"""Flower ServerApp for Flower FRL Benchmark."""

import os
import time
import numpy as np
import torch
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Union
from logging import INFO
from torch.utils.tensorboard import SummaryWriter

from flwr.common import (
    FitRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters, log,
)
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import Context
import flwr as fl

from frl_benchmark.config import get_config, get_env_info
from frl_benchmark.policy import create_policy
from frl_benchmark.core.byzantine import ByzantineFilter
from frl_benchmark.core.trajectory import sample_trajectory
from frl_benchmark.core.attacks import reset_attack_state
import frl_benchmark.strategies  # triggers auto-discovery of all strategy files
from frl_benchmark.strategies import get_strategy


def _push_metrics_to_dashboard(metrics: dict) -> None:
    """Push metrics to the dashboard via HTTP POST (best-effort, silent on failure)."""
    dashboard_url = os.environ.get("DASHBOARD_URL", "").rstrip("/")
    if not dashboard_url:
        return
    try:
        import urllib.request, json as _json
        data = _json.dumps(metrics).encode()
        req = urllib.request.Request(
            f"{dashboard_url}/api/metrics",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # Dashboard push is optional — never crash training


class FRLStrategy(Strategy):
    """Flower FRL Benchmark server strategy.

    Dispatches to the registered aggregation plugin selected by `method`:
      - 'fedpg-br': Full FedPG-BR (Byzantine filtering + SCSG)
      - 'svrpg': SCSG variance reduction only (no Byzantine filtering)
      - 'gpomdp': Simple averaging (no SCSG, no filtering)
    """

    def __init__(self, env_name: str, num_agents: int, byzantine_ratio: float = 0.0,
                 use_adaptive_batch: bool = False, method: str = 'fedpg-br', config=None,
                 num_rounds: int = 0, seed: int = 42):
        super().__init__()

        self.config = config if config is not None else get_config(env_name)
        self.env_name = env_name
        self.method = method
        env_info = get_env_info(env_name)
        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.num_agents = num_agents
        self.use_adaptive_batch = use_adaptive_batch

        self.policy = create_policy(
            self.state_dim, self.action_dim, env_name,
            self.config.hidden_units, self.config.activation, self.config.output_activation
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)

        self.byzantine_filter = ByzantineFilter(
            self.config.sigma, self.config.delta, num_agents, byzantine_ratio
        )

        if env_name == "Pursuit-v4":
            from pettingzoo.sisl import pursuit_v4
            from frl_benchmark.envs.pettingzoo_wrapper import PettingZooSingleAgentWrapper
            self.env = PettingZooSingleAgentWrapper(lambda: pursuit_v4.parallel_env(max_cycles=500))
        elif env_name == "SimpleSpread-v3":
            from pettingzoo.mpe import simple_spread_v3
            from frl_benchmark.envs.pettingzoo_wrapper import PettingZooSingleAgentWrapper
            self.env = PettingZooSingleAgentWrapper(lambda: simple_spread_v3.parallel_env(max_cycles=100, continuous_actions=False))
        else:
            self.env = gym.make(env_name)
        self.env.reset(seed=seed)
        self.theta_t_0: torch.Tensor = self.policy.get_flat_params().clone()
        self._current_batch_size = self.config.batch_size
        self._last_fit_stats: dict = {}
        self._num_rounds = num_rounds
        self._strategy = get_strategy(method)()

        # TensorBoard: one run per experiment, tagged by method + env + timestamp
        run_name = f"{method}__{env_name}__{time.strftime('%Y%m%d_%H%M%S')}"
        self._tb = SummaryWriter(log_dir=os.path.join("runs", run_name))

        log(INFO, f"Method: {self.method.upper()}")
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return ndarrays_to_parameters([p.cpu().detach().numpy() for p in self.policy.parameters()])
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        reset_attack_state()
        self.theta_t_0 = self.policy.get_flat_params().clone()
        
        if self.use_adaptive_batch:
            self._current_batch_size = np.random.randint(
                self.config.batch_size_min, self.config.batch_size_max + 1
            )
        else:
            self._current_batch_size = self.config.batch_size
        
        clients = client_manager.sample(num_clients=self.num_agents, min_num_clients=self.num_agents)
        fit_config: Dict[str, Scalar] = {"batch_size": int(self._current_batch_size), "round": server_round}
        fit_ins = fl.common.FitIns(parameters, fit_config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Filter out clients that skipped this round (num_examples == 0)
        active_results = []
        skipped_count = 0
        total_divergence = 0.0

        for client_proxy, fit_res in results:
            if fit_res.num_examples > 0:
                active_results.append((client_proxy, fit_res))
            else:
                skipped_count += 1
                # Track divergence metrics from skipped clients
                if "divergence" in fit_res.metrics:
                    total_divergence += float(fit_res.metrics["divergence"])

        # If all clients skipped, return current parameters unchanged
        if not active_results:
            log(INFO, f"Round {server_round}: All clients skipped (divergence too small)")
            return ndarrays_to_parameters([p.cpu().detach().numpy() for p in self.policy.parameters()]), {
                "num_good_agents": 0,
                "scsg_steps": 0,
                "batch_size": self._current_batch_size,
                "skipped_clients": skipped_count,
                "active_clients": 0,
                "avg_divergence": total_divergence / len(results) if results else 0.0,
            }

        gradients = []
        client_returns = []
        for _, fit_res in active_results:
            grad_numpy = parameters_to_ndarrays(fit_res.parameters)
            grad_tensor = torch.cat([torch.from_numpy(g).flatten() for g in grad_numpy]).float()
            gradients.append(grad_tensor)
            if "avg_return" in fit_res.metrics:
                client_returns.append(float(fit_res.metrics["avg_return"]))

        mu_t, good_agents = self._strategy.aggregate(
            gradients, self._current_batch_size,
            byzantine_filter=self.byzantine_filter,
        )
        actual_steps = self._strategy.server_update(
            self.policy, self.optimizer, self.theta_t_0, mu_t,
            self.config, env=self.env, env_name=self.env_name,
        )

        log(INFO, f"Round {server_round} [{self.method.upper()}]: good_agents={len(good_agents)}, "
                  f"scsg_steps={actual_steps}, active={len(active_results)}, skipped={skipped_count}")
        self._tb.add_scalar("train/good_agents", len(good_agents), server_round)
        self._tb.add_scalar("train/active_agents", len(active_results), server_round)
        self._tb.add_scalar("train/scsg_steps", actual_steps, server_round)
        self._tb.add_scalar("train/skipped_clients", skipped_count, server_round)

        # Per-client return statistics
        cr_mean = float(np.mean(client_returns)) if client_returns else 0.0
        cr_min  = float(np.min(client_returns))  if client_returns else 0.0
        cr_max  = float(np.max(client_returns))  if client_returns else 0.0
        if client_returns:
            self._tb.add_scalar("train/client_return_mean", cr_mean, server_round)
            self._tb.add_scalar("train/client_return_min",  cr_min,  server_round)
            self._tb.add_scalar("train/client_return_max",  cr_max,  server_round)

        # Policy parameter L2 norm
        param_norm = float(self.policy.get_flat_params().norm().item())
        self._tb.add_scalar("train/param_norm", param_norm, server_round)

        metrics_out = {
            "round": server_round,
            "num_good_agents": len(good_agents),
            "scsg_steps": actual_steps,
            "batch_size": self._current_batch_size,
            "skipped_clients": skipped_count,
            "active_clients": len(active_results),
            "avg_divergence": total_divergence / len(results) if results else 0.0,
            "client_reward_mean": cr_mean,
            "client_reward_min":  cr_min,
            "client_reward_max":  cr_max,
            "param_norm": param_norm,
        }
        # Cache stats — evaluate() will push everything together in one message
        self._last_fit_stats = metrics_out
        return ndarrays_to_parameters([p.cpu().detach().numpy() for p in self.policy.parameters()]), metrics_out
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        if server_round % 10 != 0:
            return []
        clients = client_manager.sample(num_clients=min(3, self.num_agents), min_num_clients=1)
        return [(c, fl.common.EvaluateIns(parameters, {"num_episodes": 10})) for c in clients]
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}
        rewards = [float(-res.loss) for _, res in results]
        avg = float(np.mean(rewards))
        log(INFO, f"Round {server_round}: Avg Reward = {avg:.2f}")
        return -avg, {"avg_reward": avg}
    
    def evaluate(self, server_round: int, parameters: Parameters):
        rewards = []
        for _ in range(10):
            _, reward = sample_trajectory(self.env, self.policy)
            rewards.append(reward)
        avg = float(np.mean(rewards))
        # Push one combined message per round: reward + agent stats from aggregate_fit
        _push_metrics_to_dashboard({
            **self._last_fit_stats,
            "round": server_round,
            "server_avg_reward": avg,
            "done": self._num_rounds > 0 and server_round >= self._num_rounds,
        })
        self._tb.add_scalar("eval/server_avg_reward", avg, server_round)
        self._tb.flush()
        return -avg, {"server_avg_reward": avg}


_dashboard_started = False


def _start_dashboard_background():
    """Start the web dashboard in a background thread (auto-starts once)."""
    global _dashboard_started
    if _dashboard_started:
        return
    _dashboard_started = True

    try:
        import threading
        from frl_benchmark.dashboard.app import app as flask_app, socketio

        def run():
            try:
                socketio.run(flask_app, host="127.0.0.1", port=8050,
                             debug=False, allow_unsafe_werkzeug=True, log_output=False)
            except Exception:
                pass  # Dashboard is optional, don't crash the server

        t = threading.Thread(target=run, daemon=True)
        t.start()
        log(INFO, "Dashboard started at http://127.0.0.1:8050/experiment")
    except ImportError:
        log(INFO, "Dashboard not available (install flask: pip install -e '.[dashboard]')")


def server_fn(context: Context):
    """Create Flower FRL Benchmark server."""
    run_config = context.run_config

    env_name = str(run_config.get("env", "CartPole-v1"))
    num_rounds = int(run_config.get("num-server-rounds", 50))
    num_workers = int(run_config.get("num-workers", 10))
    num_byzantine = int(run_config.get("num-byzantine", 0))
    use_frl_benchmark = bool(run_config.get("use-fedpg-br", False))
    method = str(run_config.get("method", "fedpg-br" if use_frl_benchmark else "gpomdp"))
    # Override config with dashboard-provided advanced settings
    cfg = get_config(env_name)
    if int(run_config.get("batch-size", 0)) > 0:
        cfg.batch_size = int(run_config["batch-size"])
    if float(run_config.get("lr", 0)) > 0:
        cfg.lr = float(run_config["lr"])
    if float(run_config.get("sigma", 0)) > 0:
        cfg.sigma = float(run_config["sigma"])
    if float(run_config.get("gamma", 0)) > 0:
        cfg.gamma = float(run_config["gamma"])
    if int(run_config.get("mini-batch-size", 0)) > 0:
        cfg.mini_batch_size = int(run_config["mini-batch-size"])
    if float(run_config.get("delta", 0)) > 0:
        cfg.delta = float(run_config["delta"])
    if int(run_config.get("max-episode-len", 0)) > 0:
        cfg.max_episode_len = int(run_config["max-episode-len"])
    if str(run_config.get("hidden-units", "")):
        hu = str(run_config["hidden-units"])
        if hu and "," in hu:
            cfg.hidden_units = tuple(int(x.strip()) for x in hu.split(","))
    if str(run_config.get("activation", "")):
        act = str(run_config["activation"])
        if act in ("ReLU", "Tanh"):
            cfg.activation = act

    _rt = int(run_config.get("round-timeout", 600))
    round_timeout = _rt if _rt > 0 else None  # 0 = no timeout (Flower default)
    byzantine_ratio = num_byzantine / num_workers if num_workers > 0 else 0.0

    # Global seed for reproducibility
    seed = int(run_config.get("seed", 42))
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Auto-start dashboard (skip if an external dashboard URL is configured)
    if not os.environ.get("DASHBOARD_URL"):
        _start_dashboard_background()

    log(INFO, f"Flower FRL Benchmark: env={env_name}, workers={num_workers}, byzantine={num_byzantine}, method={method}")

    strategy = FRLStrategy(
        env_name=env_name,
        num_agents=num_workers,
        byzantine_ratio=byzantine_ratio,
        use_adaptive_batch=use_frl_benchmark,
        method=method,
        config=cfg,
        num_rounds=num_rounds,
        seed=seed,
    )

    config = ServerConfig(num_rounds=num_rounds, round_timeout=round_timeout)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)

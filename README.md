# Flower FRL Benchmark

A benchmarking framework for federated reinforcement learning built on [Flower](https://flower.ai/), implementing the algorithms from Fan et al. (NeurIPS 2021). Designed to provide a shared, reproducible evaluation substrate for FedRL research.

Each agent explores its own copy of the environment. After every round it shares what it learned (a policy gradient), not its raw experience. A central server aggregates those gradients, updates the shared policy, and sends it back. Under Byzantine corruption, the FedPG-BR filter detects and discards adversarial updates before aggregation.

This is a bachelor's thesis project at ETH Zürich (Distributed Computing Group).

---

## How It Works

```
Round t:
                        ┌─────────────────────────────────┐
                        │         Parameter Server         │
                        │  θ  ──► Byzantine Filter ──►  θ' │
                        └──────┬────────────────▲──────────┘
                               │ broadcast θ    │ gradients
              ┌────────────────┼────────────────┼────────────────┐
              ▼                ▼                ▼                ▼
         [Agent 0]        [Agent 1]        [Agent 2]  ...  [Agent K]
         rolls out        rolls out        rolls out        rolls out
         own episodes     own episodes     own episodes     own episodes
         ∇J(θ) ──►       ∇J(θ) ──►       ✗ Byzantine      ∇J(θ) ──►
                                          (filtered out)
```

---

## Quick Start

The recommended way to run the benchmark is via Docker, which works identically on macOS, Linux, and Windows.

```bash
git clone https://github.com/JoMaag/frl-benchmark
cd frl-benchmark
docker compose up --build
# Dashboard at http://localhost:8050/experiment
```

Launch a training run from the dashboard by selecting environment, strategy, number of workers, and Byzantine ratio.

### Command line via Docker

To run a configuration file directly without the dashboard:

```bash
docker exec -it frl-benchmark flwr run . local-simulation \
    --run-config configs/paper_cartpole.toml
```

---

## Strategies

Five strategies are implemented, covering the full range from non-federated baselines to Byzantine-robust federation:

| Method      | Federated | Var. reduction | Byz. robust | Notes           |
|-------------|-----------|----------------|-------------|-----------------|
| Independent | No        | No             | No          | Lower bound     |
| Centralized | No        | No             | No          | Upper bound     |
| GPOMDP       | Yes       | No             | No          | Vanilla baseline|
| SVRPG       | Yes       | SCSG           | No          | Trusted agents  |
| FedPG-BR    | Yes       | SCSG           | Yes         | Byzantine agents|

Select a strategy via the `method` key in your config:

```toml
method = "independent"
method = "centralized"
method = "gpomdp"
method = "svrpg"
method = "fedpg-br"
```

### Add your own strategy

Drop a file into `frl_benchmark/strategies/` — it is picked up automatically at startup:

```python
from frl_benchmark.strategies import AggregationStrategy, register_strategy

@register_strategy("coordinate-median")
class CoordinateMedian(AggregationStrategy):
    description = "Coordinate-wise median aggregation"

    def aggregate(self, gradients, batch_size, **kwargs):
        stacked = torch.stack(gradients)
        return torch.median(stacked, dim=0).values, list(range(len(gradients)))

    def server_update(self, policy, optimizer, theta_t_0, mu_t, config, **kwargs):
        from frl_benchmark.strategies.base import apply_gradient
        apply_gradient(policy, optimizer, mu_t)
        return 1
```

Select it with `method = "coordinate-median"` in your TOML or from the dashboard dropdown. See `frl_benchmark/strategies/my_strategy.py` for an empty template and `example_strategy.py` for a worked trimmed-mean example.

---

## Configuration

Experiments are specified as TOML files passed to `flwr run`:

```toml
env = "CartPole-v1"           # Gymnasium or PettingZoo environment
num-server-rounds = 312       # Training rounds
num-workers = 10              # K agents
num-byzantine = 3             # Byzantine agents (must be < K/2 for FedPG-BR guarantees)
method = "fedpg-br"           # Strategy
attack-type = "sign-flip"     # Attack type for Byzantine agents
```

Any parameter not specified falls back to per-environment defaults in `config.py`.

### Pre-built configs

| Config                          | Environment     | Method      | Rounds              |
|---------------------------------|-----------------|-------------|---------------------|
| `paper_cartpole.toml`           | CartPole-v1     | FedPG-BR    | 312                 |
| `paper_cartpole_svrpg.toml`     | CartPole-v1     | SVRPG       | 312                 |
| `compare_independent.toml`      | CartPole-v1     | Independent | 312                 |
| `compare_fedpg.toml`            | CartPole-v1     | GPOMDP       | 312                 |
| `compare_centralized.toml`      | CartPole-v1     | Centralized | 312                 |
| `paper_lunarlander.toml`        | LunarLander-v3  | FedPG-BR    | 323                 |
| `paper_lunarlander_gpomdp.toml`  | LunarLander-v3  | GPOMDP       | 323                 |
| `pursuit_afedpg.toml`           | Pursuit-v4      | FedPG-BR    | 200                 |
| `pursuit_fedpg.toml`            | Pursuit-v4      | GPOMDP       | 200                 |
| `byz_afedpg_30pct.toml`         | CartPole-v1     | FedPG-BR    | 350, 30% sign-flip  |
| `byz_fedpg_30pct.toml`          | CartPole-v1     | GPOMDP       | 350, 30% sign-flip  |
| `byz_svrpg_30pct.toml`          | CartPole-v1     | SVRPG       | 350, 30% sign-flip  |
| `pursuit_byz_afedpg_30pct.toml` | Pursuit-v4      | FedPG-BR    | 200, 30% sign-flip  |
| `pursuit_byz_fedpg_30pct.toml`  | Pursuit-v4      | GPOMDP       | 200, 30% sign-flip  |

---

## Byzantine Attacks

Seven attack types for robustness evaluation, reproduced from the FedPG-BR paper:

| Key               | Name            | Description                                              |
|-------------------|-----------------|----------------------------------------------------------|
| `sign-flip`       | Sign Flip       | Sends -2.5 x true gradient                              |
| `random-noise`    | Random Noise    | Sends a random gradient vector                           |
| `random-action`   | Random Action   | Collects trajectories with a random policy               |
| `fedpg-attack`    | FedPG Attack    | Coordinated attack designed to evade the Byzantine filter|
| `variance-attack` | Variance Attack | Exploits gradient variance estimates                     |
| `zero-gradient`   | Zero Gradient   | Sends all zeros (free-rider)                             |
| `reward-flipping` | Reward Flipping | Negates rewards during rollout                           |

---

## Environments

| Environment     | Type       | Action space | Extra dependency                    |
|-----------------|------------|--------------|-------------------------------------|
| CartPole-v1     | Gymnasium  | Discrete     |                                     |
| MountainCar-v0  | Gymnasium  | Discrete     |                                     |
| Acrobot-v1      | Gymnasium  | Discrete     |                                     |
| LunarLander-v3  | Gymnasium  | Discrete     | `gymnasium[box2d]`                  |
| HalfCheetah-v5  | Gymnasium  | Continuous   | `gymnasium[mujoco]` (not evaluated) |
| Pursuit-v4      | PettingZoo | Discrete     | `pettingzoo[sisl]`                  |
| SimpleSpread-v3 | PettingZoo | Discrete     | `pettingzoo[mpe]` (not evaluated)   |

PettingZoo environments are wrapped via `PettingZooSingleAgentWrapper`, which adapts a parallel multi-agent environment to the standard Gymnasium interface. Each Flower client controls one agent; all others act randomly.

---

## Project Structure

```
frl_benchmark/
  server_app.py          # Flower ServerApp — strategy dispatch, logging
  client_app.py          # Flower ClientApp — rollout, gradient computation
  policy.py              # MLP policies (discrete + continuous action spaces)
  config.py              # Per-environment hyperparameters (from FedPG-BR paper)
  strategies/            # Plugin aggregation strategies
    base.py              #   Abstract base + decorator registry
    gpomdp.py             #   Simple averaging (GPOMDP baseline)
    svrpg.py             #   SCSG variance reduction (SVRPG)
    fedpg_br_strategy.py #   Full FedPG-BR (Byzantine filter + SCSG)
    my_strategy.py       #   Empty template for custom strategies
    example_strategy.py  #   Worked example: trimmed-mean aggregation
  core/
    byzantine.py         # Byzantine filter (FedPG-Aggregate)
    trajectory.py        # Episode sampling
    gradient.py          # Policy gradient + SCSG correction
  envs/
    pettingzoo_wrapper.py # PettingZoo to Gymnasium adapter
  flower/
    worker.py            # Environment worker — rollout + attack injection
  dashboard/             # Flask + SocketIO live web UI
configs/                 # TOML experiment configurations
```

---

## Reference

Fan, X., Ma, Y., Dai, Z., Jing, W., Tan, C., & Low, B.K.H. (2021).
**Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee.**
*Advances in Neural Information Processing Systems (NeurIPS).*

- [Paper](https://arxiv.org/abs/2110.14074)
- [Original implementation](https://github.com/flint-xf-fan/Byzantine-Federated-RL)
- [Flower framework](https://flower.ai/)

"""Flower ClientApp for Flower FRL Benchmark."""

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from frl_benchmark.config import get_config, ATTACK_ALIASES
from frl_benchmark.flower.worker import Worker


class FRLClient(NumPyClient):
    """Federated RL client that computes and returns policy gradients."""

    def __init__(self, worker: Worker):
        self.worker = worker

    def get_parameters(self, config):
        return [p.cpu().detach().numpy() for p in self.worker.policy.parameters()]

    def set_parameters(self, parameters):
        for param, new_value in zip(self.worker.policy.parameters(), parameters):
            param.data = torch.from_numpy(new_value).to(self.worker.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        batch_size = int(config.get("batch_size", 16))

        try:
            grad_list, loss, avg_return, avg_length = self.worker.compute_gradient(batch_size, sample=True)
            gradient_numpy = [g.cpu().numpy() for g in grad_list]
        except Exception:
            # Return zero gradient on failure so the server still counts this client
            gradient_numpy = [np.zeros_like(p.cpu().detach().numpy()) for p in self.worker.policy.parameters()]
            loss, avg_return, avg_length = 0.0, 0.0, 0.0

        return gradient_numpy, batch_size, {
            "loss": float(loss),
            "avg_return": float(avg_return),
            "avg_length": float(avg_length),
            "is_byzantine": self.worker.is_byzantine,
            "attack_type": self.worker.attack_type or "none",
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        num_episodes = int(config.get("num_episodes", 10))
        avg_reward, avg_length = self.worker.evaluate(num_episodes)
        return float(-avg_reward), num_episodes, {"avg_reward": float(avg_reward)}


class AdaptiveFRLClient(FRLClient):
    """Adaptive FRL client that only sends updates when policy diverges significantly.

    This reduces communication overhead by skipping rounds where the local policy
    hasn't changed much from the global policy.
    """

    def __init__(self, worker: Worker, divergence_threshold: float = 0.1, divergence_metric: str = "l2"):
        super().__init__(worker)
        self.divergence_threshold = divergence_threshold
        self.divergence_metric = divergence_metric
        self.last_global_params = None
        self.last_local_params = None
        self.rounds_skipped = 0
        self.total_rounds = 0

    def fit(self, parameters, config):
        """Fit with divergence-based adaptive communication."""
        self.total_rounds += 1

        # Store current global parameters
        current_global_params = [np.array(p) for p in parameters]

        # Train locally
        self.set_parameters(parameters)
        batch_size = int(config.get("batch_size", 16))

        grad_list, loss, avg_return, avg_length = self.worker.compute_gradient(batch_size, sample=True)
        gradient_numpy = [g.cpu().numpy() for g in grad_list]

        # Get current local parameters after training
        current_local_params = self.get_parameters(config)

        # Compute divergence if we have previous parameters
        should_skip = False
        divergence = 0.0

        if self.last_global_params is not None:
            divergence = self._compute_divergence(
                self.last_global_params,
                current_global_params
            )

            # Skip this round if divergence is below threshold
            if divergence < self.divergence_threshold:
                should_skip = True
                self.rounds_skipped += 1

        # Update stored parameters
        self.last_global_params = current_global_params
        self.last_local_params = current_local_params

        # Prepare metrics
        metrics = {
            "loss": float(loss),
            "avg_return": float(avg_return),
            "avg_length": float(avg_length),
            "is_byzantine": self.worker.is_byzantine,
            "attack_type": self.worker.attack_type or "none",
            "divergence": float(divergence),
            "skipped": should_skip,
            "rounds_skipped": self.rounds_skipped,
            "total_rounds": self.total_rounds,
            "skip_ratio": self.rounds_skipped / self.total_rounds if self.total_rounds > 0 else 0.0,
        }

        # If skipping, return zero gradients with zero examples to signal skip
        if should_skip:
            zero_gradients = [np.zeros_like(g) for g in gradient_numpy]
            return zero_gradients, 0, metrics

        return gradient_numpy, batch_size, metrics

    def _compute_divergence(self, params1, params2):
        """Compute divergence between two parameter sets.

        Args:
            params1: List of numpy arrays (first parameter set)
            params2: List of numpy arrays (second parameter set)

        Returns:
            Divergence value (float)
        """
        if self.divergence_metric == "l2":
            # L2 distance between parameter vectors
            total_diff = 0.0
            for p1, p2 in zip(params1, params2):
                total_diff += np.sum((p1 - p2) ** 2)
            return np.sqrt(total_diff)

        elif self.divergence_metric == "cosine":
            # Cosine distance (1 - cosine similarity)
            # Flatten all parameters
            vec1 = np.concatenate([p.flatten() for p in params1])
            vec2 = np.concatenate([p.flatten() for p in params2])

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance if either is zero

            cosine_sim = dot_product / (norm1 * norm2)
            return 1.0 - cosine_sim

        elif self.divergence_metric == "max":
            # Maximum absolute difference across all parameters
            max_diff = 0.0
            for p1, p2 in zip(params1, params2):
                max_diff = max(max_diff, np.max(np.abs(p1 - p2)))
            return max_diff

        else:
            # Default to L2
            total_diff = 0.0
            for p1, p2 in zip(params1, params2):
                total_diff += np.sum((p1 - p2) ** 2)
            return float(np.sqrt(total_diff))


def client_fn(context: Context):
    """Create a federated RL client."""
    run_config = context.run_config
    env_name = str(run_config.get("env", "CartPole-v1"))
    num_byzantine = int(run_config.get("num-byzantine", 0))

    # Get attack type and resolve aliases
    attack_type = str(run_config.get("attack-type", "random-noise"))
    attack_type = ATTACK_ALIASES.get(attack_type, attack_type)

    partition_id = context.node_config.get("partition-id", 0)
    client_id = int(partition_id)
    is_byzantine = client_id < num_byzantine

    # Seed per-worker for reproducibility — each worker gets a unique deterministic seed
    # derived from the global seed so different seeds produce genuinely different runs
    seed = int(run_config.get("seed", 42))
    torch.manual_seed(seed + client_id)
    np.random.seed(seed + client_id)

    config = get_config(env_name)

    worker = Worker(
        worker_id=client_id,
        env_name=env_name,
        hidden_units=config.hidden_units,
        gamma=config.gamma,
        activation=config.activation,
        output_activation=config.output_activation,
        is_byzantine=is_byzantine,
        attack_type=attack_type if is_byzantine else None,
        max_episode_len=config.max_episode_len,
        device="cpu",
    )

    # Check if adaptive communication is enabled
    use_adaptive = str(run_config.get("use-adaptive-communication", "false")).lower() == "true"

    if use_adaptive:
        # Create adaptive client with divergence-based communication
        divergence_threshold = float(run_config.get("divergence-threshold", 0.1))
        divergence_metric = str(run_config.get("divergence-metric", "l2"))

        return AdaptiveFRLClient(
            worker=worker,
            divergence_threshold=divergence_threshold,
            divergence_metric=divergence_metric,
        ).to_client()
    else:
        # Create standard client
        return FRLClient(worker).to_client()


app = ClientApp(client_fn=client_fn)

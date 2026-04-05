"""Centralized (non-federated) baseline.

Runs a single worker with batch_size = K * per_worker_batch.
Serves as the upper-bound reference: one agent with access to all trajectories.
"""

from typing import List, Tuple

import torch

from frl_benchmark.strategies.base import AggregationStrategy, register_strategy, apply_gradient


@register_strategy("centralized")
class Centralized(AggregationStrategy):
    """Centralized (non-federated) baseline.

    Uses a single worker's gradient directly — no aggregation, no filtering.
    Run with ``num_workers=1`` and ``batch_size = K * per_worker_batch`` to
    simulate the oracle centralized agent.
    """

    description = "Centralized baseline: single agent, full batch (oracle upper bound)"

    def aggregate(self, gradients: List[torch.Tensor], batch_size: int, **kwargs) -> Tuple[torch.Tensor, List[int]]:
        # Single worker — nothing to aggregate
        mu_t = gradients[0] if len(gradients) == 1 else torch.mean(torch.stack(gradients), dim=0)
        return mu_t, list(range(len(gradients)))

    def server_update(self, policy, optimizer, theta_t_0, mu_t, config, env=None, env_name="") -> int:
        policy.set_flat_params(theta_t_0)
        apply_gradient(policy, optimizer, mu_t)
        return 1

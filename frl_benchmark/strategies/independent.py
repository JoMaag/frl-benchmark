"""Independent RL: non-federated single-agent baseline.

Each round uses only one worker's gradient (no aggregation across workers).
This is equivalent to training a single agent independently, without any
federation benefit. Used as the non-federated baseline in comparisons.
"""

from typing import List, Tuple

import torch

from frl_benchmark.strategies.base import AggregationStrategy, register_strategy, apply_gradient


@register_strategy("independent")
class IndependentRL(AggregationStrategy):
    """Use only worker 0's gradient each round — no information sharing between workers."""

    description = "Non-federated baseline: single agent, no gradient sharing"

    def aggregate(self, gradients: List[torch.Tensor], batch_size: int, **kwargs) -> Tuple[torch.Tensor, List[int]]:
        # Use only worker 0 — no aggregation across workers
        return gradients[0], [0]

    def server_update(self, policy, optimizer, theta_t_0, mu_t, config, env=None, env_name="") -> int:
        policy.set_flat_params(theta_t_0)
        apply_gradient(policy, optimizer, mu_t)
        return 1

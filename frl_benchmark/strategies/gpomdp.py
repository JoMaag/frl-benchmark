"""GPOMDP: simple federated policy gradient — average gradients, single update step."""

from typing import List, Tuple

import torch

from frl_benchmark.strategies.base import AggregationStrategy, register_strategy, apply_gradient


@register_strategy("gpomdp")
class GPOMDP(AggregationStrategy):
    """Average all worker gradients and take one gradient step."""

    description = "Simple averaging, single gradient step (baseline)"

    def aggregate(self, gradients: List[torch.Tensor], batch_size: int, **kwargs) -> Tuple[torch.Tensor, List[int]]:
        mu_t = torch.mean(torch.stack(gradients), dim=0)
        good_agents = list(range(len(gradients)))
        return mu_t, good_agents

    def server_update(self, policy, optimizer, theta_t_0, mu_t, config, env=None, env_name="") -> int:
        policy.set_flat_params(theta_t_0)
        apply_gradient(policy, optimizer, mu_t)
        return 1

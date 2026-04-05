"""Example custom strategy for FedPG-BR.

This file demonstrates how to add a new aggregation method.
Drop any .py file in this directory and it's picked up automatically.

Strategy: Trimmed Mean
  - Sorts worker gradients by their L2 norm
  - Discards the top and bottom k% (trim_ratio) before averaging
  - A simple robustness heuristic that needs no prior knowledge of Byzantine count
"""

import torch
from frl_benchmark.strategies import AggregationStrategy, register_strategy, apply_gradient


@register_strategy("trimmed-mean")
class TrimmedMean(AggregationStrategy):
    """Coordinate-wise trimmed mean aggregation.

    Sorts workers by gradient L2 norm and discards the extreme outliers
    before averaging. Trim ratio controls how aggressively to filter.
    """

    description = "Trimmed mean: discard top/bottom outliers by gradient norm"

    trim_ratio: float = 0.1  # fraction of workers to discard on each end

    def aggregate(self, gradients, batch_size, **kwargs):
        # Sort workers by L2 norm of their gradient
        norms = [g.norm().item() for g in gradients]
        sorted_indices = sorted(range(len(gradients)), key=lambda i: norms[i])

        # Discard the bottom and top trim_ratio fraction
        n = len(sorted_indices)
        k = max(1, int(n * self.trim_ratio))
        good_indices = sorted_indices[k : n - k] if n - 2 * k > 0 else sorted_indices

        kept = [gradients[i] for i in good_indices]
        mu_t = torch.mean(torch.stack(kept), dim=0)
        return mu_t, good_indices

    def server_update(self, policy, optimizer, theta_t_0, mu_t, config, env=None, env_name=""):
        # Simple single-step gradient update (no SCSG variance reduction)
        policy.set_flat_params(theta_t_0)
        apply_gradient(policy, optimizer, mu_t)
        return 1

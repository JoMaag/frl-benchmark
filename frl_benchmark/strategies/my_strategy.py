"""My custom strategy for FedPG-BR.

Replace this docstring with a description of your method.
"""

import torch
from frl_benchmark.strategies import AggregationStrategy, register_strategy, apply_gradient


@register_strategy("my-method")  # <-- change this to your method name
class MyStrategy(AggregationStrategy):

    description = "My custom aggregation method"  # shown in dashboard logs

    def aggregate(self, gradients, batch_size, **kwargs):
        """Select and combine worker gradients.

        Args:
            gradients:  list of flat torch.Tensor, one per active worker
            batch_size: number of trajectories collected this round
            **kwargs:   'byzantine_filter' is available if you want to use it

        Returns:
            mu_t:         aggregated gradient tensor (same shape as each gradient)
            good_indices: list of ints — indices of workers you kept
        """
        raise NotImplementedError

    def server_update(self, policy, optimizer, theta_t_0, mu_t, config, env=None, env_name=""):
        """Apply the aggregated gradient to the server policy.

        Args:
            policy:    the server policy network
            optimizer: Adam optimizer attached to policy
            theta_t_0: flat parameter tensor at the start of this round
            mu_t:      aggregated gradient from aggregate()
            config:    Config dataclass (lr, gamma, batch_size, mini_batch_size, sigma, …)
            env:       gym environment instance (needed for SCSG variance reduction)
            env_name:  environment name string

        Returns:
            int: number of gradient steps taken
        """
        raise NotImplementedError

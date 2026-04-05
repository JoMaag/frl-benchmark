"""Base class and registry for aggregation strategies.

This module provides the plugin system that allows users to implement
custom aggregation strategies by subclassing AggregationStrategy.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import torch

# Strategy registry
_REGISTRY: Dict[str, Type["AggregationStrategy"]] = {}


def register_strategy(name: str):
    """Decorator to register a strategy class.

    Usage:
        @register_strategy("my-method")
        class MyStrategy(AggregationStrategy):
            ...
    """
    def decorator(cls):
        _REGISTRY[name] = cls
        cls.strategy_name = name
        return cls
    return decorator


def get_strategy(name: str) -> Type["AggregationStrategy"]:
    """Get a registered strategy class by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown strategy '{name}'. Available strategies: {available}"
        )
    return _REGISTRY[name]


def list_strategies() -> Dict[str, str]:
    """List all registered strategies with their descriptions."""
    return {
        name: cls.description
        for name, cls in sorted(_REGISTRY.items())
    }


def apply_gradient(policy, optimizer, gradient: torch.Tensor):
    """Helper: apply a flat gradient vector to a policy via optimizer.step()."""
    optimizer.zero_grad()
    offset = 0
    for param in policy.parameters():
        size = param.numel()
        param.grad = gradient[offset:offset + size].view(param.shape).clone()
        offset += size
    optimizer.step()


class AggregationStrategy(ABC):
    """Base class for all aggregation strategies.

    To create a custom strategy, subclass this and implement:
      - aggregate(): how to combine worker gradients
      - server_update(): how to update the global policy

    Example:
        @register_strategy("coordinate-median")
        class CoordinateMedian(AggregationStrategy):
            description = "Coordinate-wise median aggregation"

            def aggregate(self, gradients, batch_size, **kwargs):
                stacked = torch.stack(gradients)
                median = torch.median(stacked, dim=0).values
                return median, list(range(len(gradients)))

            def server_update(self, policy, optimizer, theta_t_0, mu_t, config):
                apply_gradient(policy, optimizer, mu_t)
                return 1
    """

    strategy_name: str = ""
    description: str = "Base strategy"

    @abstractmethod
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        batch_size: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Aggregate worker gradients into a single gradient estimate.

        Args:
            gradients: List of gradient tensors from K workers.
            batch_size: Number of trajectories each worker sampled.
            **kwargs: Additional context (byzantine_filter, etc.)

        Returns:
            Tuple of (aggregated_gradient, good_agent_indices).
        """
        ...

    @abstractmethod
    def server_update(
        self,
        policy,
        optimizer,
        theta_t_0: torch.Tensor,
        mu_t: torch.Tensor,
        config,
        env=None,
        env_name: str = "",
    ) -> int:
        """Update the global policy using the aggregated gradient.

        Args:
            policy: The global policy network.
            optimizer: The optimizer for the policy.
            theta_t_0: Policy parameters at the start of this round.
            mu_t: Aggregated gradient from aggregate().
            config: Environment config (has mini_batch_size, gamma, etc.)
            env: Gymnasium environment instance (for SCSG methods).
            env_name: Environment name string.

        Returns:
            Number of update steps taken.
        """
        ...

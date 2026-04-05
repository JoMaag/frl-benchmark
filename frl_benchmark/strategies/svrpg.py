"""SVRPG: Stochastic Variance-Reduced Policy Gradient.

Uses SCSG variance reduction but no Byzantine filtering.
"""

from typing import List, Tuple

import numpy as np
import torch

from frl_benchmark.strategies.base import AggregationStrategy, register_strategy, apply_gradient


@register_strategy("svrpg")
class SVRPG(AggregationStrategy):
    """Average worker gradients, then run SCSG variance-reduction steps on the server."""

    description = "SCSG variance reduction, no Byzantine filtering"

    def aggregate(self, gradients: List[torch.Tensor], batch_size: int, **kwargs) -> Tuple[torch.Tensor, List[int]]:
        mu_t = torch.mean(torch.stack(gradients), dim=0)
        good_agents = list(range(len(gradients)))
        return mu_t, good_agents

    def server_update(self, policy, optimizer, theta_t_0, mu_t, config, env=None, env_name="") -> int:
        from frl_benchmark.policy import create_policy
        from frl_benchmark.core.trajectory import sample_trajectory, compute_returns
        from frl_benchmark.core.gradient import compute_policy_gradient, compute_log_probs

        if env is None:
            raise ValueError("SVRPG server_update requires an env for SCSG sampling")

        p_geom = config.mini_batch_size / (config.batch_size + config.mini_batch_size)
        N_t = np.random.geometric(p_geom)

        theta_t_n = theta_t_0.clone()
        actual_steps = 0

        state_dim = policy.sizes[0]
        action_dim = policy.sizes[-1]

        for n in range(N_t):
            policy_n = create_policy(
                state_dim, action_dim, env_name,
                config.hidden_units, config.activation, config.output_activation
            )
            policy_n.set_flat_params(theta_t_n)

            policy_0 = create_policy(
                state_dim, action_dim, env_name,
                config.hidden_units, config.activation, config.output_activation
            )
            policy_0.set_flat_params(theta_t_0)

            all_grad_new, all_grad_old, all_ratios = [], [], []

            for _ in range(config.mini_batch_size):
                trajectory, _ = sample_trajectory(env, policy_n)
                returns = compute_returns(trajectory, config.gamma)
                grad_new, log_probs_n = compute_policy_gradient(trajectory, policy_n, config.gamma, returns)
                log_probs_0 = compute_log_probs(trajectory, policy_0)
                ratios = torch.exp(log_probs_0.detach() - log_probs_n.detach())
                all_ratios.append(ratios.mean().item())
                loss_0 = -(log_probs_0 * returns * ratios).mean()
                policy_0.zero_grad()
                loss_0.backward()
                grad_old = torch.cat([p.grad.flatten().clone() for p in policy_0.parameters() if p.grad is not None])
                all_grad_new.append(grad_new)
                all_grad_old.append(grad_old)

            v_t_n = torch.mean(torch.stack(all_grad_new), dim=0) - torch.mean(torch.stack(all_grad_old), dim=0) + mu_t
            ratio = np.mean(all_ratios)

            if ratio < 0.995 or ratio > 1.005:
                break

            policy.set_flat_params(theta_t_n)
            apply_gradient(policy, optimizer, v_t_n)
            theta_t_n = policy.get_flat_params()
            actual_steps = n + 1

        policy.set_flat_params(theta_t_n)
        return actual_steps

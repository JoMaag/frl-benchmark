"""Policy gradient computation."""

import torch
from typing import Tuple, Optional
from frl_benchmark.core.trajectory import Trajectory, compute_returns


def compute_log_probs(trajectory: Trajectory, policy, device: str = "cpu") -> torch.Tensor:
    log_probs = []
    for state, action in zip(trajectory.states, trajectory.actions):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        _, log_prob = policy(state_tensor, sample=False, fixed_action=action)
        log_probs.append(log_prob)
    return torch.stack(log_probs)


def compute_policy_gradient(trajectory: Trajectory, policy, gamma: float,
                           returns: Optional[torch.Tensor] = None, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    if returns is None:
        returns = compute_returns(trajectory, gamma)
    
    log_probs = []
    for state, action in zip(trajectory.states, trajectory.actions):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        _, log_prob = policy(state_tensor, sample=False, fixed_action=action)
        log_probs.append(log_prob)
    
    log_probs = torch.stack(log_probs)
    loss = -(log_probs * returns).mean()
    
    policy.zero_grad()
    loss.backward()
    
    gradient = torch.cat([p.grad.flatten().clone() for p in policy.parameters()])
    return gradient, log_probs

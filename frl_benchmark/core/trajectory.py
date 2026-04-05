"""Trajectory sampling and returns."""

import torch
import numpy as np
import gymnasium as gym
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Trajectory:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    
    def add(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def __len__(self):
        return len(self.rewards)
    
    @property
    def total_reward(self):
        return sum(self.rewards)


def sample_trajectory(env: gym.Env, policy, max_steps: int = 1000, device: str = "cpu") -> Tuple[Trajectory, float]:
    trajectory = Trajectory()
    reset_result = env.reset()
    state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    for _ in range(max_steps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            action, log_prob = policy(state_tensor, sample=True)
        
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.item()
        
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result
        
        trajectory.add(state, action, reward, log_prob)
        state = next_state
        
        if done:
            break
    
    return trajectory, trajectory.total_reward


def compute_returns(trajectory: Trajectory, gamma: float, normalize: bool = True) -> torch.Tensor:
    returns = []
    R = 0.0
    for reward in reversed(trajectory.rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

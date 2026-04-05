"""Policy networks for federated reinforcement learning."""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from typing import List, Tuple, Optional


def _build_mlp(sizes: List[int], activation: type, output_activation: type) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(activation() if i < len(sizes) - 2 else output_activation())
    return nn.Sequential(*layers)


def _get_activation(name: str) -> type:
    return {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "Identity": nn.Identity}[name]


class MlpPolicy(nn.Module):
    """Categorical MLP Policy for discrete actions."""
    
    def __init__(self, sizes: List[int], activation: str = "Tanh", output_activation: str = "Identity"):
        super().__init__()
        self.sizes = sizes
        self.network = _build_mlp(sizes, _get_activation(activation), _get_activation(output_activation))
        for p in self.parameters():
            p.data.uniform_(-1/math.sqrt(p.size(-1)), 1/math.sqrt(p.size(-1)))
    
    def forward(self, obs: torch.Tensor, sample: bool = True, fixed_action: Optional[int] = None):
        logits = self.network(obs.reshape(-1))
        dist = Categorical(logits=logits)
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device=obs.device)
        elif sample:
            action = dist.sample()
        else:
            action = torch.argmax(logits)
        return action.item(), dist.log_prob(action)
    
    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])
    
    def set_flat_params(self, params: torch.Tensor):
        offset = 0
        for p in self.parameters():
            p.data = params[offset:offset + p.numel()].view(p.shape)
            offset += p.numel()


class DiagonalGaussianMlpPolicy(nn.Module):
    """Gaussian MLP Policy for continuous actions."""
    
    def __init__(self, sizes: List[int], activation: str = "Tanh"):
        super().__init__()
        self.sizes = sizes
        self.features = _build_mlp(sizes[:-1], _get_activation(activation), nn.Identity)
        self.mu_head = nn.Linear(sizes[-2], sizes[-1], bias=False)
        self.log_sigma_head = nn.Linear(sizes[-2], sizes[-1], bias=False)
        for p in self.parameters():
            p.data.uniform_(-1/math.sqrt(p.size(-1)), 1/math.sqrt(p.size(-1)))
    
    def forward(self, obs: torch.Tensor, sample: bool = True, fixed_action: Optional[np.ndarray] = None):
        features = self.features(obs.reshape(-1))
        mu = torch.tanh(self.mu_head(features))
        log_sigma = torch.clamp(self.log_sigma_head(features), -20, -2)
        sigma = torch.tanh(log_sigma.exp())
        dist = Normal(mu, sigma)
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device=obs.device)
        elif sample:
            action = dist.sample()
        else:
            action = mu.detach()
        log_prob = torch.clamp(dist.log_prob(action), min=-1e5)
        return action.numpy(), log_prob.sum()
    
    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])
    
    def set_flat_params(self, params: torch.Tensor):
        offset = 0
        for p in self.parameters():
            p.data = params[offset:offset + p.numel()].view(p.shape)
            offset += p.numel()


def create_policy(state_dim: int, action_dim: int, env_name: str,
                  hidden_units: Tuple[int, ...], activation: str, output_activation: str):
    sizes = [state_dim] + list(hidden_units) + [action_dim]
    if env_name in {"HalfCheetah-v2", "HalfCheetah-v5", "Hopper-v2", "Hopper-v5", "Walker2d-v2", "Walker2d-v5"}:
        return DiagonalGaussianMlpPolicy(sizes, activation)
    return MlpPolicy(sizes, activation, output_activation)
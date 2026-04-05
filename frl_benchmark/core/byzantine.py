"""Byzantine Filtering (Algorithm 1.1: FedPG-Aggregate)."""

import numpy as np
import torch
from typing import List, Tuple


class ByzantineFilter:
    """Byzantine-resilient gradient aggregation."""
    
    def __init__(self, sigma: float, delta: float, num_agents: int, alpha: float):
        self.sigma = sigma
        self.delta = delta
        self.num_agents = num_agents
        self.alpha = alpha
    
    def aggregate(self, gradients: List[torch.Tensor], batch_size: int) -> Tuple[torch.Tensor, List[int]]:
        K = len(gradients)
        if K == 0:
            raise ValueError("No gradients to aggregate")
        
        V = 2 * np.log(2 * max(K, 1) / self.delta)
        T_mu = 2 * self.sigma * np.sqrt(V / batch_size)
        
        good_agents = self._filter(gradients, T_mu)
        
        if len(good_agents) < (1 - self.alpha) * K:
            good_agents = self._filter(gradients, 2 * self.sigma)
        
        if good_agents:
            aggregated = torch.mean(torch.stack([gradients[i] for i in good_agents]), dim=0)
        else:
            aggregated = torch.mean(torch.stack(gradients), dim=0)
        
        return aggregated, good_agents
    
    def _filter(self, gradients: List[torch.Tensor], threshold: float) -> List[int]:
        K = len(gradients)
        
        S_indices = []
        for i in range(K):
            neighbors = sum(1 for j in range(K) if torch.norm(gradients[i] - gradients[j]).item() <= threshold)
            if neighbors > K / 2:
                S_indices.append(i)
        
        if not S_indices:
            return list(range(K))
        
        S_grads = [gradients[i] for i in S_indices]
        mean_S = torch.mean(torch.stack(S_grads), dim=0)
        mu_mom_idx = min(S_indices, key=lambda i: torch.norm(gradients[i] - mean_S).item())
        mu_mom = gradients[mu_mom_idx]
        
        return [i for i in range(K) if torch.norm(gradients[i] - mu_mom).item() <= threshold]

"""Byzantine attack implementations for federated RL evaluation.

Seven attack types:
  random-noise     Send a random gradient vector
  random-action    Collect trajectories with a random policy
  sign-flip        Send -2.5 * true gradient
  fedpg-attack     Coordinated attack designed to evade the Byzantine filter
  variance-attack  Exploit gradient variance to stay within filter threshold
  zero-gradient    Send all zeros (free-rider / silent failure)
  reward-flipping  Negate rewards during trajectory collection
"""

import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AttackConfig:
    sign_flip_scale: float = -2.5
    noise_scale: float = 3.0
    va_z_max: float = 0.18
    fedpg_attack_scale: float = 3.0


# Shared state for attacks that require coordination across Byzantine workers
_byzantine_gradients: Dict[int, List[torch.Tensor]] = {}
_attack_round: int = 0


def reset_attack_state():
    global _byzantine_gradients, _attack_round
    _byzantine_gradients = {}
    _attack_round += 1


def register_byzantine_gradient(worker_id: int, gradients: List[torch.Tensor]):
    _byzantine_gradients[worker_id] = [g.clone() for g in gradients]


def get_byzantine_gradients() -> Dict[int, List[torch.Tensor]]:
    return _byzantine_gradients


class ByzantineAttack:
    def __init__(self, config: Optional[AttackConfig] = None):
        self.config = config or AttackConfig()

    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        raise NotImplementedError


class RandomNoiseAttack(ByzantineAttack):
    """Replace each gradient with a random vector of similar magnitude."""

    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        attacked = []
        for grad in gradients:
            magnitude = grad.abs().mean().item() * self.config.noise_scale
            if magnitude < 1e-6:
                magnitude = 1.0
            attacked.append((torch.rand_like(grad) * 2 - 1) * magnitude)
        return attacked


class RandomActionAttack(ByzantineAttack):
    """Gradients collected from a random policy (applied during rollout, not here)."""

    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        return gradients

    @staticmethod
    def get_random_action(action_space) -> Any:
        return action_space.sample()


class SignFlippingAttack(ByzantineAttack):
    """Send -2.5 * true gradient."""

    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        return [self.config.sign_flip_scale * grad for grad in gradients]


class FedPGAttack(ByzantineAttack):
    """Coordinated attack: Byzantine workers estimate the mean gradient and send
    mean + 3*sigma to push the aggregate while staying within the filter threshold."""

    def apply(self, gradients: List[torch.Tensor], worker_id: int = 0, **kwargs) -> List[torch.Tensor]:
        register_byzantine_gradient(worker_id, gradients)
        all_byz_grads = get_byzantine_gradients()

        if len(all_byz_grads) < 2:
            return [self.config.sign_flip_scale * grad for grad in gradients]

        num_params = len(gradients)
        mean_grads = []
        for param_idx in range(num_params):
            param_grads = [all_byz_grads[wid][param_idx] for wid in all_byz_grads]
            mean_grads.append(torch.mean(torch.stack(param_grads), dim=0))

        # Estimate sigma from pairwise distances between Byzantine gradients
        max_distance = 0.0
        worker_ids = list(all_byz_grads.keys())
        for i, wid1 in enumerate(worker_ids):
            for wid2 in worker_ids[i+1:]:
                flat1 = torch.cat([g.flatten() for g in all_byz_grads[wid1]])
                flat2 = torch.cat([g.flatten() for g in all_byz_grads[wid2]])
                max_distance = max(max_distance, torch.norm(flat1 - flat2).item())
        sigma_estimate = max_distance / 2.0

        attacked = []
        for mean_grad in mean_grads:
            direction = mean_grad / (torch.norm(mean_grad) + 1e-8)
            attacked.append(mean_grad + self.config.fedpg_attack_scale * sigma_estimate * direction)
        return attacked


class VarianceAttack(ByzantineAttack):
    """Push the gradient to the edge of the estimated variance distribution."""

    def apply(self, gradients: List[torch.Tensor], worker_id: int = 0, **kwargs) -> List[torch.Tensor]:
        register_byzantine_gradient(worker_id, gradients)
        all_byz_grads = get_byzantine_gradients()

        if len(all_byz_grads) < 2:
            return RandomNoiseAttack(self.config).apply(gradients)

        attacked = []
        for param_idx in range(len(gradients)):
            param_grads = torch.stack([all_byz_grads[wid][param_idx] for wid in all_byz_grads])
            mean_grad = param_grads.mean(dim=0)
            std_grad = param_grads.std(dim=0) + 1e-8
            attacked.append(mean_grad + self.config.va_z_max * std_grad)
        return attacked


class ZeroGradientAttack(ByzantineAttack):
    """Send zero gradients (free-rider / silent failure)."""

    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        return [torch.zeros_like(grad) for grad in gradients]


class RewardFlippingAttack(ByzantineAttack):
    """Rewards are negated during rollout; gradients arrive pre-corrupted."""

    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        return gradients


ATTACK_REGISTRY = {
    "random-noise": RandomNoiseAttack,
    "random-action": RandomActionAttack,
    "sign-flip": SignFlippingAttack,
    "sign-flipping": SignFlippingAttack,
    "fedpg-attack": FedPGAttack,
    "variance-attack": VarianceAttack,
    "zero-gradient": ZeroGradientAttack,
    "reward-flipping": RewardFlippingAttack,
}

PAPER_ATTACK_TYPES = ["random-noise", "random-action", "sign-flip", "fedpg-attack"]


def get_attack(attack_type: str, config: Optional[AttackConfig] = None) -> ByzantineAttack:
    if attack_type not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack type: {attack_type}. Available: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[attack_type](config)


def apply_attack(attack_type: str, gradients: List[torch.Tensor],
                 worker_id: int = 0, config: Optional[AttackConfig] = None) -> List[torch.Tensor]:
    return get_attack(attack_type, config).apply(gradients, worker_id=worker_id)

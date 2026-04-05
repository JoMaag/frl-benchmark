"""Environment configurations and hyperparameters for Flower FRL Benchmark."""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pygame.pkgdata')

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import gymnasium as gym

# Supported attack types
ATTACK_TYPES = [
    # Paper attacks (Section 5)
    "random-noise",      # RN: sends random vector
    "random-action",     # RA: takes random actions (hardware failure simulation)
    "sign-flip",         # SF: sends -2.5 * gradient
    "fedpg-attack",      # Sophisticated attack to evade Byzantine filter
    # Additional attacks
    "variance-attack",   # VA: exploits gradient variance (Appendix G.2)
    "zero-gradient",     # Sends zero gradients
    "reward-flipping",   # Negates rewards during training
]

# Aliases for backward compatibility
ATTACK_ALIASES = {
    "sign-flipping": "sign-flip",
    "random-reward": "random-noise",
}


@dataclass
class Config:
    env_name: str = "CartPole-v1"
    max_episode_len: int = 500
    gamma: float = 0.999
    hidden_units: Tuple[int, ...] = (16, 16)
    activation: str = "ReLU"
    output_activation: str = "Tanh"
    lr: float = 1e-3
    batch_size: int = 16
    batch_size_min: int = 12
    batch_size_max: int = 20
    mini_batch_size: int = 4
    sigma: float = 0.06
    delta: float = 0.6


_CONFIGS = {
    "CartPole-v1": Config(),
    "MountainCar-v0": Config(
        env_name="MountainCar-v0", max_episode_len=1000, gamma=0.99,
        hidden_units=(32, 32), activation="Tanh", batch_size=24,
        batch_size_min=20, batch_size_max=28, mini_batch_size=6, sigma=0.08
    ),
    "Acrobot-v1": Config(
        env_name="Acrobot-v1", max_episode_len=500, gamma=0.99,
        hidden_units=(32, 32), activation="Tanh", batch_size=24,
        batch_size_min=20, batch_size_max=28, mini_batch_size=6, sigma=0.08
    ),
    "LunarLander-v2": Config(
        env_name="LunarLander-v2", max_episode_len=1000, gamma=0.990,
        hidden_units=(64, 64), activation="Tanh", batch_size=32,
        batch_size_min=26, batch_size_max=38, mini_batch_size=8, sigma=0.07
    ),
    "LunarLander-v3": Config(
        env_name="LunarLander-v3", max_episode_len=1000, gamma=0.990,
        hidden_units=(64, 64), activation="Tanh", batch_size=32,
        batch_size_min=26, batch_size_max=38, mini_batch_size=8, sigma=0.07
    ),
    "HalfCheetah-v2": Config(
        env_name="HalfCheetah-v2", gamma=0.995, hidden_units=(64, 64),
        activation="Tanh", lr=8e-5, batch_size=48, batch_size_min=46,
        batch_size_max=50, mini_batch_size=16, sigma=0.9
    ),
    "HalfCheetah-v5": Config(
        env_name="HalfCheetah-v5", gamma=0.995, hidden_units=(64, 64),
        activation="Tanh", lr=8e-5, batch_size=48, batch_size_min=46,
        batch_size_max=50, mini_batch_size=16, sigma=0.9
    ),
}


_CONFIGS.update({
    "SimpleSpread-v3": Config(
        env_name="SimpleSpread-v3", max_episode_len=100, gamma=0.99,
        hidden_units=(64, 64), activation="Tanh", batch_size=32,
        batch_size_min=28, batch_size_max=36, mini_batch_size=8, sigma=0.05
    ),
    "Pursuit-v4": Config(
        env_name="Pursuit-v4", max_episode_len=500, gamma=0.99,
        hidden_units=(64, 64), activation="Tanh", batch_size=32,
        batch_size_min=28, batch_size_max=36, mini_batch_size=8, sigma=0.05
    ),
})


def get_config(env_name: str) -> Config:
    return _CONFIGS.get(env_name, Config(env_name=env_name))


def get_env_info(env_name: str) -> dict:
    if env_name == "Pursuit-v4":
        from pettingzoo.sisl import pursuit_v4
        from frl_benchmark.envs.pettingzoo_wrapper import PettingZooSingleAgentWrapper
        env = PettingZooSingleAgentWrapper(lambda: pursuit_v4.parallel_env(max_cycles=500))
    elif env_name == "SimpleSpread-v3":
        from pettingzoo.mpe import simple_spread_v3
        from frl_benchmark.envs.pettingzoo_wrapper import PettingZooSingleAgentWrapper
        env = PettingZooSingleAgentWrapper(lambda: simple_spread_v3.parallel_env(max_cycles=100, continuous_actions=False))
    else:
        env = gym.make(env_name)
    assert env.observation_space.shape is not None
    state_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = int(env.action_space.n)
    else:
        assert env.action_space.shape is not None
        action_dim = int(env.action_space.shape[0])
    is_continuous = not isinstance(env.action_space, gym.spaces.Discrete)
    env.close()
    return {"state_dim": state_dim, "action_dim": action_dim, "is_continuous": is_continuous}
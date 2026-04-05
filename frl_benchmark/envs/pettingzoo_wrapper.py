"""PettingZoo → Gymnasium wrapper for single-agent control.

Wraps a PettingZoo parallel environment so that one agent (identified by
agent_idx) is the "main" agent controlled by the Flower client. All other
agents use random actions. This makes PettingZoo environments drop-in
compatible with the existing Worker class.

Supported environments (registered with gym.register):
  - SimpleSpread-v3   cooperative navigation, continuous actions
  - Pursuit-v4        cooperative pursuit, discrete actions
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import gymnasium as gym


class PettingZooSingleAgentWrapper(gym.Env):
    """Gymnasium wrapper around a PettingZoo parallel environment.

    One agent (agent_idx) is controlled; all others take random actions.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_fn: Callable, agent_idx: int = 0):
        super().__init__()
        self._env_fn = env_fn
        self._agent_idx = agent_idx

        # Inspect spaces using a temporary instance
        tmp = env_fn()
        tmp.reset()
        agents = tmp.possible_agents
        self._agent_name = agents[agent_idx % len(agents)]
        self.observation_space = tmp.observation_spaces[self._agent_name]
        self.action_space = tmp.action_spaces[self._agent_name]
        tmp.close()

        self._env: Any = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if self._env is not None:
            self._env.close()
        self._env = self._env_fn()
        obs, infos = self._env.reset(seed=seed)
        agent_obs = obs.get(self._agent_name, np.zeros(self.observation_space.shape, dtype=np.float32))
        return agent_obs.astype(np.float32), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._env is None:
            raise RuntimeError("Call reset() before step()")

        actions = {}
        for agent in self._env.agents:
            if agent == self._agent_name:
                actions[agent] = action
            else:
                actions[agent] = self._env.action_spaces[agent].sample()

        obs, rewards, terminations, truncations, _ = self._env.step(actions)

        if self._agent_name in obs:
            observation = obs[self._agent_name].astype(np.float32)
            reward = float(rewards.get(self._agent_name, 0.0))
            terminated = bool(terminations.get(self._agent_name, False))
            truncated = bool(truncations.get(self._agent_name, False))
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = True
            truncated = False

        return observation, reward, terminated, truncated, {}

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def _register_pettingzoo_envs():
    """Register PettingZoo environments with gymnasium if pettingzoo is installed."""
    try:
        from mpe2 import simple_spread_v3
        from pettingzoo.sisl import pursuit_v4

        if "SimpleSpread-v3" not in gym.envs.registry:
            gym.register(
                id="SimpleSpread-v3",
                entry_point=lambda agent_idx=0: PettingZooSingleAgentWrapper(
                    lambda: simple_spread_v3.parallel_env(max_cycles=100, continuous_actions=False),
                    agent_idx=agent_idx,
                ),
            )

        if "Pursuit-v4" not in gym.envs.registry:
            gym.register(
                id="Pursuit-v4",
                entry_point=lambda agent_idx=0: PettingZooSingleAgentWrapper(
                    lambda: pursuit_v4.parallel_env(max_cycles=500),
                    agent_idx=agent_idx,
                ),
            )

    except ImportError:
        pass  # pettingzoo not installed — skip silently


_register_pettingzoo_envs()

"""Worker (RL Agent) for Federated Learning."""

import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import Any, List, Tuple, Optional

from frl_benchmark.policy import MlpPolicy, DiagonalGaussianMlpPolicy
from frl_benchmark.core.attacks import apply_attack, AttackConfig


class Worker:
    """Samples trajectories and computes policy gradients. Byzantine workers corrupt their gradients before returning them."""
    
    def __init__(self, worker_id: int, env_name: str, hidden_units: Tuple[int, ...],
                 gamma: float, activation: str = "Tanh", output_activation: str = "Identity",
                 is_byzantine: bool = False, attack_type: Optional[str] = None,
                 max_episode_len: int = 1000, device: str = "cpu",
                 attack_config: Optional[AttackConfig] = None):
        
        self.id = worker_id
        self.gamma = gamma
        self.is_byzantine = is_byzantine
        self.attack_type = attack_type
        self.max_episode_len = max_episode_len
        self.device = device
        self.attack_config = attack_config or AttackConfig()
        
        # seed with worker_id for reproducibility
        if env_name == "Pursuit-v4":
            from pettingzoo.sisl import pursuit_v4
            from frl_benchmark.envs.pettingzoo_wrapper import PettingZooSingleAgentWrapper
            self.env = PettingZooSingleAgentWrapper(lambda: pursuit_v4.parallel_env(max_cycles=500), agent_idx=worker_id)
        elif env_name == "SimpleSpread-v3":
            from pettingzoo.mpe import simple_spread_v3
            from frl_benchmark.envs.pettingzoo_wrapper import PettingZooSingleAgentWrapper
            self.env = PettingZooSingleAgentWrapper(lambda: simple_spread_v3.parallel_env(max_cycles=100, continuous_actions=False), agent_idx=worker_id)
        else:
            self.env = gym.make(env_name)
        self.env.reset(seed=worker_id)
        assert self.env.observation_space.shape is not None
        obs_dim = int(self.env.observation_space.shape[0])
        
        # Create policy based on action space type
        if isinstance(self.env.action_space, Discrete):
            action_dim = int(self.env.action_space.n)
            self.policy = MlpPolicy(
                [obs_dim] + list(hidden_units) + [action_dim], 
                activation, 
                output_activation
            )
            self.is_discrete = True
        else:
            assert self.env.action_space.shape is not None
            action_dim = int(self.env.action_space.shape[0])
            self.policy = DiagonalGaussianMlpPolicy(
                [obs_dim] + list(hidden_units) + [action_dim], 
                activation
            )
            self.is_discrete = False
        
        self.policy.to(device)
        
        if self.is_byzantine:
            print(f"Worker {worker_id}: Byzantine agent with attack '{attack_type}'")
    
    def _sample_action(self, state_tensor: torch.Tensor, sample: bool = True) -> Tuple[Any, torch.Tensor]:
        """Sample action, potentially using random action for RA attack."""
        
        # Random Action (RA) attack: ignore policy, take random actions
        if self.is_byzantine and self.attack_type == "random-action":
            # Get random action from environment
            random_action = self.env.action_space.sample()
            
            # Still need to get log_prob from policy for gradient computation
            action, log_prob = self.policy(state_tensor, sample=sample)
            return random_action, log_prob
        
        # Normal policy sampling
        action, log_prob = self.policy(state_tensor, sample=sample)
        return action, log_prob
    
    def _apply_reward_attack(self, rewards: List[float]) -> List[float]:
        """Apply reward-based attacks if applicable."""
        if not self.is_byzantine:
            return rewards
        
        if self.attack_type == "reward-flipping":
            return [-r for r in rewards]
        
        return rewards
    
    def compute_gradient(self, num_trajectories: int, sample: bool = True) -> Tuple[List[torch.Tensor], float, float, float]:
        """Collect num_trajectories episodes and return (gradients, loss, avg_return, avg_length)."""
        all_advantages = []
        all_log_probs = []
        all_returns = []
        all_lengths = []
        
        # Sample trajectories
        while len(all_returns) < num_trajectories:
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
            episode_rewards = []
            episode_log_probs = []
            
            for _ in range(self.max_episode_len):
                state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
                
                # Get action (potentially random for RA attack)
                action, log_prob = self._sample_action(state_tensor, sample=sample)
                
                # Environment step
                state, reward, terminated, truncated, _ = self.env.step(action)

                episode_rewards.append(float(reward))
                episode_log_probs.append(log_prob)

                if terminated or truncated:
                    break
            
            # Apply reward-based attacks
            episode_rewards = self._apply_reward_attack(episode_rewards)
            
            # Compute returns (discounted cumulative rewards)
            returns = []
            R = 0.0
            for r in reversed(episode_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            # Normalize advantages
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            all_advantages.extend(advantages.tolist())
            all_log_probs.extend(episode_log_probs)
            all_returns.append(sum(episode_rewards))
            all_lengths.append(len(episode_rewards))
        
        advantages = torch.tensor(all_advantages, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(all_log_probs)
        
        loss = -(log_probs * advantages).mean()
        
        self.policy.zero_grad()
        loss.backward()
        
        gradients = [param.grad.clone() for param in self.policy.parameters() if param.grad is not None]
        
        # Apply gradient-based Byzantine attacks
        if self.is_byzantine and self.attack_type is not None:
            if self.attack_type not in ["random-action", "reward-flipping"]:
                # These attacks modify gradients directly
                gradients = apply_attack(
                    self.attack_type, 
                    gradients, 
                    worker_id=self.id,
                    config=self.attack_config
                )
        
        return gradients, float(loss.item()), float(np.mean(all_returns)), float(np.mean(all_lengths))
    
    def evaluate(self, num_episodes: int = 10, max_steps: int = 1000) -> Tuple[float, float]:
        """Run num_episodes greedy episodes and return (avg_reward, avg_length)."""
        total_reward = 0.0
        total_length = 0
        
        for _ in range(num_episodes):
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            episode_reward = 0.0
            
            step = 0
            for step in range(max_steps):
                state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    action, _ = self.policy(state_tensor, sample=False)
                
                state, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += float(reward)
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            total_length += step + 1
        
        return total_reward / num_episodes, total_length / num_episodes
    
    def close(self):
        """Clean up environment."""
        self.env.close()

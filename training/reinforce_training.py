"""
REINFORCE Training Script for NutriVision Environment
Vanilla Policy Gradient implementation from scratch
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from environment.custom_env import NutriVisionEnv


class PolicyNetwork(nn.Module):
    """Neural network for policy in REINFORCE."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)


class REINFORCEAgent:
    """REINFORCE (Vanilla Policy Gradient) agent."""
    
    def __init__(self, state_size: int = 15, action_size: int = 4, 
                 hidden_size: int = 128, learning_rate: float = 1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.device = torch.device("cpu")
        
        self.policy = PolicyNetwork(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.rewards = []
        self.log_probs = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def store_reward(self, reward: float):
        """Store reward for this step."""
        self.rewards.append(reward)
    
    def store_log_prob(self, log_prob: torch.Tensor):
        """Store log probability for this step."""
        self.log_probs.append(log_prob)
    
    def compute_returns(self) -> torch.Tensor:
        """Compute discounted cumulative rewards."""
        returns = []
        cumulative = 0
        for reward in reversed(self.rewards):
            cumulative = reward + self.gamma * cumulative
            returns.insert(0, cumulative)
        
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        return returns_tensor
    
    def update(self):
        """Update policy using collected trajectories."""
        returns = self.compute_returns()
        loss = 0
        
        for log_prob, return_val in zip(self.log_probs, returns):
            loss += -log_prob * return_val
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.rewards = []
        self.log_probs = []
        
        return loss.item()


class REINFORCETrainer:
    """Trainer for REINFORCE algorithm."""
    
    def __init__(self, log_dir="models/pg"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.hyperparameter_configs = self._generate_hp_configs()
        self.results = {}
    
    def _generate_hp_configs(self) -> List[Dict]:
        """Generate 10+ REINFORCE hyperparameter configurations."""
        configs = [
            # Config 1: Default baseline
            {
                "name": "default",
                "learning_rate": 1e-3,
                "hidden_size": 128,
                "gamma": 0.99,
            },
            # Config 2: Higher learning rate
            {
                "name": "high_lr",
                "learning_rate": 5e-3,
                "hidden_size": 128,
                "gamma": 0.99,
            },
            # Config 3: Lower learning rate
            {
                "name": "low_lr",
                "learning_rate": 1e-4,
                "hidden_size": 128,
                "gamma": 0.99,
            },
            # Config 4: Larger network
            {
                "name": "large_network",
                "learning_rate": 1e-3,
                "hidden_size": 256,
                "gamma": 0.99,
            },
            # Config 5: Smaller network
            {
                "name": "small_network",
                "learning_rate": 1e-3,
                "hidden_size": 64,
                "gamma": 0.99,
            },
            # Config 6: Higher discount factor
            {
                "name": "high_gamma",
                "learning_rate": 1e-3,
                "hidden_size": 128,
                "gamma": 0.995,
            },
            # Config 7: Lower discount factor
            {
                "name": "low_gamma",
                "learning_rate": 1e-3,
                "hidden_size": 128,
                "gamma": 0.95,
            },
            # Config 8: Medium learning rate with large network
            {
                "name": "medium_lr_large",
                "learning_rate": 3e-3,
                "hidden_size": 256,
                "gamma": 0.99,
            },
            # Config 9: Aggressive learning
            {
                "name": "aggressive",
                "learning_rate": 1e-2,
                "hidden_size": 128,
                "gamma": 0.99,
            },
            # Config 10: Conservative learning
            {
                "name": "conservative",
                "learning_rate": 1e-4,
                "hidden_size": 64,
                "gamma": 0.99,
            },
        ]
        return configs
    
    def train_single_config(self, config: Dict, num_episodes: int = 500) -> Dict:
        """Train REINFORCE with a single hyperparameter configuration."""
        print(f"\n{'='*60}")
        print(f"Training REINFORCE - Config: {config['name']}")
        print(f"{'='*60}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Hidden Size: {config['hidden_size']}")
        print(f"Gamma: {config['gamma']}")
        
        env = NutriVisionEnv()
        agent = REINFORCEAgent(
            state_size=15,
            action_size=4,
            hidden_size=config["hidden_size"],
            learning_rate=config["learning_rate"]
        )
        agent.gamma = config["gamma"]
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                action, log_prob = agent.select_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                
                agent.store_reward(reward)
                agent.store_log_prob(log_prob)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            # Update policy at end of episode
            loss = agent.update()
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward (50-ep): {avg_reward:.3f}")
        
        # Evaluate final policy
        eval_rewards = []
        for _ in range(20):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                action, _ = agent.select_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        print(f"\nResults:")
        print(f"Mean Reward (20 eval episodes): {mean_reward:.3f} ± {std_reward:.3f}")
        
        # Save model
        model_path = os.path.join(self.log_dir, f"reinforce_{config['name']}.pt")
        torch.save(agent.policy.state_dict(), model_path)
        print(f"[OK] Model saved to {model_path}")
        
        result = {
            "config": config,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "model_path": model_path,
            "episode_rewards": episode_rewards,
        }
        
        env.close()
        return result
    
    def train_all_configs(self, num_episodes: int = 500) -> Dict:
        """Train REINFORCE with all hyperparameter configurations."""
        print("\n" + "="*80)
        print("REINFORCE HYPERPARAMETER TUNING - 10 CONFIGURATIONS")
        print("="*80)
        
        for config in self.hyperparameter_configs:
            result = self.train_single_config(config, num_episodes)
            self.results[config["name"]] = result
        
        self._save_results_summary()
        return self.results
    
    def _save_results_summary(self):
        """Save training results summary to JSON."""
        summary = {
            config_name: {
                "mean_reward": result["mean_reward"],
                "std_reward": result["std_reward"],
                "config": result["config"],
            }
            for config_name, result in self.results.items()
        }
        
        summary_path = os.path.join(self.log_dir, "reinforce_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[OK] Training summary saved to {summary_path}")
        
        # Print ranking
        print("\n" + "="*60)
        print("REINFORCE CONFIGURATION RANKING (by Mean Reward)")
        print("="*60)
        
        ranked = sorted(self.results.items(), 
                       key=lambda x: x[1]["mean_reward"], 
                       reverse=True)
        
        for rank, (config_name, result) in enumerate(ranked, 1):
            print(f"{rank:2d}. {config_name:30s} | "
                  f"Reward: {result['mean_reward']:7.3f} ± {result['std_reward']:6.3f}")


def main():
    """Main training function."""
    trainer = REINFORCETrainer()
    trainer.train_all_configs(num_episodes=500)


if __name__ == "__main__":
    main()

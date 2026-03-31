"""
DQN Training Script for NutriVision Environment
Deep Q-Network implementation using Stable Baselines3
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import NutriVisionEnv


class DQNTrainer:
    """Trainer class for DQN algorithm."""
    
    def __init__(self, log_dir="models/dqn"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.hyperparameter_configs = self._generate_hp_configs()
        self.results = {}
    
    def _generate_hp_configs(self) -> List[Dict]:
        """Generate 10 hyperparameter configurations for DQN."""
        configs = [
            # Config 1: Default baseline
            {
                "name": "default",
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
            # Config 2: Higher learning rate
            {
                "name": "high_lr",
                "learning_rate": 5e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
            # Config 3: Lower learning rate
            {
                "name": "low_lr",
                "learning_rate": 1e-5,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
            # Config 4: Larger batch size
            {
                "name": "large_batch",
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 64,
                "tau": 1.0,
                "gamma": 0.99,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
            # Config 5: Smaller batch size
            {
                "name": "small_batch",
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 16,
                "tau": 1.0,
                "gamma": 0.99,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
            # Config 6: Higher discount factor
            {
                "name": "high_gamma",
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.995,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
            # Config 7: Lower discount factor
            {
                "name": "low_gamma",
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.95,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
            # Config 8: Larger buffer
            {
                "name": "large_buffer",
                "learning_rate": 1e-4,
                "buffer_size": 100000,
                "learning_starts": 2000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
            # Config 9: Aggressive exploration
            {
                "name": "aggressive_exploration",
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 500,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "exploration_fraction": 0.3,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.01,
            },
            # Config 10: Conservative exploration
            {
                "name": "conservative_exploration",
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 2000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "exploration_fraction": 0.05,
                "exploration_initial_eps": 0.5,
                "exploration_final_eps": 0.1,
            },
        ]
        
        return configs
    
    def train_single_config(self, config: Dict, total_timesteps: int = 50000) -> Dict:
        """Train DQN model with a single hyperparameter configuration."""
        print(f"\n{'='*60}")
        print(f"Training DQN - Config: {config['name']}")
        print(f"{'='*60}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Buffer Size: {config['buffer_size']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Gamma: {config['gamma']}")
        print(f"Exploration Fraction: {config['exploration_fraction']}")
        
        # Create environment
        env = NutriVisionEnv()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create DQN model
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["batch_size"],
            tau=config["tau"],
            gamma=config["gamma"],
            exploration_fraction=config["exploration_fraction"],
            exploration_initial_eps=config["exploration_initial_eps"],
            exploration_final_eps=config["exploration_final_eps"],
            verbose=0,
            device=device,
        )
        
        # Train model
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
        
        print(f"\nResults:")
        print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        
        # Save model
        model_path = os.path.join(self.log_dir, f"dqn_{config['name']}")
        model.save(model_path)
        print(f"[OK] Model saved to {model_path}")
        
        # Store results
        result = {
            "config": config,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "model_path": model_path,
        }
        
        env.close()
        return result
    
    def train_all_configs(self, total_timesteps: int = 50000) -> Dict:
        """Train DQN with all hyperparameter configurations."""
        print("\n" + "="*80)
        print("DQN HYPERPARAMETER TUNING - 10 CONFIGURATIONS")
        print("="*80)
        
        for config in self.hyperparameter_configs:
            result = self.train_single_config(config, total_timesteps)
            self.results[config["name"]] = result
        
        # Save results summary
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
        
        training_path = os.path.join(self.log_dir, "training_summary.json")
        dqn_path = os.path.join(self.log_dir, "dqn_summary.json")
        with open(training_path, "w") as f:
            json.dump(summary, f, indent=2)
        with open(dqn_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[OK] Training summary saved to {training_path}")
        print(f"[OK] DQN summary saved to {dqn_path}")
        
        # Print ranking
        print("\n" + "="*60)
        print("DQN CONFIGURATION RANKING (by Mean Reward)")
        print("="*60)
        
        ranked = sorted(self.results.items(), 
                       key=lambda x: x[1]["mean_reward"], 
                       reverse=True)
        
        for rank, (config_name, result) in enumerate(ranked, 1):
            print(f"{rank:2d}. {config_name:30s} | "
                  f"Reward: {result['mean_reward']:7.3f} ± {result['std_reward']:6.3f}")


def main():
    """Main training function."""
    trainer = DQNTrainer()
    trainer.train_all_configs(total_timesteps=50000)


if __name__ == "__main__":
    main()

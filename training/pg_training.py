"""
Policy Gradient Training Script for NutriVision Environment
Implements REINFORCE, PPO, and A2C using Stable Baselines3
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import NutriVisionEnv


class PolicyGradientTrainer:
    """Base trainer class for policy gradient methods."""
    
    def __init__(self, algorithm_name: str, log_dir: str = None):
        self.algorithm_name = algorithm_name
        self.log_dir = log_dir or f"models/{algorithm_name.lower()}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.hyperparameter_configs = self._generate_hp_configs()
        self.results = {}
    
    def _generate_hp_configs(self) -> List[Dict]:
        """Generate 10+ hyperparameter configurations. Override in subclasses."""
        raise NotImplementedError
    
    def train_single_config(self, config: Dict, total_timesteps: int = 50000) -> Dict:
        """Train model with a single hyperparameter configuration."""
        print(f"\n{'='*60}")
        print(f"Training {self.algorithm_name} - Config: {config['name']}")
        print(f"{'='*60}")
        
        for key, value in config.items():
            if key != "name":
                print(f"{key}: {value}")
        
        # Create environment
        env = NutriVisionEnv()
        # SB3 warns PPO/A2C on GPU is inefficient for MLP policies; default to CPU for stable local runs.
        use_gpu = os.getenv("RL_USE_GPU", "0").strip().lower() in {"1", "true", "yes", "y"}
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        # Create model based on algorithm
        if self.algorithm_name == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                n_epochs=config["n_epochs"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                clip_range=config["clip_range"],
                ent_coef=config["ent_coef"],
                verbose=0,
                device=device,
            )
        elif self.algorithm_name == "A2C":
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                ent_coef=config["ent_coef"],
                max_grad_norm=config["max_grad_norm"],
                verbose=0,
                device=device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
        
        # Train model
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
        
        print(f"\nResults:")
        print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        
        # Save model
        model_path = os.path.join(self.log_dir, f"{self.algorithm_name.lower()}_{config['name']}")
        model.save(model_path)
        print(f"[OK] Model saved to {model_path}")
        
        result = {
            "config": config,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "model_path": model_path,
        }
        
        env.close()
        return result
    
    def train_all_configs(self, total_timesteps: int = 50000) -> Dict:
        """Train with all hyperparameter configurations."""
        print("\n" + "="*80)
        print(f"{self.algorithm_name} HYPERPARAMETER TUNING - 10 CONFIGURATIONS")
        print("="*80)
        
        for config in self.hyperparameter_configs:
            result = self.train_single_config(config, total_timesteps)
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
        
        summary_path = os.path.join(self.log_dir, f"{self.algorithm_name.lower()}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[OK] Training summary saved to {summary_path}")
        
        # Print ranking
        print("\n" + "="*60)
        print(f"{self.algorithm_name} CONFIGURATION RANKING (by Mean Reward)")
        print("="*60)
        
        ranked = sorted(self.results.items(), 
                       key=lambda x: x[1]["mean_reward"], 
                       reverse=True)
        
        for rank, (config_name, result) in enumerate(ranked, 1):
            print(f"{rank:2d}. {config_name:30s} | "
                  f"Reward: {result['mean_reward']:7.3f} ± {result['std_reward']:6.3f}")


class PPOTrainer(PolicyGradientTrainer):
    """Trainer for Proximal Policy Optimization."""
    
    def __init__(self, log_dir="models/pg"):
        super().__init__("PPO", log_dir)
    
    def _generate_hp_configs(self) -> List[Dict]:
        """Generate 10+ PPO hyperparameter configurations."""
        configs = [
            # Config 1: Default baseline
            {
                "name": "default",
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
            },
            # Config 2: Higher learning rate
            {
                "name": "high_lr",
                "learning_rate": 1e-3,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
            },
            # Config 3: Lower learning rate
            {
                "name": "low_lr",
                "learning_rate": 1e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
            },
            # Config 4: Larger batch size
            {
                "name": "large_batch",
                "learning_rate": 3e-4,
                "n_steps": 4096,
                "batch_size": 128,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
            },
            # Config 5: Smaller batch size
            {
                "name": "small_batch",
                "learning_rate": 3e-4,
                "n_steps": 1024,
                "batch_size": 32,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
            },
            # Config 6: Higher entropy coefficient (exploration)
            {
                "name": "high_entropy",
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
            },
            # Config 7: Higher clip range (more conservative)
            {
                "name": "high_clip",
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.3,
                "ent_coef": 0.0,
            },
            # Config 8: Lower clip range (more aggressive)
            {
                "name": "low_clip",
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.1,
                "ent_coef": 0.0,
            },
            # Config 9: More training epochs
            {
                "name": "many_epochs",
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 20,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
            },
            # Config 10: High discount factor
            {
                "name": "high_gamma",
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.999,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
            },
        ]
        return configs


class A2CTrainer(PolicyGradientTrainer):
    """Trainer for Advantage Actor-Critic."""
    
    def __init__(self, log_dir="models/pg"):
        super().__init__("A2C", log_dir)
    
    def _generate_hp_configs(self) -> List[Dict]:
        """Generate 10+ A2C hyperparameter configurations."""
        configs = [
            # Config 1: Default baseline
            {
                "name": "default",
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "max_grad_norm": 0.5,
            },
            # Config 2: Higher learning rate
            {
                "name": "high_lr",
                "learning_rate": 1e-3,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "max_grad_norm": 0.5,
            },
            # Config 3: Lower learning rate
            {
                "name": "low_lr",
                "learning_rate": 1e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "max_grad_norm": 0.5,
            },
            # Config 4: More steps per update
            {
                "name": "more_steps",
                "learning_rate": 7e-4,
                "n_steps": 10,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "max_grad_norm": 0.5,
            },
            # Config 5: Fewer steps per update
            {
                "name": "fewer_steps",
                "learning_rate": 7e-4,
                "n_steps": 2,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "max_grad_norm": 0.5,
            },
            # Config 6: Higher entropy coefficient
            {
                "name": "high_entropy",
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.01,
                "max_grad_norm": 0.5,
            },
            # Config 7: Lower GAE lambda
            {
                "name": "low_gae",
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 0.9,
                "ent_coef": 0.0,
                "max_grad_norm": 0.5,
            },
            # Config 8: Higher discount factor
            {
                "name": "high_gamma",
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.999,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "max_grad_norm": 0.5,
            },
            # Config 9: Stricter gradient clipping
            {
                "name": "strict_clip",
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "max_grad_norm": 0.1,
            },
            # Config 10: Combined settings
            {
                "name": "balanced",
                "learning_rate": 1e-3,
                "n_steps": 8,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.005,
                "max_grad_norm": 0.5,
            },
        ]
        return configs


def main():
    """Main training function for policy gradient methods."""
    
    # Train PPO
    print("\n" + "="*80)
    print("TRAINING PPO (PROXIMAL POLICY OPTIMIZATION)")
    print("="*80)
    ppo_trainer = PPOTrainer()
    ppo_trainer.train_all_configs(total_timesteps=50000)
    
    # Train A2C
    print("\n" + "="*80)
    print("TRAINING A2C (ADVANTAGE ACTOR-CRITIC)")
    print("="*80)
    a2c_trainer = A2CTrainer()
    a2c_trainer.train_all_configs(total_timesteps=50000)

    from training.reinforce_training import REINFORCETrainer

    print("\n" + "="*80)
    print("TRAINING REINFORCE (VANILLA POLICY GRADIENT)")
    print("="*80)
    REINFORCETrainer(log_dir="models/pg").train_all_configs(num_episodes=500)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

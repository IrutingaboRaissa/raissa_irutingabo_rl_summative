"""
Visualization module for NutriVision Environment
Supports both Pygame (2D) and terminal-based rendering
"""

import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class EnvironmentVisualizer:
    """Visualizes the NutriVision environment state."""
    
    def __init__(self, window_width=1200, window_height=800):
        self.width = window_width
        self.height = window_height
        self.fig = None
        self.ax = None
        
        # Color scheme
        self.colors = {
            "background": "#f5f5f5",
            "positive": "#4CAF50",
            "negative": "#F44336",
            "neutral": "#2196F3",
            "warning": "#FF9800",
            "text": "#333333",
        }
    
    def visualize_static_random_actions(self, env, num_steps=10, save_path="visualizations/random_agent.png"):
        """
        Create a visualization of the agent taking random actions.
        Shows environment state progression without training.
        """
        obs, _ = env.reset()
        states = []
        
        for step in range(num_steps):
            action = env.action_space.sample() # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            states.append({
                "step": step,
                "action": ["Accept", "Lower Cal", "Higher Protein", "Skip"][action],
                "reward": reward,
                "food": info.get("food_name", "Unknown"),
                "daily_cal": info.get("daily_calories", 0),
                "daily_protein": info.get("daily_protein", 0),
                "num_meals": info.get("num_meals", 0),
            })
            
            if terminated or truncated:
                break
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("NutriVision Agent - Random Actions (No Training)", fontsize=16, fontweight="bold")
        
        # 1. Action sequence
        ax = axes[0, 0]
        actions = [s["action"] for s in states]
        colors_map = {"Accept": self.colors["positive"], "Lower Cal": self.colors["warning"],
                      "Higher Protein": self.colors["neutral"], "Skip": self.colors["negative"]}
        action_colors = [colors_map[a] for a in actions]
        ax.barh(range(len(actions)), [1]*len(actions), color=action_colors)
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels([f"Step {s['step']}: {s['action']}" for s in states], fontsize=9)
        ax.set_xlabel("Action Sequence")
        ax.set_title("Actions Taken Over Time")
        ax.set_xlim(0, 1.2)
        
        # 2. Reward progression
        ax = axes[0, 1]
        rewards = np.array([s["reward"] for s in states])
        cumulative_rewards = np.cumsum(rewards)
        ax.plot(range(len(states)), cumulative_rewards, marker='o', color=self.colors["neutral"], linewidth=2)
        ax.fill_between(range(len(states)), cumulative_rewards, alpha=0.3, color=self.colors["neutral"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Cumulative Reward Over Episode")
        ax.grid(True, alpha=0.3)
        
        # 3. Calorie accumulation
        ax = axes[1, 0]
        calories = [s["daily_cal"] for s in states]
        ax.plot(range(len(states)), calories, marker='s', color=self.colors["positive"], linewidth=2, label="Actual")
        # Add target line
        ax.axhline(y=env.calorie_target, color=self.colors["warning"], linestyle='--', label="Target")
        ax.set_xlabel("Step")
        ax.set_ylabel("Calories (kcal)")
        ax.set_title("Daily Calorie Accumulation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Meals logged and protein
        ax = axes[1, 1]
        meals = [s["num_meals"] for s in states]
        proteins = [s["daily_protein"] for s in states]
        ax2 = ax.twinx()
        
        ax.bar(range(len(states)), meals, alpha=0.6, color=self.colors["neutral"], label="Meals Logged")
        ax2.plot(range(len(states)), proteins, marker='^', color=self.colors["positive"], linewidth=2, label="Protein (g)")
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Meals Logged", color=self.colors["neutral"])
        ax2.set_ylabel("Protein (g)", color=self.colors["positive"])
        ax.set_title("Meal Count & Protein Intake")
        ax.tick_params(axis='y', labelcolor=self.colors["neutral"])
        ax2.tick_params(axis='y', labelcolor=self.colors["positive"])
        
        plt.tight_layout()
        
        import os
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Random agent visualization saved to {save_path}")
        plt.close()
        
        return states
    
    def plot_training_results(self, results: Dict[str, Any], save_path="visualizations/training_results.png"):
        """
        Plot comprehensive training results across all algorithms.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("NutriVision RL Training Results - All Algorithms", fontsize=16, fontweight="bold")
        
        # 1. Cumulative rewards
        ax = axes[0, 0]
        for algo_name, data in results.items():
            if "rewards" in data:
                ax.plot(data["rewards"], label=algo_name, linewidth=2, alpha=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Cumulative Reward Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Loss curves
        ax = axes[0, 1]
        for algo_name, data in results.items():
            if "loss" in data:
                ax.plot(data["loss"], label=algo_name, linewidth=2, alpha=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        
        # 3. Moving average reward (100-episode window)
        ax = axes[1, 0]
        window = 10
        for algo_name, data in results.items():
            if "rewards" in data:
                rewards = np.array(data["rewards"])
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(moving_avg, label=algo_name, linewidth=2, alpha=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Moving Average Reward")
        ax.set_title(f"Moving Average Reward ({window}-episode window)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Final performance comparison
        ax = axes[1, 1]
        algo_names = list(results.keys())
        final_rewards = []
        for algo_name in algo_names:
            if "rewards" in results[algo_name]:
                final_rewards.append(np.mean(results[algo_name]["rewards"][-20:]))
            else:
                final_rewards.append(0)
        
        colors = [self.colors["positive"], self.colors["neutral"], self.colors["warning"], self.colors["negative"]]
        bars = ax.bar(algo_names, final_rewards, color=colors[:len(algo_names)])
        ax.set_ylabel("Average Final Reward")
        ax.set_title("Algorithm Performance Comparison (Final 20 Episodes)")
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        import os
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Training results visualization saved to {save_path}")
        plt.close()
    
    def plot_hyperparameter_comparison(self, hp_results: Dict, algorithm: str, 
                                      save_path: str = None):
        """
        Plot hyperparameter tuning results for a single algorithm.
        """
        if save_path is None:
            save_path = f"visualizations/hp_tuning_{algorithm}.png"
        
        # Extract data
        hp_names = list(hp_results.keys())
        avg_rewards = [hp_results[hp]["avg_reward"] for hp in hp_names]
        std_rewards = [hp_results[hp]["std_reward"] for hp in hp_names]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(hp_names))
        bars = ax.bar(x_pos, avg_rewards, yerr=std_rewards, capsize=5, 
                     color=self.colors["neutral"], alpha=0.8, edgecolor="black")
        
        ax.set_xlabel("Hyperparameter Configuration", fontsize=12)
        ax.set_ylabel("Average Reward", fontsize=12)
        ax.set_title(f"Hyperparameter Tuning Results - {algorithm}", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Config {i+1}" for i in range(len(hp_names))], fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Highlight best configuration
        best_idx = np.argmax(avg_rewards)
        bars[best_idx].set_color(self.colors["positive"])
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        
        import os
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Hyperparameter tuning visualization saved to {save_path}")
        plt.close()

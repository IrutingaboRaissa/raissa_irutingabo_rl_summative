"""
Main playback script for NutriVision RL Agent
Runs the best-performing trained agent in the environment with visualization
"""

import os
import json
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import torch

from environment.custom_env import NutriVisionEnv
from environment.rendering import EnvironmentVisualizer
from training.reinforce_training import PolicyNetwork


def _summary_path_for_algorithm(algo: str) -> Optional[str]:
    """Resolve training summary JSON path for an algorithm."""
    if algo == "dqn":
        for p in ("models/dqn/dqn_summary.json", "models/dqn/training_summary.json"):
            if os.path.exists(p):
                return p
        return None
    pg_paths = {
        "ppo": "models/pg/ppo_summary.json",
        "a2c": "models/pg/a2c_summary.json",
        "reinforce": "models/pg/reinforce_summary.json",
    }
    p = pg_paths.get(algo)
    if p and os.path.exists(p):
        return p
    return None


def load_best_model(algorithm: str = "all"):
    """Load the best performing model from training results."""

    if algorithm == "all":
        best_models: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        best_overall: Dict[str, Any] = {
            "algorithm": None,
            "config": None,
            "reward": -float("inf"),
        }

        for algo in ["dqn", "reinforce", "ppo", "a2c"]:
            sp = _summary_path_for_algorithm(algo)
            if not sp:
                continue
            with open(sp, "r") as f:
                summary = json.load(f)

            best_config = max(summary.items(), key=lambda x: x[1]["mean_reward"])
            best_models[algo] = best_config

            if best_config[1]["mean_reward"] > best_overall["reward"]:
                best_overall["algorithm"] = algo
                best_overall["config"] = best_config[0]
                best_overall["reward"] = best_config[1]["mean_reward"]

        return best_overall, best_models

    sp = _summary_path_for_algorithm(algorithm)
    if not sp:
        raise FileNotFoundError(
            f"No training results found for {algorithm} (expected summary under models/dqn or models/pg)"
        )
    with open(sp, "r") as f:
        summary = json.load(f)

    best_config = max(summary.items(), key=lambda x: x[1]["mean_reward"])
    return (
        {
            "algorithm": algorithm,
            "config": best_config[0],
            "reward": best_config[1]["mean_reward"],
        },
        None,
    )


def play_episode(
    env: NutriVisionEnv,
    model,
    algorithm: str,
    render: bool = True,
    pygame_viz=None,
) -> Tuple[float, list]:
    """
    Run one episode with the trained model.
    """
    obs, _ = env.reset()
    episode_reward = 0
    trajectory = []
    goal_types = ["Weight Loss", "Weight Gain", "Maintenance"]

    print(f"\n{'='*80}")
    print(f"EPISODE START - Goal: {goal_types[env.goal_type]}")
    print(f"{'='*80}")

    step = 0
    while True:
        if algorithm == "reinforce":
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs = model(state_tensor)
            action = torch.argmax(probs, dim=1).item()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        action_names = ["Accept", "Lower Calorie", "Higher Protein", "Skip"]

        traj_entry = {
            "step": step,
            "action": action_names[action],
            "reward": reward,
            "food": info.get("food_name", "Unknown"),
            "daily_cal": info.get("daily_calories", 0),
            "daily_protein": info.get("daily_protein", 0),
            "num_meals": info.get("num_meals", 0),
        }
        trajectory.append(traj_entry)

        if pygame_viz is not None:
            import pygame

            pygame_viz.render_episode(env, action, reward)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    truncated = True
                    break

        if render:
            print(f"\nStep {step + 1}:")
            print(f" Food: {info.get('food_name', 'Unknown')}")
            print(f" Action: {action_names[action]}")
            print(f" Reward: {reward:+.3f}")
            print(f" Daily Calories: {info.get('daily_calories', 0):.0f} / {env.calorie_target}")
            print(f" Daily Protein: {info.get('daily_protein', 0):.1f}g / {env.protein_target:.1f}g")
            print(f" Meals Logged: {info.get('num_meals', 0)}")
            print(f" Cumulative Reward: {episode_reward:+.3f}")

        step += 1

        if terminated or truncated:
            break

    print(f"\n{'='*80}")
    print("EPISODE COMPLETE")
    print(f"{'='*80}")
    print(f"Total Reward: {episode_reward:+.3f}")
    print(f"Meals Logged: {trajectory[-1]['num_meals'] if trajectory else 0}")
    print(f"Final Daily Calories: {trajectory[-1]['daily_cal'] if trajectory else 0:.0f}")
    print(f"Final Daily Protein: {trajectory[-1]['daily_protein'] if trajectory else 0:.1f}g")

    return episode_reward, trajectory


def run_best_agent(
    algorithm: str = "all",
    num_episodes: int = 5,
    render: bool = True,
    use_pygame: bool = False,
):
    """Run the best-performing agent."""
    print("\n" + "="*80)
    print("NutriVision RL Agent - Playback with Best Trained Model")
    print("="*80)

    best_overall, all_best = load_best_model(algorithm)

    if best_overall.get("algorithm") is None:
        raise FileNotFoundError(
            "No training summaries found. Expected DQN under models/dqn/ and "
            "PPO/A2C/REINFORCE under models/pg/."
        )

    print(f"\nBest Algorithm: {best_overall['algorithm'].upper()}")
    print(f"Best Configuration: {best_overall['config']}")
    print(f"Best Mean Reward: {best_overall['reward']:.3f}")

    if all_best:
        print("\nAll Algorithm Results:")
        for algo, (config, stats) in all_best.items():
            print(f" {algo.upper():10s}: {config:25s} (Reward: {stats['mean_reward']:7.3f})")

    algo = best_overall["algorithm"]
    config = best_overall["config"]

    if algo == "reinforce":
        sp = _summary_path_for_algorithm("reinforce")
        if not sp:
            raise FileNotFoundError("reinforce summary not found")
        with open(sp, "r") as f:
            summary = json.load(f)
        hidden_size = int(summary[config]["config"]["hidden_size"])
        model = PolicyNetwork(15, hidden_size, 4)
        model_path = os.path.join("models", "pg", f"reinforce_{config}.pt")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    else:
        from stable_baselines3 import DQN, PPO, A2C

        algo_map = {
            "dqn": DQN,
            "ppo": PPO,
            "a2c": A2C,
        }
        subdir = "dqn" if algo == "dqn" else "pg"
        model_path = os.path.join("models", subdir, f"{algo}_{config}")
        model = algo_map[algo].load(model_path)

    print(f"\nRunning {num_episodes} episodes with best model...")

    env = NutriVisionEnv()
    episode_rewards: List[float] = []
    all_trajectories = []

    pygame_viz = None
    if use_pygame:
        from environment.pygame_viz import NutriVisionVisualizer

        pygame_viz = NutriVisionVisualizer()

    try:
        for _ in range(num_episodes):
            reward, trajectory = play_episode(
                env, model, algo, render=render, pygame_viz=pygame_viz
            )
            episode_rewards.append(reward)
            all_trajectories.append(trajectory)
    finally:
        env.close()
        if pygame_viz is not None:
            pygame_viz.close()

    print(f"\n{'='*80}")
    print("PLAYBACK SUMMARY")
    print(f"{'='*80}")
    print(f"Algorithm: {algo.upper()}")
    print(f"Configuration: {config}")
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Min Reward: {np.min(episode_rewards):.3f}")
    print(f"Max Reward: {np.max(episode_rewards):.3f}")

    viz_env = NutriVisionEnv()
    try:
        visualizer = EnvironmentVisualizer()
        visualizer.visualize_static_random_actions(
            viz_env,
            num_steps=10,
            save_path="visualizations/agent_playback.png",
        )
    finally:
        viz_env.close()

    return episode_rewards, all_trajectories


def main():
    parser = argparse.ArgumentParser(description="NutriVision RL Agent Playback")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all",
        choices=["all", "dqn", "reinforce", "ppo", "a2c"],
        help="Algorithm to use (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable terminal rendering",
    )
    parser.add_argument(
        "--pygame",
        action="store_true",
        help="Open pygame visualization during play (NutriVisionVisualizer)",
    )

    args = parser.parse_args()

    run_best_agent(
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        render=not args.no_render,
        use_pygame=args.pygame,
    )


if __name__ == "__main__":
    main()

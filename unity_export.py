"""
Export NutriVision episode trajectories to Unity-friendly JSON.

This script lets you replay agent behavior in a Unity scene without changing
the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from environment.custom_env import NutriVisionEnv


ACTION_NAMES = ["Accept", "Lower Calorie", "Higher Protein", "Skip"]


def _summary_path_for_algorithm(algo: str) -> Optional[str]:
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


def _load_best_config(algorithm: str) -> Tuple[str, Dict[str, Any]]:
    sp = _summary_path_for_algorithm(algorithm)
    if not sp:
        raise FileNotFoundError(f"Summary file not found for {algorithm}")
    with open(sp, "r", encoding="utf-8") as f:
        summary = json.load(f)
    config_name, stats = max(summary.items(), key=lambda kv: kv[1]["mean_reward"])
    return config_name, stats


def _load_policy(algorithm: str):
    from training.reinforce_training import PolicyNetwork
    import torch

    config_name, stats = _load_best_config(algorithm)
    if algorithm == "reinforce":
        hidden_size = int(stats["config"]["hidden_size"])
        model = PolicyNetwork(15, hidden_size, 4)
        model_path = os.path.join("models", "pg", f"reinforce_{config_name}.pt")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model, config_name

    from stable_baselines3 import A2C, DQN, PPO

    algo_map = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
    subdir = "dqn" if algorithm == "dqn" else "pg"
    model_path = os.path.join("models", subdir, f"{algorithm}_{config_name}")
    model = algo_map[algorithm].load(model_path, device="cpu")
    return model, config_name


def _predict_action(model, algorithm: str, obs: np.ndarray) -> int:
    if algorithm == "reinforce":
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = model(state_tensor)
        return int(torch.argmax(probs, dim=1).item())
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def _episode_to_json(
    env: NutriVisionEnv,
    predict_action: Callable[[np.ndarray], int],
    episode_idx: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0

    goal_names = ["Weight Loss", "Weight Gain", "Maintenance"]
    goal_name = goal_names[int(env.goal_type)]

    step_idx = 0
    while True:
        action = predict_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        trajectory.append(
            {
                "step": step_idx,
                "action_index": action,
                "action_name": ACTION_NAMES[action],
                "reward": float(reward),
                "food_name": str(info.get("food_name", "Unknown")),
                "daily_calories": float(info.get("daily_calories", 0.0)),
                "daily_protein": float(info.get("daily_protein", 0.0)),
                "num_meals": int(info.get("num_meals", 0)),
                "observation": [float(x) for x in obs.tolist()],
                "next_observation": [float(x) for x in next_obs.tolist()],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        obs = next_obs
        step_idx += 1
        if terminated or truncated:
            break

    return {
        "episode_index": episode_idx,
        "seed": seed,
        "goal_type": int(env.goal_type),
        "goal_name": goal_name,
        "calorie_target": float(env.calorie_target),
        "protein_target": float(env.protein_target),
        "total_reward": float(total_reward),
        "num_steps": len(trajectory),
        "steps": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trajectories for Unity replay")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["dqn", "reinforce", "ppo", "a2c"],
        help="Algorithm used to select the policy to export",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to export")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/unity/replay_trajectories.json",
        help="Output JSON path for Unity replay",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=100,
        help="Base reset seed for deterministic exports (seed_base + episode_index)",
    )
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Roll random actions (no SB3/REINFORCE checkpoints). For Unity replay demo only.",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    env = NutriVisionEnv()
    try:
        if args.random_policy:
            best_config = "random_demo"

            def predict_action(obs: np.ndarray) -> int:
                return int(env.action_space.sample())

            algo_label = "random"
        else:
            model, best_config = _load_policy(args.algorithm)
            algo_label = args.algorithm

            def predict_action(obs: np.ndarray) -> int:
                return _predict_action(model, args.algorithm, obs)

        episodes = [
            _episode_to_json(env, predict_action, i, args.seed_base + i)
            for i in range(args.episodes)
        ]
    finally:
        env.close()

    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "NutriVision Africa",
        "algorithm": algo_label,
        "best_config": best_config,
        "action_names": ACTION_NAMES,
        "episodes": episodes,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] Unity replay export saved: {args.out}")
    print(
        f"[OK] Episodes: {len(episodes)} | Policy: {algo_label.upper()} | Config: {best_config}"
    )


if __name__ == "__main__":
    main()


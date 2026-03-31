"""
Record MP4 videos for each algorithm using the best-performing checkpoint.

This uses the existing Pygame GUI rendering in environment/pygame_viz.py and captures
frames with OpenCV (cv2.VideoWriter).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from environment.custom_env import NutriVisionEnv
from main import load_best_model, _summary_path_for_algorithm  # reuse checkpoint selection
from training.reinforce_training import PolicyNetwork


def _load_reinforce_model(config_name: str) -> PolicyNetwork:
    sp = _summary_path_for_algorithm("reinforce")
    if not sp:
        raise FileNotFoundError("reinforce summary not found under models/pg/")

    import json

    with open(sp, "r") as f:
        summary = json.load(f)

    hidden_size = int(summary[config_name]["config"]["hidden_size"])
    model = PolicyNetwork(15, hidden_size, 4)

    model_path = Path("models/pg") / f"reinforce_{config_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"REINFORCE checkpoint missing: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def record_for_algorithm(
    algo: str,
    *,
    out_dir: Path,
    steps: Optional[int],
    verbose: bool,
    headless: bool,
) -> Path:
    # Headless: make pygame use the dummy video driver in CI/no-display setups.
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    # Import after SDL_VIDEODRIVER is set (pygame reads it at import/display time).
    from environment.pygame_viz import NutriVisionVisualizer

    out_dir = out_dir / f"rl_agent_videos_{algo}"
    _ensure_dir(out_dir)
    out_path = out_dir / f"rl-video-episode-{algo}.mp4"

    best, _ = load_best_model(algo)
    config_name = best["config"]

    env = NutriVisionEnv()
    viz = NutriVisionVisualizer()

    try:
        if algo == "reinforce":
            model: Any = _load_reinforce_model(config_name)
        else:
            from stable_baselines3 import DQN, PPO, A2C

            algo_map = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
            subdir = "dqn" if algo == "dqn" else "pg"
            ckpt_path = Path("models") / subdir / f"{algo}_{config_name}"
            if not ckpt_path.exists():
                # SB3 may store checkpoints with a .zip extension; try the common one.
                if (ckpt_path.with_suffix(".zip")).exists():
                    ckpt_path = ckpt_path.with_suffix(".zip")
            model = algo_map[algo].load(str(ckpt_path))

        max_frames = steps if steps is not None else None
        viz.record_episode_video(
            env,
            model,
            algo,
            output_path=str(out_path),
            show_fps=False,
            verbose=verbose,
            max_frames=max_frames,
        )
    finally:
        env.close()
        viz.close()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record algorithm gameplay videos (MP4).")
    parser.add_argument("--out_dir", type=str, default="visualizations", help="Output root directory.")
    parser.add_argument("--steps", type=int, default=20, help="Max env steps per episode (default: 20).")
    parser.add_argument("--verbose", action="store_true", help="Print per-step info to terminal.")
    parser.add_argument("--headless", action="store_true", help="Use SDL_VIDEODRIVER=dummy.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    steps = args.steps

    print("\n" + "=" * 80)
    print("Recording videos for: DQN, PPO, A2C, REINFORCE")
    print("=" * 80)

    for algo in ["dqn", "ppo", "a2c", "reinforce"]:
        print(f"\n--- {algo.upper()} ---")
        p = record_for_algorithm(
            algo,
            out_dir=out_dir,
            steps=steps,
            verbose=args.verbose,
            headless=args.headless,
        )
        print("Saved:", p)


if __name__ == "__main__":
    main()


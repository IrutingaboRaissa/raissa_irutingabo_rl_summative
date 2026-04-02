"""Record gameplay videos (AVI/MJPEG) for each algorithm — best checkpoint from summaries."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from environment.custom_env import NutriVisionEnv
from main import load_best_model, _summary_path_for_algorithm
from training.reinforce_training import PolicyNetwork


def _load_reinforce_model(config_name: str) -> PolicyNetwork:
    sp = _summary_path_for_algorithm("reinforce")
    if not sp:
        raise FileNotFoundError("reinforce summary not found under models/pg/")

    import json

    with open(sp, "r", encoding="utf-8") as f:
        summary = json.load(f)

    hidden_size = int(summary[config_name]["config"]["hidden_size"])
    model = PolicyNetwork(15, hidden_size, 4)

    model_path = Path("models/pg") / f"reinforce_{config_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"REINFORCE checkpoint missing: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def _safe_filename_part(name: str) -> str:
    """Alphanumeric + underscore only for cross-platform paths."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_")
    return s or "config"


def _build_output_path(
    root: Path,
    algo: str,
    config_name: str,
    max_steps: Optional[int],
    *,
    timestamp: bool,
) -> Path:
    algo_l = algo.lower()
    sub = root / algo_l
    sub.mkdir(parents=True, exist_ok=True)
    cfg = _safe_filename_part(config_name)
    steps_part = f"max{max_steps}steps" if max_steps is not None else "max_unlimited"
    base = f"NutriVision_{algo.upper()}_best_{cfg}_{steps_part}"
    if timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = f"{base}_{ts}"
    return sub / f"{base}.avi"


def record_for_algorithm(
    algo: str,
    *,
    out_dir: Path,
    steps: Optional[int],
    verbose: bool,
    headless: bool,
    timestamp: bool,
) -> Path:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from environment.pygame_viz import NutriVisionVisualizer

    best, _ = load_best_model(algo)
    config_name = best["config"]
    out_path = _build_output_path(out_dir, algo, config_name, steps, timestamp=timestamp)

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
            if not ckpt_path.exists() and ckpt_path.with_suffix(".zip").exists():
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
    parser = argparse.ArgumentParser(description="Record algorithm videos (AVI, MJPEG).")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="videos",
        help="Output root; each algorithm gets a subfolder (e.g. videos/dqn/).",
    )
    parser.add_argument("--steps", type=int, default=200, help="Max env steps per episode.")
    parser.add_argument("--verbose", action="store_true", help="Print per-step info.")
    parser.add_argument("--headless", action="store_true", help="SDL_VIDEODRIVER=dummy.")
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append UTC timestamp to filenames so repeated runs do not overwrite.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all",
        choices=["all", "dqn", "ppo", "a2c", "reinforce"],
        help="Record one algorithm or all four.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    algos: Sequence[str] = (
        ["dqn", "ppo", "a2c", "reinforce"] if args.algorithm == "all" else [args.algorithm]
    )

    print("\n" + "=" * 80)
    print(f"Recording videos (AVI) → {out_dir.resolve()}")
    print("Subfolders:", ", ".join(a.upper() for a in algos))
    print("=" * 80)

    for algo in algos:
        print(f"\n--- {algo.upper()} ---")
        p = record_for_algorithm(
            algo,
            out_dir=out_dir,
            steps=args.steps,
            verbose=args.verbose,
            headless=args.headless,
            timestamp=args.timestamp,
        )
        print("Saved:", p)


if __name__ == "__main__":
    main()

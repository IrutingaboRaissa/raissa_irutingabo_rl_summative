"""
Local end-to-end runner for NutriVision RL.

Single entry point that can:
1) Train all algorithms
2) Generate analysis plots + detailed result tables
3) Record one video per algorithm (best checkpoint)
4) Collect all artifacts into outputs/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def _run(cmd: List[str], env: dict | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def _copytree_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _glob_copy(src_root: Path, patterns: Iterable[str], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for p in src_root.rglob(pattern):
            if p.is_file():
                shutil.copy2(p, dst / p.name)


def collect_outputs() -> None:
    out_models = OUTPUTS / "models"
    out_visualizations = OUTPUTS / "visualizations"
    out_tables = OUTPUTS / "tables"
    out_videos = OUTPUTS / "videos"
    out_reports = OUTPUTS / "reports"

    for p in (out_models, out_visualizations, out_tables, out_videos, out_reports):
        p.mkdir(parents=True, exist_ok=True)

    _copytree_if_exists(ROOT / "models", out_models)
    _copytree_if_exists(ROOT / "visualizations", out_visualizations)

    _copy_if_exists(ROOT / "TRAINING_REPORT.txt", out_reports / "TRAINING_REPORT.txt")
    _copy_if_exists(
        ROOT / "models" / "master_training_results.json",
        out_reports / "master_training_results.json",
    )

    if (ROOT / "visualizations").exists():
        _glob_copy(ROOT / "visualizations", ("*table*.png",), out_tables)
        _glob_copy(ROOT / "visualizations", ("**/*.mp4",), out_videos)

    for p in ROOT.glob("rl_agent_videos_*"):
        if p.is_dir():
            _glob_copy(p, ("**/*.mp4",), out_videos)

    print("\n=== Saved artifacts ===")
    print("models       :", out_models)
    print("visualizations:", out_visualizations)
    print("tables       :", out_tables)
    print("videos       :", out_videos)
    print("reports      :", out_reports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full local RL pipeline and save artifacts.")
    parser.add_argument("--skip-training", action="store_true", help="Skip train_all.py")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analyze_results.py")
    parser.add_argument("--skip-videos", action="store_true", help="Skip record_videos.py")
    parser.add_argument("--skip-random-demo", action="store_true", help="Skip random_demo.py")
    parser.add_argument("--timesteps", type=int, default=5000, help="SB3 timesteps per config.")
    parser.add_argument("--reinforce-episodes", type=int, default=30, help="REINFORCE episodes per config.")
    parser.add_argument("--video-steps", type=int, default=20, help="Max steps per recorded episode.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Allow SB3 trainers to use GPU when available.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use SDL dummy driver for video recording (no display).",
    )
    parser.add_argument("--video-verbose", action="store_true", help="Verbose per-step video logs.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["TOTAL_TIMESTEPS"] = str(args.timesteps)
    env["REINFORCE_EPISODES"] = str(args.reinforce_episodes)
    env["RL_USE_GPU"] = "1" if args.use_gpu else "0"

    print("=" * 90)
    print("NUTRIVISION RL - LOCAL PIPELINE")
    print("=" * 90)
    print(f"ROOT: {ROOT}")
    print(
        f"TOTAL_TIMESTEPS={env['TOTAL_TIMESTEPS']} | "
        f"REINFORCE_EPISODES={env['REINFORCE_EPISODES']} | RL_USE_GPU={env['RL_USE_GPU']}"
    )

    if not args.skip_training:
        _run([sys.executable, "train_all.py"], env=env)
    else:
        print("\n[SKIP] Training")

    if not args.skip_analysis:
        _run([sys.executable, "analyze_results.py"], env=env)
    else:
        print("\n[SKIP] Analysis")

    if not args.skip_random_demo:
        _run(
            [
                sys.executable,
                "random_demo.py",
                "--steps",
                "30",
                "--output",
                "visualizations/random_agent_demo.png",
            ],
            env=env,
        )
    else:
        print("\n[SKIP] Random demo")

    if not args.skip_videos:
        cmd = [
            sys.executable,
            "record_videos.py",
            "--out_dir",
            "visualizations",
            "--steps",
            str(args.video_steps),
        ]
        if args.headless:
            cmd.append("--headless")
        if args.video_verbose:
            cmd.append("--verbose")
        _run(cmd, env=env)
    else:
        print("\n[SKIP] Videos")

    collect_outputs()

    print("\nDone.")


if __name__ == "__main__":
    main()

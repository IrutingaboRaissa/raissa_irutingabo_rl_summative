"""Single local entry: train → analyze → optional demo → optional videos → outputs/."""

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
        _glob_copy(ROOT / "visualizations", ("**/*.avi",), out_videos)
    if (ROOT / "videos").exists():
        for p in (ROOT / "videos").rglob("*.avi"):
            if p.is_file():
                rel = p.relative_to(ROOT / "videos")
                dest = out_videos / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)

    print("\n=== Saved artifacts ===")
    print("models       :", out_models)
    print("visualizations:", out_visualizations)
    print("tables       :", out_tables)
    print("videos       :", out_videos)
    print("reports      :", out_reports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full local NutriVision RL pipeline.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--with-videos",
        action="store_true",
        help="Run record_videos.py (AVI). Skipped by default — use for demos.",
    )
    parser.add_argument("--skip-videos", action="store_true", help="Force skip videos")
    parser.add_argument("--no-videos", action="store_true", help="Alias for --skip-videos")
    parser.add_argument("--skip-random-demo", action="store_true")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--reinforce-episodes", type=int, default=500)
    parser.add_argument("--video-steps", type=int, default=200)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video-verbose", action="store_true")
    args = parser.parse_args()

    skip_videos = args.skip_videos or args.no_videos or not args.with_videos

    env = os.environ.copy()
    env["TOTAL_TIMESTEPS"] = str(args.timesteps)
    env["REINFORCE_EPISODES"] = str(args.reinforce_episodes)
    env["RL_USE_GPU"] = "1" if args.use_gpu else "0"

    print("=" * 90)
    print("NUTRIVISION RL - LOCAL PIPELINE")
    print("=" * 90)
    print(f"ROOT: {ROOT}")

    if not args.skip_training:
        _run([sys.executable, "train_all.py"], env=env)
    if not args.skip_analysis:
        _run([sys.executable, "analyze_results.py"], env=env)
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
    if not skip_videos:
        cmd = [
            sys.executable,
            "record_videos.py",
            "--out_dir",
            "videos",
            "--steps",
            str(args.video_steps),
        ]
        if args.headless:
            cmd.append("--headless")
        if args.video_verbose:
            cmd.append("--verbose")
        _run(cmd, env=env)

    collect_outputs()
    print("\nDone.")


if __name__ == "__main__":
    main()

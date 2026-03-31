"""
Run a random agent in NutriVisionEnv and save a screenshot (pygame headless).
Set SDL_VIDEODRIVER=dummy before pygame initializes.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Random agent demo (headless pygame screenshot)")
    parser.add_argument("--steps", type=int, default=30, help="Number of env steps to animate")
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations/random_agent_demo.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    import pygame

    from environment.custom_env import NutriVisionEnv
    from environment.pygame_viz import NutriVisionVisualizer

    env = NutriVisionEnv()
    viz = NutriVisionVisualizer()
    try:
        obs, _ = env.reset()
        for _ in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            viz.render_episode(env, action, reward)
            if terminated or truncated:
                obs, _ = env.reset()
        pygame.image.save(viz.screen, args.output)
        print(f"Saved: {args.output}")
    finally:
        viz.close()
        env.close()


if __name__ == "__main__":
    main()

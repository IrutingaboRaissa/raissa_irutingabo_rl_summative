"""
Watch NutriVision like a mini-game: real Pygame window + random moves.
No trained checkpoints required — use this when you just want to *see* the UI move.

If you previously set SDL_VIDEODRIVER=dummy (headless screenshots), this script clears it
so a normal desktop window can open.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="NutriVision Quest — watch mode (random agent)")
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=300,
        help="Milliseconds between random environment steps (default: 300)",
    )
    args = parser.parse_args()

    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
        print("Note: cleared SDL_VIDEODRIVER=dummy so a real window can open.\n")
    else:
        os.environ.pop("SDL_VIDEODRIVER", None)

    import pygame

    from environment.custom_env import NutriVisionEnv
    from environment.pygame_viz import NutriVisionVisualizer

    env = NutriVisionEnv()
    viz = NutriVisionVisualizer()
    env.reset()

    STEP_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(STEP_EVENT, max(80, args.delay_ms))

    last_action: int | None = None
    last_reward = 0.0

    print("NutriVision Quest — watch demo (random actions)")
    print("A window should open. Close it or press ESC to quit.\n")

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == STEP_EVENT:
                    last_action = int(env.action_space.sample())
                    _, last_reward, term, trunc, _ = env.step(last_action)
                    if term or trunc:
                        env.reset()
                        last_action = None
                        last_reward = 0.0

            viz.render_episode(env, last_action, last_reward, show_fps=True)
    finally:
        pygame.time.set_timer(STEP_EVENT, 0)
        viz.close()
        env.close()


if __name__ == "__main__":
    main()

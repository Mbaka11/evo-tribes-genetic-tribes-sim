"""
EvoTribes — Demo Script (Iteration 1)
=======================================

Instantiates the environment with default config, runs random agents,
and renders in real time via Pygame.

Usage
-----
    python -m scripts.demo

Press the window close button (X) or Ctrl-C to exit.
"""

import sys
import os

# Ensure project root is on the path so `src` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.tribes_env import TribesEnv

# ---------------------------------------------------------------------------
# Configuration — override any default here
# ---------------------------------------------------------------------------
DEMO_CONFIG = {
    "grid_width": 20,
    "grid_height": 20,
    "num_agents": 6,
    "num_tribes": 2,
    "num_food": 12,
    "max_steps": 300,
}


def main() -> None:
    env = TribesEnv(config=DEMO_CONFIG, render_mode="human")
    observations, info = env.reset(seed=42)

    print("EvoTribes — Iteration 1 demo")
    print(f"Grid {env.grid_w}x{env.grid_h}, {env.num_agents} agents, "
          f"{env.num_tribes} tribes")
    print("Running random actions — close the window to quit.\n")

    done = False
    while not done:
        # Random actions for every agent
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        env.render()

        # Periodic console log
        if env.current_step % 50 == 0:
            print(
                f"  step {env.current_step:>4d}  |  "
                f"alive {info['alive']}/{env.num_agents}  |  "
                f"total energy {info['total_energy']:.1f}"
            )

    print(f"\nEpisode finished at step {env.current_step}.")
    print(f"  Reason: {'all dead' if terminated else 'max steps reached'}")
    print(f"  Alive:  {info['alive']}/{env.num_agents}")
    env.close()


if __name__ == "__main__":
    main()

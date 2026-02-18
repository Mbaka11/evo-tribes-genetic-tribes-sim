"""
EvoTribes — Replay Script (Iteration 3)
==========================================

Load a saved chromosome (.npy file) and replay it in the environment
with Pygame rendering.

Usage
-----
    python -m scripts.replay --checkpoint runs/default/best_final.npy
    python -m scripts.replay --checkpoint runs/exp_001/best_gen0050.npy --seed 99

Press the window close button (X) or Ctrl-C to exit.
"""

import argparse
import sys
import os

import numpy as np

# Ensure project root is on the path so `src` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.tribes_env import TribesEnv, NUM_ACTIONS
from src.policies import MLPPolicy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvoTribes — Replay an evolved agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a saved .npy chromosome file",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Environment seed for the replay episode",
    )
    parser.add_argument(
        "--obs-size", type=int, default=27,
        help="Observation vector length (must match training)",
    )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[16],
        help="MLP hidden layer widths (must match training)",
    )

    args = parser.parse_args()

    # Load chromosome
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    chromosome = np.load(args.checkpoint)
    print(f"Loaded chromosome: {chromosome.shape[0]} parameters")

    # Create environment with rendering
    env = TribesEnv(render_mode="human")
    num_agents = env.num_agents
    obs_size = env.observation_space.shape[0]

    # Verify obs_size matches
    if obs_size != args.obs_size:
        print(f"Warning: env obs_size={obs_size} != --obs-size={args.obs_size}")

    # Create policies — all agents share the evolved chromosome
    policies = []
    for _ in range(num_agents):
        p = MLPPolicy(
            obs_size=obs_size,
            num_actions=NUM_ACTIONS,
            hidden_sizes=args.hidden_sizes,
            deterministic=True,
        )
        p.set_params(chromosome.copy())
        policies.append(p)

    print(f"\nEvoTribes — Replay")
    print(f"Grid {env.grid_w}x{env.grid_h}, {num_agents} agents, "
          f"{env.num_tribes} tribes")
    print(f"Checkpoint: {args.checkpoint}")
    print("Close the window to quit.\n")

    # Run episode
    observations, info = env.reset(seed=args.seed)
    total_reward = 0.0
    done = False

    while not done:
        actions = [
            policies[i].select_action(observations[i])
            for i in range(num_agents)
        ]
        observations, rewards, terminated, truncated, info = env.step(actions)
        total_reward += sum(rewards)
        done = terminated or truncated

        env.render()

        if env.current_step % 50 == 0:
            print(
                f"  step {env.current_step:>4d}  |  "
                f"alive {info['alive']}/{num_agents}  |  "
                f"total reward {total_reward:.1f}"
            )

    print(f"\nEpisode finished at step {env.current_step}.")
    print(f"  Reason: {'all dead' if terminated else 'max steps reached'}")
    print(f"  Alive:  {info['alive']}/{num_agents}")
    print(f"  Total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()

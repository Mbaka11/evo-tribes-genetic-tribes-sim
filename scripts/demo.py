"""
EvoTribes — Demo Script (Iteration 2)
=======================================

Instantiates the environment and runs agents using a selectable policy.
Renders in real time via Pygame.

Usage
-----
    python -m scripts.demo                # default: random policy
    python -m scripts.demo --policy random
    python -m scripts.demo --policy mlp
    python -m scripts.demo --policy mlp --stochastic

Press the window close button (X) or Ctrl-C to exit.
"""

import argparse
import sys
import os

# Ensure project root is on the path so `src` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.tribes_env import TribesEnv, NUM_ACTIONS
from src.policies import RandomPolicy, MLPPolicy, BasePolicy

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


def make_policies(
    policy_name: str,
    num_agents: int,
    obs_size: int,
    stochastic: bool = False,
) -> list[BasePolicy]:
    """Create a list of policies — one per agent.

    Args:
        policy_name: "random" or "mlp".
        num_agents:  How many policies to create.
        obs_size:    Observation vector length (from the environment).
        stochastic:  For MLP, sample from softmax instead of argmax.

    Returns:
        List of BasePolicy instances.

    Example:
        >>> policies = make_policies("random", 6, 27)
        >>> len(policies)
        6
    """
    policies: list[BasePolicy] = []
    for i in range(num_agents):
        if policy_name == "random":
            policies.append(RandomPolicy(num_actions=NUM_ACTIONS, seed=i))
        elif policy_name == "mlp":
            policies.append(
                MLPPolicy(
                    obs_size=obs_size,
                    num_actions=NUM_ACTIONS,
                    hidden_sizes=[16],
                    seed=i,
                    deterministic=not stochastic,
                )
            )
        else:
            raise ValueError(f"Unknown policy: {policy_name!r}")
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoTribes demo")
    parser.add_argument(
        "--policy",
        choices=["random", "mlp"],
        default="random",
        help="Which policy to use (default: random)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="For MLP: sample from softmax instead of argmax",
    )
    args = parser.parse_args()

    env = TribesEnv(config=DEMO_CONFIG, render_mode="human")
    observations, info = env.reset(seed=42)

    # Create one policy per agent
    policies = make_policies(
        args.policy, env.num_agents, env.observation_space.shape[0], args.stochastic
    )

    print(f"EvoTribes — Iteration 2 demo")
    print(f"Grid {env.grid_w}x{env.grid_h}, {env.num_agents} agents, "
          f"{env.num_tribes} tribes")
    print(f"Policy: {policies[0]}")
    print("Close the window to quit.\n")

    done = False
    while not done:
        # Each agent picks an action through its policy
        actions = [
            policies[i].select_action(observations[i])
            for i in range(env.num_agents)
        ]
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

"""
EvoTribes — Fitness Evaluation
================================

Evaluates how well an agent policy performs in the environment.

The fitness of a chromosome (flat parameter vector) is determined by
running one or more episodes in the environment and measuring the
**total reward** accumulated by the agent.

Because the environment is stochastic (random food placement, random
initial positions), we average fitness over several episodes to reduce
noise — this is called **robust evaluation**.

Key concepts
------------
* **Chromosome**: A flat float32 array of policy parameters.
  For MLPPolicy with default config, this is 533 numbers.
* **Fitness**: A single scalar summarising performance.
  Higher is better.
* **Robust evaluation**: Running multiple episodes with different
  seeds and averaging the fitness to get a more reliable estimate.

Example
-------
>>> from src.evolution.fitness import evaluate_agent
>>> from src.policies import MLPPolicy
>>> policy = MLPPolicy(obs_size=27, num_actions=5, hidden_sizes=[16])
>>> params = policy.get_params()
>>> fitness = evaluate_agent(params, obs_size=27, num_actions=5)
>>> isinstance(fitness, float)
True
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.envs.tribes_env import TribesEnv, NUM_ACTIONS
from src.policies import MLPPolicy


def evaluate_agent(
    chromosome: np.ndarray,
    obs_size: int = 27,
    num_actions: int = NUM_ACTIONS,
    hidden_sizes: Optional[List[int]] = None,
    env_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    agent_index: int = 0,
) -> float:
    """Run one episode and return the total reward for a single agent.

    This is the **core fitness function**.  It:

    1. Creates a fresh environment.
    2. Builds an MLPPolicy and loads the chromosome into it.
    3. Runs a full episode (all agents use the same chromosome).
    4. Returns the total reward summed over all agents.

    Args:
        chromosome:   Flat float32 parameter vector.
        obs_size:     Observation vector length (default 27).
        num_actions:  Number of discrete actions (default 5).
        hidden_sizes: MLP hidden layer widths (default [16]).
        env_config:   Optional environment config overrides.
        seed:         RNG seed for the environment reset.
        agent_index:  Unused for now — all agents share the chromosome.

    Returns:
        Total reward (float) accumulated across all agents over the
        episode.  Higher is better.

    Example:
        >>> import numpy as np
        >>> chromosome = np.random.randn(533).astype(np.float32)
        >>> fitness = evaluate_agent(chromosome)
        >>> isinstance(fitness, float)
        True

    Why sum all agents?
        Because we're evolving a **shared policy** — every agent in the
        tribe uses the same brain.  The fitness reflects how well that
        brain controls the whole tribe.
    """
    if hidden_sizes is None:
        hidden_sizes = [16]

    # Build environment (no rendering — headless evaluation)
    env = TribesEnv(config=env_config, render_mode=None)

    # Create policies — all agents share the same chromosome
    num_agents = env.num_agents
    policies = []
    for _ in range(num_agents):
        p = MLPPolicy(
            obs_size=obs_size,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            deterministic=True,
        )
        p.set_params(chromosome.copy())
        policies.append(p)

    # Run one episode
    observations, _info = env.reset(seed=seed)
    total_reward = 0.0
    done = False

    while not done:
        actions = [
            policies[i].select_action(observations[i])
            for i in range(num_agents)
        ]
        observations, rewards, terminated, truncated, _info = env.step(actions)
        total_reward += sum(rewards)
        done = terminated or truncated

    env.close()
    return total_reward


def evaluate_robust(
    chromosome: np.ndarray,
    num_episodes: int = 3,
    obs_size: int = 27,
    num_actions: int = NUM_ACTIONS,
    hidden_sizes: Optional[List[int]] = None,
    env_config: Optional[Dict[str, Any]] = None,
    base_seed: int = 0,
) -> float:
    """Evaluate a chromosome over multiple episodes and return the average.

    Robust evaluation reduces the noise from random environment layouts.
    Each episode uses a different seed: base_seed, base_seed+1, ...

    Args:
        chromosome:    Flat float32 parameter vector.
        num_episodes:  Number of episodes to average over (default 3).
        obs_size:      Observation vector length (default 27).
        num_actions:   Number of discrete actions (default 5).
        hidden_sizes:  MLP hidden layer widths (default [16]).
        env_config:    Environment config overrides.
        base_seed:     Starting seed; episodes use base_seed + i.

    Returns:
        Average total reward across all episodes.

    Example:
        >>> import numpy as np
        >>> chromosome = np.random.randn(533).astype(np.float32)
        >>> fitness = evaluate_robust(chromosome, num_episodes=2)
        >>> isinstance(fitness, float)
        True

    Why average?
        A single episode might get lucky (food spawns right next to the
        agent) or unlucky (food is far away).  Averaging over 3 episodes
        gives a fairer estimate of the chromosome's true quality.
    """
    total = 0.0
    for i in range(num_episodes):
        total += evaluate_agent(
            chromosome,
            obs_size=obs_size,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            env_config=env_config,
            seed=base_seed + i,
        )
    return total / num_episodes

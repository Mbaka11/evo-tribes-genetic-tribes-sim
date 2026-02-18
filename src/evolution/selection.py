"""
EvoTribes — Selection Operators
=================================

Selection determines which individuals get to reproduce.  Good
selection balances **exploitation** (picking the best) with
**exploration** (giving weaker individuals a chance).

Tournament Selection
--------------------
We use **tournament selection** because it's simple, effective, and
easy to tune:

1. Pick ``k`` individuals at random from the population.
2. The one with the highest fitness wins and becomes a parent.
3. Repeat to select the second parent.

Tuning ``k``:
    - k = 2  →  Weak pressure, more diversity, slower convergence.
    - k = 3  →  Moderate pressure (our default).
    - k = 5+ →  Strong pressure, faster convergence but risk of
                 premature convergence to a local optimum.

Example
-------
>>> import numpy as np
>>> population = [np.zeros(10) for _ in range(20)]
>>> fitnesses = np.random.randn(20)
>>> parent = tournament_selection(population, fitnesses, k=3)
>>> parent.shape
(10,)
"""

from __future__ import annotations

from typing import List

import numpy as np


def tournament_selection(
    population: List[np.ndarray],
    fitnesses: np.ndarray,
    k: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select one parent via tournament selection.

    Args:
        population: List of chromosomes (flat float32 arrays).
        fitnesses:  1-D array of fitness values, same length as population.
        k:          Tournament size — how many candidates to compare.
        rng:        NumPy random Generator (optional; creates one if None).

    Returns:
        A **copy** of the winning chromosome.

    Raises:
        ValueError: If population is empty or k < 1.

    How it works (step by step)
    ---------------------------
    Suppose population has 10 individuals and k = 3:

    1. Randomly pick 3 indices, e.g. [2, 7, 4].
    2. Look up their fitnesses: [1.5, 3.2, 0.8].
    3. Index 7 has the highest fitness (3.2) → it wins.
    4. Return a copy of population[7].

    Why a copy?
        Crossover and mutation will modify the offspring.  We don't want
        to accidentally change the parent in the population.

    Example:
        >>> import numpy as np
        >>> pop = [np.array([1.0, 2.0]), np.array([3.0, 4.0]),
        ...        np.array([5.0, 6.0])]
        >>> fit = np.array([0.1, 0.9, 0.5])
        >>> rng = np.random.default_rng(42)
        >>> winner = tournament_selection(pop, fit, k=2, rng=rng)
        >>> winner.shape
        (2,)
    """
    if len(population) == 0:
        raise ValueError("Population cannot be empty.")
    if k < 1:
        raise ValueError(f"Tournament size k must be >= 1, got {k}.")

    if rng is None:
        rng = np.random.default_rng()

    pop_size = len(population)
    k = min(k, pop_size)  # can't sample more than we have

    # Pick k random contestants
    contestants = rng.choice(pop_size, size=k, replace=False)

    # Find the one with the best fitness
    best_idx = contestants[np.argmax(fitnesses[contestants])]

    return population[best_idx].copy()

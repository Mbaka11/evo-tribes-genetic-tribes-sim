"""
EvoTribes — Mutation Operators
================================

Mutation introduces small random changes to a chromosome.  This is
essential for exploring new areas of the search space that crossover
alone cannot reach.

Gaussian Mutation
-----------------
We add a small amount of Gaussian (normal-distribution) noise to
**every** gene:

    new_gene = old_gene + N(0, σ²)

where σ (``std``) controls how large the perturbations are.

Adaptive Mutation
-----------------
Early in evolution, we want large mutations to explore widely.
Later, we want smaller mutations to fine-tune.  **Adaptive decay**
reduces σ over time:

    σ(gen) = σ₀ × decay^(gen // decay_every)

With our defaults (σ₀ = 0.02, decay = 0.9, decay_every = 20):

    Gen   0-19:  σ = 0.020
    Gen  20-39:  σ = 0.018
    Gen  40-59:  σ = 0.016
    Gen  60-79:  σ = 0.015
    Gen  80-99:  σ = 0.013

Example
-------
>>> import numpy as np
>>> chromosome = np.zeros(10)
>>> mutated = gaussian_mutate(chromosome, std=0.02)
>>> mutated.shape
(10,)
>>> np.allclose(mutated, chromosome)  # very unlikely
False
"""

from __future__ import annotations

import numpy as np


def gaussian_mutate(
    chromosome: np.ndarray,
    std: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise to every gene in the chromosome.

    Args:
        chromosome: Flat float32 parameter vector.
        std:        Standard deviation of the noise (default 0.02).
        rng:        NumPy random Generator (optional).

    Returns:
        A **new** mutated chromosome (original is not modified).

    Step-by-step example
    --------------------
    chromosome = [0.50,  -0.12,  0.03]
    noise      = [0.01,  -0.005, 0.013]   (drawn from N(0, 0.02²))
    result     = [0.51,  -0.125, 0.043]

    Example:
        >>> import numpy as np
        >>> c = np.array([1.0, 2.0, 3.0])
        >>> m = gaussian_mutate(c, std=0.01, rng=np.random.default_rng(42))
        >>> m.shape
        (3,)
        >>> np.array_equal(c, m)  # original unchanged
        False
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(loc=0.0, scale=std, size=chromosome.shape)
    return (chromosome + noise).astype(chromosome.dtype)


def adaptive_mutation_std(
    generation: int,
    initial_std: float = 0.02,
    decay: float = 0.9,
    decay_every: int = 20,
) -> float:
    """Compute the mutation std for a given generation.

    The std decreases over time following a step-decay schedule:

        std(gen) = initial_std × decay ^ (gen // decay_every)

    Args:
        generation:  Current generation number (0-indexed).
        initial_std: Starting standard deviation (default 0.02).
        decay:       Multiplicative decay factor (default 0.9).
        decay_every: Apply decay every N generations (default 20).

    Returns:
        The mutation standard deviation for this generation.

    Example values (default settings):
        >>> adaptive_mutation_std(0)
        0.02
        >>> round(adaptive_mutation_std(20), 4)
        0.018
        >>> round(adaptive_mutation_std(40), 4)
        0.0162
        >>> round(adaptive_mutation_std(99), 4)
        0.0131

    Example:
        >>> std = adaptive_mutation_std(generation=50, initial_std=0.02)
        >>> 0 < std <= 0.02
        True
    """
    steps = generation // decay_every
    return initial_std * (decay ** steps)

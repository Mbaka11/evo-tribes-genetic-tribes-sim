"""
EvoTribes — Crossover Operators
==================================

Crossover combines genetic material from two parents to create
offspring.  The hope is that good traits from both parents end up
in the child.

Uniform Crossover
-----------------
For each gene (parameter), we flip a fair coin:

    - Heads → take from parent A
    - Tails → take from parent B

This is called **uniform crossover** because every gene has an equal
(uniform) 50 % chance of coming from either parent.

Why uniform crossover?
    Our chromosomes are flat vectors of neural network weights.  There's
    no meaningful spatial structure to preserve, so simpler crossover
    methods (like single-point) don't offer an advantage.

Example
-------
>>> import numpy as np
>>> parent_a = np.ones(10)
>>> parent_b = np.zeros(10)
>>> child = uniform_crossover(parent_a, parent_b, rng=np.random.default_rng(0))
>>> child.shape
(10,)
>>> set(child).issubset({0.0, 1.0})
True
"""

from __future__ import annotations

import numpy as np


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create one child by mixing genes from two parents.

    For each position ``i``:
        child[i] = parent_a[i]  with probability 0.5
        child[i] = parent_b[i]  otherwise

    Args:
        parent_a: Flat float32 chromosome.
        parent_b: Flat float32 chromosome (same length as parent_a).
        rng:      NumPy random Generator (optional).

    Returns:
        A new chromosome (not a reference to either parent).

    Raises:
        ValueError: If parents have different lengths.

    Step-by-step example
    --------------------
    parent_a = [1.0,  2.0,  3.0,  4.0]
    parent_b = [5.0,  6.0,  7.0,  8.0]
    coin flips = [True, False, True, False]

    child    = [1.0,  6.0,  3.0,  8.0]
               ↑ from A  ↑ from B  ↑ from A  ↑ from B

    Example:
        >>> import numpy as np
        >>> a = np.array([1.0, 2.0, 3.0, 4.0])
        >>> b = np.array([5.0, 6.0, 7.0, 8.0])
        >>> rng = np.random.default_rng(42)
        >>> child = uniform_crossover(a, b, rng=rng)
        >>> len(child) == 4
        True
    """
    if len(parent_a) != len(parent_b):
        raise ValueError(
            f"Parents must have the same length: "
            f"{len(parent_a)} != {len(parent_b)}"
        )

    if rng is None:
        rng = np.random.default_rng()

    # Boolean mask: True = take from A, False = take from B
    mask = rng.random(len(parent_a)) < 0.5

    child = np.where(mask, parent_a, parent_b)
    return child.astype(parent_a.dtype)

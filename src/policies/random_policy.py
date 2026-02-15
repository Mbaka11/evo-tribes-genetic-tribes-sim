"""
EvoTribes — Random Policy
===========================

Picks a uniformly random action every step, ignoring the observation
entirely.  This is the baseline "no intelligence" policy.

It exists so that the demo and tests can use the policy interface
consistently even before real policies are trained.

Example
-------
>>> import numpy as np
>>> from src.policies.random_policy import RandomPolicy
>>> policy = RandomPolicy(num_actions=5, seed=42)
>>> obs = np.zeros(27, dtype=np.float32)
>>> action = policy.select_action(obs)
>>> 0 <= action < 5
True
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.policies.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    """Uniformly random action selection — no learning, no parameters.

    Args:
        num_actions: Size of the discrete action space.
        seed: Optional RNG seed for reproducible randomness.

    Example:
        >>> policy = RandomPolicy(num_actions=5, seed=0)
        >>> policy.select_action(np.zeros(27, dtype=np.float32))
        4
        >>> policy.param_count()
        0
    """

    def __init__(self, num_actions: int = 5, seed: Optional[int] = None):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray) -> int:
        """Return a random action, ignoring the observation.

        Args:
            observation: Ignored — present for interface compatibility.

        Returns:
            Random integer in ``[0, num_actions)``.

        Example:
            >>> obs = np.array([0.25]*25 + [0.5, 0.0], dtype=np.float32)
            >>> action = policy.select_action(obs)
            >>> 0 <= action < 5
            True
        """
        return int(self.rng.integers(0, self.num_actions))

    def __repr__(self) -> str:
        return f"RandomPolicy(num_actions={self.num_actions}, params=0)"

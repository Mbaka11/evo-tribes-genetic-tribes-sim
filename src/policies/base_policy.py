"""
EvoTribes — Base Policy Interface
===================================

Every agent brain must inherit from :class:`BasePolicy` and implement
``select_action``.  This guarantees that the environment, demo scripts,
and the genetic algorithm can work with **any** policy without knowing
its internals.

Design rules
------------
* ``select_action(obs)`` is the ONLY required method.
* ``get_params`` / ``set_params`` enable the genetic algorithm to
  read and write the policy's tuneable parameters as a flat numpy array.
* ``save`` / ``load`` handle persistence (replay, checkpointing).
* Policies must be **stateless between steps** — no hidden memory.
  (If we add recurrent policies later, they will manage their own state
  and expose a ``reset_state()`` method.)

Example
-------
>>> import numpy as np
>>> from src.policies.random_policy import RandomPolicy
>>> policy = RandomPolicy(num_actions=5)
>>> obs = np.zeros(27, dtype=np.float32)
>>> action = policy.select_action(obs)
>>> 0 <= action < 5
True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BasePolicy(ABC):
    """Abstract base class for all EvoTribes agent policies.

    Subclasses MUST implement ``select_action``.
    Subclasses SHOULD implement ``get_params`` and ``set_params`` if they
    have tuneable parameters (needed for the genetic algorithm).
    """

    # ------------------------------------------------------------------
    # Core interface — MUST override
    # ------------------------------------------------------------------
    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """Choose an action given a single agent's observation vector.

        Args:
            observation: float32 array of shape ``(obs_size,)``.

        Returns:
            Integer action id in ``[0, num_actions)``.

        Example:
            >>> obs = np.array([0.0]*25 + [0.8, 0.0], dtype=np.float32)
            >>> action = policy.select_action(obs)  # e.g. 3 (east)
        """
        ...

    # ------------------------------------------------------------------
    # Parameter interface — needed for genetic algorithm (Iteration 3)
    # ------------------------------------------------------------------
    def get_params(self) -> np.ndarray:
        """Return all tuneable parameters as a flat float32 array.

        Policies without parameters (e.g. RandomPolicy) return an empty
        array.

        Returns:
            1-D float32 numpy array.

        Example — MLPPolicy with 155 weights:
            >>> params = policy.get_params()
            >>> params.shape
            (155,)
        """
        return np.array([], dtype=np.float32)

    def set_params(self, params: np.ndarray) -> None:
        """Replace all tuneable parameters from a flat float32 array.

        This is the inverse of ``get_params``.  The array length must
        match exactly.

        Args:
            params: 1-D float32 array with the same length as
                    ``get_params()`` returns.

        Example:
            >>> old = policy.get_params()
            >>> mutated = old + np.random.normal(0, 0.1, size=old.shape)
            >>> policy.set_params(mutated)
        """
        pass  # no-op for parameter-free policies

    def param_count(self) -> int:
        """Total number of tuneable parameters.

        Example:
            >>> policy.param_count()
            155
        """
        return len(self.get_params())

    # ------------------------------------------------------------------
    # Persistence — save / load weights
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save policy parameters to a ``.npy`` file.

        Args:
            path: File path (e.g. ``"runs/gen_42/agent_0.npy"``).

        Example:
            >>> policy.save("best_agent.npy")
        """
        np.save(path, self.get_params())

    def load(self, path: str) -> None:
        """Load policy parameters from a ``.npy`` file.

        Args:
            path: File path previously saved with ``save()``.

        Example:
            >>> policy.load("best_agent.npy")
            >>> action = policy.select_action(obs)
        """
        params = np.load(path)
        self.set_params(params)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.param_count()})"

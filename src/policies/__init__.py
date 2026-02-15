"""
EvoTribes — Policy System
==========================

All policy implementations live under ``src/policies/``.

Public interface
----------------
- :class:`BasePolicy`   — abstract base class every policy must implement
- :class:`RandomPolicy`  — picks a random action each step
- :class:`MLPPolicy`     — multilayer perceptron with configurable layers
"""

from src.policies.base_policy import BasePolicy
from src.policies.random_policy import RandomPolicy
from src.policies.mlp_policy import MLPPolicy

__all__ = ["BasePolicy", "RandomPolicy", "MLPPolicy"]

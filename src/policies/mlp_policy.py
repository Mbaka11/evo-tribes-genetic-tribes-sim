"""
EvoTribes — MLP (Multi-Layer Perceptron) Policy
=================================================

A small fully-connected neural network that maps an observation vector
to action probabilities.  Built with **pure NumPy** — no PyTorch, no
TensorFlow.

Why pure NumPy?
    The genetic algorithm (Iteration 3) evolves weights directly.
    We don't need backpropagation or autograd — just forward passes and
    the ability to flatten / unflatten parameters.

Architecture
------------
The network is a chain of ``Dense → ReLU`` layers followed by a final
``Dense → Softmax`` layer.  The output has one element per action; we
pick the action with the highest probability (argmax), or sample from
the distribution.

Example with default config (obs_size=27, num_actions=5, hidden=[16]):

    Input  (27,)
      │
      ▼
    Dense  (27 → 16) + bias  →  ReLU
      │      weights: 27×16 = 432   bias: 16    total: 448
      ▼
    Dense  (16 → 5)  + bias  →  Softmax
             weights: 16×5 = 80     bias: 5     total: 85
      │
      ▼
    Output (5,)  →  argmax → action

    Total parameters: 448 + 85 = 533

How parameters are stored
-------------------------
``get_params()`` returns a single flat float32 array of all weights and
biases concatenated in layer order:

    [W0.flat, b0, W1.flat, b1, ...]

``set_params()`` slices that array back into the original shapes.

Examples
--------
>>> from src.policies.mlp_policy import MLPPolicy
>>> import numpy as np
>>> policy = MLPPolicy(obs_size=27, num_actions=5, hidden_sizes=[16])
>>> policy.param_count()
533
>>> obs = np.random.rand(27).astype(np.float32)
>>> action = policy.select_action(obs)
>>> 0 <= action < 5
True
>>> params = policy.get_params()
>>> params.shape
(533,)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.policies.base_policy import BasePolicy


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------
def _relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: max(0, x).

    Example:
        >>> _relu(np.array([-2.0, 0.0, 3.0]))
        array([0., 0., 3.])
    """
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Converts raw scores (logits) into probabilities that sum to 1.

    Example:
        >>> probs = _softmax(np.array([2.0, 1.0, 0.1]))
        >>> probs.sum()  # ≈ 1.0
        1.0
        >>> probs[0] > probs[1] > probs[2]
        True

    Math:
        softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

        Subtracting max(x) prevents overflow without changing the result.
    """
    e = np.exp(x - np.max(x))
    return e / e.sum()


class MLPPolicy(BasePolicy):
    """Fully-connected neural network policy (pure NumPy).

    Args:
        obs_size:     Length of the observation vector (default 27).
        num_actions:  Number of discrete actions (default 5).
        hidden_sizes: List of hidden layer widths (default [16]).
        seed:         Optional RNG seed for weight initialisation.
        deterministic: If True, always pick argmax action.
                       If False, sample from softmax probabilities.

    Example:
        >>> policy = MLPPolicy(obs_size=27, num_actions=5, hidden_sizes=[16])
        >>> policy.param_count()
        533
        >>> obs = np.zeros(27, dtype=np.float32)
        >>> action = policy.select_action(obs)
        >>> 0 <= action < 5
        True
    """

    def __init__(
        self,
        obs_size: int = 27,
        num_actions: int = 5,
        hidden_sizes: Optional[List[int]] = None,
        seed: Optional[int] = None,
        deterministic: bool = True,
    ):
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [16]
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

        # Build layer shapes ------------------------------------------------
        # Each layer is (input_dim, output_dim).
        # Layer list: [input→hidden0, hidden0→hidden1, ..., hiddenN→output]
        layer_dims: List[tuple] = []
        prev = self.obs_size
        for h in self.hidden_sizes:
            layer_dims.append((prev, h))
            prev = h
        layer_dims.append((prev, self.num_actions))  # final layer

        # Initialise weights and biases with Xavier uniform -----------------
        # Xavier keeps the variance of signals roughly constant across layers.
        #
        # For a layer of shape (fan_in, fan_out):
        #   limit = sqrt(6 / (fan_in + fan_out))
        #   W ~ Uniform(-limit, +limit)
        #   b = 0
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in layer_dims:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(
                np.float32
            )
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, observation: np.ndarray) -> np.ndarray:
        """Run the observation through the network and return action
        probabilities.

        Args:
            observation: float32 array of shape ``(obs_size,)``.

        Returns:
            float32 array of shape ``(num_actions,)`` summing to 1.

        Example:
            >>> obs = np.zeros(27, dtype=np.float32)
            >>> probs = policy.forward(obs)
            >>> probs.shape
            (5,)
            >>> abs(probs.sum() - 1.0) < 1e-6
            True
        """
        x = observation.astype(np.float32)

        # Hidden layers: Dense + ReLU
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = _relu(x)

        # Output layer: Dense + Softmax
        x = x @ self.weights[-1] + self.biases[-1]
        return _softmax(x)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, observation: np.ndarray) -> int:
        """Choose an action by running the forward pass.

        If ``deterministic=True`` (default), picks the action with the
        highest probability (argmax).  Otherwise, samples from the
        probability distribution.

        Args:
            observation: float32 array of shape ``(obs_size,)``.

        Returns:
            Integer action id.

        Example — deterministic:
            >>> policy = MLPPolicy(obs_size=27, num_actions=5, seed=0,
            ...                    deterministic=True)
            >>> obs = np.ones(27, dtype=np.float32) * 0.5
            >>> a1 = policy.select_action(obs)
            >>> a2 = policy.select_action(obs)
            >>> a1 == a2   # same obs → same action (deterministic)
            True

        Example — stochastic:
            >>> policy = MLPPolicy(obs_size=27, num_actions=5, seed=0,
            ...                    deterministic=False)
            >>> obs = np.ones(27, dtype=np.float32) * 0.5
            >>> actions = [policy.select_action(obs) for _ in range(100)]
            >>> len(set(actions)) > 1  # multiple different actions
            True
        """
        probs = self.forward(observation)
        if self.deterministic:
            return int(np.argmax(probs))
        else:
            return int(self.rng.choice(self.num_actions, p=probs))

    # ------------------------------------------------------------------
    # Parameter interface (for genetic algorithm)
    # ------------------------------------------------------------------
    def get_params(self) -> np.ndarray:
        """Flatten all weights and biases into a single 1-D array.

        Order: [W0.flat, b0, W1.flat, b1, ...]

        Example:
            >>> policy = MLPPolicy(obs_size=27, num_actions=5,
            ...                    hidden_sizes=[16])
            >>> p = policy.get_params()
            >>> p.shape
            (533,)
            >>> # Breakdown: W0(27×16)=432 + b0(16) + W1(16×5)=80 + b1(5)
            >>> 432 + 16 + 80 + 5
            533
        """
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts).astype(np.float32)

    def set_params(self, params: np.ndarray) -> None:
        """Restore weights and biases from a flat 1-D array.

        Args:
            params: Must have the same length as ``get_params()`` returns.

        Raises:
            ValueError: If ``params`` length doesn't match.

        Example:
            >>> p = policy.get_params()
            >>> noisy = p + np.random.normal(0, 0.01, size=p.shape)
            >>> policy.set_params(noisy.astype(np.float32))
        """
        expected = self.param_count()
        if len(params) != expected:
            raise ValueError(
                f"Expected {expected} params, got {len(params)}"
            )

        offset = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            b_size = self.biases[i].size

            self.weights[i] = params[offset : offset + w_size].reshape(
                self.weights[i].shape
            ).astype(np.float32)
            offset += w_size

            self.biases[i] = params[offset : offset + b_size].astype(
                np.float32
            )
            offset += b_size

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        arch = " → ".join(
            [str(self.obs_size)]
            + [str(h) for h in self.hidden_sizes]
            + [str(self.num_actions)]
        )
        return (
            f"MLPPolicy(arch={arch}, params={self.param_count()}, "
            f"deterministic={self.deterministic})"
        )

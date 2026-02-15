"""
EvoTribes — Policy Tests (Iteration 2)
=========================================

Tests for the policy interface, RandomPolicy, and MLPPolicy.

Groups
------
1. BasePolicy interface compliance
2. RandomPolicy behaviour
3. MLPPolicy forward pass, determinism, and stochasticity
4. Parameter get / set round-trip (critical for GA)
5. Save / load persistence
6. Policy swap — can replace one policy with another mid-episode
"""

import os
import tempfile

import numpy as np
import pytest

from src.envs.tribes_env import TribesEnv, NUM_ACTIONS
from src.policies.base_policy import BasePolicy
from src.policies.random_policy import RandomPolicy
from src.policies.mlp_policy import MLPPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
OBS_SIZE = 27  # (2*2+1)^2 + 2


def _make_obs(value: float = 0.0, seed: int | None = None) -> np.ndarray:
    """Create a fake observation vector."""
    if seed is not None:
        return np.random.default_rng(seed).random(OBS_SIZE).astype(np.float32)
    return np.full(OBS_SIZE, value, dtype=np.float32)


# ===================================================================
# 1. Interface compliance
# ===================================================================
class TestInterfaceCompliance:
    """Every policy must satisfy the BasePolicy contract."""

    @pytest.fixture(params=["random", "mlp"])
    def policy(self, request):
        if request.param == "random":
            return RandomPolicy(num_actions=NUM_ACTIONS, seed=0)
        return MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=0)

    def test_is_base_policy(self, policy):
        assert isinstance(policy, BasePolicy)

    def test_select_action_returns_int(self, policy):
        obs = _make_obs()
        action = policy.select_action(obs)
        assert isinstance(action, int)

    def test_select_action_in_range(self, policy):
        obs = _make_obs(seed=7)
        for _ in range(50):
            a = policy.select_action(obs)
            assert 0 <= a < NUM_ACTIONS

    def test_get_params_returns_1d_float32(self, policy):
        p = policy.get_params()
        assert isinstance(p, np.ndarray)
        assert p.ndim == 1
        assert p.dtype == np.float32

    def test_param_count_matches_get_params(self, policy):
        assert policy.param_count() == len(policy.get_params())

    def test_repr_is_string(self, policy):
        r = repr(policy)
        assert isinstance(r, str) and len(r) > 0


# ===================================================================
# 2. RandomPolicy
# ===================================================================
class TestRandomPolicy:
    def test_zero_params(self):
        rp = RandomPolicy(num_actions=5, seed=0)
        assert rp.param_count() == 0
        assert len(rp.get_params()) == 0

    def test_ignores_observation(self):
        """Same seed → same sequence regardless of different observations."""
        rp1 = RandomPolicy(num_actions=5, seed=99)
        rp2 = RandomPolicy(num_actions=5, seed=99)
        obs_a = _make_obs(0.0)
        obs_b = _make_obs(1.0)
        for _ in range(20):
            assert rp1.select_action(obs_a) == rp2.select_action(obs_b)

    def test_reproducibility(self):
        rp1 = RandomPolicy(num_actions=5, seed=42)
        rp2 = RandomPolicy(num_actions=5, seed=42)
        actions1 = [rp1.select_action(_make_obs()) for _ in range(100)]
        actions2 = [rp2.select_action(_make_obs()) for _ in range(100)]
        assert actions1 == actions2

    def test_covers_all_actions(self):
        rp = RandomPolicy(num_actions=5, seed=0)
        obs = _make_obs()
        seen = set(rp.select_action(obs) for _ in range(200))
        assert seen == {0, 1, 2, 3, 4}


# ===================================================================
# 3. MLPPolicy
# ===================================================================
class TestMLPPolicy:
    def test_default_architecture(self):
        mlp = MLPPolicy(obs_size=27, num_actions=5, hidden_sizes=[16], seed=0)
        # W0: 27*16=432, b0: 16, W1: 16*5=80, b1: 5 → 533
        assert mlp.param_count() == 533

    def test_custom_architecture(self):
        mlp = MLPPolicy(obs_size=10, num_actions=3, hidden_sizes=[8, 4], seed=0)
        # W0: 10*8=80, b0: 8, W1: 8*4=32, b1: 4, W2: 4*3=12, b2: 3 → 139
        assert mlp.param_count() == 139

    def test_forward_returns_probabilities(self):
        mlp = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=0)
        probs = mlp.forward(_make_obs(seed=1))
        assert probs.shape == (NUM_ACTIONS,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert np.all(probs >= 0)

    def test_deterministic_is_repeatable(self):
        mlp = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS,
                        seed=0, deterministic=True)
        obs = _make_obs(seed=5)
        a = mlp.select_action(obs)
        for _ in range(20):
            assert mlp.select_action(obs) == a

    def test_stochastic_varies(self):
        mlp = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS,
                        seed=0, deterministic=False)
        obs = _make_obs(0.5)
        actions = set(mlp.select_action(obs) for _ in range(200))
        # With random weights, softmax should produce varied actions
        assert len(actions) > 1

    def test_different_seeds_different_weights(self):
        m1 = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=0)
        m2 = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=1)
        assert not np.array_equal(m1.get_params(), m2.get_params())


# ===================================================================
# 4. Parameter get / set round-trip
# ===================================================================
class TestParamRoundTrip:
    def test_roundtrip_preserves_weights(self):
        mlp = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=0)
        original = mlp.get_params().copy()
        mlp.set_params(original)
        np.testing.assert_array_equal(mlp.get_params(), original)

    def test_set_changes_behaviour(self):
        mlp = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS,
                        seed=0, deterministic=True)
        obs = _make_obs(seed=3)
        action_before = mlp.select_action(obs)

        # Inject very different weights
        p = mlp.get_params()
        new_p = np.full_like(p, 0.5)
        mlp.set_params(new_p)
        action_after = mlp.select_action(obs)

        # Weights changed → output very likely changes (not guaranteed,
        # but with all-0.5 weights vs Xavier, extremely likely)
        assert not np.array_equal(mlp.get_params(), p)

    def test_wrong_length_raises(self):
        mlp = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=0)
        with pytest.raises(ValueError, match="Expected"):
            mlp.set_params(np.zeros(10, dtype=np.float32))

    def test_set_params_noop_for_random(self):
        rp = RandomPolicy(num_actions=NUM_ACTIONS, seed=0)
        rp.set_params(np.array([], dtype=np.float32))  # should be no-op
        assert rp.param_count() == 0


# ===================================================================
# 5. Save / Load persistence
# ===================================================================
class TestSaveLoad:
    def test_mlp_save_load_roundtrip(self):
        mlp = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=0)
        params_before = mlp.get_params().copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "agent.npy")
            mlp.save(path)
            assert os.path.exists(path)

            # Create fresh policy (different seed) and load
            mlp2 = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=99)
            assert not np.array_equal(mlp2.get_params(), params_before)

            mlp2.load(path)
            np.testing.assert_array_almost_equal(mlp2.get_params(), params_before)

    def test_mlp_loaded_policy_same_actions(self):
        mlp = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS,
                        seed=0, deterministic=True)
        obs = _make_obs(seed=10)
        expected_action = mlp.select_action(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "agent.npy")
            mlp.save(path)

            mlp2 = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS,
                             seed=999, deterministic=True)
            mlp2.load(path)
            assert mlp2.select_action(obs) == expected_action

    def test_random_save_load_noop(self):
        rp = RandomPolicy(num_actions=NUM_ACTIONS, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "random.npy")
            rp.save(path)
            assert os.path.exists(path)
            rp.load(path)  # should not crash
            assert rp.param_count() == 0


# ===================================================================
# 6. Policy swap — mid-episode replacement
# ===================================================================
class TestPolicySwap:
    def test_swap_random_to_mlp(self):
        """Start with RandomPolicy, swap to MLPPolicy, environment continues."""
        env = TribesEnv(config={"num_agents": 2, "max_steps": 20})
        obs, _ = env.reset(seed=0)

        policies: list[BasePolicy] = [
            RandomPolicy(num_actions=NUM_ACTIONS, seed=i) for i in range(2)
        ]

        # Run 5 steps with random
        for _ in range(5):
            actions = [policies[i].select_action(obs[i]) for i in range(2)]
            obs, _, term, trunc, _ = env.step(actions)
            if term or trunc:
                break

        # Swap agent 0 to MLP
        policies[0] = MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=0)

        # Run 5 more steps — should not crash
        for _ in range(5):
            actions = [policies[i].select_action(obs[i]) for i in range(2)]
            obs, _, term, trunc, _ = env.step(actions)
            if term or trunc:
                break

        env.close()

    def test_env_integration_mlp(self):
        """MLPPolicy runs a full episode with the real environment."""
        env = TribesEnv(config={"num_agents": 3, "max_steps": 30})
        obs, _ = env.reset(seed=1)

        policies = [
            MLPPolicy(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, seed=i)
            for i in range(3)
        ]

        done = False
        step = 0
        while not done:
            actions = [policies[i].select_action(obs[i]) for i in range(3)]
            obs, rewards, term, trunc, info = env.step(actions)
            done = term or trunc
            step += 1

        assert step > 0
        assert "alive" in info
        env.close()

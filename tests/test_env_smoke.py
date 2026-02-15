"""
EvoTribes — Smoke Tests for TribesEnv (Iteration 1)
=====================================================

Validates:
- environment resets without error
- observation shape matches observation_space
- action space is Discrete(5)
- stepping with random actions works for several steps
- info dict contains expected keys
- agents lose energy and episode can terminate

Run with:
    python -m pytest tests/test_env_smoke.py -v
"""

import sys
import os

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.tribes_env import TribesEnv, DEFAULT_CONFIG, NUM_ACTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def env():
    """Create a small environment for fast testing."""
    cfg = {
        "grid_width": 10,
        "grid_height": 10,
        "num_agents": 4,
        "num_tribes": 2,
        "num_food": 5,
        "max_steps": 50,
        "initial_energy": 30.0,
    }
    e = TribesEnv(config=cfg)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestReset:
    def test_reset_returns_observations_and_info(self, env: TribesEnv):
        obs, info = env.reset(seed=0)
        assert isinstance(obs, list)
        assert len(obs) == env.num_agents
        assert isinstance(info, dict)

    def test_observation_shape(self, env: TribesEnv):
        obs, _ = env.reset(seed=0)
        for o in obs:
            assert o.shape == env.observation_space.shape
            assert o.dtype == np.float32

    def test_observation_values_in_range(self, env: TribesEnv):
        obs, _ = env.reset(seed=0)
        for o in obs:
            assert np.all(o >= 0.0)
            assert np.all(o <= 1.0)


class TestActionSpace:
    def test_action_space_type(self, env: TribesEnv):
        assert env.action_space.n == NUM_ACTIONS  # 5

    def test_sample_action_is_valid(self, env: TribesEnv):
        for _ in range(20):
            a = env.action_space.sample()
            assert 0 <= a < NUM_ACTIONS


class TestStep:
    def test_step_returns_correct_structure(self, env: TribesEnv):
        env.reset(seed=1)
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        obs, rewards, terminated, truncated, info = env.step(actions)

        assert isinstance(obs, list)
        assert len(obs) == env.num_agents
        assert isinstance(rewards, list)
        assert len(rewards) == env.num_agents
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_multiple_steps(self, env: TribesEnv):
        env.reset(seed=2)
        for _ in range(10):
            actions = [env.action_space.sample() for _ in range(env.num_agents)]
            obs, rewards, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                break
        # Should have advanced at least one step
        assert env.current_step >= 1

    def test_energy_decreases(self, env: TribesEnv):
        env.reset(seed=3)
        initial_total = float(np.sum(env.agent_energy))
        # Take a few stay-actions (no food guaranteed)
        for _ in range(5):
            actions = [0] * env.num_agents  # all stay
            env.step(actions)
        final_total = float(np.sum(env.agent_energy))
        # Energy should have decreased (ignoring possible food pickup)
        # With 4 agents * 5 steps * -1.0/step = -20 base drain
        # Even if some agents eat, total should be lower overall
        assert final_total < initial_total


class TestInfo:
    def test_info_keys(self, env: TribesEnv):
        _, info = env.reset(seed=4)
        assert "step" in info
        assert "alive" in info
        assert "total_energy" in info

    def test_initial_alive_count(self, env: TribesEnv):
        _, info = env.reset(seed=5)
        assert info["alive"] == env.num_agents


class TestTermination:
    def test_episode_ends_on_max_steps(self):
        cfg = {
            "grid_width": 8,
            "grid_height": 8,
            "num_agents": 2,
            "num_tribes": 1,
            "num_food": 50,         # lots of food so agents stay alive
            "max_steps": 10,
            "initial_energy": 1000.0,
        }
        env = TribesEnv(config=cfg)
        env.reset(seed=6)
        for _ in range(20):
            actions = [0, 0]
            _, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break
        assert truncated
        assert env.current_step == 10
        env.close()

    def test_episode_ends_when_all_dead(self):
        cfg = {
            "grid_width": 8,
            "grid_height": 8,
            "num_agents": 2,
            "num_tribes": 1,
            "num_food": 0,
            "food_respawn": False,
            "max_steps": 500,
            "initial_energy": 5.0,
            "energy_per_step": -2.0,
        }
        env = TribesEnv(config=cfg)
        env.reset(seed=7)
        for _ in range(500):
            actions = [0, 0]
            _, _, terminated, truncated, info = env.step(actions)
            if terminated:
                break
        assert terminated
        assert info["alive"] == 0
        env.close()

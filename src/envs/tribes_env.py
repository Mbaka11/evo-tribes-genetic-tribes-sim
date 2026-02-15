"""
EvoTribes — Gymnasium Environment
==================================

A multi-agent grid world where agents belong to tribes, forage for food,
and spend energy to survive.  This iteration supports random agents only;
no policies or evolution logic are wired in.

Key design decisions
--------------------
* Every parameter lives in DEFAULT_CONFIG and can be overridden at init.
* Observation is a flat float32 vector:
    [local_window_flat + energy + tribe_id]
* Actions: 0=Stay, 1=North, 2=South, 3=East, 4=West.
* Reward: +food_reward on eating, +survival_bonus per step,
  -collision_penalty when bumping into another agent.
* Episode ends when max_steps is reached OR all agents are dead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Default configuration — no magic numbers anywhere else in this file
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # World
    "grid_width": 20,
    "grid_height": 20,

    # Agents
    "num_agents": 6,
    "num_tribes": 2,

    # Vision
    "view_radius": 2,  # agent sees a (2r+1)×(2r+1) window

    # Energy
    "initial_energy": 100.0,
    "energy_per_step": -1.0,       # cost of being alive
    "energy_from_food": 15.0,

    # Food
    "num_food": 10,
    "food_respawn": True,          # respawn food when eaten?

    # Rewards
    "food_reward": 1.0,
    "survival_bonus": 0.01,
    "collision_penalty": -0.1,

    # Episode
    "max_steps": 300,
}

# ---------------------------------------------------------------------------
# Tile codes used in the grid layer
# ---------------------------------------------------------------------------
TILE_EMPTY = 0
TILE_FOOD = 1
# Agents are NOT stored on the tile grid; they have their own position array.


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------
ACTION_STAY = 0
ACTION_NORTH = 1   # row - 1
ACTION_SOUTH = 2   # row + 1
ACTION_EAST = 3    # col + 1
ACTION_WEST = 4    # col - 1

NUM_ACTIONS = 5

# Movement deltas indexed by action id  (row_delta, col_delta)
_DELTAS = {
    ACTION_STAY:  ( 0,  0),
    ACTION_NORTH: (-1,  0),
    ACTION_SOUTH: ( 1,  0),
    ACTION_EAST:  ( 0,  1),
    ACTION_WEST:  ( 0, -1),
}


class TribesEnv(gym.Env):
    """Multi-agent grid environment for EvoTribes.

    Although Gymnasium is single-agent by convention, we manage all agents
    inside one env and expose a **list** interface for observations, rewards,
    and actions so the outer loop can iterate over agents cleanly.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.render_mode = render_mode

        # Unpack frequently used values
        self.grid_w: int = self.cfg["grid_width"]
        self.grid_h: int = self.cfg["grid_height"]
        self.num_agents: int = self.cfg["num_agents"]
        self.num_tribes: int = self.cfg["num_tribes"]
        self.view_r: int = self.cfg["view_radius"]
        self.max_steps: int = self.cfg["max_steps"]

        # Observation size ---------------------------------------------------
        # local window: (2r+1)² cells, each encoded as a single float
        #   0.0 = empty, 0.25 = food, 0.5..1.0 = agent of tribe k
        window_side = 2 * self.view_r + 1
        self._window_size = window_side * window_side
        # + 2 internal features: normalised energy, normalised tribe id
        self._obs_size = self._window_size + 2

        # Gymnasium spaces (for a *single* agent — callers loop over agents)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Internal state — filled in reset()
        self.grid: np.ndarray = np.empty(0)
        self.agent_positions: np.ndarray = np.empty(0)
        self.agent_energy: np.ndarray = np.empty(0)
        self.agent_tribes: np.ndarray = np.empty(0)
        self.agent_alive: np.ndarray = np.empty(0)
        self.current_step: int = 0

        # Renderer (lazy import to avoid pygame dependency when not needed)
        self._renderer = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[list[np.ndarray], dict]:
        super().reset(seed=seed)
        rng = self.np_random

        # Grid with food
        self.grid = np.full(
            (self.grid_h, self.grid_w), TILE_EMPTY, dtype=np.int8
        )
        self._spawn_food(rng, count=self.cfg["num_food"])

        # Agents
        positions = self._random_empty_cells(rng, self.num_agents)
        self.agent_positions = np.array(positions, dtype=np.int32)
        self.agent_energy = np.full(
            self.num_agents, self.cfg["initial_energy"], dtype=np.float32
        )
        self.agent_tribes = np.array(
            [i % self.num_tribes for i in range(self.num_agents)], dtype=np.int32
        )
        self.agent_alive = np.ones(self.num_agents, dtype=bool)

        self.current_step = 0

        observations = [self._get_obs(i) for i in range(self.num_agents)]
        info = self._get_info()
        return observations, info

    # ------------------------------------------------------------------
    # Step  (accepts a list of actions, one per agent)
    # ------------------------------------------------------------------
    def step(
        self, actions: list[int]
    ) -> Tuple[list[np.ndarray], list[float], bool, bool, dict]:
        assert len(actions) == self.num_agents

        rewards = [0.0] * self.num_agents
        self.current_step += 1

        # --- 1. Compute desired new positions ----------------------------
        desired = self.agent_positions.copy()
        for i, act in enumerate(actions):
            if not self.agent_alive[i]:
                continue
            dr, dc = _DELTAS[act]
            nr = self.agent_positions[i, 0] + dr
            nc = self.agent_positions[i, 1] + dc
            # Clamp to grid bounds
            nr = max(0, min(self.grid_h - 1, nr))
            nc = max(0, min(self.grid_w - 1, nc))
            desired[i] = [nr, nc]

        # --- 2. Detect collisions (two+ agents on same cell) -------------
        occupied: Dict[Tuple[int, int], list[int]] = {}
        for i in range(self.num_agents):
            if not self.agent_alive[i]:
                continue
            key = (int(desired[i, 0]), int(desired[i, 1]))
            occupied.setdefault(key, []).append(i)

        for cell, agents_on_cell in occupied.items():
            if len(agents_on_cell) > 1:
                for i in agents_on_cell:
                    rewards[i] += self.cfg["collision_penalty"]

        # Commit positions (even if colliding — they share the cell)
        self.agent_positions = desired

        # --- 3. Food consumption -----------------------------------------
        for i in range(self.num_agents):
            if not self.agent_alive[i]:
                continue
            r, c = int(self.agent_positions[i, 0]), int(self.agent_positions[i, 1])
            if self.grid[r, c] == TILE_FOOD:
                self.grid[r, c] = TILE_EMPTY
                self.agent_energy[i] += self.cfg["energy_from_food"]
                rewards[i] += self.cfg["food_reward"]
                if self.cfg["food_respawn"]:
                    self._spawn_food(self.np_random, count=1)

        # --- 4. Energy decay + survival bonus ----------------------------
        for i in range(self.num_agents):
            if not self.agent_alive[i]:
                continue
            self.agent_energy[i] += self.cfg["energy_per_step"]
            rewards[i] += self.cfg["survival_bonus"]
            if self.agent_energy[i] <= 0.0:
                self.agent_energy[i] = 0.0
                self.agent_alive[i] = False

        # --- 5. Termination conditions -----------------------------------
        all_dead = not np.any(self.agent_alive)
        time_up = self.current_step >= self.max_steps
        terminated = all_dead
        truncated = time_up and not terminated

        observations = [self._get_obs(i) for i in range(self.num_agents)]
        info = self._get_info()
        return observations, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _get_obs(self, agent_idx: int) -> np.ndarray:
        """Return a flat float32 observation for one agent."""
        obs = np.zeros(self._obs_size, dtype=np.float32)

        if not self.agent_alive[agent_idx]:
            return obs  # dead agents see zeros

        ar, ac = int(self.agent_positions[agent_idx, 0]), int(
            self.agent_positions[agent_idx, 1]
        )

        # --- Local window ------------------------------------------------
        idx = 0
        for dr in range(-self.view_r, self.view_r + 1):
            for dc in range(-self.view_r, self.view_r + 1):
                wr, wc = ar + dr, ac + dc
                if 0 <= wr < self.grid_h and 0 <= wc < self.grid_w:
                    if self.grid[wr, wc] == TILE_FOOD:
                        obs[idx] = 0.25
                    else:
                        # Check for agents on this cell
                        for j in range(self.num_agents):
                            if j == agent_idx:
                                continue
                            if not self.agent_alive[j]:
                                continue
                            if (
                                int(self.agent_positions[j, 0]) == wr
                                and int(self.agent_positions[j, 1]) == wc
                            ):
                                # Encode tribe: 0.5 for tribe 0, up to 1.0
                                tribe_val = 0.5 + 0.5 * (
                                    self.agent_tribes[j] / max(self.num_tribes - 1, 1)
                                )
                                obs[idx] = tribe_val
                                break
                else:
                    # Out of bounds — encode as 0 (empty / wall)
                    obs[idx] = 0.0
                idx += 1

        # --- Internal state ----------------------------------------------
        obs[self._window_size] = self.agent_energy[agent_idx] / self.cfg[
            "initial_energy"
        ]
        obs[self._window_size + 1] = self.agent_tribes[agent_idx] / max(
            self.num_tribes - 1, 1
        )
        return obs

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------
    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
            "alive": int(np.sum(self.agent_alive)),
            "total_energy": float(np.sum(self.agent_energy)),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self._renderer is None:
            from src.envs.rendering import Renderer
            self._renderer = Renderer(self)
        return self._renderer.render(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _spawn_food(self, rng: np.random.Generator, count: int) -> None:
        """Place *count* food items on random empty cells."""
        empties = list(zip(*np.where(self.grid == TILE_EMPTY)))
        # Exclude cells where agents sit
        agent_cells = set()
        if self.agent_positions.size > 0:
            for i in range(self.num_agents):
                agent_cells.add(
                    (int(self.agent_positions[i, 0]), int(self.agent_positions[i, 1]))
                )
        empties = [c for c in empties if c not in agent_cells]
        count = min(count, len(empties))
        if count == 0:
            return
        chosen = rng.choice(len(empties), size=count, replace=False)
        for idx in chosen:
            r, c = empties[idx]
            self.grid[r, c] = TILE_FOOD

    def _random_empty_cells(
        self, rng: np.random.Generator, n: int
    ) -> list[Tuple[int, int]]:
        """Return *n* unique empty cell coordinates."""
        empties = list(zip(*np.where(self.grid == TILE_EMPTY)))
        chosen = rng.choice(len(empties), size=n, replace=False)
        return [empties[i] for i in chosen]

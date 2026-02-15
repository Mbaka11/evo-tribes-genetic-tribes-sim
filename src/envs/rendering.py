"""
EvoTribes — Pygame Renderer
============================

Draws the grid, agents (coloured by tribe), and food.
Overlay shows step count and the energy of agent 0.

The renderer is created lazily the first time `env.render()` is called
so that importing the environment does not require pygame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from src.envs.tribes_env import TribesEnv

# ---------------------------------------------------------------------------
# Render configuration — all visual knobs in one place
# ---------------------------------------------------------------------------
CELL_SIZE = 28            # pixels per grid cell
FOOD_RADIUS = 5           # circle radius for food items
AGENT_RADIUS = 9          # circle radius for agents
OVERLAY_HEIGHT = 36        # extra pixels at the bottom for text
FONT_SIZE = 18
BG_COLOR = (30, 30, 30)
GRID_LINE_COLOR = (50, 50, 50)
FOOD_COLOR = (80, 200, 80)
TEXT_COLOR = (220, 220, 220)

# Tribe palette — cycles if more tribes than colours
TRIBE_COLORS = [
    (100, 150, 255),   # blue
    (255, 100, 100),   # red
    (255, 220, 80),    # yellow
    (180, 100, 255),   # purple
    (100, 255, 200),   # teal
    (255, 160, 80),    # orange
]


class Renderer:
    """Pygame-based renderer for :class:`TribesEnv`."""

    def __init__(self, env: TribesEnv):
        import pygame

        pygame.init()
        self._pygame = pygame

        self.cell = CELL_SIZE
        self.width = env.grid_w * CELL_SIZE
        self.height = env.grid_h * CELL_SIZE + OVERLAY_HEIGHT

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("EvoTribes — Iteration 1")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", FONT_SIZE)

    # ------------------------------------------------------------------
    def render(self, env: TribesEnv) -> Optional[np.ndarray]:
        pg = self._pygame

        # Pump events so the window stays responsive
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                raise SystemExit

        self.screen.fill(BG_COLOR)

        # --- Grid lines --------------------------------------------------
        for x in range(0, env.grid_w * self.cell + 1, self.cell):
            pg.draw.line(
                self.screen, GRID_LINE_COLOR, (x, 0), (x, env.grid_h * self.cell)
            )
        for y in range(0, env.grid_h * self.cell + 1, self.cell):
            pg.draw.line(
                self.screen, GRID_LINE_COLOR, (0, y), (env.grid_w * self.cell, y)
            )

        # --- Food --------------------------------------------------------
        from src.envs.tribes_env import TILE_FOOD

        for r in range(env.grid_h):
            for c in range(env.grid_w):
                if env.grid[r, c] == TILE_FOOD:
                    cx = c * self.cell + self.cell // 2
                    cy = r * self.cell + self.cell // 2
                    pg.draw.circle(self.screen, FOOD_COLOR, (cx, cy), FOOD_RADIUS)

        # --- Agents ------------------------------------------------------
        for i in range(env.num_agents):
            if not env.agent_alive[i]:
                continue
            r, c = int(env.agent_positions[i, 0]), int(env.agent_positions[i, 1])
            cx = c * self.cell + self.cell // 2
            cy = r * self.cell + self.cell // 2
            tribe = int(env.agent_tribes[i])
            color = TRIBE_COLORS[tribe % len(TRIBE_COLORS)]
            pg.draw.circle(self.screen, color, (cx, cy), AGENT_RADIUS)

            # Small energy bar above the agent
            bar_w = self.cell - 6
            bar_h = 3
            energy_frac = env.agent_energy[i] / env.cfg["initial_energy"]
            energy_frac = max(0.0, min(1.0, energy_frac))
            bar_x = c * self.cell + 3
            bar_y = r * self.cell + 2
            pg.draw.rect(
                self.screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h)
            )
            pg.draw.rect(
                self.screen, (0, 220, 0), (bar_x, bar_y, int(bar_w * energy_frac), bar_h)
            )

        # --- Overlay text ------------------------------------------------
        overlay_y = env.grid_h * self.cell + 4
        alive_count = int(np.sum(env.agent_alive))
        agent0_energy = float(env.agent_energy[0]) if env.num_agents > 0 else 0.0
        text = (
            f"Step {env.current_step}/{env.max_steps}  |  "
            f"Alive {alive_count}/{env.num_agents}  |  "
            f"Agent0 energy {agent0_energy:.1f}"
        )
        surf = self.font.render(text, True, TEXT_COLOR)
        self.screen.blit(surf, (6, overlay_y))

        pg.display.flip()
        self.clock.tick(env.metadata["render_fps"])

        if env.render_mode == "rgb_array":
            return np.transpose(
                np.array(pg.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        return None

    # ------------------------------------------------------------------
    def close(self):
        self._pygame.quit()

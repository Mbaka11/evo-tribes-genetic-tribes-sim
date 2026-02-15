# Environment Design

> **Status:** Implemented in Iteration 1 — random agents only, no policies.

---

## World

- 2D grid with configurable width and height.
- Multiple agents assigned to tribes (round-robin).
- Food items spawn randomly on empty cells.
- Obstacles and hazard zones are **not yet implemented** (planned).

---

## Configurable Parameters

All defaults live in `src/envs/tribes_env.py :: DEFAULT_CONFIG`.

| Parameter           | Default | Description                                    |
| ------------------- | ------- | ---------------------------------------------- |
| `grid_width`        | 20      | Number of columns                              |
| `grid_height`       | 20      | Number of rows                                 |
| `num_agents`        | 6       | Total agents in the world                      |
| `num_tribes`        | 2       | Number of tribes (agents assigned round-robin) |
| `view_radius`       | 2       | Agent sees a (2r+1)×(2r+1) window              |
| `initial_energy`    | 100.0   | Starting energy per agent                      |
| `energy_per_step`   | -1.0    | Energy cost each step (negative = drain)       |
| `energy_from_food`  | 15.0    | Energy gained when eating food                 |
| `num_food`          | 10      | Food items on the map at start                 |
| `food_respawn`      | True    | Respawn a new food item when one is eaten      |
| `food_reward`       | 1.0     | Reward for eating food                         |
| `survival_bonus`    | 0.01    | Reward for being alive each step               |
| `collision_penalty` | -0.1    | Penalty when two+ agents share a cell          |
| `max_steps`         | 300     | Episode truncates after this many steps        |

Override any value by passing a `config` dict to `TribesEnv(config={...})`.

---

## Agent State

Each agent has:

- **position** — (row, col) on the grid
- **energy** — float, decreases each step, replenished by food
- **tribe id** — integer 0 … num_tribes-1
- **alive** — boolean; dies when energy hits 0

---

## Observation Space

A **flat float32 vector** of length `(2·view_radius+1)² + 2`.

### Structure

```
[ local_window_flat ..., normalised_energy, normalised_tribe_id ]
```

### Local window encoding (per cell)

| Value     | Meaning                                |
| --------- | -------------------------------------- |
| 0.0       | Empty / out of bounds                  |
| 0.25      | Food                                   |
| 0.5 – 1.0 | Agent present (value encodes tribe id) |

### Internal features

| Index           | Feature                     | Range   |
| --------------- | --------------------------- | ------- |
| window_size     | energy / initial_energy     | [0, 1+] |
| window_size + 1 | tribe_id / (num_tribes - 1) | [0, 1]  |

With the default `view_radius=2`, the window is 5×5 = 25 cells,
so the observation vector has **27 elements**.

---

## Action Space

`Discrete(5)` — one integer per agent per step.

| Action | Direction | Delta (row, col) |
| ------ | --------- | ---------------- |
| 0      | Stay      | (0, 0)           |
| 1      | North     | (-1, 0)          |
| 2      | South     | (+1, 0)          |
| 3      | East      | (0, +1)          |
| 4      | West      | (0, -1)          |

Movement is clamped to grid bounds (no wrap-around).

Future extensions (not yet implemented):

- Attack
- Build wall
- Share resource

---

## Reward

| Source                             | Value                               |
| ---------------------------------- | ----------------------------------- |
| Eating food                        | `+food_reward` (default 1.0)        |
| Surviving a step                   | `+survival_bonus` (default 0.01)    |
| Collision (2+ agents on same cell) | `+collision_penalty` (default -0.1) |

---

## Episode Termination

- **Terminated:** all agents are dead (energy ≤ 0).
- **Truncated:** `max_steps` reached while at least one agent is alive.

---

## Rendering

Implemented in `src/envs/rendering.py` using Pygame.

Displays:

- Dark grid with lines
- Green circles = food
- Coloured circles = agents (colour = tribe)
- Small energy bars above each agent
- Overlay text: step counter, alive count, agent 0 energy

---

## How to Run

### Install dependencies

```bash
pip install gymnasium numpy pygame pytest
```

### Run the demo (random agents, Pygame window)

```bash
python -m scripts.demo
```

### Run smoke tests

```bash
python -m pytest tests/test_env_smoke.py -v
```

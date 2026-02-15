# Iteration 01 — Grid World & Random Agents

**Version:** `0.1.0`
**Date:** 2026-02-14
**Branch:** `main`

---

### 1. Goal

Build a minimal, fully runnable Gymnasium environment where multiple agents
wander a 2D grid randomly, eat food, lose energy, and eventually die — with
real-time Pygame rendering and automated smoke tests.

No intelligence, no evolution. Just the **stage** and the **physics**.

---

### 2. Why This Matters

Everything in EvoTribes sits on top of the environment.
Policies read observations from it. The genetic algorithm scores agents by
their rewards from it. Metrics are computed from what happens inside it.

If the environment has the wrong shape, a broken reward, or a silent bug,
every later iteration inherits that problem.

Iteration 1 exists to **validate the foundation** before building on it.

---

### 3. What Was Implemented

| File | Change Type | Description |
|---|---|---|
| `src/__init__.py` | Created | Package marker |
| `src/envs/__init__.py` | Created | Exports `TribesEnv` |
| `src/envs/tribes_env.py` | Created | Full Gymnasium environment |
| `src/envs/rendering.py` | Created | Pygame-based renderer |
| `scripts/demo.py` | Created | Entry point — runs random agents with rendering |
| `tests/test_env_smoke.py` | Created | 12 smoke tests |
| `docs/01_environment.md` | Modified | Full environment spec |

---

### 4. Architecture & Design Decisions

#### Multi-agent inside a single Gymnasium env

Gymnasium is designed for single-agent environments. Instead of fighting that,
we keep one `TribesEnv` instance but expose **lists** for observations, actions,
and rewards. The outer loop iterates over agents:

```
observations, info = env.reset()          # list of obs, one per agent
actions = [pick_action(obs) for obs in observations]
observations, rewards, term, trunc, info = env.step(actions)
```

**Why?** This keeps the interface simple — no message-passing framework, no
multi-env wrappers. When we add policies later, each policy just receives its
own observation and returns one action.

#### Configuration dict, not constructor args

All parameters live in `DEFAULT_CONFIG` (a plain Python dict). You override
them by passing `config={...}` at construction. This means:

- No magic numbers scattered in the code.
- Scenario files (Iteration 4+) will just be JSON/YAML dicts.
- Tests can tweak one parameter without knowing the full list.

#### Grid representation

- **Tile grid** (`self.grid`): a 2D numpy array of ints. `0` = empty, `1` = food.
- **Agent positions**: a separate `(num_agents, 2)` array of `(row, col)` ints.

Agents are NOT stored on the tile grid. This avoids collision between the food
layer and the agent layer and makes lookups simpler.

---

### 5. Mathematical Considerations

#### 5.1 Observation vector

Each agent sees a square window of `(2r+1)²` cells centred on itself, plus 2
internal features.

$$
\text{obs\_size} = (2 \cdot r + 1)^{2} + 2
$$

Where:
- $r$ = `view_radius` (default 2)

With $r = 2$: obs\_size $= 5^{2} + 2 = 27$ floats.

**Intuition:** The agent looks around in a 5×5 square (25 cells) and also
knows its own energy level and which tribe it belongs to.

#### 5.2 Cell encoding

Each cell in the window is encoded as a single float:

| Value | Meaning |
|---|---|
| $0.0$ | Empty or out-of-bounds |
| $0.25$ | Food |
| $0.5 + 0.5 \cdot \frac{t}{T - 1}$ | Agent of tribe $t$ out of $T$ tribes |

With 2 tribes:
- Tribe 0 agent → $0.5 + 0.5 \cdot \frac{0}{1} = 0.5$
- Tribe 1 agent → $0.5 + 0.5 \cdot \frac{1}{1} = 1.0$

**Why this encoding?** By packing everything into $[0, 1]$, the observation
space is a simple `Box(low=0, high=1)`. Neural network policies (later
iterations) work best when inputs are normalised.

#### 5.3 Energy dynamics

Each step, every living agent's energy changes:

$$
E_{t+1} = E_{t} + \Delta_{\text{step}} + \Delta_{\text{food}}
$$

Where:
- $E_t$ = energy at step $t$
- $\Delta_{\text{step}}$ = `energy_per_step` (default $-1.0$) — always applied
- $\Delta_{\text{food}}$ = `energy_from_food` (default $+15.0$) — only if the
  agent is on a food cell

**Death condition:** $E_t \leq 0 \Rightarrow$ agent is marked dead and
produces zero observations from that point on.

**Intuition:** Energy is a ticking clock. Agents must find food to keep the
clock from hitting zero. With default values, an agent that never eats dies
after 100 steps. An agent that eats one food extends its life by 15 steps.

#### 5.4 Reward function

Per step, each living agent receives:

$$
R = R_{\text{food}} + R_{\text{survival}} + R_{\text{collision}}
$$

| Component | Value | When |
|---|---|---|
| $R_{\text{food}}$ | $+1.0$ | Agent steps on a food cell |
| $R_{\text{survival}}$ | $+0.01$ | Every step the agent is alive |
| $R_{\text{collision}}$ | $-0.1$ | Two or more agents occupy the same cell |

**Intuition:**
- Food reward teaches agents to seek resources.
- Survival bonus provides a tiny signal for "existing is good" — this prevents
  policies from learning to die quickly if food is scarce.
- Collision penalty discourages stacking.

Right now agents are random, so rewards are just noise. But the reward
structure is designed so that **future policies** will have clear gradients
to learn from.

#### 5.5 Episode length and survival math

With default parameters:
- `initial_energy` = 100, `energy_per_step` = −1 → an agent that never eats
  survives exactly **100 steps**.
- `max_steps` = 300 → to survive the full episode, an agent must eat at least:

$$
\text{food\_needed} = \left\lceil \frac{\text{max\_steps} - \text{initial\_energy}}{\text{energy\_from\_food}} \right\rceil = \left\lceil \frac{300 - 100}{15} \right\rceil = \left\lceil 13.3 \right\rceil = 14 \text{ food items}
$$

With 6 random agents competing for 10–12 food items (that respawn), some
agents will survive the full episode and some won't — which is exactly the
right pressure for a future genetic algorithm.

---

### 6. Algorithms & Logic

#### Step loop (one call to `env.step(actions)`)

```
1. For each living agent:
      compute desired position = current position + action delta
      clamp to grid bounds

2. Detect collisions:
      group agents by their desired cell
      if 2+ agents share a cell → apply collision_penalty to each

3. Commit new positions

4. Food consumption:
      for each living agent:
         if agent is on a food cell:
            remove food from grid
            add energy_from_food to agent
            add food_reward to agent's reward
            if food_respawn: spawn a new food on a random empty cell

5. Energy decay:
      for each living agent:
         energy += energy_per_step (negative = drain)
         reward += survival_bonus
         if energy ≤ 0: mark agent as dead

6. Check termination:
      terminated = all agents dead
      truncated  = step count ≥ max_steps (and not terminated)
```

#### Food spawning

Food is placed on randomly chosen empty cells (cells that are neither food
nor occupied by an agent). Uses `numpy.random.Generator.choice` without
replacement.

#### Movement clamping

Agents cannot walk off the grid. If a move would place them outside bounds,
their position is clamped to the nearest border cell. There is no wrap-around.

---

### 7. Key Concepts Explained

#### Gymnasium Environment

[Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI Gym) is the
standard Python library for reinforcement learning environments. It defines
a simple interface:

- `reset()` → initial observation
- `step(action)` → next observation, reward, terminated, truncated, info

Every RL library (Stable Baselines, RLlib, CleanRL) expects this interface.
By building on Gymnasium, EvoTribes can plug into any of them later.

#### Observation Space

The **observation space** defines what an agent can "see" each step. In
EvoTribes it's a flat array of floats between 0 and 1. Think of it as the
agent's sensory input — like a small photograph of its neighbourhood plus a
reading of its energy gauge.

#### Action Space

The **action space** defines what an agent can "do" each step. Here it's
`Discrete(5)`: one of five integers (stay, north, south, east, west). The
agent picks one integer, and the environment moves it accordingly.

#### Reward Signal

The **reward** tells the agent how "good" that step was. Positive rewards
encourage a behaviour; negative rewards discourage it. In this iteration
agents don't learn, so rewards are just recorded — but the structure is
designed so that policies in Iteration 2+ will have meaningful signals.

#### Episode

An **episode** is one full run from `reset()` until termination or truncation.
In EvoTribes, an episode is one "life" of the simulation — up to `max_steps`
ticks, or until every agent dies.

---

### 8. Configuration & Parameters

| Parameter | Default | What It Controls |
|---|---|---|
| `grid_width` | 20 | Number of columns in the grid |
| `grid_height` | 20 | Number of rows in the grid |
| `num_agents` | 6 | Total agents spawned |
| `num_tribes` | 2 | Tribes (agents assigned round-robin: 0,1,0,1…) |
| `view_radius` | 2 | Half-size of the observation window |
| `initial_energy` | 100.0 | Energy each agent starts with |
| `energy_per_step` | −1.0 | Energy change per step (negative = drain) |
| `energy_from_food` | 15.0 | Energy gained when eating |
| `num_food` | 10 | Food items at episode start |
| `food_respawn` | True | Spawn a new food when one is eaten |
| `food_reward` | 1.0 | Reward for eating food |
| `survival_bonus` | 0.01 | Reward for surviving each step |
| `collision_penalty` | −0.1 | Penalty when agents collide |
| `max_steps` | 300 | Episode length limit |

---

### 9. How to Run

```bash
# Install dependencies
pip install gymnasium numpy pygame pytest

# Run the visual demo (random agents)
python -m scripts.demo

# Run smoke tests
python -m pytest tests/test_env_smoke.py -v
```

---

### 10. What You Should See

**Pygame window (560×596 px):**
- Dark 20×20 grid with thin lines
- Green dots = food
- Blue circles = Tribe 0, Red circles = Tribe 1
- Tiny green energy bars above agents
- Bottom overlay: `Step 42/300 | Alive 6/6 | Agent0 energy 85.0`

**Console output:**
```
EvoTribes — Iteration 1 demo
Grid 20x20, 6 agents, 2 tribes
Running random actions — close the window to quit.

  step   50  |  alive 6/6  |  total energy 542.0
  step  100  |  alive 6/6  |  total energy 480.3
  ...

Episode finished at step 300.
  Reason: max steps reached
  Alive:  4/6
```

**Tests:**
```
12 passed in ~0.7s
```

---

### 11. Known Limitations & Bugs

1. **Observation doesn't distinguish "out of bounds" from "empty"** — both
   encode as 0.0. Future iterations may add a wall channel.
2. **Collision handling is soft** — agents share cells and just take a penalty.
   There is no blocking or pushing.
3. **Only one agent can eat food per cell per step** — the first agent in the
   loop gets it. If two agents land on the same food, only one benefits.
4. **No seed control in the demo beyond the initial reset** — action sampling
   uses Gymnasium's internal RNG, not a user-controlled seed.
5. **Renderer depends on Pygame** — headless environments (CI, servers) should
   not call `render()`.

---

### 12. Docs Modified

| Doc File | What Changed |
|---|---|
| `docs/01_environment.md` | Rewritten: full spec with obs/action/reward tables, config table, run commands |

---

### 13. Test Coverage

| Test | What It Checks |
|---|---|
| `test_reset_returns_observations_and_info` | `reset()` returns a list + dict |
| `test_observation_shape` | Each obs matches `observation_space.shape` |
| `test_observation_values_in_range` | All values in [0, 1] |
| `test_action_space_type` | Action space is `Discrete(5)` |
| `test_sample_action_is_valid` | Sampled actions are in range |
| `test_step_returns_correct_structure` | `step()` returns (list, list, bool, bool, dict) |
| `test_multiple_steps` | Can run 10 steps without crash |
| `test_energy_decreases` | Total energy drops after 5 stay-actions |
| `test_info_keys` | Info dict has `step`, `alive`, `total_energy` |
| `test_initial_alive_count` | All agents alive after reset |
| `test_episode_ends_on_max_steps` | Truncation at `max_steps` |
| `test_episode_ends_when_all_dead` | Termination when all energy = 0 |

---

### 14. Version History

| Version | Change |
|---|---|
| `0.1.0` | Initial grid environment, Pygame renderer, demo script, 12 smoke tests |

---

### 15. Next Iteration Preview

**Iteration 2 — Policy Interface & Swappable Agent Brains**

1. **`BasePolicy`** — an abstract class that defines `select_action(obs) → int`.
2. **`RandomPolicy`** — wraps the current random logic behind that interface.
3. **`MLPPolicy`** — a small neural network (random weights, no training yet)
   that maps the 27-float observation to one of 5 actions.
4. **Policy swap test** — prove that swapping `RandomPolicy` for `MLPPolicy`
   requires zero changes to the environment or demo loop.
5. Update `docs/02_policies.md`.

**Why this order?** The genetic algorithm (Iteration 3) evolves policy
*weights*. We need the policy object to exist first so that evolution has
something to operate on.

---

### 16. Questions & Open Issues

- Should out-of-bounds cells have a distinct encoding (e.g., $-1$) instead of
  sharing $0.0$ with empty cells?
- Should collisions block movement instead of allowing cell-sharing?
- Should food consumption be shared when multiple agents land on the same cell?

# Iteration 03 — Genetic Algorithm

**Version:** `0.3.0`
**Date:** 2026-02-15
**Branch:** `main`

---

### 1. Goal

Implement the core genetic algorithm that evolves MLP neural network
policies. The GA treats each policy's weight vector as a chromosome
and applies selection, crossover, and mutation to breed better agents
over generations.

---

### 2. Why This Matters

Iterations 1 and 2 gave us a working environment and pluggable policies
with flat parameter access. But those policies had random or untrained
weights — they couldn't learn.

The genetic algorithm is the **learning engine**. It closes the loop:

- Policies that collect more reward survive and reproduce.
- Policies that fail are replaced by offspring of the winners.
- Over generations, the population converges toward competent agents.

This is the first time EvoTribes agents can actually **improve**.

---

### 3. What Was Implemented

| File                           | Change Type | Description                                            |
| ------------------------------ | ----------- | ------------------------------------------------------ |
| `src/evolution/__init__.py`    | Created     | Package exports for all GA components                  |
| `src/evolution/fitness.py`     | Created     | `evaluate_agent()`, `evaluate_robust()` — run episodes |
| `src/evolution/selection.py`   | Created     | `tournament_selection()` with configurable k           |
| `src/evolution/crossover.py`   | Created     | `uniform_crossover()` — gene-level random mix          |
| `src/evolution/mutation.py`    | Created     | `gaussian_mutate()`, `adaptive_mutation_std()`         |
| `src/evolution/population.py`  | Created     | `Population` class — full GA loop with logging         |
| `scripts/train.py`             | Created     | CLI training script with argparse                      |
| `scripts/replay.py`            | Created     | Load a saved .npy and replay with Pygame               |
| `tests/test_evolution.py`      | Created     | 40+ tests across 10 test classes                       |
| `docs/03_genetic_algorithm.md` | Rewritten   | Full API reference replacing the placeholder           |

---

### 4. Architecture & Design Decisions

#### Modular GA components

Each GA operator is a standalone function in its own module:

```
src/evolution/
├── fitness.py      evaluate_agent(), evaluate_robust()
├── selection.py    tournament_selection()
├── crossover.py    uniform_crossover()
├── mutation.py     gaussian_mutate(), adaptive_mutation_std()
└── population.py   Population — orchestrates the loop
```

**Why modular?** Each function can be tested, replaced, or extended
independently. Want to try rank selection? Just write a new function
and swap it into `Population._evolve_one_generation()`.

#### Shared-policy evaluation

All agents in the environment use the **same** chromosome during
evaluation. Fitness is the total reward summed across all agents.

**Why?** We're evolving a single brain that controls an entire tribe.
Individual-agent evaluation would require separate chromosomes per
agent, which we may explore in a future iteration.

#### Design decision log

| Decision          | Choice                 | Alternatives Considered           |
| ----------------- | ---------------------- | --------------------------------- |
| Selection method  | Tournament (k=3)       | Roulette wheel, rank selection    |
| Crossover method  | Uniform                | Single-point, arithmetic          |
| Mutation method   | Gaussian (σ=0.02)      | Uniform, Cauchy                   |
| Mutation schedule | Step decay (0.9/20gen) | Linear, exponential, none         |
| Elitism count     | 2                      | 0, 1, 5                           |
| Fitness episodes  | 3 (averaged)           | 1 (noisy), 5 (slow)               |
| Early stopping    | 30-gen plateau         | Fixed generations, fitness target |
| Evaluation style  | Sequential             | Parallel (multiprocessing)        |

---

### 5. Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Population   │────►│  Evaluate    │────►│  Fitnesses   │
│  (chromosomes)│     │  (N episodes)│     │  (array)     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────┘
                     ▼
              ┌─────────────┐
              │  Selection   │ Tournament k=3
              │  (2 parents) │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  Crossover   │ Uniform 50/50
              │  (1 child)   │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  Mutation    │ Gaussian σ (adaptive)
              │  (1 mutant)  │
              └──────┬──────┘
                     │
              ┌──────▼──────────┐
              │  Next generation │ Elites + offspring
              └─────────────────┘
```

---

### 6. Test Coverage

40+ tests across 10 test classes:

| Class                     | Tests | What it verifies                                |
| ------------------------- | ----- | ----------------------------------------------- |
| `TestEvaluateAgent`       | 4     | Returns float, deterministic with seed, finite  |
| `TestEvaluateRobust`      | 3     | Averaging, single-episode equivalence           |
| `TestTournamentSelection` | 7     | Copy semantics, shape, bounds, edge cases       |
| `TestUniformCrossover`    | 6     | Shape, values from parents, 50/50 split, dtype  |
| `TestGaussianMutate`      | 5     | Shape, immutability, noise bounds, dtype        |
| `TestAdaptiveMutationStd` | 4     | Initial value, decay, stability, custom params  |
| `TestPopulationInit`      | 3     | Size, chromosome shape, diversity               |
| `TestPopulationEvolution` | 5     | run() output, config.json, metrics.csv, elitism |
| `TestPopulationDiversity` | 3     | Zero for identical, nonzero for diverse         |
| `TestEvolveOneGeneration` | 3     | Size preserved, shapes correct, elites kept     |

---

### 7. How to Use

#### Training

```bash
# Quick test (fast, small)
python -m scripts.train --population 10 --generations 5

# Full run with defaults
python -m scripts.train

# Custom experiment
python -m scripts.train --population 50 --generations 100 \
    --mutation-std 0.02 --elitism 2 --output runs/exp_001 --seed 42
```

#### Replay an evolved agent

```bash
python -m scripts.replay --checkpoint runs/default/best_final.npy
```

---

### 8. Concrete Examples

#### Example 1: Tournament Selection

```
Population (5 individuals):
  Ind 0: fitness = 2.1
  Ind 1: fitness = 5.3   ← best in tournament
  Ind 2: fitness = 1.0
  Ind 3: fitness = 4.7
  Ind 4: fitness = 3.2

Tournament k=3, randomly picks indices [1, 2, 4]:
  fitnesses = [5.3, 1.0, 3.2]
  winner = Ind 1 (fitness 5.3)

A copy of Ind 1's chromosome becomes a parent.
```

#### Example 2: Uniform Crossover

```
Parent A: [0.50, -0.12,  0.03,  0.88, -0.31]
Parent B: [0.10,  0.77, -0.45, -0.02,  0.66]
Coin:     [  A  ,   B  ,   A  ,   B  ,   A  ]

Child:    [0.50,  0.77,  0.03, -0.02, -0.31]
```

#### Example 3: Gaussian Mutation

```
Before:  [0.50,  0.77,  0.03, -0.02, -0.31]
Noise:   [0.01, -0.005, 0.013, 0.002, -0.008]  (drawn from N(0, 0.02²))
After:   [0.51,  0.765, 0.043, -0.018, -0.318]
```

#### Example 4: Adaptive Mutation Decay

```
Gen  0:  σ = 0.0200  (initial — explore widely)
Gen 20:  σ = 0.0180  (first decay step)
Gen 40:  σ = 0.0162  (second decay step)
Gen 99:  σ = 0.0131  (fine-tuning mode)
```

---

### 9. Output Directory Structure

```
runs/exp_001/
├── config.json       ← reproducible experiment settings
├── metrics.csv       ← per-generation: best/avg/worst fitness, diversity, σ
├── best_gen0000.npy  ← checkpoint chromosome
├── best_gen0010.npy
├── ...
└── best_final.npy    ← overall best chromosome
```

---

### 10. Expected Behaviour

**Early generations (0–20):** Fitness is low and noisy. All chromosomes
are random Xavier-initialised weights. Diversity is high. You'll see
rapid improvement as obviously bad solutions are weeded out.

**Mid generations (20–60):** Fitness climbs steadily but more slowly.
The population starts converging — diversity drops. Mutation std
begins decaying, shifting from exploration to exploitation.

**Late generations (60–100):** Progress slows or plateaus. If the best
fitness hasn't improved for 30 generations, early stopping triggers.
The final chromosome is saved to `best_final.npy`.

**What "good" looks like:** Agents move toward food rather than
wandering randomly. Survival time increases. Total reward trends
upward across generations.

---

### 11. Gotchas & Debugging

- **Flat fitness across all individuals:** Check that `set_params()`
  is actually loading different chromosomes. A bug here means all
  agents behave identically regardless of their chromosome.

- **Fitness decreasing:** Verify elitism is working — the top 2 should
  never be lost. Also check mutation std isn't too high.

- **Very slow training:** Each generation evaluates `population_size ×
eval_episodes` full episodes. With defaults: 50 × 3 = 150 episodes
  per generation. Reduce population or eval_episodes for faster iteration.

- **`runs/` directory missing:** The training script creates it
  automatically via `os.makedirs(..., exist_ok=True)`.

---

### 12. What Changed From Previous Iterations

| Iteration | What It Added                           | What This Iteration Uses From It |
| --------- | --------------------------------------- | -------------------------------- |
| 1         | Grid environment, food, energy, render  | TribesEnv for fitness evaluation |
| 2         | Policy interface, MLPPolicy, get/set    | MLPPolicy as the evolvable brain |
| 3 (this)  | GA: selection, crossover, mutation, pop | —                                |

---

### 13. Files Modified

```
src/evolution/__init__.py     (new)
src/evolution/fitness.py      (new)
src/evolution/selection.py    (new)
src/evolution/crossover.py    (new)
src/evolution/mutation.py     (new)
src/evolution/population.py   (new)
scripts/train.py              (new)
scripts/replay.py             (new)
tests/test_evolution.py       (new)
docs/03_genetic_algorithm.md  (rewritten)
docs/notes/iteration_03.md    (new)
VERSION                       (0.2.0 → 0.3.0)
CHANGELOG.md                  (added 0.3.0 entry)
```

---

### 14. Metrics & Observations

_(To be filled after first training run.)_

Expected metrics from `metrics.csv`:

- `best_fitness`: should trend upward
- `avg_fitness`: should trend upward, lagging behind best
- `diversity`: should decrease gradually
- `mutation_std`: step-decays from 0.02

---

### 15. Conceptual Q&A (Design Session — Feb 17, 2026)

These questions came up during review and are captured here as design
context for future iterations.

---

#### Is GA a type of reinforcement learning? Are rewards the same concept?

**No — GA is not RL** — but they are related, and rewards play a
similar role in both.

| Aspect            | Genetic Algorithm (GA)           | Reinforcement Learning (RL)           |
| ----------------- | -------------------------------- | ------------------------------------- |
| How it learns     | Selection / crossover / mutation | Gradient descent on policy parameters |
| Reward use        | Fitness score (episode total)    | Per-step signal for gradient updates  |
| Needs backprop?   | No                               | Yes (usually)                         |
| Credit assignment | None — episode-level only        | Propagates reward back through steps  |
| Sample efficiency | Low — many episodes needed       | Higher — learns from each step        |
| Parallelism       | Natural — evaluate independently | Harder — depends on algorithm         |

In GA, "fitness" is just the total reward over a full episode —
the algorithm doesn't know or care which step caused it. This is
called **black-box optimization**: we only see the output (total
reward), not the internals (which action was good at step 47).

RL, by contrast, does **credit assignment** — it figures out that
moving right at step 47 caused the food pickup at step 48, and
strengthens that association.

Both use rewards to shape behavior. GA is sometimes called
**neuroevolution** when it evolves neural network weights.

---

#### What is Xavier initialization?

When you create a neural network with random weights, the scale of
those weights matters enormously.

- **Too large**: activations explode — every layer outputs huge
  numbers, gradients blow up.
- **Too small**: activations vanish — every layer outputs near-zero,
  the network can't learn anything useful.

**Xavier (Glorot) initialization** solves this by choosing the
initial weight scale based on the layer's size:

```
W ~ Uniform(-√(6 / (fan_in + fan_out)), +√(6 / (fan_in + fan_out)))
```

where `fan_in` = number of inputs, `fan_out` = number of outputs.

For our first layer (27 inputs → 16 outputs):

```
limit = √(6 / (27 + 16)) = √(6/43) ≈ 0.374
W ~ Uniform(-0.374, +0.374)
```

This keeps the signal variance roughly constant as it passes through
layers — neither exploding nor vanishing. It gives the GA a
reasonable starting distribution of weights to evolve from.

---

#### Can agents see their surroundings? How do they know where food is?

**Yes — agents have a local vision window.**

Each agent observes a **5×5 grid** centered on its position
(view_radius = 2, so 2 cells in every direction = 25 cells total),
plus 2 internal features:

```
Observation (27 values total):
  [0]  ... [24]  →  5×5 vision window (25 floats)
  [25]           →  normalized energy (your_energy / 100)
  [26]           →  tribe ID (0.0 or 1.0 when normalized)

Cell encoding in the window:
  0.00 = empty
  0.25 = food ← the agent can see this!
  0.50 = friendly agent (same tribe)
  1.00 = enemy agent (different tribe)
```

So an agent looking right at food will have `0.25` in the
corresponding window cell. In theory, the MLP can learn to:

1. Detect which window cell has `0.25`.
2. Map that position to a movement direction.
3. Move toward food reliably.

This is the core learning signal. The GA should discover this
mapping over generations.

---

#### What should we see right now? Current vs future behavior

**RIGHT NOW (Iteration 3 — random weights, before training):**

- Agents wander randomly across the grid
- Food collection is accidental — they don't navigate toward it
- All agents die at step ~100-150 when energy runs out
- No difference between tribes

**AFTER A SHORT TRAINING RUN (50 pop, 20-30 gens):**

- Agents start moving toward food visible in their window
- Average survival time increases
- Total reward per episode trends upward in `metrics.csv`
- You won't see strategy — just "walk toward the nearest food"

**WHAT IS NOT POSSIBLE YET:**

- Tribe vs tribe — both tribes run the same shared chromosome
- Attacking other agents — no attack action exists
- Building walls — no wall tile type
- Food farming — agents can't stay still to "claim" an area

Everything listed as "not possible yet" requires new actions, new
tile types, or per-tribe evolution — planned for later iterations.

---

#### Are tribes learning separately or sharing one brain?

**Currently: one shared brain for all agents** (including both tribes).

Every agent — tribe 0 and tribe 1 — loads the **same chromosome**
during evaluation. There is no tribal competition.

To get real tribal competition and divergent behavior:

- **Per-tribe chromosomes**: tribe 0 evolves its own chromosome,
  tribe 1 evolves a separate one, and fitness depends on how well
  you beat the other tribe (competitive coevolution).
- **Per-agent chromosomes**: every individual agent evolves
  independently (much more complex, computationally expensive).

With competitive coevolution, one tribe could dominate and drive
the other toward extinction — this is realistic and intentional.
It mirrors evolutionary arms races (predator/prey, competing species).

---

#### Can one tribe dominate and erase the other? Is that a problem?

**Yes, it can — and that's the interesting part.**

If one tribe evolves a better strategy, it can outcompete the other
for food, causing the losing tribe to die faster and potentially
to "go extinct" within a single episode.

This isn't a bug — it's emergent competitive dynamics. To study it:

- Tribe A and B evolve separate policies (competitive coevolution).
- Fitness = "how much food did YOUR tribe collect?" or
  "how long did YOUR tribe survive vs. the other?"
- An arms race emerges: tribe A learns to steal food → tribe B
  learns to block → tribe A learns to attack → etc.

If you want **both tribes to coexist long-term**, you'd need a
cooperative fitness function or a mechanically enforced balance
(e.g., respawn dead agents from the same tribe).

---

#### What behaviors are you aiming for long-term, and is it achievable?

**Target behaviors (roughly in order of difficulty):**

| Behavior                                   | Requires                                          | Est. Iteration |
| ------------------------------------------ | ------------------------------------------------- | -------------- |
| Navigate toward food                       | Current architecture (GA + vision)                | 3–4            |
| Survive longer than random                 | Current architecture                              | 3–4            |
| Tribal competition (one tribe beats other) | Per-tribe chromosomes                             | 4–5            |
| Avoid enemy agents                         | Per-tribe evolution + enemy encoding in obs       | 4–5            |
| Attack other agents (combat)               | New action: `ATTACK`, health stat, damage rewards | 5–6            |
| Territory control (stay near food zones)   | Reward for staying in high-food regions           | 5–6            |
| Block others with walls                    | New action: `BUILD_WALL`, wall tile type          | 6–7            |
| Farm food (stay + harvest repeatedly)      | Stationary harvest action or area-claim mechanic  | 6–7            |
| Water/terrain biomes                       | Tile types: water, fertile, barren                | 6–8            |
| Exploit weaker agents                      | Strength/aggression stat, combat system           | 7–9            |
| Physical traits (speed, strength, vision)  | Per-agent attributes in obs + effect on mechanics | 8–10           |
| Complex emergent group strategies          | All of the above + many training generations      | 10+            |

**Is it all achievable with GA?**

Yes, with caveats:

- GA works well for finding strategies in complex environments.
- The key constraint is **reward design**: GA can only optimize
  what you measure. If you want "farming" to emerge, the reward
  structure must make farming more profitable than roaming.
- The bigger the action space and behavior space, the harder the
  GA has to work. You may need larger populations, more generations,
  or eventually switch to RL for some components.
- The current architecture (Gymnasium env + MLPPolicy + GA) can
  support all these extensions — just add tile types, actions,
  observations, and reward terms incrementally.

---

#### Other mutation strategies besides Gaussian — how to choose?

| Method         | How it works                           | Best for                                  |
| -------------- | -------------------------------------- | ----------------------------------------- |
| **Gaussian**   | Add N(0, σ²) noise to every gene       | Continuous params, smooth landscapes      |
| **Uniform**    | Replace random genes with uniform rand | Wide exploration, flat fitness landscapes |
| **Cauchy**     | Add Cauchy-distributed noise           | Escaping local optima (heavy tails)       |
| **Polynomial** | Bounded perturbation with shape param  | Constrained search spaces (e.g., [0,1])   |
| **Bit-flip**   | Randomly flip 0↔1 bits                 | Binary chromosomes only                   |
| **Creep**      | Tiny step in random direction          | Fine-tuning near a known good solution    |

**How to choose:**

- Neural network weights (our case) → **Gaussian** is the standard
  choice. It's smooth, unbiased, and scales naturally.
- If training gets stuck in local optima → try **Cauchy** (heavy
  tails make large jumps more likely).
- If you want bounded mutations (e.g., weights forced to [-1, 1])
  → try **Polynomial**.

For EvoTribes we'll stay with Gaussian. It's the most studied and
well-understood choice for neuroevolution.

---

### 16. Open Questions

- Per-tribe chromosomes: when is the right iteration to introduce competitive coevolution?
- Should fitness include tribe-relative performance (beating the other tribe) or just survival?
- Would per-agent chromosomes (coevolution) for 6 individual agents emerge interesting group dynamics?
- How much would parallel evaluation (multiprocessing) speed up training?
- Is 533 parameters (27→16→5 MLP) enough depth to learn complex strategies?
- How do we visualise training progress without a notebook? (fitness curve, live plot?)

---

### 17. Iteration 4 Preview

Priority candidates based on the design session:

- **Per-tribe evolution**: give each tribe its own chromosome, introduce
  competitive coevolution — this is the biggest unlock for emergent behavior.
- **Training visualisation**: plot `metrics.csv` fitness curve and diversity
  over generations (matplotlib or ASCII plot in terminal).
- **Attack action**: add a 6th action (`ATTACK`), a health stat separate
  from energy, and rewards for combat — first step toward complex strategies.
- **Parallel evaluation**: use `multiprocessing` to evaluate chromosomes in
  parallel — 50 individuals × 3 episodes is 150 sequential episodes per
  generation, which is slow on CPU.
- **Scenario system**: configurable environment presets (dense food, sparse
  food, large grid, more agents) to test strategy robustness.

---

### 18. Version

`0.3.0` — Genetic Algorithm: selection, crossover, mutation, population
manager, training script, replay script, 40+ new tests.

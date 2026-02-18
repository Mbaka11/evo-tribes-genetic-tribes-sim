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

### 15. Open Questions

- Should we try per-tribe chromosomes instead of one shared brain?
- Would per-agent chromosomes (coevolution) emerge interesting strategies?
- How much would parallel evaluation (multiprocessing) speed things up?
- Is 533 parameters enough, or should we increase network size?

---

### 16. Iteration 4 Preview

Possible directions:

- **Metrics & visualisation**: Plot fitness curves, diversity, render
  best agent at each generation.
- **Tribe-level evolution**: Separate chromosomes per tribe.
- **Scenario system**: Configurable challenges (few/many food, large/small
  grid, more agents).
- **Parallel evaluation**: Use multiprocessing for faster training.

---

### 17. Version

`0.3.0` — Genetic Algorithm: selection, crossover, mutation, population
manager, training script, replay script, 40+ new tests.

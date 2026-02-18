# Genetic Algorithm

> Reference documentation for `src/evolution/`.
> For a beginner-friendly explanation, see
> [docs/learning/genetic_algorithms.md](learning/genetic_algorithms.md).

---

## Overview

The genetic algorithm (GA) evolves MLP neural network policies by
treating their weight vectors as **chromosomes** — flat arrays of 533
floating-point numbers that can be combined, mutated, and selected.

No backpropagation is needed. We never compute gradients. Instead,
we let natural selection do the work: policies that collect more reward
survive and reproduce; policies that perform poorly are replaced.

---

## Module Layout

```
src/evolution/
├── __init__.py      Package exports
├── fitness.py       evaluate_agent(), evaluate_robust()
├── selection.py     tournament_selection()
├── crossover.py     uniform_crossover()
├── mutation.py      gaussian_mutate(), adaptive_mutation_std()
└── population.py    Population class — orchestrates the full loop
```

---

## Evolution Loop

Each generation follows these steps:

```
┌─────────────────────────────────────────┐
│  1. Evaluate every chromosome           │
│     → run N episodes, average rewards   │
│                                         │
│  2. Log metrics (CSV)                   │
│                                         │
│  3. Check early stop                    │
│     → plateau for 30 gens? stop.        │
│                                         │
│  4. Elitism                             │
│     → copy top 2 individuals unchanged  │
│                                         │
│  5. Fill remaining slots:               │
│     a. Tournament select 2 parents      │
│     b. Uniform crossover → child        │
│     c. Gaussian mutation → mutated child│
│                                         │
│  6. Replace population                  │
└─────────────────────────────────────────┘
```

---

## API Reference

### `evaluate_agent(chromosome, ...) → float`

Run one episode with the chromosome loaded into all agents.
Returns the total reward summed across all agents.

| Parameter      | Type             | Default | Description                   |
| -------------- | ---------------- | ------- | ----------------------------- |
| `chromosome`   | `np.ndarray`     | —       | Flat float32 parameter vector |
| `obs_size`     | `int`            | 27      | Observation vector length     |
| `num_actions`  | `int`            | 5       | Number of discrete actions    |
| `hidden_sizes` | `list[int]`      | [16]    | MLP hidden layer widths       |
| `env_config`   | `Optional[dict]` | None    | Environment config overrides  |
| `seed`         | `Optional[int]`  | None    | Environment RNG seed          |

### `evaluate_robust(chromosome, num_episodes=3, ...) → float`

Average fitness over multiple episodes with different seeds.
Reduces noise from random environment layouts.

### `tournament_selection(population, fitnesses, k=3) → np.ndarray`

Pick `k` random individuals; return a **copy** of the fittest.

| Parameter    | Type                  | Default | Description         |
| ------------ | --------------------- | ------- | ------------------- |
| `population` | `list[ndarray]`       | —       | List of chromosomes |
| `fitnesses`  | `np.ndarray`          | —       | Fitness values      |
| `k`          | `int`                 | 3       | Tournament size     |
| `rng`        | `Optional[Generator]` | None    | NumPy RNG           |

### `uniform_crossover(parent_a, parent_b) → np.ndarray`

For each gene, randomly take from parent A or parent B (50/50).

### `gaussian_mutate(chromosome, std=0.02) → np.ndarray`

Add Gaussian noise N(0, σ²) to every gene. Returns a new array.

### `adaptive_mutation_std(generation, initial_std=0.02, decay=0.9, decay_every=20) → float`

Compute the mutation standard deviation for a given generation:

```
σ(gen) = initial_std × decay ^ (gen // decay_every)
```

| Gen   | σ (defaults) |
| ----- | ------------ |
| 0–19  | 0.0200       |
| 20–39 | 0.0180       |
| 40–59 | 0.0162       |
| 60–79 | 0.0146       |
| 80–99 | 0.0131       |

### `Population` class

Orchestrates the full GA loop.

```python
from src.evolution import Population

pop = Population(
    population_size=50,
    generations=100,
    tournament_k=3,
    mutation_std=0.02,
    elitism=2,
    eval_episodes=3,
    patience=30,
    output_dir="runs/exp_001",
    seed=42,
)
best_params, best_fitness = pop.run()
```

**Constructor parameters:**

| Parameter         | Default        | Description                             |
| ----------------- | -------------- | --------------------------------------- |
| `population_size` | 50             | Number of individuals                   |
| `generations`     | 100            | Max generations                         |
| `tournament_k`    | 3              | Tournament selection size               |
| `mutation_std`    | 0.02           | Initial Gaussian mutation σ             |
| `mutation_decay`  | 0.9            | σ decay factor                          |
| `decay_every`     | 20             | Apply decay every N gens                |
| `elitism`         | 2              | Elite individuals preserved             |
| `eval_episodes`   | 3              | Episodes per fitness evaluation         |
| `patience`        | 30             | Early stop after N gens without improve |
| `output_dir`      | `runs/default` | Directory for logs and checkpoints      |
| `save_every`      | 10             | Save best chromosome every N gens       |
| `seed`            | 42             | Master RNG seed                         |

---

## Output Files

After training, the output directory contains:

```
runs/exp_001/
├── config.json       Full experiment config (reproducibility)
├── metrics.csv       Per-generation stats
├── best_gen0000.npy  Checkpoint at gen 0
├── best_gen0010.npy  Checkpoint at gen 10
├── ...
└── best_final.npy    Overall best chromosome
```

**metrics.csv columns:**

| Column          | Description                          |
| --------------- | ------------------------------------ |
| `generation`    | Generation number                    |
| `best_fitness`  | Highest fitness in the generation    |
| `avg_fitness`   | Population average fitness           |
| `worst_fitness` | Lowest fitness in the generation     |
| `diversity`     | Mean pairwise Euclidean distance     |
| `mutation_std`  | Current mutation σ                   |
| `elapsed_sec`   | Wall-clock time since training start |

---

## Usage

### Training

```bash
# Quick test
python -m scripts.train --population 10 --generations 5

# Full run with defaults
python -m scripts.train

# Custom experiment
python -m scripts.train --population 50 --generations 100 \
    --mutation-std 0.02 --elitism 2 --output runs/exp_001
```

### Replay

```bash
python -m scripts.replay --checkpoint runs/exp_001/best_final.npy
```

---

## Design Decisions

| Decision       | Choice                  | Rationale                                       |
| -------------- | ----------------------- | ----------------------------------------------- |
| Selection      | Tournament (k=3)        | Simple, tuneable pressure                       |
| Crossover      | Uniform                 | No spatial structure in weight vectors          |
| Mutation       | Gaussian (σ=0.02)       | Smooth perturbations, adaptive decay            |
| Elitism        | Top 2                   | Prevents losing the best solution               |
| Fitness        | Total reward, avg 3 eps | Robust against stochastic env layouts           |
| Evaluation     | Sequential              | Simple first; parallelise in a future iteration |
| Early stopping | 30-gen plateau          | Saves compute when stuck                        |

---

## Hyperparameter Guide

| Parameter     | Too Low              | Good Range | Too High                    |
| ------------- | -------------------- | ---------- | --------------------------- |
| Population    | <10 → poor diversity | 30–100     | >200 → slow evaluation      |
| Tournament k  | 1 → random selection | 2–5        | >10 → premature convergence |
| Mutation σ    | <0.001 → stuck       | 0.01–0.05  | >0.1 → destructive noise    |
| Elitism       | 0 → can lose best    | 1–3        | >5 → reduces exploration    |
| Eval episodes | 1 → noisy fitness    | 2–5        | >10 → slow                  |

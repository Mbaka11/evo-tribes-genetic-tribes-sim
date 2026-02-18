# Changelog

All notable changes to EvoTribes are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [0.3.0] — 2026-02-15

### Added

- Genetic algorithm package (`src/evolution/`)
  - `fitness.py` — `evaluate_agent()` and `evaluate_robust()` for headless episode evaluation
  - `selection.py` — `tournament_selection()` with configurable tournament size k
  - `crossover.py` — `uniform_crossover()` for gene-level parent mixing
  - `mutation.py` — `gaussian_mutate()` with `adaptive_mutation_std()` step-decay schedule
  - `population.py` — `Population` class orchestrating the full GA loop with logging
- Training script (`scripts/train.py`) — CLI with argparse for all GA hyperparameters
- Replay script (`scripts/replay.py`) — load saved .npy chromosomes and render via Pygame
- Evolution tests (`tests/test_evolution.py`) — 40+ tests across 10 test classes
- Iteration 03 notes (`docs/notes/iteration_03.md`)

### Changed

- Rewrote `docs/03_genetic_algorithm.md` — full API reference replacing placeholder

---

## [0.2.0] — 2026-02-15

### Added

- Policy interface (`src/policies/base_policy.py`)
  - Abstract base class with `select_action`, `get_params`/`set_params`, `save`/`load`
- RandomPolicy (`src/policies/random_policy.py`)
  - Baseline random action selection, 0 tuneable parameters
- MLPPolicy (`src/policies/mlp_policy.py`)
  - Pure NumPy fully-connected neural network
  - Xavier uniform weight initialisation, ReLU activations, softmax output
  - Flat parameter get/set for genetic algorithm compatibility (533 params default)
- Policy tests (`tests/test_policies.py`) — 25+ tests across 6 test classes
- Iteration 02 notes (`docs/notes/iteration_02.md`)

### Changed

- Demo script now supports `--policy random|mlp` and `--stochastic` flags
- Rewrote `docs/02_policies.md` with full interface spec and worked examples

---

## [0.1.0] — 2026-02-14

### Added

- Grid-based Gymnasium environment (`src/envs/tribes_env.py`)
  - Configurable grid size, agent count, tribe count
  - Food spawning and respawn mechanics
  - Energy system with per-step drain
  - Collision detection with penalties
- Pygame renderer (`src/envs/rendering.py`)
  - Grid, food, tribe-coloured agents, energy bars, overlay text
  - Rendering speed: 4 FPS for easier visual tracking
- Demo script (`scripts/demo.py`) — random agents with real-time rendering
- Smoke tests (`tests/test_env_smoke.py`) — 12 tests covering reset, step, termination
- Environment documentation (`docs/01_environment.md`)
- Iteration notes system (`docs/notes/`)
- Project documentation (`README.md`, `CONTRIBUTING.md`, `LICENSE`)
- Dependency management (`requirements.txt`, `.gitignore`)
- CI/CD pipeline (`.github/workflows/ci.yml`) with automated version tagging

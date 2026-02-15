# Changelog

All notable changes to EvoTribes are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/).

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
- Demo script (`scripts/demo.py`) — random agents with real-time rendering
- Smoke tests (`tests/test_env_smoke.py`) — 12 tests covering reset, step, termination
- Environment documentation (`docs/01_environment.md`)
- Iteration notes system (`docs/notes/`)

<div align="center">

  <img src="assets/images/image.png" alt="EvoTribes Banner" style="border: 4px solid white; border-radius: 8px;" width="720" />

</div>

<br/>

# EvoTribes

**A modular, learning-first genetic multi-agent simulation built with [Gymnasium](https://gymnasium.farama.org/) to study emergent group behaviour and alignment failures.**

[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FMbaka11%2Fevo-tribes-genetic-tribes-sim%2Fmain%2FVERSION&query=%24.version&label=version)](VERSION)
[![Tests](https://github.com/Mbaka11/evo-tribes-genetic-tribes-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/Mbaka11/evo-tribes-genetic-tribes-sim/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## What is EvoTribes?

EvoTribes drops multiple AI agents onto a shared 2D grid.
Agents belong to **tribes**, compete for **food**, spend **energy** to stay alive, and — in later iterations — **evolve** through a genetic algorithm.

The goal is **not** to build the fastest simulation.
It is to **understand** every component: how observations flow, how rewards shape behaviour, and how evolution pressures produce emergent strategies.

Every change is documented, every formula is explained, and every run is reproducible.

---

## Current Version

| Iteration | Version | Summary                                                  |
| --------- | ------- | -------------------------------------------------------- |
| 1         | 0.1.0   | Grid world, random agents, Pygame rendering, smoke tests |

> Detailed notes for each iteration live in [`docs/notes/`](docs/notes/).

---

## Quick Start

### 1. Install dependencies

```bash
pip install gymnasium numpy pygame pytest
```

### 2. Run the demo (random agents, Pygame window)

```bash
python -m scripts.demo
```

### 3. Run tests

```bash
python -m pytest tests/test_env_smoke.py -v
```

---

## What You Should See

A Pygame window showing:

- A dark **20×20 grid**
- **Green dots** = food
- **Blue circles** = Tribe 0 agents, **Red circles** = Tribe 1 agents
- Tiny **energy bars** above each agent
- **Overlay** at the bottom: step count, alive agents, agent 0 energy
- Agents wander randomly, energy drains, agents die or the episode ends at step 300

---

## Project Structure

```
EvoTribes/
├── assets/images/        # project images
├── docs/
│   ├── notes/            # detailed iteration notes (start here!)
│   ├── 00_overview.md    # system goals & architecture
│   ├── 01_environment.md # env spec (obs, actions, rewards)
│   ├── 02_policies.md    # agent brains
│   ├── 03_genetic_algorithm.md
│   ├── 04_scenarios.md
│   ├── 05_metrics.md
│   └── 06_alignment_cases.md
├── scripts/
│   └── demo.py           # thin entry points
├── src/
│   └── envs/
│       ├── tribes_env.py # Gymnasium environment
│       └── rendering.py  # Pygame renderer
├── tests/
│   └── test_env_smoke.py
├── VERSION               # current version
└── README.md
```

---

## Iteration Roadmap

| Iteration | Goal                                                    | Status  |
| --------- | ------------------------------------------------------- | ------- |
| 1         | Grid environment, random agents, rendering, tests       | Done    |
| 2         | Policy interface, random policy, simple MLP policy      | Planned |
| 3         | Genetic algorithm — evaluate, select, crossover, mutate | Planned |
| 4         | Metrics logging, run tracking, reproducibility          | Planned |
| 5         | Scenarios and parameter sweeps                          | Planned |
| 6         | Alignment case studies — reward hacking experiments     | Planned |

---

## Philosophy

- **Learn, don't speed-run.** Every component must be understood before moving on.
- **Traceability.** Every change is documented in iteration notes.
- **Reproducibility.** Seeds, configs, and logged metrics make every run repeatable.
- **Modularity.** Swap policies, change rewards, resize the grid — nothing else breaks.

---

## Contributing

This is a personal learning project. If you'd like to follow along or suggest improvements, open an issue or PR on GitHub.

---

## License

MIT

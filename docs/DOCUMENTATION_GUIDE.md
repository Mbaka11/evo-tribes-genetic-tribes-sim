# Documentation Structure

EvoTribes has **three types of documentation**, each serving a different purpose:

```
docs/
├── 00-06_*.md          → What the system does (architecture)
├── notes/              → What changed (iteration history)
│   ├── TEMPLATE.md
│   ├── iteration_01.md
│   ├── iteration_02.md
│   └── ...
└── learning/           → How concepts work (education)
    ├── README.md
    ├── genetic_algorithms.md
    └── ...
```

---

## 1. Main Docs (`docs/00_*.md` through `06_*.md`)

**Purpose:** Define the system architecture and specifications

**Audience:** Developers working on the project

**Content:**

- System design decisions
- API specifications
- Configuration options
- Integration points

**Examples:**

- [00_overview.md](00_overview.md) — project goals, high-level architecture
- [01_environment.md](01_environment.md) — Gymnasium interface, observation/action spaces, rewards
- [02_policies.md](02_policies.md) — policy interface, available implementations
- [03_genetic_algorithm.md](03_genetic_algorithm.md) — evolution loop, parameters

**When to update:** When the system's API or architecture changes

**Tone:** Technical reference (assumes reader knows the domain)

---

## 2. Iteration Notes (`docs/notes/iteration_*.md`)

**Purpose:** Record what was built in each iteration and why

**Audience:** Future maintainers, project reviewers, users wanting history

**Content:**

- What was implemented (file-by-file)
- Architecture & design decisions
- Algorithms with step-by-step explanations
- Concrete worked examples
- Test coverage
- Known limitations
- Version history

**Examples:**

- [iteration_01.md](notes/iteration_01.md) — Grid world environment, random agents
- [iteration_02.md](notes/iteration_02.md) — Policy interface, RandomPolicy, MLPPolicy

**When to create:** After completing ANY iteration (mandatory)

**Tone:** Detailed narrative explaining decisions and showing examples

**Template:** [TEMPLATE.md](notes/TEMPLATE.md) (17 sections)

---

## 3. Learning Resources (`docs/learning/*.md`)

**Purpose:** Teach technical concepts from first principles

**Audience:** Anyone unfamiliar with the concept (beginners welcome)

**Content:**

- What is this concept?
- Why does it exist? What problem does it solve?
- How does it work? (algorithm, formula, code)
- Worked example with specific numbers
- Common pitfalls and fixes
- References for further reading

**Examples:**

- [genetic_algorithms.md](learning/genetic_algorithms.md) — Complete GA tutorial (selection, crossover, mutation, elitism)
- _[neural_networks.md](learning/neural_networks.md)_ (future) — MLPs, activations, weight init
- _[reinforcement_learning_basics.md](learning/reinforcement_learning_basics.md)_ (future) — RL fundamentals

**When to create:** When introducing ANY complex technical concept that requires specialized knowledge

**Tone:** Educational (assumes zero prior knowledge)

---

## Relationship Between Doc Types

```
         ┌──────────────┐
         │ Main Docs    │ ← "The environment has 5 actions"
         └──────┬───────┘
                │ references
                ▼
         ┌──────────────┐
         │ Iteration    │ ← "We built the environment with
         │ Notes        │    5 actions because..."
         └──────┬───────┘    + worked example of one step
                │ references
                ▼
         ┌──────────────┐
         │ Learning     │ ← "What is a Markov Decision Process?
         │ Resources    │    How do actions work in RL?"
         └──────────────┘    + teaches the underlying concept
```

**Example flow:**

1. **Learning doc** teaches "What is a genetic algorithm?" (theory)
2. **Iteration note** explains "We implemented tournament selection with k=3" (what we built)
3. **Main doc** specifies "Tournament selection API: `tournament_selection(population, fitness, k)`" (reference)

---

## Which Doc to Read?

| Your Goal                                  | Read This                                               |
| ------------------------------------------ | ------------------------------------------------------- |
| Understand a concept (GA, RL, neural nets) | **Learning docs**                                       |
| See how EvoTribes evolved over time        | **Iteration notes** (in order)                          |
| Look up API or config options              | **Main docs**                                           |
| Debug an issue from a past iteration       | **Iteration notes** + **Main docs**                     |
| Learn before implementing a new feature    | **Learning docs** → **Iteration notes** → **Main docs** |

---

## Writing Guidelines Summary

### Main Docs

- Concise technical reference
- Focus on "what" and "how to use"
- API signatures, config parameters, examples

### Iteration Notes

- Narrative explaining "what changed" and "why"
- Complete file list
- Worked examples for every feature
- Use [TEMPLATE.md](notes/TEMPLATE.md)

### Learning Docs

- Teach from zero knowledge
- Heavy on intuition, examples, diagrams
- Every formula explained in plain language
- Worked examples with specific numbers
- See [learning/README.md](learning/README.md) for full guidelines

---

**Last Updated:** 2026-02-15

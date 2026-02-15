# Contributing to EvoTribes

Thank you for your interest in EvoTribes!

---

## Philosophy

This is a **learning-first** project. The goal is deep understanding,
not fast feature delivery.

Code must be:

- **Simple** — readable by someone new to the codebase
- **Modular** — swappable components that don't break other systems
- **Documented** — every iteration has detailed notes explaining _why_
- **Reproducible** — seeds, configs, and tests ensure consistency

---

## How to Contribute

### 1. Open an issue first

Before submitting code, open an issue describing:

- What feature or fix you want to contribute
- Why it fits the project goals
- How it aligns with the current iteration roadmap

This prevents wasted effort on changes that may not merge.

### 2. Follow the iteration workflow

See [`.copilot-instructions.md`](.copilot-instructions.md) for the full workflow.

Every change should:

1. Explain **why** it exists
2. Be small and runnable
3. Include tests
4. Update documentation
5. Add an entry to [`CHANGELOG.md`](CHANGELOG.md)

### 3. Write iteration notes

If you're implementing a major feature, add a detailed note in `docs/notes/`.

Use the template in [`docs/notes/TEMPLATE.md`](docs/notes/TEMPLATE.md).

Iteration notes explain:

- Goal and rationale
- Architecture & design decisions
- Mathematical formulas (with plain-language explanations)
- Algorithms (step-by-step)
- Key concepts for beginners
- **Concrete examples** (worked scenarios with real values)
- How to run, what to expect, known limitations

**Critical:** Always include examples. For every technical concept, formula,
or algorithm, provide at least one worked example showing:

- Input values
- Step-by-step execution
- Output
- What it means in plain language

Examples make abstract concepts tangible and help spot bugs early.

### 4. Maintain test coverage

Every new component must have at least one test.

Run the test suite before submitting:

```bash
python -m pytest tests/ -v
```

### 5. Keep code configurable

No magic numbers. All parameters live in:

- Config dicts (environment, policies)
- Scenario files (Iteration 4+)
- Command-line arguments (scripts)

---

## Code Style

- **PEP 8** compliance (with 88-char line limit like Black)
- Type hints on function signatures
- Docstrings for modules, classes, and public functions
- Comments explain _why_, not _what_
- **Include examples in docstrings** for complex functions

Example docstring format:

```python
def compute_reward(food: bool, collision: bool) -> float:
    """Calculate reward for one agent in one step.

    Args:
        food: Whether the agent ate food this step
        collision: Whether the agent collided with another agent

    Returns:
        Total reward (sum of food, collision, survival components)

    Example:
        >>> compute_reward(food=True, collision=False)
        1.01  # food_reward (1.0) + survival_bonus (0.01)
        >>> compute_reward(food=False, collision=True)
        -0.09  # collision_penalty (-0.1) + survival_bonus (0.01)
    """
```

---

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit with clear messages: `git commit -m "Add X to support Y"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a PR against `main`

**PR description should include:**

- What changes were made
- Why they were made
- Which files were modified
- How to test the changes
- Any breaking changes or migration steps

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0) — breaking changes
- **MINOR** (0.X.0) — new features, backward-compatible
- **PATCH** (0.0.X) — bug fixes

To bump the version:

1. Update [`VERSION`](VERSION)
2. Add an entry to [`CHANGELOG.md`](CHANGELOG.md)
3. Commit and push to `main`

The GitHub Actions workflow will auto-tag the release.

---

## Questions?

Open an issue labeled `question` or reach out on GitHub Discussions.

---

## Code of Conduct

Be respectful, constructive, and collaborative.

This is a learning space — questions are encouraged, and mistakes are
part of the process.

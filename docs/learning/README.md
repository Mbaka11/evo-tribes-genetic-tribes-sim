# Learning Resources

This folder contains **educational documents** designed to teach technical concepts used in EvoTribes from first principles.

Each document assumes minimal prior knowledge and includes:

- Intuitive explanations
- Mathematical formulas with plain-language interpretations
- Concrete worked examples with specific numbers
- Code snippets showing implementation patterns
- Common pitfalls and how to avoid them

---

## Philosophy

EvoTribes is a **learning-first project**. Code alone doesn't teach — you need context, reasoning, and examples. These documents bridge the gap between "I see the code" and "I understand why it works."

---

## Available Guides

### [genetic_algorithms.md](genetic_algorithms.md)

**Topics:** Evolution, selection, crossover, mutation, elitism, fitness functions  
**Prerequisites:** None  
**When to read:** Before Iteration 3 (genetic algorithm implementation)

Complete guide to genetic algorithms with:

- Biological analogy (DNA → neural weights)
- Tournament vs rank-based selection (with worked examples)
- Crossover types (uniform, single-point, arithmetic)
- Mutation strategies (Gaussian noise, adaptive decay)
- Full walkthrough of one generation (8 agents, step-by-step)
- Hyperparameter tuning guide
- Common failure modes (premature convergence, population collapse)

### [neural_networks.md](neural_networks.md) _(coming soon)_

**Topics:** Perceptrons, MLPs, activation functions, forward pass, weight initialization  
**Prerequisites:** Basic linear algebra (dot products, matrices)  
**When to read:** Before or during Iteration 2 (policy implementation)

Will cover:

- What is a neural network (biological neuron vs artificial neuron)
- Forward propagation math (step-by-step with 3 neurons)
- Activation functions (ReLU, sigmoid, softmax) — why each exists
- Xavier initialization (preventing exploding/vanishing signals)
- Why MLPs without backprop (genetic algorithm trains weights instead)

### [reinforcement_learning_basics.md](reinforcement_learning_basics.md) _(coming soon)_

**Topics:** MDPs, agents, environments, observations, actions, rewards, episodes  
**Prerequisites:** None  
**When to read:** Before Iteration 1 (environment implementation)

Will cover:

- What is RL (vs supervised learning, vs unsupervised learning)
- The agent-environment loop (observation → action → reward)
- Markov Decision Processes (states, transitions, policies)
- Reward shaping (what makes a good reward signal)
- Gymnasium interface (`reset()`, `step()`, `render()`)
- Multi-agent RL (single env with multiple agents)

### [evolutionary_computation.md](evolutionary_computation.md) _(coming soon)_

**Topics:** Fitness landscapes, search spaces, exploration vs exploitation  
**Prerequisites:** genetic_algorithms.md  
**When to read:** After Iteration 3, before Iteration 4 (advanced evolution)

Will cover:

- Fitness landscape visualization (peaks, valleys, plateaus)
- Local vs global optima (why GAs can escape local optima)
- Exploration-exploitation tradeoff
- Diversity preservation strategies
- Co-evolution (arms race dynamics)
- Speciation and niching

---

## How to Use These Docs

### For New Concepts

1. **Before implementing:** Read the relevant learning doc
2. **During implementation:** Reference specific sections (formulas, algorithms)
3. **After implementing:** Revisit examples to verify your implementation matches

### For Debugging

If something isn't working:

1. Check the "Common Pitfalls" section in the relevant doc
2. Verify your implementation against the worked examples
3. Compare your hyperparameters to the recommended ranges

### For Teaching Others

These docs are designed to be shared. If someone asks "What's a genetic algorithm?", send them [genetic_algorithms.md](genetic_algorithms.md).

---

## Writing Guidelines (For Contributors)

When adding a new learning document:

### 1. Assume Zero Prior Knowledge

Don't say "as we know from calculus..." — explain or provide a reference.

### 2. Structure

- **What:** Define the concept in one sentence
- **Why:** Why does this concept exist? What problem does it solve?
- **How:** Show the implementation (algorithm, formula, code)
- **Example:** Work through a complete example with real numbers
- **Pitfalls:** What goes wrong and how to fix it

### 3. Examples Are Mandatory

Every technical concept needs at least one worked example showing:

- **Input:** Specific values, not abstract variables
- **Process:** Step-by-step computation (show intermediate results)
- **Output:** Final result
- **Interpretation:** What does this result mean in context?

### 4. Formulas

Every mathematical formula should have:

```markdown
$$\text{formula here}$$

Where:

- $x$ = explanation with units/context
- $y$ = explanation with units/context

**Intuition:** Plain-language explanation of what this formula does.

**Example:** [work through with specific numbers]
```

### 5. Code Snippets

- Keep them short (< 20 lines)
- Include comments explaining non-obvious parts
- Show expected output as comments

### 6. Diagrams

Use ASCII art for simple diagrams:

```
Input → [Process] → Output
```

Or reference Mermaid diagrams in the main docs.

---

## Document Lifecycle

1. **Draft:** Created when a new technical concept is introduced
2. **Review:** Read by someone unfamiliar with the concept (test comprehension)
3. **Finalized:** Added to this README and referenced in iteration notes
4. **Updated:** When implementation changes (e.g., hyperparameters tuned)

---

## Related Documentation

- **Main docs** (`docs/00_*.md` through `docs/06_*.md`) — architectural specs, loaded on-demand
- **Iteration notes** (`docs/notes/iteration_*.md`) — what was built, why, and how it works
- **Learning resources** (`docs/learning/*.md`) — **YOU ARE HERE** — educational deep-dives

**Relationship:**

- Main docs = "What the system does"
- Iteration notes = "What changed and why"
- Learning docs = "How the concepts work"

---

## Contribution Ideas

Want to add a learning doc? Consider these topics:

- **Grid-based environments** (tile maps, spatial indexing, collision detection)
- **Observation space design** (what to show agents, local vs global views)
- **Reward engineering** (shaping, dense vs sparse, pitfalls)
- **Xavier vs He initialization** (when to use each)
- **ReLU vs sigmoid vs tanh** (activation function comparison)
- **Tournament selection variations** (Boltzmann, rank-weighted)
- **Crossover operator comparison** (empirical results on different problems)
- **Mutation schedules** (linear decay, exponential, adaptive)
- **Elitism strategies** (age-based, fitness-based, diversity-preserving)
- **Multi-agent coordination** (implicit vs explicit, emergent behavior)

---

**Last Updated:** 2026-02-15  
**Maintainer:** EvoTribes Development Team

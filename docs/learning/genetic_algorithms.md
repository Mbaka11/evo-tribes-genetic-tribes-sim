# Genetic Algorithm — Complete Guide

> **Note:** This document is written for someone **new to genetic algorithms**. Every concept is explained from scratch with concrete examples, formulas, and intuition.

---

## Table of Contents

1. [What is a Genetic Algorithm?](#what-is-a-genetic-algorithm)
2. [The Biological Analogy](#the-biological-analogy)
3. [The Core Loop](#the-core-loop)
4. [Chromosome Representation](#chromosome-representation)
5. [Fitness Function](#fitness-function)
6. [Selection Methods](#selection-methods)
7. [Crossover (Recombination)](#crossover-recombination)
8. [Mutation](#mutation)
9. [Elitism](#elitism)
10. [Worked Example — Complete Generation](#worked-example--complete-generation)
11. [Hyperparameters & Tuning](#hyperparameters--tuning)
12. [Common Pitfalls](#common-pitfalls)
13. [Why This Works](#why-this-works)
14. [EvoTribes Implementation Decisions](#evotribes-implementation-decisions)

---

## What is a Genetic Algorithm?

A **genetic algorithm (GA)** is an optimization technique inspired by natural evolution. Instead of calculating gradients or testing every possibility, a GA:

1. Starts with a **population** of random solutions
2. **Evaluates** how good each solution is (fitness)
3. **Selects** the better solutions to "reproduce"
4. **Combines** selected solutions to create offspring (crossover)
5. Randomly **tweaks** offspring (mutation)
6. **Repeats** for many generations

Over time, the population evolves better and better solutions.

### When to Use GAs

✅ **Good for:**
- Problems where you can't compute gradients (non-differentiable)
- Black-box optimization (you can test solutions but don't know the function)
- Discrete or combinatorial problems
- Exploring large search spaces with many local optima

❌ **Not good for:**
- Problems with obvious gradients (use gradient descent instead)
- Very high-dimensional spaces (curse of dimensionality)
- When you need exact solutions (GAs are heuristic)

### EvoTribes Context

We're using a GA because:
- We want agents to **discover strategies** through evolution, not supervised learning
- There's no "correct answer" to learn from (no labeled data)
- The fitness landscape is complex (reward depends on environment dynamics)
- It's biologically plausible (tribes evolve, not train)

---

## The Biological Analogy

| Biology | Genetic Algorithm | EvoTribes |
|---------|-------------------|-----------|
| **Organism** | Solution / Individual | One agent with a neural network brain |
| **Chromosome** | Parameter vector | All 533 neural network weights as a flat array |
| **Gene** | Single parameter | One weight or bias value |
| **Population** | Set of solutions | 50 agents competing for survival |
| **Fitness** | Survival/reproduction success | Total reward from one episode |
| **Generation** | One reproduction cycle | Evaluate all 50 agents, breed, replace |
| **Natural selection** | Picking high-fitness parents | Tournament or rank-based selection |
| **Sexual reproduction** | Crossover | Blend two parent weight vectors |
| **Random mutation** | Gaussian noise | Add small random values to weights |
| **Evolution** | Iterative improvement | Better strategies emerge over 100+ generations |

### Key Insight

In nature, organisms don't "optimize" themselves — they reproduce, and the **next generation** is slightly different. Bad traits fade out, good traits spread.

A GA mimics this: we never directly improve an agent. We just let good agents have more children.

---

## The Core Loop

```
Generation 0:
  ┌─────────────────────────────────────┐
  │ Create 50 random agents             │
  └─────────────────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────┐
  │ EVALUATE: Run each agent, get       │
  │ fitness (total reward)              │
  └─────────────────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────┐
  │ SELECT: Pick parents based on       │
  │ fitness (better agents more likely) │
  └─────────────────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────┐
  │ CROSSOVER: Blend parent chromosomes │
  │ to create 48 offspring              │
  └─────────────────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────┐
  │ MUTATE: Add random noise to         │
  │ offspring chromosomes               │
  └─────────────────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────┐
  │ ELITISM: Keep top 2 parents         │
  │ unchanged (2 elite + 48 offspring)  │
  └─────────────────────────────────────┘
                  │
                  ▼
Generation 1: Repeat with new population
```

### Pseudocode

```python
population = create_random_agents(50)

for generation in range(100):
    # 1. Evaluate
    fitness = [evaluate(agent) for agent in population]
    
    # 2. Select parents
    parents = select_parents(population, fitness)
    
    # 3. Create offspring
    offspring = []
    for _ in range(48):  # 50 - 2 elites
        parent_a, parent_b = pick_two_random(parents)
        child = crossover(parent_a, parent_b)
        child = mutate(child)
        offspring.append(child)
    
    # 4. Elitism — keep best 2
    elites = get_top_k(population, fitness, k=2)
    
    # 5. New generation
    population = elites + offspring
```

---

## Chromosome Representation

In EvoTribes, an agent's "chromosome" is its **neural network weights**:

```python
policy = MLPPolicy(obs_size=27, num_actions=5, hidden_sizes=[16])
chromosome = policy.get_params()  # shape: (533,) float32 array
```

### Structure

```
chromosome = [w00, w01, w02, ..., w_432, b0, b1, ..., b15, w_433, ..., w_512, b16, ..., b20]
             └────────────────────┬──────────────────┘ └─────┬──────┘ └────────┬─────────┘ └───┬───┘
                    W0 (27×16=432)                        b0 (16)       W1 (16×5=80)      b1 (5)
                                                                                          
Total: 533 genes (parameters)
```

Each gene is a `float32` typically in `[-0.5, +0.5]` range after Xavier init.

### Why This Representation?

- **Simple:** One flat array, easy to manipulate
- **GA-friendly:** Crossover and mutation are just array operations
- **Complete:** The entire agent behavior is encoded in these 533 numbers

---

## Fitness Function

**Fitness** measures how good a solution is. Higher fitness = better solution.

### EvoTribes Fitness: Total Reward

```python
def evaluate(agent: MLPPolicy) -> float:
    """Run the agent for one episode, return total reward."""
    env = TribesEnv(config=EVAL_CONFIG)
    obs, _ = env.reset()
    
    total_reward = 0.0
    done = False
    
    while not done:
        actions = [agent.select_action(obs[i]) for i in range(num_agents)]
        obs, rewards, terminated, truncated, _ = env.step(actions)
        total_reward += sum(rewards)  # sum of all agents' rewards
        done = terminated or truncated
    
    env.close()
    return total_reward
```

### Reward Components (from Iteration 1)

| Event | Reward | Frequency |
|-------|--------|-----------|
| Eat food | +1.0 | Rare (requires reaching food) |
| Survive one step | +0.01 | Every step (encourages longevity) |
| Collision | -0.1 | Common if agents cluster |

**Example fitness calculation:**

```
Agent runs for 120 steps:
- Eats 8 food              → +8.0
- Survives 120 steps       → +1.2   (120 × 0.01)
- Hits 5 walls/agents      → -0.5   (5 × -0.1)
─────────────────────────────────
Total fitness = 8.7
```

A random agent typically scores **2.0 - 4.0** (few food, short survival).

An evolved agent might score **15.0 - 25.0** (efficient food-seeking, long survival).

### Multi-Episode Evaluation (Reduces Variance)

```python
def evaluate_robust(agent: MLPPolicy, num_episodes: int = 3) -> float:
    """Average fitness over multiple episodes."""
    scores = [evaluate(agent) for _ in range(num_episodes)]
    return np.mean(scores)
```

**Why average?** Food spawns are random. One lucky episode doesn't mean the agent is good. Averaging 3 episodes gives a more reliable fitness estimate.

---

## Selection Methods

**Selection** picks which agents get to reproduce. Better agents should be picked more often.

### 1. Tournament Selection (Recommended)

**Idea:** Randomly pick K agents, take the best one. Repeat N times to get N parents.

**Algorithm:**

```python
def tournament_selection(population, fitness, k=3, num_parents=50):
    """Select parents via tournament."""
    parents = []
    for _ in range(num_parents):
        # Pick k random agents
        indices = np.random.choice(len(population), size=k, replace=False)
        # Take the one with highest fitness
        winner_idx = indices[np.argmax([fitness[i] for i in indices])]
        parents.append(population[winner_idx])
    return parents
```

**Example with k=3:**

```
Population: [A, B, C, D, E, F, G, H]
Fitness:    [5, 2, 8, 1, 6, 3, 9, 4]

Tournament 1:
  - Pick random 3: [C, F, H]
  - Fitness: [8, 3, 4]
  - Winner: C (fitness 8) → parent 1

Tournament 2:
  - Pick random 3: [A, D, G]
  - Fitness: [5, 1, 9]
  - Winner: G (fitness 9) → parent 2

... repeat 48 more times
```

**Properties:**

- **Selection pressure:** Larger k → stronger pressure (only very best reproduce)
- **Diversity:** Smaller k → weaker pressure (average agents sometimes win)
- **Simplicity:** No sorting, no normalization needed
- **Typical k:** 2-5 (we use 3)

---

### 2. Rank-Based Selection

**Idea:** Sort agents by fitness, assign selection probability based on rank (not raw fitness).

**Algorithm:**

```python
def rank_selection(population, fitness):
    """Select parents based on rank."""
    # Sort by fitness
    sorted_indices = np.argsort(fitness)  # lowest to highest
    
    # Assign probabilities: rank 1 → prob 1, rank 2 → prob 2, ...
    ranks = np.arange(1, len(population) + 1)
    probs = ranks / ranks.sum()  # normalize
    
    # Sample parents
    parent_indices = np.random.choice(
        sorted_indices, size=50, replace=True, p=probs
    )
    return [population[i] for i in parent_indices]
```

**Example:**

```
Agents:    [A,   B,   C,   D,   E]
Fitness:   [1.2, 5.8, 3.1, 10.4, 2.0]
Sorted:    [A,   E,   C,   B,   D]   (by fitness)
Ranks:     [1,   2,   3,   4,   5]
Probs:     [0.07, 0.13, 0.20, 0.27, 0.33]

D has 33% chance of being selected (highest fitness)
A has 7% chance (lowest fitness)
```

**Property:** Prevents one "super fit" agent from dominating if fitness values have huge range (e.g., 100 vs 5).

---

### 3. Roulette Wheel Selection (Not Recommended Here)

**Idea:** Selection probability ∝ raw fitness.

**Problem:** If one agent has fitness 100 and others have fitness 5, the strong agent is selected 95%+ of the time → population collapses to clones.

**We don't use this.**

---

### Comparison

| Method | Pros | Cons | EvoTribes Choice |
|--------|------|------|------------------|
| **Tournament** | Simple, robust, tunable | Requires picking k | ✅ **Use this (k=3)** |
| **Rank-based** | Immune to fitness scaling | Requires sorting | Good alternative |
| **Roulette** | Intuitive | Unstable with outliers | ❌ Avoid |

---

## Crossover (Recombination)

**Crossover** combines two parent chromosomes to create a child.

### 1. Uniform Crossover (Recommended)

**Idea:** For each gene, flip a coin. Heads → take from parent A, tails → take from parent B.

**Algorithm:**

```python
def uniform_crossover(parent_a, parent_b):
    """Each gene has 50% chance from either parent."""
    mask = np.random.rand(len(parent_a)) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return child
```

**Example:**

```
Parent A: [0.2,  0.5, -0.1,  0.8,  0.3, -0.4]
Parent B: [0.1, -0.3,  0.6, -0.2,  0.7,  0.5]
Mask:     [ T,    F,    T,    F,    T,    F ]  (random)
          
Child:    [0.2, -0.3, -0.1, -0.2,  0.3,  0.5]
          └A   └B    └A    └B    └A    └B
```

**Why this works for neural networks:**

Neural network weights don't have "positional structure" like DNA codons. Gene 42 and gene 43 aren't necessarily "related". Uniform crossover maximizes genetic mixing.

---

### 2. Single-Point Crossover

**Idea:** Pick a random split point. Child gets genes 0..k from parent A, genes k+1..end from parent B.

**Algorithm:**

```python
def single_point_crossover(parent_a, parent_b):
    """Split at one random point."""
    k = np.random.randint(1, len(parent_a))
    child = np.concatenate([parent_a[:k], parent_b[k:]])
    return child
```

**Example:**

```
Parent A: [0.2,  0.5, -0.1,  0.8,  0.3, -0.4]
Parent B: [0.1, -0.3,  0.6, -0.2,  0.7,  0.5]
Split at position 3:
                     ↓
Child:    [0.2,  0.5, -0.1 | -0.2,  0.7,  0.5]
          └─────A──────┘    └──────B──────┘
```

**Property:** Preserves "gene blocks" — useful if nearby genes are related (not true for neural nets).

---

### 3. Arithmetic Crossover

**Idea:** Average the two parents, optionally weighted.

**Algorithm:**

```python
def arithmetic_crossover(parent_a, parent_b, alpha=0.5):
    """Weighted average: child = alpha*A + (1-alpha)*B"""
    child = alpha * parent_a + (1 - alpha) * parent_b
    return child
```

**Example:**

```
Parent A: [0.2,  0.5, -0.1]
Parent B: [0.6, -0.3,  0.3]
alpha = 0.5

Child:    [0.4,  0.1,  0.1]  (average)
```

**Property:** Always produces values between the parents (interpolation). Less exploratory than uniform crossover.

---

### Which to Use?

| Method | Exploration | Best For | EvoTribes Choice |
|--------|-------------|----------|------------------|
| **Uniform** | High | Unstructured problems (neural nets) | ✅ **Use this** |
| **Single-point** | Medium | Structured problems (TSP routes) | Not needed |
| **Arithmetic** | Low | Fine-tuning near optima | Maybe late-stage |

---

## Mutation

**Mutation** adds random noise to keep diversity and explore new areas.

### Gaussian Mutation (Recommended)

**Idea:** Add small random values from a normal distribution.

**Algorithm:**

```python
def mutate(chromosome, std=0.02):
    """Add Gaussian noise to each gene."""
    noise = np.random.normal(0, std, size=len(chromosome))
    mutated = chromosome + noise
    return mutated.astype(np.float32)
```

**Example:**

```
Original:  [0.25, -0.10,  0.45,  0.00]
Noise:     [0.01, -0.03,  0.00,  0.02]  (random, std=0.02)
Mutated:   [0.26, -0.13,  0.45,  0.02]
```

### The Math

$$\text{gene}_{\text{new}} = \text{gene}_{\text{old}} + \mathcal{N}(0, \sigma^2)$$

Where:
- $\mathcal{N}(0, \sigma^2)$ = normal distribution with mean 0, std deviation $\sigma$
- $\sigma$ = **mutation strength** (critical hyperparameter)

### Choosing Mutation Strength ($\sigma$)

**Context:** Xavier init produces weights in $[-0.37, +0.37]$ for our network.

| $\sigma$ | As % of Xavier range | Effect | When to Use |
|----------|----------------------|--------|-------------|
| 0.01 | ~3% | Tiny tweaks | Fine-tuning near optimum |
| 0.02 | ~5% | Moderate exploration | ✅ **Good default** |
| 0.05 | ~13% | Large jumps | Early generations, stuck populations |
| 0.10 | ~27% | Chaos | Almost like re-randomizing |

**Rule of thumb:** Start at 5% of the typical parameter range.

### Mutation Rate vs Mutation Strength

- **Mutation rate** = probability that a gene mutates (e.g., 100% = all genes, 5% = 5% of genes)
- **Mutation strength** = size of the change when it happens (std deviation)

In EvoTribes, we use:
- **Rate = 100%** (all genes get noise every generation)
- **Strength = 0.02** (small noise per gene)

This is equivalent to "always mutate, but gently."

### Adaptive Mutation

**Idea:** Reduce mutation over time as the population converges.

```python
def adaptive_mutation_std(generation, initial_std=0.02, decay=0.9, decay_every=20):
    """Decay mutation strength every N generations."""
    num_decays = generation // decay_every
    return initial_std * (decay ** num_decays)
```

**Example:**

```
Gen 0-19:   std = 0.020  (exploration)
Gen 20-39:  std = 0.018  (0.020 × 0.9)
Gen 40-59:  std = 0.016  (0.020 × 0.9²)
Gen 60-79:  std = 0.015  (0.020 × 0.9³)
...
Gen 100+:   std = 0.011  (fine-tuning)
```

**Benefits:** Early on, explore widely. Later, refine solutions.

---

## Elitism

**Elitism** forces the top K agents to survive unchanged to the next generation.

**Algorithm:**

```python
def apply_elitism(population, fitness, k=2):
    """Keep the k best agents."""
    top_k_indices = np.argsort(fitness)[-k:]  # k highest fitness
    elites = [population[i] for i in top_k_indices]
    return elites
```

**Example:**

```
Population:  [A,  B,  C,  D,  E,  F]
Fitness:     [3,  7,  5,  9,  2,  6]
k = 2

Top 2: D (fitness 9), B (fitness 7)
→ D and B copied to next generation unchanged
→ Remaining 4 slots filled by offspring
```

### Why Elitism?

**Without elitism:** The best solution might not reproduce (unlucky random selection), or its offspring might be worse due to crossover/mutation. The best solution can be **lost**.

**With elitism:** The best solution is **guaranteed** to survive. Performance never regresses.

**Cost:** Reduces diversity slightly (2 slots always occupied by previous-gen agents).

**Recommendation:** Use k=1 or k=2 (1-4% of population).

---

## Worked Example — Complete Generation

Let's trace through one full generation with a tiny population (8 agents, just for clarity).

### Setup

- Population size: 8
- Tournament size: 3
- Elites: 2
- Mutation std: 0.02

### Generation N

**Population:**

| Agent | Chromosome (only 4 genes for example) | Fitness |
|-------|---------------------------------------|---------|
| A | [0.2, -0.1, 0.3, 0.5] | 3.2 |
| B | [0.1, 0.4, -0.2, 0.0] | 7.8 |
| C | [0.5, 0.2, 0.1, -0.4] | 5.1 |
| D | [-0.3, 0.5, 0.4, 0.2] | 9.5 |
| E | [0.0, -0.5, 0.3, 0.1] | 2.0 |
| F | [0.4, 0.1, -0.1, 0.6] | 6.3 |
| G | [0.2, 0.3, 0.5, -0.2] | 4.7 |
| H | [-0.1, 0.2, 0.0, 0.4] | 8.1 |

---

### Step 1: Elitism

Keep top 2 agents (D: 9.5, H: 8.1).

```
Elites: D, H
Remaining slots to fill: 6
```

---

### Step 2: Selection (Tournament, k=3)

Generate 6 offspring, so we need 12 parent selections (2 per child).

**Tournament 1:**
- Pick random 3: [A, C, G]
- Fitness: [3.2, 5.1, 4.7]
- Winner: **C**

**Tournament 2:**
- Pick random 3: [B, E, F]
- Fitness: [7.8, 2.0, 6.3]
- Winner: **B**

**Tournament 3:**
- Pick random 3: [C, D, H]
- Fitness: [5.1, 9.5, 8.1]
- Winner: **D**

... (continue for tournaments 4-12)

**Parents selected:** C, B, D, H, F, B, D, C, A, H, D, B

---

### Step 3: Crossover (Uniform)

**Child 1** from parents C and B:

```
Parent C: [0.5,  0.2,  0.1, -0.4]
Parent B: [0.1,  0.4, -0.2,  0.0]
Mask:     [ T,    F,    T,    F ]  (random)

Child 1:  [0.5,  0.4,  0.1,  0.0]
```

**Child 2** from parents D and H:

```
Parent D: [-0.3,  0.5,  0.4,  0.2]
Parent H: [-0.1,  0.2,  0.0,  0.4]
Mask:     [  F,    T,    F,    T ]

Child 2:  [-0.1,  0.5,  0.0,  0.4]
```

... (continue for all 6 offspring)

---

### Step 4: Mutation

**Child 1** (before mutation): `[0.5, 0.4, 0.1, 0.0]`

Generate noise: `[0.01, -0.02, 0.00, 0.03]` (random, std=0.02)

**Child 1** (after mutation): `[0.51, 0.38, 0.10, 0.03]`

**Child 2** (before): `[-0.1, 0.5, 0.0, 0.4]`

Generate noise: `[-0.01, 0.02, -0.01, 0.00]`

**Child 2** (after): `[-0.11, 0.52, -0.01, 0.40]`

... (mutate all 6 offspring)

---

### Step 5: Form Generation N+1

```
Generation N+1:
  D (elite, unchanged)
  H (elite, unchanged)
  Child 1
  Child 2
  Child 3
  Child 4
  Child 5
  Child 6
```

---

### Step 6: Re-Evaluate

Run all 8 agents through the environment again. Get new fitness values.

**Key insight:** Children might be better than their parents (combination + mutation found improvement), or worse (unlucky mutation). But on average, the population improves because we selected from the better agents.

---

## Hyperparameters & Tuning

| Parameter | Symbol/Name | Typical Range | EvoTribes Default |
|-----------|-------------|---------------|-------------------|
| **Population size** | $N$ | 20-200 | 50 |
| **Generations** | $G$ | 50-500 | 100 |
| **Tournament size** | $k$ | 2-7 | 3 |
| **Elite count** | $e$ | 1-5 | 2 |
| **Mutation std** | $\sigma$ | 0.01-0.1 | 0.02 (adaptive) |
| **Eval episodes** | — | 1-5 | 3 |

### How to Tune

1. **Start with defaults** above.
2. **Too slow to converge?** Increase population size or mutation strength.
3. **Converges too fast to bad solution?** Reduce tournament size (k=2), increase mutation.
4. **Unstable fitness?** Increase eval episodes (reduces randomness).
5. **No improvement after 20 gens?** Population stuck — restart with higher mutation or larger population.

---

## Common Pitfalls

### 1. Premature Convergence

**Symptom:** After 10-20 generations, fitness stops improving. All agents nearly identical.

**Cause:** Selection pressure too high (tournament k too large) + mutation too weak.

**Fix:**
- Reduce tournament size (k=2 instead of k=5)
- Increase mutation strength (std=0.05)
- Add diversity injection (every 20 gens, replace 10% with random agents)

---

### 2. Population Collapse

**Symptom:** All agents become clones of one good agent. Zero diversity.

**Cause:** Elitism too high, or fitness scaling issues (roulette wheel with outlier).

**Fix:**
- Reduce elites (k=1 instead of k=5)
- Use tournament selection instead of roulette
- Monitor diversity: `std_dev(all_genes_across_population)` — should stay > 0.01

---

### 3. Noisy Fitness

**Symptom:** Best fitness jumps up and down wildly each generation.

**Cause:** Single-episode evaluation + randomness in environment (food spawns, collisions).

**Fix:**
- Evaluate over 3-5 episodes, average the fitness
- Use fixed seeds for evaluation (same initial conditions)

---

### 4. No Learning

**Symptom:** After 100 generations, evolved agents no better than random.

**Cause:** Fitness function doesn't reward desired behavior, or mutation is too strong.

**Fix:**
- Check that high-fitness agents actually behave better (manual inspection)
- Verify reward function aligns with goals
- Reduce mutation strength (std=0.01)

---

## Why This Works

### Intuition

Imagine searching for the highest point in a mountain range, but you're blindfolded:

- **Random search:** Walk randomly, remember best position. Slow.
- **Hill climbing:** Walk uphill from current position. Gets stuck at local peaks.
- **Genetic algorithm:** Have 50 people searching. Every hour, kill the 25 who are lowest, and the top 25 have kids (who start near their parents). Kids wander randomly (mutation). Over time, the population clusters around the highest peaks.

### Key Mechanisms

1. **Selection:** Filters out bad solutions.
2. **Crossover:** Combines partial solutions (if parent A is good at X, parent B is good at Y, child might be good at both).
3. **Mutation:** Explores nearby variations.
4. **Population:** Parallel search in many regions.

### The Search Space

For our 533-parameter neural network:

- Each parameter ∈ `[-1.0, +1.0]` (roughly)
- Search space size ≈ $2^{533}$ discrete values (if quantized) = **astronomically huge**
- Gradient descent would need differentiable fitness (we don't have that)
- Exhaustive search is impossible

**GA navigates this by:**
- Starting with 50 random guesses
- Each generation, focusing on "promising regions" (near high-fitness agents)
- Mutation ensures we don't get permanently stuck

---

## EvoTribes Implementation Decisions

### Finalized Choices

| Aspect | Decision | Reasoning |
|--------|----------|-----------|
| **Population** | 50 agents | Balances diversity with eval speed |
| **Selection** | Tournament (k=3) | Robust, tunable, prevents outlier domination |
| **Crossover** | Uniform | Neural nets have no positional gene structure |
| **Mutation** | Gaussian (std=0.02, adaptive decay) | Gentle exploration, refines over time |
| **Elitism** | Top 2 | Guarantees no regression, minimal diversity cost |
| **Fitness** | Total reward, averaged over 3 episodes | Aligns with env design, reduces randomness |
| **Generations** | 100 (early stop if plateau 30 gens) | Enough time to evolve, avoid wasted compute |

### What We're NOT Doing (For Now)

❌ **Island models** (multiple isolated populations) — Adds complexity, defer to later iteration  
❌ **Speciation** (dynamic niching) — NEAT-style, very complex  
❌ **Adaptive crossover rates** — Fixed uniform is sufficient  
❌ **Multi-objective optimization** (Pareto fronts) — Single fitness for now  
❌ **Coevolution** (agents evolving against each other) — Iteration 5+  

---

## Next Steps

After reading this document, you should understand:

✅ Why genetic algorithms work  
✅ What each operator does (selection, crossover, mutation)  
✅ The math and intuition behind each choice  
✅ Common failure modes and how to fix them  
✅ Why EvoTribes uses specific settings  

**Ready to implement?** See `docs/notes/iteration_03.md` (after Iteration 3 is built).

**Questions?** Add them to iteration notes or `.copilot-instructions.md`.

---

## References & Further Reading

- Goldberg, David E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. [Classic textbook]
- Mitchell, Melanie (1996). *An Introduction to Genetic Algorithms*. [Accessible intro]
- Whitley, Darrell (1994). "A Genetic Algorithm Tutorial". *Statistics and Computing*. [Excellent overview]
- Stanley & Miikkulainen (2002). "Evolving Neural Networks through Augmenting Topologies" (NEAT). [Advanced: topology evolution]

---

**Document Status:** Educational reference for Iteration 3 implementation  
**Last Updated:** 2026-02-15  
**Author:** EvoTribes Development Team

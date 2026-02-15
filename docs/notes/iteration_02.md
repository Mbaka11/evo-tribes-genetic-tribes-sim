# Iteration 02 — Policy Interface & Swappable Agent Brains

**Version:** `0.2.0`
**Date:** 2026-02-15
**Branch:** `main`

---

### 1. Goal

Introduce a clean policy abstraction so that every agent selects actions
through a pluggable "brain" instead of hardcoded `env.action_space.sample()`.
Ship two concrete policies — `RandomPolicy` (baseline) and `MLPPolicy`
(pure-NumPy neural network with flat parameter access for the future
genetic algorithm).

---

### 2. Why This Matters

Iteration 1 proved the environment works. But agents were wired to random
actions — there was no way to swap in smarter behaviour.

The policy interface unlocks:

- **Iteration 3 (Genetic Algorithm):** needs `get_params()` / `set_params()`
  to treat an agent's brain as a chromosome.
- **Fair comparisons:** same environment, different policies, identical
  evaluation loop.
- **Persistence:** `save()` / `load()` let us checkpoint and replay
  interesting agents.

Without this layer, evolution has nothing to evolve.

---

### 3. What Was Implemented

| File                            | Change Type | Description                                                                        |
| ------------------------------- | ----------- | ---------------------------------------------------------------------------------- | ---------------------------------------- |
| `src/policies/__init__.py`      | Created     | Package exports: `BasePolicy`, `RandomPolicy`, `MLPPolicy`                         |
| `src/policies/base_policy.py`   | Created     | Abstract base class with `select_action`, `get_params`/`set_params`, `save`/`load` |
| `src/policies/random_policy.py` | Created     | Random action selection — 0 parameters, baseline agent                             |
| `src/policies/mlp_policy.py`    | Created     | Pure NumPy MLP: Xavier init, ReLU, softmax, flat param get/set                     |
| `scripts/demo.py`               | Modified    | `--policy random                                                                   | mlp`and`--stochastic` flags via argparse |
| `tests/test_policies.py`        | Created     | 25+ tests across 6 test classes                                                    |
| `docs/02_policies.md`           | Modified    | Full rewrite — interface spec, architecture, worked examples                       |

---

### 4. Architecture & Design Decisions

#### Abstract base class pattern

All policies inherit from `BasePolicy`, which defines the contract:

```
BasePolicy (ABC)
  ├── select_action(obs)     [abstract, MUST override]
  ├── get_params() → array   [default: empty]
  ├── set_params(params)     [default: no-op]
  ├── param_count() → int    [derived: len(get_params())]
  ├── save(path)             [inherited: np.save]
  └── load(path)             [inherited: np.load + set_params]
```

**Why an ABC?** It catches missing `select_action` at instantiation
time rather than at first use deep in a simulation loop.

**Why default no-ops for params?** Policies like `RandomPolicy` have no
weights to get or set. Making the param methods optional (with safe
defaults) avoids forcing every subclass to implement dead code.

#### Pure NumPy MLP

No PyTorch or TensorFlow — intentional:

1. The genetic algorithm evolves weights by direct mutation, not
   backpropagation. We never need autograd.
2. Forward passes through a 533-parameter network are microsecond-fast
   in NumPy.
3. Zero external dependencies beyond what we already have.

#### One policy per agent

```python
policies = [RandomPolicy(seed=i) for i in range(num_agents)]
actions  = [policies[i].select_action(obs[i]) for i in range(num_agents)]
```

Each agent owns its own policy instance. This is essential for the GA:
different agents will have different weight vectors competing against
each other.

#### Flat parameter representation

Every tuneable weight is accessed as a single 1-D float32 array. This
maps directly to a genetic algorithm chromosome:

```
chromosome = policy.get_params()   # shape (533,) for default MLP
# ... crossover, mutation ...
policy.set_params(mutated_chromosome)
```

---

### 5. Mathematical Considerations

#### Xavier uniform initialisation

$$\text{limit} = \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$$

$$W_{ij} \sim \mathcal{U}(-\text{limit}, +\text{limit})$$

Where:

- $\text{fan\_in}$ = number of input neurons to the layer
- $\text{fan\_out}$ = number of output neurons from the layer

**Intuition:** Xavier keeps the variance of activations roughly constant
as the signal passes through layers. Too-large initial weights cause
exploding outputs; too-small weights cause the signal to vanish.
Xavier balances both by scaling inversely with layer width.

**Concrete calculation** for layer 1 (27 → 16):

$$\text{limit} = \sqrt{\frac{6}{27 + 16}} = \sqrt{\frac{6}{43}} \approx 0.3734$$

So every weight in $W_0$ is drawn uniformly from $[-0.3734, +0.3734]$.

#### Softmax

$$p_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$$

Where:

- $z_i$ = raw output (logit) for action $i$
- $\max(\mathbf{z})$ = maximum logit (subtracted for numerical stability)

**Intuition:** Softmax converts raw scores into a valid probability
distribution. Larger logits get exponentially more probability mass.

---

### 6. Algorithms & Logic

#### MLP forward pass

```
1. Receive observation vector x of shape (27,)
2. For each hidden layer i:
   a. Compute z = x @ W[i] + b[i]     (matrix multiply + bias)
   b. Apply ReLU: x = max(0, z)
3. Compute output logits = x @ W[-1] + b[-1]
4. Apply softmax to get probabilities p
5. If deterministic: action = argmax(p)
   If stochastic: action = sample from p
```

#### Parameter flattening / unflattening

```
Flatten (get_params):
1. For each layer: append W.ravel(), then b
2. Concatenate all into single 1-D array

Unflatten (set_params):
1. offset = 0
2. For each layer:
   a. Read W.size elements → reshape to (fan_in, fan_out)
   b. Read b.size elements
   c. Advance offset
```

---

### 7. Key Concepts Explained

#### Policy

A function that takes an observation and returns an action. Think of
it as the agent's "brain" — it decides what to do given what the agent
sees. Different policies make different decisions: random noise,
neural networks, hand-crafted rules.

#### Parameters (weights)

Numbers inside a policy that control its behaviour. For the MLP, these
are the connection weights between neurons. The genetic algorithm will
mutate these numbers to find agents that perform well.

A `RandomPolicy` has 0 parameters — its behaviour is fixed (random).
An `MLPPolicy` with architecture 27→16→5 has 533 parameters.

#### Xavier initialisation

A method for setting initial neural network weights. Named after
Xavier Glorot. The key insight: the initial weights should be
"just right" — not too big (exploding signals) and not too small
(dying signals). It scales the range based on the number of neurons
connected to each weight.

#### Softmax

Turns a vector of arbitrary numbers into proper probabilities (positive,
sum to 1). Bigger inputs get bigger probabilities, exponentially.
The name is a "soft" version of argmax — instead of picking just the
largest, it gives every option some probability proportional to its
score.

---

### 8. Concrete Examples

#### Example 1: RandomPolicy action selection

**Setup:**

- `RandomPolicy(num_actions=5, seed=42)`
- `obs = [0.0] * 27` (content irrelevant — random ignores it)

**Step-by-step:**

```
Step 1: select_action(obs) called
Step 2: Policy ignores obs entirely
Step 3: RNG(seed=42) generates random integer in [0, 5)
Step 4: Returns 0 (Stay)
```

**Result:** Action = 0

**What this shows:** RandomPolicy is completely independent of the
observation. Two RandomPolicy instances with the same seed will
produce the exact same action sequence even with different observations.

---

#### Example 2: MLP forward pass with specific numbers

**Setup:**

- `MLPPolicy(obs_size=4, num_actions=3, hidden_sizes=[2], seed=0)`
- Architecture: 4→2→3, total params = 4×2 + 2 + 2×3 + 3 = 17
- `obs = [1.0, 0.5, 0.0, −0.5]`

**Step-by-step:**

```
Step 1: Hidden layer
        z = obs @ W0 + b0
        z = [1.0, 0.5, 0.0, -0.5] @ [[w00, w01], [w10, w11],
             [w20, w21], [w30, w31]] + [0, 0]
        Suppose z = [0.42, -0.18]

Step 2: ReLU activation
        h = max(0, z) = [0.42, 0.0]

Step 3: Output layer
        logits = h @ W1 + b1
        logits = [0.42, 0.0] @ [[w00, w01, w02],
                  [w10, w11, w12]] + [0, 0, 0]
        Suppose logits = [0.15, -0.08, 0.03]

Step 4: Softmax
        max = 0.15
        exp = [e^0, e^(-0.23), e^(-0.12)]
            = [1.0, 0.795, 0.887]
        sum = 2.682
        probs = [0.373, 0.296, 0.331]

Step 5: Action (deterministic)
        argmax([0.373, 0.296, 0.331]) = 0
```

**Result:** Action = 0

**What this shows:** The MLP transforms the observation through
weighted sums and nonlinearities. ReLU zeroes out negative hidden
activations. Softmax converts raw scores to probabilities.
Different weights (from evolution) would produce different actions.

---

#### Example 3: Parameter round-trip for genetic algorithm

**Setup:**

- `MLPPolicy(obs_size=27, num_actions=5, hidden_sizes=[16])`
- 533 total parameters

**Step-by-step:**

```
Step 1: Extract chromosome
        params = policy.get_params()
        # shape: (533,) — flat float32 array
        # Layout: [W0(432), b0(16), W1(80), b1(5)]

Step 2: Simulate mutation
        noise = np.random.normal(0, 0.01, size=533)
        mutated = params + noise

Step 3: Inject mutated chromosome
        policy.set_params(mutated)
        # Internally: slices array back into W0(27×16), b0(16),
        #             W1(16×5), b1(5)

Step 4: Verify
        new_params = policy.get_params()
        assert np.allclose(new_params, mutated)  # ✓
```

**Result:** The policy now uses the mutated weights.

**What this shows:** The flat parameter interface lets the GA treat the
entire neural network as a simple array of numbers — crossover and
mutation are just array operations.

---

#### Example 4: Save, load, and verify behaviour

**Setup:**

- `mlp = MLPPolicy(obs_size=27, num_actions=5, seed=0, deterministic=True)`
- `obs = np.ones(27) * 0.5`

**Step-by-step:**

```
Step 1: Record action with original policy
        action_before = mlp.select_action(obs)  # e.g. 3 (East)

Step 2: Save to disk
        mlp.save("best_agent.npy")
        # File contains 533 float32 values = 2132 bytes

Step 3: Create brand new policy (different seed)
        mlp2 = MLPPolicy(obs_size=27, num_actions=5, seed=99,
                         deterministic=True)
        mlp2.select_action(obs)  # e.g. 1 (different action)

Step 4: Load saved weights
        mlp2.load("best_agent.npy")
        action_after = mlp2.select_action(obs)  # → 3 (East, same!)

Step 5: Verify
        assert action_before == action_after  # ✓
```

**Result:** The loaded policy produces identical actions to the original.

**What this shows:** Persistence works correctly — we can save the best
agent from a GA run and reload it later for analysis or further evolution.

---

### 9. Configuration & Parameters

| Parameter       | Default | What It Controls                                 |
| --------------- | ------- | ------------------------------------------------ |
| `num_actions`   | `5`     | Size of the discrete action space                |
| `obs_size`      | `27`    | Length of the observation vector                 |
| `hidden_sizes`  | `[16]`  | List of hidden layer widths                      |
| `seed`          | `None`  | RNG seed for weight init and stochastic sampling |
| `deterministic` | `True`  | If True: argmax. If False: sample from softmax   |

---

### 10. How to Run

```bash
# Run with random policy (baseline)
python -m scripts.demo --policy random

# Run with MLP policy (random weights)
python -m scripts.demo --policy mlp

# Run with stochastic MLP (sample from softmax)
python -m scripts.demo --policy mlp --stochastic

# Run all tests
python -m pytest tests/ -v
```

---

### 11. What You Should See

#### Random policy

Similar to Iteration 1 — agents wander aimlessly, occasionally eating
food. Movement looks chaotic with no discernible pattern.

#### MLP policy (deterministic)

Agents with the same observation consistently move the same direction.
Since weights are random (not trained), behaviour is still poor, but
you'll notice agents are more "stubborn" — they tend to repeat the same
action in similar situations. Some agents might get stuck walking into
walls repeatedly.

#### MLP policy (stochastic)

Like MLP but with more varied movement. Agents occasionally break out
of repetitive patterns because they sample from the probability
distribution instead of always picking the highest-probability action.

#### Console output

```
EvoTribes — Iteration 2 demo
Grid 20x20, 6 agents, 2 tribes
Policy: RandomPolicy(num_actions=5, params=0)
Close the window to quit.

  step   50  |  alive 5/6  |  total energy 3.2
  step  100  |  alive 4/6  |  total energy 2.1
  ...
```

#### Test output

```
tests/test_env_smoke.py::TestReset::test_... PASSED
tests/test_policies.py::TestInterfaceCompliance::test_... PASSED
...
XX passed in 0.XXs
```

---

### 12. Known Limitations & Bugs

- **MLP weights are random:** Agents don't learn anything yet. The MLP
  exists to be evolved by the genetic algorithm in Iteration 3.
- **No recurrent memory:** Policies are stateless. An agent can't
  remember where it has been.
- **Fixed architecture:** The hidden layer sizes are set at construction
  time. The GA will not evolve the architecture — only the weights.
- **Single hidden layer default:** 27→16→5 might be too small for
  complex strategies. Experimentation needed in later iterations.

---

### 13. Docs Modified

| Doc File                     | What Changed                                                              |
| ---------------------------- | ------------------------------------------------------------------------- |
| `docs/02_policies.md`        | Full rewrite — interface spec, architecture, Xavier math, worked examples |
| `docs/notes/iteration_02.md` | Created — this file                                                       |

---

### 14. Test Coverage

| Test Class                | # Tests | What It Checks                                                                      |
| ------------------------- | ------- | ----------------------------------------------------------------------------------- |
| `TestInterfaceCompliance` | 6       | Both policies satisfy BasePolicy contract                                           |
| `TestRandomPolicy`        | 4       | Zero params, ignores obs, reproducibility, covers all actions                       |
| `TestMLPPolicy`           | 6       | Architecture sizes, softmax output, determinism, stochasticity, seed independence   |
| `TestParamRoundTrip`      | 4       | get/set preserves weights, changes behaviour, wrong length raises, no-op for random |
| `TestSaveLoad`            | 3       | MLP save→load roundtrip, loaded actions match, random save/load noop                |
| `TestPolicySwap`          | 2       | Random→MLP swap mid-episode, MLP full-episode integration                           |

---

### 15. Version History

| Version | Change                                                           |
| ------- | ---------------------------------------------------------------- |
| `0.1.0` | Grid world environment, random agents, rendering                 |
| `0.2.0` | Policy interface, RandomPolicy, MLPPolicy, demo policy selection |

---

### 16. Next Iteration Preview

**Iteration 3 — Genetic Algorithm**

The genetic algorithm evolves MLPPolicy weights across generations:

1. **Population:** N agents, each with an MLPPolicy (533 params).
2. **Evaluation:** Run all agents for one episode, score by total reward.
3. **Selection:** Pick the top K agents (tournament or rank-based).
4. **Crossover:** Blend parent chromosomes (e.g. uniform crossover).
5. **Mutation:** Add Gaussian noise to offspring weights.
6. **Repeat:** New generation, hopefully better.

What you'll see in the Pygame window after many generations:

- Agents learning to seek food (directed movement instead of random)
- Tribal clustering (same-tribe agents evolving similar strategies)
- Energy-aware behaviour (agents near death moving more
  aggressively toward food)

The policy interface from this iteration makes all of this possible:
`get_params()` → crossover → `set_params()` is the entire GA-policy
bridge.

---

### 17. Questions & Open Issues

- **Architecture search:** Should the GA also evolve the number/size
  of hidden layers, or just the weights within a fixed architecture?
- **Population size:** How many agents per generation? Too few = no
  diversity; too many = slow evaluation.
- **Mutation rate:** What standard deviation for the Gaussian noise?
  This is a critical hyperparameter.
- **Elitism:** Should the best agent always survive to the next
  generation unchanged?

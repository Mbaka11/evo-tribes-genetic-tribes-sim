# Agent Policies (Iteration 2)

Every agent in EvoTribes has a **policy** — a function that maps an
observation vector to an action integer. All policies share a common
interface (`BasePolicy`) so the environment, demo scripts, and the
genetic algorithm (Iteration 3) can work with any policy without
knowing its internals.

---

## BasePolicy Interface

```python
from src.policies.base_policy import BasePolicy
```

| Method               | Required            | Purpose                                    |
| -------------------- | ------------------- | ------------------------------------------ |
| `select_action(obs)` | **Yes**             | Observation → action `int`                 |
| `get_params()`       | No (default: `[]`)  | Flat float32 array of all tuneable weights |
| `set_params(params)` | No (default: no-op) | Inverse of `get_params`                    |
| `param_count()`      | No (inherited)      | `len(get_params())`                        |
| `save(path)`         | No (inherited)      | Write params to `.npy` file                |
| `load(path)`         | No (inherited)      | Read params from `.npy` file               |

> **Design rule:** policies are **stateless between steps** — no hidden
> memory. If we add recurrent policies later they will manage their own
> state and expose a `reset_state()` method.

### Example — creating a custom policy

```python
class AlwaysNorth(BasePolicy):
    def select_action(self, observation):
        return 1  # action 1 = North
```

---

## RandomPolicy

```python
from src.policies.random_policy import RandomPolicy

policy = RandomPolicy(num_actions=5, seed=42)
action = policy.select_action(obs)  # random int in [0, 5)
```

- Ignores the observation entirely.
- Has **0 tuneable parameters** (`param_count() == 0`).
- Useful as a baseline — "how well does pure chance do?"

### Worked example

```
obs = [0.0, 0.0, …, 0.8, 0.0]   (27 floats — content irrelevant)
RandomPolicy(seed=42).select_action(obs) → 0  (Stay)
RandomPolicy(seed=42).select_action(obs) → 3  (East)  # next call
```

---

## MLPPolicy

A fully-connected neural network built with **pure NumPy** — no
PyTorch, no TensorFlow. The genetic algorithm (Iteration 3) will
evolve the weights directly, so we only need forward passes plus
the ability to flatten / unflatten parameters.

```python
from src.policies.mlp_policy import MLPPolicy

policy = MLPPolicy(
    obs_size=27,       # env observation length
    num_actions=5,     # Discrete(5)
    hidden_sizes=[16], # one hidden layer of 16 neurons
    seed=0,
    deterministic=True,
)
```

### Architecture

```
Input  (27,)
  │
  ▼
Dense  (27 → 16) + bias  →  ReLU       weights: 432  bias: 16
  │
  ▼
Dense  (16 → 5)  + bias  →  Softmax    weights:  80  bias:  5
  │
  ▼
Output (5,)  →  argmax → action

Total parameters: 533
```

### Weight initialisation — Xavier uniform

$$\text{limit} = \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$$

$$W \sim \text{Uniform}(-\text{limit}, +\text{limit}), \qquad b = 0$$

For the first layer (27 → 16):

$$\text{limit} = \sqrt{6 / (27 + 16)} = \sqrt{6/43} \approx 0.3734$$

### Forward pass

1. **Hidden layer:** $h = \text{ReLU}(x \cdot W_0 + b_0)$
2. **Output layer:** $\text{logits} = h \cdot W_1 + b_1$
3. **Softmax:** $p_i = \frac{e^{\text{logit}_i - \max}}{\sum_j e^{\text{logit}_j - \max}}$
4. **Action:** `argmax(p)` (deterministic) or `sample(p)` (stochastic)

### Parameter layout

`get_params()` returns a single flat `float32` array:

```
[W0.ravel(), b0, W1.ravel(), b1]
 ↑ 432       ↑16  ↑ 80       ↑5   = 533 total
```

`set_params()` slices this back into the original weight/bias shapes.
This is critical for the genetic algorithm: it can treat the entire
policy as a single chromosome.

### Worked example — forward pass

```
obs      = [0.5, 0.5, …, 0.5]       # 27 values, all 0.5
W0       = (27×16) Xavier init        # e.g. first element ≈ 0.12
hidden   = ReLU(obs @ W0 + b0)       # 16 values, zeros clipped
logits   = hidden @ W1 + b1          # 5 raw scores
probs    = softmax(logits)           # e.g. [0.18, 0.22, 0.19, 0.21, 0.20]
action   = argmax(probs)             # → 1 (North)
```

### Stochastic vs deterministic

| Mode          | Flag                  | Behaviour                        |
| ------------- | --------------------- | -------------------------------- |
| Deterministic | `deterministic=True`  | Always `argmax(probs)`           |
| Stochastic    | `deterministic=False` | Sample from `probs` distribution |

---

## Running the demo with policies

```bash
python -m scripts.demo --policy random
python -m scripts.demo --policy mlp
python -m scripts.demo --policy mlp --stochastic
```

---

## Policy swap at runtime

Policies are independent from the environment. You can swap a policy
mid-episode with no special API:

```python
policies[0] = MLPPolicy(obs_size=27, num_actions=5, seed=0)
# From this step onward, agent 0 uses the MLP instead of random
```

---

## Future: genetic algorithm integration (Iteration 3)

```python
# Crossover: blend two parent chromosomes
parent_a = best_agent.get_params()
parent_b = second_best.get_params()
child_params = (parent_a + parent_b) / 2 + noise
child = MLPPolicy(obs_size=27, num_actions=5)
child.set_params(child_params)
```

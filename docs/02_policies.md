# Agent Policies

The system supports multiple policy backends.

---

## MLP Policy

A small neural network that maps observations to actions.

### Advantages

- rich emergent behavior
- flexible strategy formation

### Evolution

The genetic algorithm evolves the network weights.

---

## Decision Tree / Rule Policy

A rule-based policy using thresholds.

### Advantages

- interpretable
- human-readable strategies
- easier debugging

---

## Shared Interface

All policies implement:

Policy.act(observation) → action

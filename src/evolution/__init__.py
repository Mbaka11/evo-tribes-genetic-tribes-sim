"""
EvoTribes — Evolution Package
===============================

Genetic algorithm components for evolving agent policies.

Modules
-------
fitness     Evaluate an agent's policy in the environment.
selection   Select parents from a population (tournament selection).
crossover   Combine two parent chromosomes into offspring.
mutation    Perturb a chromosome with Gaussian noise.
population  Orchestrate the full evolutionary loop.

Example
-------
>>> from src.evolution import Population
>>> pop = Population(population_size=50)
>>> pop.run(generations=100)
"""

from src.evolution.fitness import evaluate_agent, evaluate_robust
from src.evolution.selection import tournament_selection
from src.evolution.crossover import uniform_crossover
from src.evolution.mutation import gaussian_mutate, adaptive_mutation_std
from src.evolution.population import Population

__all__ = [
    "evaluate_agent",
    "evaluate_robust",
    "tournament_selection",
    "uniform_crossover",
    "gaussian_mutate",
    "adaptive_mutation_std",
    "Population",
]

"""
EvoTribes — Training Script (Iteration 3)
============================================

Evolves agent policies using a genetic algorithm.

Usage
-----
    # Quick test (small population, few generations)
    python -m scripts.train --population 10 --generations 5

    # Full run with defaults
    python -m scripts.train

    # Custom experiment
    python -m scripts.train --population 50 --generations 100 \\
        --mutation-std 0.02 --elitism 2 --tournament-size 3 \\
        --eval-episodes 3 --output runs/exp_001 --seed 42

All results are saved to the output directory:
    - config.json    — full experiment configuration
    - metrics.csv    — per-generation statistics
    - best_genNNNN.npy — checkpoint chromosomes
    - best_final.npy — best chromosome found overall
"""

import argparse
import sys
import os

# Ensure project root is on the path so `src` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evolution.population import Population


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvoTribes — Evolve agent policies with a genetic algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Population & generations
    parser.add_argument(
        "--population", type=int, default=50,
        help="Number of individuals in the population",
    )
    parser.add_argument(
        "--generations", type=int, default=100,
        help="Maximum number of generations to evolve",
    )

    # GA operators
    parser.add_argument(
        "--tournament-size", type=int, default=3,
        help="Tournament selection size (k)",
    )
    parser.add_argument(
        "--mutation-std", type=float, default=0.02,
        help="Initial Gaussian mutation standard deviation",
    )
    parser.add_argument(
        "--mutation-decay", type=float, default=0.9,
        help="Mutation std decay factor",
    )
    parser.add_argument(
        "--decay-every", type=int, default=20,
        help="Apply mutation decay every N generations",
    )
    parser.add_argument(
        "--elitism", type=int, default=2,
        help="Number of top individuals copied unchanged to next generation",
    )

    # Evaluation
    parser.add_argument(
        "--eval-episodes", type=int, default=3,
        help="Number of episodes to average for robust fitness evaluation",
    )

    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=30,
        help="Stop if best fitness doesn't improve for N generations",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="runs/default",
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save best chromosome every N generations",
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed",
    )

    args = parser.parse_args()

    pop = Population(
        population_size=args.population,
        generations=args.generations,
        tournament_k=args.tournament_size,
        mutation_std=args.mutation_std,
        mutation_decay=args.mutation_decay,
        decay_every=args.decay_every,
        elitism=args.elitism,
        eval_episodes=args.eval_episodes,
        patience=args.patience,
        output_dir=args.output,
        save_every=args.save_every,
        seed=args.seed,
    )

    best_params, best_fitness = pop.run()
    print(f"\nDone. Best fitness: {best_fitness:.4f}")
    print(f"Replay with:  python -m scripts.replay --checkpoint {args.output}/best_final.npy")


if __name__ == "__main__":
    main()

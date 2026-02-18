"""
EvoTribes — Population Manager
=================================

The :class:`Population` class orchestrates the full genetic algorithm
loop.  It manages a collection of chromosomes and evolves them over
generations.

GA Loop (one generation)
------------------------
1. **Evaluate** — Run every chromosome in the environment and record
   its fitness.
2. **Log** — Write generation statistics to CSV.
3. **Check early stop** — If the best fitness hasn't improved for
   ``patience`` generations, stop.
4. **Elitism** — Copy the top ``elitism`` individuals directly to the
   next generation (they skip crossover and mutation).
5. **Selection + Crossover + Mutation** — Fill the remaining slots:
   a. Select two parents via tournament selection.
   b. Create a child via uniform crossover.
   c. Mutate the child with adaptive Gaussian noise.
6. **Replace** — The new generation replaces the old.

Checkpointing
--------------
Every ``save_every`` generations, the best chromosome is saved as a
``.npy`` file so it can be loaded later for replay or analysis.

A ``config.json`` is saved at the start so the experiment is fully
reproducible.

Example
-------
>>> from src.evolution.population import Population
>>> pop = Population(population_size=10, generations=5)
>>> best_params, best_fitness = pop.run()
>>> best_params.shape
(533,)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.policies import MLPPolicy
from src.evolution.fitness import evaluate_robust
from src.evolution.selection import tournament_selection
from src.evolution.crossover import uniform_crossover
from src.evolution.mutation import gaussian_mutate, adaptive_mutation_std


class Population:
    """Manages a population of chromosomes through evolution.

    Args:
        population_size: Number of individuals (default 50).
        generations:     Maximum number of generations (default 100).
        obs_size:        Observation vector length (default 27).
        num_actions:     Number of actions (default 5).
        hidden_sizes:    MLP hidden widths (default [16]).
        tournament_k:    Tournament selection size (default 3).
        mutation_std:    Initial mutation std (default 0.02).
        mutation_decay:  Std decay factor (default 0.9).
        decay_every:     Apply decay every N gens (default 20).
        elitism:         Number of elite individuals preserved (default 2).
        eval_episodes:   Episodes per fitness evaluation (default 3).
        patience:        Early stop if no improvement for N gens (default 30).
        output_dir:      Directory for logs and checkpoints (default "runs/default").
        save_every:       Save best chromosome every N gens (default 10).
        env_config:      Environment config overrides.
        seed:            Master RNG seed for reproducibility.

    Example:
        >>> pop = Population(population_size=10, generations=3,
        ...                  output_dir="runs/test")
        >>> best, fitness = pop.run()
        >>> isinstance(fitness, float)
        True
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        obs_size: int = 27,
        num_actions: int = 5,
        hidden_sizes: Optional[List[int]] = None,
        tournament_k: int = 3,
        mutation_std: float = 0.02,
        mutation_decay: float = 0.9,
        decay_every: int = 20,
        elitism: int = 2,
        eval_episodes: int = 3,
        patience: int = 30,
        output_dir: str = "runs/default",
        save_every: int = 10,
        env_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        if hidden_sizes is None:
            hidden_sizes = [16]

        self.population_size = population_size
        self.generations = generations
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden_sizes = hidden_sizes
        self.tournament_k = tournament_k
        self.mutation_std = mutation_std
        self.mutation_decay = mutation_decay
        self.decay_every = decay_every
        self.elitism = elitism
        self.eval_episodes = eval_episodes
        self.patience = patience
        self.output_dir = output_dir
        self.save_every = save_every
        self.env_config = env_config or {}
        self.seed = seed

        # Master RNG — all randomness flows from this seed
        self.rng = np.random.default_rng(seed)

        # Create initial population
        self.chromosomes = self._init_population()

        # Tracking
        self.best_fitness_ever: float = -np.inf
        self.best_chromosome_ever: np.ndarray = self.chromosomes[0].copy()
        self.gens_without_improvement: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_population(self) -> List[np.ndarray]:
        """Create initial population with Xavier-initialised networks.

        Each individual starts as a fresh MLPPolicy with random weights.
        This gives a diverse starting population.

        Returns:
            List of chromosomes (flat float32 arrays).
        """
        population = []
        for i in range(self.population_size):
            policy = MLPPolicy(
                obs_size=self.obs_size,
                num_actions=self.num_actions,
                hidden_sizes=self.hidden_sizes,
                seed=int(self.rng.integers(0, 2**31)),
            )
            population.append(policy.get_params())
        return population

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate_population(self) -> np.ndarray:
        """Evaluate every chromosome and return an array of fitnesses.

        Each chromosome is evaluated over ``eval_episodes`` episodes
        with different seeds (robust evaluation).

        Returns:
            1-D float64 array of shape (population_size,).
        """
        fitnesses = np.zeros(self.population_size)
        for i, chromosome in enumerate(self.chromosomes):
            fitnesses[i] = evaluate_robust(
                chromosome,
                num_episodes=self.eval_episodes,
                obs_size=self.obs_size,
                num_actions=self.num_actions,
                hidden_sizes=self.hidden_sizes,
                env_config=self.env_config,
                base_seed=int(self.rng.integers(0, 2**31)),
            )
        return fitnesses

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------
    def _evolve_one_generation(
        self, fitnesses: np.ndarray, generation: int
    ) -> List[np.ndarray]:
        """Produce the next generation from the current one.

        Steps:
        1. Sort by fitness (descending).
        2. Copy the top ``elitism`` individuals unchanged.
        3. Fill remaining slots via tournament → crossover → mutation.

        Args:
            fitnesses:  Fitness array for the current generation.
            generation: Current generation number (for adaptive mutation).

        Returns:
            New list of chromosomes (same length as current population).
        """
        # Sort indices by fitness (best first)
        sorted_indices = np.argsort(fitnesses)[::-1]

        new_population: List[np.ndarray] = []

        # --- Elitism: keep the top individuals unchanged -----------------
        for rank in range(self.elitism):
            idx = sorted_indices[rank]
            new_population.append(self.chromosomes[idx].copy())

        # --- Fill remaining slots with offspring -------------------------
        current_std = adaptive_mutation_std(
            generation,
            initial_std=self.mutation_std,
            decay=self.mutation_decay,
            decay_every=self.decay_every,
        )

        while len(new_population) < self.population_size:
            # Select two parents
            parent_a = tournament_selection(
                self.chromosomes, fitnesses, k=self.tournament_k, rng=self.rng
            )
            parent_b = tournament_selection(
                self.chromosomes, fitnesses, k=self.tournament_k, rng=self.rng
            )

            # Crossover
            child = uniform_crossover(parent_a, parent_b, rng=self.rng)

            # Mutation
            child = gaussian_mutate(child, std=current_std, rng=self.rng)

            new_population.append(child)

        return new_population

    # ------------------------------------------------------------------
    # Diversity metric
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_diversity(chromosomes: List[np.ndarray]) -> float:
        """Mean pairwise Euclidean distance (sampled for efficiency).

        Measures how different the individuals in the population are.
        Low diversity = population is converging (everyone looks similar).
        High diversity = population is still exploring.

        For populations > 30, we sample 30 pairs to keep it fast.

        Returns:
            Mean pairwise distance (float).
        """
        n = len(chromosomes)
        if n < 2:
            return 0.0

        # Sample pairs instead of computing all n*(n-1)/2
        num_pairs = min(30, n * (n - 1) // 2)
        rng = np.random.default_rng(0)  # deterministic for logging
        distances = []
        for _ in range(num_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            dist = np.linalg.norm(chromosomes[i] - chromosomes[j])
            distances.append(dist)
        return float(np.mean(distances))

    # ------------------------------------------------------------------
    # Logging — CSV
    # ------------------------------------------------------------------
    def _init_log(self) -> str:
        """Create output directory and CSV log file.  Return CSV path."""
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "metrics.csv")
        with open(csv_path, "w") as f:
            f.write(
                "generation,best_fitness,avg_fitness,worst_fitness,"
                "diversity,mutation_std,elapsed_sec\n"
            )
        return csv_path

    def _log_generation(
        self,
        csv_path: str,
        generation: int,
        fitnesses: np.ndarray,
        diversity: float,
        current_std: float,
        elapsed: float,
    ) -> None:
        """Append one row to the CSV log."""
        row = (
            f"{generation},"
            f"{fitnesses.max():.4f},"
            f"{fitnesses.mean():.4f},"
            f"{fitnesses.min():.4f},"
            f"{diversity:.4f},"
            f"{current_std:.6f},"
            f"{elapsed:.2f}\n"
        )
        with open(csv_path, "a") as f:
            f.write(row)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def _save_config(self) -> None:
        """Save experiment configuration to JSON for reproducibility."""
        config = {
            "population_size": self.population_size,
            "generations": self.generations,
            "obs_size": self.obs_size,
            "num_actions": self.num_actions,
            "hidden_sizes": self.hidden_sizes,
            "tournament_k": self.tournament_k,
            "mutation_std": self.mutation_std,
            "mutation_decay": self.mutation_decay,
            "decay_every": self.decay_every,
            "elitism": self.elitism,
            "eval_episodes": self.eval_episodes,
            "patience": self.patience,
            "save_every": self.save_every,
            "env_config": self.env_config,
            "seed": self.seed,
        }
        path = os.path.join(self.output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def _save_checkpoint(self, generation: int) -> None:
        """Save the best chromosome as a .npy file."""
        path = os.path.join(
            self.output_dir, f"best_gen{generation:04d}.npy"
        )
        np.save(path, self.best_chromosome_ever)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> Tuple[np.ndarray, float]:
        """Execute the full evolutionary loop.

        Returns:
            Tuple of (best_chromosome, best_fitness).

        Side effects:
            - Creates output directory with config.json, metrics.csv,
              and checkpoint .npy files.
            - Prints progress to stdout.
        """
        csv_path = self._init_log()
        self._save_config()

        print(f"{'='*60}")
        print(f"EvoTribes — Genetic Algorithm Training")
        print(f"{'='*60}")
        print(f"Population: {self.population_size}  |  "
              f"Generations: {self.generations}  |  "
              f"Elitism: {self.elitism}")
        print(f"Tournament k: {self.tournament_k}  |  "
              f"Mutation std: {self.mutation_std}  |  "
              f"Eval episodes: {self.eval_episodes}")
        print(f"Patience: {self.patience}  |  "
              f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for gen in range(self.generations):
            gen_start = time.time()

            # 1. Evaluate
            fitnesses = self._evaluate_population()

            # 2. Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > self.best_fitness_ever:
                self.best_fitness_ever = gen_best_fitness
                self.best_chromosome_ever = self.chromosomes[gen_best_idx].copy()
                self.gens_without_improvement = 0
            else:
                self.gens_without_improvement += 1

            # 3. Diversity
            diversity = self._compute_diversity(self.chromosomes)

            # 4. Mutation std for this generation
            current_std = adaptive_mutation_std(
                gen,
                initial_std=self.mutation_std,
                decay=self.mutation_decay,
                decay_every=self.decay_every,
            )

            # 5. Log
            elapsed = time.time() - start_time
            self._log_generation(
                csv_path, gen, fitnesses, diversity, current_std, elapsed
            )

            gen_elapsed = time.time() - gen_start
            print(
                f"Gen {gen:>3d}/{self.generations}  |  "
                f"best {gen_best_fitness:>8.2f}  |  "
                f"avg {fitnesses.mean():>8.2f}  |  "
                f"div {diversity:>6.2f}  |  "
                f"σ {current_std:.4f}  |  "
                f"{gen_elapsed:.1f}s"
            )

            # 6. Checkpoint
            if gen % self.save_every == 0 or gen == self.generations - 1:
                self._save_checkpoint(gen)

            # 7. Early stop
            if self.gens_without_improvement >= self.patience:
                print(
                    f"\nEarly stop at gen {gen}: no improvement "
                    f"for {self.patience} generations."
                )
                break

            # 8. Evolve next generation
            self.chromosomes = self._evolve_one_generation(fitnesses, gen)

        # Final save
        self._save_checkpoint(gen)
        final_path = os.path.join(self.output_dir, "best_final.npy")
        np.save(final_path, self.best_chromosome_ever)

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {total_time:.1f}s")
        print(f"Best fitness: {self.best_fitness_ever:.4f}")
        print(f"Best chromosome saved to: {final_path}")
        print(f"Metrics log: {csv_path}")
        print(f"{'='*60}")

        return self.best_chromosome_ever, self.best_fitness_ever

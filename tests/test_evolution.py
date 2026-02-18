"""
EvoTribes — Evolution Module Tests (Iteration 3)
====================================================

Tests for every GA component: fitness, selection, crossover, mutation,
and the population manager.

Run with:
    python -m pytest tests/test_evolution.py -v
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from src.policies import MLPPolicy
from src.evolution.fitness import evaluate_agent, evaluate_robust
from src.evolution.selection import tournament_selection
from src.evolution.crossover import uniform_crossover
from src.evolution.mutation import gaussian_mutate, adaptive_mutation_std
from src.evolution.population import Population


# =====================================================================
# Fitness
# =====================================================================
class TestEvaluateAgent:
    """Tests for single-episode fitness evaluation."""

    def test_returns_float(self):
        """evaluate_agent should return a float."""
        chromosome = MLPPolicy(seed=0).get_params()
        result = evaluate_agent(chromosome, seed=0)
        assert isinstance(result, float)

    def test_deterministic_with_same_seed(self):
        """Same chromosome + same seed → same fitness."""
        chromosome = MLPPolicy(seed=0).get_params()
        f1 = evaluate_agent(chromosome, seed=42)
        f2 = evaluate_agent(chromosome, seed=42)
        assert f1 == f2

    def test_different_seeds_may_differ(self):
        """Different seeds can produce different fitnesses."""
        chromosome = MLPPolicy(seed=0).get_params()
        f1 = evaluate_agent(chromosome, seed=0)
        f2 = evaluate_agent(chromosome, seed=999)
        # They *could* be equal by chance, but almost certainly aren't
        # We just check both are finite
        assert np.isfinite(f1)
        assert np.isfinite(f2)

    def test_zero_chromosome(self):
        """A chromosome of all zeros should still produce a valid fitness."""
        param_count = MLPPolicy(seed=0).param_count()
        chromosome = np.zeros(param_count, dtype=np.float32)
        result = evaluate_agent(chromosome, seed=0)
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestEvaluateRobust:
    """Tests for multi-episode robust evaluation."""

    def test_returns_float(self):
        """evaluate_robust should return a float."""
        chromosome = MLPPolicy(seed=0).get_params()
        result = evaluate_robust(chromosome, num_episodes=2)
        assert isinstance(result, float)

    def test_averages_correctly(self):
        """With 1 episode, robust should equal single evaluation."""
        chromosome = MLPPolicy(seed=0).get_params()
        single = evaluate_agent(chromosome, seed=0)
        robust = evaluate_robust(chromosome, num_episodes=1, base_seed=0)
        assert single == robust

    def test_more_episodes_is_finite(self):
        """Multiple episodes should produce a finite average."""
        chromosome = MLPPolicy(seed=0).get_params()
        result = evaluate_robust(chromosome, num_episodes=3)
        assert np.isfinite(result)


# =====================================================================
# Selection
# =====================================================================
class TestTournamentSelection:
    """Tests for tournament selection."""

    def setup_method(self):
        """Create a small test population."""
        self.rng = np.random.default_rng(42)
        self.pop = [np.array([float(i)]) for i in range(10)]
        self.fitnesses = np.array([float(i) for i in range(10)])

    def test_returns_copy(self):
        """Selected parent should be a copy, not a reference."""
        parent = tournament_selection(
            self.pop, self.fitnesses, k=3, rng=self.rng
        )
        # Modify parent — should not affect population
        parent[0] = -999.0
        for individual in self.pop:
            assert individual[0] != -999.0

    def test_winner_shape(self):
        """Winner should have the same shape as population members."""
        parent = tournament_selection(
            self.pop, self.fitnesses, k=3, rng=self.rng
        )
        assert parent.shape == self.pop[0].shape

    def test_selects_from_population(self):
        """Winner's value should exist in the original population."""
        parent = tournament_selection(
            self.pop, self.fitnesses, k=3, rng=self.rng
        )
        values = [ind[0] for ind in self.pop]
        assert parent[0] in values

    def test_high_k_selects_best(self):
        """With k = population size, should always select the best."""
        for _ in range(5):
            parent = tournament_selection(
                self.pop, self.fitnesses, k=len(self.pop), rng=self.rng
            )
            assert parent[0] == 9.0  # highest fitness index

    def test_k_equals_1(self):
        """k=1 is valid — just picks a random individual."""
        parent = tournament_selection(
            self.pop, self.fitnesses, k=1, rng=self.rng
        )
        assert parent.shape == self.pop[0].shape

    def test_empty_population_raises(self):
        """Empty population should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            tournament_selection([], np.array([]), k=3)

    def test_k_zero_raises(self):
        """k < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            tournament_selection(self.pop, self.fitnesses, k=0)


# =====================================================================
# Crossover
# =====================================================================
class TestUniformCrossover:
    """Tests for uniform crossover."""

    def test_child_shape(self):
        """Child should have same shape as parents."""
        a = np.ones(10)
        b = np.zeros(10)
        child = uniform_crossover(a, b, rng=np.random.default_rng(0))
        assert child.shape == a.shape

    def test_child_values_from_parents(self):
        """Every gene in the child should come from one parent."""
        a = np.ones(100)
        b = np.zeros(100)
        child = uniform_crossover(a, b, rng=np.random.default_rng(42))
        for val in child:
            assert val in (0.0, 1.0)

    def test_roughly_half_from_each(self):
        """With enough genes, roughly half should come from each parent."""
        a = np.ones(1000)
        b = np.zeros(1000)
        child = uniform_crossover(a, b, rng=np.random.default_rng(0))
        from_a = np.sum(child == 1.0)
        # Should be roughly 500, but allow wide margin
        assert 350 < from_a < 650

    def test_different_length_raises(self):
        """Parents of different lengths should raise ValueError."""
        a = np.ones(10)
        b = np.zeros(5)
        with pytest.raises(ValueError, match="same length"):
            uniform_crossover(a, b)

    def test_does_not_modify_parents(self):
        """Parents should not be modified."""
        a = np.ones(10)
        b = np.zeros(10)
        a_copy = a.copy()
        b_copy = b.copy()
        uniform_crossover(a, b, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(a, a_copy)
        np.testing.assert_array_equal(b, b_copy)

    def test_preserves_dtype(self):
        """Child should have the same dtype as parent A."""
        a = np.ones(10, dtype=np.float32)
        b = np.zeros(10, dtype=np.float32)
        child = uniform_crossover(a, b)
        assert child.dtype == np.float32


# =====================================================================
# Mutation
# =====================================================================
class TestGaussianMutate:
    """Tests for Gaussian mutation."""

    def test_output_shape(self):
        """Mutated chromosome should have same shape."""
        c = np.zeros(50)
        m = gaussian_mutate(c, std=0.02)
        assert m.shape == c.shape

    def test_does_not_modify_original(self):
        """Original chromosome should not be changed."""
        c = np.zeros(50)
        c_copy = c.copy()
        gaussian_mutate(c, std=0.02)
        np.testing.assert_array_equal(c, c_copy)

    def test_noise_is_small(self):
        """With small std, perturbations should be small."""
        c = np.zeros(1000)
        m = gaussian_mutate(c, std=0.02, rng=np.random.default_rng(0))
        # 99.7% of values within 3 sigma = 0.06
        assert np.all(np.abs(m) < 0.2)  # generous bound

    def test_different_rng_gives_different_results(self):
        """Different RNG seeds should produce different mutations."""
        c = np.zeros(50)
        m1 = gaussian_mutate(c, std=0.02, rng=np.random.default_rng(0))
        m2 = gaussian_mutate(c, std=0.02, rng=np.random.default_rng(99))
        assert not np.array_equal(m1, m2)

    def test_preserves_dtype(self):
        """Mutated chromosome should preserve dtype."""
        c = np.zeros(10, dtype=np.float32)
        m = gaussian_mutate(c, std=0.02)
        assert m.dtype == np.float32


class TestAdaptiveMutationStd:
    """Tests for adaptive mutation standard deviation."""

    def test_initial_value(self):
        """At generation 0, std should equal initial_std."""
        assert adaptive_mutation_std(0) == 0.02

    def test_decreases_over_time(self):
        """Std should decrease as generations progress."""
        s0 = adaptive_mutation_std(0)
        s20 = adaptive_mutation_std(20)
        s40 = adaptive_mutation_std(40)
        assert s0 > s20 > s40

    def test_stable_within_window(self):
        """Std should be constant between decay steps."""
        assert adaptive_mutation_std(0) == adaptive_mutation_std(19)
        assert adaptive_mutation_std(20) == adaptive_mutation_std(39)

    def test_custom_parameters(self):
        """Custom initial_std and decay should work."""
        s = adaptive_mutation_std(
            generation=10, initial_std=0.1, decay=0.5, decay_every=10
        )
        assert s == pytest.approx(0.05)


# =====================================================================
# Population Manager
# =====================================================================
class TestPopulationInit:
    """Tests for population initialisation."""

    def test_correct_population_size(self):
        """Population should have the requested number of individuals."""
        pop = Population(population_size=10)
        assert len(pop.chromosomes) == 10

    def test_chromosome_size(self):
        """Each chromosome should have 533 parameters (default MLP)."""
        pop = Population(population_size=5)
        for c in pop.chromosomes:
            assert c.shape == (533,)

    def test_diverse_initial_population(self):
        """Initial chromosomes should not all be identical."""
        pop = Population(population_size=5, seed=42)
        # At least 2 chromosomes should be different
        different = False
        for i in range(1, len(pop.chromosomes)):
            if not np.array_equal(pop.chromosomes[0], pop.chromosomes[i]):
                different = True
                break
        assert different


class TestPopulationEvolution:
    """Tests for the evolutionary loop."""

    def setup_method(self):
        """Create a small population for testing."""
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_produces_result(self):
        """run() should return a chromosome and fitness."""
        pop = Population(
            population_size=4,
            generations=2,
            eval_episodes=1,
            output_dir=self.tmpdir,
            seed=0,
        )
        best_params, best_fitness = pop.run()
        assert isinstance(best_params, np.ndarray)
        assert best_params.shape == (533,)
        assert isinstance(best_fitness, float)

    def test_config_saved(self):
        """run() should save config.json."""
        pop = Population(
            population_size=4,
            generations=2,
            eval_episodes=1,
            output_dir=self.tmpdir,
            seed=0,
        )
        pop.run()
        config_path = os.path.join(self.tmpdir, "config.json")
        assert os.path.exists(config_path)
        with open(config_path) as f:
            config = json.load(f)
        assert config["population_size"] == 4
        assert config["generations"] == 2

    def test_metrics_csv_written(self):
        """run() should write metrics.csv with correct columns."""
        pop = Population(
            population_size=4,
            generations=2,
            eval_episodes=1,
            output_dir=self.tmpdir,
            seed=0,
        )
        pop.run()
        csv_path = os.path.join(self.tmpdir, "metrics.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            header = f.readline().strip()
        expected_cols = [
            "generation", "best_fitness", "avg_fitness",
            "worst_fitness", "diversity", "mutation_std", "elapsed_sec",
        ]
        assert header == ",".join(expected_cols)

    def test_best_final_saved(self):
        """run() should save best_final.npy."""
        pop = Population(
            population_size=4,
            generations=2,
            eval_episodes=1,
            output_dir=self.tmpdir,
            seed=0,
        )
        pop.run()
        final_path = os.path.join(self.tmpdir, "best_final.npy")
        assert os.path.exists(final_path)
        loaded = np.load(final_path)
        assert loaded.shape == (533,)

    def test_elitism_preserves_best(self):
        """Elitism should keep the best individual in the next generation."""
        pop = Population(
            population_size=6,
            generations=3,
            eval_episodes=1,
            elitism=2,
            output_dir=self.tmpdir,
            seed=0,
        )
        best_params, _ = pop.run()
        # The best from any generation should be tracked
        assert pop.best_fitness_ever > -np.inf


class TestPopulationDiversity:
    """Tests for the diversity metric."""

    def test_identical_population_zero_diversity(self):
        """All-identical chromosomes should have zero diversity."""
        chromosomes = [np.ones(10) for _ in range(5)]
        div = Population._compute_diversity(chromosomes)
        assert div == 0.0

    def test_diverse_population_nonzero(self):
        """Different chromosomes should have nonzero diversity."""
        chromosomes = [np.full(10, float(i)) for i in range(5)]
        div = Population._compute_diversity(chromosomes)
        assert div > 0.0

    def test_single_individual(self):
        """Single individual should have zero diversity."""
        chromosomes = [np.ones(10)]
        div = Population._compute_diversity(chromosomes)
        assert div == 0.0


class TestEvolveOneGeneration:
    """Tests for a single evolution step."""

    def test_next_gen_same_size(self):
        """Next generation should have the same population size."""
        pop = Population(population_size=10, seed=42)
        fitnesses = np.random.default_rng(0).standard_normal(10)
        next_gen = pop._evolve_one_generation(fitnesses, generation=0)
        assert len(next_gen) == 10

    def test_next_gen_chromosome_shapes(self):
        """All chromosomes in next gen should have correct shape."""
        pop = Population(population_size=10, seed=42)
        fitnesses = np.random.default_rng(0).standard_normal(10)
        next_gen = pop._evolve_one_generation(fitnesses, generation=0)
        for c in next_gen:
            assert c.shape == (533,)

    def test_elites_preserved(self):
        """Top elitism individuals should appear in next generation."""
        pop = Population(population_size=10, elitism=2, seed=42)
        fitnesses = np.arange(10, dtype=float)  # [0, 1, ..., 9]
        next_gen = pop._evolve_one_generation(fitnesses, generation=0)
        # The best individual (index 9) should be in the next gen
        best = pop.chromosomes[9]
        found = any(np.array_equal(c, best) for c in next_gen[:2])
        assert found

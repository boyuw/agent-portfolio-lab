"""Tests for portfolio_lab. Statistical tests use fixed seeds and loose
tolerances: they check the direction of effects, not exact values."""

import random

import pytest

from portfolio_lab.allocate import greedy_allocation, uniform_allocation
from portfolio_lab.metrics import odds_lift, run_experiment
from portfolio_lab.simulate import generate_candidates

SEED = 7
BUDGET = 12


@pytest.fixture(scope="module")
def candidates():
    return generate_candidates(5, random.Random(SEED))


@pytest.mark.parametrize("budget", [0, 1, 7, 12, 25])
def test_allocations_sum_to_budget(candidates, budget):
    assert sum(uniform_allocation(len(candidates), budget)) == budget
    assert sum(greedy_allocation(candidates, budget)) == budget
    assert len(greedy_allocation(candidates, budget)) == len(candidates)


def test_odds_lift_formula():
    assert odds_lift(0.5, 0.5) == pytest.approx(1.0)
    assert odds_lift(0.75, 0.5) == pytest.approx(3.0)
    assert odds_lift(1.0, 0.5) == float("inf")


def test_selector_at_chance_has_lift_near_one(candidates):
    allocation = greedy_allocation(candidates, BUDGET)
    result = run_experiment(candidates, allocation, skill=0.0, n_tasks=3000, seed=1)
    assert result.portfolio_accuracy == pytest.approx(result.chance_accuracy, abs=0.03)
    assert result.lift == pytest.approx(1.0, abs=0.25)


def test_skilled_selector_has_lift_above_one(candidates):
    allocation = greedy_allocation(candidates, BUDGET)
    result = run_experiment(candidates, allocation, skill=0.9, n_tasks=3000, seed=2)
    assert result.lift > 1.3


def test_portfolio_beats_best_single_with_decent_selector(candidates):
    allocation = greedy_allocation(candidates, BUDGET)
    result = run_experiment(candidates, allocation, skill=0.85, n_tasks=3000, seed=3)
    assert result.portfolio_accuracy >= result.best_single - 0.02


def test_oracle_bounds_portfolio(candidates):
    allocation = uniform_allocation(len(candidates), BUDGET)
    result = run_experiment(candidates, allocation, skill=0.8, n_tasks=1000, seed=4)
    assert result.oracle_accuracy >= result.portfolio_accuracy - 0.02


def test_deterministic_under_fixed_seed(candidates):
    allocation = uniform_allocation(len(candidates), BUDGET)
    first = run_experiment(candidates, allocation, skill=0.8, n_tasks=200, seed=5)
    second = run_experiment(candidates, allocation, skill=0.8, n_tasks=200, seed=5)
    assert first == second

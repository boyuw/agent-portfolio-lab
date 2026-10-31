"""Tests for the LP-relaxation allocator stub.

The stub maximises the same objective as greedy, but on fractional run
counts; tests check the structural properties (sums, determinism, edge
cases) and that both solutions land near the same objective value, not
exact allocations, which depend on the step size.
"""

import random

import pytest

from portfolio_lab.allocate import (
    fractional_allocation,
    greedy_allocation,
    lp_relaxation_allocation,
)
from portfolio_lab.simulate import generate_candidates

SEED = 7
BUDGET = 12


@pytest.fixture(scope="module")
def candidates():
    return generate_candidates(5, random.Random(SEED))


def _objective(candidates, allocation):
    """P(at least one correct run), averaged over task types."""
    n_types = candidates[0].n_types
    total = 0.0
    for t in range(n_types):
        failure = 1.0
        for candidate, count in zip(candidates, allocation):
            failure *= (1.0 - candidate.success_prob(t)) ** count
        total += 1.0 - failure
    return total / n_types


@pytest.mark.parametrize("budget", [0, 1, 7, 12, 25])
def test_lp_allocation_sums_to_budget(candidates, budget):
    allocation = lp_relaxation_allocation(candidates, budget)
    assert len(allocation) == len(candidates)
    assert all(count >= 0 for count in allocation)
    assert sum(allocation) == budget


@pytest.mark.parametrize("budget", [0, 1, 7, 12, 25])
def test_fractional_allocation_sums_to_budget(candidates, budget):
    allocation = fractional_allocation(candidates, budget)
    assert len(allocation) == len(candidates)
    assert all(x >= 0.0 for x in allocation)
    assert sum(allocation) == pytest.approx(float(budget))


def test_fractional_objective_at_least_greedy(candidates):
    # The fractional optimum upper-bounds the integer optimum; the greedy
    # fractional approximation should stay within a step-size tolerance.
    fractional = fractional_allocation(candidates, BUDGET)
    greedy = greedy_allocation(candidates, BUDGET)
    assert _objective(candidates, fractional) >= _objective(candidates, greedy) - 0.01


def test_lp_integer_objective_near_greedy(candidates):
    # Largest-remainder rounding of a near-optimal fractional solution
    # should land close to the integer greedy solution on the same objective.
    lp = lp_relaxation_allocation(candidates, BUDGET)
    greedy = greedy_allocation(candidates, BUDGET)
    assert _objective(candidates, lp) >= _objective(candidates, greedy) - 0.02


def test_lp_allocation_deterministic(candidates):
    first = lp_relaxation_allocation(candidates, BUDGET)
    second = lp_relaxation_allocation(candidates, BUDGET)
    assert first == second
    assert fractional_allocation(candidates, BUDGET) == fractional_allocation(
        candidates, BUDGET
    )


def test_single_workflow_gets_whole_budget():
    candidates = generate_candidates(1, random.Random(3))
    assert lp_relaxation_allocation(candidates, BUDGET) == [BUDGET]
    assert fractional_allocation(candidates, BUDGET) == pytest.approx([float(BUDGET)])


def test_zero_budget_is_all_zeros(candidates):
    assert lp_relaxation_allocation(candidates, 0) == [0] * len(candidates)
    assert fractional_allocation(candidates, 0) == [0.0] * len(candidates)


def test_invalid_inputs_raise(candidates):
    with pytest.raises(ValueError):
        lp_relaxation_allocation(candidates, -1)
    with pytest.raises(ValueError):
        fractional_allocation([], BUDGET)
    with pytest.raises(ValueError):
        fractional_allocation(candidates, BUDGET, step=0.0)

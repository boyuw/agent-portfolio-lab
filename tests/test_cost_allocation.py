"""Tests for cost-weighted budget allocation.

With per-run prices, the greedy rule buys gain per unit cost instead of
gain per run. Tests check the budget constraint, the equal-cost
reduction to run-count greedy, the cheap-vs-expensive trade-off on
hand-built candidates, and input validation.
"""

import random

import pytest

from portfolio_lab.allocate import (
    allocation_cost,
    greedy_allocation,
    greedy_cost_allocation,
)
from portfolio_lab.simulate import Candidate, generate_candidates

SEED = 7


@pytest.fixture(scope="module")
def candidates():
    return generate_candidates(5, random.Random(SEED))


def test_cost_never_exceeds_budget(candidates):
    costs = [1.0, 3.0, 2.0, 1.0, 4.0]
    for budget in [0.0, 1.0, 5.5, 12.0, 24.0, 100.0]:
        allocation = greedy_cost_allocation(candidates, costs, budget)
        assert len(allocation) == len(candidates)
        assert all(count >= 0 for count in allocation)
        assert allocation_cost(allocation, costs) <= budget + 1e-9


def test_leftover_is_less_than_cheapest_price(candidates):
    # Greedy stops only when nothing affordable remains, so the unspent
    # remainder is smaller than the cheapest workflow's price.
    costs = [2.0, 3.0, 5.0]
    three = candidates[:3]
    allocation = greedy_cost_allocation(three, costs, 20.0)
    spent = allocation_cost(allocation, costs)
    assert 20.0 - spent < min(costs)


def test_equal_costs_reduce_to_run_count_greedy(candidates):
    # Same price everywhere: gain per cost orders workflows exactly as
    # gain does, so a cost budget of 12 runs' worth must reproduce the
    # run-count greedy allocation for 12 runs.
    costs = [2.0] * len(candidates)
    assert greedy_cost_allocation(candidates, costs, 24.0) == greedy_allocation(
        candidates, 12
    )


def test_cheap_workflow_wins_on_gain_per_cost():
    # One type, so gains are directly comparable: the cheap workflow's
    # gain per cost (0.5) beats the expensive one's (0.09) at every
    # step, so the whole budget goes to the cheap workflow even though
    # the expensive one is more accurate per run.
    candidates = [
        Candidate(name="cheap", base_prob=0.5, type_bonus=(0.0,)),
        Candidate(name="pricey", base_prob=0.9, type_bonus=(0.0,)),
    ]
    assert greedy_cost_allocation(candidates, [1.0, 10.0], 10.0) == [10, 0]


def test_expensive_workflow_wins_when_worth_its_price():
    # Here the expensive workflow's gain per cost (0.97 / 2 = 0.485)
    # dwarfs the cheap one's (0.01), so it takes the whole budget.
    candidates = [
        Candidate(name="cheap", base_prob=0.01, type_bonus=(0.0,)),
        Candidate(name="pricey", base_prob=0.97, type_bonus=(0.0,)),
    ]
    assert greedy_cost_allocation(candidates, [1.0, 2.0], 2.0) == [0, 1]


def test_allocation_cost_sums_counts_times_prices():
    assert allocation_cost([3, 0, 2], [1.5, 10.0, 2.0]) == pytest.approx(8.5)


def test_cost_allocation_rejects_bad_inputs(candidates):
    with pytest.raises(ValueError):
        greedy_cost_allocation([], [1.0], 10.0)
    with pytest.raises(ValueError):
        greedy_cost_allocation(candidates, [1.0], 10.0)
    with pytest.raises(ValueError):
        greedy_cost_allocation(candidates, [1.0] * 4 + [0.0], 10.0)
    with pytest.raises(ValueError):
        greedy_cost_allocation(candidates, [1.0] * 4 + [-2.0], 10.0)
    with pytest.raises(ValueError):
        greedy_cost_allocation(candidates, [1.0] * 5, -1.0)
    with pytest.raises(ValueError):
        allocation_cost([1, 2], [1.0])

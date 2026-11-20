"""Tests for the multi-seed sweep in portfolio_lab.sweep."""

import random

import pytest

from portfolio_lab.allocate import greedy_allocation
from portfolio_lab.metrics import run_experiment
from portfolio_lab.simulate import generate_candidates
from portfolio_lab.sweep import quantile, summarize_values, sweep_seeds


def test_quantile_endpoints_and_median():
    values = [0.0, 10.0, 20.0, 30.0]
    assert quantile(values, 0.0) == pytest.approx(0.0)
    assert quantile(values, 1.0) == pytest.approx(30.0)
    assert quantile(values, 0.5) == pytest.approx(15.0)
    assert quantile(values, 0.25) == pytest.approx(7.5)
    assert quantile(values, 0.75) == pytest.approx(22.5)
    assert quantile([5.0], 0.5) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        quantile([], 0.5)
    with pytest.raises(ValueError):
        quantile(values, -0.1)
    with pytest.raises(ValueError):
        quantile(values, 1.1)


def test_summarize_values_known_case():
    summary = summarize_values([1.0, 2.0, 3.0, 4.0])
    assert summary.n == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.std == pytest.approx(1.2909944487358056)
    assert summary.median == pytest.approx(2.5)
    assert summary.minimum == pytest.approx(1.0)
    assert summary.maximum == pytest.approx(4.0)
    assert summary.q10 == pytest.approx(1.3)
    assert summary.q25 == pytest.approx(1.75)
    assert summary.q75 == pytest.approx(3.25)
    assert summary.q90 == pytest.approx(3.7)


def test_summarize_values_single_value():
    summary = summarize_values([7.0])
    assert summary.n == 1
    assert summary.mean == pytest.approx(7.0)
    assert summary.std == pytest.approx(0.0)
    assert summary.median == pytest.approx(7.0)
    assert summary.q10 == pytest.approx(7.0)
    assert summary.q90 == pytest.approx(7.0)
    with pytest.raises(ValueError):
        summarize_values([])


def _demo_setup():
    candidates = generate_candidates(5, random.Random(7))
    greedy = greedy_allocation(candidates, 12)
    return candidates, greedy


def test_sweep_matches_single_seed_experiment():
    candidates, greedy = _demo_setup()
    sweep = sweep_seeds(
        candidates, greedy, skill=0.8, n_tasks=2000, seeds=[7]
    )
    direct = run_experiment(candidates, greedy, skill=0.8, n_tasks=2000, seed=7)
    assert sweep.lifts == [direct.lift]
    assert sweep.accuracies == [direct.portfolio_accuracy]
    assert sweep.seeds == [7]
    assert sweep.lift_summary.n == 1
    assert sweep.lift_summary.mean == pytest.approx(direct.lift)


def test_sweep_is_deterministic_and_ordered():
    candidates, greedy = _demo_setup()
    seeds = [0, 1, 2, 3]
    first = sweep_seeds(
        candidates, greedy, skill=0.8, n_tasks=2000, seeds=seeds
    )
    second = sweep_seeds(
        candidates, greedy, skill=0.8, n_tasks=2000, seeds=seeds
    )
    assert first.lifts == second.lifts
    assert first.accuracies == second.accuracies
    # The summary is computed from the same lifts, in seed order.
    assert first.lift_summary.minimum == pytest.approx(min(first.lifts))
    assert first.lift_summary.maximum == pytest.approx(max(first.lifts))
    assert first.lift_summary.median == pytest.approx(
        quantile(sorted(first.lifts), 0.5)
    )


def test_sweep_validation():
    candidates, greedy = _demo_setup()
    with pytest.raises(ValueError):
        sweep_seeds(candidates, greedy, skill=0.8, n_tasks=2000, seeds=[])
    with pytest.raises(ValueError):
        sweep_seeds(
            candidates, greedy, skill=0.8, n_tasks=2000, seeds=[1, 2, 1]
        )


def test_sweep_distribution_on_demo_config():
    candidates, greedy = _demo_setup()
    sweep = sweep_seeds(
        candidates, greedy, skill=0.8, n_tasks=2000, seeds=list(range(20))
    )
    lift = sweep.lift_summary
    assert lift.n == 20
    # The headline single-seed lift (seed 7: 8.87) sits inside the
    # distribution, not at an extreme; the spread is Monte Carlo error
    # on 2000 tasks per seed, not a configuration effect.
    assert 7.5 < lift.mean < 9.5
    assert lift.minimum > 5.0
    assert lift.maximum < 15.0
    assert 6.0 < lift.q10 < 9.0
    assert 8.0 < lift.q90 < 11.0
    assert 0.0 < lift.std < 2.0
    assert lift.q10 <= lift.median <= lift.q90
    accuracy = sweep.accuracy_summary
    assert 0.85 < accuracy.mean < 0.93
    assert all(0.0 < a < 1.0 for a in sweep.accuracies)
    assert all(1.0 < v for v in sweep.lifts)

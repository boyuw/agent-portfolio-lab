"""Edge-case tests: single workflow, zero budget, and selector extremes.

These pin down boundary behaviour that the main suite only exercises at
interior values, so a future refactor cannot silently change it.
"""

import random

import pytest

from portfolio_lab.allocate import greedy_allocation, uniform_allocation
from portfolio_lab.metrics import odds_lift, run_experiment
from portfolio_lab.selector import select_run
from portfolio_lab.simulate import Candidate, RunResult, generate_candidates


def test_single_workflow_allocations_take_the_whole_budget():
    assert uniform_allocation(1, 12) == [12]
    single = generate_candidates(1, random.Random(11))
    assert greedy_allocation(single, 12) == [12]


def test_single_workflow_repeated_runs_still_help_selector():
    """With one workflow, chance accuracy is just that workflow's expected
    accuracy, but a perfect selector still beats chance: repeated runs give
    it several noisy signals to choose from, and the signal correlates with
    correctness. It cannot beat the oracle, which knows correctness."""
    single = generate_candidates(1, random.Random(11))
    result = run_experiment(single, [12], skill=1.0, n_tasks=3000, seed=12)
    assert result.best_single == pytest.approx(result.chance_accuracy, abs=0.03)
    assert result.portfolio_accuracy > result.chance_accuracy
    assert result.portfolio_accuracy <= result.oracle_accuracy + 0.02


def test_zero_budget_allocates_nothing_and_experiment_raises():
    candidates = generate_candidates(3, random.Random(13))
    assert uniform_allocation(3, 0) == [0, 0, 0]
    assert greedy_allocation(candidates, 0) == [0, 0, 0]
    # No runs exist to select from, so the experiment cannot proceed.
    with pytest.raises(ValueError, match="empty run set"):
        run_experiment(candidates, [0, 0, 0], skill=0.8, n_tasks=10, seed=13)


def test_allocation_errors():
    with pytest.raises(ValueError, match="at least one workflow"):
        uniform_allocation(0, 5)
    with pytest.raises(ValueError, match="non-negative"):
        uniform_allocation(3, -1)
    with pytest.raises(ValueError, match="non-negative"):
        greedy_allocation(generate_candidates(3, random.Random(14)), -1)


def test_perfect_selector_picks_highest_signal():
    runs = [
        RunResult(candidate_index=0, correct=False, signal=0.2),
        RunResult(candidate_index=1, correct=True, signal=1.4),
        RunResult(candidate_index=2, correct=False, signal=-0.3),
    ]
    chosen = select_run(runs, skill=1.0, rng=random.Random(15))
    assert chosen.candidate_index == 1


def test_selector_rejects_empty_runs_and_out_of_range_skill():
    with pytest.raises(ValueError, match="empty run set"):
        select_run([], skill=0.5, rng=random.Random(16))
    run = [RunResult(candidate_index=0, correct=True, signal=1.0)]
    for bad_skill in (-0.1, 1.1):
        with pytest.raises(ValueError, match="skill must be in"):
            select_run(run, skill=bad_skill, rng=random.Random(16))


def test_perfect_selector_beats_chance_but_not_oracle():
    candidates = generate_candidates(5, random.Random(7))
    allocation = greedy_allocation(candidates, 12)
    result = run_experiment(candidates, allocation, skill=1.0, n_tasks=3000, seed=17)
    assert result.portfolio_accuracy >= result.chance_accuracy - 0.02
    assert result.portfolio_accuracy <= result.oracle_accuracy + 0.02


def test_chance_selector_matches_chance_accuracy_with_one_run():
    """Budget 1 leaves a single run per task, so any selector must pick it:
    portfolio accuracy equals chance accuracy exactly."""
    candidates = generate_candidates(5, random.Random(7))
    result = run_experiment(candidates, [1, 0, 0, 0, 0], skill=1.0, n_tasks=500, seed=18)
    assert result.portfolio_accuracy == result.chance_accuracy


def test_success_probability_is_clamped():
    assert Candidate("hi", 0.99, (0.5,)).success_prob(0) == pytest.approx(0.97)
    assert Candidate("lo", 0.0, (-0.5,)).success_prob(0) == pytest.approx(0.01)


def test_odds_lift_boundaries():
    assert odds_lift(0.0, 0.5) == 0.0
    assert odds_lift(1.0, 0.5) == float("inf")
    with pytest.raises(ValueError, match="chance_accuracy must be in"):
        odds_lift(0.5, 1.0)


def test_single_generated_candidate_has_zero_mean_bonus():
    (candidate,) = generate_candidates(1, random.Random(19))
    assert sum(candidate.type_bonus) == pytest.approx(0.0)

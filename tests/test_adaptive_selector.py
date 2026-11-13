"""Tests for the adaptive selector: skill estimated from held-out picks.

Instead of assuming the selector's skill, `estimate_skill` recovers it
from held-out tasks where the pick can be scored: skill is the
method-of-moments ratio (observed pick accuracy - chance accuracy) /
(max-signal accuracy - chance accuracy), clamped to [0, 1]. Tests
check the exact ratio on constructed observations, the clamping and
degenerate cases, input validation, and that the estimator recovers
the true skill from simulated held-out tasks.
"""

import random

import pytest

from portfolio_lab.allocate import greedy_allocation
from portfolio_lab.selector import (
    HeldOutObservation,
    collect_observations,
    estimate_skill,
)
from portfolio_lab.simulate import RunResult, generate_candidates

SEED = 7
N_TASKS = 2000
SKILL = 0.8
HELD_OUT_SEED = 1007


@pytest.fixture(scope="module")
def candidates():
    return generate_candidates(5, random.Random(SEED))


@pytest.fixture(scope="module")
def allocation(candidates):
    return greedy_allocation(candidates, 12)


def _two_run_task(picked_correct: bool) -> HeldOutObservation:
    """One task with two runs: the max-signal run is correct, the other
    is not. Chance accuracy is 0.5, max-signal accuracy is 1.0, so the
    estimate is 2 * observed accuracy - 1."""
    good = RunResult(candidate_index=0, correct=True, signal=1.0)
    bad = RunResult(candidate_index=1, correct=False, signal=0.0)
    picked = good if picked_correct else bad
    return HeldOutObservation(runs=[good, bad], picked=picked)


def test_estimate_skill_exact_on_constructed_observations():
    # 10 tasks, the selector picked the correct (max-signal) run on 8:
    # observed accuracy 0.8, chance 0.5, max-signal 1.0, so the
    # estimate is (0.8 - 0.5) / (1.0 - 0.5) = 0.6 exactly.
    observations = [_two_run_task(True) for _ in range(8)]
    observations += [_two_run_task(False) for _ in range(2)]
    assert estimate_skill(observations) == pytest.approx(0.6)
    # Picking the max-signal run every time is skill 1 by construction.
    assert estimate_skill([_two_run_task(True) for _ in range(10)]) == pytest.approx(
        1.0
    )
    # Picking the wrong run every time is skill 0 (observed 0.0 is
    # below chance 0.5, so the ratio is negative and clamps to 0).
    assert estimate_skill([_two_run_task(False) for _ in range(10)]) == pytest.approx(
        0.0
    )


def test_estimate_skill_clamps_above_one():
    # The selector picked a correct run on all 4 tasks, but the
    # max-signal run was correct on only 3 of them (on the fourth it
    # picked a correct non-max run): observed 1.0, chance 0.5,
    # max-signal 0.75, so the raw ratio is 2.0 and clamps to 1.0.
    good = RunResult(candidate_index=0, correct=True, signal=1.0)
    bad = RunResult(candidate_index=1, correct=False, signal=0.0)
    observations = [HeldOutObservation(runs=[good, bad], picked=good)] * 3
    flipped_good = RunResult(candidate_index=0, correct=False, signal=1.0)
    flipped_bad = RunResult(candidate_index=1, correct=True, signal=0.0)
    observations.append(
        HeldOutObservation(runs=[flipped_good, flipped_bad], picked=flipped_bad)
    )
    assert estimate_skill(observations) == pytest.approx(1.0)


def test_estimate_skill_zero_when_max_signal_is_no_better_than_chance():
    # Every run correct: max-signal accuracy and chance accuracy are
    # both 1.0, so the denominator is 0 and skill is unidentified.
    good = RunResult(candidate_index=0, correct=True, signal=1.0)
    also_good = RunResult(candidate_index=1, correct=True, signal=0.0)
    observations = [HeldOutObservation(runs=[good, also_good], picked=good)] * 5
    assert estimate_skill(observations) == 0.0
    # Every run wrong: same degeneracy at 0.0.
    bad = RunResult(candidate_index=0, correct=False, signal=1.0)
    also_bad = RunResult(candidate_index=1, correct=False, signal=0.0)
    observations = [HeldOutObservation(runs=[bad, also_bad], picked=bad)] * 5
    assert estimate_skill(observations) == 0.0


def test_estimate_skill_validation():
    with pytest.raises(ValueError):
        estimate_skill([])
    good = RunResult(candidate_index=0, correct=True, signal=1.0)
    bad = RunResult(candidate_index=1, correct=False, signal=0.0)
    # Picked run is not one of the run set.
    with pytest.raises(ValueError):
        estimate_skill(
            [HeldOutObservation(runs=[good], picked=bad)]
        )
    # Empty run set.
    with pytest.raises(ValueError):
        estimate_skill([HeldOutObservation(runs=[], picked=good)])


def test_collect_observations_records_picks(candidates, allocation):
    observations = collect_observations(
        candidates, allocation, skill=SKILL, n_tasks=50, seed=HELD_OUT_SEED
    )
    assert len(observations) == 50
    for observation in observations:
        assert observation.picked in observation.runs
        assert observation.runs


def test_estimate_skill_recovers_true_skill(candidates, allocation):
    # Held-out recovery at seed 1007, 2000 tasks: 0.8 -> 0.7897,
    # 0.5 -> 0.5039, 0.0 -> 0.0094, 1.0 -> 1.0000. Tolerances sit
    # well outside the observed errors.
    for true_skill, low, high in (
        (0.0, 0.0, 0.05),
        (0.5, 0.45, 0.55),
        (0.8, 0.75, 0.85),
        (1.0, 0.95, 1.0),
    ):
        observations = collect_observations(
            candidates,
            allocation,
            skill=true_skill,
            n_tasks=N_TASKS,
            seed=HELD_OUT_SEED,
        )
        estimate = estimate_skill(observations)
        assert low <= estimate <= high, f"true skill {true_skill}: got {estimate}"


def test_estimate_skill_recovers_true_skill_on_another_seed(
    candidates, allocation
):
    # Same recovery on a different held-out seed (0.8 -> 0.8089), so
    # the estimator is not tuned to one lucky seed.
    observations = collect_observations(
        candidates, allocation, skill=SKILL, n_tasks=N_TASKS, seed=1234
    )
    assert estimate_skill(observations) == pytest.approx(SKILL, abs=0.05)

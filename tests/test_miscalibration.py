"""Tests for selector miscalibration (overstated confidence).

The selector claims a skill but behaves with a lower effective skill:
effective = claimed * (1 - overconfidence). Tests check input
validation, the endpoints (honest claim = unchanged behaviour,
total overconfidence = pure chance), that overconfidence = 0
reproduces the calibrated experiment exactly, and that the lift
degrades monotonically as overconfidence grows.
"""

import random

import pytest

from portfolio_lab.allocate import greedy_allocation
from portfolio_lab.metrics import run_experiment
from portfolio_lab.selector import effective_skill
from portfolio_lab.simulate import generate_candidates

SEED = 7
N_TASKS = 2000
SKILL = 0.8


@pytest.fixture(scope="module")
def candidates():
    return generate_candidates(5, random.Random(SEED))


@pytest.fixture(scope="module")
def allocation(candidates):
    return greedy_allocation(candidates, 12)


def test_effective_skill_endpoints_and_linearity():
    # Honest claim: behaviour unchanged. Total overconfidence: chance.
    assert effective_skill(0.8, 0.0) == 0.8
    assert effective_skill(0.8, 1.0) == 0.0
    # A claimed skill of 0 stays 0: an honest chance-level selector
    # cannot be overconfident about being better than chance.
    assert effective_skill(0.0, 1.0) == 0.0
    # Overconfidence scales the claimed skill down linearly.
    assert effective_skill(0.8, 0.5) == pytest.approx(0.4)
    assert effective_skill(0.8, 0.25) == pytest.approx(0.6)
    # Monotone: more overconfidence never means more effective skill.
    assert effective_skill(0.8, 0.75) < effective_skill(0.8, 0.5)


def test_effective_skill_validation():
    for bad_claimed in (-0.1, 1.1):
        with pytest.raises(ValueError):
            effective_skill(bad_claimed, 0.5)
    for bad_overconfidence in (-0.1, 1.1):
        with pytest.raises(ValueError):
            effective_skill(0.8, bad_overconfidence)
    with pytest.raises(ValueError):
        run_experiment(
            generate_candidates(5, random.Random(0)),
            [3, 3, 2, 2, 2],
            skill=SKILL,
            n_tasks=10,
            seed=0,
            overconfidence=1.5,
        )


def test_overconfidence_zero_reproduces_calibrated_experiment(
    candidates, allocation
):
    # overconfidence = 0 is pure arithmetic on the skill (no extra RNG
    # draws), so the experiment must be bit-identical to the default.
    default = run_experiment(
        candidates, allocation, skill=SKILL, n_tasks=N_TASKS, seed=SEED
    )
    honest = run_experiment(
        candidates,
        allocation,
        skill=SKILL,
        n_tasks=N_TASKS,
        seed=SEED,
        overconfidence=0.0,
    )
    assert honest == default


def test_lift_degrades_with_overconfidence(candidates, allocation):
    # Demo numbers at seed 7: lift 8.87 -> 3.81 -> 2.14 -> 1.45 as
    # overconfidence goes 0 -> 0.25 -> 0.5 -> 0.75. Assert the erosion
    # with margins well inside the observed gaps.
    lifts = {}
    accuracies = {}
    for overconfidence in (0.0, 0.25, 0.5, 0.75):
        result = run_experiment(
            candidates,
            allocation,
            skill=SKILL,
            n_tasks=N_TASKS,
            seed=SEED,
            overconfidence=overconfidence,
        )
        lifts[overconfidence] = result.lift
        accuracies[overconfidence] = result.portfolio_accuracy
    assert lifts[0.25] < lifts[0.0] - 1.0
    assert lifts[0.5] < lifts[0.25] - 0.5
    assert lifts[0.75] < lifts[0.5] - 0.25
    assert accuracies[0.0] > accuracies[0.25] > accuracies[0.5] > accuracies[0.75]


def test_total_overconfidence_is_chance(candidates, allocation):
    # Overconfidence 1 makes the selector pick uniformly at random, so
    # its accuracy must match the chance baseline and the lift must be
    # ~1 (both within Monte Carlo noise at 5000 tasks: demo numbers
    # 0.4882 vs 0.5027, lift 0.944).
    result = run_experiment(
        candidates,
        allocation,
        skill=SKILL,
        n_tasks=5000,
        seed=SEED,
        overconfidence=1.0,
    )
    assert result.portfolio_accuracy == pytest.approx(
        result.chance_accuracy, abs=0.04
    )
    assert 0.7 < result.lift < 1.4

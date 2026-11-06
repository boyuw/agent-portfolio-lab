"""Tests for correlated errors (common-shock model).

Task outcomes share a Gaussian common shock, so workflow errors on the
same task correlate with strength rho. Tests check input validation,
that rho = 0 reproduces the independent-runs experiment exactly, that
the copula preserves each workflow's marginal success probability,
that correctness is nested by success probability at rho = 1, and that
the oracle ceiling collapses toward the best-per-type ceiling as rho
grows — the diversification lift erodes.
"""

import random

import pytest

from portfolio_lab.allocate import greedy_allocation
from portfolio_lab.metrics import run_experiment
from portfolio_lab.simulate import (
    Candidate,
    generate_candidates,
    generate_task,
    run_once,
    run_portfolio,
)

SEED = 7
N_TASKS = 2000
SKILL = 0.8


@pytest.fixture(scope="module")
def candidates():
    return generate_candidates(5, random.Random(SEED))


def test_rho_validation():
    candidates = [Candidate(name="w", base_prob=0.5, type_bonus=(0.0,))]
    task = generate_task(random.Random(0))
    rng = random.Random(0)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            run_once(0, candidates[0], task, rng, rho=bad)
        with pytest.raises(ValueError):
            run_portfolio(candidates, [1], task, rng, rho=bad)
        with pytest.raises(ValueError):
            run_experiment(
                candidates, [1], skill=SKILL, n_tasks=10, seed=0, rho=bad
            )


def test_rho_zero_reproduces_independent_experiment(candidates):
    # rho = 0 takes the original Bernoulli code path, so the experiment
    # must be bit-identical to the default (no rho argument) run.
    allocation = greedy_allocation(candidates, 12)
    default = run_experiment(
        candidates, allocation, skill=SKILL, n_tasks=N_TASKS, seed=SEED
    )
    explicit = run_experiment(
        candidates, allocation, skill=SKILL, n_tasks=N_TASKS, seed=SEED, rho=0.0
    )
    assert explicit == default


def test_correlation_preserves_marginals(candidates):
    # Correlation changes joint outcomes, not marginals: each workflow's
    # empirical accuracy at rho = 0.5 stays near its expected accuracy.
    rng = random.Random(SEED)
    allocation = [2, 2, 2, 2, 2]
    correct = [0] * len(candidates)
    runs_seen = [0] * len(candidates)
    for _ in range(N_TASKS):
        task = generate_task(rng, 3)
        for run in run_portfolio(candidates, allocation, task, rng, rho=0.5):
            runs_seen[run.candidate_index] += 1
            correct[run.candidate_index] += run.correct
    for candidate, n, k in zip(candidates, runs_seen, correct):
        assert k / n == pytest.approx(candidate.expected_accuracy(), abs=0.03)


def test_correctness_nested_by_success_probability_at_rho_one():
    # At rho = 1 the latent draw is the shared shock alone, so on one
    # task correctness is nested: a lower-p workflow being correct
    # implies every higher-p workflow is correct too.
    candidates = [
        Candidate(name="low", base_prob=0.3, type_bonus=(0.0,)),
        Candidate(name="mid", base_prob=0.6, type_bonus=(0.0,)),
        Candidate(name="high", base_prob=0.9, type_bonus=(0.0,)),
    ]
    rng = random.Random(SEED)
    for _ in range(200):
        task = generate_task(rng, 1)
        runs = run_portfolio(candidates, [1, 1, 1], task, rng, rho=1.0)
        flags = [r.correct for r in runs]  # ordered low, mid, high by p
        # No (True, False) pair when ordered by ascending success prob:
        # correctness can only switch from False to True as p grows.
        assert flags == sorted(flags)


def test_oracle_collapses_toward_best_per_type(candidates):
    # Independent runs: the oracle is far above every per-type ceiling
    # (diversification pays: many independent tries). As rho grows the
    # common shock dominates and the oracle falls toward E[max p(type)]
    # — the "at least one correct" benefit vanishes, leaving only the
    # "best workflow for this task type" benefit. At rho = 1 it
    # coincides with that ceiling: the oracle is correct exactly when
    # the shock is below the best workflow's threshold on that type.
    n_types = candidates[0].n_types
    best_per_type = (
        sum(max(c.success_prob(t) for c in candidates) for t in range(n_types))
        / n_types
    )
    allocation = greedy_allocation(candidates, 12)
    independent = run_experiment(
        candidates, allocation, skill=SKILL, n_tasks=N_TASKS, seed=SEED, rho=0.0
    )
    correlated = run_experiment(
        candidates, allocation, skill=SKILL, n_tasks=N_TASKS, seed=SEED, rho=0.9
    )
    perfect = run_experiment(
        candidates, allocation, skill=SKILL, n_tasks=N_TASKS, seed=SEED, rho=1.0
    )
    assert independent.oracle_accuracy - best_per_type > 0.25
    assert correlated.oracle_accuracy < independent.oracle_accuracy - 0.15
    assert correlated.oracle_accuracy - best_per_type < 0.15
    assert perfect.oracle_accuracy == pytest.approx(best_per_type, abs=0.04)
    assert correlated.lift < independent.lift
    assert correlated.portfolio_accuracy < independent.portfolio_accuracy

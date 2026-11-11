"""Accuracy metrics, the odds-lift index, and a small experiment runner."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .selector import any_correct, effective_skill, select_run
from .simulate import Candidate, generate_task, run_portfolio


def best_single_accuracy(candidates: list[Candidate]) -> float:
    """Expected accuracy of the single best workflow (analytic)."""
    return max(c.expected_accuracy() for c in candidates)


def odds_lift(selector_accuracy: float, chance_accuracy: float) -> float:
    """Odds lift of a selector over chance-level selection.

    With odds(p) = p / (1 - p), the lift is

        odds(selector_accuracy) / odds(chance_accuracy),

    where chance_accuracy is the accuracy of picking a run uniformly at
    random from the same run sets. Lift = 1 means the selector adds
    nothing over chance; lift > 1 means it does. This mirrors the
    odds-lift index in arXiv:2609.18126, with the chance baseline made
    explicit. Returns inf if the selector is perfect, 0 if it never
    succeeds.
    """
    if not 0.0 < chance_accuracy < 1.0:
        raise ValueError("chance_accuracy must be in (0, 1)")
    if selector_accuracy >= 1.0:
        return float("inf")
    if selector_accuracy <= 0.0:
        return 0.0
    return (selector_accuracy / (1.0 - selector_accuracy)) / (
        chance_accuracy / (1.0 - chance_accuracy)
    )


@dataclass(frozen=True)
class ExperimentResult:
    best_single: float  # analytic expected accuracy of the best workflow
    portfolio_accuracy: float  # empirical: selector's pick is correct
    oracle_accuracy: float  # empirical: at least one allocated run correct
    chance_accuracy: float  # empirical: a uniformly random run is correct
    lift: float  # odds_lift(portfolio_accuracy, chance_accuracy)


def run_experiment(
    candidates: list[Candidate],
    allocation: list[int],
    *,
    skill: float,
    n_tasks: int,
    seed: int,
    rho: float = 0.0,
    overconfidence: float = 0.0,
) -> ExperimentResult:
    """Simulate n_tasks tasks under one allocation and selector skill.

    rho is the common-shock correlation between runs on the same task
    (see simulate.run_once); rho = 0 is the independent-runs default.
    overconfidence is the fraction of the claimed skill that is
    illusory (see selector.effective_skill); 0 is the honest default,
    and the selector then behaves with the claimed skill exactly.
    """
    rng = random.Random(seed)
    effective = effective_skill(skill, overconfidence)
    n_types = candidates[0].n_types
    selected_correct = 0
    oracle_correct = 0
    runs_correct = 0
    runs_total = 0
    for _ in range(n_tasks):
        task = generate_task(rng, n_types)
        runs = run_portfolio(candidates, allocation, task, rng, rho=rho)
        chosen = select_run(runs, effective, rng)
        selected_correct += chosen.correct
        oracle_correct += any_correct(runs)
        runs_correct += sum(r.correct for r in runs)
        runs_total += len(runs)
    portfolio_accuracy = selected_correct / n_tasks
    chance_accuracy = runs_correct / runs_total
    return ExperimentResult(
        best_single=best_single_accuracy(candidates),
        portfolio_accuracy=portfolio_accuracy,
        oracle_accuracy=oracle_correct / n_tasks,
        chance_accuracy=chance_accuracy,
        lift=odds_lift(portfolio_accuracy, chance_accuracy),
    )

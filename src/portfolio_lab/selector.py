"""An imperfect selector over run results.

The selector observes only the noisy signals, never the true correctness.
With probability `skill` it picks the run with the highest signal;
otherwise it picks uniformly at random. So skill = 0 is exactly
chance-level selection, and skill = 1 is a perfect reader of signals
that are themselves noisy. This mixture is a reduced-form stand-in for a
real judge (e.g. an LLM grader) whose signal quality varies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .simulate import Candidate, RunResult, generate_task, run_portfolio


def effective_skill(claimed_skill: float, overconfidence: float) -> float:
    """Effective selector skill when confidence systematically overstates it.

    The selector reports `claimed_skill` but behaves as if its skill
    were lower: `claimed_skill * (1 - overconfidence)`. Overconfidence
    is the fraction of the claimed skill that is illusory — 0 means the
    claim is honest (the behaviour is unchanged), 1 means the selector
    is pure chance no matter what it claims. A claimed skill of 0 stays
    0: an honest chance-level selector cannot be overconfident about
    being better than chance.
    """
    if not 0.0 <= claimed_skill <= 1.0:
        raise ValueError("claimed_skill must be in [0, 1]")
    if not 0.0 <= overconfidence <= 1.0:
        raise ValueError("overconfidence must be in [0, 1]")
    return claimed_skill * (1.0 - overconfidence)


def select_run(runs: list[RunResult], skill: float, rng: random.Random) -> RunResult:
    """Pick the final answer among runs, imperfectly (see module docstring)."""
    if not runs:
        raise ValueError("cannot select from an empty run set")
    if not 0.0 <= skill <= 1.0:
        raise ValueError("skill must be in [0, 1]")
    if rng.random() < skill:
        return max(runs, key=lambda r: r.signal)
    return rng.choice(runs)


def any_correct(runs: list[RunResult]) -> bool:
    """Oracle check: did at least one run in the set get it right?"""
    return any(r.correct for r in runs)


@dataclass(frozen=True)
class HeldOutObservation:
    """One held-out task: the run set and the run the selector picked.

    Held out means the task's correctness was known when the observation
    was recorded, so the pick can be scored — the same information a
    real evaluation set gives about a judge. `picked` must be one of
    `runs`.
    """

    runs: list[RunResult]
    picked: RunResult


def collect_observations(
    candidates: list[Candidate],
    allocation: list[int],
    *,
    skill: float,
    n_tasks: int,
    seed: int,
    rho: float = 0.0,
    overconfidence: float = 0.0,
) -> list[HeldOutObservation]:
    """Run the selector with a given true skill on fresh held-out tasks.

    Each task draws a new run set and records which run the selector
    picked, exactly as `run_experiment` would — but the picks are kept,
    so `estimate_skill` can score them against the known correctness.
    """
    rng = random.Random(seed)
    effective = effective_skill(skill, overconfidence)
    n_types = candidates[0].n_types
    observations: list[HeldOutObservation] = []
    for _ in range(n_tasks):
        task = generate_task(rng, n_types)
        runs = run_portfolio(candidates, allocation, task, rng, rho=rho)
        picked = select_run(runs, effective, rng)
        observations.append(HeldOutObservation(runs=runs, picked=picked))
    return observations


def estimate_skill(observations: list[HeldOutObservation]) -> float:
    """Estimate selector skill from held-out picks (method of moments).

    On each held-out task the selector's pick is correct with
    probability skill * P(max-signal run correct) +
    (1 - skill) * P(random run correct), because with probability
    `skill` it picks the max-signal run and otherwise a uniform random
    run. All three accuracies are measured on the held-out tasks, so

        skill = (observed pick accuracy - chance accuracy) /
                (max-signal accuracy - chance accuracy),

    clamped to [0, 1]. Returns 0.0 when the held-out max-signal pick
    is no better than chance: then the signals carry no selectable
    information and skill is unidentified (no evidence of any skill).
    """
    if not observations:
        raise ValueError("cannot estimate skill from no observations")
    n_tasks = len(observations)
    max_correct = 0
    picked_correct = 0
    runs_correct = 0
    runs_total = 0
    for observation in observations:
        runs = observation.runs
        if not runs:
            raise ValueError("observation has an empty run set")
        if observation.picked not in runs:
            raise ValueError("picked run is not one of the run set")
        max_correct += max(runs, key=lambda r: r.signal).correct
        picked_correct += observation.picked.correct
        runs_correct += sum(r.correct for r in runs)
        runs_total += len(runs)
    max_accuracy = max_correct / n_tasks
    chance_accuracy = runs_correct / runs_total
    observed_accuracy = picked_correct / n_tasks
    denominator = max_accuracy - chance_accuracy
    if denominator <= 0.0:
        return 0.0
    estimate = (observed_accuracy - chance_accuracy) / denominator
    return min(1.0, max(0.0, estimate))

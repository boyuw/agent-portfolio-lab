"""Multi-seed sweeps: report the lift distribution, not one seed's lift.

Every number in the demo so far comes from a single seed (seed 7),
so a headline like "lift 8.87" is one draw from a sampling
distribution, not a property of the portfolio. This module runs the
same experiment configuration across many seeds and summarizes the
distribution of the lift (and of the accuracy) with a mean, quantiles,
and range. The candidates and allocation are fixed across seeds —
only the task draws and run outcomes vary — so the spread here is
pure Monte Carlo error on ``n_tasks`` tasks per seed, not
configuration variation. Quantiles use linear interpolation on the
sorted values (the ``numpy.percentile`` default method).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, stdev

from .metrics import ExperimentResult, run_experiment
from .simulate import Candidate


@dataclass(frozen=True)
class DistributionSummary:
    """Mean / quantiles / range of one list of values."""

    n: int
    mean: float
    std: float  # sample standard deviation (0.0 when n == 1)
    median: float
    minimum: float
    maximum: float
    q10: float
    q25: float
    q75: float
    q90: float


@dataclass(frozen=True)
class SeedSweepResult:
    """One :func:`run_experiment` per seed, plus lift/accuracy summaries."""

    seeds: list[int]
    lifts: list[float]
    accuracies: list[float]
    lift_summary: DistributionSummary
    accuracy_summary: DistributionSummary


def quantile(sorted_values: list[float], q: float) -> float:
    """``q``-quantile of already-sorted values, by linear interpolation.

    ``q`` must lie in [0, 1]; the position is ``q * (n - 1)``, so
    ``q = 0`` is the minimum, ``q = 1`` the maximum, and ``q = 0.5``
    the median (the mean of the two middle values when ``n`` is even).
    """
    if not sorted_values:
        raise ValueError("values must be nonempty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_values(values: list[float]) -> DistributionSummary:
    """Mean / quantiles / range of ``values`` (must be nonempty)."""
    if not values:
        raise ValueError("values must be nonempty")
    ordered = sorted(values)
    return DistributionSummary(
        n=len(values),
        mean=fmean(values),
        std=stdev(values) if len(values) > 1 else 0.0,
        median=quantile(ordered, 0.5),
        minimum=ordered[0],
        maximum=ordered[-1],
        q10=quantile(ordered, 0.10),
        q25=quantile(ordered, 0.25),
        q75=quantile(ordered, 0.75),
        q90=quantile(ordered, 0.90),
    )


def sweep_seeds(
    candidates: list[Candidate],
    allocation: list[int],
    *,
    skill: float,
    n_tasks: int,
    seeds: list[int],
    rho: float = 0.0,
    overconfidence: float = 0.0,
) -> SeedSweepResult:
    """Run one experiment per seed and summarize the lift distribution.

    The candidates and allocation are fixed; each seed re-draws the
    tasks and run outcomes through :func:`run_experiment`, with the
    same ``rho`` / ``overconfidence`` throughout. ``seeds`` must be
    nonempty and duplicate-free — a repeated seed would double-count
    one draw in the distribution.
    """
    if not seeds:
        raise ValueError("seeds must be nonempty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be duplicate-free")
    lifts: list[float] = []
    accuracies: list[float] = []
    for seed in seeds:
        result: ExperimentResult = run_experiment(
            candidates,
            allocation,
            skill=skill,
            n_tasks=n_tasks,
            seed=seed,
            rho=rho,
            overconfidence=overconfidence,
        )
        lifts.append(result.lift)
        accuracies.append(result.portfolio_accuracy)
    return SeedSweepResult(
        seeds=list(seeds),
        lifts=lifts,
        accuracies=accuracies,
        lift_summary=summarize_values(lifts),
        accuracy_summary=summarize_values(accuracies),
    )

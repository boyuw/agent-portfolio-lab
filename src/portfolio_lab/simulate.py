"""Seeded synthetic tasks and workflow runs.

Toy model: tasks come in a few types, and each candidate workflow has a
base success probability plus a per-type bonus, so workflows specialise.
A run returns whether the answer was correct plus a noisy quality signal
that a selector can observe. This is a deliberate simplification of real
benchmark behaviour — see docs/DESIGN.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist

# Standard normal quantiles, for the correlated-errors latent model below.
_NORM = NormalDist()


@dataclass(frozen=True)
class Task:
    task_type: int


@dataclass(frozen=True)
class Candidate:
    """A candidate workflow: a base success rate plus per-type bonuses."""

    name: str
    base_prob: float
    type_bonus: tuple[float, ...]

    @property
    def n_types(self) -> int:
        return len(self.type_bonus)

    def success_prob(self, task_type: int) -> float:
        """P(correct answer on a task of this type), clamped to [0.01, 0.97]."""
        return min(0.97, max(0.01, self.base_prob + self.type_bonus[task_type]))

    def expected_accuracy(self) -> float:
        """Mean success probability over uniformly drawn task types."""
        return sum(self.success_prob(t) for t in range(self.n_types)) / self.n_types


@dataclass(frozen=True)
class RunResult:
    candidate_index: int
    correct: bool
    signal: float  # noisy quality signal; the only thing a selector observes


def generate_candidates(n: int, rng: random.Random, n_types: int = 3) -> list[Candidate]:
    """Draw n candidates whose per-type bonuses make them specialise.

    Bonuses are centred to mean zero per candidate, so specialisation is
    a trade-off: every workflow is above its base rate on some task types
    and below it on others, and no workflow dominates everywhere.
    """
    candidates: list[Candidate] = []
    for i in range(n):
        base = rng.uniform(0.35, 0.65)
        raw = [rng.uniform(-0.30, 0.35) for _ in range(n_types)]
        mean = sum(raw) / n_types
        bonus = tuple(b - mean for b in raw)
        candidates.append(Candidate(name=f"w{i}", base_prob=base, type_bonus=bonus))
    return candidates


def generate_task(rng: random.Random, n_types: int = 3) -> Task:
    return Task(task_type=rng.randrange(n_types))


def run_once(
    candidate_index: int,
    candidate: Candidate,
    task: Task,
    rng: random.Random,
    signal_noise: float = 0.4,
    rho: float = 0.0,
    shock: float = 0.0,
) -> RunResult:
    """Run one workflow on one task: Bernoulli correctness + noisy signal.

    With rho = 0 (the default) correctness is an independent Bernoulli
    draw at the workflow's success probability, exactly as before.
    With rho in (0, 1] the task carries a common shock shared by every
    run on it, and correctness comes from a Gaussian copula: the latent
    draw is z = sqrt(1 - rho) * eps + sqrt(rho) * shock with eps, shock
    standard normal, and the run is correct iff z < Phi^-1(p). The
    marginal P(correct) is still p for every workflow, but runs on the
    same task are now positively correlated — a hard task (large shock)
    drags every workflow down at once. At rho = 1 correctness is nested
    by success probability: if a low-p workflow is correct on a task,
    every higher-p workflow is correct too. `shock` is drawn once per
    task by run_portfolio; passing it here keeps run_once testable.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    p = candidate.success_prob(task.task_type)
    if rho == 0.0:
        correct = rng.random() < p
    else:
        z = math.sqrt(1.0 - rho) * rng.gauss(0.0, 1.0) + math.sqrt(rho) * shock
        correct = z < _NORM.inv_cdf(p)
    signal = (1.0 if correct else 0.0) + rng.gauss(0.0, signal_noise)
    return RunResult(candidate_index=candidate_index, correct=correct, signal=signal)


def run_portfolio(
    candidates: list[Candidate],
    allocation: list[int],
    task: Task,
    rng: random.Random,
    rho: float = 0.0,
) -> list[RunResult]:
    """Run each candidate `allocation[i]` times on the task; return all runs.

    With rho > 0 a single common shock is drawn for the task and shared
    by every run on it (see run_once); with rho = 0 the runs are
    independent, as before.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    shock = rng.gauss(0.0, 1.0) if rho > 0.0 else 0.0
    results: list[RunResult] = []
    for idx, (candidate, count) in enumerate(zip(candidates, allocation)):
        for _ in range(count):
            results.append(run_once(idx, candidate, task, rng, rho=rho, shock=shock))
    return results

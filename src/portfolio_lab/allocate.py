"""Budget allocation strategies.

Both strategies distribute a total run budget B across the candidate
workflows and return per-workflow run counts summing exactly to B.
"""

from __future__ import annotations

import math

from .simulate import Candidate


def uniform_allocation(n_workflows: int, budget: int) -> list[int]:
    """Split the budget as evenly as possible (remainder to the first few)."""
    if n_workflows <= 0:
        raise ValueError("need at least one workflow")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    base, remainder = divmod(budget, n_workflows)
    return [base + (1 if i < remainder else 0) for i in range(n_workflows)]


def _failure_prob(candidates: list[Candidate], allocation: list[int], task_type: int) -> float:
    """P(no correct run | task type) under the current allocation."""
    prob = 1.0
    for candidate, count in zip(candidates, allocation):
        prob *= (1.0 - candidate.success_prob(task_type)) ** count
    return prob


def _failure_prob_fractional(
    candidates: list[Candidate], allocation: list[float], task_type: int
) -> float:
    """P(no correct run | task type) under a fractional allocation.

    A fractional count x means the run is "partially bought": the failure
    factor (1 - p) is raised to x instead of to an integer count. This is
    the continuous relaxation of the integer objective greedy optimises.
    """
    prob = 1.0
    for candidate, count in zip(candidates, allocation):
        prob *= (1.0 - candidate.success_prob(task_type)) ** count
    return prob


def fractional_allocation(
    candidates: list[Candidate], budget: int, step: float = 0.05
) -> list[float]:
    """Fractional allocation maximising the relaxed objective, greedily.

    The relaxed objective is P(at least one correct run) averaged over
    task types, with fractional run counts. Each step buys `step` of the
    workflow with the largest marginal gain per unit, where the gain of
    workflow i is the average over types of F_t * (-ln(1 - p_i(t))) and
    F_t is the current failure probability for that type (the derivative
    of the relaxed objective with respect to x_i).

    This is a stub, not the paper's LP solver: it never formulates a
    linear program, never rounds randomly, and never prices new
    candidates (see docs/DESIGN.md). It exists so the demo can show what
    a fractional solution looks like next to the integer greedy one.
    """
    if not candidates:
        raise ValueError("need at least one workflow")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if step <= 0:
        raise ValueError("step must be positive")
    n_types = candidates[0].n_types
    allocation = [0.0] * len(candidates)
    remaining = float(budget)
    while remaining > 1e-9:
        chunk = min(step, remaining)
        best_index, best_gain = 0, -1.0
        for i, candidate in enumerate(candidates):
            gain = sum(
                _failure_prob_fractional(candidates, allocation, t)
                * (-_log1m(candidate.success_prob(t)))
                for t in range(n_types)
            ) / n_types
            if gain > best_gain:
                best_index, best_gain = i, gain
        allocation[best_index] += chunk
        remaining -= chunk
    return allocation


def _log1m(p: float) -> float:
    """ln(1 - p), computed directly (p is clamped away from 1)."""
    return math.log(1.0 - p)


def lp_relaxation_allocation(candidates: list[Candidate], budget: int) -> list[int]:
    """Integer allocation from the fractional solution, largest-remainder.

    Rounds the fractional allocation down, then hands the leftover runs
    (fewer than the number of workflows) to the workflows with the
    largest fractional remainders, lowest index first on ties. The result
    sums exactly to the budget and is deterministic.
    """
    fractional = fractional_allocation(candidates, budget)
    allocation = [int(x) for x in fractional]
    leftover = budget - sum(allocation)
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-(fractional[i] - allocation[i]), i),
    )
    for i in order[:leftover]:
        allocation[i] += 1
    return allocation


def greedy_allocation(candidates: list[Candidate], budget: int) -> list[int]:
    """Greedy marginal-gain allocation, one run at a time.

    The objective is P(at least one correct run) on a random task. Adding
    a run to workflow i reduces the failure probability for task type t by
    F_t * p_i(t), where F_t is the current failure probability for that
    type; the gain is averaged over types. Because workflows specialise
    by type, the greedy choice rotates as the allocation grows.

    This is a simple stand-in for the paper's LP-relaxation allocation —
    not the method itself (see docs/DESIGN.md).
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    n_types = candidates[0].n_types
    allocation = [0] * len(candidates)
    for _ in range(budget):
        best_index, best_gain = 0, -1.0
        for i, candidate in enumerate(candidates):
            gain = sum(
                _failure_prob(candidates, allocation, t) * candidate.success_prob(t)
                for t in range(n_types)
            ) / n_types
            if gain > best_gain:
                best_index, best_gain = i, gain
        allocation[best_index] += 1
    return allocation


def allocation_cost(allocation: list[int], costs: list[float]) -> float:
    """Total cost of an allocation: ``sum(count_i * cost_i)``."""
    if len(allocation) != len(costs):
        raise ValueError(
            f"costs has length {len(costs)}, expected {len(allocation)}"
        )
    return sum(count * cost for count, cost in zip(allocation, costs))


def greedy_cost_allocation(
    candidates: list[Candidate], costs: list[float], cost_budget: float
) -> list[int]:
    """Greedy marginal-gain allocation under a total *cost* budget.

    The run-count greedy (:func:`greedy_allocation`) assumes every run
    costs the same. Here workflow ``i`` costs ``costs[i]`` per run, so
    each step buys the affordable workflow with the largest marginal
    gain *per unit cost* — the average failure-probability reduction
    divided by that workflow's price — and stops when nothing is
    affordable anymore. The result's total cost never exceeds
    ``cost_budget``; the leftover (less than the cheapest workflow's
    price) is unspent, exactly as an unspent run-count remainder would be.

    With equal costs this reduces to :func:`greedy_allocation` with a
    run budget of ``cost_budget / cost`` (tested). Costs must be
    strictly positive: a zero-cost workflow would be bought infinitely,
    and a negative cost is meaningless.
    """
    if not candidates:
        raise ValueError("need at least one workflow")
    if len(costs) != len(candidates):
        raise ValueError(
            f"costs has length {len(costs)}, expected {len(candidates)}"
        )
    if any(cost <= 0.0 for cost in costs):
        raise ValueError("costs must be strictly positive")
    if cost_budget < 0.0:
        raise ValueError("cost budget must be non-negative")
    n_types = candidates[0].n_types
    allocation = [0] * len(candidates)
    remaining = float(cost_budget)
    while True:
        best_index, best_gain_per_cost = -1, 0.0
        for i, candidate in enumerate(candidates):
            if costs[i] > remaining:
                continue
            gain = sum(
                _failure_prob(candidates, allocation, t) * candidate.success_prob(t)
                for t in range(n_types)
            ) / n_types
            gain_per_cost = gain / costs[i]
            if gain_per_cost > best_gain_per_cost:
                best_index, best_gain_per_cost = i, gain_per_cost
        if best_index < 0:
            break
        allocation[best_index] += 1
        remaining -= costs[best_index]
    return allocation

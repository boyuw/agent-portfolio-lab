"""Budget allocation strategies.

Both strategies distribute a total run budget B across the candidate
workflows and return per-workflow run counts summing exactly to B.
"""

from __future__ import annotations

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

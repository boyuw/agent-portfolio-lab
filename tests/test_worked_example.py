"""Tests for the worked example in docs/WORKED_EXAMPLE.md.

Every hand-computed number in the doc is asserted against
``run_experiment`` (seed 7, 20,000 tasks, tolerance 0.01). The two
max-signal accuracies are recomputed here from the one-dimensional
integral in the doc — Simpson's rule, standard library only — rather
than hard-coded, so the doc's one computer-assisted step is itself
checked against the simulation.
"""

from __future__ import annotations

from statistics import NormalDist

from portfolio_lab.allocate import greedy_allocation, uniform_allocation
from portfolio_lab.metrics import best_single_accuracy, run_experiment
from portfolio_lab.simulate import Candidate

PHI = NormalDist().cdf
PHI_D = NormalDist().pdf
SIGMA = 0.4

CANDIDATES = [
    Candidate("A", 0.8, (0.0,)),
    Candidate("B", 0.5, (0.0,)),
    Candidate("C", 0.2, (0.0,)),
]
N_TASKS = 20_000
SEED = 7
TOL = 0.01


def max_signal_accuracy(probs: list[float]) -> float:
    """P(the max-signal run is correct), by exact 1-D quadrature.

    For a correctness pattern ``c``, run ``i`` (correct) has the max
    signal with probability
    ``∫ φ(z) ∏_{j≠i} Φ(z + (c_i − c_j) / σ) dz``; weight the patterns
    by ``∏ p_i^{c_i} (1 − p_i)^{1 − c_i}`` and sum over correct runs.
    """

    def argmax_prob(pattern: list[int], i: int) -> float:
        n = 4001
        lo, hi = -6.0, 6.0
        h = (hi - lo) / (n - 1)
        total = 0.0
        for k in range(n):
            z = lo + k * h
            val = PHI_D(z)
            for j in range(len(pattern)):
                if j != i:
                    val *= PHI(z + (pattern[i] - pattern[j]) / SIGMA)
            weight = 4 if k % 2 == 1 else 2
            if k in (0, n - 1):
                weight = 1
            total += weight * val
        return total * h / 3.0

    total = 0.0
    for mask in range(1 << len(probs)):
        pattern = [(mask >> i) & 1 for i in range(len(probs))]
        prob = 1.0
        for i, p in enumerate(probs):
            prob *= p if pattern[i] else (1.0 - p)
        for i in range(len(probs)):
            if pattern[i]:
                total += prob * argmax_prob(pattern, i)
    return total


def test_setup() -> None:
    assert [c.expected_accuracy() for c in CANDIDATES] == [0.8, 0.5, 0.2]
    assert best_single_accuracy(CANDIDATES) == 0.8
    assert uniform_allocation(3, 3) == [1, 1, 1]
    assert greedy_allocation(CANDIDATES, 3) == [3, 0, 0]


def test_uniform_hand_numbers() -> None:
    # Doc: chance 0.50, oracle 0.92, max-signal accuracy 0.8876.
    assert abs(max_signal_accuracy([0.8, 0.5, 0.2]) - 0.8876) < 0.001
    result = run_experiment(
        CANDIDATES, [1, 1, 1], skill=0.0, n_tasks=N_TASKS, seed=SEED
    )
    assert abs(result.chance_accuracy - 0.50) < TOL
    assert abs(result.oracle_accuracy - 0.92) < TOL
    assert abs(result.portfolio_accuracy - 0.50) < TOL
    assert abs(result.lift - 1.0) < 0.05


def test_uniform_selector_mixture() -> None:
    # Doc: skill 1 -> 0.8876, skill 0.8 -> 0.8*0.8876 + 0.2*0.50 = 0.8101.
    p_max = max_signal_accuracy([0.8, 0.5, 0.2])
    skilled = run_experiment(
        CANDIDATES, [1, 1, 1], skill=1.0, n_tasks=N_TASKS, seed=SEED
    )
    assert abs(skilled.portfolio_accuracy - p_max) < TOL
    mixed = run_experiment(
        CANDIDATES, [1, 1, 1], skill=0.8, n_tasks=N_TASKS, seed=SEED
    )
    assert abs(mixed.portfolio_accuracy - (0.8 * p_max + 0.2 * 0.50)) < TOL


def test_greedy_hand_numbers() -> None:
    # Doc: chance 0.80, oracle 1 - 0.2**3 = 0.992, max-signal 0.9822.
    assert abs(max_signal_accuracy([0.8, 0.8, 0.8]) - 0.9822) < 0.001
    result = run_experiment(
        CANDIDATES, [3, 0, 0], skill=0.0, n_tasks=N_TASKS, seed=SEED
    )
    assert abs(result.chance_accuracy - 0.80) < TOL
    assert abs(result.oracle_accuracy - 0.992) < TOL
    assert abs(result.portfolio_accuracy - 0.80) < TOL
    skilled = run_experiment(
        CANDIDATES, [3, 0, 0], skill=1.0, n_tasks=N_TASKS, seed=SEED
    )
    assert abs(skilled.portfolio_accuracy - 0.9822) < TOL
    mixed = run_experiment(
        CANDIDATES, [3, 0, 0], skill=0.8, n_tasks=N_TASKS, seed=SEED
    )
    assert abs(mixed.portfolio_accuracy - 0.9457) < TOL

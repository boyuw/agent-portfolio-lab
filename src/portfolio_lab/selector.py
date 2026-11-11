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

from .simulate import RunResult


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

"""Demo: best single workflow vs portfolio + imperfect selector.

Run as `python -m portfolio_lab.demo`. Fixed seed, so the output is
reproducible. The oracle row uses the greedy allocation's run sets and
is the ceiling any selector could reach on them.
"""

from __future__ import annotations

import random

from .allocate import greedy_allocation, uniform_allocation
from .metrics import run_experiment
from .simulate import generate_candidates

SEED = 7
N_WORKFLOWS = 5
BUDGET = 12
N_TASKS = 2000
SKILL = 0.8


def main() -> None:
    candidates = generate_candidates(N_WORKFLOWS, random.Random(SEED))
    uniform = uniform_allocation(N_WORKFLOWS, BUDGET)
    greedy = greedy_allocation(candidates, BUDGET)
    uniform_result = run_experiment(
        candidates, uniform, skill=SKILL, n_tasks=N_TASKS, seed=SEED
    )
    greedy_result = run_experiment(
        candidates, greedy, skill=SKILL, n_tasks=N_TASKS, seed=SEED
    )

    print(
        f"seed={SEED}  workflows={N_WORKFLOWS}  budget={BUDGET} runs/task  "
        f"tasks={N_TASKS}  selector skill={SKILL}"
    )
    print("candidates (expected accuracy):")
    for c in candidates:
        print(f"  {c.name}: {c.expected_accuracy():.3f}")
    print(f"allocations: uniform={uniform}  greedy={greedy}")
    print()
    print(f"{'strategy':<28}{'accuracy':>9}{'odds-lift':>11}")
    print(f"{'best single (expected)':<28}{greedy_result.best_single:>9.4f}{'—':>11}")
    print(
        f"{'portfolio + selector (uniform)':<28}"
        f"{uniform_result.portfolio_accuracy:>9.4f}{uniform_result.lift:>11.2f}"
    )
    print(
        f"{'portfolio + selector (greedy)':<28}"
        f"{greedy_result.portfolio_accuracy:>9.4f}{greedy_result.lift:>11.2f}"
    )
    print(
        f"{'oracle (greedy runs)':<28}{greedy_result.oracle_accuracy:>9.4f}{'—':>11}"
    )


if __name__ == "__main__":
    main()

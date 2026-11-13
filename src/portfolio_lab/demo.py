"""Demo: best single workflow vs portfolio + imperfect selector.

Run as `python -m portfolio_lab.demo`. Fixed seed, so the output is
reproducible. The oracle row uses the greedy allocation's run sets and
is the ceiling any selector could reach on them.
"""

from __future__ import annotations

import random

from .allocate import (
    allocation_cost,
    greedy_allocation,
    greedy_cost_allocation,
    lp_relaxation_allocation,
    uniform_allocation,
)
from .metrics import run_experiment
from .selector import collect_observations, estimate_skill
from .simulate import generate_candidates

SEED = 7
N_WORKFLOWS = 5
BUDGET = 12
N_TASKS = 2000
SKILL = 0.8
# Illustrative per-run prices: workflows are not equally expensive, so
# a cost budget buys a different allocation than a run-count budget.
COSTS = [1.0, 3.0, 2.0, 1.0, 4.0]
COST_BUDGET = 24.0


def main() -> None:
    candidates = generate_candidates(N_WORKFLOWS, random.Random(SEED))
    uniform = uniform_allocation(N_WORKFLOWS, BUDGET)
    greedy = greedy_allocation(candidates, BUDGET)
    lp_stub = lp_relaxation_allocation(candidates, BUDGET)
    cost_greedy = greedy_cost_allocation(candidates, COSTS, COST_BUDGET)
    uniform_result = run_experiment(
        candidates, uniform, skill=SKILL, n_tasks=N_TASKS, seed=SEED
    )
    greedy_result = run_experiment(
        candidates, greedy, skill=SKILL, n_tasks=N_TASKS, seed=SEED
    )
    lp_result = run_experiment(
        candidates, lp_stub, skill=SKILL, n_tasks=N_TASKS, seed=SEED
    )
    cost_result = run_experiment(
        candidates, cost_greedy, skill=SKILL, n_tasks=N_TASKS, seed=SEED
    )

    print(
        f"seed={SEED}  workflows={N_WORKFLOWS}  budget={BUDGET} runs/task  "
        f"tasks={N_TASKS}  selector skill={SKILL}"
    )
    print("candidates (expected accuracy):")
    for c in candidates:
        print(f"  {c.name}: {c.expected_accuracy():.3f}")
    print(f"allocations: uniform={uniform}  greedy={greedy}  lp-stub={lp_stub}")
    print(
        f"costs per run={COSTS}  cost budget={COST_BUDGET}  "
        f"cost-greedy={cost_greedy} "
        f"(spent={allocation_cost(cost_greedy, COSTS):.0f})"
    )
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
        f"{'portfolio + selector (LP stub)':<28}"
        f"{lp_result.portfolio_accuracy:>9.4f}{lp_result.lift:>11.2f}"
    )
    print(
        f"{'portfolio + selector (cost)':<28}"
        f"{cost_result.portfolio_accuracy:>9.4f}{cost_result.lift:>11.2f}"
    )
    print(
        f"{'oracle (greedy runs)':<28}{greedy_result.oracle_accuracy:>9.4f}{'—':>11}"
    )
    print()
    print("correlation sweep (greedy allocation): a common shock per task")
    print("makes workflow errors correlate, which erodes the portfolio.")
    print(f"{'rho':<28}{'accuracy':>9}{'oracle':>9}{'odds-lift':>11}")
    for rho in (0.0, 0.5, 0.9):
        result = run_experiment(
            candidates, greedy, skill=SKILL, n_tasks=N_TASKS, seed=SEED, rho=rho
        )
        print(
            f"{f'rho={rho}':<28}{result.portfolio_accuracy:>9.4f}"
            f"{result.oracle_accuracy:>9.4f}{result.lift:>11.2f}"
        )
    print()
    print("miscalibration sweep (greedy allocation): the selector claims")
    print(f"skill {SKILL} but behaves with skill {SKILL} * (1 - overconfidence).")
    print("Overstated confidence erodes the lift toward 1 (pure chance).")
    print(f"{'overconfidence':<28}{'accuracy':>9}{'eff. skill':>11}{'odds-lift':>11}")
    for overconfidence in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = run_experiment(
            candidates,
            greedy,
            skill=SKILL,
            n_tasks=N_TASKS,
            seed=SEED,
            overconfidence=overconfidence,
        )
        print(
            f"{f'overconfidence={overconfidence}':<28}"
            f"{result.portfolio_accuracy:>9.4f}"
            f"{SKILL * (1.0 - overconfidence):>11.2f}"
            f"{result.lift:>11.2f}"
        )
    print()
    print("adaptive selector (greedy allocation): the skill is estimated")
    print("from held-out tasks instead of assumed. The estimate is the")
    print("method-of-moments ratio (pick accuracy - chance accuracy) /")
    print("(max-signal accuracy - chance accuracy) on fresh tasks.")
    observations = collect_observations(
        candidates, greedy, skill=SKILL, n_tasks=N_TASKS, seed=SEED + 1000
    )
    estimated = estimate_skill(observations)
    estimated_result = run_experiment(
        candidates, greedy, skill=estimated, n_tasks=N_TASKS, seed=SEED
    )
    print(
        f"assumed skill={SKILL:.2f}  estimated skill={estimated:.2f} "
        f"(held-out tasks={N_TASKS}, seed={SEED + 1000})"
    )
    print(f"{'strategy':<28}{'accuracy':>9}{'odds-lift':>11}")
    print(
        f"{'portfolio + selector (assumed)':<28}"
        f"{greedy_result.portfolio_accuracy:>9.4f}{greedy_result.lift:>11.2f}"
    )
    print(
        f"{'portfolio + selector (estimated)':<28}"
        f"{estimated_result.portfolio_accuracy:>9.4f}{estimated_result.lift:>11.2f}"
    )


if __name__ == "__main__":
    main()

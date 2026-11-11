# Design notes

Purpose: a simulation small enough to review in one sitting that still
preserves the *shape* of the argument in arXiv:2609.18126 — portfolios
help when workflows are diverse and the selector, though imperfect, is
better than chance; allocation of a fixed compute budget matters.

## Decisions

- **Standard library only, seeded everywhere.** Every random draw flows
  from an explicit `random.Random`, so demos and tests are reproducible
  and there is nothing to install.
- **Task types are the load-bearing simplification.** Each candidate has
  a base success probability plus a per-type bonus, i.e. workflows
  specialise. Without heterogeneity like this, diversification has no
  value (for "at least one correct", concentrating the budget on the
  best workflow is optimal), and the demo would teach the wrong lesson.
- **Selector = mixture model.** With probability `skill` the selector
  picks the run with the highest noisy signal, otherwise it picks
  uniformly at random. Two lines, and it makes the endpoints exact:
  skill 0 *is* the chance baseline, so the odds-lift at skill 0 is 1 by
  construction rather than by calibration.
- **Signals are correctness + Gaussian noise** (σ = 0.4). The selector
  never sees correctness itself; at skill 1 it is a perfect reader of
  imperfect signals, which keeps the oracle (hindsight) strictly above
  any selector.
- **Greedy allocation optimises a stated objective.** Each run goes to
  the workflow with the largest marginal gain in P(at least one correct)
  averaged over task types. The objective is printed in the docstring so
  the strategy can be criticised on its own terms.
- **Best-single is analytic, the rest empirical.** Best-single accuracy
  is the best workflow's expected accuracy (no sampling noise in the
  baseline); portfolio, chance, and oracle numbers come from the same
  simulated run sets.

## Deliberately simplified vs. the paper

- **No real tasks or models.** Correctness is Bernoulli with known
  probabilities; nothing here measures any actual workflow or benchmark.
- **LP relaxation is a stub, not a solver.** `fractional_allocation`
  greedily maximises the continuous relaxation of the greedy objective
  in small fractional steps, and `lp_relaxation_allocation` rounds it
  by largest remainder. There is still no linear program, no randomized
  rounding, and no dual/pricing oracle proposing new candidates — the
  stub exists to show what a fractional solution looks like next to the
  integer greedy one, not to reproduce the paper's method.
- **Selector skill is a reduced-form parameter.** The paper models
  selection from observed scores; here `skill` collapses signal quality
  and judge quality into one number, and the within-task signal model is
  a single Gaussian. The miscalibration mode (`overconfidence` in
  `run_experiment`) keeps that reduction: the selector claims `skill`
  but behaves with `skill * (1 - overconfidence)`, so overstated
  confidence is a scaled-down skill, not a different mechanism. The
  demo sweep (claimed skill 0.8) shows the lift eroding 8.87 -> 2.14
  -> 0.90 as overconfidence goes 0 -> 0.5 -> 1.0, i.e. a fully
  overconfident selector is pure chance by construction.
- **Independent runs, uniform task types — unless rho > 0.** No
  sequential/adaptive budgets, no correlation between workflows'
  errors beyond shared task type, in the default mode. The correlated
  mode (`rho` in `run_once` / `run_portfolio` / `run_experiment`) adds
  one Gaussian common shock per task through a copula: correctness is
  z < Phi^-1(p) with z = sqrt(1-rho)*eps + sqrt(rho)*shock. Marginals
  are preserved (each workflow is still right with probability p) but
  runs on the same task are positively correlated, so the oracle
  ceiling collapses from ~1 toward E[max p(type)] as rho -> 1 — at
  rho = 1 the oracle is correct exactly when the shock is below the
  best workflow's threshold on that task type, and the demo sweep
  (rho = 0 / 0.5 / 0.9) shows the lift eroding 8.87 -> 5.37 -> 2.61.
  Budgets used to be run counts only; `greedy_cost_allocation`
  now buys gain per unit cost under a total cost budget, and reduces to
  run-count greedy when all prices are equal (tested).
- **Odds-lift baseline is explicit.** The lift here is relative to
  picking a run uniformly at random from the same run sets; the paper's
  index is defined against its own chance model. Same spirit, stated
  baseline, not the identical quantity.

## What would change first in a v2

An adaptive selector whose skill is estimated from held-out signals
instead of assumed. (Correlated errors landed 2026-11-05: the common-
shock copula in `simulate.py`, demo sweep included. Cost-weighted
budgets landed 2026-11-02: `greedy_cost_allocation` in `allocate.py`,
demo row included.)

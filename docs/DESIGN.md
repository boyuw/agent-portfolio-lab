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
  a single Gaussian.
- **Independent runs, uniform task types, fixed budget per task.** No
  cost heterogeneity across workflows, no sequential/adaptive budgets,
  no correlation between workflows' errors beyond shared task type.
- **Odds-lift baseline is explicit.** The lift here is relative to
  picking a run uniformly at random from the same run sets; the paper's
  index is defined against its own chance model. Same spirit, stated
  baseline, not the identical quantity.

## What would change first in a v2

Cost-weighted budgets (workflows with different prices), correlated
errors, and an adaptive selector whose skill is estimated from held-out
signals instead of assumed.

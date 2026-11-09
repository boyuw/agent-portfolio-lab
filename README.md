# agent-portfolio-lab

A toy-scale, runnable companion to the paper
[*Designing Agentic AI Workflow Portfolios under Imperfect Selection and
Compute Cost*](https://arxiv.org/abs/2609.18126) (Abdolmaleki, Jasin & Wang).

The idea in one paragraph: instead of betting everything on a single
agentic workflow, run a **portfolio** of candidate workflows under a fixed
compute budget and let an **imperfect selector** pick the final answer.
This repo simulates that setup end-to-end — synthetic tasks, specialised
workflows, noisy quality signals, two budget-allocation strategies, and
the paper's odds-lift view of selector quality — in ~300 lines of
standard-library Python.

> **Scope, honestly.** This is a teaching simulation, not the paper's
> method and not a benchmark result. Tasks are Bernoulli coin flips, the
> selector is a two-line mixture model, and the greedy allocator is a
> stand-in for the paper's LP relaxation with randomized rounding (there
> is no pricing oracle here). See [docs/DESIGN.md](docs/DESIGN.md) for
> exactly what is simplified and why.

## Quickstart

Requires Python 3.11+. No dependencies.

```bash
python -m portfolio_lab.demo     # from the repo root, with src/ on PYTHONPATH
PYTHONPATH=src python -m portfolio_lab.demo
pip install -e .                 # or install it properly
python -m pytest                 # run the tests
```

The demo prints, for a fixed seed, the expected accuracy of the best
single workflow vs. portfolio + selector accuracy under uniform and
greedy budget allocation, plus the oracle ceiling and each portfolio's
odds-lift over chance-level selection.

## Concepts, paper → toy

| Paper concept | Toy counterpart |
|---|---|
| Candidate workflow portfolio | `Candidate` with a base success rate and per-task-type bonuses (`simulate.py`) |
| Imperfect selector | With probability `skill`, picks the highest noisy signal; else picks at random (`selector.py`) |
| Compute budget | A fixed number of runs per task, split by an allocation (`allocate.py`) |
| LP allocation + rounding | Greedy marginal-gain allocation on P(at least one correct) — a stand-in |
| Odds-lift index | `odds(selector acc) / odds(chance acc)`, chance = picking a run uniformly at random (`metrics.py`) |
| Oracle upper bound | A run set counts as oracle-correct if any run in it is correct |

The interesting effect to look for: because workflows specialise by task
type, a portfolio with a decent selector beats the best single workflow
even though every individual workflow is mediocre — and the greedy
allocator, which chases marginal gains per task type, beats the uniform
split.

## How to read the lift number

The demo's `odds-lift` column is `odds(selector accuracy) /
odds(chance accuracy)`, with `odds(p) = p / (1 - p)`. Chance accuracy is
the accuracy of picking a run uniformly at random from the *same* run
sets the selector chose from — so the lift isolates the selector's
contribution from the allocation's.

- **Lift = 1** — the selector adds nothing over picking a run at random.
- **Lift > 1** — the selector extracts signal from the noisy run scores.
  A perfect selector has infinite lift.
- **Why odds, not a plain accuracy ratio?** Odds stretch the scale near
  the ceiling: moving accuracy from 0.51 to 0.90 is only a 1.8× accuracy
  gain but an 8.9× odds gain, which better reflects how hard the last
  few points are to win. (Demo, greedy allocation, seed 7: accuracy
  0.9010 vs chance 0.5064 → odds 9.10 vs 1.03 → lift 8.87.)
- **Compare lifts only on the same run sets.** A better allocation
  raises the chance baseline too, so a higher-accuracy portfolio can
  show a *lower* lift: the demo's uniform split is less accurate than
  greedy (0.8915 vs 0.9010) but has a higher lift (9.17 vs 8.87),
  because its run sets are easier to beat by chance (0.4727 vs 0.5064).
  Lift measures the selector, not the portfolio.
- **Correlation deflates the lift.** When workflow errors share a
  common shock, the oracle ceiling falls and there is less for the
  selector to extract: the demo's rho sweep drops the greedy lift from
  8.87 (rho = 0) to 2.61 (rho = 0.9).

## Layout

```
src/portfolio_lab/
  simulate.py   synthetic task types, candidates, noisy run signals
  selector.py   the imperfect selector (skill = P(reads signals))
  allocate.py   uniform and greedy budget allocation (both sum to B)
  metrics.py    best-single / portfolio / oracle accuracy, odds-lift
  demo.py       python -m portfolio_lab.demo
tests/          pytest: lift ≈ 1 at chance, > 1 when skilled,
                portfolio ≥ best single, allocations sum to B, seeded
                determinism
docs/DESIGN.md  design decisions and deliberate simplifications
```

## License

MIT © Boyu Wang.

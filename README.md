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

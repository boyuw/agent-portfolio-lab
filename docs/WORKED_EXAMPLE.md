# Worked example: three workflows, three runs

A tiny configuration where every number in the demo can be checked by
hand — except one one-dimensional integral, which is shown in full and
recomputed independently in `tests/test_worked_example.py`.

## Setup

One task type, so there is no specialisation to keep track of. Three
workflows with success probabilities

| workflow | P(correct) |
|---|---|
| A | 0.80 |
| B | 0.50 |
| C | 0.20 |

Budget: 3 runs per task. Signals are `correct + N(0, 0.4)` — the
selector sees the signal, never the correctness.

```python
from portfolio_lab.simulate import Candidate
candidates = [
    Candidate("A", 0.8, (0.0,)),
    Candidate("B", 0.5, (0.0,)),
    Candidate("C", 0.2, (0.0,)),
]
```

## The three quantities you can compute on paper

**Best single workflow.** A, at 0.80. No simulation involved:
`best_single_accuracy` is analytic.

**Chance accuracy** (pick a run uniformly at random) under the uniform
allocation `[1, 1, 1]` — the mean of the three success probabilities:

```
(0.80 + 0.50 + 0.20) / 3 = 0.50
```

**Oracle accuracy** (at least one run correct) under `[1, 1, 1]` —
one minus the probability that all three runs fail, runs independent:

```
1 − (1 − 0.80) × (1 − 0.50) × (1 − 0.20) = 1 − 0.20 × 0.50 × 0.80
                                          = 1 − 0.08 = 0.92
```

## The selector, step by step

With probability `skill` the selector picks the run with the highest
signal, otherwise a uniform random run, so for any allocation

```
accuracy(skill) = skill × P(max-signal run correct)
                  + (1 − skill) × chance accuracy
```

`P(max-signal run correct)` is the one number that needs a computer:
conditional on a correctness pattern `c ∈ {0, 1}³`, the signals are
independent normals centred at `c`, so for a correct run `i`

```
P(i has the max signal | c) = ∫ φ(z) ∏_{j≠i} Φ(z + (c_i − c_j) / 0.4) dz
```

and the total is that integral weighted over the 8 patterns by
`∏ p_i^{c_i} (1 − p_i)^{1 − c_i}`. Evaluated by Simpson's rule
(standard library only, in the test) it is **0.8876** for `[1, 1, 1]`
— below the oracle 0.92, because the signals are noisy: the selector
can only pick the best *signal*, not the best *run*.

| skill | accuracy | odds-lift over chance 0.50 |
|---|---|---|
| 0.0 (chance) | 0.50 | 1.00 |
| 0.8 | 0.8 × 0.8876 + 0.2 × 0.50 = **0.8101** | 4.27 |
| 1.0 (reads signals perfectly) | **0.8876** | 7.90 |

For comparison, a *noiseless*-signal selector would reach the oracle
0.92, lift `odds(0.92) / odds(0.50) = 11.50`. The gap
0.92 − 0.8876 ≈ 0.03 is the price of noisy signals at this
configuration — small here because σ = 0.4 is small next to the 0/1
correctness gap.

## Greedy allocation, also by hand

Greedy adds one run at a time to the workflow with the largest
marginal gain in P(at least one correct). With failure probability
`F` (starting at 1), adding a run to workflow `i` gains `F × p_i`:

| step | F before | gains A / B / C | pick |
|---|---|---|---|
| 1 | 1.00 | 0.80 / 0.50 / 0.20 | A |
| 2 | 0.20 | 0.16 / 0.10 / 0.04 | A |
| 3 | 0.04 | 0.032 / 0.020 / 0.008 | A |

Greedy is `[3, 0, 0]` — all three runs on the strongest workflow.
That is expected, not a bug: with a single task type there is no
specialisation for diversification to exploit (see `docs/DESIGN.md`).
Its hand-computed numbers:

- chance accuracy = 0.80 (every run is an A run)
- oracle accuracy = `1 − 0.20³ = 0.992`
- skill 1: the selector still has work to do — it picks the
  best-signalled of the three A runs, and the same max-signal integral
  with three `p = 0.8` runs gives **0.9822**, lift 13.76
- skill 0.8: `0.8 × 0.9822 + 0.2 × 0.80 = 0.9457`

## Summary

| allocation | chance | oracle | skill-1 accuracy | skill-1 lift |
|---|---|---|---|---|
| uniform `[1, 1, 1]` | 0.50 | 0.92 | 0.8876 | 7.90 |
| greedy `[3, 0, 0]` | 0.80 | 0.992 | 0.9822 | 13.76 |

Three lessons to take away:

1. **The portfolio beats the best single workflow** (0.8876 and
   0.9822 vs 0.80) — but on this configuration the win comes mostly
   from having three chances instead of one, not from diversity.
2. **Greedy maximises the oracle, not the lift.** Its accuracy is
   higher because its run set is stronger, and its chance baseline is
   higher too (0.80 vs 0.50). Lift compares selectors only on the
   same run sets.
3. **The selector's value is bounded by signal noise.** Even a
   perfect reader of the signals stops at 0.8876 on the uniform run
   set, short of the 0.92 oracle — the signals, not the selector, are
   the bottleneck.

## Reproduce it

```python
from portfolio_lab.metrics import run_experiment
run_experiment(candidates, [1, 1, 1], skill=0.8, n_tasks=200_000, seed=7)
# ExperimentResult(best_single=0.8, portfolio_accuracy≈0.8101,
#                  oracle_accuracy≈0.92, chance_accuracy≈0.50, lift≈4.27)
```

`tests/test_worked_example.py` asserts every hand-computed number
above against `run_experiment` (seed 7, 20,000 tasks, ±0.01), with the
max-signal accuracies recomputed from the integral rather than
hard-coded.

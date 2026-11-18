"""Tests for the results-table export (CSV / JSON / Markdown).

The strategy-table values are pinned to the demo's fixed seed-7
configuration, so the export is checked against the numbers the demo
itself prints — an export that silently changed a value would fail
here. Round-trip tests check that CSV/JSON parse back to the same
rows, and the Markdown test checks the table pasted into the README.
"""

import csv
import io
import json
import random

import pytest

from portfolio_lab.allocate import (
    greedy_allocation,
    greedy_cost_allocation,
    lp_relaxation_allocation,
    uniform_allocation,
)
from portfolio_lab.demo import (
    COST_BUDGET,
    COSTS,
    N_TASKS,
    SEED,
    main,
)
from portfolio_lab.export import (
    CORRELATION_COLUMNS,
    MISCALIBRATION_COLUMNS,
    STRATEGY_COLUMNS,
    correlation_rows,
    miscalibration_rows,
    rows_to_csv,
    rows_to_json,
    rows_to_markdown,
    strategy_rows,
)
from portfolio_lab.metrics import run_experiment
from portfolio_lab.simulate import generate_candidates

SKILL = 0.8


@pytest.fixture(scope="module")
def strategy_results():
    candidates = generate_candidates(5, random.Random(SEED))
    uniform = uniform_allocation(5, 12)
    greedy = greedy_allocation(candidates, 12)
    lp_stub = lp_relaxation_allocation(candidates, 12)
    cost_greedy = greedy_cost_allocation(candidates, COSTS, COST_BUDGET)
    results = {
        "portfolio + selector (uniform)": run_experiment(
            candidates, uniform, skill=SKILL, n_tasks=N_TASKS, seed=SEED
        ),
        "portfolio + selector (greedy)": run_experiment(
            candidates, greedy, skill=SKILL, n_tasks=N_TASKS, seed=SEED
        ),
        "portfolio + selector (LP stub)": run_experiment(
            candidates, lp_stub, skill=SKILL, n_tasks=N_TASKS, seed=SEED
        ),
        "portfolio + selector (cost)": run_experiment(
            candidates, cost_greedy, skill=SKILL, n_tasks=N_TASKS, seed=SEED
        ),
    }
    return results["portfolio + selector (greedy)"].best_single, results


def test_strategy_rows_match_demo_numbers(strategy_results):
    best_single, results = strategy_results
    rows = strategy_rows(
        best_single,
        results,
        results["portfolio + selector (greedy)"].oracle_accuracy,
    )
    assert [row["strategy"] for row in rows] == [
        "best single (expected)",
        "portfolio + selector (uniform)",
        "portfolio + selector (greedy)",
        "portfolio + selector (LP stub)",
        "portfolio + selector (cost)",
        "oracle (greedy runs)",
    ]
    by_label = {row["strategy"]: row for row in rows}
    # Best single is analytic (max expected accuracy), not on the
    # 1/2000 empirical grid, so allow a 4-decimal rounding tolerance.
    assert by_label["best single (expected)"]["accuracy"] == pytest.approx(
        0.5382, abs=0.00005
    )
    assert by_label["best single (expected)"]["odds_lift"] is None
    assert by_label["portfolio + selector (uniform)"]["accuracy"] == pytest.approx(
        0.8915
    )
    assert by_label["portfolio + selector (uniform)"]["odds_lift"] == pytest.approx(
        9.17, abs=0.005
    )
    assert by_label["portfolio + selector (greedy)"]["accuracy"] == pytest.approx(
        0.9010
    )
    assert by_label["portfolio + selector (greedy)"]["odds_lift"] == pytest.approx(
        8.87, abs=0.005
    )
    assert by_label["portfolio + selector (LP stub)"]["accuracy"] == pytest.approx(
        0.8960
    )
    assert by_label["portfolio + selector (cost)"]["accuracy"] == pytest.approx(
        0.8885
    )
    assert by_label["oracle (greedy runs)"]["accuracy"] == pytest.approx(0.9995)
    assert by_label["oracle (greedy runs)"]["chance"] is None
    # Oracle and chance are per-allocation empirical quantities, so the
    # greedy row's oracle must equal the oracle row's accuracy.
    assert by_label["portfolio + selector (greedy)"]["oracle"] == pytest.approx(
        0.9995
    )


def test_csv_round_trips(strategy_results):
    best_single, results = strategy_results
    rows = strategy_rows(
        best_single,
        results,
        results["portfolio + selector (greedy)"].oracle_accuracy,
    )
    text = rows_to_csv(rows, STRATEGY_COLUMNS)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert len(parsed) == len(rows)
    assert list(parsed[0]) == [
        "strategy",
        "accuracy",
        "oracle",
        "chance",
        "odds-lift",
    ]
    greedy = parsed[2]
    assert greedy["strategy"] == "portfolio + selector (greedy)"
    assert float(greedy["accuracy"]) == pytest.approx(0.9010)
    assert float(greedy["odds-lift"]) == pytest.approx(8.87, abs=0.005)
    # Missing values are empty fields, not the string "None".
    assert parsed[0]["oracle"] == ""
    assert parsed[0]["odds-lift"] == ""


def test_json_round_trips(strategy_results):
    best_single, results = strategy_results
    rows = strategy_rows(
        best_single,
        results,
        results["portfolio + selector (greedy)"].oracle_accuracy,
    )
    parsed = json.loads(rows_to_json(rows, STRATEGY_COLUMNS))
    assert len(parsed) == len(rows)
    assert set(parsed[0]) == {
        "strategy",
        "accuracy",
        "oracle",
        "chance",
        "odds_lift",
    }
    assert parsed[0]["oracle"] is None
    assert parsed[2]["accuracy"] == pytest.approx(0.9010)
    assert parsed[2]["odds_lift"] == pytest.approx(8.87, abs=0.005)


def test_markdown_table_matches_demo_print(strategy_results):
    best_single, results = strategy_results
    rows = strategy_rows(
        best_single,
        results,
        results["portfolio + selector (greedy)"].oracle_accuracy,
    )
    table = rows_to_markdown(rows, STRATEGY_COLUMNS)
    lines = table.strip().split("\n")
    assert lines[0] == "| strategy | accuracy | oracle | chance | odds-lift |"
    assert lines[1] == "| --- | --- | --- | --- | --- |"
    assert (
        "| portfolio + selector (greedy) | 0.9010 | 0.9995 | 0.5064 | 8.87 |"
        in lines
    )
    assert "| best single (expected) | 0.5382 | — | — | — |" in lines
    assert "| oracle (greedy runs) | 0.9995 | — | — | — |" in lines


def test_correlation_rows_match_demo_sweep():
    candidates = generate_candidates(5, random.Random(SEED))
    greedy = greedy_allocation(candidates, 12)
    results = {
        rho: run_experiment(
            candidates, greedy, skill=SKILL, n_tasks=N_TASKS, seed=SEED, rho=rho
        )
        for rho in (0.0, 0.5, 0.9)
    }
    rows = correlation_rows(results)
    assert [row["rho"] for row in rows] == [0.0, 0.5, 0.9]
    assert rows[0]["accuracy"] == pytest.approx(0.9010)
    assert rows[1]["oracle"] == pytest.approx(0.9475)
    assert rows[2]["odds_lift"] == pytest.approx(2.61, abs=0.005)
    table = rows_to_markdown(rows, CORRELATION_COLUMNS)
    assert "| 0.9 | 0.7270 | 0.7990 |" in table


def test_miscalibration_rows_match_demo_sweep():
    candidates = generate_candidates(5, random.Random(SEED))
    greedy = greedy_allocation(candidates, 12)
    results = {
        overconfidence: run_experiment(
            candidates,
            greedy,
            skill=SKILL,
            n_tasks=N_TASKS,
            seed=SEED,
            overconfidence=overconfidence,
        )
        for overconfidence in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    rows = miscalibration_rows(results, SKILL)
    assert [row["effective_skill"] for row in rows] == pytest.approx(
        [0.8, 0.6, 0.4, 0.2, 0.0]
    )
    assert rows[2]["odds_lift"] == pytest.approx(2.14, abs=0.005)
    assert rows[4]["odds_lift"] == pytest.approx(0.90, abs=0.005)
    table = rows_to_markdown(rows, MISCALIBRATION_COLUMNS)
    assert "| 0.50 | 0.6855 | 0.40 | 2.14 |" in table


def test_empty_rows_still_emit_a_header():
    assert rows_to_csv([], STRATEGY_COLUMNS).strip() == (
        "strategy,accuracy,oracle,chance,odds-lift"
    )
    assert json.loads(rows_to_json([], STRATEGY_COLUMNS)) == []
    assert rows_to_markdown([], STRATEGY_COLUMNS).strip().split("\n") == [
        "| strategy | accuracy | oracle | chance | odds-lift |",
        "| --- | --- | --- | --- | --- |",
    ]


def test_demo_export_flags_write_files(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    json_path = tmp_path / "results.json"
    md_path = tmp_path / "results.md"
    main(
        [
            "--export-csv",
            str(csv_path),
            "--export-json",
            str(json_path),
            "--export-md",
            str(md_path),
        ]
    )
    out = capsys.readouterr().out
    assert f"wrote {csv_path}" in out
    assert f"wrote {json_path}" in out
    assert f"wrote {md_path}" in out
    assert "portfolio + selector (greedy)" in csv_path.read_text()
    parsed = json.loads(json_path.read_text())
    assert parsed[2]["strategy"] == "portfolio + selector (greedy)"
    assert parsed[2]["accuracy"] == pytest.approx(0.9010)
    assert "| portfolio + selector (greedy) | 0.9010 |" in md_path.read_text()

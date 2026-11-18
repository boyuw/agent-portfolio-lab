"""Export the demo's results tables as CSV, JSON, or Markdown.

The demo prints its tables to stdout; this module turns the same
:class:`~portfolio_lab.metrics.ExperimentResult` objects into
machine-readable tables (CSV/JSON) and a GitHub-flavoured Markdown
table ready to paste into the README. Rows are plain dicts so the
serializers stay generic: a column spec is ``(key, header, fmt)``,
where ``fmt`` is a ``format()`` spec applied to numbers (``None``
means "leave the value as-is", for string columns) and a ``None``
value renders as an em dash in Markdown and as an empty field in
CSV / ``null`` in JSON.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence

from .metrics import ExperimentResult

# Column specs: (row key, header, number format or None).
STRATEGY_COLUMNS: list[tuple[str, str, str | None]] = [
    ("strategy", "strategy", None),
    ("accuracy", "accuracy", ".4f"),
    ("oracle", "oracle", ".4f"),
    ("chance", "chance", ".4f"),
    ("odds_lift", "odds-lift", ".2f"),
]
CORRELATION_COLUMNS: list[tuple[str, str, str | None]] = [
    ("rho", "rho", ".1f"),
    ("accuracy", "accuracy", ".4f"),
    ("oracle", "oracle", ".4f"),
    ("chance", "chance", ".4f"),
    ("odds_lift", "odds-lift", ".2f"),
]
MISCALIBRATION_COLUMNS: list[tuple[str, str, str | None]] = [
    ("overconfidence", "overconfidence", ".2f"),
    ("accuracy", "accuracy", ".4f"),
    ("effective_skill", "effective skill", ".2f"),
    ("odds_lift", "odds-lift", ".2f"),
]

Row = dict[str, object]


def strategy_rows(
    best_single: float,
    results: Mapping[str, ExperimentResult],
    oracle: float,
) -> list[Row]:
    """The demo's main table: one row per allocation strategy.

    ``results`` maps a strategy label (e.g. ``"portfolio + selector
    (greedy)"``) to its :class:`ExperimentResult`. The best-single row
    is analytic so it has no oracle/chance/lift, and the oracle row
    (the ceiling on the greedy run sets) has no chance/lift either;
    both render as ``—`` in Markdown and as blank / ``null`` in
    CSV / JSON.
    """
    rows: list[Row] = [
        {
            "strategy": "best single (expected)",
            "accuracy": best_single,
            "oracle": None,
            "chance": None,
            "odds_lift": None,
        }
    ]
    for label, result in results.items():
        rows.append(
            {
                "strategy": label,
                "accuracy": result.portfolio_accuracy,
                "oracle": result.oracle_accuracy,
                "chance": result.chance_accuracy,
                "odds_lift": result.lift,
            }
        )
    rows.append(
        {
            "strategy": "oracle (greedy runs)",
            "accuracy": oracle,
            "oracle": None,
            "chance": None,
            "odds_lift": None,
        }
    )
    return rows


def correlation_rows(results: Mapping[float, ExperimentResult]) -> list[Row]:
    """One row per rho in the correlation sweep (greedy allocation)."""
    return [
        {
            "rho": rho,
            "accuracy": result.portfolio_accuracy,
            "oracle": result.oracle_accuracy,
            "chance": result.chance_accuracy,
            "odds_lift": result.lift,
        }
        for rho, result in results.items()
    ]


def miscalibration_rows(
    results: Mapping[float, ExperimentResult], skill: float
) -> list[Row]:
    """One row per overconfidence level in the miscalibration sweep."""
    return [
        {
            "overconfidence": overconfidence,
            "accuracy": result.portfolio_accuracy,
            "effective_skill": skill * (1.0 - overconfidence),
            "odds_lift": result.lift,
        }
        for overconfidence, result in results.items()
    ]


def _column_keys(columns: Sequence[tuple[str, str, str | None]]) -> list[str]:
    return [key for key, _header, _fmt in columns]


def rows_to_csv(
    rows: Sequence[Row], columns: Sequence[tuple[str, str, str | None]]
) -> str:
    """Serialize rows to CSV; ``None`` becomes an empty field."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([header for _key, header, _fmt in columns])
    for row in rows:
        writer.writerow(
            ["" if row[key] is None else row[key] for key in _column_keys(columns)]
        )
    return buf.getvalue()


def rows_to_json(
    rows: Sequence[Row], columns: Sequence[tuple[str, str, str | None]]
) -> str:
    """Serialize rows to JSON; ``None`` becomes ``null``.

    Only the listed columns are included, in column order, so the
    JSON shape matches the CSV and Markdown tables exactly.
    """
    keys = _column_keys(columns)
    return json.dumps(
        [{key: row[key] for key in keys} for row in rows], indent=2
    ) + "\n"


def rows_to_markdown(
    rows: Sequence[Row], columns: Sequence[tuple[str, str, str | None]]
) -> str:
    """Serialize rows to a GitHub-flavoured Markdown table.

    Numbers use the column's format spec; ``None`` renders as ``—``,
    matching the demo's printed tables.
    """
    headers = [header for _key, header, _fmt in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for key, _header, fmt in columns:
            value = row[key]
            if value is None:
                cells.append("—")
            elif fmt is not None:
                cells.append(format(value, fmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

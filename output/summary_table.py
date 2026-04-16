"""Generates the top-N ranked summary table as a Markdown string and a
display-ready pandas DataFrame.

Prints a formatted summary table to the console using only built-in
string formatting (no ``tabulate`` dependency).  Designed for quick
terminal review after a pipeline run.

Data Lineage Contract
---------------------
Upstream producers:
    - ``screener.composite_ranker.generate_shortlist``
      → ``shortlist_df`` with columns: ``ticker``, ``composite_score``,
        ``rank``.  Optional enrichment columns: ``iv_weighted``,
        ``current_price_usd``, ``margin_of_safety_pct``,
        ``recommendation``.

Downstream consumers:
    - Terminal output (stdout) for operator review.

Config dependencies:
    None.  This module operates on the DataFrame passed to it.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any, TextIO

import pandas as pd

logger = logging.getLogger(__name__)

# Maximum rows to display in the console summary
_DEFAULT_MAX_ROWS = 20


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_str(value: Any, width: int, align: str = "<") -> str:
    """Format a value as a fixed-width string.

    Parameters
    ----------
    value:
        Value to format.  ``NaN`` and ``None`` render as ``"—"``.
    width:
        Target character width.
    align:
        Alignment: ``"<"`` (left), ``">"`` (right), ``"^"`` (center).

    Returns
    -------
    str
        Formatted, truncated string of exactly *width* characters.
    """
    if value is None:
        text = "—"
    elif isinstance(value, float) and math.isnan(value):
        text = "—"
    else:
        text = str(value)

    # Truncate if too long
    if len(text) > width:
        text = text[: width - 1] + "…"

    return f"{text:{align}{width}}"


def _safe_score(value: Any) -> str:
    """Format a composite score with one decimal place.

    Parameters
    ----------
    value:
        Score value.

    Returns
    -------
    str
        Formatted score like ``"78.3"`` or ``"—"`` if unavailable.
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "—"
        return f"{v:.1f}"
    except (TypeError, ValueError):
        return "—"


def _build_header_line(col_widths: dict[str, int]) -> str:
    """Build the table header line.

    Parameters
    ----------
    col_widths:
        Mapping of column label → display width.

    Returns
    -------
    str
        Formatted header row.
    """
    parts: list[str] = []
    for label, width in col_widths.items():
        parts.append(f"{label:^{width}}")
    return " | ".join(parts)


def _build_separator(col_widths: dict[str, int]) -> str:
    """Build the separator line under the header.

    Parameters
    ----------
    col_widths:
        Mapping of column label → display width.

    Returns
    -------
    str
        Dashed separator.
    """
    return "-+-".join("-" * w for w in col_widths.values())


def _safe_dollar(value: Any) -> str:
    """Format a dollar amount with two decimal places.

    Parameters
    ----------
    value:
        Dollar value.

    Returns
    -------
    str
        Formatted price like ``"$150.25"`` or ``"—"`` if unavailable.
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "—"
        return f"${v:.2f}"
    except (TypeError, ValueError):
        return "—"


def _safe_pct(value: Any) -> str:
    """Format a decimal fraction as a percentage with one decimal place.

    Parameters
    ----------
    value:
        Decimal fraction (e.g. 0.25 → ``"25.0%"``).

    Returns
    -------
    str
        Formatted percentage or ``"—"`` if unavailable.
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "—"
        return f"{v * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _build_row(
    row: pd.Series,
    col_widths: dict[str, int],
) -> str:
    """Format a single data row from a shortlist DataFrame.

    Columns: Rank, Ticker, Score, IV, Price, MoS%, Rec — matching the
    spec in the task instructions.

    Parameters
    ----------
    row:
        One row from the shortlist DataFrame.
    col_widths:
        Column widths for alignment.

    Returns
    -------
    str
        Formatted table row.
    """
    rank_val = row.get("rank", "—")
    ticker_val = row.get("ticker", "—")
    score_val = _safe_score(row.get("composite_score"))
    iv_val = _safe_dollar(row.get("iv_weighted"))
    price_val = _safe_dollar(row.get("current_price_usd"))
    mos_val = _safe_pct(row.get("margin_of_safety_pct"))
    rec_val = row.get("recommendation", "—")

    parts: list[str] = [
        _safe_str(rank_val, col_widths["Rank"], ">"),
        _safe_str(ticker_val, col_widths["Ticker"]),
        _safe_str(score_val, col_widths["Score"], ">"),
        _safe_str(iv_val, col_widths["IV"], ">"),
        _safe_str(price_val, col_widths["Price"], ">"),
        _safe_str(mos_val, col_widths["MoS%"], ">"),
        _safe_str(rec_val, col_widths["Rec"]),
    ]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def print_summary_to_console(
    shortlist_df: pd.DataFrame,
    max_rows: int = _DEFAULT_MAX_ROWS,
    output: TextIO | None = None,
) -> None:
    """Pretty-print a ranked shortlist table to the terminal.

    Displays up to *max_rows* rows in a formatted ASCII table without
    requiring any external table-formatting library.

    Parameters
    ----------
    shortlist_df:
        Shortlisted securities DataFrame.  Expected columns:
        ``ticker``, ``composite_score``, ``rank``.
        Optional: ``iv_weighted``, ``current_price_usd``,
        ``margin_of_safety_pct``, ``recommendation``.
    max_rows:
        Maximum rows to display (default 20).
    output:
        Writable text stream (default ``sys.stdout``).
    """
    if output is None:
        output = sys.stdout

    if shortlist_df.empty:
        output.write("\nNo stocks in shortlist.\n")
        return

    # Column widths — matches spec: Rank, Ticker, Score, IV, Price, MoS%, Rec
    col_widths = {
        "Rank": 4,
        "Ticker": 8,
        "Score": 6,
        "IV": 10,
        "Price": 10,
        "MoS%": 7,
        "Rec": 6,
    }

    # Title
    output.write("\n")
    output.write("=" * 70 + "\n")
    output.write("  Buffett Screener — Top Ranked Securities\n")
    output.write("=" * 70 + "\n\n")

    # Header and separator
    header = _build_header_line(col_widths)
    separator = _build_separator(col_widths)
    output.write(header + "\n")
    output.write(separator + "\n")

    # Data rows
    display_df = shortlist_df.head(max_rows)
    for _, row in display_df.iterrows():
        output.write(_build_row(row, col_widths) + "\n")

    # Footer
    total = len(shortlist_df)
    shown = len(display_df)
    output.write(separator + "\n")
    output.write(f"  Showing {shown} of {total} shortlisted securities.\n\n")

    logger.info(
        "Summary table printed to console (%d of %d rows).",
        shown,
        total,
    )

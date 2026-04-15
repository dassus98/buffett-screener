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
        ``rank``, ``score_category``, and optional ``company_name``,
        ``sector``, ``exchange``.

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


def _build_row(
    row: pd.Series,
    col_widths: dict[str, int],
) -> str:
    """Format a single data row from a shortlist DataFrame.

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
    name_val = row.get("company_name", "—")
    score_val = _safe_score(row.get("composite_score"))
    category_val = row.get("score_category", "—")
    sector_val = row.get("sector", "—")
    exchange_val = row.get("exchange", "—")

    parts: list[str] = [
        _safe_str(rank_val, col_widths["Rank"], ">"),
        _safe_str(ticker_val, col_widths["Ticker"]),
        _safe_str(name_val, col_widths["Company"]),
        _safe_str(score_val, col_widths["Score"], ">"),
        _safe_str(category_val, col_widths["Category"]),
        _safe_str(sector_val, col_widths["Sector"]),
        _safe_str(exchange_val, col_widths["Exch"]),
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
        ``ticker``, ``composite_score``, ``rank``, ``score_category``.
        Optional: ``company_name``, ``sector``, ``exchange``.
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

    # Column widths
    col_widths = {
        "Rank": 4,
        "Ticker": 8,
        "Company": 25,
        "Score": 6,
        "Category": 12,
        "Sector": 20,
        "Exch": 6,
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

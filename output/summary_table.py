"""
output.summary_table
=====================
Generates the top-N summary table in both DataFrame and Markdown formats.

The summary table is the primary deliverable of each pipeline run —
a concise ranked view of the best stocks identified by the screener,
with key metrics and valuation highlights.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_acquisition.schema import MacroSnapshot


def generate_summary_table(
    ranked_df: pd.DataFrame,
    reports: list,
    macro: MacroSnapshot,
    config: dict,
) -> str:
    """
    Render the summary table as a Markdown string using the Jinja2 template.

    Args:
        ranked_df: Ranked DataFrame from rank_universe() (top N rows).
        reports:   List of ValuationReport objects aligned with ranked_df rows.
        macro:     MacroSnapshot for the run date (used in table header).
        config:    Full filter_config.yaml dict.

    Returns:
        Rendered Markdown string for the summary table.

    Logic:
        1. Build a list of row dicts from ranked_df + reports
           (merge metrics, scores, recommendation, and margin of safety)
        2. Load summary_table_template.md via Jinja2
        3. Pass context (rows, macro, config weights, filter stats) to template
        4. Render and return
    """
    ...


def build_summary_dataframe(
    ranked_df: pd.DataFrame,
    reports: list,
) -> pd.DataFrame:
    """
    Merge ranked metrics with valuation report results into a display DataFrame.

    Args:
        ranked_df: Ranked DataFrame with metric and score columns.
        reports:   List of ValuationReport objects.

    Returns:
        Merged DataFrame with one row per ticker, containing:
            rank, ticker, name, sector, composite_score, quality_score,
            value_score, recommendation, roic_avg_5yr, gross_margin_avg_5yr,
            pe_ttm, ev_to_ebitda, mos_vs_base (margin of safety vs base case)

    Logic:
        1. Extract (ticker, recommendation, mos_vs_base) from each report
        2. Merge with ranked_df on ticker
        3. Select and reorder columns for display
        4. Return the merged DataFrame
    """
    ...


def compute_filter_elimination_stats(
    full_universe_df: pd.DataFrame,
    post_exclusion_df: pd.DataFrame,
    post_hard_filter_df: pd.DataFrame,
    filter_masks: dict[str, pd.Series],
) -> dict[str, int]:
    """
    Compute how many tickers each filter eliminated.

    Args:
        full_universe_df:      All universe tickers before any filtering.
        post_exclusion_df:     After sector/type exclusions.
        post_hard_filter_df:   After all hard filters.
        filter_masks:          Dict of filter_name → boolean Series from hard_filters.

    Returns:
        Dict mapping filter_name → count of tickers eliminated by that filter alone.

    Logic:
        For each filter mask, count the tickers that passed all *other* filters
        but failed this one. This gives a marginal elimination count per filter.
    """
    ...

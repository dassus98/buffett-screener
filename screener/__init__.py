"""
screener
========
Applies hard filters, soft filters, and composite ranking to the computed
metrics universe to produce a final ranked list of stock candidates.

Pipeline:
    1. exclusions.apply_exclusions()    → remove disqualified sectors/types
    2. hard_filters.apply_hard_filters() → binary pass/fail on minimum thresholds
    3. soft_filters.apply_soft_filters() → score-weighted soft criteria
    4. composite_ranker.rank()           → sort by composite score, return top N

The screener is stateless. It operates entirely on DataFrames of metrics
and returns DataFrames of results. No I/O.

Public surface:
    run_screener(metrics_df, config) → ranked_df
"""

from screener.exclusions import apply_exclusions
from screener.hard_filters import apply_hard_filters
from screener.soft_filters import apply_soft_filters
from screener.composite_ranker import rank_universe


def run_screener(metrics_df, config: dict):
    """
    Run the full screening pipeline on a universe metrics DataFrame.

    Args:
        metrics_df: DataFrame with one row per ticker containing all
                    computed metric columns plus profile metadata
                    (sector, is_adr, is_spac, etc.).
        config:     Full filter_config.yaml dict.

    Returns:
        DataFrame of passing tickers sorted by composite score descending,
        limited to config["output"]["top_n_stocks"] rows.
        Includes columns for each filter result and the composite score.

    Logic:
        1. apply_exclusions() → drop excluded sectors, ADRs, SPACs
        2. apply_hard_filters() → keep only rows passing ALL hard filters
        3. apply_soft_filters() → add soft_score column
        4. rank_universe() → add composite_score, sort, return top N
    """
    ...


__all__ = [
    "run_screener",
    "apply_exclusions",
    "apply_hard_filters",
    "apply_soft_filters",
    "rank_universe",
]

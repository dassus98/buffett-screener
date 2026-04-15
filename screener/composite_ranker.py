"""Ranks survivors by composite score and produces the investment shortlist.

Data Lineage Contract
---------------------
Upstream producers:
    - ``screener.soft_filters.apply_soft_scores`` → ``ranked_df``
      (survivors enriched with composite scores, sorted descending,
      with ``rank`` column).
    - ``screener.hard_filters.apply_hard_filters`` → ``filter_log_df``
      (per-ticker per-filter pass/fail log for all evaluated tickers).

Downstream consumers:
    - Pipeline runner / ``__main__`` → consumes the shortlist DataFrame
      and summary dict for reporting and output.
    - Module 4 (valuation_reports) → reads shortlist for detailed reports.

Config dependencies (all via ``get_threshold``):
    - ``recommendations.strong_buy_min_score``  (default 80)
    - ``recommendations.buy_min_score``         (default 70)
    - ``recommendations.hold_min_score``        (default 60)
    - ``output.shortlist_size``                 (default 50)

Public functions
----------------
generate_shortlist
    Select the top-N tickers from the ranked population, add percentile
    and score-category metadata columns.
generate_screener_summary
    Aggregate pipeline statistics into a summary dict for reporting.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_score_thresholds() -> tuple[float, float, float]:
    """Read score-category thresholds from config via :func:`get_threshold`.

    Returns
    -------
    tuple[float, float, float]
        ``(strong_buy_min, buy_min, hold_min)`` from the
        ``recommendations`` section of ``filter_config.yaml``.

    Raises
    ------
    ConfigError
        If any of the three config keys is missing.
    """
    # --- Fail-fast: raises ConfigError if key is absent, preventing
    #     silent use of hardcoded fallback thresholds.
    strong_buy_min = float(get_threshold("recommendations.strong_buy_min_score"))
    buy_min = float(get_threshold("recommendations.buy_min_score"))
    hold_min = float(get_threshold("recommendations.hold_min_score"))
    return strong_buy_min, buy_min, hold_min


def _assign_score_category(
    score: float,
    strong_buy_min: float,
    buy_min: float,
    hold_min: float,
) -> str:
    """Map a composite score to a human-readable category.

    Parameters
    ----------
    score:
        Composite score (0–100 scale).
    strong_buy_min:
        Minimum score for "Strong Buy".
    buy_min:
        Minimum score for "Buy".
    hold_min:
        Minimum score for "Hold".

    Returns
    -------
    str
        One of ``"Strong Buy"``, ``"Buy"``, ``"Hold"``, ``"Weak"``.
        ``NaN`` scores map to ``"Weak"``.
    """
    # --- NaN scores are unranked; treat as "Weak" rather than crashing.
    if math.isnan(score):
        return "Weak"
    # --- Cascade through thresholds (highest first).
    if score >= strong_buy_min:
        return "Strong Buy"
    if score >= buy_min:
        return "Buy"
    if score >= hold_min:
        return "Hold"
    return "Weak"


def _compute_percentile(rank: int, total: int) -> float:
    """Compute percentile for a rank within a population.

    Parameters
    ----------
    rank:
        1-based rank (1 = best).
    total:
        Total number of items in the full ranked population.

    Returns
    -------
    float
        Percentile as a value between 0 and 100.  Rank 1 of 100 → 100.0,
        rank 50 of 100 → 51.0, rank 100 of 100 → 1.0.
    """
    if total <= 0:
        return 0.0
    return ((total - rank + 1) / total) * 100.0


def _build_score_summary(scores: pd.Series) -> dict[str, float | None]:
    """Compute top / median / bottom score statistics.

    Parameters
    ----------
    scores:
        Series of composite scores (may contain NaN).

    Returns
    -------
    dict
        Keys: ``top_score``, ``median_score``, ``bottom_score``.
    """
    valid = scores.dropna()
    if valid.empty:
        return {"top_score": None, "median_score": None, "bottom_score": None}
    return {
        "top_score": float(valid.max()),
        "median_score": float(valid.median()),
        "bottom_score": float(valid.min()),
    }


def _build_distribution(
    df: pd.DataFrame,
    column: str,
) -> dict[str, int]:
    """Return value-counts dict for *column*, or empty dict if absent.

    Parameters
    ----------
    df:
        DataFrame to analyse.
    column:
        Column name whose value distribution is needed.

    Returns
    -------
    dict[str, int]
    """
    if df.empty or column not in df.columns:
        return {}
    return df[column].value_counts().to_dict()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_shortlist(
    ranked_df: pd.DataFrame,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Select the top-N stocks from *ranked_df* and add metadata columns.

    Parameters
    ----------
    ranked_df:
        Output of :func:`~screener.soft_filters.apply_soft_scores` —
        already sorted descending by ``composite_score`` with a ``rank``
        column.
    top_n:
        How many tickers to include.  When ``None`` (default), reads
        ``output.shortlist_size`` from ``filter_config.yaml``.

    Returns
    -------
    pd.DataFrame
        Top-N rows from *ranked_df* with additional columns:

        * ``rank`` — preserved from input (position in full population).
        * ``percentile`` — position expressed as a percentile of the
          full ranked population.
        * ``score_category`` — ``"Strong Buy"`` / ``"Buy"`` / ``"Hold"``
          / ``"Weak"`` based on ``composite_score``.
    """
    # --- Step 1: Guard — empty input produces empty output.
    if ranked_df.empty:
        logger.warning("generate_shortlist: ranked_df is empty.")
        return pd.DataFrame()

    # --- Step 2: Resolve top_n from config if not explicitly provided.
    if top_n is None:
        top_n = int(get_threshold("output.shortlist_size"))

    # --- Step 3: Slice the top N rows from the ranked population.
    total_population = len(ranked_df)
    shortlist = ranked_df.head(top_n).copy()

    # --- Step 4: Ensure rank column exists (should be set by soft_filters).
    if "rank" not in shortlist.columns:
        shortlist["rank"] = range(1, len(shortlist) + 1)

    # --- Step 5: Compute percentile relative to the full ranked population.
    shortlist["percentile"] = shortlist["rank"].apply(
        lambda r: _compute_percentile(int(r), total_population),
    )

    # --- Step 6: Assign score category based on composite_score thresholds.
    strong_buy_min, buy_min, hold_min = _read_score_thresholds()
    shortlist["score_category"] = shortlist["composite_score"].apply(
        lambda s: _assign_score_category(
            float(s) if pd.notna(s) else float("nan"),
            strong_buy_min,
            buy_min,
            hold_min,
        ),
    )

    logger.info(
        "Shortlist: %d of %d tickers selected (top_n=%d).",
        len(shortlist),
        total_population,
        top_n,
    )

    return shortlist


def generate_screener_summary(
    full_ranked_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    filter_log_df: pd.DataFrame,
    total_universe: int = 0,
) -> dict[str, Any]:
    """Aggregate pipeline statistics into a summary dict for reporting.

    Parameters
    ----------
    full_ranked_df:
        All Tier 1 survivors, ranked by composite score (output of
        :func:`~screener.soft_filters.apply_soft_scores`).
    shortlist_df:
        Top-N shortlist (output of :func:`generate_shortlist`).
    filter_log_df:
        Tier 1 hard-filter log (output of
        :func:`~screener.hard_filters.apply_hard_filters`).
    total_universe:
        Size of the full ticker universe BEFORE exclusions.  Passed by the
        pipeline runner; defaults to ``0`` when unavailable.

    Returns
    -------
    dict[str, Any]
        Keys:

        * ``total_universe`` — tickers before any filtering.
        * ``after_exclusions`` — unique tickers evaluated by Tier 1
          hard filters (i.e. tickers remaining after exclusions).
        * ``after_tier1`` — tickers surviving all Tier 1 filters.
        * ``shortlisted`` — tickers in the shortlist.
        * ``top_score``, ``median_score``, ``bottom_score`` — score
          statistics from *full_ranked_df*.
        * ``sector_distribution`` — ``{sector: count}`` from shortlist.
        * ``exchange_distribution`` — ``{exchange: count}`` from shortlist.
    """
    summary: dict[str, Any] = {}

    # --- Step 1: Record the pre-exclusion universe size.
    summary["total_universe"] = total_universe

    # --- Step 2: Derive after-exclusions count from the filter log.
    #     The filter log contains one row per ticker × filter, so the
    #     number of unique tickers = tickers evaluated after exclusions.
    if not filter_log_df.empty and "ticker" in filter_log_df.columns:
        summary["after_exclusions"] = int(filter_log_df["ticker"].nunique())
    else:
        summary["after_exclusions"] = 0

    # --- Step 3: Record Tier 1 survivors and shortlist counts.
    summary["after_tier1"] = len(full_ranked_df)
    summary["shortlisted"] = len(shortlist_df)

    # --- Step 4: Compute score statistics (top / median / bottom).
    if not full_ranked_df.empty and "composite_score" in full_ranked_df.columns:
        summary.update(_build_score_summary(full_ranked_df["composite_score"]))
    else:
        summary.update(
            {"top_score": None, "median_score": None, "bottom_score": None},
        )

    # --- Step 5: Compute sector and exchange distributions from shortlist.
    summary["sector_distribution"] = _build_distribution(
        shortlist_df, "sector",
    )
    summary["exchange_distribution"] = _build_distribution(
        shortlist_df, "exchange",
    )

    logger.info(
        "Summary: %d universe → %d after exclusions → %d after Tier 1 → %d shortlisted.",
        summary["total_universe"],
        summary["after_exclusions"],
        summary["after_tier1"],
        summary["shortlisted"],
    )

    return summary

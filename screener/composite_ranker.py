"""Ranks survivors by composite score and produces the investment shortlist.

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

from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_score_thresholds() -> tuple[float, float, float]:
    """Read score-category thresholds from config.

    Returns
    -------
    tuple[float, float, float]
        ``(strong_buy_min, buy_min, hold_min)`` from the
        ``recommendations`` section of ``filter_config.yaml``.
    """
    cfg = get_config()
    rec = cfg.get("recommendations", {})
    strong_buy_min = float(rec.get("strong_buy_min_score", 80))
    buy_min = float(rec.get("buy_min_score", 70))
    hold_min = float(rec.get("hold_min_score", 60))
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
    if math.isnan(score):
        return "Weak"
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
    if ranked_df.empty:
        logger.warning("generate_shortlist: ranked_df is empty.")
        return pd.DataFrame()

    cfg = get_config()
    if top_n is None:
        top_n = int(cfg.get("output", {}).get("shortlist_size", 50))

    total_population = len(ranked_df)
    shortlist = ranked_df.head(top_n).copy()

    # Ensure rank column exists
    if "rank" not in shortlist.columns:
        shortlist["rank"] = range(1, len(shortlist) + 1)

    # Percentile relative to full population
    shortlist["percentile"] = shortlist["rank"].apply(
        lambda r: _compute_percentile(int(r), total_population),
    )

    # Score category
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

    Returns
    -------
    dict[str, Any]
        Keys:

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

    # Counts derived from filter log
    if not filter_log_df.empty and "ticker" in filter_log_df.columns:
        summary["after_exclusions"] = int(filter_log_df["ticker"].nunique())
    else:
        summary["after_exclusions"] = 0

    summary["after_tier1"] = len(full_ranked_df)
    summary["shortlisted"] = len(shortlist_df)

    # Score statistics
    if not full_ranked_df.empty and "composite_score" in full_ranked_df.columns:
        summary.update(_build_score_summary(full_ranked_df["composite_score"]))
    else:
        summary.update(
            {"top_score": None, "median_score": None, "bottom_score": None},
        )

    # Distributions from shortlist
    summary["sector_distribution"] = _build_distribution(
        shortlist_df, "sector",
    )
    summary["exchange_distribution"] = _build_distribution(
        shortlist_df, "exchange",
    )

    logger.info(
        "Summary: %d evaluated → %d after Tier 1 → %d shortlisted.",
        summary["after_exclusions"],
        summary["after_tier1"],
        summary["shortlisted"],
    )

    return summary

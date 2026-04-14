"""Applies Tier 2 soft scoring by joining composite scores onto hard-filter survivors.

The heavy lifting of computing per-criterion scores (0–100) and the weighted
composite score is performed upstream in
``metrics_engine.composite_score.compute_composite_score``.  This module
joins those pre-computed scores onto the set of tickers that survived
Tier 1 hard filtering, adds a 1-based rank column, and returns the result
sorted by ``composite_score`` descending.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _join_scores(
    survivors_df: pd.DataFrame,
    composite_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join composite-score columns onto *survivors_df* by ticker.

    Parameters
    ----------
    survivors_df:
        Tickers that passed Tier 1 hard filters.
    composite_scores_df:
        Pre-computed composite scores (``ticker``, ``composite_score``,
        and ``score_*`` columns).

    Returns
    -------
    pd.DataFrame
        *survivors_df* enriched with score columns.  Tickers absent
        from *composite_scores_df* receive ``NaN`` for all score columns.
    """
    if composite_scores_df.empty:
        logger.warning("_join_scores: composite_scores_df is empty.")
        result = survivors_df.copy()
        result["composite_score"] = float("nan")
        return result

    score_cols = [c for c in composite_scores_df.columns if c != "ticker"]
    merged = survivors_df.merge(
        composite_scores_df[["ticker"] + score_cols],
        on="ticker",
        how="left",
        suffixes=("", "_dup"),
    )
    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)
    return merged


def _add_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Sort descending by ``composite_score`` and append a 1-based ``rank``.

    Parameters
    ----------
    df:
        DataFrame with a ``composite_score`` column.

    Returns
    -------
    pd.DataFrame
        Sorted copy with an integer ``rank`` column (1 = best).
    """
    sorted_df = (
        df.sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    sorted_df["rank"] = range(1, len(sorted_df) + 1)
    return sorted_df


def _log_tier2_summary(ranked_df: pd.DataFrame) -> None:
    """Log the Tier 2 summary including the top-scoring ticker.

    Parameters
    ----------
    ranked_df:
        Ranked DataFrame (output of :func:`_add_rank`).
    """
    if ranked_df.empty:
        return
    n_scored = int(ranked_df["composite_score"].notna().sum())
    top = ranked_df.iloc[0]
    logger.info(
        "Tier 2: %d tickers scored. Top: %s (composite=%.1f).",
        n_scored,
        top.get("ticker", "?"),
        float(top.get("composite_score", 0)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_soft_scores(
    survivors_df: pd.DataFrame,
    composite_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join composite scores onto Tier 1 survivors and rank them.

    Parameters
    ----------
    survivors_df:
        Tickers that passed all Tier 1 hard filters (output of
        :func:`~screener.hard_filters.apply_hard_filters`).  Must
        contain a ``ticker`` column.
    composite_scores_df:
        Pre-computed composite scores from
        :func:`~metrics_engine.composite_score.compute_all_composite_scores`.
        Must contain ``ticker`` and ``composite_score`` columns, and
        optionally ``score_*`` per-criterion columns.

    Returns
    -------
    pd.DataFrame
        *survivors_df* enriched with composite-score columns and a
        ``rank`` column (1 = highest composite score), sorted descending
        by ``composite_score``.  Tickers present in *survivors_df* but
        absent from *composite_scores_df* receive ``composite_score = NaN``
        and are ranked last.
    """
    if survivors_df.empty:
        logger.warning("apply_soft_scores: no survivors to score.")
        return pd.DataFrame()

    scored = _join_scores(survivors_df, composite_scores_df)
    ranked = _add_rank(scored)
    _log_tier2_summary(ranked)
    return ranked

"""
screener.soft_filters
======================
Applies continuous scoring criteria that reward desirable qualities without
hard binary elimination.

Unlike hard filters, soft filters produce a score contribution for each ticker.
A company that barely misses a soft threshold is penalised, not eliminated.

Soft scoring categories (all contribute to composite rank, not pass/fail):
    - Margin consistency  (low std dev in gross margin → higher score)
    - Revenue growth      (higher CAGR → higher score)
    - EPS growth          (higher CAGR → higher score)
    - CapEx intensity     (lower → higher score; asset-light preference)
    - Owner earnings yield (higher → higher score; key value metric)

Each category returns a score from 0 to 1. The soft_score column is the
weighted average of all category scores, with weights from the composite_weights
section of filter_config.yaml.
"""

from __future__ import annotations

import pandas as pd


def apply_soft_filters(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Add a "soft_score" column to the metrics DataFrame.

    Args:
        df:     Metrics DataFrame (post hard-filter, so only passing tickers).
        config: Full filter_config.yaml dict. Reads "soft_filters" and
                "composite_weights" sections.

    Returns:
        Copy of df with additional columns:
            "soft_score_margin_consistency"  — 0–1
            "soft_score_growth"              — 0–1
            "soft_score_capex"               — 0–1
            "soft_score_owner_earnings"      — 0–1
            "soft_score"                     — weighted average of all components

    Logic:
        1. Compute each component score via the individual functions below
        2. Apply weights from config["composite_weights"]
        3. Compute weighted average as "soft_score"
        4. Return augmented DataFrame
    """
    ...


def score_margin_consistency(
    df: pd.DataFrame,
    max_gross_margin_std: float,
    max_operating_margin_std: float,
) -> pd.Series:
    """
    Score margin consistency for each ticker on a 0–1 scale.

    Args:
        df:                        Metrics DataFrame. Must contain
                                   "gross_margin_std_5yr" and
                                   "operating_margin_std_5yr".
        max_gross_margin_std:      Threshold from config (e.g. 0.05 = 5pp).
        max_operating_margin_std:  Threshold from config.

    Returns:
        Series of scores from 0.0 (high variance) to 1.0 (perfectly stable).

    Logic:
        For each ticker:
            gross_score = max(0, 1 - (gross_margin_std / max_gross_margin_std))
            op_score    = max(0, 1 - (operating_margin_std / max_operating_margin_std))
            score = (gross_score + op_score) / 2
    """
    ...


def score_revenue_growth(
    df: pd.DataFrame,
    min_cagr: float,
    target_cagr: float = 0.15,
) -> pd.Series:
    """
    Score 5-year revenue CAGR for each ticker on a 0–1 scale.

    Args:
        df:          Metrics DataFrame. Must contain "revenue_cagr_5yr".
        min_cagr:    CAGR threshold below which score = 0.
        target_cagr: CAGR at which score = 1.0. Linear interpolation between
                     min_cagr and target_cagr.

    Returns:
        Series of scores from 0.0 to 1.0, clipped at 1.0 for CAGR > target_cagr.

    Logic:
        score = clip((revenue_cagr_5yr - min_cagr) / (target_cagr - min_cagr), 0, 1)
    """
    ...


def score_capex_intensity(
    df: pd.DataFrame,
    max_capex_to_revenue: float,
    target_capex_to_revenue: float = 0.02,
) -> pd.Series:
    """
    Score CapEx intensity inversely (lower CapEx/Revenue → higher score).

    Args:
        df:                       Metrics DataFrame. Must contain
                                  "capex_to_revenue_avg_5yr".
        max_capex_to_revenue:     Threshold above which score = 0 (e.g. 0.10).
        target_capex_to_revenue:  Ratio at which score = 1.0.
                                  Linear interpolation between target and max.

    Returns:
        Series of scores from 0.0 to 1.0.

    Logic:
        score = clip((max_capex_to_revenue - capex_to_revenue) /
                     (max_capex_to_revenue - target_capex_to_revenue), 0, 1)
    """
    ...


def score_owner_earnings_yield(
    df: pd.DataFrame,
    min_yield: float,
    target_yield: float = 0.08,
) -> pd.Series:
    """
    Score owner earnings yield (higher yield → higher score).

    Args:
        df:           Metrics DataFrame. Must contain "owner_earnings_yield".
        min_yield:    Minimum yield (decimal) for a non-zero score.
        target_yield: Yield at which score = 1.0 (linear interpolation).

    Returns:
        Series of scores from 0.0 to 1.0.

    Logic:
        score = clip((owner_earnings_yield - min_yield) /
                     (target_yield - min_yield), 0, 1)
    """
    ...

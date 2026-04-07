"""
metrics_engine.growth
======================
Computes revenue, earnings, and book value growth metrics.

Buffett wants businesses that grow their intrinsic value per share over time.
The key growth metrics are:
    - Revenue CAGR (5yr and 10yr)
    - EPS CAGR (5yr and 10yr)          — per-share growth matters more than total
    - Book Value per Share CAGR (5yr, 10yr)
    - Owner Earnings CAGR (5yr)
    - Consistency of growth (no large negative years)

Note on EPS vs. Owner Earnings growth:
    EPS can be inflated by share buybacks even if the underlying business is not
    growing. The screener considers both, weighting owner earnings growth more
    heavily in the composite score.
"""

from __future__ import annotations

import math

from data_acquisition.schema import TickerDataBundle


def compute_growth(bundle: TickerDataBundle) -> dict[str, float]:
    """
    Compute all growth metrics for a TickerDataBundle.

    Args:
        bundle: TickerDataBundle with at least 5 years of income statement and
                balance sheet history.

    Returns:
        Dict with keys:
            "revenue_cagr_5yr"          — decimal
            "revenue_cagr_10yr"         — decimal (NaN if < 10 years of data)
            "eps_cagr_5yr"              — decimal
            "eps_cagr_10yr"             — decimal
            "bvps_cagr_5yr"             — book value per share CAGR (decimal)
            "bvps_cagr_10yr"            — decimal
            "revenue_growth_latest"     — most recent year YoY growth (decimal)
            "eps_growth_latest"         — most recent year YoY EPS growth (decimal)
            "growth_consistency_score"  — 0–1 score (1 = never had a negative growth year)

    Logic:
        1. Extract per-year series for revenue, eps_diluted, shareholders_equity,
           and shares_outstanding from bundle
        2. Compute BVPS = shareholders_equity / shares_outstanding for each year
        3. Call cagr() for each metric × each time horizon
        4. Call growth_consistency_score() using the revenue series
    """
    ...


def cagr(
    start_value: float,
    end_value: float,
    years: int,
) -> float:
    """
    Compute compound annual growth rate between two values.

    Args:
        start_value: Value at the beginning of the period. Must be positive.
        end_value:   Value at the end of the period. Must be positive.
        years:       Number of years between start and end.

    Returns:
        CAGR as a decimal (e.g. 0.10 for 10%). Returns NaN if start_value
        or end_value is non-positive, or if years <= 0.

    Formula:
        CAGR = (end_value / start_value) ** (1 / years) - 1
    """
    ...


def yoy_growth(series: list[float]) -> list[float]:
    """
    Compute year-over-year percentage growth for each consecutive pair in a series.

    Args:
        series: List of annual values sorted ascending by year. Must have
                positive values for meaningful results.

    Returns:
        List of YoY growth rates as decimals, length = len(series) - 1.
        NaN is returned for any pair where the prior year is zero or negative.

    Formula:
        growth_t = (value_t / value_{t-1}) - 1
    """
    ...


def growth_consistency_score(growth_rates: list[float]) -> float:
    """
    Score the consistency of growth — penalising negative growth years.

    Args:
        growth_rates: List of annual YoY growth rates (decimals), NaN values
                      are excluded before scoring.

    Returns:
        Score from 0.0 to 1.0:
            1.0 = all years positive growth
            0.0 = majority of years negative or large reversals

    Logic:
        1. Count negative growth years (growth_rate < 0)
        2. score = 1 - (negative_years / total_years)
        3. Additionally apply a severity penalty: for each year with growth < -0.10
           (a 10%+ revenue drop), subtract an extra 0.1 from the score, clipped at 0
    """
    ...


def estimate_eps_from_fundamentals(
    net_income: float,
    shares_diluted: float,
) -> float:
    """
    Compute diluted EPS from fundamental components.

    Args:
        net_income:     Net income (USD thousands).
        shares_diluted: Weighted average diluted shares outstanding (thousands).

    Returns:
        EPS in USD per share. Returns NaN if shares_diluted is zero.

    Formula:
        EPS = (net_income × 1000) / (shares_diluted × 1000)
            = net_income / shares_diluted
        (both in thousands, so the scaling cancels)
    """
    ...


def book_value_per_share(
    shareholders_equity: float,
    shares_outstanding: float,
) -> float:
    """
    Compute book value per share.

    Args:
        shareholders_equity: Total book equity (USD thousands).
        shares_outstanding:  Period-end shares outstanding (thousands).

    Returns:
        Book value per share in USD. Returns NaN if shares_outstanding is zero.

    Formula:
        BVPS = shareholders_equity / shares_outstanding
        (both in thousands, so units cancel → USD per share)
    """
    ...

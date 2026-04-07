"""
valuation_reports.recommendation
==================================
Synthesises quantitative scores and valuation results into a structured
buy / watchlist / pass recommendation with a human-readable justification.

The recommendation framework mirrors Buffett's three criteria:
    1. Is it a wonderful business? (quality score)
    2. Is it run by honest, capable management? (qualitative — human judgment)
    3. Is it available at a fair price? (margin of safety)

This module automates criteria 1 and 3 only. Criterion 2 requires human review.

Recommendation tiers:
    STRONG_BUY  — top-decile quality + significant margin of safety
    BUY         — above-threshold quality + adequate margin of safety
    WATCHLIST   — high quality but price too high; monitor for entry
    PASS        — does not meet quality or valuation bar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from valuation_reports.margin_of_safety import MarginOfSafetyResult, PriceClassification


class Recommendation(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCHLIST = "WATCHLIST"
    PASS = "PASS"


@dataclass
class RecommendationResult:
    """Full recommendation output for a single stock."""

    ticker: str
    recommendation: Recommendation
    composite_score: float              # 0–100
    quality_score: float                # 0–100
    value_score: float                  # 0–100
    margin_of_safety: MarginOfSafetyResult
    key_strengths: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    justification: str = ""             # 2–4 sentence narrative summary


def generate_recommendation(
    ticker: str,
    composite_score: float,
    quality_score: float,
    value_score: float,
    margin_of_safety: MarginOfSafetyResult,
    metrics: dict,
    config: dict,
) -> RecommendationResult:
    """
    Generate a structured recommendation for a single stock.

    Args:
        ticker:             Ticker symbol.
        composite_score:    Overall composite score (0–100).
        quality_score:      Quality sub-score (0–100).
        value_score:        Value sub-score (0–100).
        margin_of_safety:   MarginOfSafetyResult from compute_margin_of_safety().
        metrics:            Full metrics dict for the stock.
        config:             Full filter_config.yaml dict (for threshold parameters).

    Returns:
        RecommendationResult with recommendation tier, key strengths/risks,
        and a justification narrative.

    Logic:
        1. Map PriceClassification to Recommendation:
               STRONG_BUY price + quality_score >= 75  → STRONG_BUY
               BUY price + quality_score >= 60          → BUY
               WATCHLIST price OR quality_score >= 60   → WATCHLIST
               Otherwise                                → PASS
        2. Identify key strengths from top 3 metrics above universe median
        3. Identify key risks from bottom 3 metrics or red flags
        4. Generate justification string from template
        5. Return RecommendationResult
    """
    ...


def identify_key_strengths(metrics: dict, threshold_pct: float = 75.0) -> list[str]:
    """
    Identify the top competitive strengths from the metrics dict.

    Args:
        metrics:        Full metrics dict for the stock.
        threshold_pct:  Percentile rank threshold above which a metric is a strength.

    Returns:
        List of 2–4 human-readable strength strings (e.g.
        "Consistently high gross margins (avg 52% over 5 years)").

    Logic:
        Check the following in priority order:
            - roic_avg_5yr > 0.15             → "High and consistent ROIC"
            - gross_margin_avg_5yr > 0.40     → "Strong pricing power (gross margin X%)"
            - gross_margin_std_5yr < 0.03     → "Stable margins (moat indicator)"
            - fcf_conversion_avg_5yr > 0.70   → "High FCF conversion"
            - revenue_cagr_5yr > 0.10         → "Consistent revenue growth (X% CAGR)"
            - has_net_cash                    → "Net cash balance sheet"
        Return up to 4 matching strengths as formatted strings
    """
    ...


def identify_key_risks(metrics: dict) -> list[str]:
    """
    Identify the top risks and watchpoints from the metrics dict.

    Args:
        metrics: Full metrics dict for the stock.

    Returns:
        List of 1–3 human-readable risk strings.

    Logic:
        Check in priority order:
            - revenue_cagr_5yr < 0.05        → "Slowing revenue growth"
            - debt_to_equity_latest > 1.0    → "Elevated leverage (D/E > 1)"
            - gross_margin_std_5yr > 0.05    → "Margin volatility"
            - capex_to_revenue_avg_5yr > 0.08 → "Capital-intensive model"
            - margin_trend == "contracting"  → "Contracting margins"
        Return up to 3 matching risk strings
    """
    ...


def format_justification(
    ticker: str,
    recommendation: Recommendation,
    quality_score: float,
    margin_of_safety_pct: float,
    key_strengths: list[str],
    key_risks: list[str],
) -> str:
    """
    Generate a 2–4 sentence justification narrative for the recommendation.

    Args:
        ticker:               Ticker symbol.
        recommendation:       Enum recommendation tier.
        quality_score:        Quality score 0–100.
        margin_of_safety_pct: MoS as a percentage (e.g. 35.0 for 35%).
        key_strengths:        List of strength strings from identify_key_strengths().
        key_risks:            List of risk strings from identify_key_risks().

    Returns:
        Formatted string suitable for inclusion in a Markdown report.

    Example output:
        "{ticker} earns a {recommendation.value} rating with a quality score of
        {quality_score:.0f}/100. The company demonstrates {strength1} and {strength2},
        with the stock currently offering a {mos:.0f}% margin of safety vs. base
        intrinsic value. Key watchpoints: {risk1}."
    """
    ...

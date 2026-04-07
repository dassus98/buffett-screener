"""
metrics_engine.composite_score
================================
Computes a single weighted composite score (0–100) that ranks stocks
by the combination of quality and value metrics.

The composite score is used by screener/composite_ranker.py to sort the
passing universe into a final ranked list.

Score components (weights configurable in filter_config.yaml):
    - ROIC score          (higher ROIC → higher score)
    - ROE score
    - Owner Earnings Yield score
    - Gross Margin Consistency score
    - Revenue CAGR score
    - FCF Margin score
    - Debt Safety score    (lower leverage → higher score)
    - Earnings Yield vs. Bond spread score

Design principle:
    Each component is first normalised to [0, 1] using empirical percentile
    ranks across the screened universe, then multiplied by its weight.
    This makes the score robust to outliers and independent of absolute scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_composite_score(
    metrics: dict[str, float],
    config: dict,
) -> float:
    """
    Compute the composite score for a single stock given its pre-computed metrics.

    Args:
        metrics: Flat dict of all computed metrics for the stock (output of
                 metrics_engine.compute_all_metrics()).
        config:  Full filter_config.yaml dict. The "composite_weights" section
                 defines the weight for each component.

    Returns:
        Composite score from 0.0 to 100.0. Higher = more attractive.

    Note:
        This function computes the *raw* weighted sum. Percentile normalisation
        across the universe is done in score_universe(), which requires the full
        universe DataFrame. Calling this function in isolation returns an
        un-normalised score useful for inspecting individual stocks.

    Logic:
        1. Extract component scores using _metric_to_component_score() for each
           metric listed in config["composite_weights"]
        2. Multiply each component score by its weight
        3. Sum and return (max possible = 100.0 if all components = 1.0 and weights sum to 1)
    """
    ...


def score_universe(
    universe_metrics: pd.DataFrame,
    config: dict,
) -> pd.Series:
    """
    Compute percentile-normalised composite scores for an entire universe.

    Args:
        universe_metrics: DataFrame with one row per ticker and metric columns
                          matching the keys expected by compute_composite_score().
        config:           Full filter_config.yaml dict.

    Returns:
        pandas Series indexed by ticker with composite scores from 0.0 to 100.0,
        sorted descending (best stocks first).

    Logic:
        1. For each metric column that is a composite weight component,
           replace raw values with their percentile rank across the universe
           (pandas.Series.rank(pct=True) × 100)
        2. For inverse metrics (e.g. debt_to_equity — lower is better),
           invert the percentile: rank = 100 - raw_rank
        3. Compute weighted sum of component percentiles per row
        4. Return as Series sorted descending
    """
    ...


def _metric_to_component_score(
    metric_name: str,
    value: float,
    percentile_rank: float,
) -> float:
    """
    Convert a single metric value to a 0–1 component score.

    Args:
        metric_name:      Name of the metric (used to determine inversion logic).
        value:            Raw metric value (used for sanity checks).
        percentile_rank:  Percentile rank of this value across the universe (0–100).

    Returns:
        Component score from 0.0 to 1.0.

    Logic:
        - For metrics where higher is better (ROIC, ROE, margins, yields):
              score = percentile_rank / 100
        - For metrics where lower is better (debt ratios, valuation multiples):
              score = (100 - percentile_rank) / 100
        - If value is NaN: return 0.0 (no credit for missing data)

    Inverse metrics (lower is better):
        debt_to_equity, net_debt_to_ebitda, capex_to_revenue, pe_ttm, ev_to_ebitda
    """
    ...


INVERSE_METRICS = frozenset({
    "debt_to_equity_latest",
    "debt_to_equity_avg_5yr",
    "net_debt_to_ebitda_latest",
    "net_debt_to_ebitda_avg_5yr",
    "capex_to_revenue_latest",
    "capex_to_revenue_avg_5yr",
    "pe_ttm",
    "pe_normalised",
    "pb_ratio",
    "ev_to_ebitda",
    "ev_to_ebit",
    "ev_to_fcf",
})
"""Set of metric names where a lower value is more desirable."""

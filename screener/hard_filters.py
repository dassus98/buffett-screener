"""
screener.hard_filters
======================
Applies binary (pass/fail) hard filters to the metrics universe.

A stock must pass ALL hard filters to advance to soft scoring.
Failing any single hard filter eliminates the stock from the screener output.

Hard filter categories (thresholds from filter_config.yaml):
    - Profitability: minimum years profitable, minimum gross/operating margin
    - Returns: minimum ROIC and ROE
    - Leverage: maximum D/E, max net debt/EBITDA, minimum interest coverage
    - Cash flow: minimum years of positive FCF, minimum FCF margin
    - Liquidity: minimum current ratio

Each individual filter function returns a boolean Series; the overall
apply_hard_filters() function ANDs all masks together.
"""

from __future__ import annotations

import pandas as pd


def apply_hard_filters(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply all hard filters to the metrics DataFrame.

    Args:
        df:     Metrics DataFrame with one row per ticker. Must contain all
                metric columns referenced by the hard filter config.
        config: Full filter_config.yaml dict. Reads "hard_filters" section.

    Returns:
        Tuple of (passing_df, failing_df).
            passing_df: Rows where ALL hard filter conditions are met.
            failing_df: Rows that failed at least one filter, with a
                        "hard_filter_failures" column listing which filters failed.

    Logic:
        1. Run each filter function, collecting boolean masks
        2. Record which filters failed for each ticker (for reporting)
        3. Return passing and failing DataFrames
    """
    ...


def filter_min_years_profitable(
    df: pd.DataFrame,
    min_years: int,
) -> pd.Series:
    """
    Pass tickers where net income was positive in at least `min_years` of the
    available historical periods.

    Args:
        df:        Metrics DataFrame. Must contain "years_positive_net_income" column.
        min_years: Minimum profitable years required (from config).

    Returns:
        Boolean Series (True = passes this filter).
    """
    ...


def filter_min_gross_margin(
    df: pd.DataFrame,
    min_gross_margin: float,
) -> pd.Series:
    """
    Pass tickers where the 5-year average gross margin meets the minimum threshold.

    Args:
        df:               Metrics DataFrame. Must contain "gross_margin_avg_5yr" column.
        min_gross_margin: Minimum gross margin as a decimal (e.g. 0.20 for 20%).

    Returns:
        Boolean Series (True = passes this filter).
    """
    ...


def filter_min_operating_margin(
    df: pd.DataFrame,
    min_operating_margin: float,
) -> pd.Series:
    """
    Pass tickers where the 5-year average operating margin meets the minimum threshold.

    Args:
        df:                   Metrics DataFrame. Must contain "operating_margin_avg_5yr".
        min_operating_margin: Minimum operating margin as a decimal.

    Returns:
        Boolean Series (True = passes this filter).
    """
    ...


def filter_min_roic(df: pd.DataFrame, min_roic: float) -> pd.Series:
    """
    Pass tickers where the 5-year average ROIC meets the minimum threshold.

    Args:
        df:       Metrics DataFrame. Must contain "roic_avg_5yr".
        min_roic: Minimum ROIC as a decimal (e.g. 0.10 for 10%).

    Returns:
        Boolean Series (True = passes this filter).
    """
    ...


def filter_min_roe(df: pd.DataFrame, min_roe: float) -> pd.Series:
    """
    Pass tickers where the 5-year average ROE meets the minimum threshold.

    Args:
        df:      Metrics DataFrame. Must contain "roe_avg_5yr".
        min_roe: Minimum ROE as a decimal.

    Returns:
        Boolean Series (True = passes this filter).
    """
    ...


def filter_max_debt_to_equity(
    df: pd.DataFrame,
    max_de: float,
) -> pd.Series:
    """
    Pass tickers where the latest debt-to-equity ratio does not exceed the maximum.

    Args:
        df:     Metrics DataFrame. Must contain "debt_to_equity_latest".
        max_de: Maximum D/E ratio allowed.

    Returns:
        Boolean Series (True = passes this filter).
        Note: Tickers with NaN D/E (negative equity) automatically fail.
    """
    ...


def filter_max_net_debt_to_ebitda(
    df: pd.DataFrame,
    max_ratio: float,
) -> pd.Series:
    """
    Pass tickers where net debt / EBITDA does not exceed the maximum.

    Args:
        df:        Metrics DataFrame. Must contain "net_debt_to_ebitda_latest".
        max_ratio: Maximum net debt/EBITDA allowed. Net cash positions (negative
                   values) always pass.

    Returns:
        Boolean Series (True = passes this filter).
    """
    ...


def filter_min_interest_coverage(
    df: pd.DataFrame,
    min_coverage: float,
) -> pd.Series:
    """
    Pass tickers where the interest coverage ratio meets the minimum threshold.

    Args:
        df:           Metrics DataFrame. Must contain "interest_coverage_latest".
        min_coverage: Minimum coverage ratio (EBIT / interest expense).
                      Tickers with no debt (coverage = inf) always pass.

    Returns:
        Boolean Series (True = passes this filter).
    """
    ...


def filter_min_fcf_margin(
    df: pd.DataFrame,
    min_fcf_margin: float,
) -> pd.Series:
    """
    Pass tickers where the 5-year average FCF margin meets the minimum threshold.

    Args:
        df:             Metrics DataFrame. Must contain "fcf_conversion_avg_5yr"
                        or a dedicated "fcf_margin_avg_5yr" column.
        min_fcf_margin: Minimum FCF margin as a decimal.

    Returns:
        Boolean Series (True = passes this filter).
    """
    ...


def filter_min_current_ratio(
    df: pd.DataFrame,
    min_ratio: float,
) -> pd.Series:
    """
    Pass tickers where the latest current ratio meets the minimum threshold.

    Args:
        df:        Metrics DataFrame. Must contain "current_ratio_latest".
        min_ratio: Minimum current ratio (default 1.0 from config).

    Returns:
        Boolean Series (True = passes this filter).
    """
    ...


def summarise_filter_results(
    df: pd.DataFrame,
    filter_masks: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Attach a human-readable failure summary column to the DataFrame.

    Args:
        df:           Full metrics DataFrame.
        filter_masks: Dict mapping filter_name → boolean Series (True = passes).

    Returns:
        Copy of df with two additional columns:
            "passed_all_hard_filters"   — bool
            "hard_filter_failures"      — comma-separated string of failed filter names,
                                          empty string for passing tickers

    Logic:
        For each row, collect names of filter_masks where the value is False,
        join with ", ", store in "hard_filter_failures".
    """
    ...

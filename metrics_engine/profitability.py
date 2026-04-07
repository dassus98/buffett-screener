"""
metrics_engine.profitability
=============================
Computes margin metrics and their consistency over time.

Buffett looks for businesses with durable competitive advantages (moats),
which manifest as consistently high and stable gross and operating margins.
Wide, stable margins indicate pricing power and low competitive threat.

Key insight: a *consistent* 40% gross margin is more valuable than a 60%
gross margin that fluctuates 20 percentage points year to year.

Metrics computed:
    - Gross margin (gross_profit / revenue)
    - Operating margin (operating_income / revenue)
    - Net margin (net_income / revenue)
    - EBITDA margin (ebitda / revenue)
    - Trailing averages (5yr, 10yr) for each margin
    - Standard deviation of each margin (consistency proxy)
    - Margin trend direction (expanding, stable, contracting)
"""

from __future__ import annotations

from data_acquisition.schema import TickerDataBundle


def compute_profitability(bundle: TickerDataBundle) -> dict[str, float]:
    """
    Compute all profitability metrics for a TickerDataBundle.

    Args:
        bundle: TickerDataBundle with income statement history populated.

    Returns:
        Dict with keys (all margin values are decimals, e.g. 0.40 for 40%):
            "gross_margin_latest"
            "gross_margin_avg_5yr"
            "gross_margin_avg_10yr"
            "gross_margin_std_5yr"        — standard deviation (consistency metric)
            "operating_margin_latest"
            "operating_margin_avg_5yr"
            "operating_margin_avg_10yr"
            "operating_margin_std_5yr"
            "net_margin_latest"
            "net_margin_avg_5yr"
            "net_margin_avg_10yr"
            "ebitda_margin_latest"
            "ebitda_margin_avg_5yr"
            "margin_trend"                — "expanding" | "stable" | "contracting"
            "years_positive_net_income"   — count of profitable years in history

    Logic:
        1. Compute per-year margin lists from income_statements
        2. Take latest, 5yr avg, 10yr avg, 5yr std for each margin type
        3. Compute margin_trend via _classify_margin_trend()
        4. Count years where net_income > 0
    """
    ...


def gross_margin(gross_profit: float, revenue: float) -> float:
    """
    Compute gross margin for a single period.

    Args:
        gross_profit: Gross profit (revenue - COGS) in USD thousands.
        revenue:      Total revenue in USD thousands.

    Returns:
        Gross margin as a decimal. Returns NaN if revenue is zero.

    Formula:
        gross_margin = gross_profit / revenue
    """
    ...


def operating_margin(operating_income: float, revenue: float) -> float:
    """
    Compute operating (EBIT) margin for a single period.

    Args:
        operating_income: EBIT from the income statement (USD thousands).
        revenue:          Total revenue (USD thousands).

    Returns:
        Operating margin as a decimal. Returns NaN if revenue is zero.
    """
    ...


def net_margin(net_income: float, revenue: float) -> float:
    """
    Compute net profit margin for a single period.

    Args:
        net_income: Net income after tax (USD thousands).
        revenue:    Total revenue (USD thousands).

    Returns:
        Net margin as a decimal. Can be negative (loss year).
    """
    ...


def ebitda_margin(ebitda: float, revenue: float) -> float:
    """
    Compute EBITDA margin for a single period.

    Args:
        ebitda:   EBITDA (USD thousands).
        revenue:  Total revenue (USD thousands).

    Returns:
        EBITDA margin as a decimal. Returns NaN if revenue is zero.
    """
    ...


def trailing_average(series: list[float], years: int) -> float:
    """
    Compute the arithmetic mean of the last `years` values in a series.

    Args:
        series: List of values sorted ascending by year. NaN values are excluded.
        years:  Number of trailing periods to include. If len(series) < years,
                uses all available values.

    Returns:
        Arithmetic mean, or NaN if no non-NaN values exist in the window.
    """
    ...


def trailing_std(series: list[float], years: int) -> float:
    """
    Compute the standard deviation of the last `years` values in a series.

    Args:
        series: List of values sorted ascending by year.
        years:  Number of trailing periods. Falls back to available data.

    Returns:
        Population standard deviation, or NaN if fewer than 2 non-NaN values.
    """
    ...


def _classify_margin_trend(
    margin_series: list[float],
    lookback_years: int = 5,
    threshold: float = 0.02,
) -> str:
    """
    Classify whether a margin series is expanding, stable, or contracting.

    Args:
        margin_series:   List of annual margin values (decimal), sorted ascending.
        lookback_years:  Number of trailing years to fit the trend line.
        threshold:       Minimum slope magnitude (in decimal margin per year)
                         to classify as expanding or contracting vs. stable.

    Returns:
        One of: "expanding" | "stable" | "contracting"

    Logic:
        1. Take the last `lookback_years` values
        2. Fit a simple linear regression (numpy.polyfit degree=1)
        3. If slope > +threshold → "expanding"
           If slope < -threshold → "contracting"
           Otherwise → "stable"
    """
    ...

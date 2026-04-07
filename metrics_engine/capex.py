"""
metrics_engine.capex
=====================
Computes capital expenditure intensity metrics.

Low CapEx intensity is one of the strongest moat proxies. Businesses that
generate high returns without needing to reinvest heavily in physical assets
(e.g. consumer brands, software, professional services) can compound owner
earnings far more reliably than capital-heavy industries.

Key metrics:
    - CapEx / Revenue (lower = more asset-light)
    - CapEx / Operating Cash Flow (lower = more FCF conversion)
    - CapEx / D&A (< 1 = not even maintaining assets; > 2 = heavy investment cycle)
    - Maintenance vs. Growth CapEx split (estimated)
    - Free Cash Flow conversion rate (FCF / EBITDA)
"""

from __future__ import annotations

from data_acquisition.schema import TickerDataBundle


def compute_capex_metrics(bundle: TickerDataBundle) -> dict[str, float]:
    """
    Compute all CapEx intensity metrics for the most recent period and 5yr averages.

    Args:
        bundle: TickerDataBundle with cash flow and income statement history.

    Returns:
        Dict with keys:
            "capex_to_revenue_latest"      — decimal
            "capex_to_revenue_avg_5yr"     — decimal
            "capex_to_ocf_latest"          — CapEx / operating cash flow (decimal)
            "capex_to_ocf_avg_5yr"         — decimal
            "capex_to_da_latest"           — CapEx / D&A (decimal)
            "capex_to_da_avg_5yr"          — decimal
            "fcf_conversion_latest"        — FCF / EBITDA (decimal)
            "fcf_conversion_avg_5yr"       — decimal
            "is_asset_light"               — True if capex_to_revenue_avg_5yr < 0.05

    Logic:
        1. Extract per-year CapEx, revenue, OCF, D&A, EBITDA, FCF from bundle
        2. Compute each ratio per year (handling division by zero as NaN)
        3. Return latest values and 5-year trailing averages
        4. Set is_asset_light flag based on 5yr average CapEx/revenue threshold
    """
    ...


def capex_to_revenue(capital_expenditures: float, revenue: float) -> float:
    """
    Compute CapEx as a percentage of revenue.

    Args:
        capital_expenditures: CapEx for the period (stored as a negative number,
                              USD thousands). Will be converted to absolute value.
        revenue:              Total revenue (USD thousands).

    Returns:
        CapEx / revenue as a positive decimal (e.g. 0.05 for 5%).
        Returns NaN if revenue is zero.

    Formula:
        capex_to_revenue = abs(capital_expenditures) / revenue
    """
    ...


def capex_to_operating_cash_flow(
    capital_expenditures: float,
    operating_cash_flow: float,
) -> float:
    """
    Compute the ratio of CapEx to operating cash flow.

    Args:
        capital_expenditures: CapEx (negative number, USD thousands).
        operating_cash_flow:  Operating cash flow (USD thousands).

    Returns:
        Ratio as a positive decimal. Returns NaN if operating_cash_flow <= 0.

    Formula:
        capex_to_ocf = abs(capital_expenditures) / operating_cash_flow
    """
    ...


def capex_to_depreciation(
    capital_expenditures: float,
    depreciation_amortization: float,
) -> float:
    """
    Compute the ratio of CapEx to Depreciation & Amortisation.

    Interpretation:
        < 1.0 → company is not replacing assets at the rate they depreciate
        ≈ 1.0 → maintenance mode; no net new investment
        > 1.5 → active investment cycle (could be growth or capital trap)

    Args:
        capital_expenditures:    CapEx (negative number, USD thousands).
        depreciation_amortization: D&A for the period (USD thousands, positive).

    Returns:
        Ratio as a positive decimal. Returns NaN if D&A is zero.
    """
    ...


def fcf_conversion_rate(fcf: float, ebitda: float) -> float:
    """
    Compute free cash flow conversion rate (FCF / EBITDA).

    High conversion (> 0.70) indicates quality earnings with minimal working
    capital drag and moderate CapEx needs.

    Args:
        fcf:    Free cash flow = OCF + CapEx (USD thousands). Can be negative.
        ebitda: EBITDA for the period (USD thousands).

    Returns:
        FCF / EBITDA as a decimal. Returns NaN if EBITDA is zero.
        Clipped to [-1, 2] to prevent extreme outlier values.
    """
    ...

"""
metrics_engine.owner_earnings
==============================
Computes Buffett's "owner earnings" — the most important single metric in
this screener.

Definition (from Berkshire Hathaway 1986 Annual Letter):
    Owner Earnings = Net Income
                   + Depreciation & Amortisation
                   - Capital Expenditures (maintenance portion only)
                   ± Changes in Working Capital

Because maintenance CapEx is not separately disclosed, we use two proxies:
    1. Conservative: all CapEx = maintenance (yields a lower, safer estimate)
    2. Graham: maintenance CapEx ≈ D&A (assumes D&A approximates economic wear)
       → Owner Earnings ≈ Free Cash Flow in this case
    3. Regression: estimate maintenance CapEx from revenue growth rate
       (the portion of CapEx attributable to growth is excluded)

The default proxy is method 1 (conservative). Callers can request other methods.
"""

from __future__ import annotations

from typing import Literal

from data_acquisition.schema import TickerDataBundle


CapExMethod = Literal["conservative", "graham", "regression"]


def compute_owner_earnings(
    bundle: TickerDataBundle,
    capex_method: CapExMethod = "conservative",
    years: int = 5,
) -> dict[str, float]:
    """
    Compute owner earnings for the most recent period and multi-year average.

    Args:
        bundle:       TickerDataBundle with financial statements populated.
        capex_method: Which CapEx proxy method to use for maintenance CapEx
                      estimation. Options: "conservative", "graham", "regression".
        years:        Number of years to average for the trailing average metric.

    Returns:
        Dict with keys:
            "owner_earnings_latest"         — most recent annual owner earnings (USD thousands)
            "owner_earnings_avg_{years}yr"  — trailing N-year average (USD thousands)
            "owner_earnings_yield"          — owner_earnings / market_cap (decimal)
            "maintenance_capex_estimated"   — estimated maintenance CapEx (USD thousands)
            "capex_method_used"             — string name of method applied

    Logic:
        1. Extract latest IncomeStatement and CashFlowStatement from bundle
        2. Call _estimate_maintenance_capex() using the requested method
        3. Compute working_capital_change = ΔCA - ΔCL (year-over-year from BalanceSheet)
        4. owner_earnings = net_income + D&A - maintenance_capex + working_capital_change
        5. Compute trailing N-year average across available periods
        6. owner_earnings_yield = owner_earnings_latest / market_cap
           (market_cap in same USD-thousands units)
    """
    ...


def _estimate_maintenance_capex(
    bundle: TickerDataBundle,
    method: CapExMethod,
) -> list[float]:
    """
    Estimate maintenance capital expenditures for each historical period.

    Args:
        bundle: TickerDataBundle with financial and cash flow statements.
        method: The estimation method to apply.

    Returns:
        List of maintenance CapEx estimates (as positive numbers, USD thousands),
        one per period, aligned with the cash flow statement periods.

    Logic (by method):
        "conservative":
            maintenance_capex = abs(capital_expenditures) for each period
            (treats all CapEx as maintenance — most conservative approach)

        "graham":
            maintenance_capex = depreciation_amortization for each period
            (Graham's approximation: D&A ≈ economic depreciation)

        "regression":
            1. Compute revenue growth rate for each period
            2. Estimate growth CapEx = total_capex × (revenue_growth / long_run_return_on_new_investment)
               where long_run_return_on_new_investment defaults to 15%
            3. maintenance_capex = total_capex - growth_capex
            4. Clip to [0, total_capex] to prevent negative estimates
    """
    ...


def compute_working_capital_change(bundle: TickerDataBundle) -> list[float]:
    """
    Compute year-over-year change in net working capital for each period.

    Args:
        bundle: TickerDataBundle with BalanceSheet history.

    Returns:
        List of working capital changes (USD thousands), one per period after
        the first (since change requires two consecutive balance sheets).
        Positive = working capital increased (cash outflow).
        Negative = working capital decreased (cash inflow).

    Formula:
        NWC = (total_current_assets - cash_and_equivalents)
              - (total_current_liabilities - short_term_debt)
        ΔNWC = NWC_t - NWC_{t-1}

    Logic:
        1. For each consecutive pair of BalanceSheet records, compute NWC
        2. Return the list of ΔNWC values
    """
    ...


def owner_earnings_cagr(
    owner_earnings_series: list[float],
    years: int,
) -> float:
    """
    Compute the compound annual growth rate of owner earnings.

    Args:
        owner_earnings_series: List of annual owner earnings values (USD thousands),
                               sorted ascending by year.
        years:                 Number of years over which to compute CAGR.
                               Must be <= len(owner_earnings_series) - 1.

    Returns:
        CAGR as a decimal (e.g. 0.12 for 12% annual growth).
        Returns NaN if insufficient data or if start value is non-positive.

    Formula:
        CAGR = (end_value / start_value) ** (1 / years) - 1
    """
    ...

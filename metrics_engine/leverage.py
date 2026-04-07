"""
metrics_engine.leverage
========================
Computes balance sheet leverage and debt safety metrics.

Buffett strongly prefers low-debt businesses. Debt amplifies both gains and
losses, reduces optionality during downturns, and introduces refinancing risk.
The hard filters in this screener use debt/equity and interest coverage as
go / no-go gates.

Metrics computed:
    - Debt-to-equity ratio (D/E)
    - Net debt-to-EBITDA
    - Interest coverage ratio (EBIT / interest expense)
    - Current ratio (current assets / current liabilities)
    - Debt-to-assets
    - Net debt (total debt - cash)
    - Debt maturity profile flag (via footnote heuristic — future feature)
"""

from __future__ import annotations

from data_acquisition.schema import TickerDataBundle


def compute_leverage(bundle: TickerDataBundle) -> dict[str, float]:
    """
    Compute all leverage metrics for the most recent period and 5-year averages.

    Args:
        bundle: TickerDataBundle with balance sheet and income statement history.

    Returns:
        Dict with keys:
            "debt_to_equity_latest"
            "debt_to_equity_avg_5yr"
            "net_debt_to_ebitda_latest"
            "net_debt_to_ebitda_avg_5yr"
            "interest_coverage_latest"    — EBIT / interest_expense
            "interest_coverage_avg_5yr"
            "current_ratio_latest"
            "debt_to_assets_latest"
            "net_debt_usd"               — in USD thousands (negative = net cash)
            "has_net_cash"               — True if cash > total_debt

    Logic:
        1. For each period, compute each ratio from the balance sheet + income stmt
        2. Compute trailing 5-year averages
        3. Derive net_debt and has_net_cash from the latest balance sheet
    """
    ...


def debt_to_equity(total_debt: float, shareholders_equity: float) -> float:
    """
    Compute the debt-to-equity ratio.

    Args:
        total_debt:          Total interest-bearing debt (USD thousands).
        shareholders_equity: Book value of shareholders' equity (USD thousands).

    Returns:
        D/E ratio (unitless). Returns NaN if shareholders_equity <= 0
        (negative equity companies are automatically flagged as CRITICAL
        in data quality and should not reach this function).

    Formula:
        D/E = total_debt / shareholders_equity
    """
    ...


def net_debt_to_ebitda(
    total_debt: float,
    cash: float,
    ebitda: float,
) -> float:
    """
    Compute the net debt to EBITDA leverage ratio.

    Args:
        total_debt: Total interest-bearing debt (USD thousands).
        cash:       Cash and equivalents (USD thousands).
        ebitda:     EBITDA for the period (USD thousands).

    Returns:
        Net debt/EBITDA ratio. Negative value indicates net cash position.
        Returns NaN if EBITDA <= 0.

    Formula:
        Net Debt = total_debt - cash
        Net Debt / EBITDA = net_debt / ebitda
    """
    ...


def interest_coverage(ebit: float, interest_expense: float) -> float:
    """
    Compute the interest coverage ratio (times interest earned).

    Args:
        ebit:             Earnings before interest and tax (USD thousands).
        interest_expense: Gross interest expense (positive number, USD thousands).

    Returns:
        Interest coverage ratio. Returns +inf if interest_expense = 0 (no debt).
        Returns NaN if ebit is unavailable.

    Formula:
        Coverage = EBIT / interest_expense
    """
    ...


def current_ratio(
    current_assets: float,
    current_liabilities: float,
) -> float:
    """
    Compute the current ratio (short-term liquidity).

    Args:
        current_assets:      Total current assets (USD thousands).
        current_liabilities: Total current liabilities (USD thousands).

    Returns:
        Current ratio. Returns NaN if current_liabilities = 0.

    Formula:
        Current Ratio = current_assets / current_liabilities
    """
    ...


def net_debt(total_debt: float, cash: float) -> float:
    """
    Compute net debt (positive = net debtor, negative = net cash position).

    Args:
        total_debt: Total interest-bearing debt (USD thousands).
        cash:       Cash and cash equivalents (USD thousands).

    Returns:
        Net debt in USD thousands.

    Formula:
        Net Debt = total_debt - cash
    """
    ...

"""
valuation_reports.earnings_yield
==================================
Compares a stock's earnings yield against current bond yields to determine
whether the equity offers a sufficient risk premium.

Buffett's bond-equivalent framework:
    "Stocks are just bonds with floating coupons."
    The earnings yield (EPS / Price) should comfortably exceed the 10-year
    Treasury yield for a stock to be attractive relative to bonds.

This module also computes the Graham Number, a simple intrinsic value
benchmark combining earnings and book value:
    Graham Number = sqrt(22.5 × EPS × BVPS)
(22.5 = 15 × 1.5, combining Graham's maximum P/E of 15 and maximum P/B of 1.5)
"""

from __future__ import annotations

from dataclasses import dataclass

from data_acquisition.schema import MacroSnapshot


@dataclass
class EarningsYieldComparison:
    """Results of the earnings yield vs. bond yield comparison."""

    ticker: str
    earnings_yield: float               # EPS / price (decimal)
    owner_earnings_yield: float         # owner earnings / market cap (decimal)
    risk_free_rate: float               # current 10Y treasury yield (decimal)
    earnings_yield_spread: float        # earnings_yield - risk_free_rate
    owner_earnings_spread: float        # owner_earnings_yield - risk_free_rate
    is_attractive_vs_bonds: bool        # True if spread >= min_spread_over_rfr
    graham_number: float                # sqrt(22.5 × EPS × BVPS)
    price_vs_graham: float              # current_price / graham_number (< 1.0 = cheap)


def compute_earnings_yield_comparison(
    ticker: str,
    eps: float,
    owner_earnings: float,
    market_cap: float,
    current_price: float,
    book_value_per_share: float,
    macro: MacroSnapshot,
    config: dict,
) -> EarningsYieldComparison:
    """
    Compute all earnings yield and bond-comparison metrics for a stock.

    Args:
        ticker:               Ticker symbol.
        eps:                  Trailing twelve-month diluted EPS (USD per share).
        owner_earnings:       Annual owner earnings (USD thousands).
        market_cap:           Market capitalisation (USD, full dollars).
        current_price:        Current stock price (USD).
        book_value_per_share: Latest BVPS (USD per share).
        macro:                MacroSnapshot with current risk-free rate.
        config:               Full filter_config.yaml dict. Reads
                              "valuation.earnings_yield.min_spread_over_rfr_pct".

    Returns:
        EarningsYieldComparison dataclass with all computed fields.

    Logic:
        1. earnings_yield = eps / current_price
        2. owner_earnings_yield = (owner_earnings × 1000) / market_cap
           (convert owner_earnings from thousands to full dollars for market_cap comparison)
        3. risk_free_rate = macro.treasury_10y_yield
        4. earnings_yield_spread = earnings_yield - risk_free_rate
        5. is_attractive = earnings_yield_spread >= config min_spread (decimal)
        6. graham_number = sqrt(22.5 × eps × book_value_per_share)
        7. price_vs_graham = current_price / graham_number
        8. Return EarningsYieldComparison
    """
    ...


def graham_number(eps: float, book_value_per_share: float) -> float:
    """
    Compute the Graham Number — a simple intrinsic value benchmark.

    Args:
        eps:                  Trailing EPS (USD per share). Must be positive.
        book_value_per_share: Book value per share (USD). Must be positive.

    Returns:
        Graham Number in USD. Returns NaN if either input is non-positive.

    Formula:
        Graham Number = sqrt(22.5 × EPS × BVPS)

    Interpretation:
        Stocks trading below their Graham Number are generally considered
        undervalued by Graham's standards. This is a conservative metric —
        many high-quality compounders will exceed their Graham Number.
    """
    ...


def required_earnings_yield(
    risk_free_rate: float,
    equity_risk_premium: float = 0.03,
) -> float:
    """
    Compute the minimum earnings yield required for equity to be attractive vs. bonds.

    Args:
        risk_free_rate:      Current 10-year Treasury yield (decimal).
        equity_risk_premium: Additional return required above risk-free for taking
                             equity risk (default 3%, based on long-run ERP estimates).

    Returns:
        Required minimum earnings yield as a decimal.

    Formula:
        required_earnings_yield = risk_free_rate + equity_risk_premium
    """
    ...

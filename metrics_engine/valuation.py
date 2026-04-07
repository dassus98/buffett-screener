"""
metrics_engine.valuation
=========================
Computes price-based valuation multiples and yield metrics.

These multiples contextualise the quality metrics from returns.py and
profitability.py with the current market price. A great business at an
outrageous price is not a Buffett investment.

Multiples computed:
    - P/E (trailing twelve months and normalised 5yr avg EPS)
    - P/B (price to book value per share)
    - EV/EBITDA (enterprise value to trailing EBITDA)
    - EV/EBIT (enterprise value to EBIT)
    - EV/FCF (enterprise value to free cash flow)
    - Earnings yield (1 / P/E — for comparison with bond yields)
    - FCF yield (FCF / market cap)
    - Owner earnings yield (from owner_earnings.py output)
"""

from __future__ import annotations

from data_acquisition.schema import MarketData, TickerDataBundle


def compute_valuation_multiples(
    bundle: TickerDataBundle,
    owner_earnings: float,
) -> dict[str, float]:
    """
    Compute all valuation multiples using current market data and trailing financials.

    Args:
        bundle:          TickerDataBundle with market data and financial history.
        owner_earnings:  Pre-computed owner earnings figure (USD thousands) from
                         the owner_earnings module. Used for owner earnings yield.

    Returns:
        Dict with keys:
            "pe_ttm"                  — trailing 12-month P/E
            "pe_normalised"           — P / (5yr average EPS)
            "pb_ratio"                — price / book value per share
            "ev_to_ebitda"            — enterprise value / trailing EBITDA
            "ev_to_ebit"              — enterprise value / trailing EBIT
            "ev_to_fcf"               — enterprise value / trailing FCF
            "earnings_yield"          — EPS / price = 1 / pe_ttm (decimal)
            "fcf_yield"               — FCF per share / price (decimal)
            "owner_earnings_yield"    — owner_earnings / market_cap (decimal)

    Logic:
        1. Pull market_cap and enterprise_value from bundle.market_data
        2. Compute TTM figures by summing the last 4 quarters (or using
           latest annual if quarterly data is unavailable)
        3. Compute each multiple using the dedicated functions below
        4. Compute yields as reciprocals / per-share ratios
    """
    ...


def price_to_earnings(price: float, eps: float) -> float:
    """
    Compute price-to-earnings ratio.

    Args:
        price: Current stock price in USD.
        eps:   Earnings per share (USD). Use trailing 12-month diluted EPS.

    Returns:
        P/E ratio. Returns NaN if eps <= 0 (loss-making companies have
        meaningless P/E and should be filtered before this stage).

    Formula:
        P/E = price / eps
    """
    ...


def price_to_book(price: float, book_value_per_share: float) -> float:
    """
    Compute price-to-book ratio.

    Args:
        price:                Current stock price in USD.
        book_value_per_share: Book value per share (USD).

    Returns:
        P/B ratio. Returns NaN if book_value_per_share <= 0.

    Formula:
        P/B = price / book_value_per_share
    """
    ...


def ev_to_ebitda(enterprise_value: float, ebitda: float) -> float:
    """
    Compute Enterprise Value / EBITDA.

    Args:
        enterprise_value: EV in USD (full dollars, not thousands).
        ebitda:           Trailing twelve-month EBITDA in USD thousands.

    Returns:
        EV/EBITDA multiple. Returns NaN if EBITDA <= 0.

    Note:
        enterprise_value is in USD (from MarketData), ebitda is in USD thousands
        (from IncomeStatement). The function handles unit conversion internally.
    """
    ...


def ev_to_fcf(enterprise_value: float, fcf: float) -> float:
    """
    Compute Enterprise Value / Free Cash Flow.

    Args:
        enterprise_value: EV in USD (full dollars).
        fcf:              Trailing twelve-month FCF in USD thousands.

    Returns:
        EV/FCF multiple. Returns NaN if FCF <= 0.
    """
    ...


def earnings_yield(eps: float, price: float) -> float:
    """
    Compute earnings yield (inverse of P/E).

    Used by Buffett to compare equity returns against bond yields.
    If earnings yield > 10-year Treasury yield by a sufficient margin,
    the stock is potentially attractive relative to bonds.

    Args:
        eps:   Trailing diluted EPS (USD per share).
        price: Current stock price (USD).

    Returns:
        Earnings yield as a decimal (e.g. 0.05 for 5%). Returns NaN if price = 0.

    Formula:
        Earnings Yield = EPS / Price
    """
    ...


def fcf_yield(
    fcf: float,
    shares_outstanding: float,
    price: float,
) -> float:
    """
    Compute free cash flow yield.

    Args:
        fcf:                Trailing FCF in USD thousands.
        shares_outstanding: Diluted shares outstanding in thousands.
        price:              Current stock price in USD.

    Returns:
        FCF yield as a decimal. Returns NaN if price = 0 or shares = 0.

    Formula:
        FCF per share = (fcf × 1000) / (shares_outstanding × 1000) = fcf / shares
        FCF Yield = FCF per share / price
    """
    ...


def earnings_yield_spread(
    earnings_yield_decimal: float,
    risk_free_rate: float,
) -> float:
    """
    Compute the spread between earnings yield and the risk-free rate.

    A positive spread indicates the equity is offering a premium over bonds.
    Buffett historically required at least a 2–3% spread before buying.

    Args:
        earnings_yield_decimal: Earnings yield as a decimal (e.g. 0.06 for 6%).
        risk_free_rate:         Current 10-year Treasury yield as a decimal.

    Returns:
        Spread in decimal terms (e.g. 0.02 for a 2pp spread).
    """
    ...

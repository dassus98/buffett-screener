"""
valuation_reports.margin_of_safety
====================================
Computes margin of safety — the discount of the current stock price below
the estimated intrinsic value.

Graham coined the term: "The margin of safety is always dependent on the
price paid." Buffett's threshold is typically 25–50% below intrinsic value.

The module also computes the "buy below" price for each scenario and
classifies the current price as:
    STRONG_BUY   — price < conservative MoS (e.g. 50% below base intrinsic)
    BUY          — price < moderate MoS    (e.g. 33% below)
    WATCHLIST    — price < aggressive MoS  (e.g. 20% below)
    EXPENSIVE    — price >= intrinsic value
    AVOID        — price >= 1.3× intrinsic value (significantly overvalued)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from valuation_reports.intrinsic_value import IntrinsicValueEstimate


class PriceClassification(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCHLIST = "WATCHLIST"
    EXPENSIVE = "EXPENSIVE"
    AVOID = "AVOID"


@dataclass
class MarginOfSafetyResult:
    """Results of the margin of safety calculation for a stock."""

    ticker: str
    current_price: float

    intrinsic_value_bear: float
    intrinsic_value_base: float
    intrinsic_value_bull: float

    mos_vs_bear: float                  # (intrinsic_bear - price) / intrinsic_bear
    mos_vs_base: float                  # (intrinsic_base - price) / intrinsic_base
    mos_vs_bull: float

    buy_below_conservative: float       # base × (1 - conservative_pct/100)
    buy_below_moderate: float           # base × (1 - moderate_pct/100)
    buy_below_aggressive: float

    classification: PriceClassification
    classification_vs: str              # "base" | "bear" — which scenario was used


def compute_margin_of_safety(
    intrinsic_value: IntrinsicValueEstimate,
    current_price: float,
    config: dict,
) -> MarginOfSafetyResult:
    """
    Compute margin of safety and classify the current stock price.

    Args:
        intrinsic_value: IntrinsicValueEstimate from compute_intrinsic_value().
        current_price:   Current stock price in USD.
        config:          Full filter_config.yaml dict. Reads
                         "valuation.margin_of_safety" section.

    Returns:
        MarginOfSafetyResult with MoS percentages, buy-below prices, and
        a price classification.

    Logic:
        1. Read MoS thresholds from config:
               conservative_pct (e.g. 50), moderate_pct (33), aggressive_pct (20)
        2. Compute mos_vs_base = (base - price) / base
           Negative MoS means the stock is trading above intrinsic value
        3. Compute buy_below prices: intrinsic_base × (1 - pct/100)
        4. Classify:
               if current_price <= buy_below_conservative → STRONG_BUY
               elif current_price <= buy_below_moderate   → BUY
               elif current_price <= buy_below_aggressive → WATCHLIST
               elif current_price <= intrinsic_base       → EXPENSIVE
               else                                       → AVOID
        5. Record which scenario (bear vs. base) was used for classification
    """
    ...


def margin_of_safety_pct(
    intrinsic_value: float,
    current_price: float,
) -> float:
    """
    Compute the percentage margin of safety.

    Args:
        intrinsic_value: Estimated intrinsic value per share (USD).
        current_price:   Current stock price (USD).

    Returns:
        Margin of safety as a decimal. Positive = trading below intrinsic value
        (attractive). Negative = trading above intrinsic value (expensive).

    Formula:
        MoS = (intrinsic_value - current_price) / intrinsic_value
    """
    ...


def buy_below_price(
    intrinsic_value: float,
    required_mos_pct: float,
) -> float:
    """
    Compute the maximum price to pay given a required margin of safety.

    Args:
        intrinsic_value:   Estimated intrinsic value per share (USD).
        required_mos_pct:  Required margin of safety as a percentage (e.g. 33.0).

    Returns:
        Buy-below price in USD.

    Formula:
        buy_below = intrinsic_value × (1 - required_mos_pct / 100)
    """
    ...

"""
valuation_reports.intrinsic_value
==================================
Estimates intrinsic value per share using an owner-earnings DCF model.

Buffett's DCF approach:
    1. Project owner earnings for N years using a tiered growth assumption
    2. Compute a terminal value using the Gordon Growth Model
    3. Discount all cash flows back at the required rate of return
    4. Divide by shares outstanding to get intrinsic value per share

The model returns a range of estimates (bear / base / bull) by varying the
growth rate assumption, providing a natural sensitivity analysis.

Parameters (all configurable in filter_config.yaml):
    projection_years:           10
    terminal_growth_rate:       3% (long-run nominal GDP)
    discount_rate:              10% (Buffett's hurdle rate)
    high_growth_rate (years 1–5): derived from trailing owner earnings CAGR
    fade_growth_rate (years 6–10): halfway between high_growth and terminal
"""

from __future__ import annotations

from dataclasses import dataclass

from data_acquisition.schema import TickerDataBundle


@dataclass
class IntrinsicValueEstimate:
    """
    Container for the three-scenario intrinsic value estimate.

    All monetary values are USD per share.
    """

    ticker: str
    bear_case: float        # conservative growth assumption
    base_case: float        # base growth assumption (default parameters)
    bull_case: float        # optimistic growth assumption
    owner_earnings_used: float          # USD thousands — the starting point
    discount_rate_used: float           # decimal
    terminal_growth_rate_used: float    # decimal
    high_growth_rate_used: float        # decimal (base case)
    projection_years: int


def compute_intrinsic_value(
    bundle: TickerDataBundle,
    owner_earnings: float,
    config: dict,
) -> IntrinsicValueEstimate:
    """
    Run the owner-earnings DCF and return a three-scenario estimate.

    Args:
        bundle:          TickerDataBundle (for shares outstanding and market data).
        owner_earnings:  Latest annual owner earnings (USD thousands), computed
                         by metrics_engine.owner_earnings.
        config:          Full filter_config.yaml dict. Reads "valuation.dcf" section.

    Returns:
        IntrinsicValueEstimate with bear / base / bull intrinsic value per share.

    Logic:
        1. Read DCF parameters from config["valuation"]["dcf"]
        2. Derive growth rate scenarios:
               bear:  high_growth_rate × 0.6
               base:  high_growth_rate (from config or trailing CAGR)
               bull:  min(high_growth_rate × 1.5, 0.25)  (cap at 25%)
        3. Call dcf_owner_earnings() for each scenario
        4. Divide by current shares_outstanding to get per-share values
        5. Return IntrinsicValueEstimate
    """
    ...


def dcf_owner_earnings(
    owner_earnings_base: float,
    high_growth_rate: float,
    high_growth_years: int,
    fade_growth_rate: float,
    terminal_growth_rate: float,
    discount_rate: float,
    total_years: int,
) -> float:
    """
    Run a two-stage owner-earnings DCF and return the total present value.

    Args:
        owner_earnings_base:  Starting owner earnings (USD thousands) for year 1.
        high_growth_rate:     Annual growth rate for the first `high_growth_years`.
        high_growth_years:    Number of years in the high-growth stage.
        fade_growth_rate:     Annual growth rate for the remaining years of the
                              projection (years high_growth_years+1 to total_years).
        terminal_growth_rate: Perpetuity growth rate for the terminal value.
        discount_rate:        Annual required rate of return (hurdle rate).
        total_years:          Total projection horizon (e.g. 10).

    Returns:
        Total present value of projected cash flows + terminal value (USD thousands).

    Formula:
        Stage 1 PV = Σ [OE_base × (1+g_high)^t / (1+r)^t]  for t=1..high_growth_years
        Stage 2 PV = Σ [OE_stage2 × (1+g_fade)^t / (1+r)^t] for remaining years
        Terminal TV = OE_final × (1+g_terminal) / (r - g_terminal)
        Terminal PV = TV / (1+r)^total_years
        Total PV = Stage 1 PV + Stage 2 PV + Terminal PV

    Raises:
        ValueError: if discount_rate <= terminal_growth_rate (model breaks down)
    """
    ...


def sensitivity_table(
    owner_earnings_base: float,
    discount_rates: list[float],
    high_growth_rates: list[float],
    config: dict,
    shares_outstanding: float,
) -> dict:
    """
    Generate a 2D sensitivity table of intrinsic value per share.

    Args:
        owner_earnings_base:  Starting owner earnings (USD thousands).
        discount_rates:       List of discount rates to test (e.g. [0.08, 0.10, 0.12]).
        high_growth_rates:    List of high-growth-rate assumptions (e.g. [0.05, 0.10, 0.15]).
        config:               Full config dict for other DCF parameters.
        shares_outstanding:   Diluted shares outstanding (thousands).

    Returns:
        Dict of the form:
            {
                "rows": high_growth_rates,
                "cols": discount_rates,
                "values": 2D list [growth_idx][discount_idx] → intrinsic_value_per_share
            }
        Suitable for rendering as a table in Streamlit or Markdown.

    Logic:
        Nested loop over discount_rates × high_growth_rates.
        Call dcf_owner_earnings() for each combination.
        Divide by shares_outstanding to get per-share values.
    """
    ...

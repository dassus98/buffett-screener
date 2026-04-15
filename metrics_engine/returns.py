"""Return formula functions: Initial Rate of Return (F2), Return on Retained
Earnings (F4).

Each public function accepts pre-computed per-share values (from the Module 2
orchestrator) and returns a summary dict consumed by the recommendation engine
and composite scoring.

Data Lineage Contract
---------------------
Upstream producers:
    - ``metrics_engine.owner_earnings.compute_owner_earnings`` (F1)
      → provides the owner earnings Series from which ``owner_earnings_per_share``
        is derived (OE / shares_outstanding_diluted).
    - ``data_acquisition.store.read_table("market_data")``
      → provides ``current_price_usd`` for the F2 initial return calculation.
    - ``data_acquisition.store.read_table("macro_data")``
      → provides risk-free rate (10-year government bond yield) for the F2
        pass test.
    - ``data_acquisition.store.read_table("income_statement")``
      → provides ``eps_diluted`` for the F4 retained earnings calculation.
    - Dividends per share: currently assumed to be zero (DPS is absent from
      ``schema.CANONICAL_COLUMNS``). The orchestrator passes a zero Series.

Downstream consumers:
    - ``metrics_engine.__init__._compute_returns``
      → calls both functions, collects summary dicts.
    - ``metrics_engine.composite_score`` (Tier 2 scoring)
      → reads summary key: ``return_on_retained`` (retained_earnings_return
        criterion).
    - ``screener.hard_filters``
      → may read ``passes_2x_test`` for Tier 1 pass/fail (F2).

Config dependencies:
    - ``config/filter_config.yaml → valuation.bond_yield_multiplier``
      (used by ``compute_initial_rate_of_return`` for the 2× pass test).
    - All thresholds loaded via ``screener.filter_config_loader.get_threshold()``.

Authoritative spec: docs/FORMULAS.md §F2, §F4.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)


def compute_initial_rate_of_return(
    owner_earnings_per_share: float,
    current_price: float,
    risk_free_rate: float,
) -> dict[str, Any]:
    """Compute Initial Rate of Return (F2) — Buffett's first-cut value check.

    ``Initial Return = owner_earnings_per_share / current_price``

    The pass test checks whether the return is at least
    ``bond_yield_multiplier × risk_free_rate`` (default: 2×).

    Parameters
    ----------
    owner_earnings_per_share:
        Owner earnings (F1) divided by diluted shares outstanding.
    current_price:
        Current market price per share.
    risk_free_rate:
        Current 10-year government bond yield (decimal, e.g. 0.045 = 4.5%).

    Returns
    -------
    dict
        Keys: ``initial_return`` (float), ``vs_bond_yield`` (spread over
        *risk_free_rate*), ``passes_2x_test`` (bool).

    Notes
    -----
    ``owner_earnings_per_share ≤ 0`` is valid but yields a negative return
    (automatic fail, logged at WARNING).  ``current_price ≤ 0`` → NaN
    (logged at ERROR).
    """
    # --- Step 1: Load the bond yield multiplier from config ---
    # Per FORMULAS.md F2: the initial return must be ≥ multiplier × risk-free rate.
    # Default multiplier is 2× (Buffett: "at least double the bond yield").
    multiplier: float = float(get_threshold("valuation.bond_yield_multiplier"))

    # --- Step 2: Guard against invalid price (edge case) ---
    # current_price ≤ 0 is a data error — cannot compute a meaningful yield.
    if pd.isna(current_price) or current_price <= 0:
        logger.error("Cannot compute Initial Rate of Return: current_price=%s ≤ 0.", current_price)
        return {"initial_return": float("nan"), "vs_bond_yield": float("nan"), "passes_2x_test": False}

    # --- Step 3: Compute the initial earnings yield ---
    # This is the Buffett framing: what return am I getting on day one if I
    # treat the purchase price as a "bond" that pays owner earnings?
    initial_return = float(owner_earnings_per_share) / float(current_price)

    # Spread over risk-free rate — positive = premium over Treasuries
    vs_bond = initial_return - float(risk_free_rate)

    # --- Step 4: Log negative OE (valid but automatic fail) ---
    if owner_earnings_per_share <= 0:
        logger.warning(
            "Owner Earnings per share ≤ 0 (%.4f) — Initial Rate of Return is negative. Automatic fail.",
            owner_earnings_per_share,
        )

    # --- Step 5: Apply the 2× test (configurable multiplier) ---
    # Per FORMULAS.md F2: passes if initial_return ≥ multiplier × risk_free_rate
    passes = initial_return >= multiplier * float(risk_free_rate)
    return {"initial_return": initial_return, "vs_bond_yield": vs_bond, "passes_2x_test": passes}


def compute_return_on_retained_earnings(
    eps_series: pd.Series,
    dividends_per_share_series: pd.Series,
) -> dict[str, Any]:
    """Compute Return on Retained Earnings (F4) — Buffett's Dollar Test.

    ``Return = (EPS_latest − EPS_earliest) / Σ(EPS_n − DPS_n)``

    Asks whether management created at least $1 of incremental earnings
    power for every $1 they retained (instead of paying as dividends).

    Parameters
    ----------
    eps_series:
        Diluted EPS values in chronological order.  Any index is accepted;
        ``iloc[0]`` and ``iloc[-1]`` are used for earliest and latest.
    dividends_per_share_series:
        Dividends per share aligned positionally to *eps_series*.

    Returns
    -------
    dict
        Keys: ``return_on_retained`` (float or NaN), ``cumulative_retained_per_share``
        (float), ``eps_growth`` (float), ``meaningful`` (bool — False when
        cumulative retained earnings ≤ 0).

    Notes
    -----
    When the company paid out more than it earned (``cumulative_retained ≤ 0``),
    the metric is not meaningful.  ``return_on_retained`` is set to NaN and
    ``meaningful = False``.  The security is NOT dropped — assess via F5, F7, F10.
    """
    # --- Step 1: Normalize inputs to float Series ---
    # Detach from original index so positional alignment is safe.
    eps = pd.Series(eps_series.values, dtype=float)
    dps = pd.Series(dividends_per_share_series.values, dtype=float)

    # --- Step 2: Compute retained earnings per share per year ---
    # retained_n = EPS_n − DPS_n  (what management kept instead of paying out)
    retained = eps - dps

    # Cumulative retained: total capital management had to deploy over the window
    cumulative = float(retained.sum())

    # --- Step 3: Compute EPS growth (numerator of the Dollar Test) ---
    # EPS growth = EPS_latest − EPS_earliest (absolute, not CAGR)
    if len(eps) < 2:
        eps_growth: float = float("nan")
    else:
        eps_growth = float(eps.iloc[-1]) - float(eps.iloc[0])

    # --- Step 4: Handle non-meaningful case (cumulative retained ≤ 0) ---
    # If the company paid out ≥ all earnings as dividends, the denominator is
    # zero or negative and the ratio has no economic meaning. Per FORMULAS.md F4:
    # do NOT drop — flag as not meaningful and assess via F5, F7, F10.
    if cumulative <= 0:
        logger.warning(
            "Return on retained earnings not meaningful: cumulative_retained=%.4f ≤ 0. "
            "Company paid out ≥ all earnings. Assess via other metrics.",
            cumulative,
        )
        return {
            "return_on_retained": float("nan"),
            "cumulative_retained_per_share": cumulative,
            "eps_growth": eps_growth,
            "meaningful": False,
        }

    # --- Step 5: Compute the Dollar Test ratio ---
    # Per FORMULAS.md F4: Return = (EPS_latest − EPS_earliest) / Σ(EPS_n − DPS_n)
    # A value ≥ 1.0 means management created ≥ $1 of incremental earnings
    # power for every $1 they retained. Buffett's benchmark is ≥ 0.12 (12%).
    ret = eps_growth / cumulative if not pd.isna(eps_growth) else float("nan")
    return {
        "return_on_retained": ret,
        "cumulative_retained_per_share": cumulative,
        "eps_growth": eps_growth,
        "meaningful": True,
    }

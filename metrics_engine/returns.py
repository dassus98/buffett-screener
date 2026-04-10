"""Computes return-on-capital metrics: ROIC, ROE, and ROCE, plus trailing averages and consistency scores."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_config

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
    cfg = get_config()
    multiplier: float = float(cfg.get("valuation", {}).get("bond_yield_multiplier", 2.0))

    if pd.isna(current_price) or current_price <= 0:
        logger.error("Cannot compute Initial Rate of Return: current_price=%s ≤ 0.", current_price)
        return {"initial_return": float("nan"), "vs_bond_yield": float("nan"), "passes_2x_test": False}

    initial_return = float(owner_earnings_per_share) / float(current_price)
    vs_bond = initial_return - float(risk_free_rate)

    if owner_earnings_per_share <= 0:
        logger.warning(
            "Owner Earnings per share ≤ 0 (%.4f) — Initial Rate of Return is negative. Automatic fail.",
            owner_earnings_per_share,
        )

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
    eps = pd.Series(eps_series.values, dtype=float)
    dps = pd.Series(dividends_per_share_series.values, dtype=float)

    retained = eps - dps
    cumulative = float(retained.sum())

    if len(eps) < 2:
        eps_growth: float = float("nan")
    else:
        eps_growth = float(eps.iloc[-1]) - float(eps.iloc[0])

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

    ret = eps_growth / cumulative if not pd.isna(eps_growth) else float("nan")
    return {
        "return_on_retained": ret,
        "cumulative_retained_per_share": cumulative,
        "eps_growth": eps_growth,
        "meaningful": True,
    }

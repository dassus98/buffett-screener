"""Computes EPS 10-Year CAGR (F11) and share buyback indicator (F13)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _count_decline_years(eps: pd.Series) -> int:
    """Count year-over-year EPS declines across the full series (NaN-safe).

    Parameters
    ----------
    eps:
        EPS values in chronological order (positional index, NaN-free
        values expected but handled gracefully via diff()).

    Returns
    -------
    int
        Number of consecutive pairs where ``eps[t] < eps[t-1]``.
    """
    if len(eps) < 2:
        return 0
    return int((eps.diff().iloc[1:] < 0).sum())


def _resolve_base(
    eps: pd.Series,
    fiscal_years: list[int],
) -> tuple[int, float, bool]:
    """Determine the base fiscal year and EPS value for CAGR.

    Tries the earliest year first.  If that EPS is ≤ 0, walks forward to
    the first year with positive EPS.

    Parameters
    ----------
    eps:
        EPS values in chronological order (positional index).
    fiscal_years:
        Integer fiscal years aligned positionally with *eps*.

    Returns
    -------
    tuple[int, float, bool]
        ``(base_year, base_eps, substituted)`` where *substituted* is
        ``True`` when the earliest year was skipped.
    """
    if float(eps.iloc[0]) > 0:
        return fiscal_years[0], float(eps.iloc[0]), False
    for i, val in enumerate(eps.values):
        if float(val) > 0:
            return fiscal_years[i], float(val), True
    return fiscal_years[0], float("nan"), False


# ---------------------------------------------------------------------------
# Public formula functions
# ---------------------------------------------------------------------------

def compute_eps_cagr(eps_series: pd.Series) -> dict[str, Any]:
    """Compute EPS 10-Year CAGR (F11) and year-over-year decline count.

    ``EPS CAGR = (EPS_current / EPS_base)^(1 / n_years) − 1``
    where ``n_years = current_fiscal_year − base_fiscal_year``.

    Parameters
    ----------
    eps_series:
        Diluted EPS in chronological order.  **Index must contain integer
        fiscal years** — used to compute ``n_years`` and to locate the
        adjusted base when the earliest EPS is ≤ 0.

    Returns
    -------
    dict
        Keys: ``eps_cagr`` (float), ``decline_years`` (int),
        ``base_year`` (int), ``base_eps`` (float), ``current_eps`` (float).

    Notes
    -----
    Fewer than 5 positive EPS years → ``eps_cagr = NaN``, logged at ERROR
    (security flagged for drop).  ``current_eps ≤ 0`` → automatic fail,
    logged at WARNING.
    """
    eps = pd.Series(eps_series.values, dtype=float)
    fiscal_years: list[int] = [int(y) for y in eps_series.index]
    decline_years = _count_decline_years(eps)
    current_eps = float(eps.iloc[-1]) if len(eps) > 0 else float("nan")
    positive_count = int((eps > 0).sum())

    if positive_count < 5:
        logger.error(
            "EPS CAGR: %d positive EPS year(s) < 5 minimum. Security flagged for drop.",
            positive_count,
        )
        return {
            "eps_cagr": float("nan"), "decline_years": decline_years,
            "base_year": fiscal_years[0] if fiscal_years else None,
            "base_eps": float("nan"), "current_eps": current_eps,
        }

    base_year, base_eps, substituted = _resolve_base(eps, fiscal_years)
    current_year = fiscal_years[-1]
    n_years = current_year - base_year

    if substituted:
        logger.warning(
            "EPS CAGR: earliest EPS ≤ 0. Using base year %d (eps=%.4f); "
            "adjusted window %d–%d (%d years).",
            base_year, base_eps, base_year, current_year, n_years,
        )

    if n_years <= 0 or pd.isna(base_eps):
        cagr = float("nan")
    elif current_eps <= 0:
        logger.warning("EPS CAGR: current_eps=%.4f ≤ 0. Automatic fail.", current_eps)
        cagr = float("nan")
    else:
        cagr = float((current_eps / base_eps) ** (1.0 / n_years) - 1.0)

    return {
        "eps_cagr": cagr, "decline_years": decline_years,
        "base_year": base_year, "base_eps": base_eps, "current_eps": current_eps,
    }


def compute_buyback_indicator(shares_series: pd.Series) -> dict[str, Any]:
    """Compute Share Buyback Indicator (F13).

    ``Buyback Rate = (Shares_10yr_ago − Shares_current) / Shares_10yr_ago``

    Positive → shares retired (good).  Negative → dilution (bad).

    Parameters
    ----------
    shares_series:
        Diluted shares outstanding in chronological order.  ``iloc[0]``
        is the 10-years-ago figure; ``iloc[-1]`` is current.

    Returns
    -------
    dict
        Keys: ``buyback_pct`` (float), ``shares_reduced`` (bool),
        ``shares_10yr_ago`` (float), ``shares_current`` (float).

    Notes
    -----
    ``shares_10yr_ago ≤ 0`` or NaN → ``buyback_pct = NaN``,
    ``shares_reduced = False`` (logged at WARNING).  Fewer than 2 data
    points receives the same treatment.
    """
    shares = pd.Series(shares_series.values, dtype=float)

    if len(shares) < 2:
        logger.warning("Buyback indicator: fewer than 2 data points. Returning NaN.")
        return {
            "buyback_pct": float("nan"), "shares_reduced": False,
            "shares_10yr_ago": float("nan"), "shares_current": float("nan"),
        }

    shares_old = float(shares.iloc[0])
    shares_new = float(shares.iloc[-1])

    if pd.isna(shares_old) or shares_old <= 0:
        logger.warning(
            "Buyback indicator: shares_10yr_ago=%.0f invalid (≤ 0 or NaN). Returning NaN.",
            shares_old,
        )
        return {
            "buyback_pct": float("nan"), "shares_reduced": False,
            "shares_10yr_ago": shares_old, "shares_current": shares_new,
        }

    buyback_pct = (shares_old - shares_new) / shares_old
    return {
        "buyback_pct": float(buyback_pct),
        "shares_reduced": bool(buyback_pct > 0),
        "shares_10yr_ago": shares_old,
        "shares_current": shares_new,
    }

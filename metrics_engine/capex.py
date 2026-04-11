"""Computes capital expenditure intensity metrics: CapEx-to-Net-Earnings ratio (F12)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def compute_capex_to_earnings(
    capex_series: pd.Series,
    net_income_series: pd.Series,
) -> dict[str, Any]:
    """Compute CapEx-to-Net-Earnings Ratio (F12) — average over valid years.

    ``CapEx/Earnings = |CapEx| / Net Income``

    Computed per year, then averaged.  Years where net income ≤ 0 are
    **excluded** from the average (logged at DEBUG) to avoid dividing by
    zero or producing economically misleading ratios.

    Parameters
    ----------
    capex_series:
        Capital expenditures in schema convention (negative = cash
        outflow).  Positionally aligned with *net_income_series*.
        ``abs()`` is applied before dividing, so positive-sign data-source
        errors are handled transparently.
    net_income_series:
        Net income values aligned positionally with *capex_series*.

    Returns
    -------
    dict
        Keys: ``avg_capex_to_ni`` (float or NaN), ``years_included``
        (int), ``years_excluded`` (int).

    Notes
    -----
    ``avg_capex_to_ni = NaN`` when no year has positive net income — the
    security should already have failed F10 (net margin consistency).
    CapEx = 0 (pure software / service business) produces a ratio of 0
    for that year; valid and included in the average.
    """
    capex = pd.Series(capex_series.values, dtype=float)
    ni = pd.Series(net_income_series.values, dtype=float)

    valid_mask = ni.notna() & (ni > 0) & capex.notna()
    excluded = int((~valid_mask).sum())
    included = int(valid_mask.sum())

    if excluded > 0:
        logger.debug(
            "CapEx/NI: excluded %d year(s) where net_income ≤ 0 or data missing.",
            excluded,
        )

    if included == 0:
        logger.warning(
            "CapEx/NI: no valid years with positive net income. avg_capex_to_ni=NaN."
        )
        return {
            "avg_capex_to_ni": float("nan"),
            "years_included": 0,
            "years_excluded": excluded,
        }

    ratios = capex.loc[valid_mask].abs() / ni.loc[valid_mask]
    return {
        "avg_capex_to_ni": float(ratios.mean()),
        "years_included": included,
        "years_excluded": excluded,
    }

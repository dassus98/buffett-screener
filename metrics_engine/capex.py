"""Capital expenditure formula function: CapEx-to-Net-Earnings Ratio (F12).

Computes the average annual ratio of |CapEx| / Net Income across all years
where net income is positive, providing a measure of capital intensity — how
much of each dollar earned must be reinvested to sustain the business.

Data Lineage Contract
---------------------
Upstream producers:
    - ``data_acquisition.store.read_table("cash_flow")``
      → provides ``capital_expenditures`` (stored negative per schema
        convention; ``abs()`` applied before dividing).
    - ``data_acquisition.store.read_table("income_statement")``
      → provides ``net_income``.
    - ``metrics_engine.__init__._compute_growth_formulas``
      → aligns CapEx from cash_flow and net_income from income_statement
        by fiscal year, then calls ``compute_capex_to_earnings``.
    - Column names are identity-mapped from ``data_acquisition.schema.py``.

Downstream consumers:
    - ``metrics_engine.__init__._compute_growth_formulas``
      → collects ``f12_result`` dict.
    - ``metrics_engine.composite_score._score_capex`` (Tier 2 scoring)
      → reads summary key: ``avg_capex_to_ni`` (capital_efficiency criterion,
        lower is better).

Config dependencies:
    - None. This function computes the raw ratio; threshold comparisons
      happen downstream in ``composite_score`` via
      ``soft_scores.capital_efficiency.excellent_capex_ratio`` and
      ``soft_scores.capital_efficiency.fail_capex_ratio``.

Authoritative spec: docs/FORMULAS.md §F12.
"""

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
    # --- Step 1: Normalize inputs to float Series ---
    # Detach from original index so positional alignment is safe.
    capex = pd.Series(capex_series.values, dtype=float)
    ni = pd.Series(net_income_series.values, dtype=float)

    # --- Step 2: Identify valid years (NI > 0 and both values non-NaN) ---
    # Per FORMULAS.md F12: exclude years where net income ≤ 0 to avoid
    # dividing by zero or producing economically misleading ratios.
    valid_mask = ni.notna() & (ni > 0) & capex.notna()
    excluded = int((~valid_mask).sum())
    included = int(valid_mask.sum())

    if excluded > 0:
        logger.debug(
            "CapEx/NI: excluded %d year(s) where net_income ≤ 0 or data missing.",
            excluded,
        )

    # --- Step 3: Handle all-excluded edge case ---
    # When no year has positive NI, the ratio is undefined. The security
    # should already have failed F10 (net margin consistency).
    if included == 0:
        logger.warning(
            "CapEx/NI: no valid years with positive net income. avg_capex_to_ni=NaN."
        )
        return {
            "avg_capex_to_ni": float("nan"),
            "years_included": 0,
            "years_excluded": excluded,
        }

    # --- Step 4: Compute per-year ratios and average ---
    # abs() handles the sign convention: CapEx is stored negative (cash
    # outflow) per schema, but the ratio is expressed as a positive fraction.
    # CapEx = 0 (pure software/service) → ratio = 0 for that year (valid).
    ratios = capex.loc[valid_mask].abs() / ni.loc[valid_mask]
    return {
        "avg_capex_to_ni": float(ratios.mean()),
        "years_included": included,
        "years_excluded": excluded,
    }

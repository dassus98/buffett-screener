"""Owner Earnings formula function: Owner Earnings (F1).

Computes Buffett's owner earnings for each fiscal year using the conservative
total-CapEx approach:

    OE = net_income + depreciation_amortization + capital_expenditures

where ``capital_expenditures`` is stored as a negative number per schema
convention, so the addition effectively subtracts the full CapEx amount.
Working capital change is tracked for transparency but NOT deducted from OE
(the Buffett 1986 formulation is ambiguous on this adjustment).

Data Lineage Contract
---------------------
Upstream producers:
    - ``data_acquisition.store.read_table("income_statement")``
      → provides ``fiscal_year``, ``net_income`` columns
        (canonical names from ``data_acquisition.schema.py``).
    - ``data_acquisition.store.read_table("cash_flow")``
      → provides ``fiscal_year``, ``depreciation_amortization``,
        ``capital_expenditures``, ``working_capital_change`` columns.
    - ``data_acquisition.store.read_table("balance_sheet")``
      → accepted for API consistency but not currently used.
    - Column names are identity-mapped from ``schema.CANONICAL_COLUMNS``.

Downstream consumers:
    - ``metrics_engine.__init__._compute_f1_and_f5``
      → calls ``compute_owner_earnings``, passes ``owner_earnings`` Series
        to ``leverage.compute_debt_payoff`` (F5).
    - ``metrics_engine.__init__._compute_returns``
      → uses owner earnings to derive ``owner_earnings_per_share`` for
        ``returns.compute_initial_rate_of_return`` (F2).
    - ``metrics_engine.composite_score`` (Tier 2 scoring)
      → reads summary key: ``owner_earnings_cagr`` (owner_earnings_growth
        criterion).
    - ``screener.hard_filters``
      → reads ``owner_earnings`` Series indirectly via F5 debt payoff.

Config dependencies:
    - None. This module uses no thresholds from ``filter_config.yaml``.
    - ``_HIGH_CAPEX_DA_RATIO`` is a methodological constant (not a tunable
      business threshold), so it lives here rather than in config.

Authoritative spec: docs/FORMULAS.md §F1.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# |CapEx| / D&A ratio above which the maintenance-CapEx approximation is
# flagged as likely understated.  Per FORMULAS.md F1 distortion-flag rule.
_HIGH_CAPEX_DA_RATIO: float = 2.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _correct_capex_sign(capex: pd.Series) -> pd.Series:
    """Ensure every CapEx value is negative (cash-outflow convention).

    Parameters
    ----------
    capex:
        Raw ``capital_expenditures`` series.  Values should be negative per
        schema convention; positive values indicate a data-source error.

    Returns
    -------
    pd.Series
        Copy with positive values negated.  Original Series is not mutated.
    """
    # Identify years where CapEx is positive (data-source error).
    # Schema convention: CapEx is a cash outflow → negative.
    positive = capex.notna() & (capex > 0)
    if not positive.any():
        return capex
    logger.warning(
        "CapEx sign correction: %d year(s) had positive CapEx. Negated per schema convention.",
        int(positive.sum()),
    )
    # Return a corrected copy — do not mutate the original Series.
    corrected = capex.copy()
    corrected.loc[positive] = -capex.loc[positive]
    return corrected


def _compute_capex_da_ratio(capex: pd.Series, da: pd.Series) -> pd.Series:
    """Compute ``|capex| / da`` per year; NaN where da ≤ 0 or either is missing.

    Parameters
    ----------
    capex:
        CapEx values (negative convention, already sign-corrected).
    da:
        Depreciation & amortization (positive).

    Returns
    -------
    pd.Series of float ratios with the same index as *capex*.
    """
    # Only compute ratio where both D&A > 0 and CapEx are available.
    # D&A ≤ 0 would produce a meaningless or infinite ratio.
    ratio = pd.Series(float("nan"), index=capex.index, dtype=float)
    both = da.notna() & (da > 0) & capex.notna()
    ratio.loc[both] = (capex.loc[both].abs() / da.loc[both]).values
    return ratio


def _compute_oe_cagr(oe: pd.Series) -> float:
    """Compute the CAGR of owner earnings over available years.

    Parameters
    ----------
    oe:
        Owner earnings Series in chronological order (may contain NaN).

    Returns
    -------
    float
        Compound annual growth rate, or NaN when fewer than two valid
        observations exist or when the first or last OE value is ≤ 0.
    """
    # Drop NaN entries — CAGR requires at least two valid data points.
    valid = oe.dropna()
    if len(valid) < 2:
        return float("nan")
    first, last = float(valid.iloc[0]), float(valid.iloc[-1])
    # CAGR is undefined when either endpoint is ≤ 0 (cannot take root of
    # negative or zero). Return NaN rather than a misleading number.
    if first <= 0 or last <= 0:
        return float("nan")
    # Standard CAGR formula: (last/first)^(1/n) − 1
    return float((last / first) ** (1.0 / (len(valid) - 1)) - 1.0)


def _merge_oe_inputs(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge income statement and cash flow data for OE computation.

    Parameters
    ----------
    income_df:
        Columns: ``fiscal_year``, ``net_income``.
    cashflow_df:
        Columns: ``fiscal_year``, ``depreciation_amortization``,
        ``capital_expenditures``, and optionally ``working_capital_change``.

    Returns
    -------
    pd.DataFrame
        Left-merged on ``fiscal_year`` with a ``wc_change`` column (NaN
        when ``working_capital_change`` is absent from *cashflow_df*).
    """
    # Isolate income columns, sorted ascending for chronological alignment.
    inc = income_df[["fiscal_year", "net_income"]].sort_values("fiscal_year").reset_index(drop=True)

    # Cash flow: always need D&A and CapEx; working_capital_change is optional.
    cf_cols = ["fiscal_year", "depreciation_amortization", "capital_expenditures"]
    if "working_capital_change" in cashflow_df.columns:
        cf_cols.append("working_capital_change")
    cf = cashflow_df[cf_cols].sort_values("fiscal_year").reset_index(drop=True)

    # Left join: every income-statement year gets cash flow data if available.
    df = inc.merge(cf, on="fiscal_year", how="left")

    # If working_capital_change was absent from the cash flow DataFrame,
    # create the column as NaN so downstream code has a consistent schema.
    if "working_capital_change" not in df.columns:
        df["working_capital_change"] = float("nan")
    return df.rename(columns={"working_capital_change": "wc_change"})


def _build_annual_df(
    fiscal_years: pd.Series,
    ni: pd.Series,
    da: pd.Series,
    capex: pd.Series,
    wc: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Compute per-year OE values and assemble the output DataFrame.

    Parameters
    ----------
    fiscal_years, ni, da, capex, wc:
        Aligned Series from the merged inputs.  *capex* must be
        sign-corrected (negative) before calling.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        ``(annual_df, oe_series, capex_da_ratio_series)``
    """
    # OE is only computable when all three components are available.
    valid = da.notna() & capex.notna() & ni.notna()
    oe = pd.Series(float("nan"), index=ni.index, dtype=float)
    # F1 formula: OE = net_income + D&A + CapEx (CapEx is negative → addition subtracts)
    oe.loc[valid] = (ni.loc[valid] + da.loc[valid] + capex.loc[valid]).values

    # CapEx/D&A ratio for distortion-flag evaluation downstream.
    ratio = _compute_capex_da_ratio(capex, da)

    # Assemble the full annual output DataFrame with all component columns
    # for transparency and downstream debugging.
    annual_df = pd.DataFrame({
        "fiscal_year": fiscal_years.values,
        "owner_earnings": oe.values,
        "net_income": ni.values,
        "da": da.values,
        "capex": capex.values,
        "wc_change": wc.values,
        "capex_to_da_ratio": ratio.values,
    })
    return annual_df, oe, ratio


# ---------------------------------------------------------------------------
# Public formula function
# ---------------------------------------------------------------------------

def compute_owner_earnings(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    balance_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute Owner Earnings (F1) for each fiscal year.

    ``OE = net_income + depreciation_amortization + capital_expenditures``

    Uses **total** CapEx (not maintenance CapEx) as the deduction, making
    this a conservative estimate.  CapEx sign is auto-corrected if positive.
    Working capital change is tracked in ``wc_change`` but **not deducted**
    from OE (the Buffett 1986 formulation is ambiguous on this adjustment).

    Parameters
    ----------
    income_df:
        Columns: ``fiscal_year``, ``net_income``.  Single ticker.
    cashflow_df:
        Columns: ``fiscal_year``, ``depreciation_amortization``,
        ``capital_expenditures``, and optionally ``working_capital_change``.
    balance_df:
        Accepted for API consistency; not used in the current computation.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        *annual_df* columns: ``fiscal_year``, ``owner_earnings``,
        ``net_income``, ``da``, ``capex``, ``wc_change``,
        ``capex_to_da_ratio``.
        *summary* keys: ``avg_owner_earnings_10yr``, ``owner_earnings_cagr``,
        ``high_capex_flag``.
    """
    # --- Step 1: Merge income statement with cash flow on fiscal_year ---
    df = _merge_oe_inputs(income_df, cashflow_df)

    # --- Step 2: Extract and type-cast component Series ---
    ni = df["net_income"].astype(float)
    da = df["depreciation_amortization"].astype(float)
    # CapEx sign correction: ensure negative (cash-outflow convention).
    # Positive values indicate a data-source error and are auto-negated.
    capex = _correct_capex_sign(df["capital_expenditures"].astype(float))
    wc = df["wc_change"].astype(float)

    # --- Step 3: Compute per-year OE and CapEx/D&A distortion ratio ---
    annual_df, oe, ratio = _build_annual_df(df["fiscal_year"], ni, da, capex, wc)

    # --- Step 4: Flag years where CapEx/D&A ratio exceeds threshold ---
    # Per FORMULAS.md F1: when |CapEx| > 2× D&A, the total-CapEx proxy may
    # significantly overstate the maintenance portion, making OE overly
    # conservative. Flag but do NOT adjust — transparency for the analyst.
    high_capex_mask = ratio.notna() & (ratio > _HIGH_CAPEX_DA_RATIO)
    if high_capex_mask.any():
        logger.warning(
            "High CapEx/D&A ratio in %d year(s) — total CapEx proxy may overstate "
            "maintenance CapEx deduction.  OE is a conservative estimate.",
            int(high_capex_mask.sum()),
        )
    logger.info("Owner Earnings: total CapEx used as maintenance CapEx proxy (conservative).")

    # --- Step 5: Build summary statistics ---
    valid_oe = oe.dropna()
    avg_oe = float(valid_oe.mean()) if not valid_oe.empty else float("nan")
    return annual_df, {
        "avg_owner_earnings_10yr": avg_oe,
        "owner_earnings_cagr": _compute_oe_cagr(oe),
        "high_capex_flag": bool(high_capex_mask.any()),
    }

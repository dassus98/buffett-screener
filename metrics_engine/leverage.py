"""Leverage and debt safety formula functions: Debt Payoff (F5), Debt-to-Equity (F6),
Interest Expense Coverage (F9).

Each public function accepts single-ticker DataFrames from DuckDB (via
``data_acquisition.store.read_table``) and returns a ``(annual_df, summary_dict)``
tuple consumed by the Module 2 orchestrator (``metrics_engine.__init__``).

Data Lineage Contract
---------------------
Upstream producers:
    - ``data_acquisition.store.read_table("balance_sheet")``
      → provides ``fiscal_year``, ``long_term_debt``, ``shareholders_equity``
        columns (canonical names from ``data_acquisition.schema.py``).
    - ``data_acquisition.store.read_table("income_statement")``
      → provides ``fiscal_year``, ``interest_expense``, ``operating_income``
        columns.
    - ``metrics_engine.owner_earnings.compute_owner_earnings``
      → provides ``owner_earnings_series`` indexed by fiscal year (input to F5).
    - Column names are identity-mapped from ``schema.CANONICAL_COLUMNS``.

Downstream consumers:
    - ``metrics_engine.__init__._compute_f1_and_f5``
      → calls ``compute_debt_payoff`` with F1 owner earnings output.
    - ``metrics_engine.__init__._compute_leverage``
      → calls ``compute_debt_to_equity`` and ``compute_interest_coverage``.
    - ``metrics_engine.composite_score`` (Tier 2 scoring)
      → reads summary keys: ``avg_de_10yr`` (debt conservatism criterion),
        ``avg_interest_pct_10yr`` (interest coverage criterion).
    - ``screener.hard_filters``
      → reads summary keys: ``debt_payoff_years``, ``pass`` for Tier 1 pass/fail.

Config dependencies:
    - ``config/filter_config.yaml → hard_filters.max_debt_payoff_years``
      (used by ``compute_debt_payoff`` to determine pass/fail).
    - All thresholds loaded via ``screener.filter_config_loader.get_threshold()``.

Authoritative spec: docs/FORMULAS.md §F5, §F6, §F9.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)


def compute_debt_payoff(
    balance_df: pd.DataFrame,
    owner_earnings_series: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute Debt Payoff Years (F5) for each fiscal year.

    ``Debt Payoff Years = long_term_debt / owner_earnings``

    Pass/fail uses the **most recent** year's values.  Zero debt → 0 years
    (automatic pass).  Owner earnings ≤ 0 → infinite payoff (automatic fail,
    logged at WARNING).

    Parameters
    ----------
    balance_df:
        Columns: ``fiscal_year``, ``long_term_debt``.  Single ticker.
    owner_earnings_series:
        pd.Series indexed by fiscal year (int) with owner earnings values.
        Supplied by ``metrics_engine/owner_earnings.py``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        *annual_df* columns: ``fiscal_year``, ``debt_payoff_years``.
        *summary* keys: ``debt_payoff_years`` (most-recent year), ``pass`` (bool).
    """
    # Debt payoff threshold from config — determines pass/fail in the summary.
    # Per FORMULAS.md F5: ≤ 5 years passes, ≥ 5 years fails.
    max_years: float = float(get_threshold("hard_filters.max_debt_payoff_years"))

    # --- Step 1: Isolate required columns, sorted ascending by fiscal year ---
    bal = balance_df[["fiscal_year", "long_term_debt"]].sort_values("fiscal_year").reset_index(drop=True)

    # Convert owner_earnings pd.Series (keyed by fiscal year) into a DataFrame
    # for merging with the balance sheet data.
    oe_df = owner_earnings_series.rename("owner_earnings").reset_index()
    oe_df.columns = ["fiscal_year", "owner_earnings"]

    # --- Step 2: Join balance sheet debt with owner earnings on fiscal_year ---
    df = bal.merge(oe_df, on="fiscal_year", how="left")
    debt = df["long_term_debt"].astype(float)
    oe = df["owner_earnings"].astype(float)

    # --- Step 3: Compute payoff years, handling edge cases per FORMULAS.md F5 ---
    # Zero or negative debt → 0 years (automatic pass, no debt to retire)
    no_debt = debt.notna() & (debt <= 0)
    # Owner earnings ≤ 0 with outstanding debt → infinite payoff (automatic fail)
    oe_bad = oe.notna() & (oe <= 0) & ~no_debt
    # Normal case: both debt > 0 and OE > 0
    valid = debt.notna() & oe.notna() & (debt > 0) & (oe > 0)

    payoff = pd.Series(float("nan"), index=df.index, dtype=float)
    payoff.loc[no_debt] = 0.0
    payoff.loc[oe_bad] = float("inf")
    payoff.loc[valid] = (debt.loc[valid] / oe.loc[valid]).values

    # Log each year where OE ≤ 0 makes debt payoff infinite
    for yr in df.loc[oe_bad, "fiscal_year"].tolist():
        logger.warning("Debt payoff infinite for fiscal_year=%s: owner_earnings ≤ 0.", yr)

    # --- Step 4: Build output DataFrame and summary using most-recent year ---
    annual_df = pd.DataFrame({"fiscal_year": df["fiscal_year"].values, "debt_payoff_years": payoff.values})

    # Summary uses the most-recent fiscal year for pass/fail determination
    recent_payoff = float(payoff.loc[df["fiscal_year"].idxmax()]) if not df.empty else float("nan")
    passes = (not pd.isna(recent_payoff)) and (recent_payoff <= max_years)
    return annual_df, {"debt_payoff_years": recent_payoff, "pass": passes}


def compute_debt_to_equity(
    balance_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute Debt-to-Equity Ratio (F6) for each fiscal year.

    ``D/E = long_term_debt / shareholders_equity``

    Zero debt → D/E = 0.0.  Non-positive equity → D/E = NaN with
    ``negative_equity = True`` (logged at WARNING).

    Parameters
    ----------
    balance_df:
        Columns: ``fiscal_year``, ``long_term_debt``, ``shareholders_equity``.
        Single ticker.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        *annual_df* columns: ``fiscal_year``, ``de_ratio``, ``negative_equity``.
        *summary* keys: ``avg_de_10yr``, ``max_de``, ``latest_de``.

    Notes
    -----
    ``avg_de_10yr`` and ``max_de`` exclude NaN years from the computation.
    ``latest_de`` is NaN when the most-recent year has negative equity.
    """
    # --- Step 1: Isolate required columns, sorted ascending by fiscal year ---
    df = (
        balance_df[["fiscal_year", "long_term_debt", "shareholders_equity"]]
        .sort_values("fiscal_year")
        .reset_index(drop=True)
    )
    debt = df["long_term_debt"].astype(float)
    equity = df["shareholders_equity"].astype(float)

    # --- Step 2: Classify each year into edge-case buckets per FORMULAS.md F6 ---
    # Zero or negative debt → D/E = 0 (automatic pass, no leverage)
    no_debt = debt.notna() & (debt <= 0)
    # Non-positive equity → D/E is mathematically meaningless (negative denominator).
    # Per FORMULAS.md: do NOT compute, flag negative_equity = True.
    neg_eq_flag = equity.notna() & (equity <= 0)
    # Normal case: both debt > 0 and equity > 0
    valid = debt.notna() & equity.notna() & (debt > 0) & (equity > 0)

    # --- Step 3: Compute D/E ratio per year ---
    de = pd.Series(float("nan"), index=df.index, dtype=float)
    de.loc[no_debt] = 0.0
    de.loc[valid] = (debt.loc[valid] / equity.loc[valid]).values

    # Log each year with non-positive equity (e.g., large buyback programs)
    for yr in df.loc[neg_eq_flag, "fiscal_year"].tolist():
        logger.warning("Negative equity for fiscal_year=%s: shareholders_equity ≤ 0. D/E set to NaN.", yr)

    # --- Step 4: Build output DataFrame and summary statistics ---
    annual_df = pd.DataFrame({
        "fiscal_year": df["fiscal_year"].values,
        "de_ratio": de.values,
        "negative_equity": neg_eq_flag.values,
    })

    # Summary statistics exclude NaN years from the computation
    valid_de = de.dropna()
    avg_de = float(valid_de.mean()) if not valid_de.empty else float("nan")
    max_de = float(valid_de.max()) if not valid_de.empty else float("nan")
    # latest_de: the D/E ratio for the most recent fiscal year
    latest_de = float(de.loc[df["fiscal_year"].idxmax()]) if not df.empty else float("nan")

    return annual_df, {"avg_de_10yr": avg_de, "max_de": max_de, "latest_de": latest_de}


def compute_interest_coverage(
    income_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute Interest Expense Coverage (F9) for each fiscal year.

    ``Interest Burden = interest_expense / operating_income``

    Lower is better. A value of 0.10 means interest consumes 10% of EBIT.
    Years with non-positive operating income produce NaN (logged at WARNING).
    When interest expense is zero the burden is 0.0 (best case).

    Parameters
    ----------
    income_df:
        Columns: ``fiscal_year``, ``interest_expense``, ``operating_income``.
        Single ticker.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        *annual_df* columns: ``fiscal_year``, ``interest_pct_of_ebit``.
        *summary* keys: ``avg_interest_pct_10yr``.

    Notes
    -----
    ``interest_expense`` is expected to be stored as a positive number
    (expense convention). ``abs()`` is applied before dividing to guard
    against sign inconsistencies in raw data.
    """
    # --- Step 1: Isolate required columns, sorted ascending by fiscal year ---
    df = (
        income_df[["fiscal_year", "interest_expense", "operating_income"]]
        .sort_values("fiscal_year")
        .reset_index(drop=True)
    )
    interest = df["interest_expense"].astype(float)
    ebit = df["operating_income"].astype(float)

    # --- Step 2: Classify each year per FORMULAS.md F9 edge cases ---
    # Non-positive EBIT → interest burden is economically undefined.
    # Per FORMULAS.md: do not compute, flag automatic fail.
    bad_ebit = ebit.notna() & (ebit <= 0)
    # Normal case: EBIT > 0 and interest data available
    valid = ebit.notna() & (ebit > 0) & interest.notna()

    # --- Step 3: Compute interest burden per year ---
    # This is the Buffett framing: Interest Expense / EBIT (lower = better).
    # abs() guards against sign inconsistencies in raw interest_expense data.
    # When interest = 0, the result is 0.0 (best case — no interest burden).
    interest_pct = pd.Series(float("nan"), index=df.index, dtype=float)
    interest_pct.loc[valid] = (interest.loc[valid].abs() / ebit.loc[valid]).values

    # Log each year where EBIT is non-positive
    for yr in df.loc[bad_ebit, "fiscal_year"].tolist():
        logger.warning("Interest coverage undefined for fiscal_year=%s: operating_income ≤ 0. Setting NaN.", yr)

    # --- Step 4: Build output DataFrame and summary ---
    annual_df = pd.DataFrame({
        "fiscal_year": df["fiscal_year"].values,
        "interest_pct_of_ebit": interest_pct.values,
    })

    # Summary: average interest burden across all valid (non-NaN) years
    valid_vals = interest_pct.dropna()
    avg_int = float(valid_vals.mean()) if not valid_vals.empty else float("nan")
    return annual_df, {"avg_interest_pct_10yr": avg_int}

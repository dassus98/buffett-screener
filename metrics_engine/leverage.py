"""Computes balance sheet leverage and debt safety metrics: D/E, net debt/EBITDA, interest coverage, and current ratio."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_config

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
    cfg = get_config()
    max_years: float = float(cfg.get("hard_filters", {}).get("max_debt_payoff_years", 5.0))

    bal = balance_df[["fiscal_year", "long_term_debt"]].sort_values("fiscal_year").reset_index(drop=True)
    oe_df = owner_earnings_series.rename("owner_earnings").reset_index()
    oe_df.columns = ["fiscal_year", "owner_earnings"]

    df = bal.merge(oe_df, on="fiscal_year", how="left")
    debt = df["long_term_debt"].astype(float)
    oe = df["owner_earnings"].astype(float)

    no_debt = debt.notna() & (debt <= 0)
    oe_bad = oe.notna() & (oe <= 0) & ~no_debt
    valid = debt.notna() & oe.notna() & (debt > 0) & (oe > 0)

    payoff = pd.Series(float("nan"), index=df.index, dtype=float)
    payoff.loc[no_debt] = 0.0
    payoff.loc[oe_bad] = float("inf")
    payoff.loc[valid] = (debt.loc[valid] / oe.loc[valid]).values

    for yr in df.loc[oe_bad, "fiscal_year"].tolist():
        logger.warning("Debt payoff infinite for fiscal_year=%s: owner_earnings ≤ 0.", yr)

    annual_df = pd.DataFrame({"fiscal_year": df["fiscal_year"].values, "debt_payoff_years": payoff.values})

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
    df = (
        balance_df[["fiscal_year", "long_term_debt", "shareholders_equity"]]
        .sort_values("fiscal_year")
        .reset_index(drop=True)
    )
    debt = df["long_term_debt"].astype(float)
    equity = df["shareholders_equity"].astype(float)

    no_debt = debt.notna() & (debt <= 0)
    neg_eq_flag = equity.notna() & (equity <= 0)
    valid = debt.notna() & equity.notna() & (debt > 0) & (equity > 0)

    de = pd.Series(float("nan"), index=df.index, dtype=float)
    de.loc[no_debt] = 0.0
    de.loc[valid] = (debt.loc[valid] / equity.loc[valid]).values

    for yr in df.loc[neg_eq_flag, "fiscal_year"].tolist():
        logger.warning("Negative equity for fiscal_year=%s: shareholders_equity ≤ 0. D/E set to NaN.", yr)

    annual_df = pd.DataFrame({
        "fiscal_year": df["fiscal_year"].values,
        "de_ratio": de.values,
        "negative_equity": neg_eq_flag.values,
    })
    valid_de = de.dropna()
    avg_de = float(valid_de.mean()) if not valid_de.empty else float("nan")
    max_de = float(valid_de.max()) if not valid_de.empty else float("nan")
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
    df = (
        income_df[["fiscal_year", "interest_expense", "operating_income"]]
        .sort_values("fiscal_year")
        .reset_index(drop=True)
    )
    interest = df["interest_expense"].astype(float)
    ebit = df["operating_income"].astype(float)

    bad_ebit = ebit.notna() & (ebit <= 0)
    valid = ebit.notna() & (ebit > 0) & interest.notna()

    interest_pct = pd.Series(float("nan"), index=df.index, dtype=float)
    interest_pct.loc[valid] = (interest.loc[valid].abs() / ebit.loc[valid]).values

    for yr in df.loc[bad_ebit, "fiscal_year"].tolist():
        logger.warning("Interest coverage undefined for fiscal_year=%s: operating_income ≤ 0. Setting NaN.", yr)

    annual_df = pd.DataFrame({
        "fiscal_year": df["fiscal_year"].values,
        "interest_pct_of_ebit": interest_pct.values,
    })
    valid_vals = interest_pct.dropna()
    avg_int = float(valid_vals.mean()) if not valid_vals.empty else float("nan")
    return annual_df, {"avg_interest_pct_10yr": avg_int}

"""Module 2 orchestrator: compute F1-F16 metrics for every surviving ticker.

Reads financial data from DuckDB tables produced by Module 1
(data_acquisition), runs all 16 Buffett formula functions in dependency
order, assembles per-ticker summary dicts, computes composite scores, and
writes results back to DuckDB for consumption by Module 3 (screener) and
Module 4 (valuation_reports).

Data Lineage Contract
---------------------
Upstream producers (DuckDB tables written by Module 1):
    - ``income_statement``  (ticker, fiscal_year, net_income, total_revenue,
      gross_profit, sga, operating_income, interest_expense, eps_diluted,
      shares_outstanding_diluted)
    - ``balance_sheet``     (ticker, fiscal_year, long_term_debt,
      shareholders_equity, treasury_stock)
    - ``cash_flow``         (ticker, fiscal_year, depreciation_amortization,
      capital_expenditures, working_capital_change)
    - ``market_data``       (ticker, current_price_usd, pe_ratio_trailing, ...)
    - ``macro_data``        (key-value: us_treasury_10yr, goc_bond_10yr)
    - ``data_quality_log``  (ticker, drop) — used by ``get_surviving_tickers``

Formula modules invoked (dependency chain):
    Independent group:
        F3  profitability.compute_roe
        F7  profitability.compute_gross_margin
        F8  profitability.compute_sga_ratio
        F10 profitability.compute_net_margin
        F6  leverage.compute_debt_to_equity
        F9  leverage.compute_interest_coverage
        F12 capex.compute_capex_to_earnings
        F13 growth.compute_buyback_indicator
    F1-dependent chain:
        F1  owner_earnings.compute_owner_earnings
        F5  leverage.compute_debt_payoff         (requires F1 output)
        F2  returns.compute_initial_rate_of_return (requires F1 output)
    F4 standalone:
        F4  returns.compute_return_on_retained_earnings
    F11-dependent chain:
        F11 growth.compute_eps_cagr
        F14 valuation.compute_intrinsic_value    (requires F11 eps_cagr)
        F15 valuation.compute_margin_of_safety   (requires F14 weighted_iv)
        F16 valuation.compute_earnings_yield

Downstream consumers:
    - ``metrics_engine.composite_score.compute_all_composite_scores``
      -> reads the flat ``metrics_summary`` dicts assembled by
         ``_merge_all_summaries``.
    - DuckDB table ``buffett_metrics``         (per-ticker per-year detail)
    - DuckDB table ``buffett_metrics_summary`` (10-year aggregates + composite)
    - DuckDB table ``composite_scores``        (ranked composite scores)
    - Module 3 (screener) reads ``buffett_metrics_summary`` for hard/soft filters.
    - Module 4 (valuation_reports) reads all three tables for report generation.

Config dependencies:
    - This module does NOT read filter_config.yaml directly. All threshold
      access is delegated to the individual formula modules (which use
      ``get_threshold()``).
    - ``_extract_macro_rfr`` uses a hardcoded ``_FALLBACK_RFR = 0.04`` as a
      data-availability fallback when the FRED API macro data is unavailable
      (see inline rationale).

Public API
----------
compute_ticker_metrics(ticker) -> dict
    Compute all Buffett formula metrics for one ticker from DuckDB and return a
    flat ``metrics_summary`` dict.

run_metrics_engine() -> pd.DataFrame
    Process all surviving tickers, write results to DuckDB, and return the
    composite-score ranking DataFrame.

Callable as a module::

    python -m metrics_engine
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from data_acquisition.store import get_connection, get_surviving_tickers, read_table
from metrics_engine.capex import compute_capex_to_earnings
from metrics_engine.composite_score import compute_all_composite_scores
from metrics_engine.growth import compute_buyback_indicator, compute_eps_cagr
from metrics_engine.leverage import (
    compute_debt_payoff,
    compute_debt_to_equity,
    compute_interest_coverage,
)
from metrics_engine.owner_earnings import compute_owner_earnings
from metrics_engine.profitability import (
    compute_gross_margin,
    compute_net_margin,
    compute_roe,
    compute_sga_ratio,
)
from metrics_engine.returns import (
    compute_initial_rate_of_return,
    compute_return_on_retained_earnings,
)
from metrics_engine.valuation import (
    compute_earnings_yield,
    compute_intrinsic_value,
    compute_margin_of_safety,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers — data I/O
# ---------------------------------------------------------------------------


def _sort_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* sorted ascending by ``fiscal_year`` with a reset integer index.

    All formula modules expect chronological order; this helper normalises any
    unsorted DataFrames coming out of DuckDB before they reach computation code.
    """
    return df.sort_values("fiscal_year").reset_index(drop=True)


def _read_ticker_data(
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series | None, dict[str, float]]:
    """Read all Module-1 DuckDB tables for *ticker* and return normalised inputs.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol.

    Returns
    -------
    tuple
        ``(income_df, balance_df, cashflow_df, market_row, macro_dict)``
        where *market_row* is ``None`` when no market-data row exists and
        *macro_dict* maps ``"us_treasury_10yr"`` / ``"goc_bond_10yr"`` → float.
    """
    q = f"ticker = '{ticker}'"

    # --- Step 1: Read the three core financial-statement tables.
    #     These are mandatory — if any read fails the ticker is un-processable.
    try:
        income_df = read_table("income_statement", where=q)
        balance_df = read_table("balance_sheet", where=q)
        cashflow_df = read_table("cash_flow", where=q)
    except RuntimeError as exc:
        logger.error("DuckDB read failed for %s: %s", ticker, exc)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, {}

    # --- Step 2: Read market_data (single row per ticker).
    #     Market data is optional; price-dependent formulas (F2, F14-F16)
    #     will produce NaN when market_row is None.
    try:
        market_df = read_table("market_data", where=q)
        market_row: pd.Series | None = market_df.iloc[0] if not market_df.empty else None
    except RuntimeError:
        logger.warning("market_data not found for %s.", ticker)
        market_row = None

    # --- Step 3: Read macro_data (key-value table: us_treasury_10yr, goc_bond_10yr).
    #     Pivoted from key-value rows into a flat dict for _extract_macro_rfr.
    #     Non-numeric values are silently skipped.
    macro_dict: dict[str, float] = {}
    try:
        for _, row in read_table("macro_data").iterrows():
            try:
                macro_dict[str(row["key"])] = float(row["value"])
            except (ValueError, TypeError):
                pass
    except RuntimeError:
        logger.warning("macro_data table unavailable.")

    return income_df, balance_df, cashflow_df, market_row, macro_dict


def _extract_macro_rfr(macro_dict: dict[str, float], ticker: str) -> float:
    """Return the 10-year risk-free rate appropriate for *ticker*'s exchange.

    TSX tickers (``*.TO``) use the GoC 10-year bond yield; all others use the
    US Treasury 10-year yield.  Falls back to 4 % when the key is absent.
    """
    # Infrastructure constant — NOT a business threshold from filter_config.yaml.
    # This value is used ONLY when the FRED API macro_data table is completely
    # empty (e.g. API key missing, network failure).  A WARNING is logged so
    # the user knows results rely on this fallback rather than live rates.
    # Rationale for keeping it here rather than in config: it represents a
    # "reasonable recent historical average" for data-availability resilience,
    # not a scoring or filtering parameter.
    _FALLBACK_RFR: float = 0.04

    # --- Step 1: Select the appropriate bond-yield key based on exchange.
    #     TSX-listed tickers (*.TO) → Government of Canada 10-year bond yield.
    #     All other tickers                → US Treasury 10-year yield.
    key = "goc_bond_10yr" if ticker.upper().endswith(".TO") else "us_treasury_10yr"

    # --- Step 2: Look up the rate in the macro dict; fall back if absent.
    rfr = macro_dict.get(key)
    if rfr is None or pd.isna(rfr):
        logger.warning(
            "Risk-free-rate key '%s' missing from macro_data for %s. "
            "Using fallback %.2f.",
            key, ticker, _FALLBACK_RFR,
        )
        return _FALLBACK_RFR
    return float(rfr)


# ---------------------------------------------------------------------------
# Private helpers — per-formula group computations
# ---------------------------------------------------------------------------


def _compute_f1_and_f5(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Compute F1 (owner earnings) and F5 (debt payoff, which depends on F1).

    Returns
    -------
    tuple
        ``(f1_annual, f1_summary, f5_annual, f5_summary)``
    """
    # --- Step 1: Compute F1 (owner earnings) from all three financial statements.
    #     Owner earnings = net_income + D&A − CapEx ± working-capital change.
    f1_annual, f1_summary = compute_owner_earnings(income_df, cashflow_df, balance_df)

    # --- Step 2: Build a year-indexed owner-earnings Series for F5.
    #     F5 (debt payoff) divides long-term debt by owner earnings per year,
    #     so it requires F1's output as a pd.Series keyed by fiscal_year.
    oe_series = pd.Series(
        f1_annual["owner_earnings"].values,
        index=f1_annual["fiscal_year"].values,
        dtype=float,
    )

    # --- Step 3: Compute F5 (debt payoff years) — depends on F1 output.
    f5_annual, f5_summary = compute_debt_payoff(balance_df, oe_series)
    return f1_annual, f1_summary, f5_annual, f5_summary


def _compute_profitability(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
) -> tuple[tuple[pd.DataFrame, ...], tuple[dict[str, Any], ...]]:
    """Compute F3 (ROE), F7 (gross margin), F8 (SGA ratio), F10 (net margin).

    Returns
    -------
    tuple
        ``((roe_annual, gm_annual, sga_annual, nm_annual),``
        ``(roe_summary, gm_summary, sga_summary, nm_summary))``
    """
    # All four profitability formulas are independent of each other; no
    # dependency ordering is required within this group.

    # --- Step 1: F3 — ROE = net_income / shareholders_equity per year.
    roe_annual, roe_summary = compute_roe(income_df, balance_df)

    # --- Step 2: F7 — Gross margin = gross_profit / total_revenue per year.
    gm_annual, gm_summary = compute_gross_margin(income_df)

    # --- Step 3: F8 — SGA ratio = sga / gross_profit per year.
    sga_annual, sga_summary = compute_sga_ratio(income_df)

    # --- Step 4: F10 — Net margin = net_income / total_revenue per year.
    nm_annual, nm_summary = compute_net_margin(income_df)

    return (
        (roe_annual, gm_annual, sga_annual, nm_annual),
        (roe_summary, gm_summary, sga_summary, nm_summary),
    )


def _compute_leverage(
    balance_df: pd.DataFrame,
    income_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Compute F6 (debt-to-equity) and F9 (interest coverage).

    Returns
    -------
    tuple
        ``(de_annual, de_summary, ic_annual, ic_summary)``
    """
    # Both leverage formulas are independent of each other.

    # --- Step 1: F6 — D/E ratio = long_term_debt / shareholders_equity per year.
    de_annual, de_summary = compute_debt_to_equity(balance_df)

    # --- Step 2: F9 — Interest coverage = interest_expense / operating_income per year.
    ic_annual, ic_summary = compute_interest_coverage(income_df)

    return de_annual, de_summary, ic_annual, ic_summary


def _compute_returns(
    income_df: pd.DataFrame,
    f1_annual: pd.DataFrame,
    market_row: pd.Series | None,
    rfr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute F2 (initial rate of return) and F4 (return on retained earnings).

    F2 uses the most-recent year's owner earnings per diluted share.
    F4 assumes zero dividends paid — ``dividends_per_share`` is absent from the
    canonical schema; zero is the conservative (lower-retention) assumption.

    Returns
    -------
    tuple
        ``(f2_result, f4_result)``
    """
    # --- Step 1: Build chronologically sorted EPS and DPS series for F4.
    #     dividends_per_share is absent from the canonical schema; zero is the
    #     conservative assumption (lower retained earnings → lower F4 score).
    inc = _sort_by_year(income_df)
    eps_series = pd.Series(inc["eps_diluted"].values, index=inc["fiscal_year"].values, dtype=float)
    dps_series = pd.Series([0.0] * len(inc), index=inc["fiscal_year"].values, dtype=float)

    # --- Step 2: F4 — Return on retained earnings (Buffett's "$1 test").
    #     Uses cumulative retained EPS over the window divided by market-cap change.
    f4_result = compute_return_on_retained_earnings(eps_series, dps_series)

    # --- Step 3: Extract current price for F2 (initial rate of return).
    current_price = float(market_row["current_price_usd"]) if market_row is not None else float("nan")
    if inc.empty:
        return compute_initial_rate_of_return(float("nan"), current_price, rfr), f4_result

    # --- Step 4: Compute owner-earnings-per-share for F2.
    #     F2 = owner_earnings_per_share / current_price; compares to rfr × bond_multiplier.
    shares_latest = float(inc["shares_outstanding_diluted"].iloc[-1])
    oe_latest = (
        float(f1_annual.sort_values("fiscal_year")["owner_earnings"].iloc[-1])
        if not f1_annual.empty
        else float("nan")
    )
    if pd.isna(shares_latest) or shares_latest <= 0 or pd.isna(oe_latest):
        oe_per_share: float = float("nan")
    else:
        oe_per_share = oe_latest / shares_latest

    # --- Step 5: F2 — Initial rate of return = owner_earnings_per_share / price.
    return compute_initial_rate_of_return(oe_per_share, current_price, rfr), f4_result


def _compute_growth_formulas(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute F11 (EPS CAGR), F12 (CapEx / NI), F13 (buyback indicator).

    Returns
    -------
    tuple
        ``(f11_result, f12_result, f13_result)``
    """
    # --- Step 1: F11 — EPS CAGR over the 10-year window.
    #     eps_cagr feeds downstream into F14 (intrinsic value).
    inc = _sort_by_year(income_df)
    eps_series = pd.Series(inc["eps_diluted"].values, index=inc["fiscal_year"].values, dtype=float)
    f11_result = compute_eps_cagr(eps_series)

    # --- Step 2: F12 — CapEx as % of net income (capital efficiency).
    #     Align net_income to cash-flow fiscal years because the two statements
    #     may have different year coverage (outer join semantics).
    cf = _sort_by_year(cashflow_df)
    ni_lookup = inc.set_index("fiscal_year")["net_income"]
    ni_aligned = pd.Series(
        [float(ni_lookup.get(yr, float("nan"))) for yr in cf["fiscal_year"].values],
        dtype=float,
    )
    capex_series = pd.Series(cf["capital_expenditures"].values, dtype=float)
    f12_result = compute_capex_to_earnings(capex_series, ni_aligned)

    # --- Step 3: F13 — Buyback indicator (net share-count change over window).
    shares_series = pd.Series(inc["shares_outstanding_diluted"].values, dtype=float)
    f13_result = compute_buyback_indicator(shares_series)

    return f11_result, f12_result, f13_result


def _compute_valuation_formulas(
    income_df: pd.DataFrame,
    f11_result: dict[str, Any],
    market_row: pd.Series | None,
    rfr: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute F14 (intrinsic value), F15 (margin of safety), F16 (earnings yield).

    F14 depends on F11's ``eps_cagr``; F15 depends on F14's ``weighted_iv``.
    Historical P/E is approximated from the single trailing P/E in ``market_data``.

    Returns
    -------
    tuple
        ``(f14_result, f15_result, f16_result)``
    """
    # --- Step 1: Extract inputs shared by F14–F16 from market data.
    inc = _sort_by_year(income_df)
    current_eps = float(inc["eps_diluted"].iloc[-1]) if not inc.empty else float("nan")
    current_price = float(market_row["current_price_usd"]) if market_row is not None else float("nan")

    # Historical P/E approximation: we only have the single trailing P/E from
    # market_data.  Wrap it in a 1-element Series; compute_intrinsic_value
    # calculates median/average internally.  Empty Series when P/E is NaN.
    pe_trailing = float(market_row["pe_ratio_trailing"]) if market_row is not None else float("nan")
    historical_pe = pd.Series(
        dtype=float, data=([] if pd.isna(pe_trailing) else [pe_trailing])
    )

    # --- Step 2: F14 — Three-scenario intrinsic value (bear/base/bull DCF).
    #     Depends on F11's eps_cagr for projected EPS growth.
    f14_result = compute_intrinsic_value(
        current_eps, f11_result.get("eps_cagr", float("nan")), historical_pe, current_price, rfr
    )

    # --- Step 3: F15 — Margin of safety = (weighted_iv − price) / weighted_iv.
    #     Depends on F14's weighted_iv output.
    f15_result = compute_margin_of_safety(f14_result.get("weighted_iv", float("nan")), current_price)

    # --- Step 4: F16 — Earnings yield = EPS / price, compared against rfr.
    f16_result = compute_earnings_yield(current_eps, current_price, rfr)

    return f14_result, f15_result, f16_result


# ---------------------------------------------------------------------------
# Private helpers — assembly
# ---------------------------------------------------------------------------


def _build_ticker_annual_df(
    ticker: str,
    f1_annual: pd.DataFrame,
    roe_annual: pd.DataFrame,
    gm_annual: pd.DataFrame,
    sga_annual: pd.DataFrame,
    nm_annual: pd.DataFrame,
    f5_annual: pd.DataFrame,
    de_annual: pd.DataFrame,
    ic_annual: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-year formula DataFrames into a single ``buffett_metrics`` row-set.

    Uses outer joins so that fiscal years present in any formula's output are
    included, with NaN where a particular formula produced no value for that year.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``fiscal_year``, ``owner_earnings``, ``roe``,
        ``gross_margin``, ``sga_ratio``, ``net_margin``, ``debt_payoff_years``,
        ``de_ratio``, ``interest_pct_of_ebit``.
    """
    # --- Step 1: Start with F1 (owner earnings) as the base table.
    base = f1_annual[["fiscal_year", "owner_earnings"]].copy()

    # --- Step 2: Outer-join each per-year formula output onto the base.
    #     Outer joins ensure fiscal years present in ANY formula's output are
    #     included; years absent from a particular formula get NaN for that column.
    per_year: list[tuple[pd.DataFrame, str]] = [
        (roe_annual, "roe"),
        (gm_annual, "gross_margin"),
        (sga_annual, "sga_ratio"),
        (nm_annual, "net_margin"),
        (f5_annual, "debt_payoff_years"),
        (de_annual, "de_ratio"),
        (ic_annual, "interest_pct_of_ebit"),
    ]
    for merge_df, col in per_year:
        if col in merge_df.columns:
            base = base.merge(merge_df[["fiscal_year", col]], on="fiscal_year", how="outer")

    # --- Step 3: Prepend ticker column and sort chronologically.
    base.insert(0, "ticker", ticker)
    return base.sort_values("fiscal_year").reset_index(drop=True)


def _merge_all_summaries(
    ticker: str,
    f1_summary: dict, f2_result: dict, roe_summary: dict, f4_result: dict,
    f5_summary: dict, de_summary: dict, gm_summary: dict, sga_summary: dict,
    ic_summary: dict, nm_summary: dict, f11_result: dict, f12_result: dict,
    f13_result: dict, f14_result: dict, f15_result: dict, f16_result: dict,
) -> dict[str, Any]:
    """Flatten all per-formula summary dicts into one ``metrics_summary`` dict.

    F14's nested scenario dicts are flattened with ``"f14_{scenario}_"`` key
    prefixes.  The ``"pass"`` key from F5 is renamed to ``"debt_payoff_pass"``
    to avoid collision with the Python keyword in some downstream contexts.

    Returns
    -------
    dict
        Flat dict containing all formula outputs, keyed by field name.
    """
    # --- Step 1: Seed the merged dict with the ticker identifier.
    merged: dict[str, Any] = {"ticker": ticker}

    # --- Step 2: Merge all non-F14 formula outputs into the flat dict.
    #     F14 is excluded here because its nested scenario dicts require
    #     special flattening (Step 4).  All other formula outputs are flat
    #     dicts whose keys merge cleanly.
    for d in (
        f1_summary, f2_result, roe_summary, f4_result, f5_summary,
        de_summary, gm_summary, sga_summary, ic_summary, nm_summary,
        f11_result, f12_result, f13_result, f15_result, f16_result,
    ):
        merged.update(d)

    # --- Step 3: Rename F5's "pass" key to "debt_payoff_pass".
    #     Avoids collision with the Python keyword in downstream dataclass or
    #     dict-to-column conversion contexts.
    if "pass" in merged:
        merged["debt_payoff_pass"] = merged.pop("pass")

    # --- Step 4: Flatten F14 scenario dicts with "f14_{scenario}_" prefixes.
    #     E.g. f14_result["bear"]["present_value"] → merged["f14_bear_present_value"].
    for scenario in ("bear", "base", "bull"):
        for k, v in f14_result.get(scenario, {}).items():
            merged[f"f14_{scenario}_{k}"] = v

    # --- Step 5: Promote F14 top-level scalars (weighted_iv, meets_hurdle).
    merged["weighted_iv"] = f14_result.get("weighted_iv", float("nan"))
    merged["meets_hurdle"] = f14_result.get("meets_hurdle", False)

    return merged


# ---------------------------------------------------------------------------
# Private helpers — pure computation (no DuckDB; directly testable)
# ---------------------------------------------------------------------------


def _compute_all_from_data(
    ticker: str,
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    market_row: pd.Series | None,
    rfr: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run all F1–F16 formulas from pre-loaded DataFrames. No DuckDB access.

    Dependency chain honoured:
    - F1 → F5 (owner earnings used by debt-payoff)
    - F11 → F14 (EPS CAGR used by intrinsic-value)
    - F14 → F15 (weighted IV used by margin-of-safety)

    Parameters
    ----------
    ticker:
        Symbol (included in returned summary as ``"ticker"`` key).
    income_df, balance_df, cashflow_df:
        Canonical-schema DataFrames for a **single** ticker, pre-filtered.
    market_row:
        Single-row market data Series (``None`` if unavailable).
    rfr:
        10-year government bond yield as a decimal (e.g. ``0.04`` = 4 %).

    Returns
    -------
    tuple[dict, pd.DataFrame]
        ``(metrics_summary, annual_df)`` — *metrics_summary* is the flat dict
        consumed by ``compute_composite_score``; *annual_df* has one row per
        fiscal year suitable for the ``buffett_metrics`` DuckDB table.
    """
    # ---- Phase A: Independent formulas (no inter-formula dependencies) ----

    # --- Step 1: F1 → F5 chain (owner earnings → debt payoff).
    f1_annual, f1_summary, f5_annual, f5_summary = _compute_f1_and_f5(
        income_df, balance_df, cashflow_df
    )

    # --- Step 2: F3, F7, F8, F10 (profitability ratios — all independent).
    (roe_annual, gm_annual, sga_annual, nm_annual), (
        roe_summary, gm_summary, sga_summary, nm_summary
    ) = _compute_profitability(income_df, balance_df)

    # --- Step 3: F6, F9 (leverage ratios — all independent).
    de_annual, de_summary, ic_annual, ic_summary = _compute_leverage(balance_df, income_df)

    # --- Step 4: F2, F4 (return ratios — F2 depends on F1 output).
    f2_result, f4_result = _compute_returns(income_df, f1_annual, market_row, rfr)

    # ---- Phase B: Growth & valuation chain (F11 → F14 → F15) ----

    # --- Step 5: F11, F12, F13 (growth & efficiency metrics).
    f11_result, f12_result, f13_result = _compute_growth_formulas(income_df, cashflow_df)

    # --- Step 6: F14 → F15 → F16 (valuation — F14 depends on F11 eps_cagr).
    f14_result, f15_result, f16_result = _compute_valuation_formulas(
        income_df, f11_result, market_row, rfr
    )

    # ---- Phase C: Assembly ----

    # --- Step 7: Merge per-year DataFrames into a single annual_df for DuckDB.
    annual_df = _build_ticker_annual_df(
        ticker, f1_annual, roe_annual, gm_annual, sga_annual,
        nm_annual, f5_annual, de_annual, ic_annual,
    )

    # --- Step 8: Flatten all summary dicts into one metrics_summary dict.
    summary = _merge_all_summaries(
        ticker, f1_summary, f2_result, roe_summary, f4_result, f5_summary,
        de_summary, gm_summary, sga_summary, ic_summary, nm_summary,
        f11_result, f12_result, f13_result, f14_result, f15_result, f16_result,
    )

    return summary, annual_df


def _compute_all(ticker: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Read DuckDB data for *ticker* then delegate to :func:`_compute_all_from_data`.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        ``(metrics_summary, annual_df)`` — empty summary / empty DataFrame when
        no income-statement data is found in DuckDB.
    """
    # --- Step 1: Read all DuckDB inputs for this ticker.
    income_df, balance_df, cashflow_df, market_row, macro_dict = _read_ticker_data(ticker)

    # --- Step 2: Guard — income_statement is mandatory; abort early if empty.
    if income_df.empty:
        logger.error("No income_statement data for %s; returning empty metrics.", ticker)
        return {"ticker": ticker}, pd.DataFrame()

    # --- Step 3: Resolve the appropriate risk-free rate (TSX vs US exchange).
    rfr = _extract_macro_rfr(macro_dict, ticker)

    # --- Step 4: Delegate to pure-computation function (no further DuckDB access).
    return _compute_all_from_data(ticker, income_df, balance_df, cashflow_df, market_row, rfr)


# ---------------------------------------------------------------------------
# Private helpers — DuckDB write
# ---------------------------------------------------------------------------


def _init_metrics_tables(conn: Any) -> None:
    """Create Module 2 output tables in DuckDB if they do not yet exist.

    These CREATE TABLE IF NOT EXISTS statements are idempotent and run at the
    start of every ``run_metrics_engine`` invocation.  The full tables are then
    replaced by ``_write_table_full_replace`` once computation is complete.
    """
    # --- Step 1: Per-year detail table (one row per ticker × fiscal_year).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS buffett_metrics (
            ticker VARCHAR, fiscal_year INTEGER,
            owner_earnings DOUBLE, roe DOUBLE, gross_margin DOUBLE,
            sga_ratio DOUBLE, net_margin DOUBLE, debt_payoff_years DOUBLE,
            de_ratio DOUBLE, interest_pct_of_ebit DOUBLE,
            PRIMARY KEY (ticker, fiscal_year)
        )
    """)

    # --- Step 2: 10-year aggregate summary (one row per ticker).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS buffett_metrics_summary (ticker VARCHAR PRIMARY KEY)"
    )

    # --- Step 3: Ranked composite scores (one row per ticker).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS composite_scores (ticker VARCHAR PRIMARY KEY)"
    )


def _write_table_full_replace(
    conn: Any,
    table_name: str,
    df: pd.DataFrame,
) -> None:
    """Replace *table_name* entirely with *df* using DROP + CREATE AS SELECT.

    This is the appropriate write pattern for Module 2 tables because the
    full ticker universe is always reprocessed on each pipeline run.

    Parameters
    ----------
    conn:
        Active DuckDB connection.
    table_name:
        Name of the table to replace.
    df:
        DataFrame whose columns define the new table schema.
    """
    # --- Step 1: Guard — skip if the DataFrame is empty or None.
    if df is None or df.empty:
        logger.warning(
            "_write_table_full_replace: empty DataFrame for '%s', skipping.", table_name
        )
        return

    # --- Step 2: Register the DataFrame as a virtual table in DuckDB.
    #     DuckDB's native pandas integration allows SELECT from the registered
    #     name without materialising an intermediate INSERT loop.
    conn.register("_metrics_staging", df)
    try:
        # --- Step 3: Atomic DROP + CREATE replaces the old table entirely.
        #     This is the correct pattern for Module 2 because the full universe
        #     is always reprocessed on each pipeline run (no incremental upsert).
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _metrics_staging")
    finally:
        # --- Step 4: Always unregister the staging view to avoid leaking
        #     references to the DataFrame's memory.
        try:
            conn.unregister("_metrics_staging")
        except Exception:
            pass

    logger.info("Wrote %d rows to '%s'.", len(df), table_name)


def _write_annual_metrics(
    conn: Any,
    annual_dfs: list[pd.DataFrame],
) -> None:
    """Concatenate all per-ticker annual DataFrames and write to ``buffett_metrics``."""
    # --- Step 1: Guard — nothing to write if no tickers produced annual data.
    if not annual_dfs:
        logger.warning("_write_annual_metrics: no per-year data to write.")
        return

    # --- Step 2: Stack all tickers' annual DataFrames and write in one shot.
    combined = pd.concat(annual_dfs, ignore_index=True)
    _write_table_full_replace(conn, "buffett_metrics", combined)


def _write_summary_with_scores(
    conn: Any,
    all_summaries: dict[str, dict],
    composite_df: pd.DataFrame,
) -> None:
    """Write per-ticker summary metrics augmented with composite scores.

    Merges composite-score values from *composite_df* into each ticker row
    before writing to ``buffett_metrics_summary``.
    """
    # --- Step 1: Guard — nothing to write if no summaries were produced.
    if not all_summaries:
        return

    # --- Step 2: Build a ticker → composite_score lookup from the scored DF.
    score_map: dict[str, float] = {}
    if not composite_df.empty and "ticker" in composite_df.columns:
        score_map = dict(zip(composite_df["ticker"], composite_df["composite_score"]))

    # --- Step 3: Augment each ticker's summary dict with its composite score.
    #     Filter out any non-serialisable values (DataFrames/Series) that may
    #     have leaked into the summary dict from intermediate formula outputs.
    rows: list[dict] = []
    for ticker, summary in all_summaries.items():
        row = {k: v for k, v in summary.items()
               if not isinstance(v, (pd.DataFrame, pd.Series))}
        row["composite_score"] = score_map.get(ticker, float("nan"))
        rows.append(row)

    # --- Step 4: Write the combined summary table.
    _write_table_full_replace(conn, "buffett_metrics_summary", pd.DataFrame(rows))


def _process_all_tickers(
    tickers: list[str],
) -> tuple[dict[str, dict], list[pd.DataFrame]]:
    """Run :func:`_compute_all` for each ticker with per-ticker error isolation.

    Tickers whose computation raises any exception are logged at ERROR and
    skipped; all other tickers are unaffected.

    Returns
    -------
    tuple[dict, list]
        ``(all_summaries, annual_dfs)`` for every successfully processed ticker.
    """
    all_summaries: dict[str, dict] = {}
    annual_dfs: list[pd.DataFrame] = []

    for i, ticker in enumerate(tickers):
        # --- Progress logging: emit a status line every 100 tickers so the
        #     user can gauge pipeline throughput on large universes.
        if i > 0 and i % 100 == 0:
            logger.info(
                "Metrics engine progress: %d / %d tickers processed.", i, len(tickers)
            )
        try:
            # --- Per-ticker error isolation: a failure in one ticker's
            #     computation must never abort the remaining tickers.
            summary, annual_df = _compute_all(ticker)
            all_summaries[ticker] = summary
            if annual_df is not None and not annual_df.empty:
                annual_dfs.append(annual_df)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to compute metrics for %s: %s", ticker, exc, exc_info=True
            )

    logger.info(
        "Metrics engine: completed %d / %d tickers.", len(all_summaries), len(tickers)
    )
    return all_summaries, annual_dfs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ticker_metrics(ticker: str) -> dict[str, Any]:
    """Read DuckDB financial data for *ticker* and compute all F1–F16 metrics.

    Dependency chain: F1 → F5; F11 → F14; F14 → F15.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol (e.g. ``"KO"``, ``"AAPL"``).

    Returns
    -------
    dict
        Flat ``metrics_summary`` dict containing all formula outputs.  Every
        key expected by :func:`~metrics_engine.composite_score.compute_composite_score`
        is present (``NaN`` when data is insufficient).
        Returns ``{"ticker": ticker}`` when no income-statement rows are found.
    """
    # Delegate to _compute_all (which reads DuckDB, resolves RFR, then runs
    # _compute_all_from_data).  Discard the annual_df — callers of the public
    # API only need the flat summary dict.
    summary, _ = _compute_all(ticker)
    return summary


def run_metrics_engine() -> pd.DataFrame:
    """Compute F1–F16 for all surviving tickers and write results to DuckDB.

    Pipeline:

    1. Fetch surviving tickers from ``data_quality_log``.
    2. Compute metrics for each ticker with per-ticker error isolation.
    3. Compute composite scores via :mod:`metrics_engine.composite_score`.
    4. Write ``buffett_metrics``, ``buffett_metrics_summary``, ``composite_scores``.
    5. Log progress every 100 tickers.

    Returns
    -------
    pd.DataFrame
        Ranked composite-score table (``ticker``, ``composite_score``, and
        ``score_*`` criterion columns), or an empty DataFrame when no tickers
        survive data-quality filtering.
    """
    # --- Step 1: Obtain DuckDB connection and ensure output tables exist.
    conn = get_connection()
    _init_metrics_tables(conn)

    # --- Step 2: Fetch surviving tickers from data_quality_log (Module 1).
    #     Empty list → nothing to process; return immediately.
    tickers = get_surviving_tickers()
    if not tickers:
        logger.warning("run_metrics_engine: no surviving tickers in data_quality_log.")
        return pd.DataFrame()

    # --- Step 3: Compute F1–F16 for every ticker with error isolation.
    logger.info("Metrics engine: starting computation for %d tickers.", len(tickers))
    all_summaries, annual_dfs = _process_all_tickers(tickers)

    if not all_summaries:
        logger.error("Metrics engine: all tickers failed; nothing written to DuckDB.")
        return pd.DataFrame()

    # --- Step 4: Compute Tier-2 composite scores from the flat summary dicts.
    composite_df = compute_all_composite_scores(all_summaries)

    # --- Step 5: Write all three Module 2 output tables to DuckDB.
    #     - buffett_metrics:         per-year detail (all tickers × years)
    #     - buffett_metrics_summary: 10-year aggregates + composite score
    #     - composite_scores:        ranked composite-score table
    _write_annual_metrics(conn, annual_dfs)
    _write_summary_with_scores(conn, all_summaries, composite_df)
    _write_table_full_replace(conn, "composite_scores", composite_df)

    # --- Step 6: Log the top-ranked ticker for quick pipeline sanity checks.
    top = composite_df.iloc[0] if not composite_df.empty else None
    logger.info(
        "Metrics engine complete: %d tickers scored. Top: %s (%.1f).",
        len(composite_df),
        top["ticker"] if top is not None else "—",
        top["composite_score"] if top is not None else 0.0,
    )
    return composite_df

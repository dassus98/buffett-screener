"""Module 2 orchestrator: compute F1–F16 metrics for every surviving ticker.

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
    """Return *df* sorted ascending by ``fiscal_year`` with a reset integer index."""
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
    try:
        income_df = read_table("income_statement", where=q)
        balance_df = read_table("balance_sheet", where=q)
        cashflow_df = read_table("cash_flow", where=q)
    except RuntimeError as exc:
        logger.error("DuckDB read failed for %s: %s", ticker, exc)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, {}

    try:
        market_df = read_table("market_data", where=q)
        market_row: pd.Series | None = market_df.iloc[0] if not market_df.empty else None
    except RuntimeError:
        logger.warning("market_data not found for %s.", ticker)
        market_row = None

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
    _FALLBACK: float = 0.04
    key = "goc_bond_10yr" if ticker.upper().endswith(".TO") else "us_treasury_10yr"
    rfr = macro_dict.get(key)
    if rfr is None or pd.isna(rfr):
        logger.warning(
            "Risk-free-rate key '%s' missing from macro_data for %s. Using fallback %.2f.",
            key, ticker, _FALLBACK,
        )
        return _FALLBACK
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
    f1_annual, f1_summary = compute_owner_earnings(income_df, cashflow_df, balance_df)
    oe_series = pd.Series(
        f1_annual["owner_earnings"].values,
        index=f1_annual["fiscal_year"].values,
        dtype=float,
    )
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
    roe_annual, roe_summary = compute_roe(income_df, balance_df)
    gm_annual, gm_summary = compute_gross_margin(income_df)
    sga_annual, sga_summary = compute_sga_ratio(income_df)
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
    de_annual, de_summary = compute_debt_to_equity(balance_df)
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
    inc = _sort_by_year(income_df)
    eps_series = pd.Series(inc["eps_diluted"].values, index=inc["fiscal_year"].values, dtype=float)
    dps_series = pd.Series([0.0] * len(inc), index=inc["fiscal_year"].values, dtype=float)
    f4_result = compute_return_on_retained_earnings(eps_series, dps_series)

    current_price = float(market_row["current_price_usd"]) if market_row is not None else float("nan")
    if inc.empty:
        return compute_initial_rate_of_return(float("nan"), current_price, rfr), f4_result

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
    inc = _sort_by_year(income_df)
    eps_series = pd.Series(inc["eps_diluted"].values, index=inc["fiscal_year"].values, dtype=float)
    f11_result = compute_eps_cagr(eps_series)

    cf = _sort_by_year(cashflow_df)
    ni_lookup = inc.set_index("fiscal_year")["net_income"]
    ni_aligned = pd.Series(
        [float(ni_lookup.get(yr, float("nan"))) for yr in cf["fiscal_year"].values],
        dtype=float,
    )
    capex_series = pd.Series(cf["capital_expenditures"].values, dtype=float)
    f12_result = compute_capex_to_earnings(capex_series, ni_aligned)
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
    inc = _sort_by_year(income_df)
    current_eps = float(inc["eps_diluted"].iloc[-1]) if not inc.empty else float("nan")
    current_price = float(market_row["current_price_usd"]) if market_row is not None else float("nan")
    pe_trailing = float(market_row["pe_ratio_trailing"]) if market_row is not None else float("nan")
    historical_pe = pd.Series(
        dtype=float, data=([] if pd.isna(pe_trailing) else [pe_trailing])
    )
    f14_result = compute_intrinsic_value(
        current_eps, f11_result.get("eps_cagr", float("nan")), historical_pe, current_price, rfr
    )
    f15_result = compute_margin_of_safety(f14_result.get("weighted_iv", float("nan")), current_price)
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
    base = f1_annual[["fiscal_year", "owner_earnings"]].copy()
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
    merged: dict[str, Any] = {"ticker": ticker}
    for d in (
        f1_summary, f2_result, roe_summary, f4_result, f5_summary,
        de_summary, gm_summary, sga_summary, ic_summary, nm_summary,
        f11_result, f12_result, f13_result, f15_result, f16_result,
    ):
        merged.update(d)
    if "pass" in merged:
        merged["debt_payoff_pass"] = merged.pop("pass")
    for scenario in ("bear", "base", "bull"):
        for k, v in f14_result.get(scenario, {}).items():
            merged[f"f14_{scenario}_{k}"] = v
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
    f1_annual, f1_summary, f5_annual, f5_summary = _compute_f1_and_f5(
        income_df, balance_df, cashflow_df
    )
    (roe_annual, gm_annual, sga_annual, nm_annual), (
        roe_summary, gm_summary, sga_summary, nm_summary
    ) = _compute_profitability(income_df, balance_df)
    de_annual, de_summary, ic_annual, ic_summary = _compute_leverage(balance_df, income_df)
    f2_result, f4_result = _compute_returns(income_df, f1_annual, market_row, rfr)
    f11_result, f12_result, f13_result = _compute_growth_formulas(income_df, cashflow_df)
    f14_result, f15_result, f16_result = _compute_valuation_formulas(
        income_df, f11_result, market_row, rfr
    )
    annual_df = _build_ticker_annual_df(
        ticker, f1_annual, roe_annual, gm_annual, sga_annual,
        nm_annual, f5_annual, de_annual, ic_annual,
    )
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
    income_df, balance_df, cashflow_df, market_row, macro_dict = _read_ticker_data(ticker)
    if income_df.empty:
        logger.error("No income_statement data for %s; returning empty metrics.", ticker)
        return {"ticker": ticker}, pd.DataFrame()
    rfr = _extract_macro_rfr(macro_dict, ticker)
    return _compute_all_from_data(ticker, income_df, balance_df, cashflow_df, market_row, rfr)


# ---------------------------------------------------------------------------
# Private helpers — DuckDB write
# ---------------------------------------------------------------------------


def _init_metrics_tables(conn: Any) -> None:
    """Create Module 2 output tables in DuckDB if they do not yet exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS buffett_metrics (
            ticker VARCHAR, fiscal_year INTEGER,
            owner_earnings DOUBLE, roe DOUBLE, gross_margin DOUBLE,
            sga_ratio DOUBLE, net_margin DOUBLE, debt_payoff_years DOUBLE,
            de_ratio DOUBLE, interest_pct_of_ebit DOUBLE,
            PRIMARY KEY (ticker, fiscal_year)
        )
    """)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS buffett_metrics_summary (ticker VARCHAR PRIMARY KEY)"
    )
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
    if df is None or df.empty:
        logger.warning(
            "_write_table_full_replace: empty DataFrame for '%s', skipping.", table_name
        )
        return
    conn.register("_metrics_staging", df)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _metrics_staging")
    finally:
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
    if not annual_dfs:
        logger.warning("_write_annual_metrics: no per-year data to write.")
        return
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
    if not all_summaries:
        return
    score_map: dict[str, float] = {}
    if not composite_df.empty and "ticker" in composite_df.columns:
        score_map = dict(zip(composite_df["ticker"], composite_df["composite_score"]))
    rows: list[dict] = []
    for ticker, summary in all_summaries.items():
        row = {k: v for k, v in summary.items()
               if not isinstance(v, (pd.DataFrame, pd.Series))}
        row["composite_score"] = score_map.get(ticker, float("nan"))
        rows.append(row)
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
        if i > 0 and i % 100 == 0:
            logger.info(
                "Metrics engine progress: %d / %d tickers processed.", i, len(tickers)
            )
        try:
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
    conn = get_connection()
    _init_metrics_tables(conn)

    tickers = get_surviving_tickers()
    if not tickers:
        logger.warning("run_metrics_engine: no surviving tickers in data_quality_log.")
        return pd.DataFrame()

    logger.info("Metrics engine: starting computation for %d tickers.", len(tickers))
    all_summaries, annual_dfs = _process_all_tickers(tickers)

    if not all_summaries:
        logger.error("Metrics engine: all tickers failed; nothing written to DuckDB.")
        return pd.DataFrame()

    composite_df = compute_all_composite_scores(all_summaries)
    _write_annual_metrics(conn, annual_dfs)
    _write_summary_with_scores(conn, all_summaries, composite_df)
    _write_table_full_replace(conn, "composite_scores", composite_df)

    top = composite_df.iloc[0] if not composite_df.empty else None
    logger.info(
        "Metrics engine complete: %d tickers scored. Top: %s (%.1f).",
        len(composite_df),
        top["ticker"] if top is not None else "—",
        top["composite_score"] if top is not None else 0.0,
    )
    return composite_df

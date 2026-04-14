"""Public surface of the data_acquisition package: fetch, validate, and cache raw financial data.

The primary entry point for external callers is :func:`run_data_acquisition`, which
executes all seven pipeline steps in sequence and persists every result to DuckDB.

Typical usage
-------------
    from data_acquisition import run_data_acquisition
    result = run_data_acquisition(use_cache=True)

    # Or directly from the command line:
    python -m data_acquisition [--no-cache]

Data lineage contract
---------------------
This module is the **orchestration layer for Module 1** (docs/ARCHITECTURE.md §2).
It imports and sequences all sub-modules; it does not implement any data-fetching
or validation logic itself.

Upstream sub-modules invoked (in pipeline order):
  store.py            → init_db() creates DuckDB schema (Step 1)
  universe.py         → get_universe() fetches investable universe (Step 2)
  financials.py       → fetch_all_financials() fetches 10yr statements (Step 3)
  market_data.py      → fetch_market_data() fetches current prices (Step 4)
  macro_data.py       → fetch_macro_data() fetches treasury/FX rates (Step 5)
  data_quality.py     → run_data_quality_check() enforces drop rules (Step 6)
  store.py            → write_dataframe() persists every result to DuckDB

Downstream consumers:
  metrics_engine/     → reads all DuckDB tables written by this pipeline
  output/pipeline_runner.py → calls run_data_acquisition() as pipeline Stage 1

Config dependencies:
  use_cache parameter → forwarded to get_universe() and fetch_macro_data()
  All sub-module configs are read internally by each sub-module via
  filter_config_loader.get_config() — this orchestrator has no direct
  config dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from data_acquisition.data_quality import run_data_quality_check
from data_acquisition.financials import fetch_all_financials
from data_acquisition.macro_data import fetch_macro_data
from data_acquisition.market_data import fetch_market_data
from data_acquisition.store import init_db, write_dataframe
from data_acquisition.universe import get_universe

logger = logging.getLogger(__name__)

__all__ = ["run_data_acquisition"]


# ---------------------------------------------------------------------------
# Public orchestration function
# ---------------------------------------------------------------------------

def run_data_acquisition(use_cache: bool = True) -> dict[str, Any]:
    """Run the full data acquisition pipeline end-to-end.

    Executes seven sequential steps: initialise the database, fetch the
    investable universe, retrieve financial statements, collect current
    market data, pull macro indicators, run data-quality checks, and log
    a final summary.  Every result is persisted to DuckDB before the next
    step begins so partial progress is preserved on failure.

    Parameters
    ----------
    use_cache:
        If ``True`` (default), use cached universe and macro data when fresh.
        Pass ``False`` to force a full re-fetch from all APIs.

    Returns
    -------
    dict with keys:
        - ``universe_size`` (int) — total tickers fetched
        - ``survivors`` (int) — tickers that passed quality checks
        - ``dropped`` (int) — tickers excluded by quality checks
        - ``macro`` (dict) — raw macro data as returned by :func:`fetch_macro_data`
    """
    logger.info("=== Data acquisition pipeline starting (use_cache=%s) ===", use_cache)

    # Step 1 — initialise DuckDB schema (idempotent).
    init_db()

    # Step 2 — universe
    universe_df = _acquire_universe(use_cache)
    n_universe = len(universe_df)

    # Step 3 — financial statements
    financials, sub_log = _acquire_financials(universe_df)

    # Step 4 — current market data
    tickers: list[str] = universe_df["ticker"].dropna().tolist() if not universe_df.empty else []
    _acquire_market_data(tickers)

    # Step 5 — macro indicators
    macro = _acquire_macro_data(use_cache)

    # Step 6 — data quality
    _, survivors_df = _run_quality(financials, sub_log)
    n_survivors = len(survivors_df)
    n_dropped = n_universe - n_survivors

    # Step 7 — summary
    logger.info(
        "%d tickers survived data quality. %d dropped. Pipeline complete.",
        n_survivors,
        n_dropped,
    )

    return {
        "universe_size": n_universe,
        "survivors": n_survivors,
        "dropped": n_dropped,
        "macro": macro,
    }


# ---------------------------------------------------------------------------
# Step helpers (one function per pipeline step)
# ---------------------------------------------------------------------------

def _acquire_universe(use_cache: bool) -> pd.DataFrame:
    """Step 2: fetch the investable universe and write it to DuckDB.

    Parameters
    ----------
    use_cache:
        Passed through to :func:`get_universe`.

    Returns
    -------
    pd.DataFrame
        Universe DataFrame (may be empty if all API calls failed).
    """
    universe_df = get_universe(use_cache=use_cache)
    if not universe_df.empty:
        write_dataframe("universe", universe_df)
    logger.info("Universe: %d tickers fetched and stored.", len(universe_df))
    return universe_df


def _acquire_financials(
    universe_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Step 3: fetch all financial statements and write them to DuckDB.

    Parameters
    ----------
    universe_df:
        Universe DataFrame with a ``ticker`` column.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], list[dict]]
        Financials dict and aggregated substitution log, as returned by
        :func:`fetch_all_financials`.
    """
    financials, sub_log = fetch_all_financials(universe_df)

    for table_name in ("income_statement", "balance_sheet", "cash_flow"):
        stmt_df = financials.get(table_name, pd.DataFrame())
        if not stmt_df.empty:
            write_dataframe(table_name, stmt_df)

    write_dataframe("substitution_log", _build_substitution_df(sub_log))

    stmt_df = financials.get("income_statement", pd.DataFrame())
    n_tickers = int(stmt_df["ticker"].nunique()) if not stmt_df.empty else 0
    logger.info("Financial statements stored for %d tickers.", n_tickers)
    return financials, sub_log


def _acquire_market_data(tickers: list[str]) -> None:
    """Step 4: fetch current market data and write it to DuckDB.

    Parameters
    ----------
    tickers:
        List of ticker symbols to fetch market data for.
    """
    if not tickers:
        logger.warning("_acquire_market_data: no tickers provided, skipping.")
        return
    market_df = fetch_market_data(tickers)
    if not market_df.empty:
        write_dataframe("market_data", market_df)
    logger.info("Market data stored for %d tickers.", len(market_df))


def _acquire_macro_data(use_cache: bool = True) -> dict[str, Any]:
    """Step 5: fetch macro indicators and write them to DuckDB.

    Parameters
    ----------
    use_cache:
        If ``True`` (default), use cached macro data when fresh (< 1 day).
        Pass ``False`` to force a fresh fetch from FRED.

    Returns
    -------
    dict
        Raw macro dict as returned by :func:`fetch_macro_data`.
    """
    macro = fetch_macro_data(use_cache=use_cache)
    macro_df = _build_macro_df(macro)
    if not macro_df.empty:
        write_dataframe("macro_data", macro_df)
    display = {k: v for k, v in macro.items() if k != "as_of_date"}
    logger.info("Macro data stored: %s", display)
    return macro


def _run_quality(
    financials: dict[str, pd.DataFrame],
    sub_log: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 6: run data quality checks and persist the report to DuckDB.

    Parameters
    ----------
    financials:
        Financials dict from :func:`_acquire_financials`.
    sub_log:
        Substitution log from :func:`fetch_all_financials`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (quality_report_df, survivors_df) from :func:`run_data_quality_check`.
    """
    quality_df, survivors_df = run_data_quality_check(financials, sub_log)
    if not quality_df.empty:
        write_dataframe("data_quality_log", quality_df)
    return quality_df, survivors_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_macro_df(macro: dict[str, Any]) -> pd.DataFrame:
    """Pivot the flat macro dict into key-value rows matching the DuckDB schema.

    The ``macro_data`` table has columns ``(key, value, as_of_date)``.  This
    function filters out the ``as_of_date`` key itself and the string
    ``as_of_date`` value is applied as a column on every numeric row.

    Parameters
    ----------
    macro:
        Dict as returned by :func:`fetch_macro_data`.  Expected keys:
        ``us_treasury_10yr``, ``usd_cad_rate``, ``as_of_date``.

    Returns
    -------
    pd.DataFrame
        Columns: ``key``, ``value``, ``as_of_date``.  Empty DataFrame with
        those columns if no numeric values are found.
    """
    as_of_date: str = str(macro.get("as_of_date", ""))
    rows = [
        {"key": k, "value": float(v), "as_of_date": as_of_date}
        for k, v in macro.items()
        if k != "as_of_date" and isinstance(v, (int, float))
    ]
    if not rows:
        return pd.DataFrame(columns=["key", "value", "as_of_date"])
    return pd.DataFrame(rows)


def _build_substitution_df(sub_log: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert substitution log list to a DataFrame with canonical columns.

    Parameters
    ----------
    sub_log:
        List of substitution-event dicts.  Each entry must include
        ``ticker``, ``fiscal_year``, ``buffett_field``, ``api_field_used``,
        and ``confidence`` keys.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with the correct columns if ``sub_log`` is empty.
    """
    _COLS = ["ticker", "fiscal_year", "buffett_field", "api_field_used", "confidence"]
    if not sub_log:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(sub_log)

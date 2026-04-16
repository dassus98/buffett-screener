"""Fetches 10 years of annual financial statements for every universe ticker.

Primary source is FMP (Financial Modeling Prep); yfinance is used as an
automatic fallback when FMP is unavailable or rate-limited.  The fallback is
transparent: yfinance field names are already listed as substitutes in
``schema.LINE_ITEM_MAP``, so ``normalize_statement()`` handles both sources
via ``resolve_all_fields()``.

Retrieves income statements, balance sheets, and cash flow statements, maps every
field to canonical schema names via ``schema.resolve_all_fields``, and persists raw
JSON responses for auditability.

Authoritative spec: docs/DATA_SOURCES.md §3, docs/FORMULAS.md F1–F16 line items.
All API configuration comes from config/filter_config.yaml via api_config / filter_config_loader.

Data lineage contract
---------------------
Upstream dependencies:
  api_config.py          → build_fmp_url, fmp_limiter, resilient_request
  schema.py              → CANONICAL_COLUMNS, LINE_ITEM_MAP, resolve_all_fields
  filter_config_loader   → get_config (for universe.required_history_years,
                            data_sources.fmp.rate_limit_per_min,
                            data_sources.store_raw_responses)

Config dependency map (all from config/filter_config.yaml):
  universe.required_history_years  → fetch_financial_statements (FMP limit param)
  data_sources.fmp.rate_limit_per_min → _log_eta (ETA calculation)
  data_sources.store_raw_responses → _persist_raw (raw JSON toggle)

Downstream consumers:
  store.py               → writes income_statement, balance_sheet, cash_flow,
                            substitution_log DataFrames to DuckDB
  data_quality.py        → uses substitution log to assess field coverage and
                            flag tickers for drop/review
  metrics_engine/        → reads normalised financial tables from DuckDB

Key exports
-----------
fetch_financial_statements(ticker) -> dict[str, pd.DataFrame | None]
    Pull three FMP statement endpoints for one ticker.

normalize_statement(raw_df, statement_type) -> tuple[pd.DataFrame, list[dict]]
    Map raw FMP columns to canonical schema names using schema.resolve_all_fields.

fetch_all_financials(universe_df, batch_size) -> tuple[dict[str, pd.DataFrame], list[dict]]
    Batch-fetch all tickers, log progress, isolate per-ticker errors.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
from typing import Any

import pandas as pd

from data_acquisition.api_config import build_fmp_url, fmp_limiter, resilient_request
from data_acquisition.schema import (
    CANONICAL_COLUMNS,
    LINE_ITEM_MAP,
    resolve_field,
)
from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: FMP stable endpoint paths for each statement type.
#: Ticker is passed as the ``symbol`` query parameter (not in the path).
_ENDPOINTS: dict[str, str] = {
    "income_statement": "/income-statement",
    "balance_sheet": "/balance-sheet-statement",
    "cash_flow": "/cash-flow-statement",
}

#: FMP date field used to derive fiscal_year.
_FMP_DATE_FIELD = "date"
_FMP_FISCAL_YEAR_FIELD = "calendarYear"

#: Columns always added to normalised DataFrames alongside canonical fields.
_METADATA_COLUMNS = ("ticker", "fiscal_year")

#: Directory for raw JSON persistence relative to project root.
_RAW_DIR_RELATIVE = pathlib.Path("data") / "raw" / "financials"

#: Requests per ticker (3 statements).  Used for estimated-time logging.
_REQUESTS_PER_TICKER = 3

#: Minimum FMP requests-per-minute assumed for ETA (conservative fallback).
_MIN_RATE = 60

#: After this many consecutive all-None FMP responses, switch entirely to
#: yfinance for the remaining tickers (avoids wasting time on 429 errors).
_MAX_CONSECUTIVE_FMP_FAILURES = 3


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_financial_statements(
    ticker: str,
) -> dict[str, pd.DataFrame | None]:
    """Fetch annual income statement, balance sheet, and cash flow for one ticker.

    Parameters
    ----------
    ticker:
        Exchange-native ticker symbol (e.g. ``"AAPL"``, ``"SHOP.TO"``).
        TSX tickers must already carry the ``.TO`` suffix (added by
        universe.py when the exchange is ``"TSX"``); if the suffix is absent
        from a TSX ticker it is appended here as a defensive measure.

    Returns
    -------
    dict[str, pd.DataFrame | None]
        Keys: ``"income_statement"``, ``"balance_sheet"``, ``"cash_flow"``.
        Value is a raw (un-normalised) DataFrame on success, ``None`` on failure.

    Notes
    -----
    Raw JSON responses are written to ``data/raw/financials/{ticker}.json``
    before any transformation, provided ``data_sources.store_raw_responses``
    is truthy in filter_config.yaml.  Errors during raw-save do not abort the
    fetch.
    """
    fmp_ticker = _resolve_fmp_ticker(ticker)
    results: dict[str, pd.DataFrame | None] = {}
    raw_responses: dict[str, Any] = {}

    # Read the history window from config instead of hardcoding limit=10.
    cfg = get_config()
    history_years: int = int(
        cfg.get("universe", {}).get("required_history_years", 10)
    )

    for stmt_type, endpoint in _ENDPOINTS.items():
        url, params = build_fmp_url(
            endpoint,
            use_stable=True,
            symbol=fmp_ticker,
            period="annual",
            limit=history_years,
        )

        try:
            # max_retries=0: fail fast on 429 so the yfinance fallback in
            # fetch_all_financials kicks in quickly instead of burning 7+
            # seconds of backoff waits per ticker.
            data = resilient_request(
                url, params=params, rate_limiter=fmp_limiter, max_retries=0,
            )
        except Exception as exc:
            logger.warning(
                "Failed to fetch %s for %s: %s", stmt_type, ticker, exc
            )
            results[stmt_type] = None
            continue

        if not isinstance(data, list) or len(data) == 0:
            logger.warning(
                "Empty response for %s / %s (got %s rows).",
                ticker,
                stmt_type,
                len(data) if isinstance(data, list) else "non-list",
            )
            results[stmt_type] = None
            continue

        results[stmt_type] = pd.DataFrame(data)
        raw_responses[stmt_type] = data
        logger.debug(
            "Fetched %s for %s: %d rows.", stmt_type, ticker, len(data)
        )

    _persist_raw(ticker, raw_responses)
    return results


def _fetch_statements_yfinance(
    ticker: str,
) -> dict[str, pd.DataFrame | None]:
    """Fetch annual financial statements for one ticker via yfinance.

    Used as a fallback when FMP is unavailable (rate-limited, paid-only, etc.).
    yfinance field names (e.g. ``"Net Income"``, ``"Total Revenue"``) are listed
    as substitutes in ``schema.LINE_ITEM_MAP``, so ``normalize_statement()``
    handles them transparently via ``resolve_all_fields()``.

    The yfinance Ticker object exposes three statement properties:

    * ``.financials`` → annual income statement
    * ``.balance_sheet`` → annual balance sheet
    * ``.cashflow`` → annual cash flow

    Each returns a DataFrame with field names as the index and period-end
    dates as columns.  This function transposes the data so each row is one
    fiscal year (matching the format ``normalize_statement`` expects).

    Parameters
    ----------
    ticker:
        Ticker symbol (e.g. ``"AAPL"``, ``"SHOP.TO"``).

    Returns
    -------
    dict[str, pd.DataFrame | None]
        Same structure as :func:`fetch_financial_statements`:
        keys ``"income_statement"``, ``"balance_sheet"``, ``"cash_flow"``.
        Value is a transposed DataFrame on success, ``None`` on failure.

    Notes
    -----
    - No API key required.  Rate limiting is handled by yfinance internally.
    - Raw responses are NOT persisted to ``data/raw/`` (only FMP raw JSON is).
    - All monetary values are in full USD dollars (same as FMP), so
      ``_apply_value_conventions`` applies the same ÷1000 conversion.
    """
    import yfinance as yf

    _YF_ATTRS: dict[str, str] = {
        "income_statement": "financials",
        "balance_sheet": "balance_sheet",
        "cash_flow": "cashflow",
    }

    results: dict[str, pd.DataFrame | None] = {}

    try:
        t = yf.Ticker(ticker)
    except Exception as exc:
        logger.warning("yfinance Ticker(%s) construction failed: %s", ticker, exc)
        return {k: None for k in _YF_ATTRS}

    for stmt_type, attr_name in _YF_ATTRS.items():
        try:
            raw = getattr(t, attr_name)
            if raw is None or raw.empty:
                logger.debug("yfinance: no %s data for %s.", stmt_type, ticker)
                results[stmt_type] = None
                continue

            # yfinance returns fields as index, dates as columns.
            # Transpose so each row = one fiscal year, columns = field names.
            df = raw.T.copy()
            dates = pd.to_datetime(df.index)
            df = df.reset_index(drop=True)
            df["calendarYear"] = dates.year.astype(str)
            df["date"] = dates.strftime("%Y-%m-%d")
            results[stmt_type] = df
            logger.debug(
                "yfinance: fetched %s for %s (%d years).",
                stmt_type,
                ticker,
                len(df),
            )
        except Exception as exc:
            logger.warning(
                "yfinance fallback failed for %s / %s: %s",
                ticker,
                stmt_type,
                exc,
            )
            results[stmt_type] = None

    return results


def normalize_statement(
    raw_df: pd.DataFrame,
    statement_type: str,
    ticker: str = "",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Map raw FMP columns to canonical schema names for one statement.

    For each row in ``raw_df``, calls ``schema.resolve_all_fields`` to try the
    ideal API field first, then substitutes in priority order.  The resolved
    value is stored under the canonical ``buffett_name``; the source field and
    confidence are recorded in the substitution log.

    Missing critical fields (resolve confidence == ``"DROP"``) are stored as
    ``NaN`` in the output — the drop decision is made downstream by
    ``data_quality.py``, not here.

    Parameters
    ----------
    raw_df:
        Raw DataFrame from FMP (one row per fiscal year, columns are FMP field
        names).  Must be non-empty.
    statement_type:
        One of ``"income_statement"``, ``"balance_sheet"``, ``"cash_flow"``.
        Used only for logging context.
    ticker:
        Ticker symbol for logging.  Optional but strongly recommended.

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        - Normalised DataFrame with columns: ``ticker``, ``fiscal_year``, and
          one column per canonical ``buffett_name`` in ``CANONICAL_COLUMNS``.
        - Substitution log: list of dicts with keys
          ``ticker``, ``fiscal_year``, ``buffett_field``, ``api_field_used``,
          ``confidence``.  One entry per row per substituted or missing field.

    Notes
    -----
    - ``fiscal_year`` is extracted from FMP's ``calendarYear`` field if present,
      falling back to the year component of the ``date`` field.
    - All monetary values arrive in full USD dollars from FMP.  Unit conversion
      to USD thousands (÷ 1000) is applied to all fields except ``eps_diluted``
      and ``shares_outstanding_diluted`` (per docs/DATA_SOURCES.md §11).
    - ``capital_expenditures`` is negated if positive (sign convention: always
      stored negative).
    """
    if raw_df is None or raw_df.empty:
        return _empty_normalised_df(ticker), []

    normalised_rows: list[dict[str, Any]] = []
    substitution_log: list[dict[str, Any]] = []

    for _, row in raw_df.iterrows():
        row_dict = row.to_dict()
        fiscal_year = _extract_fiscal_year(row_dict)

        canon_row: dict[str, Any] = {
            "ticker": ticker,
            "fiscal_year": fiscal_year,
        }

        # Only resolve fields belonging to this statement type.
        # Avoids thousands of spurious "Unresolvable" WARNINGs from trying
        # e.g. balance_sheet fields against income_statement rows.
        for buffett_name, item in LINE_ITEM_MAP.items():
            if item.statement != statement_type:
                continue

            value, field_used, confidence = resolve_field(
                row_dict, buffett_name,
            )
            canon_col = CANONICAL_COLUMNS[buffett_name]

            if value is None:
                canon_row[canon_col] = float("nan")
                if confidence in ("DROP", "FLAG"):
                    substitution_log.append({
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "buffett_field": buffett_name,
                        "api_field_used": field_used,
                        "confidence": confidence,
                    })
            else:
                processed_value = _apply_value_conventions(
                    buffett_name, value, ticker
                )
                canon_row[canon_col] = processed_value

                if field_used != item.ideal_field:
                    substitution_log.append({
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "buffett_field": buffett_name,
                        "api_field_used": field_used,
                        "confidence": confidence,
                    })

        normalised_rows.append(canon_row)

    if not normalised_rows:
        return _empty_normalised_df(ticker), substitution_log

    result_df = pd.DataFrame(normalised_rows)
    result_df = _ensure_canonical_columns(result_df, statement_type, ticker)
    result_df = result_df.sort_values("fiscal_year", ascending=True).reset_index(drop=True)
    return result_df, substitution_log


def fetch_all_financials(
    universe_df: pd.DataFrame,
    batch_size: int = 50,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Batch-fetch financial statements for every ticker in the universe.

    Parameters
    ----------
    universe_df:
        Universe DataFrame from ``universe.get_universe()``.  Must contain a
        ``ticker`` column and, optionally, an ``exchange`` column used to
        identify TSX tickers.
    batch_size:
        How often to emit a progress log.  Default: every 50 tickers.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], list[dict]]
        - Dict with keys ``"income_statement"``, ``"balance_sheet"``,
          ``"cash_flow"``, each holding a master DataFrame concatenated
          across all successfully fetched tickers.
        - Aggregated substitution log across all tickers and all statements.

    Notes
    -----
    - Per-ticker failures are isolated: a single bad ticker is logged at ERROR
      and skipped; the batch continues.
    - Estimated completion time is logged at INFO before the batch starts,
      based on the FMP rate limit from config.
    - Progress is logged every ``batch_size`` tickers at INFO level.
    - **yfinance fallback**: if FMP returns no data (all three statements
      ``None``) for ``_MAX_CONSECUTIVE_FMP_FAILURES`` consecutive tickers,
      the batch switches entirely to yfinance for the remaining tickers.
      Individual FMP failures also trigger a per-ticker yfinance retry.
      This makes the pipeline runnable with zero paid API keys.
    """
    tickers: list[str] = universe_df["ticker"].dropna().unique().tolist()

    # Pre-suffix TSX tickers with '.TO' when the universe provides an exchange
    # column.  FMP requires the '.TO' suffix to resolve Canadian equities.
    if "exchange" in universe_df.columns:
        tsx_tickers = set(
            universe_df.loc[
                universe_df["exchange"].str.upper() == "TSX", "ticker"
            ].dropna()
        )
        tickers = [
            t if t.endswith(".TO") or t not in tsx_tickers else f"{t}.TO"
            for t in tickers
        ]
        if tsx_tickers:
            logger.info(
                "Pre-suffixed %d TSX ticker(s) with '.TO' for FMP compatibility.",
                len(tsx_tickers),
            )

    total = len(tickers)

    if total == 0:
        logger.warning("fetch_all_financials: universe_df contains no tickers.")
        return _empty_master_dfs(), []

    _log_eta(total)

    master: dict[str, list[pd.DataFrame]] = {
        "income_statement": [],
        "balance_sheet": [],
        "cash_flow": [],
    }
    all_substitutions: list[dict[str, Any]] = []
    failed_tickers: list[str] = []
    consecutive_fmp_failures = 0
    use_yfinance_only = False

    for idx, ticker in enumerate(tickers, start=1):
        try:
            if use_yfinance_only:
                stmt_dfs = _fetch_statements_yfinance(ticker)
            else:
                stmt_dfs = fetch_financial_statements(ticker)
                # Detect FMP rate-limiting: all three statements returned None.
                if all(v is None for v in stmt_dfs.values()):
                    consecutive_fmp_failures += 1
                    logger.info(
                        "FMP returned no data for %s (%d consecutive). "
                        "Trying yfinance fallback...",
                        ticker,
                        consecutive_fmp_failures,
                    )
                    stmt_dfs = _fetch_statements_yfinance(ticker)
                    if (
                        consecutive_fmp_failures
                        >= _MAX_CONSECUTIVE_FMP_FAILURES
                    ):
                        logger.warning(
                            "FMP failed for %d consecutive tickers — "
                            "switching to yfinance for the remaining "
                            "%d tickers.",
                            consecutive_fmp_failures,
                            total - idx,
                        )
                        use_yfinance_only = True
                else:
                    consecutive_fmp_failures = 0

            for stmt_type in ("income_statement", "balance_sheet", "cash_flow"):
                raw = stmt_dfs.get(stmt_type)
                if raw is None or raw.empty:
                    logger.debug(
                        "No %s data for %s — skipping normalisation.",
                        stmt_type,
                        ticker,
                    )
                    continue

                norm_df, subs = normalize_statement(raw, stmt_type, ticker=ticker)
                if not norm_df.empty:
                    master[stmt_type].append(norm_df)
                all_substitutions.extend(subs)

        except Exception as exc:
            logger.error(
                "Unexpected error processing ticker %s: %s — skipping.",
                ticker,
                exc,
                exc_info=True,
            )
            failed_tickers.append(ticker)
            continue

        if idx % batch_size == 0 or idx == total:
            logger.info(
                "Financials progress: %d / %d tickers fetched (%.0f%%).",
                idx,
                total,
                100 * idx / total,
            )

    if failed_tickers:
        logger.warning(
            "fetch_all_financials: %d ticker(s) failed and were skipped: %s",
            len(failed_tickers),
            failed_tickers[:20],  # cap log length
        )

    combined: dict[str, pd.DataFrame] = {}
    for stmt_type, frames in master.items():
        if frames:
            combined[stmt_type] = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=["ticker", "fiscal_year"])
                .reset_index(drop=True)
            )
        else:
            combined[stmt_type] = _empty_normalised_df("")

    logger.info(
        "fetch_all_financials complete. income=%d rows, balance=%d rows, cashflow=%d rows. "
        "Substitutions logged: %d.",
        len(combined["income_statement"]),
        len(combined["balance_sheet"]),
        len(combined["cash_flow"]),
        len(all_substitutions),
    )

    return combined, all_substitutions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_fmp_ticker(ticker: str) -> str:
    """Return the FMP-compatible ticker string.

    TSX tickers must carry the ``.TO`` suffix for FMP to resolve them.
    If a ticker contains a ``.`` and does not already end with ``.TO``,
    leave it unchanged (it may be a share-class suffix like ``BRK.B``).

    Parameters
    ----------
    ticker:
        Raw ticker from universe DataFrame.

    Returns
    -------
    str
        FMP-ready ticker.
    """
    if ticker.endswith(".TO"):
        return ticker
    # Heuristic: TSX tickers from FMP are already ``SHOP.TO`` style; if we
    # get a bare ticker that should be TSX (universe sets exchange="TSX") that
    # detail is not available here.  The caller (fetch_all_financials) could
    # pre-suffix if needed.  For single-ticker calls, callers must pass the
    # correct symbol.
    return ticker


def _extract_fiscal_year(row: dict[str, Any]) -> int:
    """Extract an integer fiscal year from a raw FMP row.

    Tries ``calendarYear`` first, then falls back to the year component of the
    ``date`` field (format ``YYYY-MM-DD``).

    Parameters
    ----------
    row:
        One row from an FMP statement response as a dict.

    Returns
    -------
    int
        Four-digit fiscal year, or 0 if neither field is available / parseable.
    """
    year_raw = row.get(_FMP_FISCAL_YEAR_FIELD)
    if year_raw is not None:
        try:
            return int(year_raw)
        except (ValueError, TypeError):
            pass

    date_raw = row.get(_FMP_DATE_FIELD, "")
    if isinstance(date_raw, str) and len(date_raw) >= 4:
        try:
            return int(date_raw[:4])
        except (ValueError, TypeError):
            pass

    logger.debug("Could not extract fiscal year from row: %s", row)
    return 0


def _apply_value_conventions(
    buffett_name: str,
    value: Any,
    ticker: str,
) -> Any:
    """Apply sign and unit conventions to a resolved field value.

    Rules (per docs/DATA_SOURCES.md §11 and FORMULAS.md F1):

    1. ``capital_expenditures``: always stored negative.  Negate if positive.
    2. Per-share fields (``eps_diluted``, ``shares_outstanding_diluted``):
       NOT divided by 1000 — already in correct units.
    3. All other monetary fields: divide by 1000 to convert from full USD
       dollars (FMP) to USD thousands (schema convention).

    Parameters
    ----------
    buffett_name:
        Canonical field name.
    value:
        Raw resolved value (may be int, float, str, or None).
    ticker:
        Ticker for logging.

    Returns
    -------
    Any
        Processed value, or the original value if it cannot be coerced to float.
    """
    _PER_SHARE_FIELDS = {"eps_diluted", "shares_outstanding_diluted"}
    _MONETARY_FIELDS = set(LINE_ITEM_MAP.keys()) - _PER_SHARE_FIELDS

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value

    if buffett_name == "capital_expenditures":
        if numeric > 0:
            logger.warning(
                "CapEx sign corrected for %s: was +%.0f → −%.0f",
                ticker,
                numeric,
                numeric,
            )
            numeric = -numeric
        # CapEx is already in full dollars from FMP; convert to thousands.
        return numeric / 1_000.0

    if buffett_name in _PER_SHARE_FIELDS:
        # Per-share and share-count fields: return as-is (no ÷1000).
        return numeric

    # All other monetary fields: convert full USD → USD thousands.
    return numeric / 1_000.0


def _ensure_canonical_columns(
    df: pd.DataFrame,
    statement_type: str,
    ticker: str,
) -> pd.DataFrame:
    """Add any canonical columns missing from ``df`` as NaN columns.

    Ensures the output DataFrame always has the same column set regardless of
    which fields FMP returned, making downstream concatenation safe.

    Parameters
    ----------
    df:
        Partially-built normalised DataFrame.
    statement_type:
        One of the three statement type strings — used to filter LINE_ITEM_MAP.
    ticker:
        For logging.

    Returns
    -------
    pd.DataFrame
        DataFrame with all expected columns present.
    """
    expected = list(_METADATA_COLUMNS) + [
        CANONICAL_COLUMNS[name]
        for name, item in LINE_ITEM_MAP.items()
        if item.statement == statement_type
    ]
    for col in expected:
        if col not in df.columns:
            logger.debug(
                "Adding missing column '%s' as NaN for ticker %s.", col, ticker
            )
            df[col] = float("nan")

    present = [c for c in expected if c in df.columns]
    return df[present]


def _empty_normalised_df(ticker: str) -> pd.DataFrame:
    """Return an empty DataFrame with metadata columns only.

    Parameters
    ----------
    ticker:
        Ticker label (unused in output, kept for API consistency).

    Returns
    -------
    pd.DataFrame
        Zero-row DataFrame with ``ticker`` and ``fiscal_year`` columns.
    """
    return pd.DataFrame(columns=list(_METADATA_COLUMNS))


def _empty_master_dfs() -> dict[str, pd.DataFrame]:
    """Return empty DataFrames for all three statement types."""
    return {st: _empty_normalised_df("") for st in _ENDPOINTS}


def _persist_raw(ticker: str, raw_responses: dict[str, Any]) -> None:
    """Write raw API JSON to disk for auditability.

    Only writes if ``data_sources.store_raw_responses`` is true in config.
    Errors are caught and logged — a persistence failure must never abort
    the fetch pipeline.

    Parameters
    ----------
    ticker:
        Used to name the output file.
    raw_responses:
        Dict mapping statement type → raw list-of-dicts from FMP.
    """
    if not raw_responses:
        return

    try:
        cfg = get_config()
        if not cfg.get("data_sources", {}).get("store_raw_responses", True):
            return
    except Exception:
        pass  # If config unavailable, default to saving raw responses.

    project_root = pathlib.Path(__file__).parent.parent
    raw_dir = project_root / _RAW_DIR_RELATIVE
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = raw_dir / f"{ticker}.json"
        with out_path.open("w") as fh:
            json.dump(raw_responses, fh)
        logger.debug("Raw financials saved: %s", out_path)
    except Exception as exc:
        logger.error("Failed to save raw financials for %s: %s", ticker, exc)


def _log_eta(total_tickers: int) -> None:
    """Log estimated completion time for the batch.

    Parameters
    ----------
    total_tickers:
        Total number of tickers to fetch.
    """
    try:
        cfg = get_config()
        rate = cfg.get("data_sources", {}).get("fmp", {}).get(
            "rate_limit_per_min", _MIN_RATE
        )
    except Exception:
        rate = _MIN_RATE

    total_requests = total_tickers * _REQUESTS_PER_TICKER
    eta_minutes = math.ceil(total_requests / max(int(rate), 1))
    logger.info(
        "fetch_all_financials: %d tickers × %d statements = %d FMP requests. "
        "Estimated time at %d req/min: ~%d minute(s).",
        total_tickers,
        _REQUESTS_PER_TICKER,
        total_requests,
        rate,
        eta_minutes,
    )

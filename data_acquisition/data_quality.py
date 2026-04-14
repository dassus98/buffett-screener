"""Validates data completeness and enforces drop rules for the Buffett screener pipeline.

Assesses each ticker's financial statement DataFrames against two quality thresholds
(authoritative spec: docs/DATA_SOURCES.md §4 Drop Rule, §9 Data Quality Protocol):

  1. **Year coverage**: ``years_available`` must be ≥ ``universe.min_history_years``
     (default 8). Tickers with fewer fiscal years are dropped.
  2. **Field coverage**: Each field in ``get_drop_required_fields()`` must have
     non-null values for ≥ ``data_quality.min_field_coverage_years`` fiscal years.
     Tickers failing any field are dropped.

Diagnostics (not gates):
  - Substitution density: logs WARNING if a ticker used > 2 field substitutes.
  - Cross-validation: spot-checks FMP vs yfinance for ``net_income``,
    ``total_revenue``, ``eps_diluted``; logs WARNING where divergence > 5%.

All thresholds are read from ``config/filter_config.yaml`` at runtime.

Data lineage contract
---------------------
Upstream dependencies:
  schema.py              → LINE_ITEM_MAP (statement type lookup for each field),
                            get_drop_required_fields() (10 canonical fields with
                            drop_if_missing=True)
  filter_config_loader   → get_config() for all thresholds (see Config dependency map)
  financials.py          → produces ``financials`` dict and ``substitution_log`` list
                            consumed by run_data_quality_check()
  yfinance               → cross_validate_sample fetches comparison values

Config dependency map (all from config/filter_config.yaml):
  universe.min_history_years              → assess_ticker_quality (year count check)
  data_quality.min_field_coverage_years   → assess_ticker_quality (field coverage check)
  data_quality.max_substitutions_before_flag → assess_ticker_quality (substitution warning)
  data_quality.cross_validate_sample_size → cross_validate_sample (default sample size)
  data_quality.cross_validate_tolerance   → cross_validate_sample (divergence threshold)

Downstream consumers:
  __init__.py            → run_data_quality_check() called as pipeline Step 6
  store.py               → writes quality_report_df to DuckDB ``data_quality_log`` table
  screener/              → reads survivors_df (tickers with drop=False) for filtering

Key exports
-----------
assess_ticker_quality(ticker, income_df, balance_df, cashflow_df) -> dict
    Per-ticker quality dict with drop decision and reason.
run_data_quality_check(financials, substitution_log) -> tuple[pd.DataFrame, pd.DataFrame]
    Batch assessment; returns (quality_report_df, survivors_df); saves CSV.
cross_validate_sample(financials, sample_size) -> pd.DataFrame
    Diagnostic FMP-vs-yfinance comparison on a random sample.
"""

from __future__ import annotations

import logging
import pathlib
import random
from typing import Any

import pandas as pd
import yfinance as yf

from data_acquisition.schema import LINE_ITEM_MAP, get_drop_required_fields
from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Financial statement keys expected in the ``financials`` dict.
_STMT_KEYS: tuple[str, ...] = ("income_statement", "balance_sheet", "cash_flow")

#: Canonical fields compared during cross-validation.
_CROSS_VALIDATE_FIELDS: tuple[str, ...] = ("net_income", "total_revenue", "eps_diluted")

#: yfinance income-statement row labels for cross-validation (priority order).
_YF_INCOME_LABELS: dict[str, list[str]] = {
    "net_income": ["Net Income", "Net Income From Continuing Operations"],
    "total_revenue": ["Total Revenue", "Revenue"],
}

#: Output report path relative to project root.
_REPORT_RELATIVE_PATH: pathlib.Path = (
    pathlib.Path("data") / "processed" / "data_quality_report.csv"
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def assess_ticker_quality(
    ticker: str,
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    substitutions_count: int = 0,
) -> dict[str, Any]:
    """Assess data-completeness quality for a single ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol (used only for logging and the returned dict).
    income_df:
        Income statement DataFrame filtered to this ticker. May be empty.
    balance_df:
        Balance sheet DataFrame filtered to this ticker. May be empty.
    cashflow_df:
        Cash flow DataFrame filtered to this ticker. May be empty.
    substitutions_count:
        Number of distinct fields that used a non-ideal API substitute
        (sourced from the substitution log by the caller). Default 0.

    Returns
    -------
    dict with keys:

    - ``ticker`` (str)
    - ``years_available`` (int) — distinct fiscal years across all statements
    - ``missing_critical_fields`` (list[str]) — required-drop fields with
      fewer than ``min_field_coverage_years`` non-null observations
    - ``substitutions_count`` (int)
    - ``drop`` (bool) — True if quality thresholds are not met
    - ``drop_reason`` (str or None) — human-readable explanation, or None

    Notes
    -----
    Logs WARNING if ``substitutions_count`` exceeds the
    ``data_quality.max_substitutions_before_flag`` config threshold.
    """
    cfg = get_config()
    # Year count threshold: minimum distinct fiscal years across all statements.
    min_years: int = int(cfg["universe"]["min_history_years"])
    # Field coverage threshold: minimum non-null observations per drop-required field.
    # Semantically distinct from min_years — uses data_quality.min_field_coverage_years.
    min_field_years: int = int(
        cfg.get("data_quality", {}).get("min_field_coverage_years", min_years)
    )
    max_subs: int = int(cfg.get("data_quality", {}).get("max_substitutions_before_flag", 2))

    stmt_map = {
        "income_statement": income_df,
        "balance_sheet": balance_df,
        "cash_flow": cashflow_df,
    }
    years_available = _count_years_available(income_df, balance_df, cashflow_df)
    missing_critical = _find_missing_critical_fields(stmt_map, min_field_years)

    if substitutions_count > max_subs:
        logger.warning(
            "assess_ticker_quality [%s]: %d fields required substitution (threshold %d).",
            ticker, substitutions_count, max_subs,
        )

    reasons: list[str] = []
    if years_available < min_years:
        reasons.append(
            f"Insufficient fiscal year coverage: {years_available} years "
            f"(minimum {min_years})"
        )
    if missing_critical:
        reasons.append(
            f"Critical fields missing coverage: {', '.join(sorted(missing_critical))}"
        )

    drop = bool(reasons)
    drop_reason: str | None = "; ".join(reasons) if reasons else None

    if drop:
        logger.warning(
            "assess_ticker_quality [%s]: DROPPED — %s", ticker, drop_reason
        )

    return {
        "ticker": ticker,
        "years_available": years_available,
        "missing_critical_fields": missing_critical,
        "substitutions_count": substitutions_count,
        "drop": drop,
        "drop_reason": drop_reason,
    }


def run_data_quality_check(
    financials: dict[str, pd.DataFrame],
    substitution_log: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run data quality assessment for every ticker in the financials dict.

    Calls :func:`assess_ticker_quality` for each unique ticker found across
    all statement DataFrames, aggregates substitution counts from the log,
    logs a pass/fail summary, and saves the report to CSV.

    Parameters
    ----------
    financials:
        Dict produced by ``financials.fetch_all_financials``. Keys:
        ``"income_statement"``, ``"balance_sheet"``, ``"cash_flow"``.
        Each value is a concatenated DataFrame of all tickers.
    substitution_log:
        List of substitution-log dicts from ``financials.normalize_statement``.
        Each entry has keys: ``ticker``, ``fiscal_year``, ``buffett_field``,
        ``api_field_used``, ``confidence``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - **quality_report_df**: one row per ticker with columns
          ``ticker``, ``years_available``, ``missing_critical_fields``
          (comma-separated string), ``substitutions_count``, ``drop``,
          ``drop_reason``. Saved to ``data/processed/data_quality_report.csv``.
        - **survivors_df**: tickers where ``drop=False``, with a single
          ``ticker`` column. Used by downstream steps to filter the universe.

    Notes
    -----
    Per-ticker assessment errors are logged at ERROR and skipped; they do
    not abort the batch.
    """
    tickers = _collect_all_tickers(financials)
    sub_counts = _count_substitutions_per_ticker(substitution_log)

    reports: list[dict[str, Any]] = []
    for ticker in tickers:
        income_df = _filter_to_ticker(financials.get("income_statement"), ticker)
        balance_df = _filter_to_ticker(financials.get("balance_sheet"), ticker)
        cashflow_df = _filter_to_ticker(financials.get("cash_flow"), ticker)
        try:
            result = assess_ticker_quality(
                ticker, income_df, balance_df, cashflow_df, sub_counts.get(ticker, 0)
            )
        except Exception as exc:
            logger.error(
                "run_data_quality_check: assessment failed for %s: %s", ticker, exc
            )
            continue
        reports.append(result)

    quality_report_df = _build_quality_report_df(reports)
    _log_quality_summary(quality_report_df)
    _save_quality_report(quality_report_df)

    if quality_report_df.empty or "drop" not in quality_report_df.columns:
        survivors_df = pd.DataFrame(columns=["ticker"])
    else:
        survivors_df = (
            quality_report_df[~quality_report_df["drop"]][["ticker"]]
            .reset_index(drop=True)
        )
    return quality_report_df, survivors_df


def cross_validate_sample(
    financials: dict[str, pd.DataFrame],
    sample_size: int = 20,
) -> pd.DataFrame:
    """Spot-check FMP income statement values against yfinance for a random sample.

    This is a **diagnostic function only** — it logs warnings but never drops
    tickers. Intended for use after ingestion to surface systematic FMP/yfinance
    discrepancies.

    Fields compared: ``net_income``, ``total_revenue``, ``eps_diluted`` for
    the most recent fiscal year available in FMP data. The divergence threshold
    is read from ``data_quality.cross_validate_tolerance`` in config (default 5%).

    Parameters
    ----------
    financials:
        Dict produced by ``financials.fetch_all_financials``.
    sample_size:
        Maximum number of tickers to include in the sample. Defaults to 20.
        Reads ``data_quality.cross_validate_sample_size`` from config
        only when not explicitly passed.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``field``, ``fmp_value``, ``yfinance_value``,
        ``pct_difference``. One row per (ticker, field) pair where both FMP
        and yfinance values are non-null. Empty DataFrame if no data is available.

    Notes
    -----
    - yfinance ``financials`` returns values in full USD dollars; these are
      divided by 1,000 before comparison with FMP canonical values (USD thousands).
    - ``eps_diluted`` is sourced from ``yf.Ticker.info["trailingEps"]`` (per-share,
      no unit conversion required).
    - Per-ticker yfinance failures are logged at ERROR and skipped.
    """
    cfg = get_config()
    tolerance: float = float(
        cfg.get("data_quality", {}).get("cross_validate_tolerance", 0.05)
    )

    income_df = financials.get("income_statement")
    if income_df is None or income_df.empty:
        logger.warning("cross_validate_sample: no income statement data available.")
        return _empty_cross_validate_df()

    all_tickers = _collect_all_tickers(financials)
    if not all_tickers:
        return _empty_cross_validate_df()

    sample = random.sample(all_tickers, min(sample_size, len(all_tickers)))
    rows: list[dict[str, Any]] = []

    for ticker in sample:
        ticker_income = _filter_to_ticker(income_df, ticker)
        if ticker_income.empty:
            continue
        most_recent = ticker_income.sort_values("fiscal_year", ascending=False).iloc[0]
        yf_data = _safe_fetch_yf_comparison(ticker)
        if not yf_data:
            continue
        for field_name in _CROSS_VALIDATE_FIELDS:
            row = _compare_field(ticker, field_name, most_recent, yf_data, tolerance)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows) if rows else _empty_cross_validate_df()


# ---------------------------------------------------------------------------
# Internal helpers — year and field coverage
# ---------------------------------------------------------------------------

def _count_years_available(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> int:
    """Count distinct fiscal years present across all three statement DataFrames.

    Parameters
    ----------
    income_df, balance_df, cashflow_df:
        Statement DataFrames for a single ticker. May be None or empty.

    Returns
    -------
    int
        Cardinality of the union of ``fiscal_year`` values across all statements.
    """
    all_years: set[Any] = set()
    for df in (income_df, balance_df, cashflow_df):
        if df is not None and not df.empty and "fiscal_year" in df.columns:
            all_years.update(df["fiscal_year"].dropna().unique())
    return len(all_years)


def _find_missing_critical_fields(
    stmt_map: dict[str, pd.DataFrame],
    min_years: int,
) -> list[str]:
    """Return required-drop fields with fewer than ``min_years`` non-null values.

    Parameters
    ----------
    stmt_map:
        Mapping of statement type → filtered single-ticker DataFrame.
    min_years:
        Minimum required non-null observations per field.

    Returns
    -------
    list[str]
        Canonical field names failing the coverage threshold.
    """
    missing: list[str] = []
    for field_name in get_drop_required_fields():
        stmt_type = LINE_ITEM_MAP[field_name].statement
        df = stmt_map.get(stmt_type)
        if df is None or df.empty or field_name not in df.columns:
            non_null_count = 0
        else:
            non_null_count = int(df[field_name].notna().sum())
        if non_null_count < min_years:
            missing.append(field_name)
    return missing


# ---------------------------------------------------------------------------
# Internal helpers — ticker collection and filtering
# ---------------------------------------------------------------------------

def _collect_all_tickers(financials: dict[str, pd.DataFrame]) -> list[str]:
    """Return sorted unique tickers present in any statement DataFrame.

    Parameters
    ----------
    financials:
        Dict of statement DataFrames (may contain None values).

    Returns
    -------
    list[str]
        Alphabetically sorted list of ticker symbols.
    """
    all_tickers: set[str] = set()
    for df in financials.values():
        if df is not None and not df.empty and "ticker" in df.columns:
            all_tickers.update(df["ticker"].dropna().unique())
    return sorted(all_tickers)


def _filter_to_ticker(df: pd.DataFrame | None, ticker: str) -> pd.DataFrame:
    """Return the rows of ``df`` for a single ticker, or an empty DataFrame.

    Parameters
    ----------
    df:
        Statement DataFrame (may be None).
    ticker:
        Ticker symbol to filter on.

    Returns
    -------
    pd.DataFrame
        Filtered copy with reset index, or empty DataFrame.
    """
    if df is None or df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    return df[df["ticker"] == ticker].copy().reset_index(drop=True)


def _count_substitutions_per_ticker(
    substitution_log: list[dict[str, Any]],
) -> dict[str, int]:
    """Count distinct fields using non-ideal substitutes, per ticker.

    Only counts entries where ``confidence`` is not ``"DROP"`` or ``"FLAG"``
    (i.e., actual field substitutions, not missing-field records).

    Parameters
    ----------
    substitution_log:
        Aggregated substitution log from ``financials.normalize_statement``.

    Returns
    -------
    dict[str, int]
        Mapping of ticker → number of distinct substituted fields.
    """
    ticker_fields: dict[str, set[str]] = {}
    for entry in substitution_log:
        ticker = entry.get("ticker", "")
        confidence = entry.get("confidence", "")
        field = entry.get("buffett_field", "")
        if confidence not in ("DROP", "FLAG") and field and ticker:
            ticker_fields.setdefault(ticker, set()).add(field)
    return {t: len(fields) for t, fields in ticker_fields.items()}


# ---------------------------------------------------------------------------
# Internal helpers — report building and persistence
# ---------------------------------------------------------------------------

def _build_quality_report_df(reports: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert list of assess_ticker_quality result dicts to a typed DataFrame.

    ``missing_critical_fields`` is serialized from list to a comma-separated
    string for CSV compatibility. ``drop`` is cast to bool.

    Parameters
    ----------
    reports:
        List of dicts returned by :func:`assess_ticker_quality`.

    Returns
    -------
    pd.DataFrame
        One row per ticker. Empty DataFrame with correct columns if ``reports``
        is empty.
    """
    if not reports:
        return pd.DataFrame(
            columns=[
                "ticker", "years_available", "missing_critical_fields",
                "substitutions_count", "drop", "drop_reason",
            ]
        )
    df = pd.DataFrame(reports)
    df["missing_critical_fields"] = df["missing_critical_fields"].apply(
        lambda lst: ", ".join(sorted(lst)) if isinstance(lst, list) else str(lst or "")
    )
    df["drop"] = df["drop"].astype(bool)
    return df


def _log_quality_summary(quality_report_df: pd.DataFrame) -> None:
    """Log a summary of pass/fail counts and the top drop reasons.

    Parameters
    ----------
    quality_report_df:
        DataFrame produced by :func:`_build_quality_report_df`.
    """
    if quality_report_df.empty or "drop" not in quality_report_df.columns:
        logger.info("run_data_quality_check: no tickers assessed.")
        return

    n_total = len(quality_report_df)
    n_dropped = int(quality_report_df["drop"].sum())
    n_passed = n_total - n_dropped
    logger.info(
        "%d tickers passed quality check, %d dropped.", n_passed, n_dropped
    )

    if n_dropped > 0 and "drop_reason" in quality_report_df.columns:
        dropped_reasons = quality_report_df.loc[
            quality_report_df["drop"], "drop_reason"
        ].dropna()
        # Summarise by leading clause (before first semicolon).
        reason_counts = (
            dropped_reasons.str.split(";").str[0].str.strip().value_counts()
        )
        logger.info("Top drop reasons:")
        for reason, count in reason_counts.head(5).items():
            logger.info("  %d tickers — %s", count, reason)


def _save_quality_report(quality_report_df: pd.DataFrame) -> None:
    """Save the quality report DataFrame to ``data/processed/data_quality_report.csv``.

    Creates the output directory if it does not exist. Errors are logged at
    ERROR level but do not raise — a failed CSV write must not abort the pipeline.

    Parameters
    ----------
    quality_report_df:
        Quality report DataFrame to persist.
    """
    project_root = pathlib.Path(__file__).parent.parent
    report_path = project_root / _REPORT_RELATIVE_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        quality_report_df.to_csv(report_path, index=False)
        logger.info("Data quality report saved to %s (%d rows).", report_path, len(quality_report_df))
    except Exception as exc:
        logger.error("Failed to save quality report to %s: %s", report_path, exc)


# ---------------------------------------------------------------------------
# Internal helpers — cross-validation
# ---------------------------------------------------------------------------

def _safe_fetch_yf_comparison(ticker: str) -> dict[str, float | None]:
    """Fetch yfinance comparison values; return empty dict on any error.

    Parameters
    ----------
    ticker:
        Ticker symbol to fetch from yfinance.

    Returns
    -------
    dict
        Keys: ``net_income``, ``total_revenue``, ``eps_diluted`` (any subset).
    """
    try:
        return _fetch_yf_comparison_values(ticker)
    except Exception as exc:
        logger.error(
            "cross_validate_sample: yfinance fetch failed for %s: %s", ticker, exc
        )
        return {}


def _fetch_yf_comparison_values(ticker: str) -> dict[str, float | None]:
    """Fetch net_income, total_revenue, eps_diluted from yfinance for most recent year.

    Values for ``net_income`` and ``total_revenue`` are sourced from
    ``yf.Ticker.financials`` (full USD dollars ÷ 1,000 → USD thousands, to match
    the canonical unit convention). ``eps_diluted`` is sourced from
    ``yf.Ticker.info["trailingEps"]`` (already per-share, no unit conversion).

    Parameters
    ----------
    ticker:
        Ticker symbol.

    Returns
    -------
    dict
        Keys: fields for which a non-null value was found.
    """
    t = yf.Ticker(ticker)
    result: dict[str, float | None] = {}

    try:
        fin = t.financials
        if fin is not None and not fin.empty:
            col = fin.columns[0]   # Most recent fiscal year (columns sorted desc)
            for field, labels in _YF_INCOME_LABELS.items():
                for label in labels:
                    if label in fin.index:
                        val = fin.loc[label, col]
                        result[field] = float(val) / 1_000.0 if pd.notna(val) else None
                        break
    except Exception as exc:
        logger.warning(
            "_fetch_yf_comparison_values: financials error for %s: %s", ticker, exc
        )

    try:
        info = t.info or {}
        eps = info.get("trailingEps")
        result["eps_diluted"] = float(eps) if eps is not None else None
    except Exception as exc:
        logger.warning(
            "_fetch_yf_comparison_values: info error for %s: %s", ticker, exc
        )

    return result


def _compare_field(
    ticker: str,
    field_name: str,
    fmp_row: pd.Series,
    yf_data: dict[str, float | None],
    tolerance: float,
) -> dict[str, Any] | None:
    """Build a comparison row dict for one (ticker, field) pair.

    Parameters
    ----------
    ticker:
        Ticker symbol for the log/output row.
    field_name:
        Canonical field name being compared.
    fmp_row:
        One-row Series (most recent fiscal year) from the FMP DataFrame.
    yf_data:
        yfinance comparison values dict from :func:`_fetch_yf_comparison_values`.
    tolerance:
        Fractional threshold above which divergence triggers a WARNING log.

    Returns
    -------
    dict or None
        Comparison row dict, or ``None`` if either value is missing/NaN.
    """
    if field_name not in fmp_row.index:
        return None
    fmp_val = fmp_row[field_name]
    yf_val = yf_data.get(field_name)

    if pd.isna(fmp_val) or yf_val is None or pd.isna(yf_val):
        return None

    denom = abs(float(yf_val))
    pct_diff = (
        abs(float(fmp_val) - float(yf_val)) / denom if denom != 0.0 else float("nan")
    )

    if not pd.isna(pct_diff) and pct_diff > tolerance:
        logger.warning(
            "cross_validate_sample [%s] %s diverges >%.0f%%: "
            "FMP=%.4g  yfinance=%.4g  (%.1f%%)",
            ticker, field_name, tolerance * 100,
            fmp_val, yf_val, pct_diff * 100,
        )

    return {
        "ticker": ticker,
        "field": field_name,
        "fmp_value": float(fmp_val),
        "yfinance_value": float(yf_val),
        "pct_difference": pct_diff,
    }


def _empty_cross_validate_df() -> pd.DataFrame:
    """Return an empty cross-validation DataFrame with the canonical columns."""
    return pd.DataFrame(
        columns=["ticker", "field", "fmp_value", "yfinance_value", "pct_difference"]
    )

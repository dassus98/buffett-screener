"""Applies binary pass/fail hard filters (Tier 1) from filter_config.yaml.

Each ticker must pass ALL five hard filters to survive into Tier 2 soft
scoring.  A single failure on any filter permanently excludes the ticker.

Filters
-------
1. Earnings Consistency — ``profitable_years >= min_profitable_years``
2. ROE Floor — ``avg_roe >= min_avg_roe``
3. EPS Growth — ``eps_cagr > min_eps_cagr``  (strictly greater than)
4. Debt Sustainability — ``debt_payoff_years <= max_debt_payoff_years``
5. Data Sufficiency — ``years_available >= min_history_years``

NaN handling: a missing or NaN metric value causes the ticker to **fail**
that filter — a missing metric cannot be assumed to pass.

Data Lineage Contract
---------------------
Upstream producers:
    - ``metrics_engine.run_metrics_engine`` → DuckDB table
      ``buffett_metrics_summary`` (one row per ticker with 10-year aggregates).
      Required columns: ``ticker``, ``profitable_years``, ``avg_roe``,
      ``eps_cagr``, ``debt_payoff_years``, ``years_available``.
    - ``screener.exclusions.apply_exclusions`` → the input DataFrame has
      already had financial-sector and SPAC/shell tickers removed.

Downstream consumers:
    - ``screener.soft_filters.apply_soft_scores`` — receives ``survivors_df``
      (tickers that passed all five hard filters).
    - ``screener.composite_ranker.generate_screener_summary`` — receives
      ``filter_log_df`` for per-filter failure counts.
    - Module 4 (valuation_reports) — reads survivors for report generation.

Config dependencies (all via ``get_threshold``):
    - ``hard_filters.min_profitable_years``   (default 8)
    - ``hard_filters.min_avg_roe``            (default 0.15)
    - ``hard_filters.min_eps_cagr``           (default 0.0)
    - ``hard_filters.max_debt_payoff_years``  (default 5)
    - ``universe.min_history_years``          (default 8)
"""

from __future__ import annotations

import logging

import pandas as pd

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter specifications
#
# Each tuple: (filter_name, column_in_df, comparison_op, config_section, config_key)
# The threshold value is read from get_threshold("{section}.{key}") at runtime.
# ---------------------------------------------------------------------------

_FILTER_SPECS: list[tuple[str, str, str, str, str]] = [
    ("earnings_consistency", "profitable_years",  "gte", "hard_filters", "min_profitable_years"),
    ("roe_floor",            "avg_roe",           "gte", "hard_filters", "min_avg_roe"),
    ("eps_growth",           "eps_cagr",          "gt",  "hard_filters", "min_eps_cagr"),
    ("debt_sustainability",  "debt_payoff_years", "lte", "hard_filters", "max_debt_payoff_years"),
    ("data_sufficiency",     "years_available",   "gte", "universe",     "min_history_years"),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_threshold_value(section: str, key: str) -> float:
    """Read a single numeric threshold via :func:`get_threshold` (fail-fast).

    Parameters
    ----------
    section:
        Top-level config section (e.g. ``"hard_filters"``).
    key:
        Key within the section (e.g. ``"min_avg_roe"``).

    Returns
    -------
    float
        The threshold value.

    Raises
    ------
    ConfigError
        If the config path ``{section}.{key}`` does not exist.
    """
    # --- Fail-fast: raises ConfigError if the key is missing from config,
    #     ensuring no silent fallback to a hardcoded default.
    return float(get_threshold(f"{section}.{key}"))


def _safe_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column *col* as float, or all-NaN if the column is absent.

    Parameters
    ----------
    df:
        Input DataFrame.
    col:
        Column name to extract.

    Returns
    -------
    pd.Series
        Float series aligned with *df*.
    """
    # --- Return the column as float if it exists; otherwise all-NaN so
    #     every ticker fails this filter (NaN never passes a comparison).
    if col in df.columns:
        return df[col].astype(float)
    logger.warning(
        "Column '%s' not found in metrics_summary_df; "
        "all tickers will fail this filter.",
        col,
    )
    return pd.Series(float("nan"), index=df.index, dtype=float)


def _compare(values: pd.Series, threshold: float, op: str) -> pd.Series:
    """Element-wise comparison; NaN always evaluates to ``False`` (fail).

    Parameters
    ----------
    values:
        Metric values.
    threshold:
        Threshold to compare against.
    op:
        ``"gte"`` (>=), ``"gt"`` (>), or ``"lte"`` (<=).

    Returns
    -------
    pd.Series[bool]
    """
    # --- NaN guard: notna() is ANDed into every comparison so NaN → False.
    #     This implements the SCORING.md rule: "a missing metric cannot be
    #     assumed to pass."
    if op == "gte":
        return values.notna() & (values >= threshold)
    if op == "gt":
        return values.notna() & (values > threshold)
    if op == "lte":
        return values.notna() & (values <= threshold)
    raise ValueError(f"Unknown comparison operator: {op!r}")


def _warn_nan_tickers(
    tickers: pd.Series, values: pd.Series, filter_name: str,
) -> None:
    """Log a WARNING for every ticker whose metric value is NaN.

    Parameters
    ----------
    tickers:
        Ticker symbols aligned with *values*.
    values:
        Metric values (NaN entries trigger the warning).
    filter_name:
        Name of the hard filter (for the log message).
    """
    for ticker in tickers.loc[values.isna()]:
        logger.warning("%s failed %s: metric is NaN", ticker, filter_name)


def _evaluate_one_filter(
    df: pd.DataFrame,
    filter_name: str,
    col: str,
    op: str,
    threshold: float,
) -> pd.DataFrame:
    """Evaluate a single hard filter for every ticker.

    Parameters
    ----------
    df:
        Metrics summary DataFrame (one row per ticker).
    filter_name:
        Human-readable filter name for the log.
    col:
        Column in *df* containing the metric.
    op:
        Comparison operator (``"gte"``, ``"gt"``, ``"lte"``).
    threshold:
        Numeric threshold from config.

    Returns
    -------
    pd.DataFrame
        One row per ticker, columns: ``ticker``, ``filter_name``,
        ``filter_value``, ``threshold``, ``pass_fail``.
    """
    # --- Step 1: Extract the metric column (NaN if column is missing).
    values = _safe_column(df, col)

    # --- Step 2: Apply the comparison; NaN always evaluates to False (fail).
    passes = _compare(values, threshold, op)

    # --- Step 3: Log individual NaN-failed tickers at WARNING level.
    _warn_nan_tickers(df["ticker"], values, filter_name)

    # --- Step 4: Build a one-row-per-ticker log for this filter.
    return pd.DataFrame({
        "ticker": df["ticker"].values,
        "filter_name": filter_name,
        "filter_value": values.values,
        "threshold": float(threshold),
        "pass_fail": passes.values,
    })


def _build_filter_log(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all hard filters and return the combined filter log.

    Each filter's threshold is read from ``filter_config.yaml`` via
    :func:`get_threshold` (fail-fast — raises ``ConfigError`` if a
    required key is missing).

    Parameters
    ----------
    df:
        Metrics summary DataFrame.

    Returns
    -------
    pd.DataFrame
        Concatenated filter results for all tickers and all filters.
    """
    parts: list[pd.DataFrame] = []
    for filter_name, col, op, section, key in _FILTER_SPECS:
        # --- Read threshold from config (fail-fast, no hardcoded fallback).
        threshold = _read_threshold_value(section, key)
        parts.append(_evaluate_one_filter(df, filter_name, col, op, threshold))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _select_survivors(
    df: pd.DataFrame,
    filter_log: pd.DataFrame,
) -> pd.DataFrame:
    """Return rows from *df* whose ticker passed ALL filters.

    Parameters
    ----------
    df:
        Original metrics summary DataFrame.
    filter_log:
        Combined filter log from :func:`_build_filter_log`.

    Returns
    -------
    pd.DataFrame
        Subset of *df* containing only surviving tickers (index reset).
    """
    # --- Step 1: If no filter log exists, all tickers survive by default.
    if filter_log.empty:
        return df.copy()

    # --- Step 2: Collect tickers that failed ANY filter (at least one False).
    failed_tickers = set(filter_log.loc[~filter_log["pass_fail"], "ticker"])

    # --- Step 3: Return only tickers NOT in the failed set.
    mask = ~df["ticker"].isin(failed_tickers)
    return df.loc[mask].reset_index(drop=True)


def _log_tier1_summary(
    filter_log: pd.DataFrame,
    total_tickers: int,
) -> None:
    """Log the Tier 1 pass/fail summary with per-filter breakdown.

    Parameters
    ----------
    filter_log:
        Combined filter log.
    total_tickers:
        Total number of tickers evaluated.
    """
    if filter_log.empty:
        logger.info("Tier 1: 0 tickers evaluated.")
        return

    # --- Step 1: Determine how many tickers passed ALL five filters.
    passed_all = filter_log.groupby("ticker")["pass_fail"].all()
    n_pass = int(passed_all.sum())
    n_fail = total_tickers - n_pass

    # --- Step 2: Count per-filter failures for the breakdown string.
    fail_counts: dict[str, int] = {}
    for spec in _FILTER_SPECS:
        name = spec[0]
        subset = filter_log[filter_log["filter_name"] == name]
        fail_counts[name] = int((~subset["pass_fail"]).sum())
    breakdown = ", ".join(f"{k}={v}" for k, v in fail_counts.items())

    # --- Step 3: Emit the summary log line (matches task spec format).
    logger.info(
        "Tier 1: %d passed, %d failed. Breakdown: %s",
        n_pass, n_fail, breakdown,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_hard_filters(
    metrics_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Tier 1 binary pass/fail hard filters to every ticker.

    Reads thresholds from ``config/filter_config.yaml`` and evaluates
    five hard filters.  A ticker must pass ALL five to survive into
    Tier 2 soft scoring.

    Parameters
    ----------
    metrics_summary_df:
        DataFrame with one row per ticker.  Expected columns:
        ``ticker``, ``profitable_years``, ``avg_roe``, ``eps_cagr``,
        ``debt_payoff_years``, ``years_available``.  Missing columns
        cause all tickers to fail the corresponding filter.  ``NaN``
        values cause that ticker to fail (logged at WARNING).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(survivors_df, filter_log_df)``

        *survivors_df*: subset of *metrics_summary_df* containing only
        tickers that passed all five filters.

        *filter_log_df*: one row per ticker per filter.  Columns:
        ``ticker``, ``filter_name``, ``filter_value``, ``threshold``,
        ``pass_fail`` (bool).  Contains entries for ALL tickers
        (both passing and failing).
    """
    # --- Step 1: Guard — empty input produces empty output.
    if metrics_summary_df.empty:
        logger.warning("apply_hard_filters: received empty DataFrame.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Step 2: Evaluate every ticker against all 5 hard filters.
    #     Thresholds are read from filter_config.yaml via get_threshold
    #     inside _build_filter_log (fail-fast if any key is missing).
    filter_log = _build_filter_log(metrics_summary_df)

    # --- Step 3: Select tickers that passed ALL five filters.
    survivors = _select_survivors(metrics_summary_df, filter_log)

    # --- Step 4: Log the Tier 1 summary with per-filter breakdown.
    _log_tier1_summary(filter_log, len(metrics_summary_df))

    return survivors, filter_log

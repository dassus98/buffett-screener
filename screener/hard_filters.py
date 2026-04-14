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

All thresholds are read from ``config/filter_config.yaml`` at runtime via
:func:`~screener.filter_config_loader.get_config`.

NaN handling: a missing or NaN metric value causes the ticker to **fail**
that filter — a missing metric cannot be assumed to pass.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter specifications
#
# Each tuple: (filter_name, column_in_df, comparison_op, config_section, config_key)
# The threshold value is read from cfg[section][key] at runtime.
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


def _read_threshold(cfg: dict[str, Any], section: str, key: str) -> float:
    """Read a single numeric threshold from the config dict.

    Parameters
    ----------
    cfg:
        Full config dict from :func:`get_config`.
    section:
        Top-level config section (e.g. ``"hard_filters"``).
    key:
        Key within the section (e.g. ``"min_avg_roe"``).

    Returns
    -------
    float
        The threshold value, or ``0.0`` if the key is missing.
    """
    return float(cfg.get(section, {}).get(key, 0))


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
    values = _safe_column(df, col)
    passes = _compare(values, threshold, op)
    _warn_nan_tickers(df["ticker"], values, filter_name)
    return pd.DataFrame({
        "ticker": df["ticker"].values,
        "filter_name": filter_name,
        "filter_value": values.values,
        "threshold": float(threshold),
        "pass_fail": passes.values,
    })


def _build_filter_log(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate all hard filters and return the combined filter log.

    Parameters
    ----------
    df:
        Metrics summary DataFrame.
    cfg:
        Full config dict.

    Returns
    -------
    pd.DataFrame
        Concatenated filter results for all tickers and all filters.
    """
    parts: list[pd.DataFrame] = []
    for filter_name, col, op, section, key in _FILTER_SPECS:
        threshold = _read_threshold(cfg, section, key)
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
    if filter_log.empty:
        return df.copy()
    failed_tickers = set(filter_log.loc[~filter_log["pass_fail"], "ticker"])
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
    passed_all = filter_log.groupby("ticker")["pass_fail"].all()
    n_pass = int(passed_all.sum())
    n_fail = total_tickers - n_pass
    fail_counts: dict[str, int] = {}
    for spec in _FILTER_SPECS:
        name = spec[0]
        subset = filter_log[filter_log["filter_name"] == name]
        fail_counts[name] = int((~subset["pass_fail"]).sum())
    breakdown = ", ".join(f"{k}={v}" for k, v in fail_counts.items())
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
    if metrics_summary_df.empty:
        logger.warning("apply_hard_filters: received empty DataFrame.")
        return pd.DataFrame(), pd.DataFrame()

    cfg = get_config()
    filter_log = _build_filter_log(metrics_summary_df, cfg)
    survivors = _select_survivors(metrics_summary_df, filter_log)
    _log_tier1_summary(filter_log, len(metrics_summary_df))
    return survivors, filter_log

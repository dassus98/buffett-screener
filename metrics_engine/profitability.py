"""Computes gross, operating, net, and EBITDA margins along with their trailing averages, standard deviations, and trend direction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)

# Minimum absolute slope magnitude for classifying a trend as non-stable.
# Applied to net-margin values (range 0–1); 0.005 ≈ 0.5 percentage-point
# shift per year — separates directionless noise from a sustained move.
_TREND_SLOPE_THRESHOLD: float = 0.005


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_avg_equity(
    equity_t: pd.Series,
    equity_prior: pd.Series,
) -> pd.Series:
    """Average current and prior-year equity; fall back to current when prior absent.

    Parameters
    ----------
    equity_t:
        Current-year shareholders' equity, aligned by DataFrame index.
    equity_prior:
        Prior-year shareholders' equity shifted forward one fiscal year.
        Contains NaN for the first available year in the series.

    Returns
    -------
    pd.Series
        Average equity per row.  NaN where equity_t itself is NaN.
    """
    has_prior = equity_prior.notna()
    avg = pd.Series(np.nan, index=equity_t.index, dtype=float)
    avg.loc[~has_prior] = equity_t.loc[~has_prior].values
    avg.loc[has_prior] = (
        (equity_t.loc[has_prior] + equity_prior.loc[has_prior]) / 2.0
    ).values
    return avg


def _compute_trend(values: np.ndarray) -> str:
    """Classify a time-series as improving, stable, or deteriorating.

    Uses an ordinary-least-squares slope. Returns ``"stable"`` when fewer
    than two valid observations are available.

    Parameters
    ----------
    values:
        1-D float array in chronological order (NaN-free).

    Returns
    -------
    str
        One of ``"improving"``, ``"deteriorating"``, or ``"stable"``.
    """
    if len(values) < 2:
        return "stable"
    slope = float(np.polyfit(np.arange(len(values)), values.astype(float), 1)[0])
    if slope > _TREND_SLOPE_THRESHOLD:
        return "improving"
    if slope < -_TREND_SLOPE_THRESHOLD:
        return "deteriorating"
    return "stable"


# ---------------------------------------------------------------------------
# Public formula functions
# ---------------------------------------------------------------------------

def compute_roe(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute Return on Equity (F3) for each fiscal year.

    ``ROE = net_income / avg(equity_{t-1}, equity_t)``.  First year uses
    current equity alone.  Years where average equity ≤ 0 yield NaN and
    ``negative_equity = True`` (logged at WARNING).

    Parameters
    ----------
    income_df:
        Columns: ``fiscal_year``, ``net_income``.  Single ticker.
    balance_df:
        Columns: ``fiscal_year``, ``shareholders_equity``.  Single ticker.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        *annual_df* columns: ``fiscal_year``, ``roe``, ``negative_equity``.
        *summary* keys: ``avg_roe``, ``roe_stdev``, ``years_above_threshold``.
    """
    cfg = get_config()
    roe_threshold: float = float(cfg.get("hard_filters", {}).get("min_avg_roe", 0.15))

    inc = income_df[["fiscal_year", "net_income"]].sort_values("fiscal_year").reset_index(drop=True)
    bal = balance_df[["fiscal_year", "shareholders_equity"]].sort_values("fiscal_year").reset_index(drop=True)
    bal_prior = bal.rename(columns={"shareholders_equity": "equity_prior"}).copy()
    bal_prior["fiscal_year"] = bal_prior["fiscal_year"] + 1

    df = inc.merge(bal.rename(columns={"shareholders_equity": "equity_t"}), on="fiscal_year", how="left")
    df = df.merge(bal_prior, on="fiscal_year", how="left")

    avg_eq = _compute_avg_equity(df["equity_t"], df["equity_prior"])
    valid_mask = avg_eq.notna() & (avg_eq > 0)
    roe_series = (df["net_income"].astype(float) / avg_eq.where(valid_mask)).where(valid_mask, other=np.nan)

    for yr in df.loc[~valid_mask, "fiscal_year"].tolist():
        logger.warning("ROE undefined for fiscal_year=%s: avg_equity ≤ 0. Setting ROE=NaN.", yr)

    annual_df = pd.DataFrame({
        "fiscal_year": df["fiscal_year"].values,
        "roe": roe_series.values,
        "negative_equity": (~valid_mask).values,
    })
    valid_roe = roe_series.dropna()
    avg_roe = float(valid_roe.mean()) if not valid_roe.empty else float("nan")
    roe_stdev = float(valid_roe.std()) if len(valid_roe) > 1 else float("nan")
    years_above = int((valid_roe >= roe_threshold).sum())
    return annual_df, {"avg_roe": avg_roe, "roe_stdev": roe_stdev, "years_above_threshold": years_above}


def compute_gross_margin(
    income_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute Gross Margin (F7) for each fiscal year.

    ``Gross Margin = gross_profit / total_revenue``

    Years with zero or missing revenue produce NaN (logged at WARNING).

    Parameters
    ----------
    income_df:
        Columns: ``fiscal_year``, ``gross_profit``, ``total_revenue``.
        Single ticker, sorted by ``fiscal_year`` ascending.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        *annual_df* columns: ``fiscal_year``, ``gross_margin``.
        *summary* keys: ``avg_gross_margin``, ``min_gross_margin``.
    """
    df = income_df[["fiscal_year", "gross_profit", "total_revenue"]].sort_values("fiscal_year").reset_index(drop=True)

    invalid = df["total_revenue"].isna() | (df["total_revenue"] == 0)
    for yr in df.loc[invalid, "fiscal_year"].tolist():
        logger.warning("Gross margin undefined for fiscal_year=%s: total_revenue=0. Setting NaN.", yr)

    gross_margin = (
        df["gross_profit"].astype(float) / df["total_revenue"].where(~invalid).astype(float)
    ).where(~invalid, other=np.nan)

    annual_df = pd.DataFrame({"fiscal_year": df["fiscal_year"].values, "gross_margin": gross_margin.values})
    valid = gross_margin.dropna()
    avg_gm = float(valid.mean()) if not valid.empty else float("nan")
    min_gm = float(valid.min()) if not valid.empty else float("nan")
    return annual_df, {"avg_gross_margin": avg_gm, "min_gross_margin": min_gm}


def compute_sga_ratio(
    income_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute SGA-to-Gross-Profit Ratio (F8) for each fiscal year.

    ``SGA Ratio = sga / gross_profit``

    Years with zero or negative gross profit produce NaN (logged at WARNING).

    Parameters
    ----------
    income_df:
        Columns: ``fiscal_year``, ``sga``, ``gross_profit``.
        Single ticker, sorted by ``fiscal_year`` ascending.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        *annual_df* columns: ``fiscal_year``, ``sga_ratio``.
        *summary* keys: ``avg_sga_ratio``.
    """
    df = income_df[["fiscal_year", "sga", "gross_profit"]].sort_values("fiscal_year").reset_index(drop=True)

    invalid = df["gross_profit"].isna() | (df["gross_profit"] <= 0)
    for yr in df.loc[invalid, "fiscal_year"].tolist():
        logger.warning("SGA ratio undefined for fiscal_year=%s: gross_profit ≤ 0. Setting NaN.", yr)

    sga_ratio = (
        df["sga"].astype(float) / df["gross_profit"].where(~invalid).astype(float)
    ).where(~invalid, other=np.nan)

    annual_df = pd.DataFrame({"fiscal_year": df["fiscal_year"].values, "sga_ratio": sga_ratio.values})
    valid = sga_ratio.dropna()
    avg_sga = float(valid.mean()) if not valid.empty else float("nan")
    return annual_df, {"avg_sga_ratio": avg_sga}


def compute_net_margin(
    income_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute Net Margin Consistency (F10) for each fiscal year.

    ``Net Margin = net_income / total_revenue``

    Also reports ``profitable_years`` (years where net_income > 0) and
    a linear-regression trend direction.

    Parameters
    ----------
    income_df:
        Columns: ``fiscal_year``, ``net_income``, ``total_revenue``.
        Single ticker, sorted by ``fiscal_year`` ascending.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        *annual_df* columns: ``fiscal_year``, ``net_margin``.
        *summary* keys: ``avg_net_margin``, ``profitable_years``, ``trend``.

    Notes
    -----
    ``profitable_years`` counts rows where ``net_income > 0`` regardless of
    whether revenue is zero.  Zero-revenue years are excluded from the margin
    average but still assessed for profitability.
    """
    df = income_df[["fiscal_year", "net_income", "total_revenue"]].sort_values("fiscal_year").reset_index(drop=True)

    invalid = df["total_revenue"].isna() | (df["total_revenue"] == 0)
    for yr in df.loc[invalid, "fiscal_year"].tolist():
        logger.warning("Net margin undefined for fiscal_year=%s: total_revenue=0. Setting NaN.", yr)

    net_margin = (
        df["net_income"].astype(float) / df["total_revenue"].where(~invalid).astype(float)
    ).where(~invalid, other=np.nan)

    annual_df = pd.DataFrame({"fiscal_year": df["fiscal_year"].values, "net_margin": net_margin.values})
    profitable_years = int((df["net_income"] > 0).sum())
    valid = net_margin.dropna()
    avg_nm = float(valid.mean()) if not valid.empty else float("nan")
    trend = _compute_trend(valid.values)
    return annual_df, {"avg_net_margin": avg_nm, "profitable_years": profitable_years, "trend": trend}

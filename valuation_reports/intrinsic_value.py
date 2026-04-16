"""Computes a full valuation for a single ticker by reading DuckDB data
and calling metrics_engine/valuation.py formula functions (F14/F15/F16).

Data Lineage Contract
---------------------
Upstream producers:
    - ``data_acquisition.store.read_table("universe")``
      → provides ``exchange`` for determining the correct bond yield key
        (TSX → GoC 10yr; NYSE/NASDAQ → US Treasury 10yr).
    - ``data_acquisition.store.read_table("income_statement")``
      → provides ``eps_diluted`` per fiscal year (10-year series for F11 CAGR).
    - ``data_acquisition.store.read_table("market_data")``
      → provides ``current_price_usd`` and ``pe_ratio_trailing``.
    - ``data_acquisition.store.read_table("macro_data")``
      → provides risk-free rate (``us_treasury_10yr`` for US exchanges,
        ``goc_bond_10yr`` for TSX).
    - ``metrics_engine.growth.compute_eps_cagr``
      → computes EPS CAGR from the diluted EPS series.
    - ``metrics_engine.valuation.compute_intrinsic_value`` (F14)
    - ``metrics_engine.valuation.compute_margin_of_safety`` (F15)
    - ``metrics_engine.valuation.compute_earnings_yield`` (F16)

Downstream consumers:
    - ``valuation_reports.report_generator``
      → reads the valuation dict to populate the Deep-Dive Report Template.
      → accesses ``scenarios.{bear,base,bull}.discount_rate`` for the
        Valuation table's Discount Rate column.
    - ``valuation_reports.margin_of_safety.compute_sensitivity_table``
      → reads the base valuation dict for sensitivity analysis.

Config dependencies:
    - None directly — all config access is handled by the upstream formula
      functions via ``get_threshold()``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from data_acquisition.store import read_table
from metrics_engine.growth import compute_eps_cagr
from metrics_engine.valuation import (
    compute_earnings_yield,
    compute_intrinsic_value,
    compute_margin_of_safety,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _nan_valuation(ticker: str, reason: str) -> dict[str, Any]:
    """Return a complete valuation dict with all float fields set to NaN.

    Used for early exits when required data is unavailable.
    """
    logger.warning(
        "compute_full_valuation(%s): %s — returning NaN valuation.",
        ticker,
        reason,
    )
    nan = float("nan")
    return {
        "ticker": ticker,
        "current_price": nan,
        "scenarios": {
            "bear": {
                "growth": nan, "pe": nan, "discount_rate": nan,
                "projected_price": nan, "present_value": nan,
                "annual_return": nan, "probability": nan,
            },
            "base": {
                "growth": nan, "pe": nan, "discount_rate": nan,
                "projected_price": nan, "present_value": nan,
                "annual_return": nan, "probability": nan,
            },
            "bull": {
                "growth": nan, "pe": nan, "discount_rate": nan,
                "projected_price": nan, "present_value": nan,
                "annual_return": nan, "probability": nan,
            },
        },
        "weighted_iv": nan,
        "margin_of_safety": nan,
        "earnings_yield": nan,
        "bond_yield": nan,
        "spread": nan,
        "meets_hurdle": False,
        "is_undervalued": False,
    }


def _read_current_price(ticker: str) -> tuple[float, float]:
    """Read current price and trailing P/E from market_data.

    Returns
    -------
    tuple[float, float]
        ``(current_price_usd, pe_ratio_trailing)``.
        Both are ``NaN`` if the ticker is absent from market_data.
    """
    mkt = read_table("market_data", where=f"ticker = '{ticker}'")
    if mkt.empty:
        logger.warning("No market_data for %s.", ticker)
        return float("nan"), float("nan")
    row = mkt.iloc[0]
    return float(row["current_price_usd"]), float(row["pe_ratio_trailing"])


def _read_exchange(ticker: str) -> str:
    """Read the ticker's exchange from the universe table.

    Used to select the appropriate bond yield for valuation:
    TSX-listed tickers use GoC 10yr; all other exchanges use US Treasury 10yr.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.

    Returns
    -------
    str
        Exchange code (e.g. ``"TSX"``, ``"NYSE"``).  Empty string if the
        ticker is not found in the universe table (falls back to US yield).
    """
    uni = read_table("universe", where=f"ticker = '{ticker}'")
    if uni.empty:
        logger.warning("No universe row for %s; defaulting to US bond yield.", ticker)
        return ""
    return str(uni.iloc[0].get("exchange", ""))


def _read_risk_free_rate(exchange: str) -> float:
    """Read the appropriate 10-year bond yield from macro_data.

    TSX-listed securities use the Government of Canada 10-year bond yield
    (``goc_bond_10yr``) per the Canada-US tax treaty context in REPORT_SPEC.md.
    All other exchanges use the US Treasury 10-year yield (``us_treasury_10yr``).

    This mirrors the exchange-aware selection in
    ``metrics_engine.__init__._extract_macro_rfr``, ensuring that the
    valuation module and the metrics engine use the same risk-free rate for
    any given ticker.

    Parameters
    ----------
    exchange:
        Exchange code from the universe table.

    Returns
    -------
    float
        Risk-free rate (decimal, e.g. 0.04 for 4 %).
        ``NaN`` if the key is not present in macro_data.
    """
    # --- Select bond yield key based on exchange ---
    key = "goc_bond_10yr" if exchange == "TSX" else "us_treasury_10yr"
    macro = read_table("macro_data", where=f"key = '{key}'")
    if macro.empty:
        logger.warning(
            "%s not found in macro_data for exchange=%s.",
            key, exchange,
        )
        return float("nan")
    return float(macro.iloc[0]["value"])


def _read_eps_series(ticker: str) -> pd.Series:
    """Read the historical eps_diluted series from income_statement.

    Returns
    -------
    pd.Series
        Index: fiscal_year (int), values: eps_diluted (float).
        Empty Series if no data is found.
    """
    inc = read_table(
        "income_statement",
        where=f"ticker = '{ticker}'",
    )
    if inc.empty or "eps_diluted" not in inc.columns:
        logger.warning("No income_statement data for %s.", ticker)
        return pd.Series(dtype=float)

    # --- Build a fiscal-year-indexed Series (required by compute_eps_cagr) ---
    df = inc[["fiscal_year", "eps_diluted"]].dropna(subset=["eps_diluted"])
    df = df.sort_values("fiscal_year")
    return pd.Series(
        df["eps_diluted"].values,
        index=df["fiscal_year"].astype(int).values,
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_full_valuation(ticker: str) -> dict[str, Any]:
    """Read DuckDB data and compute F14/F15/F16 for *ticker*.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (must exist in DuckDB tables).

    Returns
    -------
    dict
        Structured valuation result with keys:

        * ``ticker`` — input ticker symbol
        * ``current_price`` — current market price (USD)
        * ``scenarios`` — ``{"bear": {...}, "base": {...}, "bull": {...}}``
          each containing ``growth``, ``pe``, ``present_value``, etc.
        * ``weighted_iv`` — probability-weighted intrinsic value
        * ``margin_of_safety`` — MoS as decimal (0.25 = 25 %)
        * ``earnings_yield`` — EPS / current price
        * ``bond_yield`` — 10-year risk-free rate
        * ``spread`` — earnings yield minus bond yield
        * ``meets_hurdle`` — True if projected return ≥ hurdle rate
        * ``is_undervalued`` — True if MoS > 0
    """
    # --- Step 1: Read exchange to determine which bond yield to use ---
    #     TSX tickers use GoC 10yr; NYSE/NASDAQ tickers use US Treasury 10yr.
    exchange = _read_exchange(ticker)

    # --- Step 2: Read current market price and trailing P/E ---
    current_price, pe_trailing = _read_current_price(ticker)
    if math.isnan(current_price):
        return _nan_valuation(ticker, "current_price unavailable")

    # --- Step 3: Read exchange-appropriate risk-free rate ---
    risk_free_rate = _read_risk_free_rate(exchange)
    if math.isnan(risk_free_rate):
        return _nan_valuation(ticker, "risk_free_rate unavailable")

    # --- Step 4: Read historical EPS series ---
    eps_series = _read_eps_series(ticker)
    if eps_series.empty:
        return _nan_valuation(ticker, "eps_diluted series is empty")

    # --- Step 5: Compute EPS CAGR (F11) from the historical EPS series ---
    cagr_result = compute_eps_cagr(eps_series)
    eps_cagr = cagr_result.get("eps_cagr", float("nan"))
    current_eps = float(eps_series.iloc[-1])

    # --- Step 6: Build a historical P/E series ---
    #     In production, a richer historical P/E series would come from
    #     annual price/eps data.  Here we use the single trailing P/E as
    #     a one-element series — _resolve_pe_estimates handles this gracefully.
    pe_series = pd.Series(dtype=float)
    if not math.isnan(pe_trailing):
        pe_series = pd.Series([pe_trailing])

    # --- Step 7: F14 — Three-Scenario Intrinsic Value ---
    iv_result = compute_intrinsic_value(
        current_eps, eps_cagr, pe_series, current_price, risk_free_rate,
    )

    # --- Step 8: F15 — Margin of Safety ---
    mos_result = compute_margin_of_safety(
        iv_result["weighted_iv"], current_price,
    )

    # --- Step 9: F16 — Earnings Yield vs Bond Yield ---
    ey_result = compute_earnings_yield(
        current_eps, current_price, risk_free_rate,
    )

    # --- Step 10: Package into the report-ready structure ---
    return {
        "ticker": ticker,
        "current_price": current_price,
        "scenarios": {
            label: {
                "growth": iv_result[label]["growth"],
                "pe": iv_result[label]["pe"],
                "discount_rate": iv_result[label]["discount_rate"],
                "projected_price": iv_result[label]["projected_price"],
                "present_value": iv_result[label]["present_value"],
                "annual_return": iv_result[label]["annual_return"],
                "probability": iv_result[label]["probability"],
            }
            for label in ("bear", "base", "bull")
        },
        "weighted_iv": iv_result["weighted_iv"],
        "margin_of_safety": mos_result["margin_of_safety"],
        "earnings_yield": ey_result["earnings_yield"],
        "bond_yield": ey_result["bond_yield"],
        "spread": ey_result["spread"],
        "meets_hurdle": iv_result["meets_hurdle"],
        "is_undervalued": mos_result["is_undervalued"],
    }

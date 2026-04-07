"""Fetches current price, market cap, enterprise value, beta, and volume via yfinance.

Provides a current market snapshot and annual price history for each ticker in the
investment universe. yfinance is the primary source (no API key required).

Authoritative spec: docs/DATA_SOURCES.md §1 (current market data), §6 (yfinance labels).
Rate limiting: reads ``data_sources.yfinance.rate_limit_per_sec`` from
config/filter_config.yaml (default 2 req/sec = 120 req/min).

Key exports
-----------
fetch_market_data(tickers) -> pd.DataFrame
    Current snapshot: price, market cap, shares, 52-week range, volume, PE, dividend.
fetch_historical_pe(ticker, years) -> pd.DataFrame
    Annual year-end closing prices for historical P/E computation (F14).
MARKET_DATA_COLUMNS : tuple[str, ...]
    Canonical output columns for fetch_market_data.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

from data_acquisition.api_config import RateLimiter
from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical columns returned by :func:`fetch_market_data`.
MARKET_DATA_COLUMNS: tuple[str, ...] = (
    "ticker",
    "current_price_usd",
    "market_cap_usd",
    "enterprise_value_usd",
    "shares_outstanding",
    "high_52w",
    "low_52w",
    "avg_volume_3m",
    "pe_ratio_trailing",
    "dividend_yield",
    "as_of_date",
)

#: Numeric columns within MARKET_DATA_COLUMNS (excludes string fields).
_NUMERIC_MARKET_COLS: tuple[str, ...] = tuple(
    c for c in MARKET_DATA_COLUMNS if c not in ("ticker", "as_of_date")
)


def _yfinance_rate_limit() -> int:
    """Return yfinance req/min from config (rate_limit_per_sec × 60, default 120)."""
    cfg = get_config()
    per_sec = int(
        cfg.get("data_sources", {}).get("yfinance", {}).get("rate_limit_per_sec", 2)
    )
    return per_sec * 60


#: Module-level RateLimiter for yfinance. Initialized at import time from config.
_yf_limiter: RateLimiter = RateLimiter(max_requests_per_minute=_yfinance_rate_limit())


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_market_data(tickers: list[str]) -> pd.DataFrame:
    """Fetch current market snapshot for a list of tickers via yfinance.

    Iterates over each ticker and calls yfinance for current price, market cap,
    52-week range, shares outstanding, 3-month average volume, trailing P/E ratio,
    and dividend yield. Failures produce NaN rows rather than raising exceptions.

    Parameters
    ----------
    tickers:
        Ticker symbols. TSX tickers must include the ``.TO`` suffix (e.g.
        ``"SHOP.TO"``), which yfinance requires for Canadian equities.

    Returns
    -------
    pd.DataFrame
        One row per ticker. Columns match ``MARKET_DATA_COLUMNS``. All numeric
        columns are ``float64``; ``as_of_date`` is an ISO 8601 date string.

    Notes
    -----
    - Per-ticker failures are isolated: the ticker gets a NaN row and processing
      continues. Errors are logged at ERROR level.
    - Rate limited to the configured yfinance req/sec via ``_yf_limiter``.
    - TSX tickers are stored with ``.TO`` suffix and passed directly to yfinance.
    """
    today = date.today().isoformat()
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            row = _fetch_single_market_row(ticker, today)
        except Exception as exc:
            logger.error("fetch_market_data: failed for %s: %s", ticker, exc)
            row = _empty_market_row(ticker, today)
        rows.append(row)
    return _assemble_market_df(rows)


def fetch_historical_pe(ticker: str, years: int = 10) -> pd.DataFrame:
    """Fetch annual year-end closing prices for historical P/E computation.

    Downloads daily OHLCV history via yfinance and resamples to calendar
    year-end closing prices. Combined with annual EPS from ``financials.py``,
    this supports computation of trailing P/E history (formula F14).

    Parameters
    ----------
    ticker:
        Ticker symbol. TSX tickers must include ``.TO`` suffix.
    years:
        Number of calendar years of history to fetch. Default is 10.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker`` (str), ``calendar_year`` (int), ``close_price``
        (float). Rows are sorted ascending by ``calendar_year``.
        Returns an empty DataFrame on failure.

    Notes
    -----
    - On yfinance failure, returns an empty DataFrame. Caller should treat
      missing years as NaN when computing P/E history.
    - Annual price is the last trading-day close of each calendar year.
    """
    _yf_limiter.wait_if_needed()
    try:
        return _download_annual_prices(ticker, years)
    except Exception as exc:
        logger.error("fetch_historical_pe: failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["ticker", "calendar_year", "close_price"])


# ---------------------------------------------------------------------------
# Internal helpers — single-ticker fetch
# ---------------------------------------------------------------------------

def _fetch_single_market_row(ticker: str, as_of_date: str) -> dict[str, Any]:
    """Fetch all market data fields for one ticker and return as a dict."""
    _yf_limiter.wait_if_needed()
    t = yf.Ticker(ticker)
    fast = _extract_fast_info(t, ticker)
    extended = _extract_extended_info(t, ticker)
    return {"ticker": ticker, "as_of_date": as_of_date, **fast, **extended}


def _extract_fast_info(t: yf.Ticker, ticker: str) -> dict[str, Any]:
    """Extract price and volume fields from yfinance fast_info.

    Uses ``fast_info`` (low-latency endpoint) for: current price, market cap,
    shares outstanding, 52-week range, and 3-month average volume.

    Parameters
    ----------
    t:
        yfinance Ticker object.
    ticker:
        Symbol string, used for log messages.

    Returns
    -------
    dict
        Keys: current_price_usd, market_cap_usd, shares_outstanding,
        high_52w, low_52w, avg_volume_3m. Values are float (NaN on error).
    """
    fi = t.fast_info

    def _get(attr: str) -> float:
        try:
            val = getattr(fi, attr, None)
            return float(val) if val is not None else float("nan")
        except Exception:
            return float("nan")

    return {
        "current_price_usd": _get("last_price"),
        "market_cap_usd": _get("market_cap"),
        "shares_outstanding": _get("shares"),
        "high_52w": _get("year_high"),
        "low_52w": _get("year_low"),
        "avg_volume_3m": _get("three_month_average_volume"),
    }


def _extract_extended_info(t: yf.Ticker, ticker: str) -> dict[str, Any]:
    """Extract enterprise value, trailing PE, and dividend yield from yfinance info.

    Uses the slower ``info`` dict endpoint for fields not available in ``fast_info``.

    Parameters
    ----------
    t:
        yfinance Ticker object.
    ticker:
        Symbol string, used for log messages.

    Returns
    -------
    dict
        Keys: enterprise_value_usd, pe_ratio_trailing, dividend_yield.
        Values are float (NaN when the key is absent or non-numeric).
    """
    try:
        info = t.info or {}
    except Exception as exc:
        logger.warning(
            "_extract_extended_info: info fetch failed for %s: %s", ticker, exc
        )
        info = {}

    def _safe(key: str) -> float:
        v = info.get(key)
        try:
            return float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    return {
        "enterprise_value_usd": _safe("enterpriseValue"),
        "pe_ratio_trailing": _safe("trailingPE"),
        "dividend_yield": _safe("dividendYield"),
    }


def _download_annual_prices(ticker: str, years: int) -> pd.DataFrame:
    """Download and resample price history to annual year-end closes.

    Parameters
    ----------
    ticker:
        Ticker symbol passed directly to yfinance.
    years:
        Number of years of history to request.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, calendar_year, close_price. Sorted ascending.
    """
    period = f"{years}y"
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if hist.empty:
        logger.warning("fetch_historical_pe: empty history for %s.", ticker)
        return pd.DataFrame(columns=["ticker", "calendar_year", "close_price"])
    try:
        annual = hist["Close"].resample("YE").last().dropna()
    except ValueError:
        # Older pandas versions use "A" instead of "YE".
        annual = hist["Close"].resample("A").last().dropna()
    df = pd.DataFrame({
        "ticker": ticker,
        "calendar_year": annual.index.year.astype(int),
        "close_price": annual.values.astype(float),
    })
    return df.sort_values("calendar_year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers — assembly and empty rows
# ---------------------------------------------------------------------------

def _empty_market_row(ticker: str, as_of_date: str) -> dict[str, Any]:
    """Return a market data dict with NaN for all numeric fields.

    Used when a per-ticker fetch fails, to preserve error isolation.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    as_of_date:
        ISO 8601 date string to populate the as_of_date field.

    Returns
    -------
    dict
        All numeric fields set to ``float("nan")``.
    """
    row: dict[str, Any] = {"ticker": ticker, "as_of_date": as_of_date}
    for col in _NUMERIC_MARKET_COLS:
        row[col] = float("nan")
    return row


def _assemble_market_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of market row dicts to a typed DataFrame.

    Parameters
    ----------
    rows:
        List of dicts produced by ``_fetch_single_market_row`` or
        ``_empty_market_row``. May be empty.

    Returns
    -------
    pd.DataFrame
        Columns in ``MARKET_DATA_COLUMNS`` order. Numeric columns are float64.
    """
    df = pd.DataFrame(rows, columns=list(MARKET_DATA_COLUMNS))
    for col in _NUMERIC_MARKET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df

"""
data_acquisition.macro_data
============================
Fetches macro-economic indicators from the Federal Reserve Economic Data (FRED)
API, used as inputs to valuation models (discount rate, risk-free rate) and
to contextualise screening results (rate environment, inflation).

FRED series used:
    DGS10   — 10-Year Treasury Constant Maturity Rate (daily)
    DGS2    — 2-Year Treasury Constant Maturity Rate (daily)
    CPIAUCSL — Consumer Price Index for All Urban Consumers (monthly)
    FEDFUNDS — Effective Federal Funds Rate (monthly)
    A191RL1Q225SBEA — Real GDP Growth Rate (quarterly, annualised)

Requires:
    FRED_API_KEY set in .env (via api_config.FredConfig)
"""

from __future__ import annotations

import dataclasses
import logging
import math
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from data_acquisition.api_config import FredConfig
from data_acquisition.schema import MacroSnapshot

logger = logging.getLogger(__name__)


def fetch_macro_snapshot(
    as_of_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
    use_cache: bool = True,
    store=None,
) -> MacroSnapshot:
    """
    Fetch a complete MacroSnapshot as of the given date.

    Args:
        as_of_date: Date for which to retrieve macro data. Defaults to today.
        config:     FredConfig instance. If None, loaded from environment.
        use_cache:  If True, read from DuckDB cache if fresh data exists.
        store:      DuckDBStore instance for cache read/write.

    Returns:
        MacroSnapshot dataclass instance.
    """
    target_date = as_of_date or date.today()

    # ---- Cache check -------------------------------------------------------
    if use_cache and store is not None:
        cached = store.load_macro_snapshot(as_of_date=target_date)
        if cached is not None:
            cached_date = pd.Timestamp(cached["as_of_date"]).date()
            age = (target_date - cached_date).days
            if age <= 1:
                logger.debug("Cache hit for macro_snapshot (as_of=%s).", cached_date)
                return MacroSnapshot(
                    as_of_date=cached_date,
                    treasury_10y_yield=float(cached.get("treasury_10y_yield", float("nan"))),
                    treasury_2y_yield=float(cached.get("treasury_2y_yield", float("nan"))),
                    cpi_yoy=float(cached.get("cpi_yoy", float("nan"))),
                    fed_funds_rate=float(cached.get("fed_funds_rate", float("nan"))),
                    real_gdp_growth_yoy=float(cached.get("real_gdp_growth_yoy", float("nan"))),
                    sp500_pe_ratio=float(cached.get("sp500_pe_ratio", float("nan"))),
                )

    if config is None:
        config = FredConfig.from_env()

    logger.info("Fetching macro snapshot as_of=%s from FRED.", target_date)

    # Fetch each series; wrap individually so one failure doesn't abort all
    treasury_10y = _safe_fetch(fetch_treasury_yield, "10Y", target_date, config)
    treasury_2y = _safe_fetch(fetch_treasury_yield, "2Y", target_date, config)
    cpi_yoy = _safe_fetch(fetch_cpi_yoy, target_date, config)
    fed_funds = _safe_fetch(fetch_fed_funds_rate, target_date, config)
    gdp_growth = _safe_fetch(fetch_real_gdp_growth, target_date, config)
    sp500_pe = _safe_fetch_no_config(fetch_sp500_pe, target_date)

    snapshot = MacroSnapshot(
        as_of_date=target_date,
        treasury_10y_yield=treasury_10y,
        treasury_2y_yield=treasury_2y,
        cpi_yoy=cpi_yoy,
        fed_funds_rate=fed_funds,
        real_gdp_growth_yoy=gdp_growth,
        sp500_pe_ratio=sp500_pe,
    )

    # ---- Write to cache ----------------------------------------------------
    if use_cache and store is not None:
        try:
            store.save_macro_snapshot(
                pd.DataFrame([dataclasses.asdict(snapshot)])
            )
        except Exception as exc:
            logger.warning("Failed to write macro_snapshot to cache: %s.", exc)

    return snapshot


def _safe_fetch(func, *args):
    """Call a FRED fetch function, returning NaN on any failure."""
    try:
        return func(*args)
    except Exception as exc:
        logger.warning("Macro fetch %s failed: %s. Storing NaN.", func.__name__, exc)
        return float("nan")


def _safe_fetch_no_config(func, *args):
    """Call a fetch function that takes no config, returning NaN on failure."""
    try:
        return func(*args)
    except Exception as exc:
        logger.warning("Macro fetch %s failed: %s. Storing NaN.", func.__name__, exc)
        return float("nan")


def fetch_fred_series(
    series_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
) -> pd.Series:
    """
    Fetch a single FRED time series as a pandas Series indexed by date.

    Args:
        series_id:  FRED series ID string (e.g. "DGS10", "CPIAUCSL").
        start_date: First date of the range. Defaults to 10 years ago.
        end_date:   Last date of the range. Defaults to today.
        config:     FredConfig instance. If None, loaded from environment.

    Returns:
        pandas.Series with DatetimeIndex and float values. Values are in
        original FRED units (percent, index, etc.) — divide by 100 for decimals.
    """
    if config is None:
        config = FredConfig.from_env()

    end = end_date or date.today()
    start = start_date or (end - timedelta(days=365 * 10))

    url = f"{config.base_url}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": config.api_key,
        "observation_start": str(start),
        "observation_end": str(end),
        "file_type": "json",
        "sort_order": "asc",
    }

    logger.info("Fetching FRED series %s (%s → %s).", series_id, start, end)
    resp = requests.get(url, params=params, timeout=config.timeout)
    resp.raise_for_status()
    data = resp.json()

    observations = data.get("observations", [])
    if not observations:
        logger.warning("FRED series %s returned no observations.", series_id)
        return pd.Series(dtype=float, name=series_id)

    dates = []
    values = []
    for obs in observations:
        date_str = obs.get("date")
        val_str = obs.get("value", ".")
        if date_str is None:
            continue
        # FRED uses "." to denote missing values
        if val_str == ".":
            val = float("nan")
        else:
            try:
                val = float(val_str)
            except (TypeError, ValueError):
                val = float("nan")
        dates.append(pd.Timestamp(date_str))
        values.append(val)

    series = pd.Series(values, index=dates, name=series_id)
    series.dropna(inplace=True)

    logger.debug(
        "FRED series %s: %d observations fetched (%s → %s).",
        series_id, len(series),
        series.index.min() if len(series) else "N/A",
        series.index.max() if len(series) else "N/A",
    )
    return series


def fetch_treasury_yield(
    maturity: str = "10Y",
    as_of_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
) -> float:
    """
    Return the treasury yield for a given maturity as a decimal rate.

    Args:
        maturity:   "10Y" or "2Y".
        as_of_date: Date for which to retrieve the yield. Defaults to today.
        config:     FredConfig instance.

    Returns:
        Yield as a decimal (e.g. 0.043 for 4.3%).
    """
    maturity_map = {
        "10Y": "DGS10",
        "2Y": "DGS2",
        "3M": "DGS3MO",
        "5Y": "DGS5",
        "30Y": "DGS30",
    }
    series_id = maturity_map.get(maturity.upper())
    if series_id is None:
        raise ValueError(
            f"Unsupported maturity {maturity!r}. "
            f"Supported: {list(maturity_map.keys())}"
        )

    target_date = as_of_date or date.today()
    # Fetch recent data (last 30 days to find the latest obs on or before target)
    start = target_date - timedelta(days=30)
    series = fetch_fred_series(series_id, start_date=start, end_date=target_date, config=config)

    if series.empty:
        logger.warning(
            "Treasury yield series %s returned no data as_of=%s. Returning NaN.",
            series_id, target_date,
        )
        return float("nan")

    # Last observation on or before target_date
    valid = series[series.index <= pd.Timestamp(target_date)]
    if valid.empty:
        logger.warning(
            "No %s treasury observations on or before %s. Returning NaN.",
            maturity, target_date,
        )
        return float("nan")

    yield_pct = float(valid.iloc[-1])
    yield_decimal = yield_pct / 100.0
    logger.debug(
        "Treasury yield %s as_of=%s: %.4f (%.2f%%).",
        maturity, target_date, yield_decimal, yield_pct,
    )
    return yield_decimal


def fetch_cpi_yoy(
    as_of_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
) -> float:
    """
    Return the year-over-year CPI inflation rate as a decimal.

    Returns:
        CPI YoY inflation rate as a decimal (e.g. 0.032 for 3.2%).
    """
    target_date = as_of_date or date.today()
    # Need 13 months of data to compute YoY
    start = target_date - timedelta(days=400)

    series = fetch_fred_series(
        "CPIAUCSL", start_date=start, end_date=target_date, config=config
    )

    if len(series) < 13:
        logger.warning(
            "CPI series has only %d observations (need ≥13). Returning NaN.", len(series)
        )
        return float("nan")

    # Latest observation on or before target_date
    valid = series[series.index <= pd.Timestamp(target_date)]
    if len(valid) < 13:
        logger.warning(
            "Fewer than 13 valid CPI observations on or before %s. Returning NaN.",
            target_date,
        )
        return float("nan")

    cpi_current = float(valid.iloc[-1])
    cpi_year_ago = float(valid.iloc[-13])
    if cpi_year_ago == 0:
        logger.warning("CPI 12 months ago is zero; cannot compute YoY. Returning NaN.")
        return float("nan")

    yoy = (cpi_current / cpi_year_ago) - 1.0
    logger.debug(
        "CPI YoY as_of=%s: %.4f (current=%.2f, 12m_ago=%.2f).",
        target_date, yoy, cpi_current, cpi_year_ago,
    )
    return yoy


def fetch_fed_funds_rate(
    as_of_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
) -> float:
    """
    Return the effective federal funds rate as a decimal.

    Returns:
        Fed funds rate as a decimal (e.g. 0.053 for 5.3%).
    """
    target_date = as_of_date or date.today()
    start = target_date - timedelta(days=60)

    series = fetch_fred_series(
        "FEDFUNDS", start_date=start, end_date=target_date, config=config
    )

    if series.empty:
        logger.warning(
            "FEDFUNDS series returned no data as_of=%s. Returning NaN.", target_date
        )
        return float("nan")

    valid = series[series.index <= pd.Timestamp(target_date)]
    if valid.empty:
        logger.warning(
            "No FEDFUNDS observations on or before %s. Returning NaN.", target_date
        )
        return float("nan")

    rate_pct = float(valid.iloc[-1])
    rate_decimal = rate_pct / 100.0
    logger.debug(
        "Fed funds rate as_of=%s: %.4f (%.2f%%).", target_date, rate_decimal, rate_pct
    )
    return rate_decimal


def fetch_real_gdp_growth(
    as_of_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
) -> float:
    """
    Return the most recent annualised real GDP growth rate as a decimal.

    Returns:
        Real GDP growth rate as a decimal (e.g. 0.025 for 2.5%).
    """
    target_date = as_of_date or date.today()
    # GDP is quarterly — look back up to 2 years
    start = target_date - timedelta(days=730)

    series = fetch_fred_series(
        "A191RL1Q225SBEA", start_date=start, end_date=target_date, config=config
    )

    if series.empty:
        logger.warning(
            "A191RL1Q225SBEA series returned no data as_of=%s. Returning NaN.", target_date
        )
        return float("nan")

    valid = series[series.index <= pd.Timestamp(target_date)]
    if valid.empty:
        logger.warning(
            "No GDP observations on or before %s. Returning NaN.", target_date
        )
        return float("nan")

    gdp_pct = float(valid.iloc[-1])
    gdp_decimal = gdp_pct / 100.0
    logger.debug(
        "Real GDP growth as_of=%s: %.4f (%.2f%%).", target_date, gdp_decimal, gdp_pct
    )
    return gdp_decimal


def fetch_sp500_pe(as_of_date: Optional[date] = None) -> float:
    """
    Return the S&P 500 trailing P/E ratio.

    Falls back from SPY to ^GSPC if the first attempt yields None.

    Returns:
        P/E ratio as a float (e.g. 25.3).
    """
    target_date = as_of_date or date.today()

    # Primary: SPY trailing P/E
    try:
        spy_info = yf.Ticker("SPY").info or {}
        pe = spy_info.get("trailingPE")
        if pe is not None and not math.isnan(float(pe)):
            logger.debug("S&P 500 P/E from SPY.trailingPE: %.2f.", float(pe))
            return float(pe)
        logger.debug("SPY trailingPE is None or NaN; trying ^GSPC.")
    except Exception as exc:
        logger.warning("Failed to fetch SPY info for P/E (%s). Trying ^GSPC.", exc)

    # Fallback: ^GSPC trailing P/E
    try:
        gspc_info = yf.Ticker("^GSPC").info or {}
        pe = gspc_info.get("trailingPE")
        if pe is not None and not math.isnan(float(pe)):
            logger.debug("S&P 500 P/E from ^GSPC.trailingPE: %.2f.", float(pe))
            return float(pe)
        logger.warning("^GSPC trailingPE is also None or NaN. Returning NaN.")
    except Exception as exc:
        logger.warning("Failed to fetch ^GSPC info for P/E (%s). Returning NaN.", exc)

    return float("nan")

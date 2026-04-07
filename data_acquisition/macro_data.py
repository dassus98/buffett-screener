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

from datetime import date
from typing import Optional

import pandas as pd

from data_acquisition.api_config import FredConfig
from data_acquisition.schema import MacroSnapshot


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
                    Data may be slightly lagged (FRED updates with a delay).
        config:     FredConfig instance. If None, loaded from environment.
        use_cache:  If True, read from DuckDB cache if fresh data exists.
        store:      DuckDBStore instance for cache read/write.

    Returns:
        MacroSnapshot dataclass instance with treasury yields, CPI, fed funds
        rate, real GDP growth, and S&P 500 P/E ratio.

    Logic:
        1. Build FredConfig if not provided
        2. Check DuckDB cache for today's snapshot (if use_cache)
        3. Fetch each FRED series via fetch_fred_series()
        4. Take the most recent observation on or before as_of_date
        5. Fetch Shiller CAPE from online source (see fetch_sp500_pe())
        6. Construct MacroSnapshot, write to cache, return
    """
    ...


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
        pandas.Series with DatetimeIndex and float values. All rates are in
        their original FRED units (percent, index, etc.) — callers must
        divide by 100 to get decimals when needed.

    Logic:
        1. Build request URL: {base_url}/series/observations
           Parameters: series_id, api_key, observation_start, observation_end,
                       file_type=json, sort_order=asc
        2. GET the URL with timeout from config
        3. Parse JSON → extract "observations" list → build Series
        4. Cast values to float; replace "." (FRED's missing value marker) with NaN
        5. Drop NaN values, return
    """
    ...


def fetch_treasury_yield(
    maturity: str = "10Y",
    as_of_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
) -> float:
    """
    Return the treasury yield for a given maturity as a decimal rate.

    Args:
        maturity:   "10Y" or "2Y" (other maturities require FRED series lookup).
        as_of_date: Date for which to retrieve the yield. Defaults to today.
        config:     FredConfig instance.

    Returns:
        Yield as a decimal (e.g. 0.043 for 4.3%).

    Logic:
        1. Map maturity to FRED series: "10Y" → "DGS10", "2Y" → "DGS2"
        2. Fetch series via fetch_fred_series()
        3. Take the last available observation on or before as_of_date
        4. Divide by 100 to convert from percent to decimal
        5. Return float
    """
    ...


def fetch_cpi_yoy(
    as_of_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
) -> float:
    """
    Return the year-over-year CPI inflation rate as a decimal.

    Args:
        as_of_date: Reference month-end date. Defaults to most recent available.
        config:     FredConfig instance.

    Returns:
        CPI YoY inflation rate as a decimal (e.g. 0.032 for 3.2%).

    Logic:
        1. Fetch CPIAUCSL series for the last 13 months
        2. Compute: yoy = (cpi_current / cpi_12_months_ago) - 1
        3. Return the rate for the most recent month on or before as_of_date
    """
    ...


def fetch_fed_funds_rate(
    as_of_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
) -> float:
    """
    Return the effective federal funds rate as a decimal.

    Args:
        as_of_date: Date for the observation. Defaults to most recent.
        config:     FredConfig instance.

    Returns:
        Fed funds rate as a decimal (e.g. 0.053 for 5.3%).

    Logic:
        1. Fetch FEDFUNDS series
        2. Take last observation on or before as_of_date
        3. Divide by 100 and return
    """
    ...


def fetch_real_gdp_growth(
    as_of_date: Optional[date] = None,
    config: Optional[FredConfig] = None,
) -> float:
    """
    Return the most recent annualised real GDP growth rate as a decimal.

    Args:
        as_of_date: Quarter-end date. Defaults to most recent available.
        config:     FredConfig instance.

    Returns:
        Real GDP growth rate as a decimal (e.g. 0.025 for 2.5% annualised).

    Logic:
        1. Fetch A191RL1Q225SBEA (real GDP percent change, quarterly annualised)
        2. Take last observation on or before as_of_date
        3. Divide by 100 and return
    """
    ...


def fetch_sp500_pe(as_of_date: Optional[date] = None) -> float:
    """
    Return the S&P 500 Shiller CAPE (cyclically adjusted P/E) ratio.

    Args:
        as_of_date: Month-end date. Defaults to most recent available.

    Returns:
        CAPE ratio as a float (e.g. 30.5).

    Logic:
        1. Download Shiller's CAPE data from his Yale website (public CSV)
           URL: http://www.econ.yale.edu/~shiller/data/ie_data.xls
           (or use a FRED proxy series: MULTPL/SHILLER_PE_RATIO_MONTH)
        2. Parse the relevant sheet, extract Date and CAPE columns
        3. Return the value for the most recent month on or before as_of_date
    """
    ...

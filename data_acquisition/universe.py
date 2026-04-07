"""
data_acquisition.universe
==========================
Builds and maintains the stock universe — the master list of tickers
that the screener will consider.

Sources (in priority order):
    1. S&P 500 constituent list scraped from Wikipedia (free, no API key)
    2. Russell 1000 ETF holdings (iShares IWB) downloaded as CSV
    3. Manual override list from config (add / remove specific tickers)

The output is a list of CompanyProfile objects that include sector,
industry, exchange, and metadata flags (is_adr, is_spac).

Caching:
    Universe snapshots are persisted in DuckDB via the store module to avoid
    re-fetching on every pipeline run. The cache TTL is configurable
    (default: 7 days).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from data_acquisition.schema import CompanyProfile


def fetch_sp500_tickers() -> list[str]:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.

    Args:
        None

    Returns:
        List of ticker symbols (e.g. ["AAPL", "MSFT", ...]) as they appear
        in the Wikipedia table. BRK.B is returned as "BRK-B" to match
        yfinance conventions.

    Logic:
        1. GET https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
        2. Parse the first HTML table with pandas.read_html()
        3. Extract the "Symbol" column, strip whitespace, replace "." with "-"
        4. Return sorted list of unique tickers
    """
    ...


def fetch_russell1000_tickers() -> list[str]:
    """
    Download Russell 1000 ETF (IWB) holdings from iShares and extract tickers.

    Args:
        None

    Returns:
        List of ticker symbols for all equity holdings in the IWB ETF.
        Cash, derivatives, and non-equity rows are excluded.

    Logic:
        1. Download the IWB holdings CSV from iShares (URL from api_config)
        2. Read with pandas, skip header rows until the actual data starts
        3. Filter rows where "Asset Class" == "Equity"
        4. Return cleaned ticker list
    """
    ...


def build_universe(
    config: dict,
    additional_tickers: Optional[list[str]] = None,
    excluded_tickers: Optional[list[str]] = None,
) -> list[str]:
    """
    Merge S&P 500 and Russell 1000 tickers into a deduplicated universe,
    applying additions and exclusions from the config.

    Args:
        config:             The full filter_config.yaml dict (or just the
                            "universe" sub-section).
        additional_tickers: Optional list of extra tickers to force-include
                            regardless of index membership.
        excluded_tickers:   Optional list of tickers to remove from the
                            universe regardless of index membership.

    Returns:
        Sorted list of unique ticker strings ready for data fetching.

    Logic:
        1. Call fetch_sp500_tickers() and fetch_russell1000_tickers()
        2. Union the two sets
        3. Add additional_tickers, remove excluded_tickers
        4. Apply any exchange or index filters from config["universe"]
        5. Return sorted list
    """
    ...


def enrich_with_profiles(
    tickers: list[str],
) -> list[CompanyProfile]:
    """
    Fetch CompanyProfile metadata for each ticker via yfinance.

    Args:
        tickers: List of ticker symbols to enrich.

    Returns:
        List of CompanyProfile dataclass instances. Tickers that fail
        to fetch (delisted, invalid) are logged as warnings and skipped.

    Logic:
        1. For each ticker, call yf.Ticker(ticker).info
        2. Map yfinance fields to CompanyProfile attributes:
               sector, industry, exchange, country, currency,
               longBusinessSummary → description
        3. Detect is_adr via "quoteType" == "ETF" check or country != "US"
           (configurable heuristic)
        4. Set is_spac = True if company name contains "Acquisition" or
           "SPAC" (simple heuristic; can be improved with SEC EDGAR lookup)
        5. Return list, logging any skipped tickers
    """
    ...


def filter_by_market_cap(
    tickers: list[str],
    min_market_cap_usd: float,
) -> list[str]:
    """
    Remove tickers whose current market cap is below the minimum threshold.

    Args:
        tickers:           List of ticker symbols.
        min_market_cap_usd: Minimum market cap in USD (e.g. 500_000_000 for $500M).

    Returns:
        Filtered list of tickers that meet the market cap threshold.

    Logic:
        1. Batch-fetch market caps using yf.download() or Ticker.fast_info
        2. For each ticker, compare marketCap to min_market_cap_usd
        3. Log how many tickers were filtered and why
        4. Return passing tickers
    """
    ...


def load_universe_from_cache(store) -> Optional[list[CompanyProfile]]:
    """
    Load a previously saved universe snapshot from DuckDB if it is fresh
    (within the configured TTL).

    Args:
        store: A DuckDBStore instance connected to the local database.

    Returns:
        List of CompanyProfile objects if a fresh cache exists, else None.

    Logic:
        1. Query the "universe_snapshots" table for the most recent entry
        2. Check if snapshot date is within TTL (default 7 days)
        3. Deserialise rows back into CompanyProfile dataclass instances
        4. Return None if no valid cache found
    """
    ...


def save_universe_to_cache(
    profiles: list[CompanyProfile],
    store,
    snapshot_date: Optional[date] = None,
) -> None:
    """
    Persist a universe snapshot to DuckDB for later retrieval.

    Args:
        profiles:      List of CompanyProfile objects to persist.
        store:         A DuckDBStore instance connected to the local database.
        snapshot_date: Date to tag the snapshot with. Defaults to today.

    Returns:
        None

    Logic:
        1. Convert CompanyProfile list to a pandas DataFrame
        2. Add a "snapshot_date" column
        3. Upsert into the "universe_snapshots" DuckDB table
           (replace rows with the same snapshot_date)
    """
    ...

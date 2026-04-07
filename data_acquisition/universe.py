"""
data_acquisition.universe
==========================
Builds and maintains the stock universe — the master list of tickers
that the screener will consider.

Sources (in priority order):
    1. S&P 500 constituent list scraped from Wikipedia (free, no API key)
    2. Russell 1000 ETF holdings (iShares IWB) downloaded as CSV
    3. S&P/TSX 60 constituent list scraped from Wikipedia (if TSX in exchanges)
    4. Manual override list from config (add / remove specific tickers)

The output is a list of CompanyProfile objects that include sector,
industry, exchange, and metadata flags (is_adr, is_spac).

Caching:
    Universe snapshots are persisted in DuckDB via the store module to avoid
    re-fetching on every pipeline run. The cache TTL is configurable
    (default: 7 days).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf

from data_acquisition.schema import CompanyProfile

logger = logging.getLogger(__name__)

# iShares IWB holdings CSV download URL
_IWB_CSV_URL = (
    "https://www.ishares.com/us/products/239707/IWB/1467271812596.ajax"
    "?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

# Wikipedia URLs
_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_TSX60_WIKI_URL = "https://en.wikipedia.org/wiki/S%26P/TSX_60"


def fetch_sp500_tickers() -> list[str]:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.

    Returns:
        List of ticker symbols with "." replaced by "-" for yfinance compat.
    """
    logger.info("Fetching S&P 500 tickers from Wikipedia.")
    tables = pd.read_html(_SP500_WIKI_URL, header=0)
    # The first table on the page is the constituent list
    df = tables[0]

    # The Symbol column may be named "Symbol" or "Ticker"
    if "Symbol" in df.columns:
        symbols = df["Symbol"].dropna().astype(str).str.strip()
    elif "Ticker" in df.columns:
        symbols = df["Ticker"].dropna().astype(str).str.strip()
    else:
        raise ValueError(
            f"Could not find 'Symbol' or 'Ticker' column in Wikipedia table. "
            f"Columns found: {list(df.columns)}"
        )

    tickers = sorted(set(symbols.str.replace(".", "-", regex=False).tolist()))
    logger.info("S&P 500: fetched %d tickers.", len(tickers))
    return tickers


def fetch_russell1000_tickers() -> list[str]:
    """
    Download Russell 1000 ETF (IWB) holdings from iShares and extract tickers.

    Returns:
        List of ticker symbols for all equity holdings in the IWB ETF.
    """
    logger.info("Fetching Russell 1000 tickers from iShares IWB CSV.")
    try:
        # iShares CSV has several header rows before the data starts
        # We read with a high skiprows budget and detect where data begins
        raw = pd.read_csv(_IWB_CSV_URL, skiprows=2, header=0, thousands=",")
    except Exception as exc:
        logger.warning(
            "Failed to download IWB CSV from iShares (%s). "
            "Russell 1000 tickers will not be included.",
            exc,
        )
        return []

    # The holdings data has an "Asset Class" column; keep only Equity rows
    if "Asset Class" not in raw.columns:
        logger.warning(
            "IWB CSV does not contain 'Asset Class' column. "
            "Columns found: %s. Skipping Russell 1000.",
            list(raw.columns),
        )
        return []

    equity_rows = raw[raw["Asset Class"].str.strip() == "Equity"]

    # Ticker column may be "Ticker" or "Name"
    if "Ticker" in equity_rows.columns:
        ticker_col = "Ticker"
    else:
        logger.warning(
            "IWB CSV does not have a 'Ticker' column. "
            "Columns: %s. Skipping Russell 1000.",
            list(equity_rows.columns),
        )
        return []

    tickers = (
        equity_rows[ticker_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    tickers = sorted(t for t in tickers if t and t != "-")
    logger.info("Russell 1000 (IWB): fetched %d equity tickers.", len(tickers))
    return tickers


def fetch_tsx60_tickers() -> list[str]:
    """
    Scrape the S&P/TSX 60 constituent list from Wikipedia and append '.TO' suffix.

    Returns:
        List of ticker symbols with '.TO' suffix for yfinance compatibility.
    """
    logger.info("Fetching S&P/TSX 60 tickers from Wikipedia.")
    try:
        tables = pd.read_html(_TSX60_WIKI_URL, header=0)
    except Exception as exc:
        logger.warning(
            "Failed to scrape TSX 60 from Wikipedia (%s). Skipping TSX.", exc
        )
        return []

    # Find the table that contains the constituent list (has a "Ticker" column)
    df = None
    for table in tables:
        for col_candidate in ("Ticker", "Symbol", "Ticker symbol"):
            if col_candidate in table.columns:
                df = table
                ticker_col = col_candidate
                break
        if df is not None:
            break

    if df is None:
        logger.warning(
            "Could not find ticker column in TSX 60 Wikipedia page. Skipping TSX."
        )
        return []

    raw_tickers = df[ticker_col].dropna().astype(str).str.strip().tolist()
    # Strip any existing exchange suffixes and append .TO
    tickers = []
    for t in raw_tickers:
        base = t.split(".")[0].strip()
        if base:
            tickers.append(f"{base}.TO")

    tickers = sorted(set(tickers))
    logger.info("S&P/TSX 60: fetched %d tickers (with .TO suffix).", len(tickers))
    return tickers


def build_universe(
    config: dict,
    additional_tickers: Optional[list[str]] = None,
    excluded_tickers: Optional[list[str]] = None,
) -> list[str]:
    """
    Merge index tickers into a deduplicated universe, applying additions
    and exclusions from the config.

    Args:
        config:             The full filter_config.yaml dict (or just the
                            "universe" sub-section).
        additional_tickers: Optional list of extra tickers to force-include.
        excluded_tickers:   Optional list of tickers to remove.

    Returns:
        Sorted list of unique ticker strings ready for data fetching.
    """
    # Support both the full config dict and just the "universe" sub-section
    universe_cfg: dict = config.get("universe", config)

    include_indices: list[str] = universe_cfg.get(
        "include_indices", ["S&P 500", "Russell 1000"]
    )
    exchanges: list[str] = universe_cfg.get("exchanges", ["NYSE", "NASDAQ", "AMEX"])

    ticker_set: set[str] = set()

    if "S&P 500" in include_indices:
        try:
            ticker_set.update(fetch_sp500_tickers())
        except Exception as exc:
            logger.warning("S&P 500 fetch failed: %s. Continuing without it.", exc)

    if "Russell 1000" in include_indices:
        try:
            ticker_set.update(fetch_russell1000_tickers())
        except Exception as exc:
            logger.warning("Russell 1000 fetch failed: %s. Continuing without it.", exc)

    if "TSX" in exchanges:
        try:
            ticker_set.update(fetch_tsx60_tickers())
        except Exception as exc:
            logger.warning("TSX 60 fetch failed: %s. Continuing without it.", exc)

    # Apply additions
    if additional_tickers:
        before = len(ticker_set)
        ticker_set.update(additional_tickers)
        logger.info(
            "Added %d tickers from additional_tickers list.",
            len(ticker_set) - before,
        )

    # Apply exclusions
    excluded_set = set(excluded_tickers or [])
    if excluded_set:
        removed = ticker_set & excluded_set
        ticker_set -= excluded_set
        logger.info("Removed %d tickers from excluded_tickers: %s.", len(removed), sorted(removed))

    result = sorted(ticker_set)
    logger.info("Universe built: %d total tickers.", len(result))
    return result


def enrich_with_profiles(
    tickers: list[str],
) -> list[CompanyProfile]:
    """
    Fetch CompanyProfile metadata for each ticker via yfinance.

    Args:
        tickers: List of ticker symbols to enrich.

    Returns:
        List of CompanyProfile dataclass instances.
    """
    profiles: list[CompanyProfile] = []
    failed = 0

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
                # Attempt to detect totally empty/invalid responses
                if not any(
                    k in info for k in ("longName", "shortName", "sector", "exchange")
                ):
                    logger.warning(
                        "Ticker %s returned empty info dict. Skipping.", ticker
                    )
                    failed += 1
                    continue

            name = (
                info.get("longName")
                or info.get("shortName")
                or ticker
            )
            sector = info.get("sector") or ""
            industry = info.get("industry") or ""
            exchange = info.get("exchange") or ""
            country = info.get("country") or ""
            currency = info.get("currency") or "USD"
            description = info.get("longBusinessSummary") or None

            # Derive fiscal_year_end_month from fiscalYearEnd string (e.g. "December")
            fiscal_year_end_str: str = info.get("fiscalYearEnd", "") or ""
            fiscal_month = _month_name_to_int(fiscal_year_end_str)

            # ADR detection: country != "United States" and quoteType not in
            # standard domestic types, or quoteType == "ADR"
            quote_type = info.get("quoteType", "")
            is_adr = quote_type == "ADR" or (
                country not in ("", "United States") and not ticker.endswith(".TO")
            )

            # SPAC detection: simple name heuristic
            name_upper = name.upper()
            is_spac = any(
                kw in name_upper
                for kw in ("ACQUISITION", "SPAC", "BLANK CHECK")
            )

            profile = CompanyProfile(
                ticker=ticker,
                name=name,
                sector=sector,
                industry=industry,
                exchange=exchange,
                country=country,
                currency=currency,
                fiscal_year_end_month=fiscal_month,
                is_adr=is_adr,
                is_spac=is_spac,
                description=description,
            )
            profiles.append(profile)
            logger.debug(
                "Enriched %s: sector=%s, exchange=%s, is_adr=%s, is_spac=%s.",
                ticker, sector, exchange, is_adr, is_spac,
            )

        except Exception as exc:
            logger.warning(
                "Failed to fetch profile for ticker %s: %s. Skipping.", ticker, exc
            )
            failed += 1

    logger.info(
        "enrich_with_profiles: %d profiles fetched, %d skipped.", len(profiles), failed
    )
    return profiles


def _month_name_to_int(month_name: str) -> int:
    """Map a month name string to its 1-based integer, defaulting to 12."""
    mapping = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    return mapping.get(month_name.strip().lower(), 12)


def filter_by_market_cap(
    tickers: list[str],
    min_market_cap_usd: float,
) -> list[str]:
    """
    Remove tickers whose current market cap is below the minimum threshold.

    Args:
        tickers:           List of ticker symbols.
        min_market_cap_usd: Minimum market cap in USD.

    Returns:
        Filtered list of tickers that meet the market cap threshold.
    """
    passing: list[str] = []
    filtered_out = 0
    unavailable = 0

    for ticker in tickers:
        try:
            fast_info = yf.Ticker(ticker).fast_info
            market_cap = getattr(fast_info, "market_cap", None)
            if market_cap is None:
                logger.warning(
                    "Market cap unavailable for ticker %s (fast_info returned None). "
                    "Skipping this ticker from universe.",
                    ticker,
                )
                unavailable += 1
                continue
            if market_cap < min_market_cap_usd:
                logger.debug(
                    "Ticker %s filtered out: market_cap=%.0f < threshold=%.0f.",
                    ticker, market_cap, min_market_cap_usd,
                )
                filtered_out += 1
            else:
                passing.append(ticker)
        except Exception as exc:
            logger.warning(
                "Could not fetch market cap for %s (%s). Skipping.", ticker, exc
            )
            unavailable += 1

    logger.info(
        "filter_by_market_cap: %d pass, %d filtered (below %.0f USD), %d unavailable.",
        len(passing), filtered_out, min_market_cap_usd, unavailable,
    )
    return passing


def load_universe_from_cache(store) -> Optional[list[CompanyProfile]]:
    """
    Load a previously saved universe snapshot from DuckDB if it is fresh.

    Args:
        store: A DuckDBStore instance connected to the local database.

    Returns:
        List of CompanyProfile objects if a fresh cache exists, else None.
    """
    df = store.load_latest_universe()
    if df is None or df.empty:
        return None

    profiles: list[CompanyProfile] = []
    for _, row in df.iterrows():
        try:
            profile = CompanyProfile(
                ticker=str(row.get("ticker", "")),
                name=str(row.get("name", "") or ""),
                sector=str(row.get("sector", "") or ""),
                industry=str(row.get("industry", "") or ""),
                exchange=str(row.get("exchange", "") or ""),
                country=str(row.get("country", "") or ""),
                currency=str(row.get("currency", "USD") or "USD"),
                fiscal_year_end_month=int(row.get("fiscal_year_end_month") or 12),
                is_adr=bool(row.get("is_adr", False)),
                is_spac=bool(row.get("is_spac", False)),
                description=row.get("description") or None,
            )
            profiles.append(profile)
        except Exception as exc:
            logger.warning(
                "Could not deserialise CompanyProfile row for ticker %s: %s.",
                row.get("ticker", "?"), exc,
            )

    logger.info("Loaded %d profiles from universe cache.", len(profiles))
    return profiles if profiles else None


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
    """
    if not profiles:
        logger.warning("save_universe_to_cache: empty profiles list, nothing saved.")
        return

    target_date = snapshot_date or date.today()
    rows = [asdict(p) for p in profiles]
    df = pd.DataFrame(rows)
    store.save_universe(df, snapshot_date=target_date)
    logger.info(
        "Saved %d profiles to universe cache (snapshot_date=%s).",
        len(profiles), target_date,
    )

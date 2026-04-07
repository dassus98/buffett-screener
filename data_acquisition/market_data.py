"""
data_acquisition.market_data
=============================
Fetches current and historical market data (price, market cap, enterprise
value, beta, volume) for a single ticker or a batch of tickers.

Source: yfinance (primary) — no API key required.

All returned values are point-in-time as of the `as_of_date` parameter.
For the live pipeline this defaults to today; for backtesting it should be
set to the desired historical date.

Note on enterprise value:
    EV = market_cap + total_debt - cash_and_equivalents
    This is computed here using balance sheet data from the most recent
    quarter, cross-referenced with the BalanceSheet from financials.py.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from data_acquisition.schema import MarketData

logger = logging.getLogger(__name__)


def fetch_market_data(
    ticker: str,
    as_of_date: Optional[date] = None,
    use_cache: bool = True,
    store=None,
) -> MarketData:
    """
    Fetch a complete MarketData snapshot for a single ticker.

    Args:
        ticker:      Stock ticker symbol (e.g. "KO").
        as_of_date:  Date for which to retrieve data. Defaults to today.
        use_cache:   If True, check DuckDB for a cached snapshot first.
        store:       DuckDBStore instance for cache read/write.

    Returns:
        MarketData dataclass instance.

    Raises:
        ValueError: If the ticker returns no data.
    """
    target_date = as_of_date or date.today()

    # ---- Cache check -------------------------------------------------------
    if use_cache and store is not None:
        cached = store.load_market_data(ticker, as_of_date=target_date)
        if cached is not None:
            # Accept cache hit if within 1 day of target_date
            cached_date = pd.Timestamp(cached["as_of_date"]).date()
            age = abs((target_date - cached_date).days)
            if age <= 1:
                logger.debug(
                    "Cache hit for market_data ticker=%s (as_of=%s).", ticker, cached_date
                )
                return MarketData(
                    ticker=ticker,
                    as_of_date=cached_date,
                    price=float(cached.get("price", float("nan"))),
                    market_cap=float(cached.get("market_cap", float("nan"))),
                    enterprise_value=float(cached.get("enterprise_value", float("nan"))),
                    shares_outstanding=float(cached.get("shares_outstanding", float("nan"))),
                    beta=float(cached.get("beta", float("nan"))),
                    avg_daily_volume_30d=float(cached.get("avg_daily_volume_30d", float("nan"))),
                    fifty_two_week_high=float(cached.get("fifty_two_week_high", float("nan"))),
                    fifty_two_week_low=float(cached.get("fifty_two_week_low", float("nan"))),
                )

    # ---- yfinance fetch ----------------------------------------------------
    logger.info("Fetching market data for ticker=%s as_of=%s.", ticker, target_date)
    yf_ticker = yf.Ticker(ticker)

    try:
        fast_info = yf_ticker.fast_info
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch fast_info for ticker {ticker}: {exc}"
        ) from exc

    # Price
    price: float = float("nan")
    if target_date >= date.today() - timedelta(days=1):
        # Live / near-live price
        price_raw = getattr(fast_info, "last_price", None)
        if price_raw is not None:
            price = float(price_raw)
    else:
        # Historical close price
        start = target_date
        end = target_date + timedelta(days=5)  # buffer for weekends/holidays
        hist = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        if not hist.empty:
            # Use the closest available date on or before target_date
            close_col = "Close"
            if isinstance(hist.columns, pd.MultiIndex):
                close_col = ("Close", ticker)
            hist_on_or_before = hist[hist.index <= pd.Timestamp(target_date)]
            if not hist_on_or_before.empty:
                price = float(hist_on_or_before[close_col].iloc[-1])

    if math.isnan(price):
        logger.warning(
            "Price unavailable for ticker=%s as_of=%s. Storing NaN.", ticker, target_date
        )

    # Market cap
    market_cap: float = float("nan")
    mc_raw = getattr(fast_info, "market_cap", None)
    if mc_raw is not None:
        market_cap = float(mc_raw)
    else:
        logger.warning(
            "market_cap unavailable from fast_info for ticker=%s (as_of=%s). Storing NaN.",
            ticker, target_date,
        )

    # Shares outstanding (in thousands)
    shares_outstanding = get_shares_outstanding(ticker)

    # beta, 52-week hi/lo — from full info dict (slower but more complete)
    beta: float = float("nan")
    fifty_two_week_high: float = float("nan")
    fifty_two_week_low: float = float("nan")
    try:
        info = yf_ticker.info or {}
        beta_raw = info.get("beta")
        if beta_raw is not None:
            beta = float(beta_raw)
        else:
            logger.warning(
                "beta unavailable for ticker=%s. Storing NaN.", ticker
            )

        hi_raw = info.get("fiftyTwoWeekHigh") or info.get("52WeekHigh")
        lo_raw = info.get("fiftyTwoWeekLow") or info.get("52WeekLow")
        if hi_raw is not None:
            fifty_two_week_high = float(hi_raw)
        else:
            logger.warning("52-week high unavailable for ticker=%s. Storing NaN.", ticker)
        if lo_raw is not None:
            fifty_two_week_low = float(lo_raw)
        else:
            logger.warning("52-week low unavailable for ticker=%s. Storing NaN.", ticker)
    except Exception as exc:
        logger.warning(
            "Failed to fetch info dict for ticker=%s (%s). beta/52wk unavailable.", ticker, exc
        )

    # 30-day average daily volume
    avg_daily_volume_30d: float = float("nan")
    try:
        vol_end = target_date + timedelta(days=1)
        vol_start = target_date - timedelta(days=35)
        vol_hist = yf.download(
            ticker, start=vol_start, end=vol_end, interval="1d", progress=False
        )
        if not vol_hist.empty:
            vol_col = "Volume"
            if isinstance(vol_hist.columns, pd.MultiIndex):
                vol_col = ("Volume", ticker)
            avg_daily_volume_30d = float(vol_hist[vol_col].dropna().tail(30).mean())
        else:
            logger.warning(
                "No volume history found for ticker=%s. avg_daily_volume_30d=NaN.", ticker
            )
    except Exception as exc:
        logger.warning(
            "Failed to compute 30d avg volume for ticker=%s: %s. Storing NaN.", ticker, exc
        )

    # Enterprise value — prefer fast_info.enterprise_value, then compute manually
    enterprise_value: float = float("nan")
    ev_raw = getattr(fast_info, "enterprise_value", None)
    if ev_raw is not None:
        enterprise_value = float(ev_raw)
    else:
        # Compute from balance sheet if market_cap is available
        logger.warning(
            "enterprise_value unavailable from fast_info for ticker=%s. "
            "Attempting manual computation from balance sheet.",
            ticker,
        )
        try:
            info_dict = yf_ticker.info or {}
            total_debt_raw = info_dict.get("totalDebt")
            cash_raw = info_dict.get("totalCash")
            if (
                not math.isnan(market_cap)
                and total_debt_raw is not None
                and cash_raw is not None
            ):
                enterprise_value = compute_enterprise_value(
                    market_cap=market_cap,
                    total_debt=float(total_debt_raw),
                    cash_and_equivalents=float(cash_raw),
                )
            else:
                logger.warning(
                    "Cannot compute EV for ticker=%s: missing totalDebt or totalCash. Storing NaN.",
                    ticker,
                )
        except Exception as exc:
            logger.warning(
                "Manual EV computation failed for ticker=%s: %s. Storing NaN.", ticker, exc
            )

    md = MarketData(
        ticker=ticker,
        as_of_date=target_date,
        price=price,
        market_cap=market_cap,
        enterprise_value=enterprise_value,
        shares_outstanding=shares_outstanding,
        beta=beta,
        avg_daily_volume_30d=avg_daily_volume_30d,
        fifty_two_week_high=fifty_two_week_high,
        fifty_two_week_low=fifty_two_week_low,
    )

    # ---- Write to cache ----------------------------------------------------
    if use_cache and store is not None:
        try:
            store.save_market_data(
                pd.DataFrame([dataclasses.asdict(md)])
            )
        except Exception as exc:
            logger.warning(
                "Failed to write market_data for %s to cache: %s.", ticker, exc
            )

    return md


def fetch_market_data_batch(
    tickers: list[str],
    as_of_date: Optional[date] = None,
    use_cache: bool = True,
    store=None,
    max_workers: int = 8,
) -> dict[str, MarketData]:
    """
    Fetch MarketData for a list of tickers concurrently.

    Args:
        tickers:     List of ticker symbols.
        as_of_date:  Date for data retrieval. Defaults to today.
        use_cache:   Whether to read/write DuckDB cache.
        store:       DuckDBStore instance.
        max_workers: Thread pool size.

    Returns:
        Dict mapping ticker → MarketData. Tickers that fail are logged and omitted.
    """
    results: dict[str, MarketData] = {}
    target_date = as_of_date or date.today()

    logger.info(
        "fetch_market_data_batch: fetching %d tickers with %d workers (as_of=%s).",
        len(tickers), max_workers, target_date,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(
                fetch_market_data, ticker, target_date, use_cache, store
            ): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                md = future.result()
                results[ticker] = md
            except Exception as exc:
                logger.warning(
                    "fetch_market_data_batch: failed for ticker=%s (%s). Omitting.", ticker, exc
                )

    logger.info(
        "fetch_market_data_batch: completed %d/%d successfully.",
        len(results), len(tickers),
    )
    return results


def fetch_price_history(
    ticker: str,
    start_date: date,
    end_date: Optional[date] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV price history for a ticker over a date range.

    Args:
        ticker:     Ticker symbol.
        start_date: First date of the range (inclusive).
        end_date:   Last date of the range (inclusive). Defaults to today.
        interval:   yfinance interval string: "1d", "1wk", "1mo".

    Returns:
        DataFrame with columns: [Open, High, Low, Close, Adj Close, Volume]
        indexed by date, sorted ascending.
    """
    end = end_date or date.today()
    # yfinance's end is exclusive, so add 1 day
    end_exclusive = end + timedelta(days=1)

    logger.info(
        "Fetching price history for ticker=%s (%s → %s, interval=%s).",
        ticker, start_date, end, interval,
    )

    hist = yf.download(
        ticker,
        start=str(start_date),
        end=str(end_exclusive),
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if hist.empty:
        logger.warning(
            "No price history returned for ticker=%s (%s → %s).", ticker, start_date, end
        )
        return pd.DataFrame()

    # Flatten MultiIndex columns if present (happens with single ticker too in newer yfinance)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [col[0] for col in hist.columns]

    # Ensure Adj Close exists
    if "Adj Close" not in hist.columns:
        if "Close" in hist.columns:
            logger.warning(
                "ticker=%s: 'Adj Close' column missing; using 'Close' as fallback.",
                ticker,
            )
            hist["Adj Close"] = hist["Close"]
        else:
            logger.warning(
                "ticker=%s: Neither 'Adj Close' nor 'Close' found in price history.",
                ticker,
            )

    # Drop rows with NaN in Close or Volume
    required_cols = [c for c in ["Close", "Volume"] if c in hist.columns]
    if required_cols:
        before = len(hist)
        hist.dropna(subset=required_cols, inplace=True)
        dropped = before - len(hist)
        if dropped:
            logger.debug(
                "ticker=%s: dropped %d rows with NaN in Close/Volume.", ticker, dropped
            )

    hist.sort_index(inplace=True)
    return hist


def compute_enterprise_value(
    market_cap: float,
    total_debt: float,
    cash_and_equivalents: float,
    minority_interest: float = 0.0,
    preferred_equity: float = 0.0,
) -> float:
    """
    Compute enterprise value from its components.

    Formula:
        EV = market_cap + total_debt + minority_interest + preferred_equity
             - cash_and_equivalents
    """
    return market_cap + total_debt + minority_interest + preferred_equity - cash_and_equivalents


def get_shares_outstanding(ticker: str) -> float:
    """
    Retrieve the most current diluted shares outstanding for a ticker.

    Returns:
        Diluted shares outstanding in thousands.

    Raises:
        ValueError: If neither source returns a non-zero value.
    """
    yf_ticker = yf.Ticker(ticker)

    # Try fast_info first
    try:
        fast_info = yf_ticker.fast_info
        shares_raw = getattr(fast_info, "shares", None)
        if shares_raw is not None and float(shares_raw) > 0:
            shares_k = float(shares_raw) / 1_000.0
            logger.debug(
                "ticker=%s: shares from fast_info.shares = %.0fk.", ticker, shares_k
            )
            return shares_k
    except Exception as exc:
        logger.debug(
            "ticker=%s: fast_info.shares failed (%s). Trying info dict.", ticker, exc
        )

    # Fall back to info dict
    try:
        info = yf_ticker.info or {}
        shares_raw = info.get("sharesOutstanding")
        if shares_raw is not None and float(shares_raw) > 0:
            shares_k = float(shares_raw) / 1_000.0
            logger.debug(
                "ticker=%s: shares from info['sharesOutstanding'] = %.0fk.", ticker, shares_k
            )
            return shares_k
    except Exception as exc:
        logger.warning(
            "ticker=%s: info['sharesOutstanding'] failed (%s). Returning NaN.", ticker, exc
        )

    logger.warning(
        "ticker=%s: shares_outstanding unavailable from both fast_info and info dict. "
        "Returning NaN.",
        ticker,
    )
    return float("nan")

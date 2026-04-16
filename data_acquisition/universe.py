"""Builds the investable stock universe from FMP's stock screener endpoint.

Fetches all actively-traded securities meeting the minimum market-cap threshold
across the configured exchanges (TSX, NYSE, NASDAQ, or any subset), applies
sector/industry exclusions, and persists the result to a Parquet cache.

Authoritative spec: docs/DATA_SOURCES.md §1, §10.
All thresholds and exchange lists come from config/filter_config.yaml at runtime
via screener.filter_config_loader.

Data lineage contract
---------------------
Upstream dependencies:
  api_config.py        → build_fmp_url, fmp_limiter, resilient_request
  filter_config_loader → get_config (for universe, exclusions config sections)

Config dependency map (all from config/filter_config.yaml):
  universe.exchanges           → fetch_universe (which exchanges to query)
  universe.min_market_cap_usd  → fetch_universe (marketCapMoreThan param)
  exclusions.financial_sector_label → filter_universe (sector exclusion)
  exclusions.sectors           → filter_universe (additional sector exclusion)
  exclusions.industry_keywords → filter_universe (industry keyword exclusion)

Downstream consumers:
  store.py             → writes universe DataFrame to DuckDB universe table
  screener/exclusions.py → applies finer-grained SIC/sector/industry/flag filters

Key exports
-----------
fetch_universe() -> pd.DataFrame
    Hit FMP /stock-screener, one exchange at a time, return merged DataFrame.
filter_universe(df) -> pd.DataFrame
    Remove excluded sectors / industries (reads rules from config).
get_universe(use_cache=True) -> pd.DataFrame
    Orchestrator: cache-read or fetch-then-cache.
"""

from __future__ import annotations

import logging
import pathlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from data_acquisition.api_config import build_fmp_url, fmp_limiter, resilient_request
from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Columns guaranteed to be present in the returned DataFrame.
UNIVERSE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "exchange",
    "company_name",
    "market_cap_usd",
    "sector",
    "industry",
    "country",
)

#: Maximum rows FMP returns per screener request. If a response is exactly
#: this size we warn that results may be truncated.
_FMP_SCREENER_LIMIT = 10_000

#: Age in seconds beyond which the cached universe file is considered stale.
#: Not currently in config; could be added as universe.cache_ttl_days.
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

#: FMP exchange identifiers. Keys are canonical config names; values are the
#: parameter string sent to the API.
_EXCHANGE_MAP: dict[str, str] = {
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
    "AMEX": "AMEX",
    "TSX": "TSX",   # FMP uses "TSX"; fallback to "TO" handled in fetch_universe
}

# ---------------------------------------------------------------------------
# Field name mapping from FMP response → canonical schema
# ---------------------------------------------------------------------------
_FMP_FIELD_MAP: dict[str, str] = {
    "symbol": "ticker",
    "companyName": "company_name",
    "marketCap": "market_cap_usd",
    "sector": "sector",
    "industry": "industry",
    "country": "country",
    "exchangeShortName": "exchange",
}

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_universe() -> pd.DataFrame:
    """Fetch all actively-traded securities meeting eligibility criteria from FMP.

    Iterates over each exchange listed in ``config.universe.exchanges``, calls
    the FMP ``/stock-screener`` endpoint with ``marketCapMoreThan`` and
    ``isActivelyTrading=true``, and concatenates the results.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``exchange``, ``company_name``, ``market_cap_usd``,
        ``sector``, ``industry``, ``country``. All string columns are stripped
        of whitespace. ``market_cap_usd`` is stored as float (full USD dollars).

    Notes
    -----
    - If any exchange returns exactly ``_FMP_SCREENER_LIMIT`` rows, a WARNING is
      logged indicating possible truncation; increase the ``limit`` parameter or
      paginate in that case.
    - TSX tickers are returned by FMP as ``SHOP.TO`` style; these are stored
      as-is and later used directly with yfinance (which requires the ``.TO``
      suffix). For FMP endpoints the base symbol is used.
    - If FMP returns 0 rows for a configured exchange, logs WARNING and
      continues (TSX coverage may be limited on lower FMP tiers).
    """
    cfg = get_config()
    exchanges: list[str] = cfg["universe"]["exchanges"]
    min_cap: int = int(cfg["universe"]["min_market_cap_usd"])

    all_frames: list[pd.DataFrame] = []

    for exchange in exchanges:
        fmp_exchange = _EXCHANGE_MAP.get(exchange.upper(), exchange.upper())
        logger.info(
            "Fetching universe for exchange=%s (FMP identifier=%s), min_cap=$%s",
            exchange,
            fmp_exchange,
            f"{min_cap:,}",
        )

        rows = _fetch_single_exchange(fmp_exchange, min_cap)

        if len(rows) == 0:
            logger.warning(
                "Exchange %s returned 0 tickers. "
                "Verify FMP coverage for this exchange on your API tier.",
                exchange,
            )
            continue

        if len(rows) >= _FMP_SCREENER_LIMIT:
            logger.warning(
                "Exchange %s returned exactly %d rows — results may be truncated. "
                "Consider filtering further or paginating.",
                exchange,
                len(rows),
            )

        frame = _normalise_fmp_rows(rows, canonical_exchange=exchange)
        all_frames.append(frame)
        logger.info(
            "Exchange %s: %d tickers fetched (after normalisation).",
            exchange,
            len(frame),
        )

    if not all_frames:
        logger.warning(
            "FMP screener returned 0 tickers for all exchanges %s. "
            "Attempting Wikipedia + FMP profile fallback.",
            exchanges,
        )
        fallback = _fetch_universe_via_wikipedia(exchanges, min_cap)
        if not fallback.empty:
            return fallback
        logger.error("Both FMP screener and Wikipedia fallback returned 0 tickers.")
        return _empty_universe_df()

    combined = pd.concat(all_frames, ignore_index=True)
    # Drop duplicates that may arise if FMP lists a ticker on multiple exchanges.
    combined = combined.drop_duplicates(subset=["ticker"], keep="first")
    logger.info(
        "Total universe after deduplication: %d tickers across %d exchanges.",
        len(combined),
        len(all_frames),
    )
    return combined


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove tickers in excluded sectors/industries from the universe DataFrame.

    Uses sector and industry strings from FMP because FMP's stock screener does
    not return SIC codes directly.  The exclusion criteria mirror the SIC ranges
    in ``config.exclusions.sic_codes`` (see docs/DATA_SOURCES.md §10).

    All exclusion values are read from ``config/filter_config.yaml`` at runtime:

    * **Sector exclusion** (case-insensitive exact match):
      ``exclusions.financial_sector_label`` + ``exclusions.sectors``
    * **Industry exclusion** (case-insensitive substring match):
      ``exclusions.industry_keywords``

    A row is removed if it matches either the sector or industry rule (OR logic).
    This is intentionally broad; ``screener/exclusions.py`` applies finer-grained
    SIC + sector + industry + flag checks on the surviving set.

    Parameters
    ----------
    df:
        Universe DataFrame produced by :func:`fetch_universe`. Must contain
        ``sector`` and ``industry`` columns.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with excluded rows removed. Index is reset.

    Logs
    ----
    - INFO: total removed and count per exclusion reason.
    - WARNING: if ``sector`` or ``industry`` column is missing (skips that check).
    """
    cfg = get_config()
    excl_cfg: dict[str, Any] = cfg.get("exclusions", {})

    # --- Build excluded-sectors set from config ---
    # Primary financial sector label (e.g. "Financial Services")
    financial_label: str = excl_cfg.get("financial_sector_label", "")
    # Additional GICS sector names (e.g. ["Financials", "Utilities", "Real Estate"])
    extra_sectors: list[str] = [
        s.strip() for s in excl_cfg.get("sectors", []) if s and str(s).strip()
    ]
    excluded_sectors: set[str] = set()
    if financial_label.strip():
        excluded_sectors.add(financial_label.strip().lower())
    excluded_sectors.update(s.lower() for s in extra_sectors)

    # --- Build industry keyword list from config ---
    # Keywords like "Banks", "Insurance", "REIT" — substring match against industry
    industry_keywords: list[str] = [
        str(k).strip()
        for k in excl_cfg.get("industry_keywords", [])
        if k and str(k).strip()
    ]

    original_count = len(df)
    mask_keep = pd.Series(True, index=df.index)

    # --- Sector filter ---
    if "sector" not in df.columns:
        logger.warning("filter_universe: 'sector' column missing — skipping sector exclusion.")
    elif excluded_sectors:
        sector_lower = df["sector"].fillna("").str.lower()
        sector_mask = sector_lower.isin(excluded_sectors)
        n_sector = int(sector_mask.sum())
        if n_sector:
            logger.info(
                "filter_universe: removing %d tickers by sector exclusion %s.",
                n_sector,
                sorted(excluded_sectors),
            )
        mask_keep &= ~sector_mask

    # --- Industry keyword filter ---
    if "industry" not in df.columns:
        logger.warning(
            "filter_universe: 'industry' column missing — skipping industry exclusion."
        )
    elif industry_keywords:
        industry_lower = df["industry"].fillna("").str.lower()
        industry_mask = pd.Series(False, index=df.index)
        for keyword in industry_keywords:
            kw_mask = industry_lower.str.contains(keyword.lower(), regex=False)
            n_kw = int((kw_mask & mask_keep).sum())
            if n_kw:
                logger.info(
                    "filter_universe: removing %d tickers matching industry keyword '%s'.",
                    n_kw,
                    keyword,
                )
            industry_mask |= kw_mask
        mask_keep &= ~industry_mask

    filtered = df[mask_keep].reset_index(drop=True)
    removed = original_count - len(filtered)
    logger.info(
        "filter_universe: %d → %d tickers (%d removed by exclusions).",
        original_count,
        len(filtered),
        removed,
    )
    return filtered


def get_universe(use_cache: bool = True) -> pd.DataFrame:
    """Orchestrate universe fetch with optional Parquet caching.

    Parameters
    ----------
    use_cache:
        If ``True`` and a non-stale cache file exists at
        ``data/raw/universe.parquet``, load from cache and skip the API call.
        The cache TTL is 7 days (``_CACHE_TTL_SECONDS``). Defaults to ``True``.
        Set to ``False`` to force a fresh fetch regardless of cache state.

    Returns
    -------
    pd.DataFrame
        Filtered universe with columns in ``UNIVERSE_COLUMNS``.

    Logs
    ----
    - INFO: cache hit/miss, universe sizes before and after filtering.
    """
    cache_path = _resolve_cache_path()

    if use_cache and _cache_is_fresh(cache_path):
        logger.info("Loading universe from cache: %s", cache_path)
        df = pd.read_parquet(cache_path)
        logger.info("Cached universe: %d tickers.", len(df))
        return df

    logger.info("Fetching fresh universe from FMP (use_cache=%s).", use_cache)
    raw = fetch_universe()
    logger.info("Raw universe: %d tickers before filtering.", len(raw))

    filtered = filter_universe(raw)
    logger.info("Filtered universe: %d tickers after exclusions.", len(filtered))

    _save_cache(filtered, cache_path)
    return filtered


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_single_exchange(fmp_exchange: str, min_cap: int) -> list[dict[str, Any]]:
    """Call FMP /company-screener for one exchange and return raw rows.

    Uses the FMP ``/stable/company-screener`` endpoint (the replacement
    for the deprecated ``/api/v3/stock-screener``).

    Parameters
    ----------
    fmp_exchange:
        The exchange identifier string to pass to FMP (e.g. ``"NYSE"``).
    min_cap:
        Minimum market cap in full USD dollars.

    Returns
    -------
    list[dict]
        Raw JSON rows from FMP. Empty list on error.
    """
    url, params = build_fmp_url(
        "/company-screener",
        use_stable=True,
        marketCapMoreThan=min_cap,
        exchange=fmp_exchange,
        isActivelyTrading="true",
        limit=_FMP_SCREENER_LIMIT,
    )
    try:
        result = resilient_request(url, params=params, rate_limiter=fmp_limiter)
    except Exception as exc:
        logger.error(
            "Failed to fetch universe for exchange %s: %s", fmp_exchange, exc
        )
        return []

    if not isinstance(result, list):
        logger.error(
            "Unexpected response type for exchange %s: %s", fmp_exchange, type(result)
        )
        return []

    return result


def _normalise_fmp_rows(
    rows: list[dict[str, Any]], canonical_exchange: str
) -> pd.DataFrame:
    """Convert raw FMP screener rows to a canonical DataFrame.

    Renames FMP fields to canonical schema names, coerces types, fills missing
    columns with ``None`` / empty string, and appends the canonical exchange name.

    Parameters
    ----------
    rows:
        Raw list of dicts from FMP /stock-screener.
    canonical_exchange:
        The canonical exchange name (e.g. ``"NYSE"``). Written to the
        ``exchange`` column, overriding whatever FMP returns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching ``UNIVERSE_COLUMNS``.
    """
    df = pd.DataFrame(rows)

    # Rename FMP fields to canonical names.
    df = df.rename(columns=_FMP_FIELD_MAP)

    # Ensure all required columns exist (FMP may omit some).
    for col in UNIVERSE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Overwrite exchange with the canonical name (e.g. "TSX" not "TSX" or "TO").
    df["exchange"] = canonical_exchange

    # Coerce market cap to float64; set to NaN if unparseable.
    # Always float64 (not int64) so NaN can be represented for missing values.
    df["market_cap_usd"] = pd.to_numeric(
        df["market_cap_usd"], errors="coerce"
    ).astype("float64")

    # Strip whitespace from string columns.
    str_cols = ["ticker", "company_name", "sector", "industry", "country"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df[list(UNIVERSE_COLUMNS)]


def _resolve_cache_path() -> pathlib.Path:
    """Return the absolute path to the universe Parquet cache file.

    Creates the parent directory if it does not exist.

    Returns
    -------
    pathlib.Path
        ``{project_root}/data/raw/universe.parquet``
    """
    project_root = pathlib.Path(__file__).parent.parent
    cache_dir = project_root / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "universe.parquet"


def _cache_is_fresh(cache_path: pathlib.Path) -> bool:
    """Return True if the cache file exists and is younger than _CACHE_TTL_SECONDS.

    Parameters
    ----------
    cache_path:
        Path to the Parquet cache file.

    Returns
    -------
    bool
        ``True`` if the file exists and its modification time is within the TTL.
    """
    if not cache_path.exists():
        return False
    mtime = cache_path.stat().st_mtime
    age_seconds = datetime.now(tz=timezone.utc).timestamp() - mtime
    if age_seconds <= _CACHE_TTL_SECONDS:
        logger.debug(
            "Cache is fresh (age=%.0fs, ttl=%ds): %s",
            age_seconds,
            _CACHE_TTL_SECONDS,
            cache_path,
        )
        return True
    logger.info(
        "Cache is stale (age=%.0fs > ttl=%ds): %s",
        age_seconds,
        _CACHE_TTL_SECONDS,
        cache_path,
    )
    return False


def _save_cache(df: pd.DataFrame, cache_path: pathlib.Path) -> None:
    """Persist universe DataFrame to Parquet.

    Parameters
    ----------
    df:
        Universe DataFrame to cache.
    cache_path:
        Destination path for the Parquet file.
    """
    try:
        df.to_parquet(cache_path, index=False)
        logger.info("Universe cached to %s (%d rows).", cache_path, len(df))
    except Exception as exc:
        logger.error("Failed to write universe cache to %s: %s", cache_path, exc)


def _empty_universe_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical universe columns.

    Returns
    -------
    pd.DataFrame
        Zero-row DataFrame with typed columns matching ``UNIVERSE_COLUMNS``.
    """
    return pd.DataFrame(columns=list(UNIVERSE_COLUMNS))


# ---------------------------------------------------------------------------
# Wikipedia + FMP profile fallback (for FMP plans without the screener)
# ---------------------------------------------------------------------------

_WIKI_SP500_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
_WIKI_TSX_URL = (
    "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
)
_WIKI_HEADERS = {"User-Agent": "BuffettScreener/1.0 (research project)"}


def _scrape_sp500_tickers() -> list[str]:
    """Scrape S&P 500 constituent tickers from Wikipedia.

    Returns
    -------
    list[str]
        Ticker symbols (e.g. ``["AAPL", "MSFT", ...]``).
        Empty list on failure.
    """
    import io

    import requests as _req

    try:
        resp = _req.get(_WIKI_SP500_URL, headers=_WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        symbols = df["Symbol"].astype(str).str.strip().tolist()
        logger.info("Wikipedia S&P 500: %d tickers scraped.", len(symbols))
        return symbols
    except Exception as exc:
        logger.warning("Failed to scrape S&P 500 from Wikipedia: %s", exc)
        return []


def _scrape_tsx_tickers() -> list[str]:
    """Scrape S&P/TSX Composite constituent tickers from Wikipedia.

    Returns
    -------
    list[str]
        Ticker symbols with ``.TO`` suffix (e.g. ``["RY.TO", "TD.TO"]``).
        Empty list on failure.
    """
    import io

    import requests as _req

    try:
        resp = _req.get(_WIKI_TSX_URL, headers=_WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        # Find the table with a "Symbol" or "Ticker" column
        for tbl in tables:
            for col in ("Symbol", "Ticker"):
                if col in tbl.columns:
                    syms = tbl[col].astype(str).str.strip().tolist()
                    # Ensure .TO suffix
                    syms = [
                        s if s.endswith(".TO") else f"{s}.TO"
                        for s in syms if s and s != "nan"
                    ]
                    logger.info(
                        "Wikipedia TSX Composite: %d tickers scraped.",
                        len(syms),
                    )
                    return syms
        logger.warning("Wikipedia TSX page: no Symbol/Ticker column found.")
        return []
    except Exception as exc:
        logger.warning("Failed to scrape TSX from Wikipedia: %s", exc)
        return []


def _fetch_profile(ticker: str) -> dict[str, Any] | None:
    """Fetch a single ticker profile from FMP ``/stable/profile``.

    Parameters
    ----------
    ticker:
        Ticker symbol (e.g. ``"AAPL"`` or ``"RY.TO"``).

    Returns
    -------
    dict or None
        Profile dict on success, ``None`` on failure.
    """
    url, params = build_fmp_url(
        "/profile", use_stable=True, symbol=ticker,
    )
    try:
        result = resilient_request(url, params=params, rate_limiter=fmp_limiter)
    except Exception:
        return None
    if isinstance(result, list) and result:
        return result[0]
    return None


def _fetch_universe_via_wikipedia(
    exchanges: list[str],
    min_cap: int,
) -> pd.DataFrame:
    """Build the universe from Wikipedia index lists + FMP profile data.

    Fallback used when the FMP screener endpoint is unavailable (e.g.
    free-tier plans). Scrapes S&P 500 and S&P/TSX Composite constituent
    lists from Wikipedia, then enriches each ticker with market cap,
    sector, and industry via the FMP ``/stable/profile`` endpoint.

    Parameters
    ----------
    exchanges:
        List of canonical exchange names from config (e.g.
        ``["TSX", "NYSE", "NASDAQ"]``).
    min_cap:
        Minimum market cap in full USD dollars.

    Returns
    -------
    pd.DataFrame
        Universe DataFrame with columns matching ``UNIVERSE_COLUMNS``.
    """
    tickers_by_exchange: dict[str, list[str]] = {}

    # Scrape US indices
    us_exchanges = {"NYSE", "NASDAQ", "AMEX"}
    if us_exchanges & set(e.upper() for e in exchanges):
        sp500 = _scrape_sp500_tickers()
        if sp500:
            tickers_by_exchange["US"] = sp500

    # Scrape TSX
    if "TSX" in (e.upper() for e in exchanges):
        tsx = _scrape_tsx_tickers()
        if tsx:
            tickers_by_exchange["TSX"] = tsx

    all_tickers = []
    for group_tickers in tickers_by_exchange.values():
        all_tickers.extend(group_tickers)

    if not all_tickers:
        logger.error("Wikipedia fallback: no tickers scraped.")
        return _empty_universe_df()

    logger.info(
        "Wikipedia fallback: enriching %d tickers via FMP profile...",
        len(all_tickers),
    )

    rows = _enrich_tickers_via_profile(all_tickers, min_cap)

    if not rows:
        return _empty_universe_df()

    df = pd.DataFrame(rows)
    # Ensure all columns exist
    for col in UNIVERSE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(UNIVERSE_COLUMNS)]
    df = df.drop_duplicates(subset=["ticker"], keep="first")

    logger.info(
        "Wikipedia fallback: %d tickers after profile enrichment "
        "and min_cap filter ($%s).",
        len(df),
        f"{min_cap:,}",
    )
    return df


def _enrich_tickers_via_profile(
    tickers: list[str],
    min_cap: int,
) -> list[dict[str, Any]]:
    """Fetch FMP profile for each ticker and filter by market cap.

    Parameters
    ----------
    tickers:
        List of ticker symbols to enrich.
    min_cap:
        Minimum market cap in USD.

    Returns
    -------
    list[dict]
        Rows matching the universe schema for tickers above min_cap.
    """
    rows: list[dict[str, Any]] = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0 or (i + 1) == total:
            logger.info(
                "Profile enrichment: %d/%d tickers processed...",
                i + 1,
                total,
            )

        profile = _fetch_profile(ticker)
        if profile is None:
            continue

        mkt_cap = profile.get("marketCap") or 0
        if mkt_cap < min_cap:
            continue

        exchange = profile.get("exchange", "") or ""
        exchange = exchange.upper()
        # Normalise exchange names
        if "NASDAQ" in exchange:
            exchange = "NASDAQ"
        elif "NYSE" in exchange:
            exchange = "NYSE"
        elif "TSX" in exchange or "TORONTO" in exchange:
            exchange = "TSX"

        rows.append({
            "ticker": profile.get("symbol", ticker),
            "exchange": exchange,
            "company_name": profile.get("companyName", ""),
            "market_cap_usd": float(mkt_cap),
            "sector": profile.get("sector", ""),
            "industry": profile.get("industry", ""),
            "country": profile.get("country", ""),
        })

    return rows

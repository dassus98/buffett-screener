"""Fetches macro indicators (treasury yields, CPI, GDP growth) from the FRED API.

Primary source: FRED REST API (free, key required via ``FRED_API_KEY`` env var).
Fallback: yfinance tickers ``^TNX`` (treasury) and ``CADUSD=X`` (USD/CAD rate).
Results are cached for 1 day at ``data/raw/macro_data.json``.

Authoritative spec: docs/DATA_SOURCES.md §7 (FRED series reference), §11 (unit conventions).
FRED series IDs and rate limits are read from config/filter_config.yaml at runtime.

Unit conventions (docs/DATA_SOURCES.md §11):
- ``us_treasury_10yr``: raw FRED percent ÷ 100 → stored as decimal (e.g. 0.0425).
- ``usd_cad_rate``: USD per 1 CAD from FRED ``DEXCAUS`` (e.g. 0.74); use directly.
- FRED ``"."`` missing-value markers are replaced with ``float("nan")``.

Data lineage contract
---------------------
Upstream dependencies:
  api_config.py          → fred_limiter, get_fred_key, resilient_request
  filter_config_loader   → get_config (for data_sources.fred.base_url,
                            data_sources.fred.series.treasury_10yr,
                            data_sources.fred.series.usd_cad)

Config dependency map (all from config/filter_config.yaml):
  data_sources.fred.base_url          → _fetch_fred_latest (FRED endpoint URL)
  data_sources.fred.rate_limit_per_min→ fred_limiter (initialised in api_config.py)
  data_sources.fred.series.treasury_10yr → _fetch_all_macro_series (DGS10)
  data_sources.fred.series.goc_10yr  → _fetch_all_macro_series (IRLTLT01CAM156N)
  data_sources.fred.series.usd_cad   → _fetch_all_macro_series (DEXCAUS)

Downstream consumers:
  store.py               → writes macro_data dict as key-value rows to DuckDB
  metrics_engine/        → reads us_treasury_10yr for F14 (DCF discount rate),
                            F16 (earnings yield spread); goc_bond_10yr for TSX
  valuation_reports/     → reads us_treasury_10yr for risk-free rate, usd_cad_rate
                            for CAD conversion, goc_bond_10yr for TSX valuations

Key exports
-----------
fetch_macro_data(use_cache) -> dict
    Fetch (or load from cache) macro indicators.
get_risk_free_rate() -> float
    Return ``us_treasury_10yr`` as a decimal fraction. Used by valuation module.
get_usd_cad_rate() -> float
    Return USD per 1 CAD. Used for CAD → USD currency normalization.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from data_acquisition.api_config import fred_limiter, get_fred_key, resilient_request
from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cache TTL for macro data (1 day).
_MACRO_CACHE_TTL_SECONDS: int = 24 * 60 * 60

#: FRED API sentinel for missing observations.
_FRED_MISSING: str = "."

#: yfinance fallback ticker for US 10-year Treasury yield (in percent).
_YF_TREASURY_TICKER: str = "^TNX"

#: yfinance fallback ticker for CAD/USD rate (USD per 1 CAD).
_YF_USDCAD_TICKER: str = "CADUSD=X"

#: yfinance fallback ticker for GoC 10-year bond (no direct ticker;
#: we accept FRED-only and log a warning if unavailable).
_GOC_YF_FALLBACK: str | None = None  # No reliable yfinance proxy


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def fetch_macro_data(use_cache: bool = True) -> dict[str, Any]:
    """Fetch macro indicators from FRED, returning cached data if fresh.

    Retrieves the following series (FRED primary, yfinance fallback):

    - ``us_treasury_10yr``: 10-year US Treasury yield as a **decimal**
      (e.g. 0.0425). Raw FRED percent is divided by 100 on ingest.
    - ``goc_bond_10yr``: GoC 10-year bond yield as a decimal (FRED only).
    - ``usd_cad_rate``: USD per 1 CAD from FRED ``DEXCAUS`` (e.g. 0.74).
    - ``as_of_date``: ISO 8601 date string when data was fetched.

    Cache location: ``data/raw/macro_data.json``. TTL: 1 day.

    Parameters
    ----------
    use_cache:
        If ``True`` (default) and the cache file is younger than 1 day,
        return cached data without hitting FRED. Set to ``False`` to force
        a fresh fetch regardless of cache state.

    Returns
    -------
    dict
        Keys: ``us_treasury_10yr`` (float or NaN), ``usd_cad_rate``
        (float or NaN), ``as_of_date`` (str).

    Notes
    -----
    - If FRED is unavailable, yfinance fallbacks are tried automatically.
    - If both sources fail for a field, its value is ``float("nan")``.
    - ``float("nan")`` values are serialized as ``null`` in the JSON cache and
      restored as ``float("nan")`` when loaded.
    """
    cache_path = _resolve_macro_cache_path()
    if use_cache and _macro_cache_is_fresh(cache_path):
        logger.info("Loading macro data from cache: %s", cache_path)
        return _load_macro_cache(cache_path)

    logger.info("Fetching fresh macro data from FRED (use_cache=%s).", use_cache)
    result = _fetch_all_macro_series()
    _save_macro_cache(result, cache_path)
    return result


def get_risk_free_rate() -> float:
    """Return the US 10-year Treasury yield as a decimal fraction.

    Fetches macro data (cache-aware) and returns ``us_treasury_10yr``.

    Returns
    -------
    float
        10-year Treasury yield as decimal (e.g. 0.0425). Returns
        ``float("nan")`` if both FRED and yfinance fallback failed.
    """
    data = fetch_macro_data()
    rate = data.get("us_treasury_10yr", float("nan"))
    if rate is None or (isinstance(rate, float) and math.isnan(rate)):
        logger.warning("get_risk_free_rate: us_treasury_10yr unavailable; returning NaN.")
        return float("nan")
    return float(rate)


def get_usd_cad_rate() -> float:
    """Return the current USD per 1 CAD exchange rate.

    Used by the currency normalization step to convert CAD financial statement
    values to USD (docs/DATA_SOURCES.md §5). Sourced from FRED ``DEXCAUS``.

    Returns
    -------
    float
        USD per 1 CAD (e.g. 0.74). Returns ``float("nan")`` if unavailable.
    """
    data = fetch_macro_data()
    rate = data.get("usd_cad_rate", float("nan"))
    if rate is None or (isinstance(rate, float) and math.isnan(rate)):
        logger.warning("get_usd_cad_rate: usd_cad_rate unavailable; returning NaN.")
        return float("nan")
    return float(rate)


# ---------------------------------------------------------------------------
# Internal helpers — FRED fetch
# ---------------------------------------------------------------------------

def _fetch_all_macro_series() -> dict[str, Any]:
    """Fetch all configured macro series and assemble the result dict.

    Returns
    -------
    dict
        Keys: us_treasury_10yr, goc_bond_10yr, usd_cad_rate, as_of_date.
    """
    cfg = get_config()
    series_map: dict[str, str] = (
        cfg.get("data_sources", {}).get("fred", {}).get("series", {})
    )
    treasury_id = series_map.get("treasury_10yr", "DGS10")
    usd_cad_id = series_map.get("usd_cad", "DEXCAUS")
    goc_10yr_id = series_map.get("goc_10yr", "IRLTLT01CAM156N")

    us_10yr = _fetch_fred_latest(treasury_id)
    if us_10yr is not None and not math.isnan(us_10yr):
        us_10yr = us_10yr / 100.0  # percent → decimal (docs/DATA_SOURCES.md §11)
    else:
        us_10yr = _fetch_yf_treasury_fallback()

    usd_cad = _fetch_fred_latest(usd_cad_id)
    if usd_cad is None or math.isnan(usd_cad):
        usd_cad = _fetch_yf_usdcad_fallback()

    # GoC 10-year bond yield (FRED only — no reliable yfinance proxy).
    # Stored as "goc_bond_10yr" to match downstream consumers
    # (metrics_engine, valuation_reports, streamlit_app).
    goc_10yr = _fetch_fred_latest(goc_10yr_id)
    if goc_10yr is not None and not math.isnan(goc_10yr):
        goc_10yr = goc_10yr / 100.0  # percent → decimal
    else:
        logger.warning(
            "GoC 10-year bond yield unavailable from FRED (%s). "
            "TSX valuations will use the 4%% hardcoded fallback.",
            goc_10yr_id,
        )
        goc_10yr = float("nan")

    return {
        "us_treasury_10yr": us_10yr if us_10yr is not None else float("nan"),
        "goc_bond_10yr": goc_10yr,
        "usd_cad_rate": usd_cad if usd_cad is not None else float("nan"),
        "as_of_date": datetime.now(tz=timezone.utc).date().isoformat(),
    }


def _fetch_fred_latest(series_id: str) -> float | None:
    """Fetch the most recent non-missing observation for a FRED series.

    Parameters
    ----------
    series_id:
        FRED series identifier (e.g. ``"DGS10"``).

    Returns
    -------
    float or None
        Most recent valid observation value, or ``None`` on network/key error.
        FRED ``"."`` missing-value markers are skipped to find the latest actual
        reading (up to 10 most recent observations are checked).
    """
    cfg = get_config()
    base_url: str = cfg.get("data_sources", {}).get("fred", {}).get(
        "base_url", "https://api.stlouisfed.org/fred/series/observations"
    )
    try:
        key = get_fred_key()
    except EnvironmentError as exc:
        logger.warning("FRED API key unavailable for series %s: %s", series_id, exc)
        return None

    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": key,
        "sort_order": "desc",
        "limit": 10,
        "file_type": "json",
    }
    try:
        data = resilient_request(base_url, params=params, rate_limiter=fred_limiter)
    except Exception as exc:
        logger.error("FRED fetch failed for series %s: %s", series_id, exc)
        return None

    return _parse_fred_latest(data, series_id)


def _parse_fred_latest(data: Any, series_id: str) -> float | None:
    """Extract the most recent non-missing float from a FRED observations response.

    Parameters
    ----------
    data:
        Parsed JSON from the FRED API. Expected to be a dict with an
        ``"observations"`` list, each entry having a ``"value"`` key.
    series_id:
        FRED series ID, used for log messages.

    Returns
    -------
    float or None
        First numeric value found, or ``None`` if all observations are missing.
    """
    if not isinstance(data, dict):
        logger.error(
            "Unexpected FRED response type for %s: %s", series_id, type(data)
        )
        return None
    observations: list[dict[str, str]] = data.get("observations", [])
    for obs in observations:
        raw_value = obs.get("value", _FRED_MISSING)
        if raw_value == _FRED_MISSING:
            continue
        try:
            return float(raw_value)
        except (ValueError, TypeError):
            continue
    logger.warning(
        "No valid observations found in FRED response for %s.", series_id
    )
    return None


# ---------------------------------------------------------------------------
# Internal helpers — yfinance fallbacks
# ---------------------------------------------------------------------------

def _fetch_yf_treasury_fallback() -> float:
    """Fetch US 10yr Treasury yield from yfinance (^TNX) as a decimal fallback.

    Returns
    -------
    float
        Yield as a decimal (e.g. 0.0425). ``float("nan")`` on failure.
    """
    logger.info("Using yfinance fallback for US Treasury yield (%s).", _YF_TREASURY_TICKER)
    try:
        price = yf.Ticker(_YF_TREASURY_TICKER).fast_info.last_price
        if price is not None:
            return float(price) / 100.0  # ^TNX quotes in percent
    except Exception as exc:
        logger.error("yfinance fallback failed for %s: %s", _YF_TREASURY_TICKER, exc)
    return float("nan")


def _fetch_yf_usdcad_fallback() -> float:
    """Fetch USD/CAD rate from yfinance (CADUSD=X) as a fallback.

    ``CADUSD=X`` quotes the price of 1 CAD in USD (USD per 1 CAD), consistent
    with FRED ``DEXCAUS``. No inversion is required.

    Returns
    -------
    float
        USD per 1 CAD (e.g. 0.74). ``float("nan")`` on failure.
    """
    logger.info("Using yfinance fallback for USD/CAD rate (%s).", _YF_USDCAD_TICKER)
    try:
        price = yf.Ticker(_YF_USDCAD_TICKER).fast_info.last_price
        if price is not None:
            return float(price)
    except Exception as exc:
        logger.error("yfinance fallback failed for %s: %s", _YF_USDCAD_TICKER, exc)
    return float("nan")


# ---------------------------------------------------------------------------
# Internal helpers — cache
# ---------------------------------------------------------------------------

def _resolve_macro_cache_path() -> pathlib.Path:
    """Return the absolute path to the macro data JSON cache file.

    Creates the parent directory if it does not exist.

    Returns
    -------
    pathlib.Path
        ``{project_root}/data/raw/macro_data.json``
    """
    project_root = pathlib.Path(__file__).parent.parent
    cache_dir = project_root / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "macro_data.json"


def _macro_cache_is_fresh(cache_path: pathlib.Path) -> bool:
    """Return True if the cache file exists and is within the 1-day TTL.

    Parameters
    ----------
    cache_path:
        Path to the macro JSON cache file.

    Returns
    -------
    bool
        ``True`` if the file exists and its modification time is within
        ``_MACRO_CACHE_TTL_SECONDS``.
    """
    if not cache_path.exists():
        return False
    age = datetime.now(tz=timezone.utc).timestamp() - cache_path.stat().st_mtime
    if age <= _MACRO_CACHE_TTL_SECONDS:
        logger.debug("Macro cache is fresh (age=%.0fs): %s", age, cache_path)
        return True
    logger.info("Macro cache is stale (age=%.0fs): %s", age, cache_path)
    return False


def _load_macro_cache(cache_path: pathlib.Path) -> dict[str, Any]:
    """Load and return the macro data dict from the JSON cache file.

    ``null`` values in JSON are restored as ``float("nan")``.

    Parameters
    ----------
    cache_path:
        Path to the macro JSON cache file.

    Returns
    -------
    dict
        Macro data with NaN restored for missing numeric fields.
    """
    with cache_path.open("r") as fh:
        raw: dict[str, Any] = json.load(fh)
    # Restore None (JSON null) → float("nan") for numeric fields.
    for key in ("us_treasury_10yr", "goc_bond_10yr", "usd_cad_rate"):
        if raw.get(key) is None:
            raw[key] = float("nan")
    return raw


def _save_macro_cache(data: dict[str, Any], cache_path: pathlib.Path) -> None:
    """Persist macro data dict to the JSON cache file.

    ``float("nan")`` values are serialized as JSON ``null`` because JSON does
    not support NaN. They are restored as ``float("nan")`` by ``_load_macro_cache``.

    Parameters
    ----------
    data:
        Macro data dict to persist.
    cache_path:
        Destination path for the JSON file.
    """
    serialisable = {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in data.items()
    }
    try:
        with cache_path.open("w") as fh:
            json.dump(serialisable, fh, indent=2)
        logger.info("Macro data cached to %s.", cache_path)
    except Exception as exc:
        logger.error("Failed to write macro cache to %s: %s", cache_path, exc)

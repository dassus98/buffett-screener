"""API key management, rate limiting, and resilient HTTP request logic for all external
data sources used by the buffett-screener pipeline.

Authoritative spec: docs/DATA_SOURCES.md §1–2.
Rate limit values and base URLs come from config/filter_config.yaml at import time.

Note on yaml loading
--------------------
This is the ONE module in the codebase that calls ``yaml.safe_load`` directly. It is the
lowest-level config consumer and must not depend on ``screener/filter_config_loader.py``
(which would create a circular-dependency path). All other modules must load config
exclusively via ``filter_config_loader``.

Key exports
-----------
RateLimiter
    Sliding-window, thread-safe rate limiter. Constructed from a requests-per-minute cap.
resilient_request(url, params, rate_limiter, max_retries) -> dict
    GET with exponential backoff on 429 / 5xx. Raises on 4xx.
fmp_limiter : RateLimiter
    Preconfigured for FMP at the rate in filter_config.yaml.
fred_limiter : RateLimiter
    Preconfigured for FRED at the rate in filter_config.yaml.
build_fmp_url(endpoint, **params) -> tuple[str, dict]
    Construct a full FMP URL with API key injected into params.
get_fmp_key() -> str
    Return the loaded FMP API key (raises if not set).
get_fred_key() -> str
    Return the loaded FRED API key (raises if not set).
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading
import time
from collections import deque
from typing import Any
from urllib.parse import urljoin

import requests
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
# Walk up from this file to find the project root .env.  This keeps the module
# importable from any working directory.
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_fmp_key() -> str:
    """Return the FMP API key loaded from the .env file.

    Returns
    -------
    str
        The value of ``FMP_API_KEY`` from the environment.

    Raises
    ------
    EnvironmentError
        If ``FMP_API_KEY`` is not set or is empty. Add it to your .env file:
        ``FMP_API_KEY=your_key_here``
    """
    key = os.getenv("FMP_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "FMP_API_KEY is not set. "
            "Add FMP_API_KEY=<your_key> to the .env file at the project root. "
            "Get a key at https://financialmodelingprep.com/developer/docs"
        )
    return key


def get_fred_key() -> str:
    """Return the FRED API key loaded from the .env file.

    Returns
    -------
    str
        The value of ``FRED_API_KEY`` from the environment.

    Raises
    ------
    EnvironmentError
        If ``FRED_API_KEY`` is not set or is empty. Add it to your .env file:
        ``FRED_API_KEY=your_key_here``
    """
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY is not set. "
            "Add FRED_API_KEY=<your_key> to the .env file at the project root. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


# ---------------------------------------------------------------------------
# Config loading (direct yaml — see module docstring for rationale)
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    """Load filter_config.yaml from the project root config/ directory.

    Returns
    -------
    dict
        The full parsed YAML document.

    Raises
    ------
    FileNotFoundError
        If config/filter_config.yaml does not exist at the expected path.
    """
    config_path = _PROJECT_ROOT / "config" / "filter_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"filter_config.yaml not found at {config_path}. "
            "Run from the project root or ensure config/ is present."
        )
    with config_path.open("r") as fh:
        return yaml.safe_load(fh)


_CONFIG: dict[str, Any] = _load_config()
_DATA_SOURCES: dict[str, Any] = _CONFIG.get("data_sources", {})


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Tracks the timestamps of recent requests in a deque. Before allowing a new
    request, it inspects how many requests occurred within the last 60 seconds
    and sleeps for the minimum time needed to drop below the cap.

    Parameters
    ----------
    max_requests_per_minute : int
        Maximum number of requests allowed in any 60-second sliding window.
        Read from ``config/filter_config.yaml`` for the preconfigured instances.

    Examples
    --------
    >>> limiter = RateLimiter(max_requests_per_minute=300)
    >>> limiter.wait_if_needed()   # blocks if at the limit; returns immediately otherwise
    """

    _WINDOW_SECONDS: float = 60.0

    def __init__(self, max_requests_per_minute: int) -> None:
        if max_requests_per_minute <= 0:
            raise ValueError(
                f"max_requests_per_minute must be positive, got {max_requests_per_minute}"
            )
        self._max: int = max_requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock: threading.Lock = threading.Lock()

    @property
    def max_requests_per_minute(self) -> int:
        """The configured maximum requests per 60-second window."""
        return self._max

    def wait_if_needed(self) -> None:
        """Block until a request slot is available within the rate limit window.

        Removes timestamps older than 60 seconds, then sleeps if the window is
        full. After sleeping (if any), records the current timestamp and returns.

        This method is thread-safe: concurrent callers share the same deque and
        lock, so no two threads will be granted the same slot.

        Side effects
        ------------
        Appends the current monotonic timestamp to the internal deque once a
        slot is available.
        """
        with self._lock:
            while True:
                now = time.monotonic()
                # Drop timestamps outside the sliding window.
                cutoff = now - self._WINDOW_SECONDS
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self._max:
                    # Slot available — record and proceed.
                    self._timestamps.append(now)
                    return

                # Window is full. Sleep until the oldest request ages out.
                oldest = self._timestamps[0]
                sleep_for = (oldest + self._WINDOW_SECONDS) - now
                if sleep_for > 0:
                    logger.debug(
                        "RateLimiter: window full (%d/%d). Sleeping %.3fs.",
                        len(self._timestamps),
                        self._max,
                        sleep_for,
                    )
                    # Release the lock while sleeping so other threads can check.
                    self._lock.release()
                    try:
                        time.sleep(sleep_for)
                    finally:
                        self._lock.acquire()
                # Re-evaluate after waking.


# ---------------------------------------------------------------------------
# resilient_request
# ---------------------------------------------------------------------------

#: HTTP status codes that warrant a retry with exponential backoff.
_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Seconds to wait before each retry attempt (index = attempt number, 0-based).
_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0)

#: Default per-request timeout in seconds.
_REQUEST_TIMEOUT: int = 30


def resilient_request(
    url: str,
    params: dict[str, Any] | None = None,
    rate_limiter: RateLimiter | None = None,
    max_retries: int = 3,
) -> dict[str, Any] | list[Any]:
    """Make a GET request with rate limiting and exponential-backoff retry logic.

    Parameters
    ----------
    url:
        Full URL to request (must not contain the API key in the URL string;
        pass it via ``params`` to keep it out of log messages).
    params:
        Query-string parameters, including any API key. The key is stripped from
        log messages automatically.
    rate_limiter:
        Optional :class:`RateLimiter` instance. If provided,
        ``wait_if_needed()`` is called before every attempt (including retries).
    max_retries:
        Maximum number of retry attempts after the initial request failure.
        Total attempts = ``max_retries + 1``. Read from
        ``config/filter_config.yaml`` ``data_sources.retry.max_attempts`` for
        the default; callers may override.

    Returns
    -------
    dict or list
        Parsed JSON response body.

    Raises
    ------
    requests.HTTPError
        For 4xx responses other than 429 (not retried).
    requests.HTTPError
        If all retry attempts are exhausted on 429 / 5xx.
    requests.RequestException
        For connection errors, timeouts, or other transport failures after all
        retries are exhausted.

    Notes
    -----
    - URL is logged at DEBUG level with the ``apikey`` param redacted.
    - Every retry is logged at WARNING level with the attempt number and status.
    - Backoff schedule: 1 s → 2 s → 4 s (configurable via ``_BACKOFF_SECONDS``).
    """
    params = params or {}
    safe_url = _redact_key(url, params)
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        if rate_limiter is not None:
            rate_limiter.wait_if_needed()

        logger.debug("GET %s (attempt %d/%d)", safe_url, attempt + 1, total_attempts)

        try:
            response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            if attempt < max_retries:
                wait = _backoff_wait(attempt)
                logger.warning(
                    "Request error on attempt %d/%d for %s: %s. Retrying in %.1fs.",
                    attempt + 1,
                    total_attempts,
                    safe_url,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue
            logger.error(
                "Request failed after %d attempts for %s: %s",
                total_attempts,
                safe_url,
                exc,
            )
            raise

        if response.status_code == 200:
            return response.json()

        if response.status_code in _RETRY_STATUSES:
            if attempt < max_retries:
                wait = _backoff_wait(attempt)
                logger.warning(
                    "HTTP %d on attempt %d/%d for %s. Retrying in %.1fs.",
                    response.status_code,
                    attempt + 1,
                    total_attempts,
                    safe_url,
                    wait,
                )
                time.sleep(wait)
                continue
            # Exhausted retries.
            logger.error(
                "HTTP %d — all %d attempts exhausted for %s.",
                response.status_code,
                total_attempts,
                safe_url,
            )
            response.raise_for_status()

        # 4xx (other than 429): do not retry.
        logger.error(
            "HTTP %d — not retrying for %s. Response: %.200s",
            response.status_code,
            safe_url,
            response.text,
        )
        response.raise_for_status()

    # Unreachable, but satisfies type checkers.
    raise RuntimeError(f"resilient_request exhausted without returning for {safe_url}")


def _backoff_wait(attempt: int) -> float:
    """Return the backoff sleep duration for a given attempt index (0-based).

    Parameters
    ----------
    attempt:
        Zero-based attempt index (0 = first retry).

    Returns
    -------
    float
        Seconds to sleep before the next attempt.
    """
    idx = min(attempt, len(_BACKOFF_SECONDS) - 1)
    return _BACKOFF_SECONDS[idx]


def _redact_key(url: str, params: dict[str, Any]) -> str:
    """Return a log-safe URL string with the API key value replaced by '***'.

    Parameters
    ----------
    url:
        The base URL (may or may not contain 'apikey' as a path segment).
    params:
        Query-string parameters dict; 'apikey' value is redacted if present.

    Returns
    -------
    str
        URL string safe to write to log output.
    """
    if not params:
        return url
    safe_params = {
        k: ("***" if k.lower() in {"apikey", "api_key"} else v)
        for k, v in params.items()
    }
    param_str = "&".join(f"{k}={v}" for k, v in safe_params.items())
    return f"{url}?{param_str}"


# ---------------------------------------------------------------------------
# build_fmp_url
# ---------------------------------------------------------------------------

def build_fmp_url(endpoint: str, **params: Any) -> tuple[str, dict[str, Any]]:
    """Construct a fully-qualified FMP API URL with the API key in the params dict.

    The API key is placed in ``params`` (not the URL path) so that log messages
    produced by :func:`resilient_request` never expose the key.

    Parameters
    ----------
    endpoint:
        Path segment starting with ``/``, e.g. ``"/income-statement/AAPL"``.
    **params:
        Additional query-string parameters (e.g. ``period="annual"``,
        ``limit=10``). These are merged with the ``apikey`` entry.

    Returns
    -------
    tuple[str, dict]
        ``(url, params_with_apikey)`` where ``url`` is the full base URL plus
        endpoint path and ``params_with_apikey`` is the params dict including
        the ``"apikey"`` key.

    Examples
    --------
    >>> url, p = build_fmp_url("/income-statement/AAPL", period="annual", limit=10)
    >>> url
    'https://financialmodelingprep.com/api/v3/income-statement/AAPL'
    >>> p["period"]
    'annual'
    >>> "apikey" in p
    True
    """
    base_url: str = _DATA_SOURCES.get("fmp", {}).get(
        "base_url", "https://financialmodelingprep.com/api/v3"
    )
    # urljoin handles trailing/leading slashes correctly.
    full_url = base_url.rstrip("/") + endpoint
    params_with_key: dict[str, Any] = {"apikey": get_fmp_key(), **params}
    return full_url, params_with_key


# ---------------------------------------------------------------------------
# Preconfigured limiter instances
# ---------------------------------------------------------------------------

def _fmp_rate_limit() -> int:
    """Read FMP rate limit from config; fall back to 300 if missing."""
    return int(_DATA_SOURCES.get("fmp", {}).get("rate_limit_per_min", 300))


def _fred_rate_limit() -> int:
    """Read FRED rate limit from config; fall back to 120 if missing."""
    return int(_DATA_SOURCES.get("fred", {}).get("rate_limit_per_min", 120))


#: RateLimiter for Financial Modeling Prep API.
#: Initialized at import time from ``data_sources.fmp.rate_limit_per_min`` in
#: ``config/filter_config.yaml`` (default: 300 req/min for FMP Starter tier).
fmp_limiter: RateLimiter = RateLimiter(max_requests_per_minute=_fmp_rate_limit())

#: RateLimiter for FRED API.
#: Initialized at import time from ``data_sources.fred.rate_limit_per_min`` in
#: ``config/filter_config.yaml`` (default: 120 req/min).
fred_limiter: RateLimiter = RateLimiter(max_requests_per_minute=_fred_rate_limit())

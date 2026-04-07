"""
data_acquisition.api_config
============================
Centralised API configuration loader.

Reads credentials from environment variables (via python-dotenv) and exposes
typed config objects consumed by each data-source module.  No credentials
should ever be hardcoded; they must come exclusively from the .env file.

Environment variables expected:
    FRED_API_KEY        — Federal Reserve Economic Data API key
    SEC_USER_AGENT      — "FirstName LastName email@example.com" (EDGAR policy)
    FMP_API_KEY         — Financial Modeling Prep (optional fallback source)
"""

from __future__ import annotations

import functools
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(max_attempts: int = 3, backoff_base: float = 2.0) -> Callable[[F], F]:
    """
    Module-level retry decorator with exponential back-off.

    Args:
        max_attempts: Maximum number of total attempts (including the first).
        backoff_base: Base for exponential back-off; sleep = backoff_base ** attempt.

    Returns:
        Decorator that wraps a function with retry logic.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        sleep_secs = backoff_base ** attempt
                        logger.warning(
                            "Attempt %d/%d for %s failed (%s). Retrying in %.1fs.",
                            attempt + 1,
                            max_attempts,
                            func.__qualname__,
                            exc,
                            sleep_secs,
                        )
                        time.sleep(sleep_secs)
                    else:
                        logger.error(
                            "All %d attempts for %s failed. Last error: %s",
                            max_attempts,
                            func.__qualname__,
                            exc,
                        )
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


def load_env(env_path: Path | None = None) -> None:
    """
    Load environment variables from a .env file into os.environ.

    Args:
        env_path: Explicit path to the .env file. If None, searches upward
                  from the current working directory until a .env is found
                  (standard python-dotenv behaviour).

    Returns:
        None. Side-effect: os.environ is populated.

    Logic:
        Call dotenv.load_dotenv(env_path, override=False) so that variables
        already set in the shell environment take precedence over the file.
        Raise EnvironmentError if no .env file is found and env_path was
        explicitly provided.
    """
    if env_path is not None:
        resolved = Path(env_path).resolve()
        if not resolved.exists():
            raise EnvironmentError(
                f"Explicit .env path not found: {resolved}. "
                "Check the path passed to load_env()."
            )
        loaded = load_dotenv(dotenv_path=resolved, override=False)
        logger.info("load_dotenv(%s) → loaded=%s", resolved, loaded)
    else:
        loaded = load_dotenv(override=False)
        logger.info("load_dotenv(auto-search) → loaded=%s", loaded)


@dataclass(frozen=True)
class FredConfig:
    """
    Configuration for the FRED (Federal Reserve Economic Data) API.

    Attributes:
        api_key:  FRED API key (loaded from FRED_API_KEY env var).
        base_url: FRED REST API base URL.
        timeout:  HTTP request timeout in seconds.
    """

    api_key: str
    base_url: str = "https://api.stlouisfed.org/fred"
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "FredConfig":
        """
        Construct FredConfig by reading FRED_API_KEY from os.environ.

        Returns:
            FredConfig instance.

        Raises:
            EnvironmentError: if FRED_API_KEY is not set.
        """
        load_env()
        api_key = os.environ.get("FRED_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "FRED_API_KEY is not set or is empty. "
                "Add it to your .env file: FRED_API_KEY=your_key_here"
            )
        logger.debug("FredConfig loaded from environment (key length=%d).", len(api_key))
        return cls(api_key=api_key)


@dataclass(frozen=True)
class SecEdgarConfig:
    """
    Configuration for the SEC EDGAR REST API.

    SEC requires a User-Agent header of the form:
        "CompanyName Contact email@example.com"
    Requests without a valid User-Agent may be rate-limited or blocked.

    Attributes:
        user_agent: Full User-Agent string (loaded from SEC_USER_AGENT env var).
        base_url:   EDGAR REST API base URL.
        timeout:    HTTP request timeout in seconds.
        rate_limit_rps: Maximum requests per second to respect EDGAR's 10 req/s limit.
    """

    user_agent: str
    base_url: str = "https://data.sec.gov"
    timeout: int = 30
    rate_limit_rps: float = 8.0         # stay comfortably under SEC's 10 req/s cap

    @classmethod
    def from_env(cls) -> "SecEdgarConfig":
        """
        Construct SecEdgarConfig by reading SEC_USER_AGENT from os.environ.

        Returns:
            SecEdgarConfig instance.

        Raises:
            EnvironmentError: if SEC_USER_AGENT is not set or does not look
                              like a valid "Name email" string.
        """
        load_env()
        user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
        if not user_agent:
            raise EnvironmentError(
                "SEC_USER_AGENT is not set or is empty. "
                "Add it to your .env file: SEC_USER_AGENT=YourName contact@example.com"
            )
        if "@" not in user_agent:
            raise EnvironmentError(
                f"SEC_USER_AGENT does not appear to contain an email address: "
                f"'{user_agent}'. SEC EDGAR requires a valid contact email in the "
                "User-Agent string (e.g. 'MyApp contact@example.com')."
            )
        logger.debug("SecEdgarConfig loaded from environment (user_agent=%r).", user_agent)
        return cls(user_agent=user_agent)


@dataclass(frozen=True)
class FmpConfig:
    """
    Configuration for Financial Modeling Prep API (optional enrichment source).

    Attributes:
        api_key:  FMP API key (loaded from FMP_API_KEY env var).
        base_url: FMP REST API base URL.
        timeout:  HTTP request timeout in seconds.
    """

    api_key: str
    base_url: str = "https://financialmodelingprep.com/api/v3"
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "FmpConfig":
        """
        Construct FmpConfig by reading FMP_API_KEY from os.environ.

        Returns:
            FmpConfig instance, or raises EnvironmentError if key is absent.

        Logic:
            FMP is an optional source; callers should catch EnvironmentError
            and fall back gracefully rather than failing the pipeline.
        """
        load_env()
        api_key = os.environ.get("FMP_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "FMP_API_KEY is not set or is empty. "
                "Add it to your .env file: FMP_API_KEY=your_key_here  "
                "(FMP is optional; the pipeline will continue without it.)"
            )
        logger.debug("FmpConfig loaded from environment (key length=%d).", len(api_key))
        return cls(api_key=api_key)


def get_all_configs() -> dict[str, object]:
    """
    Load and return all API config objects in a single dictionary.

    Args:
        None

    Returns:
        dict with keys "fred", "sec", "fmp" mapped to their respective
        config dataclass instances. FmpConfig is included only if
        FMP_API_KEY is present in the environment.

    Logic:
        Call load_env() first, then instantiate each config via from_env().
        Suppress EnvironmentError only for FmpConfig (optional dependency).
        Re-raise EnvironmentError for FRED and SEC configs (required).
    """
    load_env()

    configs: dict[str, object] = {}

    # Required configs — propagate EnvironmentError to caller
    configs["fred"] = FredConfig.from_env()
    logger.info("FredConfig loaded successfully.")

    configs["sec"] = SecEdgarConfig.from_env()
    logger.info("SecEdgarConfig loaded successfully.")

    # Optional config — swallow error and log a warning
    try:
        configs["fmp"] = FmpConfig.from_env()
        logger.info("FmpConfig loaded successfully.")
    except EnvironmentError as exc:
        logger.warning(
            "FmpConfig not available (FMP_API_KEY missing or invalid): %s. "
            "FMP fallback disabled.",
            exc,
        )

    return configs

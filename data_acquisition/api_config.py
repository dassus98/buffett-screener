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

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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
    ...


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

        Logic:
            Call load_env() to ensure the .env file has been loaded, then
            read os.environ["FRED_API_KEY"]. Raise a descriptive error if
            the key is missing or empty.
        """
        ...


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

        Logic:
            Validate the user_agent string contains an "@" (basic email check).
            Raise a descriptive error pointing at .env if validation fails.
        """
        ...


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
        ...


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
    ...

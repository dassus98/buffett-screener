"""
screener.exclusions
====================
Removes structurally ineligible tickers from the universe before any
financial analysis is performed.

Exclusion categories:
    1. Sector exclusions (Financials, Utilities, Real Estate by default)
       — these sectors require fundamentally different analytical frameworks
    2. ADR flag  — foreign-listed ADRs (optional; configurable)
    3. SPAC flag — Special Purpose Acquisition Companies (blank-check)
    4. Shell companies — no real operating business
    5. Market cap below minimum (pre-filter before expensive API calls)

All exclusions are driven by the "exclusions" section of filter_config.yaml.
"""

from __future__ import annotations

import pandas as pd


def apply_exclusions(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Remove all structurally excluded tickers from the universe DataFrame.

    Args:
        df:     Universe DataFrame with at minimum these columns:
                    ticker, sector, is_adr (bool), is_spac (bool),
                    market_cap (float, USD)
        config: Full filter_config.yaml dict. Reads "exclusions" and
                "universe.min_market_cap_usd" sections.

    Returns:
        Filtered DataFrame with excluded rows removed. Adds a column
        "exclusion_reason" to the *excluded* rows (returned separately
        via the also-returned exclusions_log DataFrame).

    Logic:
        1. Build exclusion mask for each category
        2. Apply sector exclusion: df["sector"].isin(config["exclusions"]["sectors"])
        3. Apply ADR flag: df["is_adr"] == True (if "ADR" in config["exclusions"]["flags"])
        4. Apply SPAC flag: df["is_spac"] == True (if "SPAC" in config["exclusions"]["flags"])
        5. Apply market cap floor: df["market_cap"] < min_market_cap_usd
        6. Return (passing_df, excluded_df_with_reason)
    """
    ...


def is_financial_sector(sector: str) -> bool:
    """
    Check whether a sector string maps to the Financials sector.

    Args:
        sector: Sector string as returned by yfinance (e.g. "Financial Services").

    Returns:
        True if the sector should be excluded as a financial company.

    Logic:
        Match against known financial sector labels (case-insensitive):
            "Financials", "Financial Services", "Banks", "Insurance",
            "Asset Management", "Capital Markets", "Mortgage Finance"
    """
    ...


def is_excluded_sector(sector: str, excluded_sectors: list[str]) -> bool:
    """
    Check whether a sector matches any entry in the excluded sectors list.

    Args:
        sector:           Sector string from company profile.
        excluded_sectors: List of sector strings from config["exclusions"]["sectors"].

    Returns:
        True if the sector matches any exclusion (case-insensitive partial match).

    Logic:
        Lower-case both sides and check if any excluded_sector is a substring
        of the company's sector string. This handles slight name variations
        (e.g. "Real Estate" matching "Real Estate Investment Trusts").
    """
    ...


def flag_shell_companies(df: pd.DataFrame) -> pd.Series:
    """
    Produce a boolean mask identifying probable shell companies.

    Args:
        df: Universe DataFrame with columns: revenue (float), total_assets (float),
            employees (int, optional).

    Returns:
        Boolean pandas Series (True = likely shell company).

    Logic:
        A company is flagged as a probable shell if:
            - revenue < $1M (USD thousands: revenue < 1000)
            - AND total_assets < $10M (USD thousands: total_assets < 10000)
        This is a heuristic; false positives should be reviewed manually.
    """
    ...

"""Removes structurally ineligible tickers before financial analysis.

Exclusions are applied BEFORE Tier 1 hard filters and Tier 2 soft scoring.
Three mechanisms, all config-driven via ``config/filter_config.yaml``:

1. **SIC code ranges** — removes banks, insurers, REITs, holding cos.
2. **Sector-level exclusion** — removes entire GICS sectors (if configured).
3. **Industry name patterns** — catches financials when SIC data is
   unavailable (FMP provides industry strings but not SIC codes).
4. **Flags** — removes SPACs, shell companies (via ``is_<flag>`` columns).

A ticker is excluded if it matches ANY mechanism.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_sic_set(sic_ranges: list[list[int]]) -> set[int]:
    """Expand a list of ``[start, end]`` inclusive ranges into a flat set.

    Parameters
    ----------
    sic_ranges:
        Each element is a two-element list ``[lo, hi]`` read from
        ``config.exclusions.sic_codes``.

    Returns
    -------
    set[int]
        Every integer in the union of all ranges.
    """
    codes: set[int] = set()
    for pair in sic_ranges:
        if len(pair) == 2:
            codes.update(range(int(pair[0]), int(pair[1]) + 1))
    return codes


def _check_sic_exclusion(
    df: pd.DataFrame,
    sic_set: set[int],
) -> pd.Series:
    """Return boolean mask — ``True`` where ticker is excluded by SIC code.

    Parameters
    ----------
    df:
        Universe DataFrame; may or may not contain a ``sic_code`` column.
    sic_set:
        Set of excluded SIC codes from :func:`_build_sic_set`.

    Returns
    -------
    pd.Series[bool]
        ``True`` for rows whose ``sic_code`` falls inside *sic_set*.
        All-``False`` if column is absent or *sic_set* is empty.
    """
    if "sic_code" not in df.columns or not sic_set:
        return pd.Series(False, index=df.index, dtype=bool)
    sic = pd.to_numeric(df["sic_code"], errors="coerce")
    return sic.isin(sic_set)


def _check_sector_exclusion(
    df: pd.DataFrame,
    excluded_sectors: list[str],
) -> pd.Series:
    """Return boolean mask — ``True`` where ticker is excluded by sector.

    Parameters
    ----------
    df:
        Universe DataFrame.
    excluded_sectors:
        Sector names from ``config.exclusions.sectors`` (case-insensitive).

    Returns
    -------
    pd.Series[bool]
    """
    if "sector" not in df.columns or not excluded_sectors:
        return pd.Series(False, index=df.index, dtype=bool)
    sector_lower = df["sector"].astype(str).str.lower()
    excluded_lower = {s.lower() for s in excluded_sectors}
    return sector_lower.isin(excluded_lower)


def _check_industry_pattern_exclusion(
    df: pd.DataFrame,
    keywords: list[str],
    financial_sector_label: str,
) -> pd.Series:
    """Return boolean mask — ``True`` where industry keyword matches.

    Exclusion fires only when BOTH conditions hold:

    * ``industry`` contains one of *keywords* (case-insensitive substring).
    * ``sector`` equals *financial_sector_label*.

    Parameters
    ----------
    df:
        Universe DataFrame.
    keywords:
        Industry keyword strings from ``config.exclusions.industry_keywords``.
    financial_sector_label:
        Sector value that must match for the exclusion to apply
        (from ``config.exclusions.financial_sector_label``).

    Returns
    -------
    pd.Series[bool]
    """
    if "industry" not in df.columns or "sector" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    if not keywords:
        return pd.Series(False, index=df.index, dtype=bool)

    industry = df["industry"].astype(str)
    sector = df["sector"].astype(str)

    is_financial_sector = sector == financial_sector_label
    keyword_match = pd.Series(False, index=df.index, dtype=bool)
    for kw in keywords:
        keyword_match = keyword_match | industry.str.contains(
            kw, case=False, na=False,
        )

    return is_financial_sector & keyword_match


def _check_flag_exclusion(
    df: pd.DataFrame,
    flags: list[str],
) -> pd.Series:
    """Return boolean mask — ``True`` where a flag column indicates exclusion.

    Checks for boolean columns named ``is_<flag>`` (e.g. ``is_SPAC``).

    Parameters
    ----------
    df:
        Universe DataFrame.
    flags:
        Flag names from ``config.exclusions.flags``.

    Returns
    -------
    pd.Series[bool]
    """
    mask = pd.Series(False, index=df.index, dtype=bool)
    for flag in flags:
        col = f"is_{flag}"
        if col in df.columns:
            mask = mask | df[col].astype(bool)
    return mask


def _build_exclusion_log(
    df: pd.DataFrame,
    sic_mask: pd.Series,
    sector_mask: pd.Series,
    industry_mask: pd.Series,
    flag_mask: pd.Series,
) -> pd.DataFrame:
    """Build a log DataFrame with one row per excluded ticker and reason.

    Parameters
    ----------
    df:
        Universe DataFrame (must contain ``ticker`` column).
    sic_mask, sector_mask, industry_mask, flag_mask:
        Boolean masks from each exclusion check.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``reason``.  Empty (zero rows, two columns)
        when nothing is excluded.
    """
    records: list[dict[str, str]] = []
    for idx in df.index:
        ticker = str(df.at[idx, "ticker"])
        reasons: list[str] = []
        if sic_mask.at[idx]:
            reasons.append("sic_code_excluded")
        if sector_mask.at[idx]:
            reasons.append("sector_excluded")
        if industry_mask.at[idx]:
            reasons.append("industry_pattern_excluded")
        if flag_mask.at[idx]:
            reasons.append("flag_excluded")
        if reasons:
            records.append({"ticker": ticker, "reason": "; ".join(reasons)})

    if not records:
        return pd.DataFrame(columns=["ticker", "reason"])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_exclusions(
    universe_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove structurally ineligible tickers from the screening universe.

    Reads exclusion rules from ``config/filter_config.yaml`` (section
    ``exclusions``) and removes tickers matching SIC code ranges,
    sector names, financial-industry keyword patterns, or company-type
    flags (SPAC, shell company).

    Parameters
    ----------
    universe_df:
        Full universe DataFrame.  Expected columns: ``ticker``, ``sector``,
        ``industry``.  Optional columns: ``sic_code``, ``is_SPAC``,
        ``is_shell_company``.  Missing columns cause the corresponding
        exclusion mechanism to be silently skipped.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(filtered_df, exclusion_log_df)``

        *filtered_df*: subset of *universe_df* with excluded rows removed
        (index reset).

        *exclusion_log_df*: one row per excluded ticker.  Columns:
        ``ticker``, ``reason`` (semicolon-separated if multiple reasons).
    """
    if universe_df.empty:
        logger.warning("apply_exclusions: received empty DataFrame.")
        return pd.DataFrame(), pd.DataFrame(columns=["ticker", "reason"])

    cfg = get_config()
    excl_cfg: dict[str, Any] = cfg.get("exclusions", {})

    # Read config values
    sic_ranges: list[list[int]] = excl_cfg.get("sic_codes", [])
    sic_set = _build_sic_set(sic_ranges)
    excluded_sectors: list[str] = excl_cfg.get("sectors", [])
    keywords: list[str] = excl_cfg.get("industry_keywords", [])
    financial_sector_label: str = excl_cfg.get(
        "financial_sector_label", "Financial Services",
    )
    flags: list[str] = excl_cfg.get("flags", [])

    # Evaluate each exclusion mechanism
    sic_mask = _check_sic_exclusion(universe_df, sic_set)
    sector_mask = _check_sector_exclusion(universe_df, excluded_sectors)
    industry_mask = _check_industry_pattern_exclusion(
        universe_df, keywords, financial_sector_label,
    )
    flag_mask = _check_flag_exclusion(universe_df, flags)

    combined_mask = sic_mask | sector_mask | industry_mask | flag_mask

    exclusion_log = _build_exclusion_log(
        universe_df, sic_mask, sector_mask, industry_mask, flag_mask,
    )

    filtered_df = universe_df.loc[~combined_mask].reset_index(drop=True)

    n_excluded = int(combined_mask.sum())
    logger.info(
        "Exclusions: %d of %d tickers removed (%d remain).",
        n_excluded,
        len(universe_df),
        len(filtered_df),
    )

    return filtered_df, exclusion_log

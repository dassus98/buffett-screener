"""Removes structurally ineligible tickers before financial analysis.

Exclusions are applied BEFORE Tier 1 hard filters and Tier 2 soft scoring.
Four mechanisms, all config-driven via ``config/filter_config.yaml``:

1. **SIC code ranges** — removes banks, insurers, REITs, holding cos.
2. **Sector-level exclusion** — removes entire GICS sectors (if configured).
3. **Industry name patterns** — catches financials when SIC data is
   unavailable (FMP provides industry strings but not SIC codes).
4. **Flags** — removes SPACs, shell companies (via ``is_<flag>`` columns).

A ticker is excluded if it matches ANY mechanism.

Data Lineage Contract
---------------------
Upstream producers:
    - ``data_acquisition.universe.get_universe`` → ``universe_df`` with
      columns ``ticker``, ``sector``, ``industry``, and optionally
      ``sic_code``, ``is_SPAC``, ``is_shell_company``.

Downstream consumers:
    - ``screener.hard_filters.apply_hard_filters`` — receives
      ``filtered_df`` (universe minus excluded tickers).
    - ``screener.composite_ranker.generate_screener_summary`` — receives
      ``exclusion_log_df`` for pipeline statistics.

Config dependencies (all via ``get_threshold``):
    - ``exclusions.sic_codes``              (list of [lo, hi] SIC ranges)
    - ``exclusions.sectors``                (list of GICS sectors)
    - ``exclusions.industry_keywords``      (list of keyword strings)
    - ``exclusions.financial_sector_label``  (sector for keyword matching)
    - ``exclusions.flags``                  (list of flag names)
"""

from __future__ import annotations

import logging

import pandas as pd

from screener.filter_config_loader import get_threshold

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
    # --- Expand each [lo, hi] pair into a contiguous range of integers.
    #     Config example: [6020, 6029] → {6020, 6021, ..., 6029}.
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
    # --- Guard: skip when the column is absent or no SIC codes to check.
    if "sic_code" not in df.columns or not sic_set:
        return pd.Series(False, index=df.index, dtype=bool)
    # --- Coerce to numeric so non-integer SIC values become NaN (no match).
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
    # --- Guard: skip when column or exclusion list is absent.
    if "sector" not in df.columns or not excluded_sectors:
        return pd.Series(False, index=df.index, dtype=bool)
    # --- Case-insensitive comparison for robust matching.
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
    # --- Guard: skip when required columns or keyword list are absent.
    if "industry" not in df.columns or "sector" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    if not keywords:
        return pd.Series(False, index=df.index, dtype=bool)

    industry = df["industry"].astype(str)
    sector = df["sector"].astype(str)

    # --- Both conditions must be true: sector matches the financial
    #     sector label AND industry contains at least one keyword.
    #     This prevents excluding e.g. "Data Banks & Storage" in Tech.
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
    # --- Step 1: Guard — empty input produces empty output.
    if universe_df.empty:
        logger.warning("apply_exclusions: received empty DataFrame.")
        return pd.DataFrame(), pd.DataFrame(columns=["ticker", "reason"])

    # --- Step 2: Read all exclusion config via get_threshold (fail-fast).
    sic_ranges: list[list[int]] = get_threshold("exclusions.sic_codes")
    sic_set = _build_sic_set(sic_ranges)
    excluded_sectors: list[str] = get_threshold("exclusions.sectors")
    keywords: list[str] = get_threshold("exclusions.industry_keywords")
    financial_sector_label: str = get_threshold(
        "exclusions.financial_sector_label",
    )
    flags: list[str] = get_threshold("exclusions.flags")

    # --- Step 3: Evaluate each exclusion mechanism independently.
    #     A ticker is excluded if ANY mechanism matches.
    sic_mask = _check_sic_exclusion(universe_df, sic_set)
    sector_mask = _check_sector_exclusion(universe_df, excluded_sectors)
    industry_mask = _check_industry_pattern_exclusion(
        universe_df, keywords, financial_sector_label,
    )
    flag_mask = _check_flag_exclusion(universe_df, flags)

    # --- Step 4: Combine all masks with OR (any match = excluded).
    combined_mask = sic_mask | sector_mask | industry_mask | flag_mask

    # --- Step 5: Build the exclusion log (one row per excluded ticker).
    exclusion_log = _build_exclusion_log(
        universe_df, sic_mask, sector_mask, industry_mask, flag_mask,
    )

    # --- Step 6: Return surviving tickers (not in any exclusion mask).
    filtered_df = universe_df.loc[~combined_mask].reset_index(drop=True)

    n_excluded = int(combined_mask.sum())
    logger.info(
        "Exclusions: %d of %d tickers removed (%d remain).",
        n_excluded,
        len(universe_df),
        len(filtered_df),
    )

    return filtered_df, exclusion_log

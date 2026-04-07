"""Canonical line-item mapping for all financial statement fields used by the buffett-screener
pipeline.

This module is the single source of truth for every financial field name referenced by any
downstream module (metrics_engine/, screener/, valuation_reports/). Downstream code must
never reference raw API field names directly — only the canonical ``buffett_name`` keys
defined here.

Authoritative spec: docs/DATA_SOURCES.md §3 (Financial Statement Line-Item Mapping Table).

Key exports
-----------
LINE_ITEM_MAP : dict[str, LineItem]
    One entry per canonical field. Each entry describes the ideal API field name,
    acceptable substitutes in priority order, substitution confidence, and whether
    the security should be dropped when the field is unavailable.

CANONICAL_COLUMNS : dict[str, str]
    Maps each buffett_name to the snake_case column name stored in DuckDB. Currently
    an identity mapping (buffett_name == column name), but kept separate so downstream
    code has a stable reference point if column naming conventions change.

Functions
---------
resolve_field(raw_data, buffett_name) -> tuple[Any, str, str]
    Try ideal field then substitutes; return (value, field_used, confidence).

resolve_all_fields(raw_data) -> dict[str, tuple[Any, str, str]]
    Run resolve_field for every entry in LINE_ITEM_MAP.

get_drop_required_fields() -> list[str]
    Return buffett_names where drop_if_missing is True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LineItem dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineItem:
    """Descriptor for a single canonical financial statement field.

    Parameters
    ----------
    buffett_name:
        Internal canonical name used throughout the pipeline (e.g. ``"net_income"``).
        This is the key under which the value is stored in DuckDB and referenced by
        metrics_engine, screener, and valuation_reports.
    ideal_field:
        The preferred field name as returned by the primary API (FMP). Tried first
        when resolving a raw API response.
    substitutes:
        Acceptable alternative field names in priority order. Each is tried in sequence
        if the ideal field is absent from the raw data. May include derived-field
        sentinel strings (e.g. ``"DERIVED:totalRevenue-costOfRevenue"``); derivation
        logic is handled by the individual acquisition modules, not here.
    substitution_confidence:
        ``"High"``, ``"Medium"``, or ``"Low"``. Reflects how semantically equivalent
        the substitute is to the ideal field. Logged alongside every substitution.
    drop_if_missing:
        If ``True`` and neither the ideal field nor any substitute resolves to a
        non-null value, the security is flagged for exclusion by data_quality.py.
        If ``False``, the field is set to NaN and flagged but the security is retained.
    statement:
        Which financial statement this field belongs to:
        ``"income_statement"``, ``"balance_sheet"``, or ``"cash_flow"``.
    notes:
        Free-text implementation caveats (sign conventions, derivation rules,
        special-case handling). Referenced by financials.py and data_quality.py.
    """

    buffett_name: str
    ideal_field: str
    substitutes: list[str] = field(default_factory=list)
    substitution_confidence: str = "High"
    drop_if_missing: bool = True
    statement: str = "income_statement"
    notes: str = ""


# ---------------------------------------------------------------------------
# LINE_ITEM_MAP  — 14 canonical entries per docs/DATA_SOURCES.md §3
# ---------------------------------------------------------------------------
#
# The 14 entries correspond exactly to the required line items listed in the
# task specification: net_income, depreciation_amortization, capital_expenditures,
# total_revenue, gross_profit, sga, operating_income, interest_expense,
# long_term_debt, shareholders_equity, eps_diluted, shares_outstanding_diluted,
# treasury_stock, working_capital_change.
#
# yfinance candidate labels are included as additional substitutes after FMP
# substitutes, because financials.py falls back to yfinance when FMP is
# unavailable. All candidates are tried in the listed order.

LINE_ITEM_MAP: dict[str, LineItem] = {
    # ------------------------------------------------------------------
    # Income Statement fields
    # ------------------------------------------------------------------
    "net_income": LineItem(
        buffett_name="net_income",
        ideal_field="netIncome",
        substitutes=[
            "netIncomeFromContinuingOperations",  # FMP substitute
            "Net Income",                         # yfinance label
            "Net Income From Continuing Operations",  # yfinance label
        ],
        substitution_confidence="High",
        drop_if_missing=True,
        statement="income_statement",
        notes=(
            "Core profitability figure used in F3 (ROE), F10 (consistency), and as "
            "the base for owner earnings (F1). Prefer continuing-operations figure "
            "to exclude one-time items. If only total net income is available, use "
            "it and log at INFO."
        ),
    ),
    "depreciation_amortization": LineItem(
        buffett_name="depreciation_amortization",
        ideal_field="depreciationAndAmortization",
        substitutes=[
            "depreciationAmortizationDepletion",  # FMP CF substitute
            "Reconciled Depreciation",            # yfinance CF label
            "Depreciation And Amortization",      # yfinance IS/CF label
            "depreciation_amortization_is",       # internal IS fallback field name
        ],
        substitution_confidence="High",
        drop_if_missing=True,
        statement="cash_flow",
        notes=(
            "Preferred source is the cash flow statement (non-cash add-back). "
            "IS-sourced D&A is an acceptable substitute with confidence downgraded "
            "to Medium. Used as the maintenance CapEx proxy in F1 (known low-confidence "
            "approximation per 1986 Berkshire letter). Flag when total CapEx > 2×D&A. "
            "Drop rule: if unavailable for ≥5 of 10 years the security is excluded "
            "(stricter than standard 8-of-10 rule)."
        ),
    ),
    "capital_expenditures": LineItem(
        buffett_name="capital_expenditures",
        ideal_field="capitalExpenditure",
        substitutes=[
            "purchasesOfPropertyPlantAndEquipment",  # FMP substitute
            "purchaseOfPPE",                          # FMP short form
            "Capital Expenditure",                    # yfinance label
            "Purchase Of Ppe",                        # yfinance label
        ],
        substitution_confidence="High",
        drop_if_missing=True,
        statement="cash_flow",
        notes=(
            "SIGN CONVENTION: always stored as a negative number (cash outflow). "
            "Negate and log WARNING if the source returns a positive value. "
            "FMP and yfinance both sometimes return CapEx as positive — always audit "
            "after ingestion. Used in F1 (owner earnings) and F12 (CapEx/NI ratio)."
        ),
    ),
    "total_revenue": LineItem(
        buffett_name="total_revenue",
        ideal_field="totalRevenue",
        substitutes=[
            "revenue",       # FMP short form
            "salesRevenue",  # FMP alternative
            "Total Revenue", # yfinance label
            "Revenue",       # yfinance short label
        ],
        substitution_confidence="High",
        drop_if_missing=True,
        statement="income_statement",
        notes=(
            "Top-line revenue. Used in F7 (gross margin = gross_profit / total_revenue), "
            "F8 (SGA ratio denominator chain), FCF margin, and revenue CAGR. "
            "For financial companies this field exists but the ratio formulas do not apply "
            "— those companies are excluded by SIC code before metrics computation."
        ),
    ),
    "gross_profit": LineItem(
        buffett_name="gross_profit",
        ideal_field="grossProfit",
        substitutes=[
            "DERIVED:totalRevenue-costOfRevenue",  # Derivation sentinel
            "Gross Profit",                        # yfinance label
        ],
        substitution_confidence="High",
        drop_if_missing=True,
        statement="income_statement",
        notes=(
            "Used in F7 (gross margin) and F8 (SGA/gross_profit). "
            "If not available directly, derive as totalRevenue − costOfRevenue and log "
            "at INFO (confidence stays High when both inputs are present). "
            "Derivation sentinel 'DERIVED:totalRevenue-costOfRevenue' is handled by "
            "financials.py, not by resolve_field()."
        ),
    ),
    "sga": LineItem(
        buffett_name="sga",
        ideal_field="sellingGeneralAndAdministrativeExpenses",
        substitutes=[
            "generalAndAdministrativeExpenses",       # FMP alternative
            "DERIVED:operatingExpenses-costOfRevenue-depreciationAndAmortization",
            "Selling General Administrative",         # yfinance label
            "Selling And Marketing Expense",          # yfinance alternative
        ],
        substitution_confidence="Medium",
        drop_if_missing=False,
        statement="income_statement",
        notes=(
            "Used in F8 (SGA/gross_profit ratio). drop_if_missing=False: if unavailable, "
            "omit F8 for this ticker and log WARNING. "
            "Derivation sentinel handled by financials.py: if derived value is negative "
            "(inconsistent decomposition), set to NaN and skip F8. "
            "D&A must be normalised out of SGA when sourced from operating expenses "
            "to avoid double-counting per docs/FORMULAS.md F8 notes."
        ),
    ),
    "operating_income": LineItem(
        buffett_name="operating_income",
        ideal_field="operatingIncome",
        substitutes=[
            "ebit",                              # FMP/standard alternative
            "Operating Income",                  # yfinance label
            "Total Operating Income As Reported",# yfinance alternative
            "EBIT",                              # yfinance alternative
        ],
        substitution_confidence="High",
        drop_if_missing=True,
        statement="income_statement",
        notes=(
            "Used in F9 (interest coverage = interest_expense / operating_income). "
            "If operating_income ≤ 0, F9 is automatically flagged as failing per "
            "docs/FORMULAS.md F9 edge cases."
        ),
    ),
    "interest_expense": LineItem(
        buffett_name="interest_expense",
        ideal_field="interestExpense",
        substitutes=[
            "interestAndDebtExpense",      # FMP alternative
            "Interest Expense",            # yfinance label
            "Interest Expense Non Operating",  # yfinance alternative
        ],
        substitution_confidence="High",
        drop_if_missing=False,
        statement="income_statement",
        notes=(
            "drop_if_missing=False. Zero-fill rule: if missing AND long_term_debt=0 "
            "for that period, set to 0 and log INFO (expected for debt-free companies). "
            "If missing AND long_term_debt>0, set to NaN and log WARNING. "
            "Used in F9 (interest coverage) and F5 indirectly through owner earnings."
        ),
    ),
    # ------------------------------------------------------------------
    # Balance Sheet fields
    # ------------------------------------------------------------------
    "long_term_debt": LineItem(
        buffett_name="long_term_debt",
        ideal_field="longTermDebt",
        substitutes=[
            "totalLongTermDebt",            # FMP alternative
            "longTermDebtNoncurrent",       # FMP alternative
            "Long Term Debt",               # yfinance label
            "Long Term Debt And Capital Lease Obligation",  # yfinance alternative
        ],
        substitution_confidence="High",
        drop_if_missing=True,
        statement="balance_sheet",
        notes=(
            "Used in F5 (debt payoff years = long_term_debt / owner_earnings) and F6 "
            "(D/E ratio). Includes capital lease obligations when reported together — "
            "acceptable for Buffett-style analysis as leases are economic debt."
        ),
    ),
    "shareholders_equity": LineItem(
        buffett_name="shareholders_equity",
        ideal_field="totalStockholdersEquity",
        substitutes=[
            "totalEquity",        # FMP alternative
            "Stockholders Equity",# yfinance label
            "Common Stock Equity",# yfinance alternative
        ],
        substitution_confidence="High",
        drop_if_missing=True,
        statement="balance_sheet",
        notes=(
            "Used in F3 (ROE = net_income / shareholders_equity) and F6 "
            "(D/E = long_term_debt / shareholders_equity). "
            "EDGE CASE: if shareholders_equity ≤ 0, set ROE and D/E to NaN, "
            "set negative_equity_flag=True, do NOT drop the security. "
            "Assessment relies on F5, F7, F10, F11 per docs/FORMULAS.md."
        ),
    ),
    "eps_diluted": LineItem(
        buffett_name="eps_diluted",
        ideal_field="epsdiluted",
        substitutes=[
            "eps",       # FMP fallback
            "Diluted EPS",  # yfinance label
        ],
        substitution_confidence="High",
        drop_if_missing=True,
        statement="income_statement",
        notes=(
            "Diluted EPS is used in F11 (10yr EPS CAGR) and F14 (projected intrinsic "
            "value via DCF). Always use diluted figure to account for options and "
            "convertibles. Units: USD per share (not divided by 1000)."
        ),
    ),
    "shares_outstanding_diluted": LineItem(
        buffett_name="shares_outstanding_diluted",
        ideal_field="weightedAverageSharesDiluted",
        substitutes=[
            "sharesOutstanding",      # FMP fallback
            "Diluted Average Shares", # yfinance label
            "Average Dilution Earnings",  # yfinance alternative
        ],
        substitution_confidence="Medium",
        drop_if_missing=True,
        statement="income_statement",
        notes=(
            "Weighted-average diluted share count. Used in F13 (buyback indicator: "
            "compare current to 10-years-ago). Also used to cross-check EPS: "
            "net_income / shares_outstanding_diluted should ≈ eps_diluted. "
            "Units: full share count (not divided by 1000)."
        ),
    ),
    "treasury_stock": LineItem(
        buffett_name="treasury_stock",
        ideal_field="treasuryStock",
        substitutes=[
            "DERIVED:shares_outstanding_diluted_delta",  # year-over-year delta sentinel
            "Treasury Stock",           # yfinance label
            "Treasury Shares Number",   # yfinance alternative
        ],
        substitution_confidence="Medium",
        drop_if_missing=False,
        statement="balance_sheet",
        notes=(
            "drop_if_missing=False: if unavailable, omit buyback test (F13) for this "
            "ticker rather than dropping. Derivation sentinel means financials.py "
            "computes share count change year-over-year as a proxy when direct "
            "treasury stock value is not reported. Units: USD thousands when reported "
            "as a dollar value; share count when derived from delta."
        ),
    ),
    "working_capital_change": LineItem(
        buffett_name="working_capital_change",
        ideal_field="changeInWorkingCapital",
        substitutes=[
            "changesInWorkingCapital",    # FMP alternative spelling
            "Changes In Working Capital", # yfinance label
            "DERIVED:delta_current_assets-delta_current_liabilities",  # derivation sentinel
        ],
        substitution_confidence="Medium",
        drop_if_missing=False,
        statement="cash_flow",
        notes=(
            "Change in working capital used in F1 (owner earnings). "
            "Positive ΔWC = working capital increased = cash consumed = subtract. "
            "Negative ΔWC = working capital decreased = cash released = add. "
            "DERIVATION: ΔWC = Δ(current_assets − cash) − Δ(current_liabilities − "
            "short_term_debt) per docs/FORMULAS.md F1 definition. "
            "drop_if_missing=False: if unavailable, owner earnings omits the WC "
            "adjustment and logs WARNING."
        ),
    ),
}


# ---------------------------------------------------------------------------
# CANONICAL_COLUMNS  — DuckDB column name for each buffett_name
# ---------------------------------------------------------------------------
# Currently an identity mapping (buffett_name == column name). Kept as a
# separate dict so downstream code has a single import point and any future
# column rename only requires changing this dict, not every consumer.

CANONICAL_COLUMNS: dict[str, str] = {name: name for name in LINE_ITEM_MAP}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def resolve_field(
    raw_data: dict[str, Any],
    buffett_name: str,
) -> tuple[Any, str, str]:
    """Resolve a single canonical field from a raw API response row.

    Tries the ideal field first, then each substitute in priority order. Returns
    the first non-None value found, along with which field name was used and the
    applicable confidence level.

    Parameters
    ----------
    raw_data:
        A flat dict representing one row of raw API data (e.g., one fiscal year
        of an income statement). Keys are source API field names.
    buffett_name:
        The canonical field name to resolve (must be a key in ``LINE_ITEM_MAP``).

    Returns
    -------
    tuple of (value, field_used, confidence) where:

    - ``value``: the resolved value, or ``None`` if unresolvable.
    - ``field_used``: the source field name that was used, ``"MISSING"`` if none
      resolved, or ``"DERIVED:..."`` sentinel strings are passed through as-is
      (derivation is handled by the calling acquisition module, not here).
    - ``confidence``: ``"High"``, ``"Medium"``, ``"Low"``, ``"DROP"`` (missing +
      drop_if_missing=True), or ``"FLAG"`` (missing + drop_if_missing=False).

    Raises
    ------
    KeyError
        If ``buffett_name`` is not found in ``LINE_ITEM_MAP``.
    """
    item = LINE_ITEM_MAP[buffett_name]
    candidates = [item.ideal_field] + item.substitutes

    for candidate in candidates:
        # Derivation sentinels are not resolvable here — skip them.
        # The calling module handles derivation logic.
        if candidate.startswith("DERIVED:"):
            continue
        if candidate in raw_data and raw_data[candidate] is not None:
            confidence = (
                item.substitution_confidence
                if candidate != item.ideal_field
                else "High"
            )
            if candidate != item.ideal_field:
                logger.warning(
                    "Line-item substitution: buffett_name=%s | ideal=%s | used=%s "
                    "| confidence=%s",
                    buffett_name,
                    item.ideal_field,
                    candidate,
                    confidence,
                )
            return raw_data[candidate], candidate, confidence

    # Nothing resolved.
    outcome = "DROP" if item.drop_if_missing else "FLAG"
    logger.warning(
        "Unresolvable field: buffett_name=%s | ideal=%s | outcome=%s",
        buffett_name,
        item.ideal_field,
        outcome,
    )
    return None, "MISSING", outcome


def resolve_all_fields(
    raw_data: dict[str, Any],
) -> dict[str, tuple[Any, str, str]]:
    """Resolve every canonical field in LINE_ITEM_MAP from a raw API response row.

    Calls :func:`resolve_field` for each entry in ``LINE_ITEM_MAP`` and collects
    the results.

    Parameters
    ----------
    raw_data:
        A flat dict representing one row of raw API data for a single fiscal year
        and statement type. In practice callers pass a merged dict combining
        income statement, balance sheet, and cash flow fields for one period.

    Returns
    -------
    dict[buffett_name, (value, field_used, confidence)]
        One entry per canonical field. See :func:`resolve_field` for the meaning
        of each tuple element.
    """
    return {
        buffett_name: resolve_field(raw_data, buffett_name)
        for buffett_name in LINE_ITEM_MAP
    }


def get_drop_required_fields() -> list[str]:
    """Return the list of canonical field names where ``drop_if_missing`` is True.

    Securities missing any of these fields for ≥ ``data_quality.min_field_coverage_years``
    fiscal years (configured in filter_config.yaml) are excluded from all downstream
    processing by data_quality.py.

    Returns
    -------
    list[str]
        Canonical field names (buffett_names) that trigger a drop when absent.
    """
    return [
        name
        for name, item in LINE_ITEM_MAP.items()
        if item.drop_if_missing
    ]

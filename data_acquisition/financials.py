"""
data_acquisition.financials
============================
Fetches multi-year financial statements (income statement, balance sheet,
cash flow statement) for a given ticker.

Primary source: yfinance (fast, no API key required for basic data)
Fallback source: SEC EDGAR XBRL REST API (authoritative, requires SEC_USER_AGENT)

Data is normalised into the schema.IncomeStatement, schema.BalanceSheet, and
schema.CashFlowStatement dataclasses before being returned, so callers never
need to know which source was used.

Caching:
    Raw JSON/DataFrame responses are cached in DuckDB to avoid redundant API
    calls during iterative development. Cache TTL is 24 hours for financial
    statements (statements change at most quarterly).
"""

from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from data_acquisition.api_config import SecEdgarConfig
from data_acquisition.schema import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    IncomeStatement,
    TickerDataBundle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# yfinance label maps — each value is a prioritised list of candidate labels
# ---------------------------------------------------------------------------

INCOME_LABELS: dict[str, list[str]] = {
    "revenue":                   ["Total Revenue", "Revenue"],
    "cost_of_revenue":           ["Cost Of Revenue", "Cost of Revenue"],
    "gross_profit":              ["Gross Profit"],
    "operating_income":          ["Operating Income", "Total Operating Income As Reported", "EBIT"],
    "interest_expense":          ["Interest Expense", "Interest Expense Non Operating"],
    "pretax_income":             ["Pretax Income", "Income Before Tax"],
    "income_tax":                ["Tax Provision", "Income Tax Expense"],
    "net_income":                ["Net Income", "Net Income From Continuing Operations"],
    "depreciation_amortization": [
        "Reconciled Depreciation",
        "Depreciation And Amortization",
        "Depreciation Depletion And Amortization",
    ],
    "shares_diluted":            ["Diluted Average Shares", "Average Dilution Earnings"],
    "eps_diluted":               ["Diluted EPS"],
}

BALANCE_LABELS: dict[str, list[str]] = {
    "cash_and_equivalents":      ["Cash And Cash Equivalents", "Cash"],
    "short_term_investments":    ["Short Term Investments", "Available For Sale Securities"],
    "total_current_assets":      ["Current Assets", "Total Current Assets"],
    "total_assets":              ["Total Assets"],
    "total_current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "short_term_debt":           ["Current Debt", "Short Long Term Debt", "Short Term Debt"],
    "long_term_debt":            ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "total_liabilities":         ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "shareholders_equity":       ["Stockholders Equity", "Common Stock Equity"],
    "retained_earnings":         ["Retained Earnings"],
    "shares_outstanding":        ["Ordinary Shares Number", "Share Issued"],
}

CASHFLOW_LABELS: dict[str, list[str]] = {
    "operating_cash_flow":   ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capital_expenditures":  ["Capital Expenditure", "Purchase Of Ppe", "Capital Expenditures"],
    "free_cash_flow":        ["Free Cash Flow"],
    "dividends_paid":        ["Cash Dividends Paid", "Payment Of Dividends"],
    "stock_buybacks":        ["Repurchase Of Capital Stock", "Common Stock Repurchase"],
    "net_debt_issuance":     ["Net Issuance Payments Of Debt", "Long Term Debt Issuance"],
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_field(
    df: pd.DataFrame,
    candidates: list[str],
    col,
    field_name: str,
    ticker: str,
) -> float:
    """
    Extract a single field value from a yfinance DataFrame column.

    Tries each candidate row label in priority order. Returns float("nan")
    with a warning if none of the candidates are found in the DataFrame index.

    Args:
        df:         yfinance financials/balance_sheet/cashflow DataFrame.
                    Rows = line items, columns = period-end dates.
        candidates: Ordered list of row label strings to try.
        col:        Column (period-end date) to extract the value from.
        field_name: Schema field name (for logging).
        ticker:     Ticker symbol (for logging).

    Returns:
        float value, or float("nan") if not found / not convertible.
    """
    for label in candidates:
        if label in df.index:
            raw = df.at[label, col]
            logger.debug(
                "Field mapping: ticker=%s field=%s label=%r (period=%s).",
                ticker, field_name, label, col,
            )
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                logger.warning(
                    "Field %s for ticker %s (period %s) found label %r but value is NaN/None.",
                    field_name, ticker, col, label,
                )
                return float("nan")
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Field %s for ticker %s (period %s): could not convert %r to float: %s.",
                    field_name, ticker, col, raw, exc,
                )
                return float("nan")

    logger.warning(
        "Field %s for ticker %s (period %s): none of the candidate labels %s "
        "found in DataFrame index. Storing NaN.",
        field_name, ticker, col, candidates,
    )
    return float("nan")


def _col_to_date(col) -> date:
    """Convert a yfinance column (Timestamp or str) to a date object."""
    if isinstance(col, date):
        return col
    try:
        return pd.Timestamp(col).date()
    except Exception:
        return datetime.today().date()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_financials(
    ticker: str,
    years: int = 10,
    use_cache: bool = True,
    store=None,
) -> TickerDataBundle:
    """
    Fetch the last `years` annual financial statements for a single ticker
    and return them as a TickerDataBundle.

    Args:
        ticker:    Stock ticker symbol (e.g. "AAPL").
        years:     Number of historical annual periods to retrieve.
        use_cache: If True and a fresh cache entry exists in DuckDB, return
                   cached data without hitting external APIs.
        store:     DuckDBStore instance for cache read/write.

    Returns:
        TickerDataBundle with income_statements, balance_sheets, and
        cash_flow_statements populated.

    Raises:
        ValueError: If the ticker is not found on any source.
        RuntimeError: If all data sources fail after retries.
    """
    # ---- Cache check -------------------------------------------------------
    if use_cache and store is not None:
        cached_income = store.load_income_statements(ticker)
        cached_bs = store.load_balance_sheets(ticker)
        cached_cf = store.load_cash_flow_statements(ticker)

        if not cached_income.empty and not cached_bs.empty and not cached_cf.empty:
            logger.debug("Cache hit for ticker %s financial statements.", ticker)
            income_list = _df_to_income_statements(cached_income)
            bs_list = _df_to_balance_sheets(cached_bs)
            cf_list = _df_to_cash_flow_statements(cached_cf)

            if len(income_list) >= years:
                profile = CompanyProfile(
                    ticker=ticker, name="", sector="", industry="",
                    exchange="", country="", currency="USD",
                    fiscal_year_end_month=12, is_adr=False, is_spac=False,
                )
                return TickerDataBundle(
                    profile=profile,
                    income_statements=income_list,
                    balance_sheets=bs_list,
                    cash_flow_statements=cf_list,
                )
            else:
                logger.debug(
                    "Cache has only %d income rows for %s (need %d). Refreshing.",
                    len(income_list), ticker, years,
                )

    # ---- yfinance fetch ----------------------------------------------------
    logger.info("Fetching financial statements for %s via yfinance.", ticker)
    yf_ticker = yf.Ticker(ticker)

    try:
        inc_df: pd.DataFrame = yf_ticker.financials       # rows=items, cols=dates
        bs_df: pd.DataFrame = yf_ticker.balance_sheet
        cf_df: pd.DataFrame = yf_ticker.cashflow
    except Exception as exc:
        raise RuntimeError(
            f"yfinance failed to fetch financials for {ticker}: {exc}"
        ) from exc

    income_list = _parse_income_stmt(inc_df, ticker) if inc_df is not None and not inc_df.empty else []
    bs_list = _parse_balance_sheet(bs_df, ticker) if bs_df is not None and not bs_df.empty else []
    cf_list = _parse_cash_flow(cf_df, ticker) if cf_df is not None and not cf_df.empty else []

    # ---- EDGAR fallback if yfinance is insufficient -----------------------
    if len(income_list) < years:
        logger.info(
            "yfinance returned %d periods for %s (need %d). Attempting EDGAR fallback.",
            len(income_list), ticker, years,
        )
        try:
            sec_cfg = SecEdgarConfig.from_env()
            cik = lookup_cik(ticker, sec_cfg)
            edgar_data = fetch_from_edgar(ticker, cik, years, sec_cfg)

            edgar_income = _parse_income_stmt(edgar_data.get("income", pd.DataFrame()), ticker)
            edgar_bs = _parse_balance_sheet(edgar_data.get("balance", pd.DataFrame()), ticker)
            edgar_cf = _parse_cash_flow(edgar_data.get("cashflow", pd.DataFrame()), ticker)

            # Merge: prefer yfinance data where it exists, supplement with EDGAR
            existing_years_income = {s.fiscal_year_end for s in income_list}
            for stmt in edgar_income:
                if stmt.fiscal_year_end not in existing_years_income:
                    income_list.append(stmt)

            existing_years_bs = {s.fiscal_year_end for s in bs_list}
            for stmt in edgar_bs:
                if stmt.fiscal_year_end not in existing_years_bs:
                    bs_list.append(stmt)

            existing_years_cf = {s.fiscal_year_end for s in cf_list}
            for stmt in edgar_cf:
                if stmt.fiscal_year_end not in existing_years_cf:
                    cf_list.append(stmt)

            income_list.sort(key=lambda s: s.fiscal_year_end)
            bs_list.sort(key=lambda s: s.fiscal_year_end)
            cf_list.sort(key=lambda s: s.fiscal_year_end)

        except Exception as exc:
            logger.warning(
                "EDGAR fallback for %s failed: %s. Proceeding with yfinance data only.",
                ticker, exc,
            )

    if not income_list and not bs_list and not cf_list:
        raise ValueError(
            f"No financial data found for ticker {ticker} from any source."
        )

    # Build a minimal profile (caller enriches later via enrich_with_profiles)
    info = {}
    try:
        info = yf_ticker.info or {}
    except Exception:
        pass

    profile = CompanyProfile(
        ticker=ticker,
        name=info.get("longName") or info.get("shortName") or ticker,
        sector=info.get("sector") or "",
        industry=info.get("industry") or "",
        exchange=info.get("exchange") or "",
        country=info.get("country") or "",
        currency=info.get("currency") or "USD",
        fiscal_year_end_month=12,
        is_adr=False,
        is_spac=False,
    )

    bundle = TickerDataBundle(
        profile=profile,
        income_statements=income_list,
        balance_sheets=bs_list,
        cash_flow_statements=cf_list,
    )

    # ---- Write to cache ----------------------------------------------------
    if use_cache and store is not None and income_list:
        try:
            import dataclasses
            store.save_income_statements(
                pd.DataFrame([dataclasses.asdict(s) for s in income_list])
            )
            store.save_balance_sheets(
                pd.DataFrame([dataclasses.asdict(s) for s in bs_list])
            )
            store.save_cash_flow_statements(
                pd.DataFrame([dataclasses.asdict(s) for s in cf_list])
            )
        except Exception as exc:
            logger.warning(
                "Failed to write financial data for %s to cache: %s.", ticker, exc
            )

    return bundle


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_income_stmt(
    df: pd.DataFrame,
    ticker: str,
) -> list[IncomeStatement]:
    """
    Convert a raw income statement DataFrame into IncomeStatement instances.

    Args:
        df:     yfinance financials DataFrame (rows=items, columns=period dates).
        ticker: Ticker symbol.

    Returns:
        List of IncomeStatement instances sorted ascending by fiscal_year_end.
    """
    if df is None or df.empty:
        return []

    results: list[IncomeStatement] = []

    for col in df.columns:
        period_date = _col_to_date(col)

        def get(field_name: str) -> float:
            return _get_field(df, INCOME_LABELS[field_name], col, field_name, ticker)

        revenue = get("revenue")
        cost_of_revenue = get("cost_of_revenue")
        gross_profit = get("gross_profit")
        operating_income = get("operating_income")
        interest_expense = get("interest_expense")
        pretax_income = get("pretax_income")
        income_tax = get("income_tax")
        net_income = get("net_income")
        depreciation_amortization = get("depreciation_amortization")
        shares_diluted = get("shares_diluted")
        eps_diluted = get("eps_diluted")

        # Derive gross_profit if not directly available
        if math.isnan(gross_profit) and not math.isnan(revenue) and not math.isnan(cost_of_revenue):
            gross_profit = revenue - cost_of_revenue

        # Derive ebitda
        if not math.isnan(operating_income) and not math.isnan(depreciation_amortization):
            ebitda = operating_income + depreciation_amortization
        elif not math.isnan(operating_income):
            ebitda = operating_income
        else:
            ebitda = float("nan")

        # Derive eps_diluted if missing
        if math.isnan(eps_diluted):
            if not math.isnan(net_income) and not math.isnan(shares_diluted) and shares_diluted != 0:
                # shares_diluted from yfinance is in units — divide net_income by shares_diluted
                # Both should be in same units after /1000 conversion below
                eps_diluted = float("nan")  # Will be computed after unit conversion

        # Convert monetary values from USD (yfinance) to USD thousands
        def to_thousands(v: float) -> float:
            return v / 1_000.0 if not math.isnan(v) else float("nan")

        revenue = to_thousands(revenue)
        cost_of_revenue = to_thousands(cost_of_revenue)
        gross_profit = to_thousands(gross_profit)
        operating_income = to_thousands(operating_income)
        interest_expense = to_thousands(interest_expense)
        pretax_income = to_thousands(pretax_income)
        income_tax = to_thousands(income_tax)
        net_income = to_thousands(net_income)
        depreciation_amortization = to_thousands(depreciation_amortization)
        ebitda = to_thousands(ebitda)

        # shares_diluted from yfinance is already in units of shares; convert to thousands
        shares_diluted_k = to_thousands(shares_diluted)

        # eps_diluted is per-share, no unit conversion needed (already in dollars/share)
        if math.isnan(eps_diluted):
            if (
                not math.isnan(net_income)
                and not math.isnan(shares_diluted_k)
                and shares_diluted_k != 0
            ):
                eps_diluted = net_income / shares_diluted_k

        stmt = IncomeStatement(
            ticker=ticker,
            fiscal_year_end=period_date,
            period="annual",
            revenue=revenue,
            cost_of_revenue=cost_of_revenue,
            gross_profit=gross_profit,
            operating_income=operating_income,
            interest_expense=interest_expense,
            pretax_income=pretax_income,
            income_tax=income_tax,
            net_income=net_income,
            depreciation_amortization=depreciation_amortization,
            ebitda=ebitda,
            shares_diluted=shares_diluted_k,
            eps_diluted=eps_diluted,
        )
        results.append(stmt)

    results.sort(key=lambda s: s.fiscal_year_end)
    return results


def _parse_balance_sheet(
    df: pd.DataFrame,
    ticker: str,
) -> list[BalanceSheet]:
    """
    Convert a raw balance sheet DataFrame into BalanceSheet instances.

    Args:
        df:     yfinance balance_sheet DataFrame (rows=items, columns=period dates).
        ticker: Ticker symbol.

    Returns:
        List of BalanceSheet instances sorted ascending by fiscal_year_end.
    """
    if df is None or df.empty:
        return []

    results: list[BalanceSheet] = []

    for col in df.columns:
        period_date = _col_to_date(col)

        def get(field_name: str) -> float:
            return _get_field(df, BALANCE_LABELS[field_name], col, field_name, ticker)

        cash_and_equivalents = get("cash_and_equivalents")
        short_term_investments = get("short_term_investments")
        total_current_assets = get("total_current_assets")
        total_assets = get("total_assets")
        total_current_liabilities = get("total_current_liabilities")
        short_term_debt = get("short_term_debt")
        long_term_debt = get("long_term_debt")
        total_liabilities = get("total_liabilities")
        shareholders_equity = get("shareholders_equity")
        retained_earnings = get("retained_earnings")
        shares_outstanding = get("shares_outstanding")

        # Derive total_debt
        if not math.isnan(short_term_debt) and not math.isnan(long_term_debt):
            total_debt = short_term_debt + long_term_debt
        elif not math.isnan(long_term_debt):
            total_debt = long_term_debt
        elif not math.isnan(short_term_debt):
            total_debt = short_term_debt
        else:
            total_debt = float("nan")

        # Convert all monetary values to USD thousands
        def to_thousands(v: float) -> float:
            return v / 1_000.0 if not math.isnan(v) else float("nan")

        bs = BalanceSheet(
            ticker=ticker,
            fiscal_year_end=period_date,
            cash_and_equivalents=to_thousands(cash_and_equivalents),
            short_term_investments=to_thousands(short_term_investments),
            total_current_assets=to_thousands(total_current_assets),
            total_assets=to_thousands(total_assets),
            total_current_liabilities=to_thousands(total_current_liabilities),
            short_term_debt=to_thousands(short_term_debt),
            long_term_debt=to_thousands(long_term_debt),
            total_debt=to_thousands(total_debt),
            total_liabilities=to_thousands(total_liabilities),
            shareholders_equity=to_thousands(shareholders_equity),
            retained_earnings=to_thousands(retained_earnings),
            shares_outstanding=to_thousands(shares_outstanding),
        )
        results.append(bs)

    results.sort(key=lambda s: s.fiscal_year_end)
    return results


def _parse_cash_flow(
    df: pd.DataFrame,
    ticker: str,
) -> list[CashFlowStatement]:
    """
    Convert a raw cash flow DataFrame into CashFlowStatement instances.

    Args:
        df:     yfinance cashflow DataFrame (rows=items, columns=period dates).
        ticker: Ticker symbol.

    Returns:
        List of CashFlowStatement instances sorted ascending by fiscal_year_end.
    """
    if df is None or df.empty:
        return []

    results: list[CashFlowStatement] = []

    for col in df.columns:
        period_date = _col_to_date(col)

        def get(field_name: str) -> float:
            return _get_field(df, CASHFLOW_LABELS[field_name], col, field_name, ticker)

        operating_cash_flow = get("operating_cash_flow")
        capital_expenditures = get("capital_expenditures")
        free_cash_flow = get("free_cash_flow")
        dividends_paid = get("dividends_paid")
        stock_buybacks = get("stock_buybacks")
        net_debt_issuance = get("net_debt_issuance")

        # Constraint: capital_expenditures must be negative (cash outflow)
        if not math.isnan(capital_expenditures) and capital_expenditures > 0:
            logger.warning(
                "ticker=%s period=%s: capital_expenditures=%.0f is positive — "
                "negating to enforce sign convention (outflow).",
                ticker, period_date, capital_expenditures,
            )
            capital_expenditures = -capital_expenditures

        # Derive free_cash_flow if missing
        if math.isnan(free_cash_flow):
            if not math.isnan(operating_cash_flow) and not math.isnan(capital_expenditures):
                free_cash_flow = operating_cash_flow + capital_expenditures
                logger.debug(
                    "ticker=%s period=%s: free_cash_flow computed as OCF + CapEx = %.0f.",
                    ticker, period_date, free_cash_flow,
                )

        # Convert all monetary values to USD thousands
        def to_thousands(v: float) -> float:
            return v / 1_000.0 if not math.isnan(v) else float("nan")

        cf = CashFlowStatement(
            ticker=ticker,
            fiscal_year_end=period_date,
            period="annual",
            operating_cash_flow=to_thousands(operating_cash_flow),
            capital_expenditures=to_thousands(capital_expenditures),
            free_cash_flow=to_thousands(free_cash_flow),
            dividends_paid=to_thousands(dividends_paid),
            stock_buybacks=to_thousands(stock_buybacks),
            net_debt_issuance=to_thousands(net_debt_issuance),
        )
        results.append(cf)

    results.sort(key=lambda s: s.fiscal_year_end)
    return results


# ---------------------------------------------------------------------------
# EDGAR fallback
# ---------------------------------------------------------------------------

def fetch_from_edgar(
    ticker: str,
    cik: str,
    years: int,
    config: SecEdgarConfig,
) -> dict[str, pd.DataFrame]:
    """
    Pull financial statement data directly from SEC EDGAR XBRL REST API.

    Args:
        ticker: Ticker symbol (used for logging only).
        cik:    SEC Central Index Key (10-digit zero-padded string).
        years:  Number of annual periods to retrieve.
        config: SecEdgarConfig with user_agent and base_url.

    Returns:
        Dict with keys "income", "balance", "cashflow", each mapped to a
        DataFrame in the same row-label format as yfinance (rows=items, cols=dates).
    """
    url = f"{config.base_url}/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": config.user_agent}

    logger.info("Fetching EDGAR company facts for ticker=%s (CIK=%s).", ticker, cik)

    resp = requests.get(url, headers=headers, timeout=config.timeout)
    resp.raise_for_status()
    facts = resp.json()

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        logger.warning(
            "EDGAR response for CIK=%s has no us-gaap facts. Returning empty DataFrames.",
            cik,
        )
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}

    def extract_concept(concept_name: str) -> pd.Series:
        """Extract annual 10-K USD values for a single XBRL concept."""
        concept = us_gaap.get(concept_name, {})
        units = concept.get("units", {})
        usd_entries = units.get("USD", [])
        if not usd_entries:
            return pd.Series(dtype=float, name=concept_name)

        rows = []
        for entry in usd_entries:
            if entry.get("form") != "10-K":
                continue
            if entry.get("fp") not in ("FY", "Q4", None):
                continue
            end_date = entry.get("end")
            val = entry.get("val")
            if end_date and val is not None:
                rows.append({"date": pd.Timestamp(end_date), "val": float(val)})

        if not rows:
            return pd.Series(dtype=float, name=concept_name)

        series_df = pd.DataFrame(rows).sort_values("date").drop_duplicates(
            subset="date", keep="last"
        )
        s = pd.Series(series_df["val"].values, index=series_df["date"], name=concept_name)
        # Keep only the last `years` data points
        return s.iloc[-years:]

    # Concept name → yfinance-compatible row label
    concept_map = {
        # Income statement
        "Revenues":                                        "Total Revenue",
        "RevenueFromContractWithCustomerExcludingAssessedTax": "Total Revenue",
        "CostOfRevenue":                                   "Cost Of Revenue",
        "GrossProfit":                                     "Gross Profit",
        "OperatingIncomeLoss":                             "Operating Income",
        "InterestExpense":                                 "Interest Expense",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
                                                           "Pretax Income",
        "IncomeTaxExpenseBenefit":                         "Tax Provision",
        "NetIncomeLoss":                                   "Net Income",
        "DepreciationDepletionAndAmortization":            "Reconciled Depreciation",
        # Balance sheet
        "CashAndCashEquivalentsAtCarryingValue":           "Cash And Cash Equivalents",
        "ShortTermInvestments":                            "Short Term Investments",
        "AssetsCurrent":                                   "Current Assets",
        "Assets":                                          "Total Assets",
        "LiabilitiesCurrent":                             "Current Liabilities",
        "LongTermDebt":                                    "Long Term Debt",
        "Liabilities":                                     "Total Liabilities Net Minority Interest",
        "StockholdersEquity":                              "Stockholders Equity",
        "RetainedEarningsAccumulatedDeficit":              "Retained Earnings",
        "CommonStockSharesOutstanding":                    "Ordinary Shares Number",
        # Cash flow
        "NetCashProvidedByUsedInOperatingActivities":      "Operating Cash Flow",
        "PaymentsToAcquirePropertyPlantAndEquipment":      "Capital Expenditure",
        "PaymentsOfDividendsCommonStock":                  "Cash Dividends Paid",
        "PaymentsForRepurchaseOfCommonStock":              "Repurchase Of Capital Stock",
        "ProceedsFromIssuanceOfLongTermDebt":              "Net Issuance Payments Of Debt",
    }

    all_series: list[pd.Series] = []
    for concept_name, row_label in concept_map.items():
        s = extract_concept(concept_name)
        if not s.empty:
            s.name = row_label
            all_series.append(s)
        time.sleep(1.0 / config.rate_limit_rps)

    if not all_series:
        logger.warning("EDGAR returned no usable data for CIK=%s.", cik)
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}

    # Combine into one wide DataFrame (rows = labels, columns = dates)
    combined = pd.DataFrame({s.name: s for s in all_series}).T

    # Separate into three statement DataFrames using row label membership
    income_labels = {
        "Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income",
        "Interest Expense", "Pretax Income", "Tax Provision", "Net Income",
        "Reconciled Depreciation",
    }
    balance_labels = {
        "Cash And Cash Equivalents", "Short Term Investments", "Current Assets",
        "Total Assets", "Current Liabilities", "Long Term Debt",
        "Total Liabilities Net Minority Interest", "Stockholders Equity",
        "Retained Earnings", "Ordinary Shares Number",
    }
    cashflow_labels = {
        "Operating Cash Flow", "Capital Expenditure", "Cash Dividends Paid",
        "Repurchase Of Capital Stock", "Net Issuance Payments Of Debt",
    }

    def subset(labels: set) -> pd.DataFrame:
        rows = [r for r in combined.index if r in labels]
        return combined.loc[rows] if rows else pd.DataFrame()

    return {
        "income": subset(income_labels),
        "balance": subset(balance_labels),
        "cashflow": subset(cashflow_labels),
    }


def lookup_cik(ticker: str, config: SecEdgarConfig) -> str:
    """
    Resolve a ticker symbol to its SEC CIK (Central Index Key).

    Args:
        ticker: Stock ticker symbol.
        config: SecEdgarConfig for the HTTP request.

    Returns:
        Zero-padded 10-digit CIK string (e.g. "0000320193" for Apple).

    Raises:
        ValueError: If the ticker cannot be found in the EDGAR company tickers list.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": config.user_agent}

    logger.info("Looking up CIK for ticker=%s from SEC company tickers JSON.", ticker)
    resp = requests.get(url, headers=headers, timeout=config.timeout)
    resp.raise_for_status()
    data: dict = resp.json()

    ticker_upper = ticker.upper()
    for _key, entry in data.items():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik_raw: int = entry["cik_str"]
            cik_padded = str(cik_raw).zfill(10)
            logger.debug(
                "CIK lookup: ticker=%s → CIK=%s (company=%s).",
                ticker, cik_padded, entry.get("title", ""),
            )
            return cik_padded

    raise ValueError(
        f"Ticker {ticker!r} not found in SEC EDGAR company tickers list "
        f"({url}). Check the ticker symbol or use CIK directly."
    )


# ---------------------------------------------------------------------------
# Cache deserialisation helpers
# ---------------------------------------------------------------------------

def _df_to_income_statements(df: pd.DataFrame) -> list[IncomeStatement]:
    """Reconstruct IncomeStatement objects from a cached DataFrame."""
    results = []
    for _, row in df.iterrows():
        try:
            stmt = IncomeStatement(
                ticker=str(row["ticker"]),
                fiscal_year_end=pd.Timestamp(row["fiscal_year_end"]).date(),
                period=str(row.get("period", "annual")),
                revenue=float(row.get("revenue", float("nan"))),
                cost_of_revenue=float(row.get("cost_of_revenue", float("nan"))),
                gross_profit=float(row.get("gross_profit", float("nan"))),
                operating_income=float(row.get("operating_income", float("nan"))),
                interest_expense=float(row.get("interest_expense", float("nan"))),
                pretax_income=float(row.get("pretax_income", float("nan"))),
                income_tax=float(row.get("income_tax", float("nan"))),
                net_income=float(row.get("net_income", float("nan"))),
                depreciation_amortization=float(row.get("depreciation_amortization", float("nan"))),
                ebitda=float(row.get("ebitda", float("nan"))),
                shares_diluted=float(row.get("shares_diluted", float("nan"))),
                eps_diluted=float(row.get("eps_diluted", float("nan"))),
            )
            results.append(stmt)
        except Exception as exc:
            logger.warning("Failed to deserialise income statement row: %s.", exc)
    results.sort(key=lambda s: s.fiscal_year_end)
    return results


def _df_to_balance_sheets(df: pd.DataFrame) -> list[BalanceSheet]:
    """Reconstruct BalanceSheet objects from a cached DataFrame."""
    results = []
    for _, row in df.iterrows():
        try:
            bs = BalanceSheet(
                ticker=str(row["ticker"]),
                fiscal_year_end=pd.Timestamp(row["fiscal_year_end"]).date(),
                cash_and_equivalents=float(row.get("cash_and_equivalents", float("nan"))),
                short_term_investments=float(row.get("short_term_investments", float("nan"))),
                total_current_assets=float(row.get("total_current_assets", float("nan"))),
                total_assets=float(row.get("total_assets", float("nan"))),
                total_current_liabilities=float(row.get("total_current_liabilities", float("nan"))),
                short_term_debt=float(row.get("short_term_debt", float("nan"))),
                long_term_debt=float(row.get("long_term_debt", float("nan"))),
                total_debt=float(row.get("total_debt", float("nan"))),
                total_liabilities=float(row.get("total_liabilities", float("nan"))),
                shareholders_equity=float(row.get("shareholders_equity", float("nan"))),
                retained_earnings=float(row.get("retained_earnings", float("nan"))),
                shares_outstanding=float(row.get("shares_outstanding", float("nan"))),
            )
            results.append(bs)
        except Exception as exc:
            logger.warning("Failed to deserialise balance sheet row: %s.", exc)
    results.sort(key=lambda s: s.fiscal_year_end)
    return results


def _df_to_cash_flow_statements(df: pd.DataFrame) -> list[CashFlowStatement]:
    """Reconstruct CashFlowStatement objects from a cached DataFrame."""
    results = []
    for _, row in df.iterrows():
        try:
            cf = CashFlowStatement(
                ticker=str(row["ticker"]),
                fiscal_year_end=pd.Timestamp(row["fiscal_year_end"]).date(),
                period=str(row.get("period", "annual")),
                operating_cash_flow=float(row.get("operating_cash_flow", float("nan"))),
                capital_expenditures=float(row.get("capital_expenditures", float("nan"))),
                free_cash_flow=float(row.get("free_cash_flow", float("nan"))),
                dividends_paid=float(row.get("dividends_paid", float("nan"))),
                stock_buybacks=float(row.get("stock_buybacks", float("nan"))),
                net_debt_issuance=float(row.get("net_debt_issuance", float("nan"))),
            )
            results.append(cf)
        except Exception as exc:
            logger.warning("Failed to deserialise cash flow statement row: %s.", exc)
    results.sort(key=lambda s: s.fiscal_year_end)
    return results

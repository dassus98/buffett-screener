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

from datetime import date
from typing import Optional

import pandas as pd

from data_acquisition.api_config import SecEdgarConfig
from data_acquisition.schema import (
    BalanceSheet,
    CashFlowStatement,
    IncomeStatement,
    TickerDataBundle,
)


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
        years:     Number of historical annual periods to retrieve (default 10).
        use_cache: If True and a fresh cache entry exists in DuckDB, return
                   cached data without hitting external APIs.
        store:     DuckDBStore instance for cache read/write. Required if
                   use_cache=True.

    Returns:
        TickerDataBundle with income_statements, balance_sheets, and
        cash_flow_statements populated. market_data and macro are left None
        (populated by separate fetchers).

    Raises:
        ValueError: If the ticker is not found on any source.
        RuntimeError: If all data sources fail after retries.

    Logic:
        1. Check DuckDB cache for fresh data (if use_cache)
        2. Try yfinance first: yf.Ticker(ticker).financials / balance_sheet / cashflow
        3. If yfinance returns fewer than `years` periods, supplement or replace
           with SEC EDGAR XBRL data via fetch_from_edgar()
        4. Normalise raw DataFrames into dataclass lists via _parse_income_stmt(),
           _parse_balance_sheet(), _parse_cash_flow()
        5. Write to cache (if use_cache)
        6. Return TickerDataBundle
    """
    ...


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
        cik:    SEC Central Index Key for the company (10-digit zero-padded string).
        years:  Number of annual periods to retrieve.
        config: SecEdgarConfig with user_agent and base_url.

    Returns:
        Dict with keys "income", "balance", "cashflow", each mapped to a
        DataFrame in a standardised column layout matching the schema fields.

    Logic:
        1. GET {base_url}/api/xbrl/companyfacts/CIK{cik}.json
           (returns all reported XBRL facts for the company)
        2. Extract relevant concepts (us-gaap namespace):
               Revenues, NetIncomeLoss, OperatingIncomeLoss,
               Assets, LiabilitiesAndStockholdersEquity,
               NetCashProvidedByUsedInOperatingActivities, etc.
        3. Filter to "10-K" form, annual frequency, USD denomination
        4. Pivot into DataFrames with fiscal_year_end as the index
        5. Respect rate_limit_rps from config (use time.sleep between requests)
    """
    ...


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

    Logic:
        1. GET https://www.sec.gov/files/company_tickers.json (cached locally)
        2. Search for ticker (case-insensitive) in the returned dict
        3. Zero-pad the CIK to 10 digits and return
    """
    ...


def _parse_income_stmt(
    df: pd.DataFrame,
    ticker: str,
) -> list[IncomeStatement]:
    """
    Convert a raw income statement DataFrame (columns = fiscal year dates) into
    a list of IncomeStatement dataclass instances.

    Args:
        df:     Raw DataFrame as returned by yfinance or _edgar_to_df().
                Rows are financial line items; columns are period-end dates.
        ticker: Ticker symbol to populate the IncomeStatement.ticker field.

    Returns:
        List of IncomeStatement instances, one per column (fiscal year),
        sorted ascending by fiscal_year_end.

    Logic:
        1. Iterate over columns (period dates)
        2. Map yfinance row labels to IncomeStatement fields:
               "Total Revenue" → revenue
               "Cost Of Revenue" → cost_of_revenue
               "Gross Profit" → gross_profit
               "Operating Income" → operating_income
               "Interest Expense" → interest_expense  (may be negative in yf)
               "Pretax Income" → pretax_income
               "Income Tax Expense" → income_tax
               "Net Income" → net_income
               "Reconciled Depreciation" → depreciation_amortization
        3. Compute derived fields: ebitda = operating_income + D&A
        4. Handle missing values: set to 0.0 and log a warning
        5. Convert all amounts from units returned by source to USD thousands
    """
    ...


def _parse_balance_sheet(
    df: pd.DataFrame,
    ticker: str,
) -> list[BalanceSheet]:
    """
    Convert a raw balance sheet DataFrame into a list of BalanceSheet instances.

    Args:
        df:     Raw DataFrame with rows = line items, columns = period-end dates.
        ticker: Ticker symbol.

    Returns:
        List of BalanceSheet instances sorted ascending by fiscal_year_end.

    Logic:
        1. Map yfinance / EDGAR labels to BalanceSheet fields:
               "Cash And Cash Equivalents" → cash_and_equivalents
               "Short Term Investments" → short_term_investments
               "Current Assets" → total_current_assets
               "Total Assets" → total_assets
               "Current Liabilities" → total_current_liabilities
               "Short Long Term Debt" → short_term_debt
               "Long Term Debt" → long_term_debt
               "Stockholders Equity" → shareholders_equity
               "Retained Earnings" → retained_earnings
        2. Derive: total_debt = short_term_debt + long_term_debt
        3. Handle missing values gracefully (0.0 with warning)
    """
    ...


def _parse_cash_flow(
    df: pd.DataFrame,
    ticker: str,
) -> list[CashFlowStatement]:
    """
    Convert a raw cash flow DataFrame into a list of CashFlowStatement instances.

    Args:
        df:     Raw DataFrame with rows = line items, columns = period-end dates.
        ticker: Ticker symbol.

    Returns:
        List of CashFlowStatement instances sorted ascending by fiscal_year_end.

    Logic:
        1. Map labels to CashFlowStatement fields:
               "Operating Cash Flow" → operating_cash_flow
               "Capital Expenditure" → capital_expenditures (ensure negative sign)
               "Free Cash Flow" → free_cash_flow (if absent, compute: OCF + CapEx)
               "Cash Dividends Paid" → dividends_paid (ensure negative)
               "Repurchase Of Capital Stock" → stock_buybacks (ensure negative)
               "Proceeds From Issuance Of Debt" / "Repayment Of Debt" → net_debt_issuance
        2. Validate: free_cash_flow ≈ operating_cash_flow + capital_expenditures
           (warn if discrepancy > 5%)
    """
    ...

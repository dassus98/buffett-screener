"""
data_acquisition
================
Responsible for fetching, validating, and persisting raw financial data
from external sources (yfinance, SEC EDGAR, FRED, FMP).

Public surface:
    - fetch_universe()      → list of tickers matching universe config
    - fetch_financials()    → income statement, balance sheet, cash flow
    - fetch_market_data()   → current price, market cap, shares outstanding
    - fetch_macro_data()    → risk-free rate, CPI, macro indicators
    - run_data_quality()    → validation report for a given dataset
    - store / load via DuckDBStore
"""

from data_acquisition.universe import fetch_universe
from data_acquisition.financials import fetch_financials
from data_acquisition.market_data import fetch_market_data
from data_acquisition.macro_data import fetch_macro_data
from data_acquisition.data_quality import run_data_quality_checks
from data_acquisition.store import DuckDBStore

__all__ = [
    "fetch_universe",
    "fetch_financials",
    "fetch_market_data",
    "fetch_macro_data",
    "run_data_quality_checks",
    "DuckDBStore",
]

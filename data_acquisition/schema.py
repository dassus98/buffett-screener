"""
data_acquisition.schema
=======================
Canonical dataclass schemas for every dataset flowing through the pipeline.

All downstream modules (metrics_engine, screener, valuation_reports) consume
these typed structures rather than raw DataFrames, ensuring consistent column
names, units, and dtypes across the entire system.

Units convention:
    - Monetary values  → USD, in *thousands* (as reported on financial statements)
    - Ratios / margins → decimal (e.g. 0.25 = 25%), NOT percentage
    - Dates            → datetime.date
    - Rates            → decimal (e.g. 0.04 = 4%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class IncomeStatement:
    """
    Annual or TTM income statement for a single ticker and fiscal year.

    Fields map directly to SEC XBRL / yfinance column names where possible.
    All monetary amounts are in USD thousands.
    """

    ticker: str
    fiscal_year_end: date
    period: str                         # "annual" | "ttm"

    revenue: float                      # Total revenue / net sales
    cost_of_revenue: float              # COGS
    gross_profit: float                 # revenue - cost_of_revenue
    operating_income: float             # EBIT
    interest_expense: float             # gross interest paid on debt
    pretax_income: float
    income_tax: float
    net_income: float
    depreciation_amortization: float    # D&A from income stmt or cash flow
    ebitda: float                       # operating_income + D&A
    shares_diluted: float               # weighted average diluted shares (thousands)
    eps_diluted: float                  # net_income / shares_diluted


@dataclass
class BalanceSheet:
    """
    Annual balance sheet snapshot for a single ticker and fiscal year end.
    All monetary amounts are in USD thousands.
    """

    ticker: str
    fiscal_year_end: date

    cash_and_equivalents: float
    short_term_investments: float
    total_current_assets: float
    total_assets: float

    total_current_liabilities: float
    short_term_debt: float
    long_term_debt: float
    total_debt: float                   # short_term_debt + long_term_debt
    total_liabilities: float

    shareholders_equity: float          # book value of equity
    retained_earnings: float
    shares_outstanding: float           # period-end diluted shares (thousands)


@dataclass
class CashFlowStatement:
    """
    Annual cash flow statement for a single ticker and fiscal year.
    All monetary amounts are in USD thousands.
    """

    ticker: str
    fiscal_year_end: date
    period: str                         # "annual" | "ttm"

    operating_cash_flow: float          # CFO
    capital_expenditures: float         # always stored as a negative number (outflow)
    free_cash_flow: float               # operating_cash_flow + capital_expenditures
    dividends_paid: float               # negative outflow
    stock_buybacks: float               # negative outflow
    net_debt_issuance: float            # positive = new debt raised


@dataclass
class MarketData:
    """
    Point-in-time market data for a single ticker.

    Monetary amounts are in USD (full dollars, not thousands).
    """

    ticker: str
    as_of_date: date

    price: float                        # closing price
    market_cap: float                   # price × shares_outstanding (USD)
    enterprise_value: float             # market_cap + total_debt - cash
    shares_outstanding: float           # current diluted shares (thousands)
    beta: float                         # 5-year monthly beta vs. S&P 500
    avg_daily_volume_30d: float         # 30-day average daily trading volume
    fifty_two_week_high: float
    fifty_two_week_low: float


@dataclass
class MacroSnapshot:
    """
    Macro-economic indicators at a given date, sourced from FRED.
    All rates stored as decimals (e.g., 0.04 = 4%).
    """

    as_of_date: date

    treasury_10y_yield: float           # FRED series: DGS10
    treasury_2y_yield: float            # FRED series: DGS2
    cpi_yoy: float                      # CPI year-over-year inflation rate
    fed_funds_rate: float               # effective federal funds rate
    real_gdp_growth_yoy: float          # real GDP growth (annualised)
    sp500_pe_ratio: float               # Shiller CAPE or trailing P/E of index


@dataclass
class CompanyProfile:
    """
    Static company metadata fetched once and cached.
    """

    ticker: str
    name: str
    sector: str
    industry: str
    exchange: str
    country: str
    currency: str
    fiscal_year_end_month: int          # 1–12
    is_adr: bool
    is_spac: bool
    description: Optional[str] = None


@dataclass
class TickerDataBundle:
    """
    Aggregated container holding all raw data for a single ticker.
    Passed as a unit between pipeline stages.
    """

    profile: CompanyProfile
    income_statements: list[IncomeStatement] = field(default_factory=list)
    balance_sheets: list[BalanceSheet] = field(default_factory=list)
    cash_flow_statements: list[CashFlowStatement] = field(default_factory=list)
    market_data: Optional[MarketData] = None
    macro: Optional[MacroSnapshot] = None

    def latest_income(self) -> Optional[IncomeStatement]:
        """Return the most recent annual income statement by fiscal_year_end."""
        annual = [s for s in self.income_statements if s.period == "annual"]
        if not annual:
            return None
        return max(annual, key=lambda s: s.fiscal_year_end)

    def latest_balance_sheet(self) -> Optional[BalanceSheet]:
        """Return the most recent annual balance sheet by fiscal_year_end."""
        if not self.balance_sheets:
            return None
        return max(self.balance_sheets, key=lambda s: s.fiscal_year_end)

    def latest_cash_flow(self) -> Optional[CashFlowStatement]:
        """Return the most recent annual cash flow statement by fiscal_year_end."""
        annual = [s for s in self.cash_flow_statements if s.period == "annual"]
        if not annual:
            return None
        return max(annual, key=lambda s: s.fiscal_year_end)

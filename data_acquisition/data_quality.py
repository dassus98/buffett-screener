"""
data_acquisition.data_quality
==============================
Validates raw financial data before it enters the metrics engine.

Runs a suite of checks on TickerDataBundle objects to detect:
    - Missing required fields
    - Implausible values (e.g. negative revenue, total_assets < total_liabilities)
    - Accounting identities that must balance (Assets = Liabilities + Equity)
    - Insufficient historical depth (fewer years than the screener requires)
    - Stale data (last fiscal year older than expected)
    - Inconsistent signs (CapEx should be negative, dividends should be negative)

Each check produces a DataQualityIssue with a severity level:
    CRITICAL  — data is unusable; ticker should be skipped
    WARNING   — data is suspicious; flag for review but continue
    INFO      — minor note; logged but does not affect processing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from data_acquisition.schema import (
    BalanceSheet,
    CashFlowStatement,
    IncomeStatement,
    TickerDataBundle,
)


class Severity(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class DataQualityIssue:
    """Represents a single data quality finding for a ticker."""

    ticker: str
    check_name: str
    severity: Severity
    message: str
    field_name: Optional[str] = None
    period: Optional[str] = None        # fiscal year end date as string, if applicable


@dataclass
class DataQualityReport:
    """Aggregated quality report for a single ticker's TickerDataBundle."""

    ticker: str
    issues: list[DataQualityIssue] = field(default_factory=list)

    def is_critical(self) -> bool:
        """Return True if any CRITICAL issue exists."""
        ...

    def summary(self) -> str:
        """Return a human-readable summary string of all issues."""
        ...

    def to_dict(self) -> dict:
        """Serialise the report to a plain dict for storage in DuckDB."""
        ...


def run_data_quality_checks(bundle: TickerDataBundle) -> DataQualityReport:
    """
    Run the full suite of data quality checks on a TickerDataBundle.

    Args:
        bundle: A TickerDataBundle returned by fetch_financials() +
                fetch_market_data().

    Returns:
        DataQualityReport containing all detected issues. Callers should check
        report.is_critical() before passing the bundle to the metrics engine.

    Logic:
        Run each individual check function below and aggregate their issues:
            check_required_fields()
            check_historical_depth()
            check_accounting_identity()
            check_sign_conventions()
            check_implausible_values()
            check_data_staleness()
        Return the consolidated DataQualityReport.
    """
    ...


def check_required_fields(bundle: TickerDataBundle) -> list[DataQualityIssue]:
    """
    Verify that all non-optional fields in IncomeStatement, BalanceSheet, and
    CashFlowStatement are non-zero and non-NaN for each period.

    Args:
        bundle: TickerDataBundle to inspect.

    Returns:
        List of DataQualityIssue. Missing required fields → CRITICAL;
        missing optional fields → WARNING.

    Logic:
        Required fields for IncomeStatement: revenue, gross_profit, net_income,
            operating_income, shares_diluted
        Required fields for BalanceSheet: total_assets, shareholders_equity,
            total_current_liabilities
        Required fields for CashFlowStatement: operating_cash_flow, capital_expenditures
        Flag each missing/zero field with the period it belongs to.
    """
    ...


def check_historical_depth(
    bundle: TickerDataBundle,
    min_years: int = 7,
) -> list[DataQualityIssue]:
    """
    Verify that the bundle contains at least `min_years` of annual statements.

    Args:
        bundle:    TickerDataBundle to inspect.
        min_years: Minimum number of annual periods required.

    Returns:
        List with a single CRITICAL DataQualityIssue if depth is insufficient,
        or an empty list if depth is adequate.

    Logic:
        Count the number of IncomeStatement records with period="annual".
        If count < min_years, add a CRITICAL issue noting the actual depth.
    """
    ...


def check_accounting_identity(bundle: TickerDataBundle) -> list[DataQualityIssue]:
    """
    Verify that Assets ≈ Liabilities + Shareholders' Equity for each period.

    Args:
        bundle: TickerDataBundle to inspect.

    Returns:
        List of DataQualityIssue. A discrepancy > 1% of total assets → WARNING.

    Logic:
        For each BalanceSheet:
            implied_equity = total_assets - total_liabilities
            discrepancy_pct = abs(implied_equity - shareholders_equity) / total_assets
            If discrepancy_pct > 0.01: add a WARNING issue
    """
    ...


def check_sign_conventions(bundle: TickerDataBundle) -> list[DataQualityIssue]:
    """
    Verify that cash outflow items are stored with the expected sign.

    Args:
        bundle: TickerDataBundle to inspect.

    Returns:
        List of DataQualityIssue. Wrong sign → WARNING (we can correct it),
        or CRITICAL if the magnitude is implausibly large after sign correction.

    Logic:
        For each CashFlowStatement:
            capital_expenditures should be <= 0
            dividends_paid should be <= 0
            stock_buybacks should be <= 0
        For each IncomeStatement:
            interest_expense should be >= 0 (cost, positive by convention here)
        Add a WARNING for each violation, noting the period and sign observed.
    """
    ...


def check_implausible_values(bundle: TickerDataBundle) -> list[DataQualityIssue]:
    """
    Detect financially implausible values that indicate data errors.

    Args:
        bundle: TickerDataBundle to inspect.

    Returns:
        List of DataQualityIssue. CRITICAL for definitive impossibilities;
        WARNING for suspicious but theoretically possible values.

    Logic:
        Checks to run:
            - revenue <= 0 → CRITICAL
            - gross_profit > revenue → CRITICAL (impossible)
            - total_assets <= 0 → CRITICAL
            - net_income / revenue > 0.95 → WARNING (>95% net margin is suspicious)
            - operating_cash_flow / revenue > 0.80 → WARNING
            - shares_diluted <= 0 → CRITICAL
            - market_cap / revenue < 0.001 → WARNING (possible delisting)
    """
    ...


def check_data_staleness(
    bundle: TickerDataBundle,
    max_age_days: int = 400,
) -> list[DataQualityIssue]:
    """
    Check whether the most recent financial statements are recent enough.

    Args:
        bundle:       TickerDataBundle to inspect.
        max_age_days: Maximum acceptable age of the most recent annual period,
                      in calendar days from today. Default 400 accommodates
                      companies with fiscal year ends up to 13 months ago.

    Returns:
        List with a WARNING if the latest statement is older than max_age_days,
        or a CRITICAL if older than 2 × max_age_days (likely delisted or error).

    Logic:
        1. Find the max fiscal_year_end across all IncomeStatements
        2. Compute days_since = (today - max_fiscal_year_end).days
        3. Compare against max_age_days and 2 × max_age_days thresholds
    """
    ...

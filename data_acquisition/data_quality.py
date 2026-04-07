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

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

from data_acquisition.schema import (
    BalanceSheet,
    CashFlowStatement,
    IncomeStatement,
    TickerDataBundle,
)

logger = logging.getLogger(__name__)


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
        return any(issue.severity == Severity.CRITICAL for issue in self.issues)

    def summary(self) -> str:
        """Return a human-readable summary string of all issues."""
        n_critical = sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
        n_warning = sum(1 for i in self.issues if i.severity == Severity.WARNING)
        n_info = sum(1 for i in self.issues if i.severity == Severity.INFO)

        lines = [
            f"DataQualityReport for {self.ticker}: "
            f"{len(self.issues)} issues "
            f"(CRITICAL={n_critical}, WARNING={n_warning}, INFO={n_info})"
        ]
        # Show first 5 issues as examples
        for issue in self.issues[:5]:
            period_str = f" [{issue.period}]" if issue.period else ""
            field_str = f" (field: {issue.field_name})" if issue.field_name else ""
            lines.append(
                f"  [{issue.severity.value}] {issue.check_name}{period_str}{field_str}: "
                f"{issue.message}"
            )
        if len(self.issues) > 5:
            lines.append(f"  ... and {len(self.issues) - 5} more.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise the report to a plain dict for storage in DuckDB."""
        return {
            "ticker": self.ticker,
            "is_critical": self.is_critical(),
            "issue_count": len(self.issues),
            "summary": self.summary(),
            "issues": [
                {
                    "ticker": issue.ticker,
                    "check_name": issue.check_name,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "field_name": issue.field_name,
                    "period": issue.period,
                }
                for issue in self.issues
            ],
        }


def run_data_quality_checks(bundle: TickerDataBundle) -> DataQualityReport:
    """
    Run the full suite of data quality checks on a TickerDataBundle.

    Args:
        bundle: A TickerDataBundle returned by fetch_financials() +
                fetch_market_data().

    Returns:
        DataQualityReport containing all detected issues.
    """
    ticker = bundle.profile.ticker
    report = DataQualityReport(ticker=ticker)

    check_functions = [
        check_required_fields,
        check_historical_depth,
        check_accounting_identity,
        check_sign_conventions,
        check_implausible_values,
        check_data_staleness,
    ]

    for check_fn in check_functions:
        try:
            issues = check_fn(bundle)
            report.issues.extend(issues)
        except Exception as exc:
            logger.warning(
                "Data quality check %s raised an unexpected error for ticker=%s: %s. "
                "Adding a CRITICAL issue.",
                check_fn.__name__, ticker, exc,
            )
            report.issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name=check_fn.__name__,
                    severity=Severity.CRITICAL,
                    message=f"Check raised unexpected exception: {exc}",
                )
            )

    severity_counts = {
        Severity.CRITICAL: sum(1 for i in report.issues if i.severity == Severity.CRITICAL),
        Severity.WARNING: sum(1 for i in report.issues if i.severity == Severity.WARNING),
        Severity.INFO: sum(1 for i in report.issues if i.severity == Severity.INFO),
    }
    logger.info(
        "Data quality checks for ticker=%s: %d CRITICAL, %d WARNING, %d INFO.",
        ticker,
        severity_counts[Severity.CRITICAL],
        severity_counts[Severity.WARNING],
        severity_counts[Severity.INFO],
    )

    return report


def check_required_fields(bundle: TickerDataBundle) -> list[DataQualityIssue]:
    """
    Verify that all non-optional fields in IncomeStatement, BalanceSheet, and
    CashFlowStatement are non-zero and non-NaN for each period.

    Required fields:
        IncomeStatement: revenue, gross_profit, net_income, operating_income, shares_diluted
        BalanceSheet: total_assets, shareholders_equity, total_current_liabilities
        CashFlowStatement: operating_cash_flow, capital_expenditures
    """
    ticker = bundle.profile.ticker
    issues: list[DataQualityIssue] = []

    REQUIRED_INCOME = [
        "revenue", "gross_profit", "net_income", "operating_income", "shares_diluted"
    ]
    REQUIRED_BALANCE = [
        "total_assets", "shareholders_equity", "total_current_liabilities"
    ]
    REQUIRED_CASHFLOW = [
        "operating_cash_flow", "capital_expenditures"
    ]

    for stmt in bundle.income_statements:
        period_str = str(stmt.fiscal_year_end)
        for field_name in REQUIRED_INCOME:
            val = getattr(stmt, field_name, None)
            if val is None or (isinstance(val, float) and (math.isnan(val) or val == 0.0)):
                issues.append(
                    DataQualityIssue(
                        ticker=ticker,
                        check_name="check_required_fields",
                        severity=Severity.CRITICAL,
                        message=(
                            f"Required field '{field_name}' is missing or zero "
                            f"in income statement for period {period_str}."
                        ),
                        field_name=field_name,
                        period=period_str,
                    )
                )

    for stmt in bundle.balance_sheets:
        period_str = str(stmt.fiscal_year_end)
        for field_name in REQUIRED_BALANCE:
            val = getattr(stmt, field_name, None)
            if val is None or (isinstance(val, float) and (math.isnan(val) or val == 0.0)):
                issues.append(
                    DataQualityIssue(
                        ticker=ticker,
                        check_name="check_required_fields",
                        severity=Severity.CRITICAL,
                        message=(
                            f"Required field '{field_name}' is missing or zero "
                            f"in balance sheet for period {period_str}."
                        ),
                        field_name=field_name,
                        period=period_str,
                    )
                )

    for stmt in bundle.cash_flow_statements:
        period_str = str(stmt.fiscal_year_end)
        for field_name in REQUIRED_CASHFLOW:
            val = getattr(stmt, field_name, None)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                issues.append(
                    DataQualityIssue(
                        ticker=ticker,
                        check_name="check_required_fields",
                        severity=Severity.CRITICAL,
                        message=(
                            f"Required field '{field_name}' is missing "
                            f"in cash flow statement for period {period_str}."
                        ),
                        field_name=field_name,
                        period=period_str,
                    )
                )

    return issues


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
        List with a single CRITICAL DataQualityIssue if depth is insufficient.
    """
    ticker = bundle.profile.ticker
    annual_income = [s for s in bundle.income_statements if s.period == "annual"]
    actual_depth = len(annual_income)

    if actual_depth < min_years:
        return [
            DataQualityIssue(
                ticker=ticker,
                check_name="check_historical_depth",
                severity=Severity.CRITICAL,
                message=(
                    f"Only {actual_depth} annual income statement period(s) available; "
                    f"minimum required is {min_years}. "
                    "Historical depth is insufficient for reliable screening."
                ),
                field_name="income_statements",
            )
        ]

    return []


def check_accounting_identity(bundle: TickerDataBundle) -> list[DataQualityIssue]:
    """
    Verify that Assets ≈ Liabilities + Shareholders' Equity for each period.

    A discrepancy > 1% of total_assets → WARNING.
    """
    ticker = bundle.profile.ticker
    issues: list[DataQualityIssue] = []

    for bs in bundle.balance_sheets:
        period_str = str(bs.fiscal_year_end)

        total_assets = bs.total_assets
        total_liabilities = bs.total_liabilities
        shareholders_equity = bs.shareholders_equity

        # Skip if any of the key fields are NaN
        if any(math.isnan(v) for v in [total_assets, total_liabilities, shareholders_equity]):
            logger.debug(
                "ticker=%s period=%s: skipping accounting identity check "
                "(one or more fields are NaN).",
                ticker, period_str,
            )
            continue

        if total_assets == 0:
            continue

        implied_equity = total_assets - total_liabilities
        discrepancy = abs(implied_equity - shareholders_equity)
        discrepancy_pct = discrepancy / abs(total_assets)

        if discrepancy_pct > 0.01:
            issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name="check_accounting_identity",
                    severity=Severity.WARNING,
                    message=(
                        f"Accounting identity violated for period {period_str}: "
                        f"implied_equity ({implied_equity:,.0f}) differs from "
                        f"reported shareholders_equity ({shareholders_equity:,.0f}) "
                        f"by {discrepancy_pct:.1%} of total_assets."
                    ),
                    field_name="shareholders_equity",
                    period=period_str,
                )
            )

    return issues


def check_sign_conventions(bundle: TickerDataBundle) -> list[DataQualityIssue]:
    """
    Verify that cash outflow items are stored with the expected sign.

    Checks:
        CashFlowStatement: capital_expenditures <= 0, dividends_paid <= 0, stock_buybacks <= 0
        IncomeStatement: interest_expense >= 0 (cost, positive convention)
    """
    ticker = bundle.profile.ticker
    issues: list[DataQualityIssue] = []

    for cf in bundle.cash_flow_statements:
        period_str = str(cf.fiscal_year_end)

        # capital_expenditures should be negative
        capex = cf.capital_expenditures
        if not math.isnan(capex) and capex > 0:
            issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name="check_sign_conventions",
                    severity=Severity.WARNING,
                    message=(
                        f"capital_expenditures is positive ({capex:,.0f}) for period "
                        f"{period_str}. Expected negative (cash outflow). "
                        "Data may need sign correction."
                    ),
                    field_name="capital_expenditures",
                    period=period_str,
                )
            )

        # dividends_paid should be negative
        divs = cf.dividends_paid
        if not math.isnan(divs) and divs > 0:
            issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name="check_sign_conventions",
                    severity=Severity.WARNING,
                    message=(
                        f"dividends_paid is positive ({divs:,.0f}) for period {period_str}. "
                        "Expected negative (cash outflow)."
                    ),
                    field_name="dividends_paid",
                    period=period_str,
                )
            )

        # stock_buybacks should be negative
        buybacks = cf.stock_buybacks
        if not math.isnan(buybacks) and buybacks > 0:
            issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name="check_sign_conventions",
                    severity=Severity.WARNING,
                    message=(
                        f"stock_buybacks is positive ({buybacks:,.0f}) for period {period_str}. "
                        "Expected negative (cash outflow)."
                    ),
                    field_name="stock_buybacks",
                    period=period_str,
                )
            )

    for stmt in bundle.income_statements:
        period_str = str(stmt.fiscal_year_end)
        interest = stmt.interest_expense
        # interest_expense stored as positive cost by convention
        if not math.isnan(interest) and interest < 0:
            issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name="check_sign_conventions",
                    severity=Severity.WARNING,
                    message=(
                        f"interest_expense is negative ({interest:,.0f}) for period {period_str}. "
                        "Expected positive (cost). Check source data."
                    ),
                    field_name="interest_expense",
                    period=period_str,
                )
            )

    return issues


def check_implausible_values(bundle: TickerDataBundle) -> list[DataQualityIssue]:
    """
    Detect financially implausible values that indicate data errors.

    Checks (CRITICAL):
        - revenue <= 0
        - gross_profit > revenue
        - total_assets <= 0
        - shares_diluted <= 0

    Checks (WARNING):
        - net_income / revenue > 0.95
        - operating_cash_flow / revenue > 0.80
        - market_cap / revenue < 0.001
    """
    ticker = bundle.profile.ticker
    issues: list[DataQualityIssue] = []

    for stmt in bundle.income_statements:
        period_str = str(stmt.fiscal_year_end)

        revenue = stmt.revenue
        gross_profit = stmt.gross_profit
        net_income = stmt.net_income
        operating_income = stmt.operating_income
        shares_diluted = stmt.shares_diluted

        # revenue <= 0
        if not math.isnan(revenue) and revenue <= 0:
            issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name="check_implausible_values",
                    severity=Severity.CRITICAL,
                    message=(
                        f"revenue is non-positive ({revenue:,.0f}) for period {period_str}. "
                        "This is financially impossible for an operating company."
                    ),
                    field_name="revenue",
                    period=period_str,
                )
            )

        # gross_profit > revenue (impossible)
        if (
            not math.isnan(gross_profit)
            and not math.isnan(revenue)
            and revenue > 0
            and gross_profit > revenue
        ):
            issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name="check_implausible_values",
                    severity=Severity.CRITICAL,
                    message=(
                        f"gross_profit ({gross_profit:,.0f}) > revenue ({revenue:,.0f}) "
                        f"for period {period_str}. This is arithmetically impossible."
                    ),
                    field_name="gross_profit",
                    period=period_str,
                )
            )

        # shares_diluted <= 0
        if not math.isnan(shares_diluted) and shares_diluted <= 0:
            issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name="check_implausible_values",
                    severity=Severity.CRITICAL,
                    message=(
                        f"shares_diluted is non-positive ({shares_diluted:,.0f}) "
                        f"for period {period_str}."
                    ),
                    field_name="shares_diluted",
                    period=period_str,
                )
            )

        # net margin > 95% (suspicious but not impossible for some financials)
        if (
            not math.isnan(net_income)
            and not math.isnan(revenue)
            and revenue > 0
        ):
            net_margin = net_income / revenue
            if net_margin > 0.95:
                issues.append(
                    DataQualityIssue(
                        ticker=ticker,
                        check_name="check_implausible_values",
                        severity=Severity.WARNING,
                        message=(
                            f"net margin is {net_margin:.1%} for period {period_str}. "
                            "Net margin > 95% is highly unusual; verify data."
                        ),
                        field_name="net_income",
                        period=period_str,
                    )
                )

    for bs in bundle.balance_sheets:
        period_str = str(bs.fiscal_year_end)
        total_assets = bs.total_assets

        # total_assets <= 0
        if not math.isnan(total_assets) and total_assets <= 0:
            issues.append(
                DataQualityIssue(
                    ticker=ticker,
                    check_name="check_implausible_values",
                    severity=Severity.CRITICAL,
                    message=(
                        f"total_assets is non-positive ({total_assets:,.0f}) "
                        f"for period {period_str}."
                    ),
                    field_name="total_assets",
                    period=period_str,
                )
            )

    # Cash flow margin check (OCF / revenue)
    for cf in bundle.cash_flow_statements:
        period_str = str(cf.fiscal_year_end)
        ocf = cf.operating_cash_flow

        # Find matching income statement for revenue
        matching_income = next(
            (s for s in bundle.income_statements if s.fiscal_year_end == cf.fiscal_year_end),
            None,
        )
        if matching_income is not None:
            revenue = matching_income.revenue
            if (
                not math.isnan(ocf)
                and not math.isnan(revenue)
                and revenue > 0
                and ocf / revenue > 0.80
            ):
                issues.append(
                    DataQualityIssue(
                        ticker=ticker,
                        check_name="check_implausible_values",
                        severity=Severity.WARNING,
                        message=(
                            f"operating_cash_flow / revenue = {ocf / revenue:.1%} "
                            f"for period {period_str}. > 80% is unusual; verify data."
                        ),
                        field_name="operating_cash_flow",
                        period=period_str,
                    )
                )

    # Market cap / revenue sanity check
    if bundle.market_data is not None:
        market_cap = bundle.market_data.market_cap
        latest_income = bundle.latest_income()
        if latest_income is not None:
            revenue = latest_income.revenue
            # revenue is in USD thousands; market_cap is in full USD
            revenue_usd = revenue * 1_000.0
            if (
                not math.isnan(market_cap)
                and not math.isnan(revenue_usd)
                and revenue_usd > 0
                and market_cap / revenue_usd < 0.001
            ):
                issues.append(
                    DataQualityIssue(
                        ticker=ticker,
                        check_name="check_implausible_values",
                        severity=Severity.WARNING,
                        message=(
                            f"market_cap / revenue = {market_cap / revenue_usd:.4f} "
                            "(< 0.001). Possible delisting or data error."
                        ),
                        field_name="market_cap",
                    )
                )

    return issues


def check_data_staleness(
    bundle: TickerDataBundle,
    max_age_days: int = 400,
) -> list[DataQualityIssue]:
    """
    Check whether the most recent financial statements are recent enough.

    Returns:
        WARNING if latest statement is older than max_age_days;
        CRITICAL if older than 2 × max_age_days (likely delisted or data error).
    """
    ticker = bundle.profile.ticker
    issues: list[DataQualityIssue] = []

    if not bundle.income_statements:
        issues.append(
            DataQualityIssue(
                ticker=ticker,
                check_name="check_data_staleness",
                severity=Severity.CRITICAL,
                message=(
                    "No income statements present in bundle. "
                    "Cannot assess data staleness."
                ),
            )
        )
        return issues

    max_fiscal_year_end = max(s.fiscal_year_end for s in bundle.income_statements)
    today = date.today()
    days_since = (today - max_fiscal_year_end).days

    if days_since > max_age_days * 2:
        issues.append(
            DataQualityIssue(
                ticker=ticker,
                check_name="check_data_staleness",
                severity=Severity.CRITICAL,
                message=(
                    f"Most recent fiscal year end is {max_fiscal_year_end} "
                    f"({days_since} days ago), which exceeds 2× the max_age_days threshold "
                    f"({max_age_days * 2} days). Data likely stale or company delisted."
                ),
                field_name="fiscal_year_end",
                period=str(max_fiscal_year_end),
            )
        )
    elif days_since > max_age_days:
        issues.append(
            DataQualityIssue(
                ticker=ticker,
                check_name="check_data_staleness",
                severity=Severity.WARNING,
                message=(
                    f"Most recent fiscal year end is {max_fiscal_year_end} "
                    f"({days_since} days ago), exceeding the max_age_days threshold "
                    f"({max_age_days} days). Consider re-fetching financial data."
                ),
                field_name="fiscal_year_end",
                period=str(max_fiscal_year_end),
            )
        )

    return issues

"""
tests.test_data_quality
========================
Unit tests for the data_acquisition.data_quality module.

All tests construct synthetic TickerDataBundle objects with controlled
defects and verify that the appropriate DataQualityIssue is raised with
the correct severity.

No external I/O. All tests should complete in under 1 second.

Coverage targets:
    data_quality.check_required_fields      — missing fields → CRITICAL
    data_quality.check_historical_depth     — insufficient years → CRITICAL
    data_quality.check_accounting_identity  — balance sheet imbalance → WARNING
    data_quality.check_sign_conventions     — wrong-sign CapEx → WARNING
    data_quality.check_implausible_values   — impossible ratios → CRITICAL/WARNING
    data_quality.check_data_staleness       — stale data → WARNING/CRITICAL
    data_quality.run_data_quality_checks    — aggregation and is_critical()
"""

import pytest
from datetime import date, timedelta

from data_acquisition.data_quality import (
    DataQualityReport,
    Severity,
    check_required_fields,
    check_historical_depth,
    check_accounting_identity,
    check_sign_conventions,
    check_implausible_values,
    check_data_staleness,
    run_data_quality_checks,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_valid_bundle(num_years: int = 10):
    """
    Create a TickerDataBundle with valid, complete data for `num_years`.

    All financial figures are set to plausible values that pass all quality checks.
    """
    ...


def make_income_stmt(
    ticker: str = "TEST",
    fiscal_year_end: date = date(2024, 12, 31),
    **overrides,
):
    """Create an IncomeStatement with valid defaults; apply overrides."""
    ...


def make_balance_sheet(
    ticker: str = "TEST",
    fiscal_year_end: date = date(2024, 12, 31),
    **overrides,
):
    """Create a BalanceSheet with valid defaults; apply overrides."""
    ...


def make_cash_flow(
    ticker: str = "TEST",
    fiscal_year_end: date = date(2024, 12, 31),
    **overrides,
):
    """Create a CashFlowStatement with valid defaults; apply overrides."""
    ...


# ---------------------------------------------------------------------------
# check_required_fields
# ---------------------------------------------------------------------------

class TestCheckRequiredFields:
    def test_no_issues_on_complete_bundle(self):
        """Fully populated bundle → empty issues list."""
        ...

    def test_critical_on_zero_revenue(self):
        """revenue = 0 in any period → CRITICAL issue."""
        ...

    def test_critical_on_zero_total_assets(self):
        ...

    def test_critical_on_zero_shares_diluted(self):
        ...

    def test_warns_on_missing_optional_field(self):
        """dividends_paid = 0 → WARNING (optional, but surprising for large companies)."""
        ...

    def test_issue_includes_correct_period(self):
        """DataQualityIssue.period should match the fiscal_year_end where the problem occurs."""
        ...


# ---------------------------------------------------------------------------
# check_historical_depth
# ---------------------------------------------------------------------------

class TestCheckHistoricalDepth:
    def test_no_issue_with_sufficient_years(self):
        """Bundle with 10 years of income statements → no issues."""
        ...

    def test_critical_with_insufficient_years(self):
        """Bundle with only 3 years of income statements → CRITICAL."""
        ...

    def test_exactly_at_minimum_passes(self):
        """Exactly min_years annual statements → no issue."""
        ...


# ---------------------------------------------------------------------------
# check_accounting_identity
# ---------------------------------------------------------------------------

class TestCheckAccountingIdentity:
    def test_no_issue_on_balanced_sheet(self):
        """total_assets == total_liabilities + shareholders_equity → no issues."""
        ...

    def test_warns_on_small_discrepancy(self):
        """Discrepancy > 1% of total assets → WARNING."""
        ...

    def test_no_issue_on_tiny_rounding_error(self):
        """Discrepancy < 1% → no issue (rounding tolerance)."""
        ...


# ---------------------------------------------------------------------------
# check_sign_conventions
# ---------------------------------------------------------------------------

class TestCheckSignConventions:
    def test_no_issue_on_correct_signs(self):
        """CapEx < 0, dividends < 0, buybacks < 0, interest_expense > 0 → clean."""
        ...

    def test_warns_on_positive_capex(self):
        """CapEx stored as positive → WARNING."""
        ...

    def test_warns_on_positive_dividends(self):
        ...

    def test_warns_on_negative_interest_expense(self):
        ...


# ---------------------------------------------------------------------------
# check_implausible_values
# ---------------------------------------------------------------------------

class TestCheckImplausibleValues:
    def test_no_issue_on_normal_company(self):
        ...

    def test_critical_on_negative_revenue(self):
        """revenue < 0 is impossible → CRITICAL."""
        ...

    def test_critical_when_gross_profit_exceeds_revenue(self):
        """gross_profit > revenue → accounting impossibility → CRITICAL."""
        ...

    def test_warns_on_suspiciously_high_net_margin(self):
        """net_income / revenue > 0.95 → WARNING."""
        ...

    def test_critical_on_negative_total_assets(self):
        ...


# ---------------------------------------------------------------------------
# check_data_staleness
# ---------------------------------------------------------------------------

class TestCheckDataStaleness:
    def test_no_issue_on_fresh_data(self):
        """Latest fiscal year < 400 days ago → no issue."""
        ...

    def test_warns_on_stale_data(self):
        """Latest fiscal year > 400 days ago → WARNING."""
        ...

    def test_critical_on_very_old_data(self):
        """Latest fiscal year > 800 days ago → CRITICAL."""
        ...


# ---------------------------------------------------------------------------
# run_data_quality_checks (integration)
# ---------------------------------------------------------------------------

class TestRunDataQualityChecks:
    def test_returns_report_object(self):
        """run_data_quality_checks() should return a DataQualityReport instance."""
        ...

    def test_is_critical_false_on_clean_bundle(self):
        ...

    def test_is_critical_true_when_critical_issue_present(self):
        ...

    def test_summary_is_non_empty_string(self):
        ...

    def test_to_dict_is_serialisable(self):
        """DataQualityReport.to_dict() output should be JSON-serialisable."""
        ...

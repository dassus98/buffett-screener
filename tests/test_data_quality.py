"""Unit tests for data_acquisition/data_quality.py.

All external I/O (yfinance, filesystem, config) is mocked.
No real network requests or disk writes are required.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from data_acquisition.data_quality import (
    _build_quality_report_df,
    _collect_all_tickers,
    _compare_field,
    _count_substitutions_per_ticker,
    _count_years_available,
    _empty_cross_validate_df,
    _filter_to_ticker,
    _find_missing_critical_fields,
    assess_ticker_quality,
    cross_validate_sample,
    run_data_quality_check,
)
from data_acquisition.schema import LINE_ITEM_MAP, get_drop_required_fields


# ===========================================================================
# Fixtures / helpers
# ===========================================================================

def _make_income_df(
    ticker: str = "AAPL",
    n_years: int = 10,
    null_fields: list[str] | None = None,
    n_null_years: int | None = None,
) -> pd.DataFrame:
    """Build a minimal income statement DataFrame for testing.

    Parameters
    ----------
    null_fields:
        Fields to set entirely to NaN (or partially, if n_null_years given).
    n_null_years:
        If given, sets only the first ``n_null_years`` rows of each null_field
        to NaN, leaving the rest non-null.
    """
    years = list(range(2014, 2014 + n_years))
    data: dict[str, Any] = {
        "ticker": [ticker] * n_years,
        "fiscal_year": years,
        "net_income": [96_995.0] * n_years,
        "total_revenue": [383_285.0] * n_years,
        "gross_profit": [169_148.0] * n_years,
        "operating_income": [114_301.0] * n_years,
        "eps_diluted": [6.13] * n_years,
        "shares_outstanding_diluted": [15_812_547.0] * n_years,
        "sga": [24_609.0] * n_years,
        "interest_expense": [3_933.0] * n_years,
    }
    df = pd.DataFrame(data)
    if null_fields:
        for field in null_fields:
            if field in df.columns:
                if n_null_years is not None:
                    df.loc[: n_null_years - 1, field] = float("nan")
                else:
                    df[field] = float("nan")
    return df


def _make_balance_df(
    ticker: str = "AAPL",
    n_years: int = 10,
    null_fields: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal balance sheet DataFrame for testing."""
    years = list(range(2014, 2014 + n_years))
    data: dict[str, Any] = {
        "ticker": [ticker] * n_years,
        "fiscal_year": years,
        "long_term_debt": [111_000.0] * n_years,
        "shareholders_equity": [62_146.0] * n_years,
        "treasury_stock": [-3_000_000.0] * n_years,
    }
    df = pd.DataFrame(data)
    if null_fields:
        for field in null_fields:
            if field in df.columns:
                df[field] = float("nan")
    return df


def _make_cashflow_df(
    ticker: str = "AAPL",
    n_years: int = 10,
    null_fields: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal cash flow statement DataFrame for testing."""
    years = list(range(2014, 2014 + n_years))
    data: dict[str, Any] = {
        "ticker": [ticker] * n_years,
        "fiscal_year": years,
        "depreciation_amortization": [12_547.0] * n_years,
        "capital_expenditures": [-10_959.0] * n_years,
        "working_capital_change": [3_651.0] * n_years,
    }
    df = pd.DataFrame(data)
    if null_fields:
        for field in null_fields:
            if field in df.columns:
                df[field] = float("nan")
    return df


def _make_financials(
    tickers: list[str] | None = None,
    n_years: int = 10,
) -> dict[str, pd.DataFrame]:
    """Build a financials dict with complete data for the given tickers."""
    if tickers is None:
        tickers = ["AAPL"]
    income_frames = [_make_income_df(t, n_years) for t in tickers]
    balance_frames = [_make_balance_df(t, n_years) for t in tickers]
    cash_frames = [_make_cashflow_df(t, n_years) for t in tickers]
    return {
        "income_statement": pd.concat(income_frames, ignore_index=True),
        "balance_sheet": pd.concat(balance_frames, ignore_index=True),
        "cash_flow": pd.concat(cash_frames, ignore_index=True),
    }


# ===========================================================================
# Tests: _count_years_available
# ===========================================================================

class TestCountYearsAvailable:
    def test_all_dfs_same_years(self) -> None:
        inc = _make_income_df(n_years=10)
        bal = _make_balance_df(n_years=10)
        cf = _make_cashflow_df(n_years=10)
        assert _count_years_available(inc, bal, cf) == 10

    def test_empty_dfs_returns_zero(self) -> None:
        result = _count_years_available(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert result == 0

    def test_different_year_coverage_unions_correctly(self) -> None:
        """income_df covers 2014–2021 (8), balance/cashflow cover 2014–2023 (10)."""
        inc = _make_income_df(n_years=8)
        bal = _make_balance_df(n_years=10)
        cf = _make_cashflow_df(n_years=10)
        assert _count_years_available(inc, bal, cf) == 10

    def test_none_dfs_handled_gracefully(self) -> None:
        inc = _make_income_df(n_years=5)
        result = _count_years_available(inc, None, None)  # type: ignore[arg-type]
        assert result == 5

    def test_three_years_returns_three(self) -> None:
        inc = _make_income_df(n_years=3)
        bal = _make_balance_df(n_years=3)
        cf = _make_cashflow_df(n_years=3)
        assert _count_years_available(inc, bal, cf) == 3


# ===========================================================================
# Tests: _find_missing_critical_fields
# ===========================================================================

class TestFindMissingCriticalFields:
    def test_complete_data_no_missing(self) -> None:
        stmt_map = {
            "income_statement": _make_income_df(n_years=10),
            "balance_sheet": _make_balance_df(n_years=10),
            "cash_flow": _make_cashflow_df(n_years=10),
        }
        missing = _find_missing_critical_fields(stmt_map, min_years=8)
        assert missing == []

    def test_net_income_all_null_flagged(self) -> None:
        stmt_map = {
            "income_statement": _make_income_df(n_years=10, null_fields=["net_income"]),
            "balance_sheet": _make_balance_df(n_years=10),
            "cash_flow": _make_cashflow_df(n_years=10),
        }
        missing = _find_missing_critical_fields(stmt_map, min_years=8)
        assert "net_income" in missing

    def test_field_with_exactly_min_years_not_flagged(self) -> None:
        """Exactly min_years (8) non-null values — not below threshold."""
        # 10 rows, first 2 are NaN → 8 non-null
        inc = _make_income_df(n_years=10, null_fields=["net_income"], n_null_years=2)
        stmt_map = {
            "income_statement": inc,
            "balance_sheet": _make_balance_df(n_years=10),
            "cash_flow": _make_cashflow_df(n_years=10),
        }
        missing = _find_missing_critical_fields(stmt_map, min_years=8)
        assert "net_income" not in missing

    def test_field_one_below_threshold_flagged(self) -> None:
        """7 non-null values (< 8) → flagged."""
        inc = _make_income_df(n_years=10, null_fields=["net_income"], n_null_years=3)
        stmt_map = {
            "income_statement": inc,
            "balance_sheet": _make_balance_df(n_years=10),
            "cash_flow": _make_cashflow_df(n_years=10),
        }
        missing = _find_missing_critical_fields(stmt_map, min_years=8)
        assert "net_income" in missing

    def test_missing_statement_df_flags_its_required_fields(self) -> None:
        """Absent balance sheet → both balance_sheet required fields flagged."""
        stmt_map = {
            "income_statement": _make_income_df(n_years=10),
            "balance_sheet": None,  # type: ignore[dict-item]
            "cash_flow": _make_cashflow_df(n_years=10),
        }
        missing = _find_missing_critical_fields(stmt_map, min_years=8)
        assert "long_term_debt" in missing
        assert "shareholders_equity" in missing

    def test_only_drop_required_fields_checked(self) -> None:
        """Non-required (drop_if_missing=False) fields never appear in missing list."""
        non_required = [f for f, item in LINE_ITEM_MAP.items() if not item.drop_if_missing]
        stmt_map = {
            "income_statement": _make_income_df(n_years=10),
            "balance_sheet": _make_balance_df(n_years=10),
            "cash_flow": _make_cashflow_df(n_years=10),
        }
        missing = _find_missing_critical_fields(stmt_map, min_years=8)
        for field in non_required:
            assert field not in missing, f"Non-required field {field} must not appear in missing"

    def test_cashflow_field_checked_in_cashflow_df(self) -> None:
        """capital_expenditures lives in cash_flow — blank CF df triggers flag."""
        stmt_map = {
            "income_statement": _make_income_df(n_years=10),
            "balance_sheet": _make_balance_df(n_years=10),
            "cash_flow": _make_cashflow_df(n_years=10, null_fields=["capital_expenditures"]),
        }
        missing = _find_missing_critical_fields(stmt_map, min_years=8)
        assert "capital_expenditures" in missing


# ===========================================================================
# Tests: assess_ticker_quality
# ===========================================================================

class TestAssessTickerQuality:
    def test_complete_data_drop_false(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        assert result["drop"] is False
        assert result["drop_reason"] is None

    def test_returns_all_required_keys(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        expected_keys = {
            "ticker", "years_available", "missing_critical_fields",
            "substitutions_count", "drop", "drop_reason",
        }
        assert set(result.keys()) == expected_keys

    def test_ticker_in_result(self) -> None:
        result = assess_ticker_quality(
            "MSFT",
            _make_income_df(ticker="MSFT", n_years=10),
            _make_balance_df(ticker="MSFT", n_years=10),
            _make_cashflow_df(ticker="MSFT", n_years=10),
        )
        assert result["ticker"] == "MSFT"

    def test_years_available_correct(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        assert result["years_available"] == 10

    def test_missing_critical_fields_empty_when_passing(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        assert result["missing_critical_fields"] == []

    def test_missing_critical_fields_is_list(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        assert isinstance(result["missing_critical_fields"], list)

    def test_insufficient_years_drop_true(self) -> None:
        result = assess_ticker_quality(
            "TINY",
            _make_income_df(n_years=3),
            _make_balance_df(n_years=3),
            _make_cashflow_df(n_years=3),
        )
        assert result["drop"] is True
        assert result["years_available"] == 3
        assert "Insufficient fiscal year coverage" in result["drop_reason"]

    def test_insufficient_years_drop_reason_mentions_count(self) -> None:
        result = assess_ticker_quality(
            "TINY",
            _make_income_df(n_years=4),
            _make_balance_df(n_years=4),
            _make_cashflow_df(n_years=4),
        )
        assert "4 years" in result["drop_reason"]

    def test_missing_critical_field_drop_true(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10, null_fields=["net_income"]),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        assert result["drop"] is True
        assert "net_income" in result["missing_critical_fields"]

    def test_missing_critical_field_reason_mentions_field_name(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10, null_fields=["net_income"]),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        assert "net_income" in result["drop_reason"]

    def test_empty_dfs_drop_true(self) -> None:
        result = assess_ticker_quality(
            "GHOST", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert result["drop"] is True
        assert result["years_available"] == 0

    def test_substitutions_count_passed_through(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
            substitutions_count=3,
        )
        assert result["substitutions_count"] == 3

    def test_substitutions_default_zero(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        assert result["substitutions_count"] == 0

    def test_both_failure_reasons_in_drop_reason(self) -> None:
        """Short years AND missing field → both clauses appear in drop_reason."""
        result = assess_ticker_quality(
            "FAIL",
            _make_income_df(n_years=3, null_fields=["net_income"]),
            _make_balance_df(n_years=3),
            _make_cashflow_df(n_years=3),
        )
        assert result["drop"] is True
        assert "Insufficient" in result["drop_reason"]

    def test_drop_reason_none_when_passing(self) -> None:
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        assert result["drop_reason"] is None

    def test_exactly_eight_years_passes(self) -> None:
        """8 years = exactly min_history_years → should pass (not dropped)."""
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=8),
            _make_balance_df(n_years=8),
            _make_cashflow_df(n_years=8),
        )
        assert result["drop"] is False

    @patch("data_acquisition.data_quality.get_config")
    def test_uses_min_field_coverage_years_for_field_check(
        self, mock_cfg: MagicMock,
    ) -> None:
        """Field coverage must use data_quality.min_field_coverage_years, NOT
        universe.min_history_years. When they differ, only min_field_coverage_years
        governs the per-field non-null check (docs/DATA_SOURCES.md §4)."""
        # year count threshold = 5 → 8 years pass year check
        # field coverage threshold = 9 → 8 non-null per field FAILS
        mock_cfg.return_value = {
            "universe": {"min_history_years": 5},
            "data_quality": {
                "min_field_coverage_years": 9,
                "max_substitutions_before_flag": 2,
            },
        }
        result = assess_ticker_quality(
            "TEST",
            _make_income_df(n_years=8),
            _make_balance_df(n_years=8),
            _make_cashflow_df(n_years=8),
        )
        # Year coverage passes (8 >= 5).
        assert result["years_available"] == 8
        # Field coverage fails (8 non-null < 9 required) → drop.
        assert result["drop"] is True
        assert len(result["missing_critical_fields"]) > 0
        # Drop reason should mention "Critical fields", not "Insufficient fiscal year".
        assert "Critical fields" in result["drop_reason"]
        assert "Insufficient" not in result["drop_reason"]

    @patch("data_acquisition.data_quality.get_config")
    def test_min_field_coverage_defaults_to_min_history_years(
        self, mock_cfg: MagicMock,
    ) -> None:
        """If min_field_coverage_years is absent from config, it should fall back
        to min_history_years as a safe default."""
        mock_cfg.return_value = {
            "universe": {"min_history_years": 8},
            "data_quality": {
                # min_field_coverage_years intentionally omitted.
                "max_substitutions_before_flag": 2,
            },
        }
        result = assess_ticker_quality(
            "AAPL",
            _make_income_df(n_years=10),
            _make_balance_df(n_years=10),
            _make_cashflow_df(n_years=10),
        )
        # 10 years of complete data → passes both checks with fallback.
        assert result["drop"] is False

    @patch("data_acquisition.data_quality.get_config")
    def test_substitution_warning_threshold_from_config(
        self, mock_cfg: MagicMock,
    ) -> None:
        """Substitution warning threshold must come from
        data_quality.max_substitutions_before_flag config key."""
        mock_cfg.return_value = {
            "universe": {"min_history_years": 8},
            "data_quality": {
                "min_field_coverage_years": 8,
                "max_substitutions_before_flag": 1,  # Low threshold.
            },
        }
        with patch("data_acquisition.data_quality.logger") as mock_logger:
            assess_ticker_quality(
                "AAPL",
                _make_income_df(n_years=10),
                _make_balance_df(n_years=10),
                _make_cashflow_df(n_years=10),
                substitutions_count=2,
            )
            # 2 > 1 → should warn.
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "substitution" in str(c).lower()
            ]
            assert len(warning_calls) >= 1


# ===========================================================================
# Tests: _collect_all_tickers
# ===========================================================================

class TestCollectAllTickers:
    def test_returns_sorted_unique_tickers(self) -> None:
        fin = _make_financials(tickers=["MSFT", "AAPL", "GOOG"])
        tickers = _collect_all_tickers(fin)
        assert tickers == sorted(tickers)
        assert set(tickers) == {"AAPL", "GOOG", "MSFT"}

    def test_empty_financials_returns_empty(self) -> None:
        fin = {
            "income_statement": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
        }
        assert _collect_all_tickers(fin) == []

    def test_deduplicates_across_statements(self) -> None:
        fin = _make_financials(tickers=["AAPL"])
        assert _collect_all_tickers(fin) == ["AAPL"]

    def test_ticker_only_in_one_statement_included(self) -> None:
        inc = _make_income_df("AAPL", 10)
        bal = pd.concat([_make_balance_df("AAPL", 10), _make_balance_df("BAL_ONLY", 5)], ignore_index=True)
        cf = _make_cashflow_df("AAPL", 10)
        fin = {"income_statement": inc, "balance_sheet": bal, "cash_flow": cf}
        tickers = _collect_all_tickers(fin)
        assert "BAL_ONLY" in tickers


# ===========================================================================
# Tests: _filter_to_ticker
# ===========================================================================

class TestFilterToTicker:
    def test_returns_only_matching_rows(self) -> None:
        df = pd.concat([_make_income_df("AAPL", 5), _make_income_df("MSFT", 3)], ignore_index=True)
        result = _filter_to_ticker(df, "AAPL")
        assert len(result) == 5
        assert (result["ticker"] == "AAPL").all()

    def test_none_df_returns_empty(self) -> None:
        result = _filter_to_ticker(None, "AAPL")  # type: ignore[arg-type]
        assert len(result) == 0

    def test_empty_df_returns_empty(self) -> None:
        result = _filter_to_ticker(pd.DataFrame(), "AAPL")
        assert len(result) == 0

    def test_no_matching_ticker_returns_empty(self) -> None:
        df = _make_income_df("AAPL", 5)
        result = _filter_to_ticker(df, "MSFT")
        assert len(result) == 0

    def test_result_has_reset_index(self) -> None:
        df = pd.concat([_make_income_df("AAPL", 5), _make_income_df("MSFT", 3)], ignore_index=True)
        result = _filter_to_ticker(df, "MSFT")
        assert list(result.index) == list(range(len(result)))


# ===========================================================================
# Tests: _count_substitutions_per_ticker
# ===========================================================================

class TestCountSubstitutionsPerTicker:
    def test_counts_distinct_fields(self) -> None:
        log = [
            {"ticker": "AAPL", "buffett_field": "net_income", "confidence": "High"},
            {"ticker": "AAPL", "buffett_field": "total_revenue", "confidence": "Medium"},
            {"ticker": "AAPL", "buffett_field": "net_income", "confidence": "High"},  # duplicate
        ]
        counts = _count_substitutions_per_ticker(log)
        assert counts["AAPL"] == 2

    def test_drop_and_flag_excluded(self) -> None:
        log = [
            {"ticker": "AAPL", "buffett_field": "net_income", "confidence": "DROP"},
            {"ticker": "AAPL", "buffett_field": "total_revenue", "confidence": "FLAG"},
            {"ticker": "AAPL", "buffett_field": "gross_profit", "confidence": "High"},
        ]
        counts = _count_substitutions_per_ticker(log)
        assert counts.get("AAPL", 0) == 1

    def test_multiple_tickers(self) -> None:
        log = [
            {"ticker": "AAPL", "buffett_field": "net_income", "confidence": "High"},
            {"ticker": "MSFT", "buffett_field": "total_revenue", "confidence": "Medium"},
            {"ticker": "MSFT", "buffett_field": "gross_profit", "confidence": "Low"},
        ]
        counts = _count_substitutions_per_ticker(log)
        assert counts["AAPL"] == 1
        assert counts["MSFT"] == 2

    def test_empty_log_returns_empty_dict(self) -> None:
        assert _count_substitutions_per_ticker([]) == {}


# ===========================================================================
# Tests: _build_quality_report_df
# ===========================================================================

class TestBuildQualityReportDf:
    def test_empty_reports_returns_correct_columns(self) -> None:
        df = _build_quality_report_df([])
        expected = {
            "ticker", "years_available", "missing_critical_fields",
            "substitutions_count", "drop", "drop_reason",
        }
        assert set(df.columns) == expected
        assert len(df) == 0

    def test_missing_critical_fields_is_string_in_df(self) -> None:
        reports = [{
            "ticker": "AAPL",
            "years_available": 10,
            "missing_critical_fields": ["net_income", "eps_diluted"],
            "substitutions_count": 0,
            "drop": True,
            "drop_reason": "fields missing",
        }]
        df = _build_quality_report_df(reports)
        assert isinstance(df.iloc[0]["missing_critical_fields"], str)
        assert "net_income" in df.iloc[0]["missing_critical_fields"]

    def test_drop_column_is_bool_dtype(self) -> None:
        reports = [{
            "ticker": "AAPL",
            "years_available": 10,
            "missing_critical_fields": [],
            "substitutions_count": 0,
            "drop": False,
            "drop_reason": None,
        }]
        df = _build_quality_report_df(reports)
        assert df["drop"].dtype == bool

    def test_empty_missing_fields_list_serializes_to_empty_string(self) -> None:
        reports = [{
            "ticker": "AAPL",
            "years_available": 10,
            "missing_critical_fields": [],
            "substitutions_count": 0,
            "drop": False,
            "drop_reason": None,
        }]
        df = _build_quality_report_df(reports)
        assert df.iloc[0]["missing_critical_fields"] == ""


# ===========================================================================
# Tests: run_data_quality_check
# ===========================================================================

class TestRunDataQualityCheck:
    @patch("data_acquisition.data_quality._save_quality_report")
    def test_returns_tuple_of_two_dataframes(self, mock_save: MagicMock) -> None:
        fin = _make_financials(["AAPL"])
        result = run_data_quality_check(fin, [])
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)

    @patch("data_acquisition.data_quality._save_quality_report")
    def test_quality_report_has_one_row_per_ticker(self, mock_save: MagicMock) -> None:
        fin = _make_financials(["AAPL", "MSFT", "GOOG"])
        quality_df, _ = run_data_quality_check(fin, [])
        assert len(quality_df) == 3

    @patch("data_acquisition.data_quality._save_quality_report")
    def test_survivors_contains_only_passing_tickers(self, mock_save: MagicMock) -> None:
        inc = pd.concat([_make_income_df("AAPL", 10), _make_income_df("TINY", 3)], ignore_index=True)
        bal = pd.concat([_make_balance_df("AAPL", 10), _make_balance_df("TINY", 3)], ignore_index=True)
        cf = pd.concat([_make_cashflow_df("AAPL", 10), _make_cashflow_df("TINY", 3)], ignore_index=True)
        fin = {"income_statement": inc, "balance_sheet": bal, "cash_flow": cf}

        _, survivors_df = run_data_quality_check(fin, [])

        assert "AAPL" in survivors_df["ticker"].values
        assert "TINY" not in survivors_df["ticker"].values

    @patch("data_acquisition.data_quality._save_quality_report")
    def test_survivors_df_has_only_ticker_column(self, mock_save: MagicMock) -> None:
        fin = _make_financials(["AAPL"])
        _, survivors_df = run_data_quality_check(fin, [])
        assert list(survivors_df.columns) == ["ticker"]

    @patch("data_acquisition.data_quality._save_quality_report")
    def test_empty_financials_returns_empty_report_and_survivors(
        self, mock_save: MagicMock
    ) -> None:
        fin = {
            "income_statement": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
        }
        quality_df, survivors_df = run_data_quality_check(fin, [])
        assert len(quality_df) == 0
        assert len(survivors_df) == 0

    @patch("data_acquisition.data_quality._save_quality_report")
    def test_substitution_counts_aggregated_from_log(self, mock_save: MagicMock) -> None:
        fin = _make_financials(["AAPL"])
        sub_log = [
            {"ticker": "AAPL", "buffett_field": "net_income", "confidence": "High",
             "fiscal_year": 2023, "api_field_used": "netIncomeFromContinuingOperations"},
            {"ticker": "AAPL", "buffett_field": "total_revenue", "confidence": "Medium",
             "fiscal_year": 2023, "api_field_used": "revenue"},
        ]
        quality_df, _ = run_data_quality_check(fin, sub_log)
        aapl_row = quality_df[quality_df["ticker"] == "AAPL"].iloc[0]
        assert aapl_row["substitutions_count"] == 2

    @patch("data_acquisition.data_quality._save_quality_report")
    def test_passing_ticker_has_drop_false(self, mock_save: MagicMock) -> None:
        fin = _make_financials(["AAPL"])
        quality_df, _ = run_data_quality_check(fin, [])
        assert quality_df[quality_df["ticker"] == "AAPL"]["drop"].item() is False

    @patch("data_acquisition.data_quality._save_quality_report")
    def test_save_quality_report_called_once(self, mock_save: MagicMock) -> None:
        fin = _make_financials(["AAPL"])
        run_data_quality_check(fin, [])
        mock_save.assert_called_once()

    @patch("data_acquisition.data_quality._save_quality_report")
    def test_ticker_only_in_one_statement_still_assessed(self, mock_save: MagicMock) -> None:
        """A ticker present only in balance_sheet still appears in the quality report."""
        inc = _make_income_df("AAPL", 10)
        bal = pd.concat(
            [_make_balance_df("AAPL", 10), _make_balance_df("BS_ONLY", 10)],
            ignore_index=True,
        )
        cf = _make_cashflow_df("AAPL", 10)
        fin = {"income_statement": inc, "balance_sheet": bal, "cash_flow": cf}
        quality_df, _ = run_data_quality_check(fin, [])
        assert "BS_ONLY" in quality_df["ticker"].values

    @patch("data_acquisition.data_quality._save_quality_report")
    def test_failing_ticker_drop_true_in_report(self, mock_save: MagicMock) -> None:
        inc = pd.concat([
            _make_income_df("AAPL", 10),
            _make_income_df("TINY", 2),
        ], ignore_index=True)
        bal = pd.concat([_make_balance_df("AAPL", 10), _make_balance_df("TINY", 2)], ignore_index=True)
        cf = pd.concat([_make_cashflow_df("AAPL", 10), _make_cashflow_df("TINY", 2)], ignore_index=True)
        fin = {"income_statement": inc, "balance_sheet": bal, "cash_flow": cf}
        quality_df, _ = run_data_quality_check(fin, [])
        tiny_row = quality_df[quality_df["ticker"] == "TINY"]
        assert tiny_row["drop"].item() is True


# ===========================================================================
# Tests: _compare_field
# ===========================================================================

class TestCompareField:
    def _make_series(self, **kwargs: float) -> pd.Series:
        return pd.Series(kwargs)

    def test_returns_dict_for_valid_inputs(self) -> None:
        row = self._make_series(net_income=100.0)
        result = _compare_field("AAPL", "net_income", row, {"net_income": 101.0}, 0.05)
        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["field"] == "net_income"

    def test_pct_difference_computed_correctly(self) -> None:
        row = self._make_series(net_income=110.0)
        result = _compare_field("AAPL", "net_income", row, {"net_income": 100.0}, 0.05)
        assert result is not None
        assert result["pct_difference"] == pytest.approx(0.10)

    def test_returns_none_when_fmp_nan(self) -> None:
        row = self._make_series(net_income=float("nan"))
        result = _compare_field("AAPL", "net_income", row, {"net_income": 100.0}, 0.05)
        assert result is None

    def test_returns_none_when_yf_missing(self) -> None:
        row = self._make_series(net_income=100.0)
        result = _compare_field("AAPL", "net_income", row, {}, 0.05)
        assert result is None

    def test_returns_none_when_field_not_in_row(self) -> None:
        row = self._make_series(total_revenue=100.0)
        result = _compare_field("AAPL", "net_income", row, {"net_income": 100.0}, 0.05)
        assert result is None

    def test_warning_logged_on_excessive_divergence(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        row = self._make_series(net_income=200.0)
        with caplog.at_level(logging.WARNING, logger="data_acquisition.data_quality"):
            _compare_field("AAPL", "net_income", row, {"net_income": 100.0}, 0.05)
        assert any("diverges" in r.message for r in caplog.records)

    def test_no_warning_within_tolerance(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        row = self._make_series(net_income=102.0)
        with caplog.at_level(logging.WARNING, logger="data_acquisition.data_quality"):
            _compare_field("AAPL", "net_income", row, {"net_income": 100.0}, 0.05)
        assert not any("diverges" in r.message for r in caplog.records)


# ===========================================================================
# Tests: cross_validate_sample
# ===========================================================================

class TestCrossValidateSample:
    def test_no_income_data_returns_empty_with_columns(self) -> None:
        fin = {
            "income_statement": pd.DataFrame(),
            "balance_sheet": _make_balance_df(),
            "cash_flow": _make_cashflow_df(),
        }
        result = cross_validate_sample(fin, sample_size=5)
        assert len(result) == 0
        expected = {"ticker", "field", "fmp_value", "yfinance_value", "pct_difference"}
        assert expected.issubset(set(result.columns))

    def test_empty_cross_validate_df_has_correct_columns(self) -> None:
        from data_acquisition.data_quality import _empty_cross_validate_df
        df = _empty_cross_validate_df()
        assert set(df.columns) == {"ticker", "field", "fmp_value", "yfinance_value", "pct_difference"}

    @patch("data_acquisition.data_quality.random.sample")
    @patch("data_acquisition.data_quality._safe_fetch_yf_comparison")
    def test_returns_expected_columns(
        self, mock_fetch: MagicMock, mock_sample: MagicMock
    ) -> None:
        fin = _make_financials(["AAPL"])
        mock_sample.return_value = ["AAPL"]
        mock_fetch.return_value = {
            "net_income": 96_995.0,
            "total_revenue": 383_285.0,
            "eps_diluted": 6.13,
        }
        result = cross_validate_sample(fin, sample_size=1)
        expected_cols = {"ticker", "field", "fmp_value", "yfinance_value", "pct_difference"}
        assert expected_cols.issubset(set(result.columns))

    @patch("data_acquisition.data_quality.random.sample")
    @patch("data_acquisition.data_quality._safe_fetch_yf_comparison")
    def test_yfinance_empty_result_skips_ticker(
        self, mock_fetch: MagicMock, mock_sample: MagicMock
    ) -> None:
        fin = _make_financials(["AAPL"])
        mock_sample.return_value = ["AAPL"]
        mock_fetch.return_value = {}
        result = cross_validate_sample(fin, sample_size=1)
        assert len(result) == 0

    @patch("data_acquisition.data_quality.random.sample")
    @patch("data_acquisition.data_quality._safe_fetch_yf_comparison")
    def test_pct_difference_computed_per_row(
        self, mock_fetch: MagicMock, mock_sample: MagicMock
    ) -> None:
        fin = _make_financials(["AAPL"])
        mock_sample.return_value = ["AAPL"]
        # FMP net_income = 96_995; yfinance = 100_000
        mock_fetch.return_value = {
            "net_income": 100_000.0,
            "total_revenue": None,
            "eps_diluted": None,
        }
        result = cross_validate_sample(fin, sample_size=1)
        ni_row = result[result["field"] == "net_income"]
        assert len(ni_row) == 1
        expected_pct = abs(96_995.0 - 100_000.0) / 100_000.0
        assert ni_row.iloc[0]["pct_difference"] == pytest.approx(expected_pct, rel=1e-3)

    @patch("data_acquisition.data_quality.random.sample")
    @patch("data_acquisition.data_quality._safe_fetch_yf_comparison")
    def test_multiple_fields_compared_per_ticker(
        self, mock_fetch: MagicMock, mock_sample: MagicMock
    ) -> None:
        fin = _make_financials(["AAPL"])
        mock_sample.return_value = ["AAPL"]
        mock_fetch.return_value = {
            "net_income": 96_995.0,
            "total_revenue": 383_285.0,
            "eps_diluted": 6.13,
        }
        result = cross_validate_sample(fin, sample_size=1)
        # All three fields should have rows (all values match → no divergence)
        assert set(result["field"]) == {"net_income", "total_revenue", "eps_diluted"}

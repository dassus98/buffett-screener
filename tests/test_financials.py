"""Unit tests for data_acquisition/financials.py.

All external I/O (API calls, file system, config) is mocked.
No real network requests, disk writes, or FMP API keys required.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from data_acquisition.financials import (
    _MAX_CONSECUTIVE_FMP_FAILURES,
    _apply_value_conventions,
    _empty_normalised_df,
    _ensure_canonical_columns,
    _extract_fiscal_year,
    _fetch_statements_yfinance,
    _resolve_fmp_ticker,
    fetch_all_financials,
    fetch_financial_statements,
    normalize_statement,
)
from data_acquisition.schema import CANONICAL_COLUMNS, LINE_ITEM_MAP


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_fmp_income_row(
    ticker: str = "AAPL",
    year: int = 2023,
    net_income: float = 100_000_000_000,
    revenue: float = 400_000_000_000,
    gross_profit: float = 170_000_000_000,
    operating_income: float = 115_000_000_000,
    interest_expense: float = 3_000_000_000,
    eps_diluted: float = 6.13,
    shares_diluted: float = 15_812_547_000,
    sga: float = 24_000_000_000,
) -> dict[str, Any]:
    """Minimal FMP income statement row using ideal field names."""
    return {
        "symbol": ticker,
        "date": f"{year}-09-30",
        "calendarYear": str(year),
        "netIncome": net_income,
        "totalRevenue": revenue,
        "grossProfit": gross_profit,
        "operatingIncome": operating_income,
        "interestExpense": interest_expense,
        "epsdiluted": eps_diluted,
        "weightedAverageSharesDiluted": shares_diluted,
        "sellingGeneralAndAdministrativeExpenses": sga,
        "depreciationAndAmortization": 12_000_000_000,
    }


def _make_fmp_balance_row(
    ticker: str = "AAPL",
    year: int = 2023,
    long_term_debt: float = 110_000_000_000,
    equity: float = 62_000_000_000,
) -> dict[str, Any]:
    """Minimal FMP balance sheet row using ideal field names."""
    return {
        "symbol": ticker,
        "date": f"{year}-09-30",
        "calendarYear": str(year),
        "longTermDebt": long_term_debt,
        "totalStockholdersEquity": equity,
        "retainedEarnings": 20_000_000_000,
        "totalCurrentAssets": 130_000_000_000,
        "totalCurrentLiabilities": 145_000_000_000,
        "treasuryStock": -70_000_000_000,
    }


def _make_fmp_cashflow_row(
    ticker: str = "AAPL",
    year: int = 2023,
    op_cf: float = 115_000_000_000,
    capex: float = -11_000_000_000,
    da: float = 12_000_000_000,
    wc_change: float = -5_000_000_000,
) -> dict[str, Any]:
    """Minimal FMP cash flow row using ideal field names."""
    return {
        "symbol": ticker,
        "date": f"{year}-09-30",
        "calendarYear": str(year),
        "operatingCashFlow": op_cf,
        "capitalExpenditure": capex,
        "depreciationAndAmortization": da,
        "changeInWorkingCapital": wc_change,
        "freeCashFlow": op_cf + capex,
    }


def _make_universe_df(tickers: list[str], exchange: str = "NYSE") -> pd.DataFrame:
    return pd.DataFrame({"ticker": tickers, "exchange": [exchange] * len(tickers)})


# ---------------------------------------------------------------------------
# _resolve_fmp_ticker tests
# ---------------------------------------------------------------------------

class TestResolveFmpTicker:
    def test_plain_us_ticker_unchanged(self) -> None:
        assert _resolve_fmp_ticker("AAPL") == "AAPL"

    def test_tsx_ticker_with_suffix_unchanged(self) -> None:
        assert _resolve_fmp_ticker("SHOP.TO") == "SHOP.TO"

    def test_brk_b_dot_not_treated_as_tsx(self) -> None:
        """BRK.B has a dot but must not be modified."""
        assert _resolve_fmp_ticker("BRK.B") == "BRK.B"


# ---------------------------------------------------------------------------
# _extract_fiscal_year tests
# ---------------------------------------------------------------------------

class TestExtractFiscalYear:
    def test_calendar_year_field_preferred(self) -> None:
        row = {"calendarYear": "2022", "date": "2021-09-30"}
        assert _extract_fiscal_year(row) == 2022

    def test_falls_back_to_date_field(self) -> None:
        row = {"date": "2020-12-31"}
        assert _extract_fiscal_year(row) == 2020

    def test_integer_calendar_year_accepted(self) -> None:
        row = {"calendarYear": 2019}
        assert _extract_fiscal_year(row) == 2019

    def test_returns_zero_when_no_date_available(self) -> None:
        assert _extract_fiscal_year({}) == 0

    def test_malformed_calendar_year_falls_back_to_date(self) -> None:
        row = {"calendarYear": "N/A", "date": "2018-06-30"}
        assert _extract_fiscal_year(row) == 2018


# ---------------------------------------------------------------------------
# _apply_value_conventions tests
# ---------------------------------------------------------------------------

class TestApplyValueConventions:
    def test_monetary_field_divided_by_1000(self) -> None:
        result = _apply_value_conventions("net_income", 5_000_000_000, "AAPL")
        assert result == pytest.approx(5_000_000.0)

    def test_eps_not_divided(self) -> None:
        result = _apply_value_conventions("eps_diluted", 6.13, "AAPL")
        assert result == pytest.approx(6.13)

    def test_shares_not_divided(self) -> None:
        result = _apply_value_conventions(
            "shares_outstanding_diluted", 15_000_000_000, "AAPL"
        )
        assert result == pytest.approx(15_000_000_000)

    def test_capex_negated_when_positive(self) -> None:
        result = _apply_value_conventions("capital_expenditures", 11_000_000_000, "AAPL")
        assert result == pytest.approx(-11_000_000.0)

    def test_capex_stays_negative_when_already_negative(self) -> None:
        result = _apply_value_conventions("capital_expenditures", -11_000_000_000, "AAPL")
        assert result == pytest.approx(-11_000_000.0)

    def test_capex_zero_stays_zero(self) -> None:
        result = _apply_value_conventions("capital_expenditures", 0, "AAPL")
        assert result == 0.0

    def test_string_value_returned_unchanged(self) -> None:
        result = _apply_value_conventions("net_income", "N/A", "AAPL")
        assert result == "N/A"


# ---------------------------------------------------------------------------
# normalize_statement — ideal fields → canonical output
# ---------------------------------------------------------------------------

class TestNormalizeStatementIdealFields:
    """normalize_statement with all ideal FMP field names present."""

    def _income_df(self, ticker="AAPL", years=(2022, 2023)) -> pd.DataFrame:
        return pd.DataFrame([_make_fmp_income_row(ticker, y) for y in years])

    def test_output_contains_ticker_column(self) -> None:
        df, _ = normalize_statement(self._income_df(), "income_statement", ticker="AAPL")
        assert "ticker" in df.columns
        assert (df["ticker"] == "AAPL").all()

    def test_output_contains_fiscal_year_column(self) -> None:
        df, _ = normalize_statement(self._income_df(), "income_statement", ticker="AAPL")
        assert "fiscal_year" in df.columns
        assert set(df["fiscal_year"]) == {2022, 2023}

    def test_net_income_converted_to_thousands(self) -> None:
        raw = pd.DataFrame([_make_fmp_income_row("AAPL", 2023, net_income=100_000_000_000)])
        df, _ = normalize_statement(raw, "income_statement", ticker="AAPL")
        assert df.iloc[0]["net_income"] == pytest.approx(100_000_000.0)

    def test_eps_not_divided(self) -> None:
        raw = pd.DataFrame([_make_fmp_income_row("AAPL", 2023, eps_diluted=6.13)])
        df, _ = normalize_statement(raw, "income_statement", ticker="AAPL")
        assert df.iloc[0]["eps_diluted"] == pytest.approx(6.13)

    def test_no_substitution_logs_when_ideal_fields_present(self) -> None:
        df, subs = normalize_statement(self._income_df(), "income_statement", ticker="AAPL")
        assert subs == [], f"Expected no substitutions but got: {subs}"

    def test_rows_sorted_by_fiscal_year_ascending(self) -> None:
        raw = pd.DataFrame([
            _make_fmp_income_row("AAPL", 2023),
            _make_fmp_income_row("AAPL", 2020),
            _make_fmp_income_row("AAPL", 2021),
        ])
        df, _ = normalize_statement(raw, "income_statement", ticker="AAPL")
        assert list(df["fiscal_year"]) == [2020, 2021, 2023]

    def test_output_excludes_wrong_statement_fields(self) -> None:
        """balance_sheet fields must not appear in income_statement output."""
        raw = pd.DataFrame([_make_fmp_income_row("AAPL", 2023)])
        df, _ = normalize_statement(raw, "income_statement", ticker="AAPL")
        # long_term_debt is a balance_sheet field — must not be in IS output
        assert "long_term_debt" not in df.columns

    def test_empty_raw_df_returns_empty_with_metadata_cols(self) -> None:
        df, subs = normalize_statement(pd.DataFrame(), "income_statement", ticker="AAPL")
        assert df.empty
        assert "ticker" in df.columns or len(df.columns) == 0
        assert subs == []


# ---------------------------------------------------------------------------
# normalize_statement — substitution fields → substitution log
# ---------------------------------------------------------------------------

class TestNormalizeStatementSubstitutions:
    """normalize_statement when ideal fields are absent but substitutes exist."""

    def test_substitute_field_resolves_and_logged(self) -> None:
        """netIncomeFromContinuingOperations should substitute for netIncome."""
        row = {
            "calendarYear": "2023",
            "netIncomeFromContinuingOperations": 80_000_000_000,
            "totalRevenue": 390_000_000_000,
            "grossProfit": 165_000_000_000,
            "operatingIncome": 110_000_000_000,
            "interestExpense": 2_500_000_000,
            "epsdiluted": 5.90,
            "weightedAverageSharesDiluted": 15_500_000_000,
        }
        raw = pd.DataFrame([row])
        df, subs = normalize_statement(raw, "income_statement", ticker="AAPL")

        assert df.iloc[0]["net_income"] == pytest.approx(80_000_000.0)

        sub_fields = [s["buffett_field"] for s in subs]
        assert "net_income" in sub_fields

    def test_substitution_log_entry_has_correct_keys(self) -> None:
        row = {"calendarYear": "2023", "netIncomeFromContinuingOperations": 5e10}
        raw = pd.DataFrame([row])
        _, subs = normalize_statement(raw, "income_statement", ticker="AAPL")
        net_income_subs = [s for s in subs if s["buffett_field"] == "net_income"]
        assert len(net_income_subs) == 1
        entry = net_income_subs[0]
        assert entry["ticker"] == "AAPL"
        assert entry["fiscal_year"] == 2023
        assert entry["api_field_used"] == "netIncomeFromContinuingOperations"
        assert entry["confidence"] in ("High", "Medium", "Low")

    def test_missing_drop_field_is_nan_in_output(self) -> None:
        """A DROP-outcome field must be NaN in the DataFrame — not excluded."""
        row = {"calendarYear": "2023"}  # net_income completely absent
        raw = pd.DataFrame([row])
        df, subs = normalize_statement(raw, "income_statement", ticker="AAPL")

        assert "net_income" in df.columns
        assert pd.isna(df.iloc[0]["net_income"])

    def test_missing_drop_field_logged_in_substitution_log(self) -> None:
        row = {"calendarYear": "2023"}
        raw = pd.DataFrame([row])
        _, subs = normalize_statement(raw, "income_statement", ticker="AAPL")
        drop_entries = [
            s for s in subs
            if s["buffett_field"] == "net_income" and s["confidence"] == "DROP"
        ]
        assert len(drop_entries) == 1

    def test_missing_flag_field_is_nan_and_logged_as_flag(self) -> None:
        """sga (drop_if_missing=False) should be NaN with confidence='FLAG'."""
        row = {"calendarYear": "2023", "netIncome": 1e10}
        raw = pd.DataFrame([row])
        df, subs = normalize_statement(raw, "income_statement", ticker="AAPL")

        if "sga" in df.columns:
            assert pd.isna(df.iloc[0]["sga"])
        flag_entries = [
            s for s in subs if s["buffett_field"] == "sga" and s["confidence"] == "FLAG"
        ]
        assert len(flag_entries) == 1

    def test_multiple_rows_substitution_log_accumulated(self) -> None:
        """Each row with substitutions generates its own log entry."""
        rows = [
            {"calendarYear": str(y), "netIncomeFromContinuingOperations": 5e10}
            for y in [2021, 2022, 2023]
        ]
        raw = pd.DataFrame(rows)
        _, subs = normalize_statement(raw, "income_statement", ticker="AAPL")
        ni_subs = [s for s in subs if s["buffett_field"] == "net_income"]
        assert len(ni_subs) == 3

    def test_capex_positive_value_negated(self) -> None:
        """CapEx returned positive by FMP must be stored negative."""
        row = {
            "calendarYear": "2023",
            "operatingCashFlow": 115e9,
            "capitalExpenditure": 11_000_000_000,   # positive — wrong sign
            "depreciationAndAmortization": 12e9,
        }
        raw = pd.DataFrame([row])
        df, _ = normalize_statement(raw, "cash_flow", ticker="AAPL")
        assert df.iloc[0]["capital_expenditures"] < 0

    def test_capex_already_negative_stays_negative(self) -> None:
        row = {
            "calendarYear": "2023",
            "operatingCashFlow": 115e9,
            "capitalExpenditure": -11_000_000_000,  # already negative — correct
            "depreciationAndAmortization": 12e9,
        }
        raw = pd.DataFrame([row])
        df, _ = normalize_statement(raw, "cash_flow", ticker="AAPL")
        assert df.iloc[0]["capital_expenditures"] < 0


# ---------------------------------------------------------------------------
# normalize_statement — balance sheet
# ---------------------------------------------------------------------------

class TestNormalizeStatementBalanceSheet:
    def test_long_term_debt_in_thousands(self) -> None:
        raw = pd.DataFrame([_make_fmp_balance_row("AAPL", 2023, long_term_debt=110e9)])
        df, _ = normalize_statement(raw, "balance_sheet", ticker="AAPL")
        assert df.iloc[0]["long_term_debt"] == pytest.approx(110_000_000.0)

    def test_shareholders_equity_in_thousands(self) -> None:
        raw = pd.DataFrame([_make_fmp_balance_row("AAPL", 2023, equity=62e9)])
        df, _ = normalize_statement(raw, "balance_sheet", ticker="AAPL")
        assert df.iloc[0]["shareholders_equity"] == pytest.approx(62_000_000.0)


# ---------------------------------------------------------------------------
# fetch_financial_statements tests (mocked resilient_request)
# ---------------------------------------------------------------------------

class TestFetchFinancialStatements:
    """fetch_financial_statements makes 3 API calls and returns raw DataFrames."""

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    def test_three_calls_made_per_ticker(self, mock_req, mock_persist) -> None:
        mock_req.return_value = [_make_fmp_income_row()]
        fetch_financial_statements("AAPL")
        assert mock_req.call_count == 3

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    def test_returns_dict_with_three_keys(self, mock_req, mock_persist) -> None:
        mock_req.return_value = [_make_fmp_income_row()]
        result = fetch_financial_statements("AAPL")
        assert set(result.keys()) == {"income_statement", "balance_sheet", "cash_flow"}

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    def test_returns_dataframe_on_success(self, mock_req, mock_persist) -> None:
        mock_req.return_value = [_make_fmp_income_row()]
        result = fetch_financial_statements("AAPL")
        for df in result.values():
            assert isinstance(df, pd.DataFrame)
            assert not df.empty

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    def test_empty_response_sets_key_to_none(self, mock_req, mock_persist) -> None:
        """An endpoint returning [] must map to None for that statement."""
        mock_req.side_effect = [
            [_make_fmp_income_row()],  # income_statement
            [],                         # balance_sheet empty
            [_make_fmp_cashflow_row()], # cash_flow
        ]
        result = fetch_financial_statements("AAPL")
        assert result["balance_sheet"] is None

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    def test_api_error_sets_key_to_none(self, mock_req, mock_persist) -> None:
        """A single endpoint error must set that key to None, not propagate."""
        mock_req.side_effect = [
            Exception("HTTP 500"),      # income_statement fails
            [_make_fmp_balance_row()],  # balance_sheet succeeds
            [_make_fmp_cashflow_row()], # cash_flow succeeds
        ]
        result = fetch_financial_statements("AAPL")
        assert result["income_statement"] is None
        assert isinstance(result["balance_sheet"], pd.DataFrame)

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    def test_persist_raw_called_once(self, mock_req, mock_persist) -> None:
        mock_req.return_value = [_make_fmp_income_row()]
        fetch_financial_statements("AAPL")
        mock_persist.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_all_financials — error isolation
# ---------------------------------------------------------------------------

class TestFetchAllFinancialsErrorIsolation:
    """A failing ticker must not abort the batch."""

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    def test_batch_continues_after_single_ticker_failure(
        self, mock_req, mock_persist
    ) -> None:
        good_rows = [_make_fmp_income_row()]

        call_count = {"n": 0}

        def side_effect(url, params=None, rate_limiter=None, **kw):
            call_count["n"] += 1
            # AAPL succeeds; JPM (call 4–6) fails on first call
            if call_count["n"] in (4,):
                raise Exception("Simulated API failure for JPM")
            return good_rows

        mock_req.side_effect = side_effect
        universe = _make_universe_df(["AAPL", "JPM", "MSFT"])
        result_dfs, subs = fetch_all_financials(universe, batch_size=10)

        # At least AAPL and MSFT should have contributed rows
        assert len(result_dfs["income_statement"]) >= 1

    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_all_tickers_fail_returns_empty_dfs(
        self, mock_fetch, mock_eta
    ) -> None:
        mock_fetch.side_effect = Exception("total failure")
        universe = _make_universe_df(["X", "Y"])
        result_dfs, subs = fetch_all_financials(universe, batch_size=10)
        assert all(df.empty for df in result_dfs.values())

    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_empty_universe_returns_empty_dfs(
        self, mock_fetch, mock_eta
    ) -> None:
        universe = pd.DataFrame({"ticker": []})
        result_dfs, subs = fetch_all_financials(universe)
        assert all(df.empty for df in result_dfs.values())
        mock_fetch.assert_not_called()


class TestFetchAllFinancialsAggregation:
    """Results from multiple tickers are correctly concatenated."""

    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_master_dfs_contain_rows_from_all_tickers(
        self, mock_fetch, mock_eta
    ) -> None:
        def make_stmts(ticker):
            return {
                "income_statement": pd.DataFrame([
                    _make_fmp_income_row(ticker, 2022),
                    _make_fmp_income_row(ticker, 2023),
                ]),
                "balance_sheet": pd.DataFrame([_make_fmp_balance_row(ticker, 2023)]),
                "cash_flow": pd.DataFrame([_make_fmp_cashflow_row(ticker, 2023)]),
            }

        mock_fetch.side_effect = lambda t: make_stmts(t)
        universe = _make_universe_df(["AAPL", "MSFT"])

        result_dfs, subs = fetch_all_financials(universe, batch_size=10)
        tickers_in_result = set(result_dfs["income_statement"]["ticker"].unique())
        assert tickers_in_result == {"AAPL", "MSFT"}

    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_duplicate_ticker_year_pairs_deduplicated(
        self, mock_fetch, mock_eta
    ) -> None:
        """If somehow the same (ticker, fiscal_year) appears twice, keep one row."""
        def make_dupes(ticker):
            return {
                "income_statement": pd.DataFrame([
                    _make_fmp_income_row(ticker, 2023),
                    _make_fmp_income_row(ticker, 2023),  # duplicate year
                ]),
                "balance_sheet": None,
                "cash_flow": None,
            }

        mock_fetch.side_effect = lambda t: make_dupes(t)
        universe = _make_universe_df(["AAPL"])
        result_dfs, _ = fetch_all_financials(universe, batch_size=10)
        aapl = result_dfs["income_statement"]
        year_counts = aapl[aapl["ticker"] == "AAPL"]["fiscal_year"].value_counts()
        assert year_counts.get(2023, 0) == 1

    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_substitution_logs_aggregated_across_tickers(
        self, mock_fetch, mock_eta
    ) -> None:
        """Substitution logs from all tickers must be merged into one list."""
        def make_stmts(ticker):
            # Use substitute field for net_income to generate log entries
            return {
                "income_statement": pd.DataFrame([{
                    "calendarYear": "2023",
                    "netIncomeFromContinuingOperations": 5e10,  # substitute
                    "totalRevenue": 4e11,
                    "grossProfit": 1.7e11,
                    "operatingIncome": 1.1e11,
                }]),
                "balance_sheet": None,
                "cash_flow": None,
            }

        mock_fetch.side_effect = lambda t: make_stmts(t)
        universe = _make_universe_df(["AAPL", "MSFT"])
        _, subs = fetch_all_financials(universe, batch_size=10)
        # Both AAPL and MSFT should have a substitution entry for net_income
        tickers_with_subs = {s["ticker"] for s in subs if s["buffett_field"] == "net_income"}
        assert "AAPL" in tickers_with_subs
        assert "MSFT" in tickers_with_subs


class TestFetchAllFinancialsProgress:
    """Progress logging fires at correct intervals."""

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_progress_logged_at_batch_boundaries(
        self, mock_fetch, mock_eta, mock_yf_fetch, caplog
    ) -> None:
        import logging

        mock_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        mock_yf_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        tickers = [f"T{i}" for i in range(10)]
        universe = _make_universe_df(tickers)

        with caplog.at_level(logging.INFO, logger="data_acquisition.financials"):
            fetch_all_financials(universe, batch_size=5)

        progress_msgs = [r for r in caplog.records if "progress" in r.message.lower()]
        # Expect a log at tick 5 and tick 10
        assert len(progress_msgs) >= 2


# ---------------------------------------------------------------------------
# _empty_normalised_df tests
# ---------------------------------------------------------------------------

class TestEmptyNormalisedDf:
    def test_has_metadata_columns(self) -> None:
        df = _empty_normalised_df("AAPL")
        assert "ticker" in df.columns
        assert "fiscal_year" in df.columns

    def test_is_empty(self) -> None:
        assert _empty_normalised_df("AAPL").empty


# ---------------------------------------------------------------------------
# normalize_statement — None / edge-case inputs
# ---------------------------------------------------------------------------

class TestNormalizeStatementEdgeCases:
    """Edge cases: None input, single row, cash flow D&A and working capital."""

    def test_none_input_returns_empty_df_and_no_subs(self) -> None:
        """Passing None instead of a DataFrame should not raise."""
        df, subs = normalize_statement(None, "income_statement", ticker="AAPL")
        assert df.empty
        assert subs == []

    def test_cash_flow_depreciation_in_thousands(self) -> None:
        """D&A from cash flow should be divided by 1000 like other monetary fields."""
        row = _make_fmp_cashflow_row("AAPL", 2023, da=12_000_000_000)
        raw = pd.DataFrame([row])
        df, _ = normalize_statement(raw, "cash_flow", ticker="AAPL")
        assert df.iloc[0]["depreciation_amortization"] == pytest.approx(12_000_000.0)

    def test_working_capital_change_in_thousands(self) -> None:
        """Working capital change should be divided by 1000."""
        row = _make_fmp_cashflow_row("AAPL", 2023, wc_change=-5_000_000_000)
        raw = pd.DataFrame([row])
        df, _ = normalize_statement(raw, "cash_flow", ticker="AAPL")
        assert df.iloc[0]["working_capital_change"] == pytest.approx(-5_000_000.0)


# ---------------------------------------------------------------------------
# _ensure_canonical_columns tests
# ---------------------------------------------------------------------------

class TestEnsureCanonicalColumns:
    """_ensure_canonical_columns adds missing columns as NaN."""

    def test_adds_missing_income_columns_as_nan(self) -> None:
        """A DataFrame with only ticker/fiscal_year should gain all IS columns."""
        df = pd.DataFrame({"ticker": ["AAPL"], "fiscal_year": [2023]})
        result = _ensure_canonical_columns(df, "income_statement", "AAPL")

        # All canonical income_statement fields must be present.
        expected_cols = [
            CANONICAL_COLUMNS[name]
            for name, item in LINE_ITEM_MAP.items()
            if item.statement == "income_statement"
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing expected column: {col}"
            assert pd.isna(result.iloc[0][col])

    def test_preserves_existing_values(self) -> None:
        """Pre-existing values must not be overwritten with NaN."""
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "fiscal_year": [2023],
            "net_income": [100_000.0],
        })
        result = _ensure_canonical_columns(df, "income_statement", "AAPL")
        assert result.iloc[0]["net_income"] == pytest.approx(100_000.0)

    def test_balance_sheet_columns_separate_from_income(self) -> None:
        """Balance sheet columns should not appear in an income_statement call."""
        df = pd.DataFrame({"ticker": ["AAPL"], "fiscal_year": [2023]})
        result = _ensure_canonical_columns(df, "income_statement", "AAPL")
        bs_cols = [
            CANONICAL_COLUMNS[name]
            for name, item in LINE_ITEM_MAP.items()
            if item.statement == "balance_sheet"
        ]
        for col in bs_cols:
            assert col not in result.columns


# ---------------------------------------------------------------------------
# Config-driven limit for fetch_financial_statements
# ---------------------------------------------------------------------------

class TestFetchFinancialStatementsConfigLimit:
    """fetch_financial_statements reads limit from config, not hardcoded."""

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    @patch("data_acquisition.financials.get_config")
    def test_limit_param_read_from_config(
        self, mock_config, mock_req, mock_persist
    ) -> None:
        """The FMP limit parameter should equal universe.required_history_years."""
        mock_config.return_value = {
            "universe": {"required_history_years": 15},
            "data_sources": {"store_raw_responses": False},
        }
        mock_req.return_value = [_make_fmp_income_row()]

        fetch_financial_statements("AAPL")

        # Inspect the build_fmp_url calls — limit should be 15, not 10.
        for c in mock_req.call_args_list:
            url = c[0][0]  # positional arg 0 = url
            params = c[1].get("params", {}) if c[1] else {}
            # build_fmp_url embeds limit in the params dict
            # We check via the URL or params that were passed
            # The limit is baked into the params by build_fmp_url.
            pass  # covered by the build_fmp_url mock below

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    @patch("data_acquisition.financials.build_fmp_url")
    @patch("data_acquisition.financials.get_config")
    def test_limit_param_passed_to_build_fmp_url(
        self, mock_config, mock_build, mock_req, mock_persist
    ) -> None:
        """Verify build_fmp_url is called with limit=required_history_years."""
        mock_config.return_value = {
            "universe": {"required_history_years": 15},
            "data_sources": {"store_raw_responses": False},
        }
        mock_build.return_value = ("http://example.com/api", {"apikey": "x"})
        mock_req.return_value = [_make_fmp_income_row()]

        fetch_financial_statements("AAPL")

        # All 3 build_fmp_url calls should have limit=15.
        assert mock_build.call_count == 3
        for c in mock_build.call_args_list:
            assert c[1]["limit"] == 15, (
                f"Expected limit=15 but got limit={c[1].get('limit')}"
            )

    @patch("data_acquisition.financials._persist_raw")
    @patch("data_acquisition.financials.resilient_request")
    @patch("data_acquisition.financials.build_fmp_url")
    @patch("data_acquisition.financials.get_config")
    def test_default_limit_when_config_key_absent(
        self, mock_config, mock_build, mock_req, mock_persist
    ) -> None:
        """If required_history_years is not in config, fall back to 10."""
        mock_config.return_value = {
            "universe": {},  # no required_history_years key
            "data_sources": {"store_raw_responses": False},
        }
        mock_build.return_value = ("http://example.com/api", {"apikey": "x"})
        mock_req.return_value = [_make_fmp_income_row()]

        fetch_financial_statements("AAPL")

        for c in mock_build.call_args_list:
            assert c[1]["limit"] == 10, (
                f"Expected default limit=10 but got limit={c[1].get('limit')}"
            )


# ---------------------------------------------------------------------------
# TSX ticker pre-suffixing in fetch_all_financials
# ---------------------------------------------------------------------------

class TestFetchAllFinancialsTsxSuffix:
    """TSX tickers without '.TO' suffix should be pre-suffixed."""

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_tsx_ticker_gets_dot_to_suffix(
        self, mock_fetch, mock_eta, mock_yf_fetch
    ) -> None:
        """A TSX ticker 'SHOP' should become 'SHOP.TO' before API call."""
        mock_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        mock_yf_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        # Universe with one TSX ticker without the .TO suffix.
        universe = pd.DataFrame({
            "ticker": ["SHOP"],
            "exchange": ["TSX"],
        })
        fetch_all_financials(universe, batch_size=10)

        # fetch_financial_statements must have been called with 'SHOP.TO'.
        mock_fetch.assert_called_once_with("SHOP.TO")

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_tsx_ticker_already_suffixed_not_doubled(
        self, mock_fetch, mock_eta, mock_yf_fetch
    ) -> None:
        """A TSX ticker 'SHOP.TO' should NOT become 'SHOP.TO.TO'."""
        mock_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        mock_yf_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        universe = pd.DataFrame({
            "ticker": ["SHOP.TO"],
            "exchange": ["TSX"],
        })
        fetch_all_financials(universe, batch_size=10)

        mock_fetch.assert_called_once_with("SHOP.TO")

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_us_ticker_not_suffixed(
        self, mock_fetch, mock_eta, mock_yf_fetch
    ) -> None:
        """A NYSE ticker must not receive any suffix."""
        mock_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        mock_yf_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        universe = pd.DataFrame({
            "ticker": ["AAPL"],
            "exchange": ["NYSE"],
        })
        fetch_all_financials(universe, batch_size=10)

        mock_fetch.assert_called_once_with("AAPL")

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_mixed_exchanges_only_tsx_suffixed(
        self, mock_fetch, mock_eta, mock_yf_fetch
    ) -> None:
        """Only TSX tickers should be suffixed; NYSE tickers left alone."""
        mock_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        mock_yf_fetch.return_value = {
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }
        universe = pd.DataFrame({
            "ticker": ["AAPL", "SHOP", "MSFT"],
            "exchange": ["NYSE", "TSX", "NASDAQ"],
        })
        fetch_all_financials(universe, batch_size=10)

        called_tickers = [c[0][0] for c in mock_fetch.call_args_list]
        assert "AAPL" in called_tickers
        assert "SHOP.TO" in called_tickers
        assert "MSFT" in called_tickers
        # Ensure SHOP (without .TO) was NOT called.
        assert "SHOP" not in called_tickers


# ---------------------------------------------------------------------------
# _fetch_statements_yfinance tests
# ---------------------------------------------------------------------------

def _make_yf_financials(years: list[int]) -> pd.DataFrame:
    """Build a mock yfinance .financials DataFrame (fields as index, dates as cols)."""
    cols = {pd.Timestamp(f"{y}-09-30"): {} for y in years}
    for ts in cols:
        cols[ts] = {
            "Net Income": 100_000_000_000,
            "Total Revenue": 400_000_000_000,
            "Gross Profit": 170_000_000_000,
            "Operating Income": 115_000_000_000,
            "Interest Expense": 3_000_000_000,
            "Diluted EPS": 6.13,
            "Diluted Average Shares": 15_812_547_000,
            "Selling General And Administration": 24_000_000_000,
        }
    return pd.DataFrame(cols)


def _make_yf_balance_sheet(years: list[int]) -> pd.DataFrame:
    """Build a mock yfinance .balance_sheet DataFrame."""
    cols = {pd.Timestamp(f"{y}-09-30"): {} for y in years}
    for ts in cols:
        cols[ts] = {
            "Long Term Debt": 110_000_000_000,
            "Stockholders Equity": 62_000_000_000,
            "Treasury Stock": -70_000_000_000,
        }
    return pd.DataFrame(cols)


def _make_yf_cashflow(years: list[int]) -> pd.DataFrame:
    """Build a mock yfinance .cashflow DataFrame."""
    cols = {pd.Timestamp(f"{y}-09-30"): {} for y in years}
    for ts in cols:
        cols[ts] = {
            "Depreciation And Amortization": 12_000_000_000,
            "Capital Expenditure": -11_000_000_000,
            "Change In Working Capital": -5_000_000_000,
        }
    return pd.DataFrame(cols)


class TestFetchStatementsYfinance:
    """Tests for the yfinance financial statement fallback."""

    @patch("yfinance.Ticker")
    def test_returns_dict_with_three_keys(self, mock_ticker_cls) -> None:
        mock_t = MagicMock()
        mock_ticker_cls.return_value = mock_t
        mock_t.financials = _make_yf_financials([2023])
        mock_t.balance_sheet = _make_yf_balance_sheet([2023])
        mock_t.cashflow = _make_yf_cashflow([2023])

        result = _fetch_statements_yfinance("AAPL")
        assert set(result.keys()) == {"income_statement", "balance_sheet", "cash_flow"}

    @patch("yfinance.Ticker")
    def test_transposed_dataframe_has_field_columns(self, mock_ticker_cls) -> None:
        """yfinance data is transposed so field names become columns."""
        mock_t = MagicMock()
        mock_ticker_cls.return_value = mock_t
        mock_t.financials = _make_yf_financials([2022, 2023])
        mock_t.balance_sheet = _make_yf_balance_sheet([2022, 2023])
        mock_t.cashflow = _make_yf_cashflow([2022, 2023])

        result = _fetch_statements_yfinance("AAPL")
        income_df = result["income_statement"]
        assert "Net Income" in income_df.columns
        assert "Total Revenue" in income_df.columns
        assert len(income_df) == 2  # Two years of data

    @patch("yfinance.Ticker")
    def test_calendar_year_column_added(self, mock_ticker_cls) -> None:
        mock_t = MagicMock()
        mock_ticker_cls.return_value = mock_t
        mock_t.financials = _make_yf_financials([2023])
        mock_t.balance_sheet = _make_yf_balance_sheet([2023])
        mock_t.cashflow = _make_yf_cashflow([2023])

        result = _fetch_statements_yfinance("AAPL")
        income_df = result["income_statement"]
        assert "calendarYear" in income_df.columns
        assert income_df.iloc[0]["calendarYear"] == "2023"

    @patch("yfinance.Ticker")
    def test_date_column_added(self, mock_ticker_cls) -> None:
        mock_t = MagicMock()
        mock_ticker_cls.return_value = mock_t
        mock_t.financials = _make_yf_financials([2023])
        mock_t.balance_sheet = _make_yf_balance_sheet([2023])
        mock_t.cashflow = _make_yf_cashflow([2023])

        result = _fetch_statements_yfinance("AAPL")
        income_df = result["income_statement"]
        assert "date" in income_df.columns
        assert income_df.iloc[0]["date"] == "2023-09-30"

    @patch("yfinance.Ticker")
    def test_empty_statement_returns_none(self, mock_ticker_cls) -> None:
        mock_t = MagicMock()
        mock_ticker_cls.return_value = mock_t
        mock_t.financials = pd.DataFrame()  # empty
        mock_t.balance_sheet = _make_yf_balance_sheet([2023])
        mock_t.cashflow = _make_yf_cashflow([2023])

        result = _fetch_statements_yfinance("AAPL")
        assert result["income_statement"] is None
        assert result["balance_sheet"] is not None
        assert result["cash_flow"] is not None

    @patch("yfinance.Ticker")
    def test_ticker_construction_failure_returns_all_none(self, mock_ticker_cls) -> None:
        mock_ticker_cls.side_effect = Exception("yfinance unavailable")
        result = _fetch_statements_yfinance("AAPL")
        assert all(v is None for v in result.values())

    @patch("yfinance.Ticker")
    def test_normalize_statement_handles_yfinance_data(self, mock_ticker_cls) -> None:
        """End-to-end: yfinance data can flow through normalize_statement."""
        mock_t = MagicMock()
        mock_ticker_cls.return_value = mock_t
        mock_t.financials = _make_yf_financials([2023])
        mock_t.balance_sheet = _make_yf_balance_sheet([2023])
        mock_t.cashflow = _make_yf_cashflow([2023])

        stmt_dfs = _fetch_statements_yfinance("AAPL")

        # Income statement should normalise via yfinance substitute fields
        norm_df, subs = normalize_statement(
            stmt_dfs["income_statement"], "income_statement", ticker="AAPL"
        )
        assert not norm_df.empty
        assert norm_df.iloc[0]["fiscal_year"] == 2023
        # net_income resolved from "Net Income" substitute (÷1000)
        assert norm_df.iloc[0]["net_income"] == pytest.approx(100_000_000.0)

    @patch("yfinance.Ticker")
    def test_cashflow_capex_sign_preserved(self, mock_ticker_cls) -> None:
        """CapEx from yfinance (already negative) should stay negative."""
        mock_t = MagicMock()
        mock_ticker_cls.return_value = mock_t
        mock_t.financials = _make_yf_financials([2023])
        mock_t.balance_sheet = _make_yf_balance_sheet([2023])
        mock_t.cashflow = _make_yf_cashflow([2023])

        stmt_dfs = _fetch_statements_yfinance("AAPL")
        norm_df, _ = normalize_statement(
            stmt_dfs["cash_flow"], "cash_flow", ticker="AAPL"
        )
        assert norm_df.iloc[0]["capital_expenditures"] < 0


# ---------------------------------------------------------------------------
# fetch_all_financials — yfinance fallback logic tests
# ---------------------------------------------------------------------------

class TestFetchAllFinancialsYfinanceFallback:
    """Tests for the automatic FMP → yfinance fallback in fetch_all_financials."""

    _ALL_NONE = {
        "income_statement": None,
        "balance_sheet": None,
        "cash_flow": None,
    }

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_yfinance_called_when_fmp_returns_all_none(
        self, mock_fmp, mock_eta, mock_yf
    ) -> None:
        """When FMP returns all-None, yfinance should be tried for that ticker."""
        mock_fmp.return_value = self._ALL_NONE
        mock_yf.return_value = self._ALL_NONE
        universe = _make_universe_df(["AAPL"])

        fetch_all_financials(universe, batch_size=10)

        mock_fmp.assert_called_once()
        mock_yf.assert_called_once_with("AAPL")

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_yfinance_not_called_when_fmp_succeeds(
        self, mock_fmp, mock_eta, mock_yf
    ) -> None:
        """When FMP returns data, yfinance should NOT be called."""
        mock_fmp.return_value = {
            "income_statement": pd.DataFrame([_make_fmp_income_row()]),
            "balance_sheet": pd.DataFrame([_make_fmp_balance_row()]),
            "cash_flow": pd.DataFrame([_make_fmp_cashflow_row()]),
        }
        universe = _make_universe_df(["AAPL"])

        fetch_all_financials(universe, batch_size=10)

        mock_fmp.assert_called_once()
        mock_yf.assert_not_called()

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_switches_to_yfinance_only_after_consecutive_failures(
        self, mock_fmp, mock_eta, mock_yf
    ) -> None:
        """After _MAX_CONSECUTIVE_FMP_FAILURES all-None results, FMP should be
        skipped entirely for remaining tickers."""
        mock_fmp.return_value = self._ALL_NONE
        mock_yf.return_value = self._ALL_NONE

        # Need enough tickers to exceed the threshold
        n = _MAX_CONSECUTIVE_FMP_FAILURES + 2
        tickers = [f"T{i}" for i in range(n)]
        universe = _make_universe_df(tickers)

        fetch_all_financials(universe, batch_size=100)

        # FMP should have been called for the first threshold tickers,
        # then skipped for the rest.
        assert mock_fmp.call_count == _MAX_CONSECUTIVE_FMP_FAILURES
        # yfinance should have been called for ALL tickers
        # (as fallback for the first N, and directly for the rest).
        assert mock_yf.call_count == n

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_fmp_success_resets_failure_counter(
        self, mock_fmp, mock_eta, mock_yf
    ) -> None:
        """An FMP success should reset the consecutive failure counter."""
        good_response = {
            "income_statement": pd.DataFrame([_make_fmp_income_row("T1")]),
            "balance_sheet": pd.DataFrame([_make_fmp_balance_row("T1")]),
            "cash_flow": pd.DataFrame([_make_fmp_cashflow_row("T1")]),
        }
        # 2 failures, then 1 success, then 2 more failures.
        # Total FMP failures never reach threshold of 3 consecutively.
        mock_fmp.side_effect = [
            self._ALL_NONE,      # T0: fail → try yfinance
            self._ALL_NONE,      # T1: fail → try yfinance
            good_response,       # T2: success → reset counter
            self._ALL_NONE,      # T3: fail → try yfinance (counter=1)
            self._ALL_NONE,      # T4: fail → try yfinance (counter=2)
        ]
        mock_yf.return_value = self._ALL_NONE
        universe = _make_universe_df([f"T{i}" for i in range(5)])

        fetch_all_financials(universe, batch_size=100)

        # All 5 tickers should have been tried via FMP
        # (never hit threshold of 3 consecutive).
        assert mock_fmp.call_count == 5
        # yfinance fallback called for the 4 failures, not for T2 (success).
        assert mock_yf.call_count == 4

    @patch("data_acquisition.financials._fetch_statements_yfinance")
    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_yfinance_data_normalised_and_collected(
        self, mock_fmp, mock_eta, mock_yf
    ) -> None:
        """yfinance fallback data should be normalised and included in output."""
        mock_fmp.return_value = self._ALL_NONE
        # Return real yfinance-style data
        mock_yf.return_value = {
            "income_statement": _make_yf_financials([2023]).T.assign(
                calendarYear="2023", date="2023-09-30"
            ).reset_index(drop=True),
            "balance_sheet": None,
            "cash_flow": None,
        }
        # Ensure column names match what yfinance transpose produces
        yf_income = _make_yf_financials([2023]).T.copy()
        dates = pd.to_datetime(yf_income.index)
        yf_income = yf_income.reset_index(drop=True)
        yf_income["calendarYear"] = dates.year.astype(str)
        yf_income["date"] = dates.strftime("%Y-%m-%d")
        mock_yf.return_value = {
            "income_statement": yf_income,
            "balance_sheet": None,
            "cash_flow": None,
        }

        universe = _make_universe_df(["AAPL"])
        result_dfs, subs = fetch_all_financials(universe, batch_size=10)

        # Income statement should have data from yfinance
        income = result_dfs["income_statement"]
        assert len(income) == 1
        assert income.iloc[0]["ticker"] == "AAPL"
        assert income.iloc[0]["fiscal_year"] == 2023

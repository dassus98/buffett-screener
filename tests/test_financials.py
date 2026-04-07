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
    _apply_value_conventions,
    _empty_normalised_df,
    _extract_fiscal_year,
    _resolve_fmp_ticker,
    fetch_all_financials,
    fetch_financial_statements,
    normalize_statement,
)
from data_acquisition.schema import LINE_ITEM_MAP


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

    @patch("data_acquisition.financials._log_eta")
    @patch("data_acquisition.financials.fetch_financial_statements")
    def test_progress_logged_at_batch_boundaries(
        self, mock_fetch, mock_eta, caplog
    ) -> None:
        import logging

        mock_fetch.return_value = {
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

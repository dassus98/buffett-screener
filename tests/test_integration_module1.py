"""Mock integration test for Module 1 — data_acquisition end-to-end.

Exercises the full normalise → quality-check → DuckDB-write → read-back
pipeline using synthetic FMP-shaped data.  No live API calls are made.

Universe:
  Complete tickers (10 fiscal years each): AAPL, KO, MSFT, RY.TO, CNR.TO
  Thin ticker   (3 fiscal years only):     THIN

Design decisions:
  - KO income statement uses ``netIncomeFromContinuingOperations`` (a valid
    substitute for ``net_income``) instead of the ideal ``netIncome`` field, so
    that the substitution log is non-empty.
  - THIN has only 3 years of data, below the configured
    ``universe.min_history_years = 4``, and must be marked drop=True.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import data_acquisition.store as store_module
from data_acquisition.data_quality import run_data_quality_check
from data_acquisition.financials import normalize_statement
from data_acquisition.store import (
    close,
    get_surviving_tickers,
    init_db,
    read_table,
    write_dataframe,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMPLETE_TICKERS: list[str] = ["AAPL", "KO", "MSFT", "RY.TO", "CNR.TO"]
_THIN_TICKER: str = "THIN"
_ALL_TICKERS: list[str] = _COMPLETE_TICKERS + [_THIN_TICKER]

_N_COMPLETE_YEARS: int = 10
_N_THIN_YEARS: int = 3
_START_YEAR: int = 2014  # oldest fiscal year in complete tickers

# Expected total rows per statement table (5 × 10 + 1 × 3 = 53).
_EXPECTED_STMT_ROWS: int = _N_COMPLETE_YEARS * len(_COMPLETE_TICKERS) + _N_THIN_YEARS

# KO uses a substitute for net_income → one sub-log entry per fiscal year.
_KO_SUB_LOG_ROWS: int = _N_COMPLETE_YEARS

# ---------------------------------------------------------------------------
# Raw-data helpers (FMP field names, full USD dollars)
# ---------------------------------------------------------------------------

def _raw_income_rows(ticker: str, n_years: int) -> pd.DataFrame:
    """Build a raw FMP income statement DataFrame for a single ticker.

    KO intentionally uses ``netIncomeFromContinuingOperations`` (a substitute)
    instead of the ideal ``netIncome`` field to exercise the substitution log.
    """
    rows: list[dict[str, Any]] = []
    for i, year in enumerate(range(_START_YEAR, _START_YEAR + n_years)):
        base = float(i + 1)
        row: dict[str, Any] = {
            "calendarYear": year,
            "totalRevenue": 10_000_000_000 * base,
            "grossProfit": 4_000_000_000 * base,
            "sellingGeneralAndAdministrativeExpenses": 500_000_000 * base,
            "operatingIncome": 2_000_000_000 * base,
            "interestExpense": 100_000_000 * base,
            "epsdiluted": round(3.5 + i * 0.1, 2),
            "weightedAverageSharesDiluted": 1_000_000_000.0,
        }
        # KO: use substitute field; all others: use ideal field.
        if ticker == "KO":
            row["netIncomeFromContinuingOperations"] = 1_000_000_000 * base
        else:
            row["netIncome"] = 1_000_000_000 * base
        rows.append(row)
    return pd.DataFrame(rows)


def _raw_balance_rows(n_years: int) -> pd.DataFrame:
    """Build a raw FMP balance sheet DataFrame (same values across all tickers)."""
    rows: list[dict[str, Any]] = []
    for i, year in enumerate(range(_START_YEAR, _START_YEAR + n_years)):
        rows.append({
            "calendarYear": year,
            "longTermDebt": 5_000_000_000.0,
            "totalStockholdersEquity": 30_000_000_000.0,
            "treasuryStock": 2_000_000_000.0,
        })
    return pd.DataFrame(rows)


def _raw_cashflow_rows(n_years: int) -> pd.DataFrame:
    """Build a raw FMP cash flow DataFrame (same values across all tickers)."""
    rows: list[dict[str, Any]] = []
    for i, year in enumerate(range(_START_YEAR, _START_YEAR + n_years)):
        rows.append({
            "calendarYear": year,
            "depreciationAndAmortization": 3_000_000_000.0,
            "capitalExpenditure": -2_000_000_000.0,   # negative = outflow
            "changeInWorkingCapital": 500_000_000.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _build_financials() -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Normalise synthetic raw data for all tickers and concatenate into financials dict."""
    income_frames: list[pd.DataFrame] = []
    balance_frames: list[pd.DataFrame] = []
    cashflow_frames: list[pd.DataFrame] = []
    all_sub_log: list[dict[str, Any]] = []

    for ticker in _ALL_TICKERS:
        n = _N_COMPLETE_YEARS if ticker != _THIN_TICKER else _N_THIN_YEARS

        inc_df, sub1 = normalize_statement(_raw_income_rows(ticker, n), "income_statement", ticker)
        bal_df, sub2 = normalize_statement(_raw_balance_rows(n), "balance_sheet", ticker)
        cf_df, sub3 = normalize_statement(_raw_cashflow_rows(n), "cash_flow", ticker)

        income_frames.append(inc_df)
        balance_frames.append(bal_df)
        cashflow_frames.append(cf_df)
        all_sub_log.extend(sub1 + sub2 + sub3)

    financials: dict[str, pd.DataFrame] = {
        "income_statement": pd.concat(income_frames, ignore_index=True),
        "balance_sheet": pd.concat(balance_frames, ignore_index=True),
        "cash_flow": pd.concat(cashflow_frames, ignore_index=True),
    }
    return financials, all_sub_log


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_connection():
    """Ensure a clean DuckDB connection state around each test."""
    close()
    yield
    close()


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Full Module 1 pipeline run against a temporary DuckDB.

    Returns a dict with keys:
        financials, sub_log, quality_df, survivors_df, db_path
    """
    db_path = tmp_path / "integration.duckdb"
    monkeypatch.setattr(store_module, "DB_PATH", db_path)
    init_db(db_path=db_path)

    financials, sub_log = _build_financials()
    quality_df, survivors_df = run_data_quality_check(financials, sub_log)

    # Persist all results to DuckDB.
    for table_name in ("income_statement", "balance_sheet", "cash_flow"):
        stmt_df = financials[table_name]
        if not stmt_df.empty:
            write_dataframe(table_name, stmt_df)

    _COLS = ["ticker", "fiscal_year", "buffett_field", "api_field_used", "confidence"]
    sub_df = pd.DataFrame(sub_log) if sub_log else pd.DataFrame(columns=_COLS)
    write_dataframe("substitution_log", sub_df)

    if not quality_df.empty:
        write_dataframe("data_quality_log", quality_df)

    yield {
        "financials": financials,
        "sub_log": sub_log,
        "quality_df": quality_df,
        "survivors_df": survivors_df,
        "db_path": db_path,
    }


# ===========================================================================
# Tests — normalize_statement behaviour
# ===========================================================================

class TestNormalizeStatement:
    def test_income_output_has_canonical_columns(self, pipeline):
        inc = pipeline["financials"]["income_statement"]
        expected_cols = {
            "ticker", "fiscal_year", "net_income", "total_revenue", "gross_profit",
            "sga", "operating_income", "interest_expense", "eps_diluted",
            "shares_outstanding_diluted",
        }
        assert expected_cols == set(inc.columns), f"Unexpected columns: {set(inc.columns)}"

    def test_monetary_fields_converted_to_thousands(self, pipeline):
        # netIncome = 1_000_000_000 (full USD) → 1_000_000 (thousands)
        aapl = pipeline["financials"]["income_statement"]
        aapl_row = aapl[aapl["ticker"] == "AAPL"].sort_values("fiscal_year").iloc[0]
        assert aapl_row["net_income"] == pytest.approx(1_000_000.0)
        assert aapl_row["total_revenue"] == pytest.approx(10_000_000.0)

    def test_eps_not_divided_by_1000(self, pipeline):
        # epsdiluted is per-share → returned as-is (not ÷1000)
        inc = pipeline["financials"]["income_statement"]
        row = inc[inc["ticker"] == "AAPL"].sort_values("fiscal_year").iloc[0]
        assert row["eps_diluted"] == pytest.approx(3.5)

    def test_shares_not_divided_by_1000(self, pipeline):
        inc = pipeline["financials"]["income_statement"]
        row = inc[inc["ticker"] == "AAPL"].iloc[0]
        assert row["shares_outstanding_diluted"] == pytest.approx(1_000_000_000.0)

    def test_capex_sign_preserved_when_already_negative(self, pipeline):
        # capitalExpenditure = -2_000_000_000 → stored as -2_000_000 (no sign flip)
        cf = pipeline["financials"]["cash_flow"]
        row = cf[cf["ticker"] == "AAPL"].iloc[0]
        assert row["capital_expenditures"] < 0

    def test_ko_substitute_generates_sub_log_entries(self, pipeline):
        # KO uses netIncomeFromContinuingOperations → 10 sub-log entries
        sub_log = pipeline["sub_log"]
        ko_net_income_subs = [
            e for e in sub_log
            if e["ticker"] == "KO" and e["buffett_field"] == "net_income"
        ]
        assert len(ko_net_income_subs) == _N_COMPLETE_YEARS

    def test_ko_substitute_field_name_logged(self, pipeline):
        sub_log = pipeline["sub_log"]
        entry = next(e for e in sub_log if e["ticker"] == "KO" and e["buffett_field"] == "net_income")
        assert entry["api_field_used"] == "netIncomeFromContinuingOperations"
        assert entry["confidence"] == "High"

    def test_complete_tickers_have_10_fiscal_years_each(self, pipeline):
        inc = pipeline["financials"]["income_statement"]
        for ticker in _COMPLETE_TICKERS:
            rows = inc[inc["ticker"] == ticker]
            assert len(rows) == _N_COMPLETE_YEARS, f"{ticker}: expected 10 rows, got {len(rows)}"

    def test_thin_ticker_has_3_fiscal_years(self, pipeline):
        inc = pipeline["financials"]["income_statement"]
        thin_rows = inc[inc["ticker"] == _THIN_TICKER]
        assert len(thin_rows) == _N_THIN_YEARS


# ===========================================================================
# Tests — run_data_quality_check behaviour
# ===========================================================================

class TestDataQualityCheck:
    def test_complete_tickers_have_drop_false(self, pipeline):
        qdf = pipeline["quality_df"]
        for ticker in _COMPLETE_TICKERS:
            row = qdf[qdf["ticker"] == ticker]
            assert len(row) == 1, f"Expected 1 row for {ticker}"
            assert row["drop"].iloc[0] == False, f"{ticker} unexpectedly dropped"  # noqa: E712

    def test_thin_ticker_has_drop_true(self, pipeline):
        qdf = pipeline["quality_df"]
        thin_row = qdf[qdf["ticker"] == _THIN_TICKER]
        assert len(thin_row) == 1
        assert thin_row["drop"].iloc[0] == True  # noqa: E712

    def test_thin_ticker_drop_reason_mentions_insufficient(self, pipeline):
        qdf = pipeline["quality_df"]
        reason = qdf[qdf["ticker"] == _THIN_TICKER]["drop_reason"].iloc[0]
        assert reason is not None
        assert "Insufficient" in reason or "insufficient" in reason.lower()

    def test_survivors_df_excludes_thin(self, pipeline):
        survivors = pipeline["survivors_df"]["ticker"].tolist()
        assert _THIN_TICKER not in survivors

    def test_survivors_df_contains_all_complete_tickers(self, pipeline):
        survivors = set(pipeline["survivors_df"]["ticker"].tolist())
        assert set(_COMPLETE_TICKERS).issubset(survivors)

    def test_quality_df_has_one_row_per_ticker(self, pipeline):
        assert len(pipeline["quality_df"]) == len(_ALL_TICKERS)


# ===========================================================================
# Tests — DuckDB write-and-read-back
# ===========================================================================

class TestDuckDBRoundtrip:
    def test_income_statement_row_count(self, pipeline):
        result = read_table("income_statement")
        assert len(result) == _EXPECTED_STMT_ROWS

    def test_balance_sheet_row_count(self, pipeline):
        result = read_table("balance_sheet")
        assert len(result) == _EXPECTED_STMT_ROWS

    def test_cash_flow_row_count(self, pipeline):
        result = read_table("cash_flow")
        assert len(result) == _EXPECTED_STMT_ROWS

    def test_income_statement_column_names(self, pipeline):
        result = read_table("income_statement")
        expected = {
            "ticker", "fiscal_year", "net_income", "total_revenue", "gross_profit",
            "sga", "operating_income", "interest_expense", "eps_diluted",
            "shares_outstanding_diluted",
        }
        assert set(result.columns) == expected

    def test_substitution_log_has_ko_entries(self, pipeline):
        result = read_table("substitution_log")
        ko_rows = result[result["ticker"] == "KO"]
        assert len(ko_rows) == _KO_SUB_LOG_ROWS

    def test_substitution_log_field_and_confidence(self, pipeline):
        result = read_table("substitution_log")
        net_income_subs = result[
            (result["ticker"] == "KO") & (result["buffett_field"] == "net_income")
        ]
        assert len(net_income_subs) > 0
        assert (net_income_subs["confidence"] == "High").all()
        assert (net_income_subs["api_field_used"] == "netIncomeFromContinuingOperations").all()

    def test_data_quality_log_row_count(self, pipeline):
        result = read_table("data_quality_log")
        assert len(result) == len(_ALL_TICKERS)

    def test_data_quality_log_thin_is_dropped(self, pipeline):
        result = read_table("data_quality_log")
        thin_row = result[result["ticker"] == _THIN_TICKER]
        assert len(thin_row) == 1
        assert thin_row["drop"].iloc[0] == True  # noqa: E712

    def test_data_quality_log_complete_tickers_not_dropped(self, pipeline):
        result = read_table("data_quality_log")
        for ticker in _COMPLETE_TICKERS:
            row = result[result["ticker"] == ticker]
            assert row["drop"].iloc[0] == False, f"{ticker} unexpectedly dropped in DB"  # noqa: E712

    def test_get_surviving_tickers_excludes_thin(self, pipeline):
        survivors = get_surviving_tickers()
        assert _THIN_TICKER not in survivors
        assert len(survivors) == len(_COMPLETE_TICKERS)

    def test_balance_sheet_column_names(self, pipeline):
        result = read_table("balance_sheet")
        expected = {"ticker", "fiscal_year", "long_term_debt", "shareholders_equity", "treasury_stock"}
        assert set(result.columns) == expected

    def test_cash_flow_column_names(self, pipeline):
        result = read_table("cash_flow")
        expected = {"ticker", "fiscal_year", "depreciation_amortization",
                    "capital_expenditures", "working_capital_change"}
        assert set(result.columns) == expected

    def test_income_values_are_in_thousands(self, pipeline):
        result = read_table("income_statement")
        aapl_2014 = result[(result["ticker"] == "AAPL") & (result["fiscal_year"] == _START_YEAR)]
        assert len(aapl_2014) == 1
        # netIncome = 1_000_000_000 full USD → 1_000_000 USD thousands
        assert aapl_2014["net_income"].iloc[0] == pytest.approx(1_000_000.0)

"""Unit tests for data_acquisition/store.py.

All tests use a temporary DuckDB file (via tmp_path) so the production
database is never touched. The connection singleton is closed after each
test that opens one.

Coverage:
  - get_connection / close: singleton lifecycle, parent dir creation
  - init_db: idempotent table creation, column verification
  - write_dataframe: replace mode, append mode (upsert), empty/None df,
    extra columns, invalid table/mode, all 8 table types roundtrip
  - read_table: full reads, WHERE clause, empty tables, missing tables
  - get_surviving_tickers: pass/fail filtering, empty/all-dropped cases
  - Column alignment: store._TABLE_COLUMNS vs upstream module constants
"""

from __future__ import annotations

import pandas as pd
import pytest

import data_acquisition.store as store_module
from data_acquisition.store import (
    _TABLE_COLUMNS,
    _TABLE_DDL,
    _TABLE_PK_COLUMNS,
    close,
    get_connection,
    get_surviving_tickers,
    init_db,
    read_table,
    write_dataframe,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_connection():
    """Close any open connection before and after every test."""
    close()
    yield
    close()


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Initialised temporary DuckDB with DB_PATH patched; closes on teardown.

    Patching DB_PATH ensures that write_dataframe / read_table / get_surviving_tickers,
    which call get_connection() with no explicit path, all use the temp file.
    """
    path = tmp_path / "test.duckdb"
    monkeypatch.setattr(store_module, "DB_PATH", path)
    init_db(db_path=path)
    yield path
    close()


def _universe_df(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": tickers,
        "exchange": ["NASDAQ"] * len(tickers),
        "company_name": [f"{t} Inc" for t in tickers],
        "market_cap_usd": [1e12] * len(tickers),
        "sector": ["Technology"] * len(tickers),
        "industry": ["Software"] * len(tickers),
        "country": ["US"] * len(tickers),
    })


def _income_df(ticker: str, years: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [ticker] * len(years),
        "fiscal_year": years,
        "net_income": [1000.0 * y for y in years],
        "total_revenue": [5000.0 * y for y in years],
        "gross_profit": [2000.0 * y for y in years],
        "sga": [300.0] * len(years),
        "operating_income": [800.0] * len(years),
        "interest_expense": [50.0] * len(years),
        "eps_diluted": [3.5] * len(years),
        "shares_outstanding_diluted": [1e7] * len(years),
    })


def _quality_log_df(
    tickers: list[str],
    drops: list[bool],
    reasons: list[str | None] | None = None,
) -> pd.DataFrame:
    n = len(tickers)
    reasons = reasons or [None] * n
    return pd.DataFrame({
        "ticker": tickers,
        "years_available": [10] * n,
        "missing_critical_fields": [""] * n,
        "substitutions_count": [0] * n,
        "drop": drops,
        "drop_reason": reasons,
    })


def _balance_df(ticker: str, years: list[int]) -> pd.DataFrame:
    """Build a minimal balance sheet DataFrame for testing."""
    return pd.DataFrame({
        "ticker": [ticker] * len(years),
        "fiscal_year": years,
        "long_term_debt": [50_000.0] * len(years),
        "shareholders_equity": [120_000.0] * len(years),
        "treasury_stock": [-5_000.0] * len(years),
    })


def _cashflow_df(ticker: str, years: list[int]) -> pd.DataFrame:
    """Build a minimal cash flow DataFrame for testing."""
    return pd.DataFrame({
        "ticker": [ticker] * len(years),
        "fiscal_year": years,
        "depreciation_amortization": [8_000.0] * len(years),
        "capital_expenditures": [-6_500.0] * len(years),
        "working_capital_change": [1_200.0] * len(years),
    })


def _market_data_df(tickers: list[str]) -> pd.DataFrame:
    """Build a minimal market_data DataFrame for testing."""
    return pd.DataFrame({
        "ticker": tickers,
        "current_price_usd": [150.0] * len(tickers),
        "market_cap_usd": [2e12] * len(tickers),
        "enterprise_value_usd": [2.1e12] * len(tickers),
        "shares_outstanding": [15e9] * len(tickers),
        "high_52w": [200.0] * len(tickers),
        "low_52w": [120.0] * len(tickers),
        "avg_volume_3m": [50e6] * len(tickers),
        "pe_ratio_trailing": [28.5] * len(tickers),
        "dividend_yield": [0.005] * len(tickers),
        "as_of_date": ["2024-01-15"] * len(tickers),
    })


def _macro_data_df() -> pd.DataFrame:
    """Build a minimal macro_data DataFrame for testing."""
    return pd.DataFrame({
        "key": ["us_treasury_10yr", "usd_cad_rate"],
        "value": [0.0425, 0.74],
        "as_of_date": ["2024-01-15", "2024-01-15"],
    })


def _substitution_log_df() -> pd.DataFrame:
    """Build a minimal substitution_log DataFrame for testing."""
    return pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "fiscal_year": [2023, 2023, 2022],
        "buffett_field": ["sga", "gross_profit", "operating_income"],
        "api_field_used": ["generalAndAdminExpenses", "DERIVED:totalRevenue-costOfRevenue", "ebit"],
        "confidence": ["Medium", "High", "High"],
    })


# ===========================================================================
# TestGetConnection
# ===========================================================================

class TestGetConnection:
    def test_returns_duckdb_connection(self, tmp_path):
        import duckdb
        path = tmp_path / "conn.duckdb"
        conn = get_connection(db_path=path)
        assert isinstance(conn, duckdb.DuckDBPyConnection)

    def test_creates_db_file(self, tmp_path):
        path = tmp_path / "created.duckdb"
        assert not path.exists()
        get_connection(db_path=path)
        assert path.exists()

    def test_singleton_same_path_returns_same_object(self, tmp_path):
        path = tmp_path / "singleton.duckdb"
        conn1 = get_connection(db_path=path)
        conn2 = get_connection(db_path=path)
        assert conn1 is conn2

    def test_different_path_creates_new_connection(self, tmp_path):
        path1 = tmp_path / "db1.duckdb"
        path2 = tmp_path / "db2.duckdb"
        conn1 = get_connection(db_path=path1)
        conn2 = get_connection(db_path=path2)
        assert conn1 is not conn2

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.duckdb"
        get_connection(db_path=path)
        assert path.exists()


# ===========================================================================
# TestClose
# ===========================================================================

class TestClose:
    def test_close_with_no_connection_is_safe(self):
        # autouse fixture already closed; second close must not raise.
        close()

    def test_close_allows_reconnect(self, tmp_path):
        import duckdb
        path = tmp_path / "recon.duckdb"
        get_connection(db_path=path)
        close()
        conn = get_connection(db_path=path)
        assert isinstance(conn, duckdb.DuckDBPyConnection)

    def test_close_resets_singleton_so_new_object_returned(self, tmp_path):
        path = tmp_path / "clear.duckdb"
        conn1 = get_connection(db_path=path)
        close()
        conn2 = get_connection(db_path=path)
        assert conn1 is not conn2


# ===========================================================================
# TestInitDb
# ===========================================================================

class TestInitDb:
    def test_all_eight_tables_created(self, db):
        conn = get_connection()
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchdf()["table_name"].tolist()
        for expected in _TABLE_DDL:
            assert expected in tables, f"Table '{expected}' not found after init_db"

    def test_idempotent_second_call_does_not_raise(self, db, tmp_path):
        # init_db already called by fixture; repeating must succeed.
        path = tmp_path / "test.duckdb"
        init_db(db_path=path)

    def test_data_quality_log_has_drop_column(self, db):
        df = read_table("data_quality_log")
        assert "drop" in df.columns

    def test_income_statement_has_correct_columns(self, db):
        df = read_table("income_statement")
        for col in ("ticker", "fiscal_year", "net_income", "eps_diluted",
                    "shares_outstanding_diluted"):
            assert col in df.columns


# ===========================================================================
# TestWriteDataframe
# ===========================================================================

class TestWriteDataframe:
    def test_roundtrip_universe(self, db):
        df = _universe_df(["AAPL", "MSFT"])
        write_dataframe("universe", df)
        result = read_table("universe")
        assert set(result["ticker"]) == {"AAPL", "MSFT"}

    def test_replace_mode_removes_existing_rows(self, db):
        write_dataframe("universe", _universe_df(["AAPL", "MSFT"]))
        write_dataframe("universe", _universe_df(["GOOG"]), mode="replace")
        result = read_table("universe")
        assert list(result["ticker"]) == ["GOOG"]

    def test_append_mode_adds_new_pk_rows(self, db):
        write_dataframe("income_statement", _income_df("AAPL", [2023]))
        write_dataframe("income_statement", _income_df("AAPL", [2022]), mode="append")
        result = read_table("income_statement")
        assert set(result["fiscal_year"]) == {2022, 2023}

    def test_append_mode_upserts_conflicting_pk(self, db):
        write_dataframe("income_statement", _income_df("AAPL", [2023]))
        updated = _income_df("AAPL", [2023])
        updated["net_income"] = 9999.0
        write_dataframe("income_statement", updated, mode="append")
        result = read_table("income_statement")
        assert len(result) == 1
        assert result["net_income"].iloc[0] == pytest.approx(9999.0)

    def test_empty_dataframe_is_skipped(self, db):
        write_dataframe("universe", _universe_df(["AAPL"]))
        write_dataframe("universe", pd.DataFrame(), mode="replace")
        result = read_table("universe")
        assert len(result) == 1  # Empty write skipped; original row preserved.

    def test_unknown_table_raises_value_error(self, db):
        with pytest.raises(ValueError, match="Unknown table"):
            write_dataframe("no_such_table", _universe_df(["AAPL"]))

    def test_extra_df_columns_are_ignored(self, db):
        df = _universe_df(["AAPL"])
        df["extra_col"] = "surprise"
        write_dataframe("universe", df)
        result = read_table("universe")
        assert "extra_col" not in result.columns
        assert "AAPL" in result["ticker"].values

    def test_data_quality_log_roundtrip_with_drop_column(self, db):
        log_df = _quality_log_df(["AAPL", "JUNK"], [False, True])
        write_dataframe("data_quality_log", log_df)
        result = read_table("data_quality_log")
        assert set(result["ticker"]) == {"AAPL", "JUNK"}
        aapl_drop = result.loc[result["ticker"] == "AAPL", "drop"].iloc[0]
        assert aapl_drop == False  # noqa: E712 — explicit bool comparison

    def test_invalid_mode_raises_value_error(self, db):
        with pytest.raises(ValueError, match="mode must be"):
            write_dataframe("universe", _universe_df(["AAPL"]), mode="overwrite")


# ===========================================================================
# TestReadTable
# ===========================================================================

class TestReadTable:
    def test_reads_all_rows(self, db):
        write_dataframe("universe", _universe_df(["AAPL", "MSFT", "GOOG"]))
        result = read_table("universe")
        assert len(result) == 3

    def test_where_clause_filters_rows(self, db):
        write_dataframe("universe", _universe_df(["AAPL", "MSFT"]))
        result = read_table("universe", where="ticker = 'AAPL'")
        assert len(result) == 1
        assert result["ticker"].iloc[0] == "AAPL"

    def test_empty_table_returns_empty_df(self, db):
        result = read_table("universe")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_unknown_table_raises_value_error(self, db):
        with pytest.raises(ValueError, match="Unknown table"):
            read_table("no_such_table")

    def test_table_not_created_raises_runtime_error(self, tmp_path, monkeypatch):
        # Connect to an empty DB (no init_db) → SELECT fails with CatalogException.
        path = tmp_path / "empty.duckdb"
        monkeypatch.setattr(store_module, "DB_PATH", path)
        get_connection(db_path=path)
        with pytest.raises(RuntimeError, match="not found in database"):
            read_table("universe")


# ===========================================================================
# TestGetSurvivingTickers
# ===========================================================================

class TestGetSurvivingTickers:
    def test_returns_tickers_where_drop_is_false(self, db):
        write_dataframe("data_quality_log", _quality_log_df(
            ["AAPL", "MSFT", "JUNK"], [False, False, True]
        ))
        result = get_surviving_tickers()
        assert "AAPL" in result
        assert "MSFT" in result

    def test_excludes_dropped_tickers(self, db):
        write_dataframe("data_quality_log", _quality_log_df(
            ["AAPL", "JUNK"], [False, True]
        ))
        result = get_surviving_tickers()
        assert "JUNK" not in result

    def test_result_is_sorted_alphabetically(self, db):
        write_dataframe("data_quality_log", _quality_log_df(
            ["MSFT", "AAPL", "GOOG"], [False, False, False]
        ))
        result = get_surviving_tickers()
        assert result == sorted(result)

    def test_empty_table_returns_empty_list(self, db):
        result = get_surviving_tickers()
        assert result == []

    def test_all_dropped_returns_empty_list(self, db):
        write_dataframe("data_quality_log", _quality_log_df(
            ["JUNK1", "JUNK2"], [True, True]
        ))
        result = get_surviving_tickers()
        assert result == []


# ===========================================================================
# TestWriteDataframeAllTables — roundtrip coverage for every table type
# ===========================================================================

class TestWriteDataframeAllTables:
    """Roundtrip write+read tests for table types not covered above."""

    def test_balance_sheet_roundtrip(self, db):
        df = _balance_df("AAPL", [2022, 2023])
        write_dataframe("balance_sheet", df)
        result = read_table("balance_sheet")
        assert len(result) == 2
        assert set(result["fiscal_year"]) == {2022, 2023}
        assert result["long_term_debt"].iloc[0] == pytest.approx(50_000.0)

    def test_cash_flow_roundtrip(self, db):
        df = _cashflow_df("AAPL", [2022, 2023])
        write_dataframe("cash_flow", df)
        result = read_table("cash_flow")
        assert len(result) == 2
        # Verify CapEx sign convention preserved (stored as negative).
        assert result["capital_expenditures"].iloc[0] < 0

    def test_market_data_roundtrip(self, db):
        df = _market_data_df(["AAPL", "MSFT"])
        write_dataframe("market_data", df)
        result = read_table("market_data")
        assert set(result["ticker"]) == {"AAPL", "MSFT"}
        assert result["current_price_usd"].iloc[0] == pytest.approx(150.0)
        assert result["as_of_date"].iloc[0] == "2024-01-15"

    def test_macro_data_roundtrip(self, db):
        df = _macro_data_df()
        write_dataframe("macro_data", df)
        result = read_table("macro_data")
        assert len(result) == 2
        treasury = result[result["key"] == "us_treasury_10yr"]
        assert treasury["value"].iloc[0] == pytest.approx(0.0425)

    def test_substitution_log_roundtrip(self, db):
        df = _substitution_log_df()
        write_dataframe("substitution_log", df)
        result = read_table("substitution_log")
        assert len(result) == 3
        aapl_rows = result[result["ticker"] == "AAPL"]
        assert len(aapl_rows) == 2

    def test_substitution_log_composite_pk_upsert(self, db):
        """Substitution_log has a 3-column composite PK (ticker, fiscal_year,
        buffett_field). Append mode should upsert correctly."""
        df = _substitution_log_df()
        write_dataframe("substitution_log", df)

        # Update the confidence for (AAPL, 2023, sga) — existing PK.
        update = pd.DataFrame({
            "ticker": ["AAPL"],
            "fiscal_year": [2023],
            "buffett_field": ["sga"],
            "api_field_used": ["new_field"],
            "confidence": ["Low"],
        })
        write_dataframe("substitution_log", update, mode="append")
        result = read_table("substitution_log")
        # 3 original rows, 1 upserted → still 3 total.
        assert len(result) == 3
        # Check the updated row.
        sga_row = result[
            (result["ticker"] == "AAPL") & (result["buffett_field"] == "sga")
        ]
        assert sga_row["confidence"].iloc[0] == "Low"
        assert sga_row["api_field_used"].iloc[0] == "new_field"


# ===========================================================================
# TestWriteDataframeEdgeCases — additional edge cases
# ===========================================================================

class TestWriteDataframeEdgeCases:
    """Edge cases for write_dataframe not covered above."""

    def test_none_dataframe_is_skipped(self, db):
        """Passing None for df should be a no-op (logged warning, no crash)."""
        write_dataframe("universe", _universe_df(["AAPL"]))
        write_dataframe("universe", None)  # type: ignore[arg-type]
        result = read_table("universe")
        assert len(result) == 1  # Original row preserved.

    def test_replace_then_append_accumulates(self, db):
        """Replace clears, then append adds without clearing."""
        write_dataframe("income_statement", _income_df("AAPL", [2022]))
        write_dataframe("income_statement", _income_df("MSFT", [2023]), mode="replace")
        # Only MSFT 2023 should exist after replace.
        result = read_table("income_statement")
        assert len(result) == 1
        assert result["ticker"].iloc[0] == "MSFT"

        # Append AAPL 2022 — should add without clearing MSFT.
        write_dataframe("income_statement", _income_df("AAPL", [2022]), mode="append")
        result = read_table("income_statement")
        assert len(result) == 2
        assert set(result["ticker"]) == {"AAPL", "MSFT"}

    def test_macro_data_replace_overwrites_all_keys(self, db):
        """Replace mode on macro_data should clear all rows, not just matching PKs."""
        write_dataframe("macro_data", _macro_data_df())
        assert len(read_table("macro_data")) == 2
        # Replace with a single row.
        new_df = pd.DataFrame({
            "key": ["new_indicator"],
            "value": [99.9],
            "as_of_date": ["2025-01-01"],
        })
        write_dataframe("macro_data", new_df, mode="replace")
        result = read_table("macro_data")
        assert len(result) == 1
        assert result["key"].iloc[0] == "new_indicator"


# ===========================================================================
# TestColumnAlignment — verify store columns match upstream module constants
# ===========================================================================

class TestColumnAlignment:
    """Verify _TABLE_COLUMNS matches the column definitions in upstream modules."""

    def test_universe_columns_match_upstream(self):
        from data_acquisition.universe import UNIVERSE_COLUMNS
        assert _TABLE_COLUMNS["universe"] == UNIVERSE_COLUMNS

    def test_market_data_columns_match_upstream(self):
        from data_acquisition.market_data import MARKET_DATA_COLUMNS
        assert _TABLE_COLUMNS["market_data"] == MARKET_DATA_COLUMNS

    def test_income_statement_columns_match_schema(self):
        """Income statement table columns should include ticker, fiscal_year,
        plus all 8 income_statement fields from schema.py."""
        from data_acquisition.schema import LINE_ITEM_MAP
        schema_fields = [
            name for name, item in LINE_ITEM_MAP.items()
            if item.statement == "income_statement"
        ]
        table_cols = list(_TABLE_COLUMNS["income_statement"])
        for field in schema_fields:
            assert field in table_cols, f"schema field '{field}' missing from income_statement DDL"
        # ticker and fiscal_year are structural columns, not schema fields.
        assert "ticker" in table_cols
        assert "fiscal_year" in table_cols

    def test_balance_sheet_columns_match_schema(self):
        from data_acquisition.schema import LINE_ITEM_MAP
        schema_fields = [
            name for name, item in LINE_ITEM_MAP.items()
            if item.statement == "balance_sheet"
        ]
        table_cols = list(_TABLE_COLUMNS["balance_sheet"])
        for field in schema_fields:
            assert field in table_cols, f"schema field '{field}' missing from balance_sheet DDL"

    def test_cash_flow_columns_match_schema(self):
        from data_acquisition.schema import LINE_ITEM_MAP
        schema_fields = [
            name for name, item in LINE_ITEM_MAP.items()
            if item.statement == "cash_flow"
        ]
        table_cols = list(_TABLE_COLUMNS["cash_flow"])
        for field in schema_fields:
            assert field in table_cols, f"schema field '{field}' missing from cash_flow DDL"

    def test_all_tables_have_pk_columns_defined(self):
        """Every table in _TABLE_DDL must have a corresponding entry in _TABLE_PK_COLUMNS."""
        for table_name in _TABLE_DDL:
            assert table_name in _TABLE_PK_COLUMNS, (
                f"Table '{table_name}' missing from _TABLE_PK_COLUMNS"
            )

    def test_pk_columns_are_subset_of_table_columns(self):
        """PK columns for every table must be present in that table's column list."""
        for table_name, pk_cols in _TABLE_PK_COLUMNS.items():
            table_cols = set(_TABLE_COLUMNS[table_name])
            for pk_col in pk_cols:
                assert pk_col in table_cols, (
                    f"PK column '{pk_col}' not in _TABLE_COLUMNS['{table_name}']"
                )

    def test_eight_tables_defined(self):
        """Pipeline requires exactly 8 tables."""
        expected = {
            "universe", "income_statement", "balance_sheet", "cash_flow",
            "market_data", "macro_data", "data_quality_log", "substitution_log",
        }
        assert set(_TABLE_DDL.keys()) == expected

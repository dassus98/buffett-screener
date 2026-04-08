"""Unit tests for data_acquisition/store.py.

All tests use a temporary DuckDB file (via tmp_path) so the production
database is never touched. The connection singleton is closed after each
test that opens one.
"""

from __future__ import annotations

import pandas as pd
import pytest

import data_acquisition.store as store_module
from data_acquisition.store import (
    _TABLE_DDL,
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

    def test_table_not_created_raises_runtime_error(self, tmp_path):
        # Connect to an empty DB (no init_db) → SELECT fails with CatalogException.
        path = tmp_path / "empty.duckdb"
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

"""DuckDB-backed persistent store: schema initialisation, upsert, and cache TTL management.

Provides the central persistence layer for the Buffett screener pipeline. All
upstream data-acquisition modules produce pandas DataFrames; this module writes
them to DuckDB tables and reads them back for downstream consumers
(metrics_engine, screener, valuation_reports).

Database path: ``data/processed/buffett.duckdb`` by default, overridable via
the ``BUFFETT_DB_PATH`` environment variable.

Authoritative spec: docs/ARCHITECTURE.md §10 (DuckDB Table Reference),
§store.py (upsert pattern: delete-then-insert).

Key exports
-----------
init_db(db_path) -> None
    Create all tables (idempotent).
write_dataframe(table_name, df, mode) -> None
    Bulk-write a DataFrame to a table. Modes: ``"replace"`` or ``"append"``.
read_table(table_name, where) -> pd.DataFrame
    Read a full table or a filtered subset.
get_surviving_tickers() -> list[str]
    Return tickers from ``data_quality_log`` where ``drop = FALSE``.
get_connection(db_path) -> duckdb.DuckDBPyConnection
    Lazy-singleton accessor to the DuckDB connection.
close() -> None
    Close and release the cached connection.

Notes
-----
- DuckDB connections are **not** thread-safe. The pipeline runs sequentially;
  concurrent access from multiple threads is unsupported.
- ``register`` / ``unregister`` (DuckDB ≥ 0.9) is used for efficient bulk
  insertion from pandas DataFrames.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default database path.  Override with env var ``BUFFETT_DB_PATH``.
DB_PATH: pathlib.Path = pathlib.Path(
    os.environ.get("BUFFETT_DB_PATH", "data/processed/buffett.duckdb")
)

#: Ordered column definitions for each table.  Used to build explicit
#: ``SELECT`` clauses so DataFrames with extra columns are tolerated.
_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "universe": (
        "ticker", "exchange", "company_name", "market_cap_usd",
        "sector", "industry", "country",
    ),
    "income_statement": (
        "ticker", "fiscal_year", "net_income", "total_revenue",
        "gross_profit", "sga", "operating_income", "interest_expense",
        "eps_diluted", "shares_outstanding_diluted",
    ),
    "balance_sheet": (
        "ticker", "fiscal_year", "long_term_debt",
        "shareholders_equity", "treasury_stock",
    ),
    "cash_flow": (
        "ticker", "fiscal_year", "depreciation_amortization",
        "capital_expenditures", "working_capital_change",
    ),
    "market_data": (
        "ticker", "current_price_usd", "market_cap_usd",
        "enterprise_value_usd", "shares_outstanding", "high_52w",
        "low_52w", "avg_volume_3m", "pe_ratio_trailing",
        "dividend_yield", "as_of_date",
    ),
    "macro_data": ("key", "value", "as_of_date"),
    "data_quality_log": (
        "ticker", "years_available", "missing_critical_fields",
        "substitutions_count", "drop", "drop_reason",
    ),
    "substitution_log": (
        "ticker", "fiscal_year", "buffett_field",
        "api_field_used", "confidence",
    ),
}

#: Primary-key columns for each table (used by upsert delete-then-insert).
_TABLE_PK_COLUMNS: dict[str, tuple[str, ...]] = {
    "universe": ("ticker",),
    "income_statement": ("ticker", "fiscal_year"),
    "balance_sheet": ("ticker", "fiscal_year"),
    "cash_flow": ("ticker", "fiscal_year"),
    "market_data": ("ticker",),
    "macro_data": ("key",),
    "data_quality_log": ("ticker",),
    "substitution_log": ("ticker", "fiscal_year", "buffett_field"),
}

#: DDL statements for every table.  Executed by :func:`init_db`.
#: ``"drop"`` is quoted because it is a SQL keyword.
_TABLE_DDL: dict[str, str] = {
    "universe": """
        CREATE TABLE IF NOT EXISTS universe (
            ticker VARCHAR PRIMARY KEY,
            exchange VARCHAR,
            company_name VARCHAR,
            market_cap_usd DOUBLE,
            sector VARCHAR,
            industry VARCHAR,
            country VARCHAR
        )
    """,
    "income_statement": """
        CREATE TABLE IF NOT EXISTS income_statement (
            ticker VARCHAR,
            fiscal_year INTEGER,
            net_income DOUBLE,
            total_revenue DOUBLE,
            gross_profit DOUBLE,
            sga DOUBLE,
            operating_income DOUBLE,
            interest_expense DOUBLE,
            eps_diluted DOUBLE,
            shares_outstanding_diluted DOUBLE,
            PRIMARY KEY (ticker, fiscal_year)
        )
    """,
    "balance_sheet": """
        CREATE TABLE IF NOT EXISTS balance_sheet (
            ticker VARCHAR,
            fiscal_year INTEGER,
            long_term_debt DOUBLE,
            shareholders_equity DOUBLE,
            treasury_stock DOUBLE,
            PRIMARY KEY (ticker, fiscal_year)
        )
    """,
    "cash_flow": """
        CREATE TABLE IF NOT EXISTS cash_flow (
            ticker VARCHAR,
            fiscal_year INTEGER,
            depreciation_amortization DOUBLE,
            capital_expenditures DOUBLE,
            working_capital_change DOUBLE,
            PRIMARY KEY (ticker, fiscal_year)
        )
    """,
    "market_data": """
        CREATE TABLE IF NOT EXISTS market_data (
            ticker VARCHAR PRIMARY KEY,
            current_price_usd DOUBLE,
            market_cap_usd DOUBLE,
            enterprise_value_usd DOUBLE,
            shares_outstanding DOUBLE,
            high_52w DOUBLE,
            low_52w DOUBLE,
            avg_volume_3m DOUBLE,
            pe_ratio_trailing DOUBLE,
            dividend_yield DOUBLE,
            as_of_date VARCHAR
        )
    """,
    "macro_data": """
        CREATE TABLE IF NOT EXISTS macro_data (
            key VARCHAR PRIMARY KEY,
            value DOUBLE,
            as_of_date VARCHAR
        )
    """,
    "data_quality_log": """
        CREATE TABLE IF NOT EXISTS data_quality_log (
            ticker VARCHAR PRIMARY KEY,
            years_available INTEGER,
            missing_critical_fields VARCHAR,
            substitutions_count INTEGER,
            "drop" BOOLEAN,
            drop_reason VARCHAR
        )
    """,
    "substitution_log": """
        CREATE TABLE IF NOT EXISTS substitution_log (
            ticker VARCHAR,
            fiscal_year INTEGER,
            buffett_field VARCHAR,
            api_field_used VARCHAR,
            confidence VARCHAR,
            PRIMARY KEY (ticker, fiscal_year, buffett_field)
        )
    """,
}


# ---------------------------------------------------------------------------
# Connection singleton
# ---------------------------------------------------------------------------

_connection: duckdb.DuckDBPyConnection | None = None
_active_db_path: pathlib.Path | None = None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_connection(db_path: pathlib.Path | None = None) -> duckdb.DuckDBPyConnection:
    """Return the cached DuckDB connection, creating one if necessary.

    Parameters
    ----------
    db_path:
        Path to the DuckDB file.  ``None`` uses :data:`DB_PATH` (which
        itself defaults to ``data/processed/buffett.duckdb`` unless
        overridden by ``BUFFETT_DB_PATH``).

    Returns
    -------
    duckdb.DuckDBPyConnection
        A persistent connection to the specified database file.
    """
    global _connection, _active_db_path
    target = db_path or DB_PATH
    if _connection is not None and _active_db_path == target:
        return _connection
    close()
    _ensure_parent_dir(target)
    _connection = duckdb.connect(str(target))
    _active_db_path = target
    logger.info("Opened DuckDB connection: %s", target)
    return _connection


def close() -> None:
    """Close the cached DuckDB connection and release the file lock.

    Safe to call multiple times or when no connection is open.
    """
    global _connection, _active_db_path
    if _connection is not None:
        try:
            _connection.close()
        except Exception:
            pass
        _connection = None
        _active_db_path = None


def init_db(db_path: pathlib.Path | None = None) -> None:
    """Create all pipeline tables in the DuckDB database (idempotent).

    Uses ``CREATE TABLE IF NOT EXISTS`` so repeated calls are safe.

    Parameters
    ----------
    db_path:
        Path to the DuckDB file.  ``None`` uses the default :data:`DB_PATH`.
    """
    conn = get_connection(db_path)
    for table_name, ddl in _TABLE_DDL.items():
        conn.execute(ddl)
    logger.info(
        "Initialized %d DuckDB tables at %s.", len(_TABLE_DDL), _active_db_path
    )


def write_dataframe(
    table_name: str,
    df: pd.DataFrame,
    mode: str = "replace",
) -> None:
    """Write a DataFrame to a DuckDB table.

    Uses DuckDB's native ``register`` for efficient bulk insert.

    Parameters
    ----------
    table_name:
        Target table.  Must be one of the tables defined in :data:`_TABLE_DDL`.
    df:
        DataFrame to write.  Must contain all columns defined for the table
        in :data:`_TABLE_COLUMNS`.  Extra columns are ignored.
    mode:
        ``"replace"`` (default): delete all existing rows, then insert.
        ``"append"``: delete-then-insert only for rows whose primary key
        matches the incoming data (upsert pattern from ARCHITECTURE.md).

    Raises
    ------
    ValueError
        If ``table_name`` is not recognised or ``mode`` is invalid.
    """
    _validate_table_name(table_name)
    if df is None or df.empty:
        logger.warning(
            "write_dataframe: empty DataFrame for '%s', skipping.", table_name
        )
        return
    if mode not in ("replace", "append"):
        raise ValueError(f"mode must be 'replace' or 'append', got '{mode}'")
    conn = get_connection()
    col_sql = _col_list_sql(table_name)
    conn.register("_staging_df", df)
    try:
        if mode == "replace":
            conn.execute(f"DELETE FROM {table_name}")
        else:
            _delete_conflicting_rows(conn, table_name)
        conn.execute(
            f"INSERT INTO {table_name} SELECT {col_sql} FROM _staging_df"
        )
    finally:
        try:
            conn.unregister("_staging_df")
        except Exception:
            pass
    logger.info(
        "Wrote %d rows to '%s' (mode=%s).", len(df), table_name, mode
    )


def read_table(
    table_name: str,
    where: str | None = None,
) -> pd.DataFrame:
    """Read a table (or filtered subset) from DuckDB as a DataFrame.

    Parameters
    ----------
    table_name:
        Table to read.  Validated against :data:`_TABLE_DDL` keys.
    where:
        Optional SQL ``WHERE`` clause body (e.g. ``"ticker = 'AAPL'"``).
        Caller is responsible for valid SQL; this is for internal pipeline
        use, not user-facing input.

    Returns
    -------
    pd.DataFrame
        All columns of the table, with zero or more rows.

    Raises
    ------
    ValueError
        If ``table_name`` is not recognised.
    RuntimeError
        If the table does not exist in the database (call :func:`init_db` first).
    """
    _validate_table_name(table_name)
    conn = get_connection()
    sql = f"SELECT * FROM {table_name}"
    if where:
        sql += f" WHERE {where}"
    try:
        result: pd.DataFrame = conn.execute(sql).fetchdf()
    except duckdb.CatalogException as exc:
        raise RuntimeError(
            f"Table '{table_name}' not found in database. "
            "Call init_db() first."
        ) from exc
    logger.debug("read_table: %d rows from '%s'.", len(result), table_name)
    return result


def get_surviving_tickers() -> list[str]:
    """Return tickers from ``data_quality_log`` where ``drop = FALSE``.

    Returns
    -------
    list[str]
        Alphabetically sorted ticker symbols that passed quality checks.
        Empty list if the table does not exist or contains no survivors.
    """
    conn = get_connection()
    try:
        result = conn.execute(
            'SELECT ticker FROM data_quality_log WHERE "drop" = FALSE '
            "ORDER BY ticker"
        ).fetchdf()
    except duckdb.CatalogException:
        logger.warning(
            "get_surviving_tickers: data_quality_log table not found."
        )
        return []
    return result["ticker"].tolist()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_table_name(table_name: str) -> None:
    """Raise ``ValueError`` if *table_name* is not a known table."""
    if table_name not in _TABLE_DDL:
        raise ValueError(
            f"Unknown table '{table_name}'. "
            f"Valid tables: {sorted(_TABLE_DDL.keys())}"
        )


def _ensure_parent_dir(db_path: pathlib.Path) -> None:
    """Create parent directories for the database file if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _col_list_sql(table_name: str) -> str:
    """Return a comma-separated, double-quoted column list for SQL clauses.

    Quoting all identifiers avoids collisions with SQL keywords (e.g.
    the ``drop`` column in ``data_quality_log``).
    """
    return ", ".join(f'"{c}"' for c in _TABLE_COLUMNS[table_name])


def _delete_conflicting_rows(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    """Delete rows from *table_name* whose PK matches the staging DataFrame.

    Called by :func:`write_dataframe` in ``mode="append"`` to implement the
    delete-then-insert upsert pattern (docs/ARCHITECTURE.md §store.py).
    Assumes ``_staging_df`` is already registered on *conn*.
    """
    pk_cols = _TABLE_PK_COLUMNS[table_name]
    pk_sql = ", ".join(f'"{c}"' for c in pk_cols)
    conn.execute(
        f"DELETE FROM {table_name} "
        f"WHERE ({pk_sql}) IN (SELECT {pk_sql} FROM _staging_df)"
    )

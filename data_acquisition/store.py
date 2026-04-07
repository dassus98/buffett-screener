"""
data_acquisition.store
=======================
Persistent local data store backed by DuckDB.

DuckDB is used as an embedded analytical database — no separate server process
required. All raw fetched data, computed metrics, and quality reports are stored
here so that pipeline runs are incremental (re-fetch only stale data) and
analytical queries over the full universe are fast.

Database layout (tables):
    universe_snapshots      — CompanyProfile records with snapshot_date
    income_statements       — IncomeStatement records per ticker + period
    balance_sheets          — BalanceSheet records per ticker + period
    cash_flow_statements    — CashFlowStatement records per ticker + period
    market_data             — MarketData records per ticker + as_of_date
    macro_snapshots         — MacroSnapshot records per as_of_date
    data_quality_reports    — DataQualityReport summaries per ticker + run_date
    metrics                 — Computed MetricsBundle records per ticker + run_date

The database file lives at data/processed/buffett_screener.duckdb by default.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd


DB_DEFAULT_PATH = Path("data/processed/buffett_screener.duckdb")


class DuckDBStore:
    """
    Manages all read/write operations against the local DuckDB database.

    Usage:
        store = DuckDBStore()          # connects to default DB path
        store = DuckDBStore("data/processed/custom.duckdb")
        store.initialise_schema()      # idempotent — safe to call on startup
    """

    def __init__(self, db_path: Path | str = DB_DEFAULT_PATH) -> None:
        """
        Open a connection to the DuckDB database at `db_path`.

        Args:
            db_path: Path to the .duckdb file. Created automatically if absent.

        Logic:
            1. Resolve db_path to an absolute Path
            2. Ensure parent directories exist (mkdir -p)
            3. Open duckdb.connect(str(db_path))
            4. Store the connection as self._conn
        """
        ...

    def initialise_schema(self) -> None:
        """
        Create all required tables if they do not already exist (idempotent).

        Logic:
            Execute CREATE TABLE IF NOT EXISTS statements for each table listed
            in the module docstring. Use appropriate DuckDB column types:
                VARCHAR, INTEGER, DOUBLE, DATE, BOOLEAN, TIMESTAMP
            Add primary key / unique constraints on (ticker, fiscal_year_end)
            for financial statement tables, and (as_of_date) for macro snapshots.
        """
        ...

    def upsert_dataframe(
        self,
        table: str,
        df: pd.DataFrame,
        conflict_columns: list[str],
    ) -> int:
        """
        Insert rows from a DataFrame into `table`, replacing on conflict.

        Args:
            table:            Target DuckDB table name.
            df:               DataFrame whose columns match the table schema.
            conflict_columns: List of column names forming the unique key for
                              conflict detection (e.g. ["ticker", "fiscal_year_end"]).

        Returns:
            Number of rows upserted.

        Logic:
            1. Register df as a DuckDB view: self._conn.register("_upsert_staging", df)
            2. Build and execute an INSERT INTO ... SELECT ... ON CONFLICT DO UPDATE statement
            3. Unregister the staging view
            4. Return row count
        """
        ...

    def query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """
        Execute an arbitrary SQL query and return results as a DataFrame.

        Args:
            sql:    SQL query string. Use ? placeholders for parameters.
            params: Optional list of parameter values for ? placeholders.

        Returns:
            pandas DataFrame with query results. Empty DataFrame if no rows.

        Logic:
            self._conn.execute(sql, params or []).df()
        """
        ...

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def save_universe(self, df: pd.DataFrame, snapshot_date: date) -> None:
        """
        Persist a universe snapshot DataFrame to the universe_snapshots table.

        Args:
            df:            DataFrame with CompanyProfile columns.
            snapshot_date: Date label for this snapshot.

        Logic:
            Add "snapshot_date" column to df, then call upsert_dataframe()
            with conflict_columns=["ticker", "snapshot_date"].
        """
        ...

    def load_latest_universe(self, max_age_days: int = 7) -> Optional[pd.DataFrame]:
        """
        Load the most recent universe snapshot if it is within max_age_days.

        Args:
            max_age_days: Maximum acceptable age of the snapshot in days.

        Returns:
            DataFrame of CompanyProfile rows, or None if no fresh snapshot found.

        Logic:
            1. SELECT MAX(snapshot_date) FROM universe_snapshots
            2. If snapshot is older than max_age_days, return None
            3. Otherwise SELECT * WHERE snapshot_date = max_date
        """
        ...

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------

    def save_income_statements(self, df: pd.DataFrame) -> None:
        """
        Upsert income statement rows into the income_statements table.

        Args:
            df: DataFrame with IncomeStatement fields as columns,
                including ticker and fiscal_year_end.
        """
        ...

    def load_income_statements(
        self, ticker: str, min_fiscal_year: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Load income statement rows for a ticker, optionally from a minimum year.

        Args:
            ticker:         Ticker symbol.
            min_fiscal_year: If provided, only return rows where fiscal_year_end
                             >= min_fiscal_year.

        Returns:
            DataFrame sorted ascending by fiscal_year_end.
        """
        ...

    def save_balance_sheets(self, df: pd.DataFrame) -> None:
        """Upsert balance sheet rows into the balance_sheets table."""
        ...

    def load_balance_sheets(
        self, ticker: str, min_fiscal_year: Optional[date] = None
    ) -> pd.DataFrame:
        """Load balance sheet rows for a ticker, sorted by fiscal_year_end."""
        ...

    def save_cash_flow_statements(self, df: pd.DataFrame) -> None:
        """Upsert cash flow statement rows into the cash_flow_statements table."""
        ...

    def load_cash_flow_statements(
        self, ticker: str, min_fiscal_year: Optional[date] = None
    ) -> pd.DataFrame:
        """Load cash flow statement rows for a ticker, sorted by fiscal_year_end."""
        ...

    # ------------------------------------------------------------------
    # Market & macro data
    # ------------------------------------------------------------------

    def save_market_data(self, df: pd.DataFrame) -> None:
        """Upsert market data rows into the market_data table."""
        ...

    def load_market_data(
        self, ticker: str, as_of_date: Optional[date] = None
    ) -> Optional[pd.Series]:
        """
        Load a single market data row for a ticker as of a given date.

        Args:
            ticker:      Ticker symbol.
            as_of_date:  Target date. Returns the row with the nearest date
                         on or before as_of_date. Defaults to the most recent row.

        Returns:
            pandas Series of market data fields, or None if no row found.
        """
        ...

    def save_macro_snapshot(self, df: pd.DataFrame) -> None:
        """Upsert a macro snapshot row into the macro_snapshots table."""
        ...

    def load_macro_snapshot(
        self, as_of_date: Optional[date] = None
    ) -> Optional[pd.Series]:
        """
        Load the macro snapshot for the most recent date on or before as_of_date.

        Args:
            as_of_date: Target date. Defaults to today.

        Returns:
            pandas Series of macro fields, or None if no snapshot found.
        """
        ...

    # ------------------------------------------------------------------
    # Metrics & reports
    # ------------------------------------------------------------------

    def save_metrics(self, df: pd.DataFrame, run_date: date) -> None:
        """
        Persist computed metrics (MetricsBundle) for all tickers in a run.

        Args:
            df:       DataFrame with one row per ticker and metric columns.
            run_date: Date of the pipeline run to tag this batch.
        """
        ...

    def load_latest_metrics(self) -> pd.DataFrame:
        """
        Load metrics from the most recent pipeline run.

        Returns:
            DataFrame with one row per ticker. Empty DataFrame if no runs found.
        """
        ...

    def save_quality_reports(self, reports: list[dict], run_date: date) -> None:
        """
        Persist data quality report summaries for a pipeline run.

        Args:
            reports:  List of dicts from DataQualityReport.to_dict().
            run_date: Date of the pipeline run.
        """
        ...

    def close(self) -> None:
        """Close the DuckDB connection."""
        ...

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

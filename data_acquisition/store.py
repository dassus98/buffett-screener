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

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

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
        """
        resolved = Path(db_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Connecting to DuckDB at %s", resolved)
        self._conn = duckdb.connect(str(resolved))
        self._path = resolved

    def initialise_schema(self) -> None:
        """
        Create all required tables if they do not already exist (idempotent).
        """
        ddl_statements = [
            # universe_snapshots
            """
            CREATE TABLE IF NOT EXISTS universe_snapshots (
                ticker                  VARCHAR NOT NULL,
                name                    VARCHAR,
                sector                  VARCHAR,
                industry                VARCHAR,
                exchange                VARCHAR,
                country                 VARCHAR,
                currency                VARCHAR,
                fiscal_year_end_month   INTEGER,
                is_adr                  BOOLEAN,
                is_spac                 BOOLEAN,
                description             VARCHAR,
                snapshot_date           DATE NOT NULL,
                PRIMARY KEY (ticker, snapshot_date)
            )
            """,
            # income_statements
            """
            CREATE TABLE IF NOT EXISTS income_statements (
                ticker                      VARCHAR NOT NULL,
                fiscal_year_end             DATE NOT NULL,
                period                      VARCHAR,
                revenue                     DOUBLE,
                cost_of_revenue             DOUBLE,
                gross_profit                DOUBLE,
                operating_income            DOUBLE,
                interest_expense            DOUBLE,
                pretax_income               DOUBLE,
                income_tax                  DOUBLE,
                net_income                  DOUBLE,
                depreciation_amortization   DOUBLE,
                ebitda                      DOUBLE,
                shares_diluted              DOUBLE,
                eps_diluted                 DOUBLE,
                PRIMARY KEY (ticker, fiscal_year_end)
            )
            """,
            # balance_sheets
            """
            CREATE TABLE IF NOT EXISTS balance_sheets (
                ticker                      VARCHAR NOT NULL,
                fiscal_year_end             DATE NOT NULL,
                cash_and_equivalents        DOUBLE,
                short_term_investments      DOUBLE,
                total_current_assets        DOUBLE,
                total_assets                DOUBLE,
                total_current_liabilities   DOUBLE,
                short_term_debt             DOUBLE,
                long_term_debt              DOUBLE,
                total_debt                  DOUBLE,
                total_liabilities           DOUBLE,
                shareholders_equity         DOUBLE,
                retained_earnings           DOUBLE,
                shares_outstanding          DOUBLE,
                PRIMARY KEY (ticker, fiscal_year_end)
            )
            """,
            # cash_flow_statements
            """
            CREATE TABLE IF NOT EXISTS cash_flow_statements (
                ticker                  VARCHAR NOT NULL,
                fiscal_year_end         DATE NOT NULL,
                period                  VARCHAR,
                operating_cash_flow     DOUBLE,
                capital_expenditures    DOUBLE,
                free_cash_flow          DOUBLE,
                dividends_paid          DOUBLE,
                stock_buybacks          DOUBLE,
                net_debt_issuance       DOUBLE,
                PRIMARY KEY (ticker, fiscal_year_end)
            )
            """,
            # market_data
            """
            CREATE TABLE IF NOT EXISTS market_data (
                ticker                  VARCHAR NOT NULL,
                as_of_date              DATE NOT NULL,
                price                   DOUBLE,
                market_cap              DOUBLE,
                enterprise_value        DOUBLE,
                shares_outstanding      DOUBLE,
                beta                    DOUBLE,
                avg_daily_volume_30d    DOUBLE,
                fifty_two_week_high     DOUBLE,
                fifty_two_week_low      DOUBLE,
                PRIMARY KEY (ticker, as_of_date)
            )
            """,
            # macro_snapshots
            """
            CREATE TABLE IF NOT EXISTS macro_snapshots (
                as_of_date              DATE NOT NULL PRIMARY KEY,
                treasury_10y_yield      DOUBLE,
                treasury_2y_yield       DOUBLE,
                cpi_yoy                 DOUBLE,
                fed_funds_rate          DOUBLE,
                real_gdp_growth_yoy     DOUBLE,
                sp500_pe_ratio          DOUBLE
            )
            """,
            # data_quality_reports
            """
            CREATE TABLE IF NOT EXISTS data_quality_reports (
                ticker          VARCHAR NOT NULL,
                run_date        DATE NOT NULL,
                is_critical     BOOLEAN,
                issue_count     INTEGER,
                summary         VARCHAR,
                issues_json     VARCHAR,
                PRIMARY KEY (ticker, run_date)
            )
            """,
            # metrics
            """
            CREATE TABLE IF NOT EXISTS metrics (
                ticker      VARCHAR NOT NULL,
                run_date    DATE NOT NULL,
                metrics_json VARCHAR,
                PRIMARY KEY (ticker, run_date)
            )
            """,
        ]

        for stmt in ddl_statements:
            self._conn.execute(stmt)

        logger.info("DuckDB schema initialised (all tables verified).")

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
                              conflict detection.

        Returns:
            Number of rows upserted.
        """
        if df.empty:
            logger.debug("upsert_dataframe(%s): empty DataFrame, skipping.", table)
            return 0

        staging_name = "_upsert_staging"
        self._conn.register(staging_name, df)
        try:
            # Build WHERE clause to detect conflicts
            where_parts = " AND ".join(
                f"{table}.{col} = {staging_name}.{col}" for col in conflict_columns
            )
            # Delete existing rows that conflict
            delete_sql = f"""
                DELETE FROM {table}
                WHERE EXISTS (
                    SELECT 1 FROM {staging_name}
                    WHERE {where_parts}
                )
            """
            self._conn.execute(delete_sql)

            # Insert all rows from staging
            cols = ", ".join(df.columns)
            insert_sql = f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {staging_name}"
            self._conn.execute(insert_sql)

            row_count = len(df)
            logger.debug("upsert_dataframe(%s): upserted %d rows.", table, row_count)
            return row_count
        finally:
            self._conn.unregister(staging_name)

    def query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """
        Execute an arbitrary SQL query and return results as a DataFrame.

        Args:
            sql:    SQL query string. Use ? placeholders for parameters.
            params: Optional list of parameter values for ? placeholders.

        Returns:
            pandas DataFrame with query results. Empty DataFrame if no rows.
        """
        result = self._conn.execute(sql, params or []).df()
        return result

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def save_universe(self, df: pd.DataFrame, snapshot_date: date) -> None:
        """
        Persist a universe snapshot DataFrame to the universe_snapshots table.

        Args:
            df:            DataFrame with CompanyProfile columns.
            snapshot_date: Date label for this snapshot.
        """
        df = df.copy()
        df["snapshot_date"] = snapshot_date
        count = self.upsert_dataframe(
            "universe_snapshots", df, conflict_columns=["ticker", "snapshot_date"]
        )
        logger.info("Saved %d rows to universe_snapshots (snapshot_date=%s).", count, snapshot_date)

    def load_latest_universe(self, max_age_days: int = 7) -> Optional[pd.DataFrame]:
        """
        Load the most recent universe snapshot if it is within max_age_days.

        Args:
            max_age_days: Maximum acceptable age of the snapshot in days.

        Returns:
            DataFrame of CompanyProfile rows, or None if no fresh snapshot found.
        """
        try:
            result = self._conn.execute(
                "SELECT MAX(snapshot_date) AS max_date FROM universe_snapshots"
            ).df()
        except Exception:
            logger.debug("universe_snapshots table not yet populated.")
            return None

        if result.empty or result["max_date"].iloc[0] is None:
            logger.debug("No universe snapshot found in cache.")
            return None

        max_date = pd.to_datetime(result["max_date"].iloc[0]).date()
        age_days = (date.today() - max_date).days

        if age_days > max_age_days:
            logger.info(
                "Universe cache is %d days old (max_age=%d). Cache miss.",
                age_days, max_age_days,
            )
            return None

        df = self._conn.execute(
            "SELECT * FROM universe_snapshots WHERE snapshot_date = ?", [max_date]
        ).df()
        logger.debug("Loaded universe snapshot from cache (date=%s, rows=%d).", max_date, len(df))
        return df

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------

    def save_income_statements(self, df: pd.DataFrame) -> None:
        """
        Upsert income statement rows into the income_statements table.
        """
        count = self.upsert_dataframe(
            "income_statements", df, conflict_columns=["ticker", "fiscal_year_end"]
        )
        logger.info("Saved %d rows to income_statements.", count)

    def load_income_statements(
        self, ticker: str, min_fiscal_year: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Load income statement rows for a ticker, optionally from a minimum year.

        Returns:
            DataFrame sorted ascending by fiscal_year_end.
        """
        if min_fiscal_year is not None:
            sql = (
                "SELECT * FROM income_statements "
                "WHERE ticker = ? AND fiscal_year_end >= ? "
                "ORDER BY fiscal_year_end ASC"
            )
            params = [ticker, min_fiscal_year]
        else:
            sql = (
                "SELECT * FROM income_statements "
                "WHERE ticker = ? "
                "ORDER BY fiscal_year_end ASC"
            )
            params = [ticker]
        return self.query(sql, params)

    def save_balance_sheets(self, df: pd.DataFrame) -> None:
        """Upsert balance sheet rows into the balance_sheets table."""
        count = self.upsert_dataframe(
            "balance_sheets", df, conflict_columns=["ticker", "fiscal_year_end"]
        )
        logger.info("Saved %d rows to balance_sheets.", count)

    def load_balance_sheets(
        self, ticker: str, min_fiscal_year: Optional[date] = None
    ) -> pd.DataFrame:
        """Load balance sheet rows for a ticker, sorted by fiscal_year_end."""
        if min_fiscal_year is not None:
            sql = (
                "SELECT * FROM balance_sheets "
                "WHERE ticker = ? AND fiscal_year_end >= ? "
                "ORDER BY fiscal_year_end ASC"
            )
            params = [ticker, min_fiscal_year]
        else:
            sql = (
                "SELECT * FROM balance_sheets "
                "WHERE ticker = ? "
                "ORDER BY fiscal_year_end ASC"
            )
            params = [ticker]
        return self.query(sql, params)

    def save_cash_flow_statements(self, df: pd.DataFrame) -> None:
        """Upsert cash flow statement rows into the cash_flow_statements table."""
        count = self.upsert_dataframe(
            "cash_flow_statements", df, conflict_columns=["ticker", "fiscal_year_end"]
        )
        logger.info("Saved %d rows to cash_flow_statements.", count)

    def load_cash_flow_statements(
        self, ticker: str, min_fiscal_year: Optional[date] = None
    ) -> pd.DataFrame:
        """Load cash flow statement rows for a ticker, sorted by fiscal_year_end."""
        if min_fiscal_year is not None:
            sql = (
                "SELECT * FROM cash_flow_statements "
                "WHERE ticker = ? AND fiscal_year_end >= ? "
                "ORDER BY fiscal_year_end ASC"
            )
            params = [ticker, min_fiscal_year]
        else:
            sql = (
                "SELECT * FROM cash_flow_statements "
                "WHERE ticker = ? "
                "ORDER BY fiscal_year_end ASC"
            )
            params = [ticker]
        return self.query(sql, params)

    # ------------------------------------------------------------------
    # Market & macro data
    # ------------------------------------------------------------------

    def save_market_data(self, df: pd.DataFrame) -> None:
        """Upsert market data rows into the market_data table."""
        count = self.upsert_dataframe(
            "market_data", df, conflict_columns=["ticker", "as_of_date"]
        )
        logger.info("Saved %d rows to market_data.", count)

    def load_market_data(
        self, ticker: str, as_of_date: Optional[date] = None
    ) -> Optional[pd.Series]:
        """
        Load a single market data row for a ticker as of a given date.

        Returns:
            pandas Series of market data fields, or None if no row found.
        """
        target_date = as_of_date or date.today()
        sql = (
            "SELECT * FROM market_data "
            "WHERE ticker = ? AND as_of_date <= ? "
            "ORDER BY as_of_date DESC "
            "LIMIT 1"
        )
        df = self.query(sql, [ticker, target_date])
        if df.empty:
            logger.debug("No market_data found for ticker=%s as_of=%s.", ticker, target_date)
            return None
        logger.debug("Loaded market_data for ticker=%s (as_of=%s).", ticker, df["as_of_date"].iloc[0])
        return df.iloc[0]

    def save_macro_snapshot(self, df: pd.DataFrame) -> None:
        """Upsert a macro snapshot row into the macro_snapshots table."""
        count = self.upsert_dataframe(
            "macro_snapshots", df, conflict_columns=["as_of_date"]
        )
        logger.info("Saved %d rows to macro_snapshots.", count)

    def load_macro_snapshot(
        self, as_of_date: Optional[date] = None
    ) -> Optional[pd.Series]:
        """
        Load the macro snapshot for the most recent date on or before as_of_date.

        Returns:
            pandas Series of macro fields, or None if no snapshot found.
        """
        target_date = as_of_date or date.today()
        sql = (
            "SELECT * FROM macro_snapshots "
            "WHERE as_of_date <= ? "
            "ORDER BY as_of_date DESC "
            "LIMIT 1"
        )
        df = self.query(sql, [target_date])
        if df.empty:
            logger.debug("No macro_snapshot found for as_of_date=%s.", target_date)
            return None
        logger.debug("Loaded macro_snapshot (as_of=%s).", df["as_of_date"].iloc[0])
        return df.iloc[0]

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
        import json

        records = []
        for _, row in df.iterrows():
            ticker = row.get("ticker", "")
            metrics_dict = row.drop(labels=["ticker"], errors="ignore").to_dict()
            records.append(
                {
                    "ticker": ticker,
                    "run_date": run_date,
                    "metrics_json": json.dumps(metrics_dict, default=str),
                }
            )

        if not records:
            logger.debug("save_metrics: no records to save.")
            return

        metrics_df = pd.DataFrame(records)
        count = self.upsert_dataframe(
            "metrics", metrics_df, conflict_columns=["ticker", "run_date"]
        )
        logger.info("Saved %d metric rows for run_date=%s.", count, run_date)

    def load_latest_metrics(self) -> pd.DataFrame:
        """
        Load metrics from the most recent pipeline run.

        Returns:
            DataFrame with one row per ticker. Empty DataFrame if no runs found.
        """
        import json

        try:
            run_date_df = self.query("SELECT MAX(run_date) AS max_date FROM metrics")
        except Exception:
            logger.debug("metrics table not yet populated.")
            return pd.DataFrame()

        if run_date_df.empty or run_date_df["max_date"].iloc[0] is None:
            logger.debug("No metrics runs found in cache.")
            return pd.DataFrame()

        max_date = run_date_df["max_date"].iloc[0]
        raw_df = self.query(
            "SELECT ticker, metrics_json FROM metrics WHERE run_date = ?", [max_date]
        )

        rows = []
        for _, row in raw_df.iterrows():
            record: dict = {"ticker": row["ticker"]}
            if row["metrics_json"]:
                record.update(json.loads(row["metrics_json"]))
            rows.append(record)

        return pd.DataFrame(rows)

    def save_quality_reports(self, reports: list[dict], run_date: date) -> None:
        """
        Persist data quality report summaries for a pipeline run.

        Args:
            reports:  List of dicts from DataQualityReport.to_dict().
            run_date: Date of the pipeline run.
        """
        import json

        if not reports:
            logger.debug("save_quality_reports: no reports to save.")
            return

        records = []
        for r in reports:
            records.append(
                {
                    "ticker": r.get("ticker", ""),
                    "run_date": run_date,
                    "is_critical": r.get("is_critical", False),
                    "issue_count": r.get("issue_count", 0),
                    "summary": r.get("summary", ""),
                    "issues_json": json.dumps(r.get("issues", []), default=str),
                }
            )

        df = pd.DataFrame(records)
        count = self.upsert_dataframe(
            "data_quality_reports", df, conflict_columns=["ticker", "run_date"]
        )
        logger.info("Saved %d quality reports for run_date=%s.", count, run_date)

    def close(self) -> None:
        """Close the DuckDB connection."""
        if hasattr(self, "_conn") and self._conn is not None:
            self._conn.close()
            logger.debug("DuckDB connection closed (%s).", self._path)

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

"""Full end-to-end integration test with 30 mock tickers.

Populates a temporary DuckDB with synthetic financial data for 30 tickers
across four quality tiers, then runs the real pipeline from metrics_engine
through report generation, verifying every stage's output.

Ticker tiers:
    - 10 high-quality stocks (HQ01–HQ10): strong financials, should be shortlisted
    - 8 medium-quality stocks (MQ01–MQ08): pass Tier 1 but mixed quality
    - 7 low-quality stocks (LQ01–LQ07): fail one or more Tier 1 hard filters
    - 5 financial-sector stocks (FN01–FN05): excluded by sector/industry rules
"""

from __future__ import annotations

import json
import math
import pathlib
import shutil
from typing import Any

import pandas as pd
import pytest

from data_acquisition.store import close, get_connection, init_db, write_dataframe


def _read_metrics_summary() -> pd.DataFrame:
    """Read buffett_metrics_summary directly from DuckDB.

    This table is created by the metrics engine (Module 2) using
    ``_write_table_full_replace``, so it is NOT in store's
    ``_TABLE_DDL`` and cannot be read via ``read_table()``.

    Returns
    -------
    pd.DataFrame
    """
    conn = get_connection()
    try:
        return conn.execute("SELECT * FROM buffett_metrics_summary").fetchdf()
    except Exception:
        return pd.DataFrame()


def _enrich_metrics_with_quality(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Merge ``years_available`` from data_quality_log into metrics_df.

    The hard filters require ``years_available`` for the data_sufficiency
    check, but the metrics engine does not produce this column — it comes
    from Module 1's ``data_quality_log``.

    Parameters
    ----------
    metrics_df:
        buffett_metrics_summary DataFrame.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with ``years_available`` column.
    """
    from data_acquisition.store import read_table

    quality_df = read_table("data_quality_log")
    if quality_df.empty or "years_available" not in quality_df.columns:
        metrics_df["years_available"] = 10  # Fallback
        return metrics_df

    return metrics_df.merge(
        quality_df[["ticker", "years_available"]],
        on="ticker",
        how="left",
    )


# ---------------------------------------------------------------------------
# Mock data factory helpers
# ---------------------------------------------------------------------------


def _make_income_rows(
    ticker: str,
    years: range,
    *,
    base_revenue: float = 10_000_000_000,
    revenue_growth: float = 0.08,
    gross_margin: float = 0.55,
    sga_pct_of_gp: float = 0.35,
    net_margin: float = 0.15,
    interest_pct: float = 0.02,
    base_eps: float = 3.0,
    eps_growth: float = 0.10,
    shares: float = 1_000_000_000,
) -> list[dict[str, Any]]:
    """Generate 10 years of income statement rows.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    years:
        Range of fiscal years.
    base_revenue:
        Starting revenue (grows by *revenue_growth* per year).
    gross_margin:
        Gross profit / revenue ratio.
    sga_pct_of_gp:
        SGA / gross profit ratio.
    net_margin:
        Net income / revenue ratio.
    interest_pct:
        Interest expense / revenue ratio.
    base_eps:
        Starting EPS.
    eps_growth:
        Annual EPS growth rate.
    shares:
        Diluted shares outstanding.

    Returns
    -------
    list[dict]
    """
    rows = []
    for i, yr in enumerate(years):
        revenue = base_revenue * (1 + revenue_growth) ** i
        gross_profit = revenue * gross_margin
        sga = gross_profit * sga_pct_of_gp
        operating_income = gross_profit - sga
        interest_expense = revenue * interest_pct
        net_income = revenue * net_margin
        eps = base_eps * (1 + eps_growth) ** i
        rows.append({
            "ticker": ticker,
            "fiscal_year": yr,
            "net_income": net_income,
            "total_revenue": revenue,
            "gross_profit": gross_profit,
            "sga": sga,
            "operating_income": operating_income,
            "interest_expense": interest_expense,
            "eps_diluted": eps,
            "shares_outstanding_diluted": shares,
        })
    return rows


def _make_balance_rows(
    ticker: str,
    years: range,
    *,
    base_equity: float = 20_000_000_000,
    equity_growth: float = 0.08,
    debt_ratio: float = 0.3,
) -> list[dict[str, Any]]:
    """Generate balance sheet rows.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    years:
        Range of fiscal years.
    base_equity:
        Starting shareholders' equity.
    equity_growth:
        Annual equity growth.
    debt_ratio:
        Long-term debt / equity ratio.

    Returns
    -------
    list[dict]
    """
    rows = []
    for i, yr in enumerate(years):
        equity = base_equity * (1 + equity_growth) ** i
        debt = equity * debt_ratio
        rows.append({
            "ticker": ticker,
            "fiscal_year": yr,
            "long_term_debt": debt,
            "shareholders_equity": equity,
            "treasury_stock": 0.0,
        })
    return rows


def _make_cashflow_rows(
    ticker: str,
    years: range,
    *,
    base_da: float = 500_000_000,
    da_growth: float = 0.05,
    capex_ratio: float = 1.2,
    wc_change: float = 50_000_000,
) -> list[dict[str, Any]]:
    """Generate cash flow rows.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    years:
        Fiscal year range.
    base_da:
        Starting depreciation and amortisation.
    da_growth:
        Annual D&A growth.
    capex_ratio:
        CapEx as multiple of D&A.
    wc_change:
        Annual working capital change.

    Returns
    -------
    list[dict]
    """
    rows = []
    for i, yr in enumerate(years):
        da = base_da * (1 + da_growth) ** i
        capex = da * capex_ratio
        rows.append({
            "ticker": ticker,
            "fiscal_year": yr,
            "depreciation_amortization": da,
            "capital_expenditures": capex,
            "working_capital_change": wc_change,
        })
    return rows


def _make_universe_row(
    ticker: str,
    *,
    exchange: str = "NYSE",
    sector: str = "Technology",
    industry: str = "Software",
    country: str = "US",
) -> dict[str, Any]:
    """Create a single universe row.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    exchange:
        Exchange code.
    sector:
        GICS sector.
    industry:
        GICS industry.
    country:
        Country code.

    Returns
    -------
    dict
    """
    return {
        "ticker": ticker,
        "exchange": exchange,
        "company_name": f"{ticker} Corp",
        "market_cap_usd": 50_000_000_000,
        "sector": sector,
        "industry": industry,
        "country": country,
    }


def _make_market_row(
    ticker: str,
    *,
    price: float = 150.0,
    pe: float = 20.0,
    dividend_yield: float = 0.02,
) -> dict[str, Any]:
    """Create a single market data row.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    price:
        Current share price in USD.
    pe:
        Trailing P/E ratio.
    dividend_yield:
        Dividend yield (decimal).

    Returns
    -------
    dict
    """
    return {
        "ticker": ticker,
        "current_price_usd": price,
        "market_cap_usd": 50_000_000_000,
        "enterprise_value_usd": 55_000_000_000,
        "shares_outstanding": 1_000_000_000,
        "high_52w": price * 1.2,
        "low_52w": price * 0.8,
        "avg_volume_3m": 5_000_000,
        "pe_ratio_trailing": pe,
        "dividend_yield": dividend_yield,
        "as_of_date": "2025-12-31",
    }


def _make_quality_row(
    ticker: str,
    *,
    years_available: int = 10,
    subs_count: int = 0,
    drop: bool = False,
    drop_reason: str = "",
) -> dict[str, Any]:
    """Create a data quality log row.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    years_available:
        Years of data available.
    subs_count:
        Number of line-item substitutions.
    drop:
        Whether this ticker should be dropped.
    drop_reason:
        Reason for dropping.

    Returns
    -------
    dict
    """
    return {
        "ticker": ticker,
        "years_available": years_available,
        "missing_critical_fields": "",
        "substitutions_count": subs_count,
        "drop": drop,
        "drop_reason": drop_reason,
    }


# ---------------------------------------------------------------------------
# Build the full 30-ticker mock dataset
# ---------------------------------------------------------------------------

_YEARS = range(2015, 2025)  # 10 years


def _build_all_mock_data() -> dict[str, pd.DataFrame]:
    """Build DataFrames for all 8 DuckDB tables with 30 synthetic tickers.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by table name.
    """
    universe_rows: list[dict] = []
    income_rows: list[dict] = []
    balance_rows: list[dict] = []
    cashflow_rows: list[dict] = []
    market_rows: list[dict] = []
    quality_rows: list[dict] = []

    # ---- HIGH QUALITY (HQ01–HQ10): Strong financials ----
    for i in range(1, 11):
        t = f"HQ{i:02d}"
        sectors = ["Technology", "Healthcare", "Consumer Staples",
                    "Industrials", "Technology", "Healthcare",
                    "Consumer Staples", "Technology", "Industrials",
                    "Consumer Staples"]
        exchanges = ["NYSE", "NASDAQ", "NYSE", "NYSE", "NASDAQ",
                     "NYSE", "TSX", "NASDAQ", "NYSE", "TSX"]
        universe_rows.append(_make_universe_row(
            t,
            exchange=exchanges[i - 1],
            sector=sectors[i - 1],
            industry=f"{sectors[i - 1]} Products",
        ))
        # High ROE (>15%), good growth, low debt
        # ROE = NI / equity. With net_margin=0.25, revenue=10e9:
        #   NI = 2.5e9, equity=8e9 → ROE = 0.31 (well above 0.15)
        income_rows.extend(_make_income_rows(
            t, _YEARS,
            base_revenue=8e9 + i * 1e9,
            gross_margin=0.55 + i * 0.02,
            net_margin=0.25,
            base_eps=2.5 + i * 0.5,
            eps_growth=0.12,
            sga_pct_of_gp=0.30,
        ))
        balance_rows.extend(_make_balance_rows(
            t, _YEARS,
            base_equity=5e9 + i * 1e9,
            equity_growth=0.06,
            debt_ratio=0.20 + i * 0.01,
        ))
        cashflow_rows.extend(_make_cashflow_rows(
            t, _YEARS,
            base_da=400e6,
            capex_ratio=1.1,
        ))
        market_rows.append(_make_market_row(
            t,
            price=80.0 + i * 10,
            pe=18.0 + i * 0.5,
            dividend_yield=0.02 if exchanges[i - 1] != "TSX" else 0.03,
        ))
        quality_rows.append(_make_quality_row(t, years_available=10))

    # ---- MEDIUM QUALITY (MQ01–MQ08): Pass Tier 1 but not all shortlisted ----
    for i in range(1, 9):
        t = f"MQ{i:02d}"
        universe_rows.append(_make_universe_row(
            t,
            exchange="NYSE",
            sector="Industrials",
            industry="Industrial Machinery",
        ))
        # Moderate ROE (~0.17) and growth; must pass Tier 1 hard filters
        # ROE = base_revenue * net_margin / base_equity = 5e9 * 0.12 / 3.5e9 = 0.17
        income_rows.extend(_make_income_rows(
            t, _YEARS,
            base_revenue=5e9,
            gross_margin=0.40,
            net_margin=0.12,
            base_eps=2.0,
            eps_growth=0.05,
            sga_pct_of_gp=0.45,
            interest_pct=0.04,
        ))
        balance_rows.extend(_make_balance_rows(
            t, _YEARS,
            base_equity=3.5e9,
            debt_ratio=0.40,
        ))
        cashflow_rows.extend(_make_cashflow_rows(
            t, _YEARS,
            base_da=300e6,
            capex_ratio=1.4,
        ))
        market_rows.append(_make_market_row(
            t,
            price=50.0 + i * 5,
            pe=14.0,
            dividend_yield=0.015,
        ))
        quality_rows.append(_make_quality_row(t, years_available=10))

    # ---- LOW QUALITY (LQ01–LQ07): Fail various Tier 1 filters ----
    for i in range(1, 8):
        t = f"LQ{i:02d}"
        universe_rows.append(_make_universe_row(
            t,
            exchange="NASDAQ",
            sector="Consumer Discretionary",
            industry="Retail",
        ))

        # Each LQ ticker fails a different filter
        if i <= 2:
            # Fail earnings consistency: negative net income in most years
            income_rows.extend(_make_income_rows(
                t, _YEARS,
                base_revenue=3e9,
                gross_margin=0.25,
                net_margin=-0.05,  # Negative!
                base_eps=-0.50,
                eps_growth=0.0,
            ))
        elif i <= 4:
            # Fail ROE: very low equity returns
            income_rows.extend(_make_income_rows(
                t, _YEARS,
                base_revenue=3e9,
                gross_margin=0.30,
                net_margin=0.02,  # Very low
                base_eps=0.30,
                eps_growth=0.01,
            ))
        elif i <= 5:
            # Fail EPS growth: negative CAGR
            income_rows.extend(_make_income_rows(
                t, _YEARS,
                base_revenue=3e9,
                gross_margin=0.35,
                net_margin=0.08,
                base_eps=5.0,
                eps_growth=-0.05,  # Declining EPS!
            ))
        else:
            # Fail debt sustainability: extremely high debt
            income_rows.extend(_make_income_rows(
                t, _YEARS,
                base_revenue=3e9,
                gross_margin=0.35,
                net_margin=0.08,
                base_eps=1.5,
                eps_growth=0.03,
            ))

        # All LQ get balance sheets; LQ06-07 get very high debt
        if i >= 6:
            balance_rows.extend(_make_balance_rows(
                t, _YEARS,
                base_equity=2e9,
                debt_ratio=5.0,  # 500% D/E → enormous debt payoff
            ))
        else:
            balance_rows.extend(_make_balance_rows(
                t, _YEARS,
                base_equity=5e9,
                debt_ratio=0.4,
            ))

        cashflow_rows.extend(_make_cashflow_rows(
            t, _YEARS,
            base_da=200e6,
            capex_ratio=1.5,
        ))
        market_rows.append(_make_market_row(
            t,
            price=20.0 + i * 3,
            pe=12.0,
        ))
        quality_rows.append(_make_quality_row(t, years_available=10))

    # ---- FINANCIAL SECTOR (FN01–FN05): Should be excluded ----
    for i in range(1, 6):
        t = f"FN{i:02d}"
        industries = ["Banks", "Insurance Carriers", "REIT Services",
                       "Mortgage Finance Corp", "Savings Institutions"]
        universe_rows.append(_make_universe_row(
            t,
            exchange="NYSE",
            sector="Financial Services",
            industry=industries[i - 1],
        ))
        income_rows.extend(_make_income_rows(
            t, _YEARS,
            base_revenue=5e9,
            gross_margin=0.50,
            net_margin=0.15,
            base_eps=3.0,
            eps_growth=0.08,
        ))
        balance_rows.extend(_make_balance_rows(
            t, _YEARS,
            base_equity=10e9,
            debt_ratio=0.3,
        ))
        cashflow_rows.extend(_make_cashflow_rows(t, _YEARS))
        market_rows.append(_make_market_row(t, price=80.0, pe=15.0))
        quality_rows.append(_make_quality_row(t, years_available=10))

    # Macro data
    macro_rows = [
        {"key": "us_treasury_10yr", "value": 0.042, "as_of_date": "2025-12-31"},
        {"key": "usd_cad_rate", "value": 1.36, "as_of_date": "2025-12-31"},
    ]

    return {
        "universe": pd.DataFrame(universe_rows),
        "income_statement": pd.DataFrame(income_rows),
        "balance_sheet": pd.DataFrame(balance_rows),
        "cash_flow": pd.DataFrame(cashflow_rows),
        "market_data": pd.DataFrame(market_rows),
        "data_quality_log": pd.DataFrame(quality_rows),
        "macro_data": pd.DataFrame(macro_rows),
        "substitution_log": pd.DataFrame(
            columns=["ticker", "fiscal_year", "buffett_field",
                     "api_field_used", "confidence"],
        ),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def integration_db(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create and populate a temporary DuckDB with all 30 tickers.

    Yields
    ------
    pathlib.Path
        Path to the temporary DuckDB file.
    """
    import os

    db_path = tmp_path / "integration_test.duckdb"
    os.environ["BUFFETT_DB_PATH"] = str(db_path)

    # Force store module to pick up the new path
    import data_acquisition.store as store_mod

    store_mod.DB_PATH = db_path
    close()  # Clear any cached connection

    init_db(db_path)

    data = _build_all_mock_data()
    for table_name, df in data.items():
        if not df.empty:
            write_dataframe(table_name, df)

    yield db_path

    close()
    # Restore original DB_PATH
    store_mod.DB_PATH = pathlib.Path(
        os.environ.pop("BUFFETT_DB_PATH", "data/processed/buffett.duckdb")
    )


@pytest.fixture()
def report_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary report output directory.

    Yields
    ------
    pathlib.Path
    """
    rd = tmp_path / "reports"
    rd.mkdir()
    return rd


# ===========================================================================
# Part A: Full Pipeline Integration Tests
# ===========================================================================


class TestFullPipelineIntegration:
    """Run the full pipeline from metrics_engine through report generation."""

    def test_metrics_engine_processes_all_survivors(
        self, integration_db: pathlib.Path,
    ) -> None:
        """Metrics engine scores all 30 non-dropped tickers."""
        from metrics_engine import run_metrics_engine

        composite_df = run_metrics_engine()

        # All 30 tickers are non-dropped in data_quality_log
        assert len(composite_df) == 30
        assert "composite_score" in composite_df.columns
        assert "ticker" in composite_df.columns

    def test_exclusions_remove_financial_stocks(
        self, integration_db: pathlib.Path,
    ) -> None:
        """apply_exclusions removes all 5 FN* tickers."""
        from data_acquisition.store import read_table
        from screener.exclusions import apply_exclusions

        universe_df = read_table("universe")
        filtered_df, exclusion_log = apply_exclusions(universe_df)

        excluded_tickers = set(exclusion_log["ticker"].tolist())
        fn_tickers = {f"FN{i:02d}" for i in range(1, 6)}

        # All 5 financial tickers should be excluded
        assert fn_tickers.issubset(excluded_tickers)
        # No HQ/MQ/LQ tickers should be excluded
        for t in filtered_df["ticker"].tolist():
            assert not t.startswith("FN")

    def test_no_ticker_in_both_excluded_and_filtered(
        self, integration_db: pathlib.Path,
    ) -> None:
        """No ticker appears in both excluded and surviving sets."""
        from data_acquisition.store import read_table
        from screener.exclusions import apply_exclusions

        universe_df = read_table("universe")
        filtered_df, exclusion_log = apply_exclusions(universe_df)

        filtered_set = set(filtered_df["ticker"].tolist())
        excluded_set = set(exclusion_log["ticker"].tolist())

        assert filtered_set.isdisjoint(excluded_set)

    def test_hard_filters_remove_low_quality(
        self, integration_db: pathlib.Path,
    ) -> None:
        """Tier 1 hard filters remove LQ tickers that fail financial tests."""
        from data_acquisition.store import read_table
        from metrics_engine import run_metrics_engine
        from screener.exclusions import apply_exclusions
        from screener.hard_filters import apply_hard_filters

        # Run metrics engine first to populate buffett_metrics_summary
        run_metrics_engine()

        # Apply exclusions
        universe_df = read_table("universe")
        filtered_df, _ = apply_exclusions(universe_df)

        # Read metrics summary and filter to post-exclusion tickers
        metrics_df = _enrich_metrics_with_quality(_read_metrics_summary())
        surviving_tickers = set(filtered_df["ticker"].tolist())
        metrics_df = metrics_df[
            metrics_df["ticker"].isin(surviving_tickers)
        ].reset_index(drop=True)

        # Apply hard filters
        survivors_df, filter_log = apply_hard_filters(metrics_df)

        # LQ tickers with negative earnings (LQ01, LQ02) should fail
        survivor_set = set(survivors_df["ticker"].tolist())
        for t in ["LQ01", "LQ02"]:
            assert t not in survivor_set, (
                f"{t} should have been eliminated by hard filters"
            )

    def test_full_screening_pipeline(
        self, integration_db: pathlib.Path,
    ) -> None:
        """Full screening pipeline produces a shortlist of top-10."""
        from data_acquisition.store import read_table
        from metrics_engine import run_metrics_engine
        from screener.composite_ranker import generate_shortlist
        from screener.exclusions import apply_exclusions
        from screener.hard_filters import apply_hard_filters
        from screener.soft_filters import apply_soft_scores

        # Stage 2: Metrics
        composite_df = run_metrics_engine()

        # Stage 3a: Exclusions
        universe_df = read_table("universe")
        filtered_df, _ = apply_exclusions(universe_df)

        # Stage 3b: Hard filters
        metrics_df = _enrich_metrics_with_quality(_read_metrics_summary())
        surviving_tickers = set(filtered_df["ticker"].tolist())
        metrics_df = metrics_df[
            metrics_df["ticker"].isin(surviving_tickers)
        ].reset_index(drop=True)
        survivors_df, filter_log = apply_hard_filters(metrics_df)

        # Stage 3c: Soft scores
        ranked_df = apply_soft_scores(survivors_df, composite_df)

        # Stage 3d: Shortlist
        shortlist_df = generate_shortlist(ranked_df, top_n=10)

        # Verify shortlist
        assert len(shortlist_df) == 10
        assert "rank" in shortlist_df.columns
        assert "composite_score" in shortlist_df.columns

        # No excluded tickers in shortlist
        shortlisted = set(shortlist_df["ticker"].tolist())
        fn_tickers = {f"FN{i:02d}" for i in range(1, 6)}
        assert shortlisted.isdisjoint(fn_tickers)

    def test_report_generation(
        self,
        integration_db: pathlib.Path,
        report_dir: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """generate_all_reports produces one report per shortlisted ticker."""
        from data_acquisition.store import read_table
        from metrics_engine import run_metrics_engine
        from screener.composite_ranker import (
            generate_screener_summary,
            generate_shortlist,
        )
        from screener.exclusions import apply_exclusions
        from screener.hard_filters import apply_hard_filters
        from screener.soft_filters import apply_soft_scores
        from valuation_reports.report_generator import generate_all_reports

        # Redirect report output
        monkeypatch.setattr(
            "valuation_reports.report_generator._PROJECT_ROOT",
            report_dir.parent,
        )
        from screener.filter_config_loader import _config_cache
        # Don't need to modify — generate_all_reports reads output.report_dir

        # We need the report dir to match what generate_all_reports creates
        actual_report_dir = report_dir.parent / "data" / "reports"
        actual_report_dir.mkdir(parents=True, exist_ok=True)

        # Run full pipeline
        composite_df = run_metrics_engine()
        universe_df = read_table("universe")
        filtered_df, exclusion_log = apply_exclusions(universe_df)
        metrics_df = _enrich_metrics_with_quality(_read_metrics_summary())
        surviving_tickers = set(filtered_df["ticker"].tolist())
        metrics_df = metrics_df[
            metrics_df["ticker"].isin(surviving_tickers)
        ].reset_index(drop=True)
        survivors_df, filter_log = apply_hard_filters(metrics_df)
        ranked_df = apply_soft_scores(survivors_df, composite_df)
        shortlist_df = generate_shortlist(ranked_df, top_n=10)

        screener_summary = generate_screener_summary(
            full_ranked_df=ranked_df,
            shortlist_df=shortlist_df,
            filter_log_df=filter_log,
            total_universe=30,
        )

        # Enrich screener_summary with keys that the summary template
        # expects.  generate_screener_summary() uses its own key naming
        # convention, so we map them and add macro data that only the
        # pipeline runner would otherwise inject.
        screener_summary["universe_size"] = screener_summary.get(
            "total_universe", 30,
        )
        screener_summary["passed_hard_filters"] = screener_summary.get(
            "after_tier1", 0,
        )
        screener_summary["macro"] = {
            "us_treasury_10yr": 0.0425,
            "usd_cad_rate": 1.36,
        }

        # Generate reports
        paths = generate_all_reports(shortlist_df, screener_summary)

        # Verify: one analysis report per shortlisted ticker + summary
        analysis_reports = [p for p in paths if p.name.endswith("_analysis.md")]
        summary_reports = [p for p in paths if p.name == "summary.md"]

        assert len(analysis_reports) == 10
        assert len(summary_reports) == 1

        # Verify every shortlisted ticker has a corresponding report
        shortlisted_tickers = set(shortlist_df["ticker"].tolist())
        report_tickers = {
            p.stem.replace("_analysis", "") for p in analysis_reports
        }
        assert shortlisted_tickers == report_tickers

    def test_reports_contain_required_sections(
        self,
        integration_db: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """Each generated report contains all required sections."""
        from data_acquisition.store import read_table
        from metrics_engine import run_metrics_engine
        from screener.composite_ranker import generate_shortlist
        from screener.exclusions import apply_exclusions
        from screener.hard_filters import apply_hard_filters
        from screener.soft_filters import apply_soft_scores
        from valuation_reports.report_generator import generate_all_reports

        actual_report_dir = tmp_path / "data" / "reports"
        actual_report_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "valuation_reports.report_generator._PROJECT_ROOT",
            tmp_path,
        )

        composite_df = run_metrics_engine()
        universe_df = read_table("universe")
        filtered_df, _ = apply_exclusions(universe_df)
        metrics_df = _enrich_metrics_with_quality(_read_metrics_summary())
        surviving_tickers = set(filtered_df["ticker"].tolist())
        metrics_df = metrics_df[
            metrics_df["ticker"].isin(surviving_tickers)
        ].reset_index(drop=True)
        survivors_df, _ = apply_hard_filters(metrics_df)
        ranked_df = apply_soft_scores(survivors_df, composite_df)
        shortlist_df = generate_shortlist(ranked_df, top_n=5)

        paths = generate_all_reports(shortlist_df)
        analysis_reports = [p for p in paths if p.name.endswith("_analysis.md")]

        for report_path in analysis_reports:
            content = report_path.read_text(encoding="utf-8")
            ticker = report_path.stem.replace("_analysis", "")

            # Required sections
            assert "Executive Summary" in content, (
                f"{ticker}: missing Executive Summary"
            )
            assert "Financial Statement Analysis" in content or \
                   "Income Statement" in content, (
                f"{ticker}: missing Financial Analysis"
            )
            assert "Valuation" in content, (
                f"{ticker}: missing Valuation section"
            )
            assert "Assumption Log" in content, (
                f"{ticker}: missing Assumption Log"
            )
            assert "Bear Case" in content or "Devil" in content, (
                f"{ticker}: missing Bear Case"
            )
            assert "Investment Strategy" in content, (
                f"{ticker}: missing Investment Strategy"
            )

    def test_executive_summary_no_nan(
        self,
        integration_db: pathlib.Path,
    ) -> None:
        """Executive Summary key fields are not NaN."""
        from valuation_reports.report_generator import build_report_context

        # Build context for a high-quality ticker
        ctx = build_report_context("HQ01")

        # Key executive summary fields should not be NaN
        assert ctx["recommendation"] in ("BUY", "HOLD", "PASS")
        assert ctx["confidence_level"] in ("High", "Moderate", "Low")
        assert ctx["account_recommendation"] != ""
        assert ctx["time_horizon_years"] > 0

        # IV should be a real number (not NaN) for HQ tickers
        iv = ctx.get("iv_weighted", 0.0)
        assert not math.isnan(iv), "Weighted IV should not be NaN for HQ01"
        assert iv > 0, "Weighted IV should be positive for HQ01"

    def test_account_recommendation_exchange_logic(
        self,
        integration_db: pathlib.Path,
    ) -> None:
        """Account recommendations follow exchange-based logic."""
        from valuation_reports.report_generator import build_report_context

        # HQ01 is NYSE with dividend_yield=0.02 → RRSP
        ctx_nyse = build_report_context("HQ01")
        assert ctx_nyse["account_recommendation"] == "RRSP"

        # HQ07 is TSX → TFSA
        ctx_tsx = build_report_context("HQ07")
        assert ctx_tsx["account_recommendation"] == "TFSA"

    def test_composite_score_weights_sum_to_one(self) -> None:
        """Config soft_scores weights must sum to 1.0."""
        from screener.filter_config_loader import get_threshold

        ss = get_threshold("soft_scores")
        total_weight = sum(
            v.get("weight", 0)
            for v in ss.values()
            if isinstance(v, dict) and "weight" in v
        )
        assert abs(total_weight - 1.0) < 1e-9, (
            f"Soft score weights sum to {total_weight}, expected 1.0"
        )

    def test_screener_summary_statistics(
        self, integration_db: pathlib.Path,
    ) -> None:
        """Screener summary contains valid pipeline statistics."""
        from data_acquisition.store import read_table
        from metrics_engine import run_metrics_engine
        from screener.composite_ranker import (
            generate_screener_summary,
            generate_shortlist,
        )
        from screener.exclusions import apply_exclusions
        from screener.hard_filters import apply_hard_filters
        from screener.soft_filters import apply_soft_scores

        composite_df = run_metrics_engine()
        universe_df = read_table("universe")
        filtered_df, exclusion_log = apply_exclusions(universe_df)
        metrics_df = _enrich_metrics_with_quality(_read_metrics_summary())
        surviving_tickers = set(filtered_df["ticker"].tolist())
        metrics_df = metrics_df[
            metrics_df["ticker"].isin(surviving_tickers)
        ].reset_index(drop=True)
        survivors_df, filter_log = apply_hard_filters(metrics_df)
        ranked_df = apply_soft_scores(survivors_df, composite_df)
        shortlist_df = generate_shortlist(ranked_df, top_n=10)

        summary = generate_screener_summary(
            full_ranked_df=ranked_df,
            shortlist_df=shortlist_df,
            filter_log_df=filter_log,
            total_universe=30,
        )

        assert summary["total_universe"] == 30
        assert summary["shortlisted"] == 10
        # After exclusions should be 25 (30 minus 5 financials)
        assert summary["after_exclusions"] == 25
        # After Tier 1 should be less than 25 (LQ tickers fail)
        assert summary["after_tier1"] <= 25
        assert summary["after_tier1"] >= 10  # At least our 10 HQ pass

    def test_run_log_json(
        self,
        integration_db: pathlib.Path,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pipeline runner writes a valid run_log.json."""
        import argparse

        from output.pipeline_runner import _write_run_log

        monkeypatch.setattr(
            "output.pipeline_runner._PROJECT_ROOT",
            tmp_path,
        )
        monkeypatch.setattr(
            "output.pipeline_runner.get_threshold",
            lambda key: str(tmp_path / "data" / "reports")
            if key == "output.report_dir" else "INFO",
        )

        args = argparse.Namespace(
            mode="reports",
            top=10,
            exchange="ALL",
            skip_acquisition=True,
            skip_metrics=False,
            no_cache=False,
            no_moat=False,
            verbose=False,
        )

        log_path = _write_run_log(
            args=args,
            elapsed_seconds=15.3,
            stages_run=["metrics_engine", "screening", "reports"],
            report_paths=[pathlib.Path("HQ01_analysis.md")],
        )

        assert log_path.exists()
        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert data["status"] == "success"
        assert data["mode"] == "reports"
        assert data["elapsed_seconds"] == 15.3


# ===========================================================================
# Part A continued: Data Integrity Checks
# ===========================================================================


class TestDataIntegrity:
    """Verify data integrity across the pipeline."""

    def test_shortlist_tickers_disjoint_from_excluded(
        self, integration_db: pathlib.Path,
    ) -> None:
        """No ticker appears in both excluded and shortlisted sets."""
        from data_acquisition.store import read_table
        from metrics_engine import run_metrics_engine
        from screener.composite_ranker import generate_shortlist
        from screener.exclusions import apply_exclusions
        from screener.hard_filters import apply_hard_filters
        from screener.soft_filters import apply_soft_scores

        composite_df = run_metrics_engine()
        universe_df = read_table("universe")
        filtered_df, exclusion_log = apply_exclusions(universe_df)
        metrics_df = _enrich_metrics_with_quality(_read_metrics_summary())
        surviving_tickers = set(filtered_df["ticker"].tolist())
        metrics_df = metrics_df[
            metrics_df["ticker"].isin(surviving_tickers)
        ].reset_index(drop=True)
        survivors_df, _ = apply_hard_filters(metrics_df)
        ranked_df = apply_soft_scores(survivors_df, composite_df)
        shortlist_df = generate_shortlist(ranked_df, top_n=10)

        excluded_set = set(exclusion_log["ticker"].tolist())
        shortlisted_set = set(shortlist_df["ticker"].tolist())

        assert excluded_set.isdisjoint(shortlisted_set), (
            f"Overlap: {excluded_set & shortlisted_set}"
        )

    def test_high_quality_tickers_rank_above_medium(
        self, integration_db: pathlib.Path,
    ) -> None:
        """HQ tickers should generally score higher than MQ tickers."""
        from data_acquisition.store import read_table
        from metrics_engine import run_metrics_engine
        from screener.composite_ranker import generate_shortlist
        from screener.exclusions import apply_exclusions
        from screener.hard_filters import apply_hard_filters
        from screener.soft_filters import apply_soft_scores

        composite_df = run_metrics_engine()
        universe_df = read_table("universe")
        filtered_df, _ = apply_exclusions(universe_df)
        metrics_df = _enrich_metrics_with_quality(_read_metrics_summary())
        surviving_tickers = set(filtered_df["ticker"].tolist())
        metrics_df = metrics_df[
            metrics_df["ticker"].isin(surviving_tickers)
        ].reset_index(drop=True)
        survivors_df, _ = apply_hard_filters(metrics_df)
        ranked_df = apply_soft_scores(survivors_df, composite_df)

        # Get average score by tier
        hq_scores = ranked_df[
            ranked_df["ticker"].str.startswith("HQ")
        ]["composite_score"].mean()
        mq_scores = ranked_df[
            ranked_df["ticker"].str.startswith("MQ")
        ]["composite_score"].mean()

        assert hq_scores > mq_scores, (
            f"HQ avg score ({hq_scores:.1f}) should exceed "
            f"MQ avg score ({mq_scores:.1f})"
        )

    def test_all_scores_within_bounds(
        self, integration_db: pathlib.Path,
    ) -> None:
        """All composite scores are between 0 and 100."""
        from metrics_engine import run_metrics_engine

        composite_df = run_metrics_engine()
        scores = composite_df["composite_score"].dropna()
        assert (scores >= 0).all(), "Some scores below 0"
        assert (scores <= 100).all(), "Some scores above 100"

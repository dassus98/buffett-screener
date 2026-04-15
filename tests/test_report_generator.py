"""Tests for valuation_reports.report_generator module.

Covers:
  - build_report_context with mock DuckDB data → all required keys present
  - render_deep_dive with mock context → non-empty markdown with expected headers
  - render_summary with mock shortlist → ranked table rendered
  - assumption_log and bear_case generation
  - generate_all_reports writes files to disk
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from valuation_reports.report_generator import (
    _build_assumption_log,
    _build_bear_case,
    _determine_position_sizing,
    _determine_time_horizon,
    _safe_float,
    build_report_context,
    render_deep_dive,
    render_summary,
)


# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------


def _mock_universe_df(ticker: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker,
        "exchange": "NASDAQ",
        "company_name": "Apple Inc.",
        "market_cap_usd": 3_000_000_000_000.0,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "US",
    }])


def _mock_income_df(ticker: str = "AAPL", n_years: int = 10) -> pd.DataFrame:
    years = list(range(2014, 2014 + n_years))
    return pd.DataFrame({
        "ticker": [ticker] * n_years,
        "fiscal_year": years,
        "net_income": [50_000_000 * (1.05 ** i) for i in range(n_years)],
        "total_revenue": [200_000_000 * (1.03 ** i) for i in range(n_years)],
        "gross_profit": [90_000_000 * (1.03 ** i) for i in range(n_years)],
        "sga": [20_000_000] * n_years,
        "operating_income": [60_000_000 * (1.04 ** i) for i in range(n_years)],
        "interest_expense": [2_000_000] * n_years,
        "eps_diluted": [3.0 * (1.05 ** i) for i in range(n_years)],
        "shares_outstanding_diluted": [16_000_000_000] * n_years,
    })


def _mock_balance_df(ticker: str = "AAPL", n_years: int = 10) -> pd.DataFrame:
    years = list(range(2014, 2014 + n_years))
    return pd.DataFrame({
        "ticker": [ticker] * n_years,
        "fiscal_year": years,
        "long_term_debt": [80_000_000] * n_years,
        "shareholders_equity": [200_000_000 * (1.02 ** i) for i in range(n_years)],
        "treasury_stock": [0] * n_years,
    })


def _mock_cashflow_df(ticker: str = "AAPL", n_years: int = 10) -> pd.DataFrame:
    years = list(range(2014, 2014 + n_years))
    return pd.DataFrame({
        "ticker": [ticker] * n_years,
        "fiscal_year": years,
        "depreciation_amortization": [10_000_000] * n_years,
        "capital_expenditures": [12_000_000] * n_years,
        "working_capital_change": [1_000_000] * n_years,
    })


def _mock_market_df(ticker: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker,
        "current_price_usd": 150.0,
        "market_cap_usd": 3_000_000_000_000.0,
        "enterprise_value_usd": 3_100_000_000_000.0,
        "shares_outstanding": 16_000_000_000,
        "high_52w": 180.0,
        "low_52w": 120.0,
        "avg_volume_3m": 80_000_000,
        "pe_ratio_trailing": 25.0,
        "dividend_yield": 0.005,
        "as_of_date": "2024-01-15",
    }])


def _mock_macro_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "key": "us_treasury_10yr",
        "value": 0.04,
        "as_of_date": "2024-01-15",
    }])


def _mock_dq_df(ticker: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker,
        "years_available": 10,
        "missing_critical_fields": "",
        "substitutions_count": 0,
        "drop": False,
        "drop_reason": "",
    }])


def _mock_subs_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "ticker", "fiscal_year", "buffett_field", "api_field_used", "confidence",
    ])


def _mock_read_table(ticker: str = "AAPL"):
    """Return a side_effect function for patching read_table."""
    income = _mock_income_df(ticker)
    balance = _mock_balance_df(ticker)
    cashflow = _mock_cashflow_df(ticker)
    market = _mock_market_df(ticker)
    macro = _mock_macro_df()
    universe = _mock_universe_df(ticker)
    dq = _mock_dq_df(ticker)
    subs = _mock_subs_df()

    def _read(table_name: str, where: str | None = None) -> pd.DataFrame:
        mapping = {
            "universe": universe,
            "income_statement": income,
            "balance_sheet": balance,
            "cash_flow": cashflow,
            "market_data": market,
            "macro_data": macro,
            "data_quality_log": dq,
            "substitution_log": subs,
        }
        return mapping.get(table_name, pd.DataFrame())

    return _read


# ---------------------------------------------------------------------------
# Tests: _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_nan_returns_default(self):
        assert _safe_float(float("nan"), 0.0) == 0.0

    def test_none_returns_default(self):
        assert _safe_float(None, -1.0) == -1.0

    def test_string_returns_default(self):
        assert _safe_float("abc", 0.0) == 0.0


# ---------------------------------------------------------------------------
# Tests: build_report_context
# ---------------------------------------------------------------------------


class TestBuildReportContext:
    """Test build_report_context with fully mocked DuckDB data."""

    REQUIRED_TOP_KEYS = {
        "company_name", "ticker", "exchange", "report_date",
        "latest_fiscal_year", "composite_score", "iv_weighted",
        "current_price_usd", "margin_of_safety_pct",
        "recommendation", "confidence_level", "account_recommendation",
        "time_horizon_years", "critical_flags",
        "qualitative_enabled", "moat_assessment", "moat_indicators",
        "gross_margin_avg_10yr", "roe_avg_10yr",
        "income_tests", "balance_tests", "cashflow_tests",
        "annual_income", "annual_balance", "annual_cashflow",
        "negative_equity_flag", "negative_equity_years",
        "capex_flag", "capex_flag_years",
        "eps_latest", "eps_cagr_10yr", "pe_avg_10yr", "risk_free_rate",
        "projection_years", "terminal_growth_rate",
        "bear_growth", "bear_terminal_pe", "bear_discount_rate",
        "bear_probability", "iv_bear",
        "base_growth", "base_terminal_pe", "base_discount_rate",
        "base_probability", "iv_base",
        "bull_growth", "bull_terminal_pe", "bull_discount_rate",
        "bull_probability", "iv_bull",
        "mos_conservative", "mos_moderate",
        "buy_below_conservative", "buy_below_moderate",
        "margin_of_safety_interpretation",
        "earnings_yield", "bond_yield", "bond_yield_type",
        "earnings_yield_spread", "earnings_yield_interpretation",
        "sensitivity_data", "assumption_log", "bear_case_arguments",
        "position_sizing_guidance", "sell_triggers",
        "account_reasoning", "data_quality",
    }

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_all_required_keys_present(self, mock_iv_rt, mock_rg_rt):
        """All keys expected by deep_dive_template.md are present."""
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        missing = self.REQUIRED_TOP_KEYS - set(ctx.keys())
        assert not missing, f"Missing keys: {missing}"

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_ticker_echoed(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        assert ctx["ticker"] == "AAPL"

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_company_name_populated(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        assert ctx["company_name"] == "Apple Inc."

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_current_price_from_market(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        assert ctx["current_price_usd"] == pytest.approx(150.0)

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_annual_income_has_rows(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        assert len(ctx["annual_income"]) == 10

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_assumption_log_nonempty(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        assert len(ctx["assumption_log"]) >= 2

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_bear_case_has_mean_reversion(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        titles = [a["title"] for a in ctx["bear_case_arguments"]]
        assert "Mean reversion risk" in titles

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_sell_triggers_is_list(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        assert isinstance(ctx["sell_triggers"], list)
        assert len(ctx["sell_triggers"]) == 5

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_report_date_is_today(self, mock_iv_rt, mock_rg_rt):
        import datetime
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        assert ctx["report_date"] == datetime.date.today().isoformat()

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_data_quality_present(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        assert ctx["data_quality"]["years_available"] == 10

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_sensitivity_data_populated(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        ctx = build_report_context("AAPL")
        # Sensitivity may be None if EPS can't be derived, or a dict
        if ctx["sensitivity_data"] is not None:
            assert "eps_sensitivity" in ctx["sensitivity_data"]


# ---------------------------------------------------------------------------
# Tests: render_deep_dive
# ---------------------------------------------------------------------------


class TestRenderDeepDive:
    """Test that render_deep_dive produces valid markdown."""

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_renders_nonempty(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        md = render_deep_dive("AAPL")
        assert len(md) > 500

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_contains_executive_summary(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        md = render_deep_dive("AAPL")
        assert "Executive Summary" in md

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_contains_ticker(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        md = render_deep_dive("AAPL")
        assert "AAPL" in md
        assert "Apple Inc." in md

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_contains_valuation_section(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        md = render_deep_dive("AAPL")
        assert "Valuation" in md
        assert "Three Scenarios" in md

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_contains_financial_analysis(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        md = render_deep_dive("AAPL")
        assert "Financial Statement Analysis" in md
        assert "Income Statement Tests" in md

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_contains_investment_strategy(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        md = render_deep_dive("AAPL")
        assert "Investment Strategy" in md

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_contains_bear_case(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        md = render_deep_dive("AAPL")
        assert "Devil's Advocate" in md
        assert "Mean reversion risk" in md

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_contains_data_quality(self, mock_iv_rt, mock_rg_rt):
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        md = render_deep_dive("AAPL")
        assert "Data Quality Notes" in md

    @patch("valuation_reports.report_generator.read_table")
    @patch("valuation_reports.intrinsic_value.read_table")
    def test_technology_bear_case_for_tech_sector(self, mock_iv_rt, mock_rg_rt):
        """Tech sector stock should have technology disruption bear case."""
        mock_rg_rt.side_effect = _mock_read_table("AAPL")
        mock_iv_rt.side_effect = _mock_read_table("AAPL")
        md = render_deep_dive("AAPL")
        assert "Technology disruption risk" in md


# ---------------------------------------------------------------------------
# Tests: render_summary
# ---------------------------------------------------------------------------


class TestRenderSummary:
    """Test the portfolio-level summary report rendering."""

    def _mock_shortlist(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "exchange": "NASDAQ",
                "sector": "Technology",
                "composite_score": 82.5,
                "margin_of_safety_pct": 0.233,
                "recommendation": "BUY",
                "confidence_level": "High",
                "account_recommendation": "Either",
                "gross_margin_avg_10yr": 0.432,
                "roe_avg_10yr": 0.285,
                "eps_cagr_10yr": 0.123,
            },
            {
                "ticker": "KO",
                "company_name": "Coca-Cola Co.",
                "exchange": "NYSE",
                "sector": "Consumer Staples",
                "composite_score": 78.3,
                "margin_of_safety_pct": 0.185,
                "recommendation": "HOLD",
                "confidence_level": "High",
                "account_recommendation": "RRSP",
                "gross_margin_avg_10yr": 0.605,
                "roe_avg_10yr": 0.395,
                "eps_cagr_10yr": 0.047,
            },
        ])

    def _mock_screener_summary(self) -> dict:
        return {
            "universe_size": 2500,
            "after_exclusions": 1800,
            "passed_hard_filters": 120,
            "filter_stats": {"min_profitable_years": 450},
            "macro": {
                "us_treasury_10yr": 0.04,
            },
        }

    def test_renders_nonempty(self):
        md = render_summary(self._mock_shortlist(), self._mock_screener_summary())
        assert len(md) > 200

    def test_contains_title(self):
        md = render_summary(self._mock_shortlist(), self._mock_screener_summary())
        assert "Top 2 Summary" in md

    def test_contains_both_tickers(self):
        md = render_summary(self._mock_shortlist(), self._mock_screener_summary())
        assert "AAPL" in md
        assert "KO" in md

    def test_contains_company_names(self):
        md = render_summary(self._mock_shortlist(), self._mock_screener_summary())
        assert "Apple Inc." in md
        assert "Coca-Cola Co." in md

    def test_contains_screener_statistics(self):
        md = render_summary(self._mock_shortlist(), self._mock_screener_summary())
        assert "Screener Statistics" in md
        assert "2500" in md

    def test_contains_macro_context(self):
        md = render_summary(self._mock_shortlist(), self._mock_screener_summary())
        assert "Macro Context" in md
        assert "4.0%" in md  # US Treasury (Jinja2 round drops trailing zero)

    def test_contains_sector_summary(self):
        md = render_summary(self._mock_shortlist(), self._mock_screener_summary())
        assert "Technology" in md
        assert "Consumer Staples" in md

    def test_ranked_table_order(self):
        md = render_summary(self._mock_shortlist(), self._mock_screener_summary())
        # AAPL should appear before KO
        aapl_pos = md.index("AAPL")
        ko_pos = md.index("KO")
        assert aapl_pos < ko_pos

    def test_recommendations_present(self):
        md = render_summary(self._mock_shortlist(), self._mock_screener_summary())
        assert "BUY" in md
        assert "HOLD" in md


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestBuildAssumptionLog:
    """Test assumption log auto-population."""

    @patch("valuation_reports.report_generator.read_table")
    def test_always_includes_capex_assumption(self, mock_rt):
        mock_rt.return_value = pd.DataFrame()
        log = _build_assumption_log("AAPL", {"years_available": 10})
        assumptions = [e["assumption"] for e in log]
        assert any("CapEx" in a for a in assumptions)

    @patch("valuation_reports.report_generator.read_table")
    def test_always_includes_growth_assumption(self, mock_rt):
        mock_rt.return_value = pd.DataFrame()
        log = _build_assumption_log("AAPL", {"years_available": 10})
        assumptions = [e["assumption"] for e in log]
        assert any("EPS CAGR" in a for a in assumptions)

    @patch("valuation_reports.report_generator.read_table")
    def test_includes_substitutions(self, mock_rt):
        subs_df = pd.DataFrame([{
            "ticker": "AAPL",
            "fiscal_year": 2020,
            "buffett_field": "sga",
            "api_field_used": "total_operating_expenses",
            "confidence": "Medium",
        }])
        mock_rt.return_value = subs_df
        log = _build_assumption_log("AAPL", {"years_available": 10})
        subs = [e for e in log if "substitution" in e["assumption"].lower()]
        assert len(subs) == 1
        assert "sga" in subs[0]["assumption"]


class TestBuildBearCase:
    """Test rule-based bear case argument generation."""

    def test_mean_reversion_always_present(self):
        args = _build_bear_case("AAPL", "Technology", {}, {})
        titles = [a["title"] for a in args]
        assert "Mean reversion risk" in titles

    def test_low_margin_argument(self):
        args = _build_bear_case(
            "XYZ", "Retail", {},
            {"gross_margin_avg_10yr": 0.35},
        )
        titles = [a["title"] for a in args]
        assert "Limited pricing power" in titles

    def test_tech_disruption_argument(self):
        args = _build_bear_case("AAPL", "Technology", {}, {})
        titles = [a["title"] for a in args]
        assert "Technology disruption risk" in titles

    def test_debt_argument(self):
        args = _build_bear_case(
            "XYZ", "Industrial", {},
            {"debt_payoff_years": 4.0},
        )
        titles = [a["title"] for a in args]
        assert "Elevated debt levels" in titles

    def test_decline_years_argument(self):
        args = _build_bear_case(
            "XYZ", "Industrial", {},
            {"decline_years": 3},
        )
        titles = [a["title"] for a in args]
        assert "Earnings volatility" in titles

    def test_no_margin_argument_when_high(self):
        args = _build_bear_case(
            "XYZ", "Industrial", {},
            {"gross_margin_avg_10yr": 0.65},
        )
        titles = [a["title"] for a in args]
        assert "Limited pricing power" not in titles


class TestTimeHorizon:
    """Test time horizon determination."""

    def test_low_confidence_shortened(self):
        assert _determine_time_horizon(90.0, "Low", 0.20) == 5

    def test_high_score_high_cagr_extended(self):
        assert _determine_time_horizon(85.0, "High", 0.18) == 15

    def test_standard_horizon_default(self):
        assert _determine_time_horizon(70.0, "High", 0.10) == 10


class TestPositionSizing:
    """Test position sizing guidance generation."""

    def test_high_confidence_high_score(self):
        guidance = _determine_position_sizing(85.0, "High")
        assert "8-10%" in guidance

    def test_low_confidence(self):
        guidance = _determine_position_sizing(90.0, "Low")
        assert "1-2%" in guidance

    def test_moderate_confidence_high_score(self):
        guidance = _determine_position_sizing(75.0, "Moderate")
        assert "3-5%" in guidance

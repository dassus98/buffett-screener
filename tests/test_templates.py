"""Tests for Jinja2 report templates.

Validates that both deep_dive_template.md and summary_table_template.md:
  - Load without Jinja2 syntax errors
  - Render with mock data to produce non-empty output
  - Contain expected section headings and key data in rendered output
"""

from __future__ import annotations

import pathlib

import jinja2
import pytest

_TEMPLATES_DIR = (
    pathlib.Path(__file__).parent.parent / "valuation_reports" / "templates"
)


# ---------------------------------------------------------------------------
# Jinja2 environment (shared across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jinja_env() -> jinja2.Environment:
    """Create a Jinja2 environment pointing at the templates directory."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=jinja2.StrictUndefined,
    )


# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------


def _deep_dive_context() -> dict:
    """Comprehensive mock context for deep_dive_template.md."""
    return {
        # Executive Summary
        "company_name": "Apple Inc.",
        "ticker": "AAPL",
        "exchange": "NASDAQ",
        "report_date": "2024-06-15",
        "latest_fiscal_year": 2023,
        "composite_score": 82.5,
        "iv_weighted": 195.40,
        "current_price_usd": 150.00,
        "margin_of_safety_pct": 0.2325,
        "recommendation": "BUY",
        "confidence_level": "High",
        "account_recommendation": "Either",
        "time_horizon_years": 10,
        "critical_flags": [],
        # Moat
        "qualitative_enabled": False,
        "moat_assessment": None,
        "moat_indicators": [
            "10-year average gross margin: 43.2% — Consistent with pricing power",
            "10-year average ROE: 28.5% — Consistent with durable advantage",
            "Gross margin standard deviation: 2.1% — Stable",
        ],
        "gross_margin_avg_10yr": 0.432,
        "roe_avg_10yr": 0.285,
        # Income Statement Tests
        "income_tests": [
            {
                "name": "Earnings Consistency (F10)",
                "result": "10/10 years profitable",
                "threshold": ">= 8",
                "status": "Pass",
                "notes": "All years profitable",
            },
            {
                "name": "Average ROE (F3)",
                "result": "28.5%",
                "threshold": ">= 15%",
                "status": "Pass",
                "notes": "Strong and consistent",
            },
            {
                "name": "EPS CAGR (F11)",
                "result": "12.3%",
                "threshold": "> 0%",
                "status": "Pass",
                "notes": "1 decline year",
            },
        ],
        "annual_income": [
            {
                "fiscal_year": 2022,
                "revenue": 394328,
                "gross_margin": 0.433,
                "operating_margin": 0.302,
                "net_margin": 0.253,
                "eps_diluted": 6.15,
                "roe": 0.287,
            },
            {
                "fiscal_year": 2023,
                "revenue": 383285,
                "gross_margin": 0.441,
                "operating_margin": 0.298,
                "net_margin": 0.259,
                "eps_diluted": 6.42,
                "roe": 0.291,
            },
        ],
        # Balance Sheet Tests
        "balance_tests": [
            {
                "name": "Debt Payoff Years (F5)",
                "result": "2.1 years",
                "threshold": "<= 5",
                "status": "Pass",
                "notes": "Based on owner earnings",
            },
            {
                "name": "Debt-to-Equity (F6)",
                "result": "0.45",
                "threshold": "< 0.80",
                "status": "Pass",
                "notes": "10-yr avg: 0.52",
            },
        ],
        "negative_equity_flag": False,
        "negative_equity_years": 0,
        "annual_balance": [
            {
                "fiscal_year": 2022,
                "long_term_debt": 98959,
                "shareholders_equity": 50672,
                "de_ratio": 0.45,
                "retained_earnings": 35000,
            },
            {
                "fiscal_year": 2023,
                "long_term_debt": 95281,
                "shareholders_equity": 62146,
                "de_ratio": 0.41,
                "retained_earnings": 42000,
            },
        ],
        # Cash Flow Tests
        "cashflow_tests": [
            {
                "name": "Owner Earnings (F1)",
                "result": "$95,000K",
                "threshold": "Positive",
                "status": "Pass",
                "notes": "",
            },
            {
                "name": "CapEx / Net Income (F12)",
                "result": "22.5%",
                "threshold": "< 50%",
                "status": "Pass",
                "notes": "Capital light",
            },
        ],
        "capex_flag": False,
        "capex_flag_years": 0,
        "annual_cashflow": [
            {
                "fiscal_year": 2022,
                "operating_cash_flow": 122151,
                "capital_expenditures": 10708,
                "free_cash_flow": 111443,
                "owner_earnings": 95000,
                "depreciation_amortization": 11104,
            },
            {
                "fiscal_year": 2023,
                "operating_cash_flow": 110543,
                "capital_expenditures": 11000,
                "free_cash_flow": 99543,
                "owner_earnings": 90000,
                "depreciation_amortization": 11500,
            },
        ],
        # Valuation
        "eps_latest": 6.42,
        "eps_cagr_10yr": 0.123,
        "pe_avg_10yr": 24.5,
        "risk_free_rate": 0.0425,
        "projection_years": 10,
        "terminal_growth_rate": 0.03,
        "bear_growth": 0.0615,
        "bear_terminal_pe": 12.0,
        "bear_discount_rate": 0.0925,
        "bear_probability": 0.25,
        "iv_bear": 95.20,
        "base_growth": 0.123,
        "base_terminal_pe": 24.5,
        "base_discount_rate": 0.0725,
        "base_probability": 0.50,
        "iv_base": 195.40,
        "bull_growth": 0.1599,
        "bull_terminal_pe": 24.5,
        "bull_discount_rate": 0.0625,
        "bull_probability": 0.25,
        "iv_bull": 320.80,
        # Margin of Safety
        "mos_conservative": 0.50,
        "mos_moderate": 0.33,
        "buy_below_conservative": 97.70,
        "buy_below_moderate": 130.92,
        "margin_of_safety_interpretation": (
            "Current price offers a 23.3% margin of safety relative to the "
            "weighted intrinsic value. This exceeds the moderate 33% buy-below "
            "threshold but falls short of the conservative 50% level."
        ),
        # Earnings Yield
        "earnings_yield": 0.0428,
        "bond_yield": 0.0425,
        "bond_yield_type": "US Treasury 10yr",
        "earnings_yield_spread": 0.0003,
        "earnings_yield_interpretation": "Moderate",
        # Sensitivity
        "sensitivity_data": {
            "eps_sensitivity": [
                (0.0861, 120.50, 0.196),
                (0.1046, 155.20, 0.334),
                (0.1230, 195.40, 0.232),
                (0.1415, 242.80, 0.382),
                (0.1599, 298.50, 0.497),
            ],
            "pe_sensitivity": [
                (18.4, 146.55, 0.023),
                (21.4, 170.97, 0.122),
                (24.5, 195.40, 0.232),
                (27.6, 219.83, 0.318),
                (30.6, 244.25, 0.386),
            ],
            "discount_sensitivity": [
                (0.0525, 234.48, 0.360),
                (0.0625, 213.20, 0.296),
                (0.0725, 195.40, 0.232),
                (0.0825, 178.60, 0.160),
                (0.0925, 163.50, 0.083),
            ],
        },
        # Assumption Log
        "assumption_log": [
            {
                "assumption": "D&A sourced from income statement",
                "confidence": "Medium",
                "failure_mode": "May include non-operating charges",
                "consequence": "Owner earnings (F1) may be overstated",
            },
        ],
        # Bear Case
        "bear_case_arguments": [
            {
                "title": "Smartphone market saturation",
                "body": (
                    "Global smartphone shipments have plateaued. iPhone revenue "
                    "growth depends increasingly on ASP increases and services "
                    "attach rate rather than unit volume growth."
                ),
            },
            {
                "title": "Regulatory risk",
                "body": (
                    "Antitrust scrutiny of the App Store's 30% commission rate "
                    "could reduce services margins if forced to open to third-"
                    "party payment processors."
                ),
            },
        ],
        # Investment Strategy
        "entry_strategy": {
            "current_price": 150.00,
            "ideal_entry": 130.92,
            "discount_needed_pct": 0.127,
            "strategy": (
                "Current price of $150.00 is 14.6% above the ideal entry "
                "of $130.92. Wait for a pullback or accumulate in tranches."
            ),
        },
        "position_sizing_guidance": (
            "High confidence, score >= 80: Up to 8-10% of portfolio."
        ),
        "sell_triggers": [
            {
                "signal": "roe_deterioration",
                "threshold": 0.12,
                "current_value": 0.291,
                "status": "OK",
            },
            {
                "signal": "leverage_spike",
                "threshold": 1.0,
                "current_value": 0.41,
                "status": "OK",
            },
        ],
        "account_reasoning": (
            "No withholding tax issue (minimal dividend). TFSA gains are "
            "fully tax-free vs. taxed as ordinary income on RRSP withdrawal."
        ),
        # Data Quality
        "data_quality": {
            "years_available": 10,
            "substitutions_count": 1,
            "missing_critical_fields": "",
            "drop_reason": "",
        },
    }


def _summary_context() -> dict:
    """Comprehensive mock context for summary_table_template.md."""
    return {
        "top_n": 10,
        "run_date": "2024-06-15",
        "universe_size": 2500,
        "passed_hard_filters": 120,
        "after_exclusions": 1800,
        "shortlist_count": 10,
        "macro": {
            "us_treasury_10yr": 0.0425,
            "goc_bond_10yr": 0.0380,
            "usd_cad_rate": 1.3650,
        },
        "rows": [
            {
                "rank": 1,
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "exchange": "NASDAQ",
                "composite_score": 82.5,
                "iv_weighted": 195.40,
                "current_price_usd": 150.00,
                "margin_of_safety_pct": 0.233,
                "recommendation": "BUY",
                "confidence_level": "High",
                "account_recommendation": "Either",
                "gross_margin_avg_10yr": 0.432,
                "roe_avg_10yr": 0.285,
                "eps_cagr_10yr": 0.123,
            },
            {
                "rank": 2,
                "ticker": "KO",
                "company_name": "Coca-Cola Co.",
                "exchange": "NYSE",
                "composite_score": 78.3,
                "iv_weighted": 62.50,
                "current_price_usd": 55.00,
                "margin_of_safety_pct": 0.185,
                "recommendation": "HOLD",
                "confidence_level": "High",
                "account_recommendation": "RRSP",
                "gross_margin_avg_10yr": 0.605,
                "roe_avg_10yr": 0.395,
                "eps_cagr_10yr": 0.047,
            },
            {
                "rank": 3,
                "ticker": "RY.TO",
                "company_name": "Royal Bank of Canada",
                "exchange": "TSX",
                "composite_score": 71.2,
                "iv_weighted": 145.80,
                "current_price_usd": 125.00,
                "margin_of_safety_pct": 0.142,
                "recommendation": "HOLD",
                "confidence_level": "Moderate",
                "account_recommendation": "TFSA",
                "gross_margin_avg_10yr": 0.550,
                "roe_avg_10yr": 0.168,
                "eps_cagr_10yr": 0.082,
            },
        ],
        "filter_stats": {
            "min_profitable_years": 450,
            "min_avg_roe": 380,
            "min_eps_cagr": 220,
            "max_debt_payoff_years": 150,
        },
        "sector_summary": [
            {"name": "Technology", "count": 4, "avg_score": 79.2},
            {"name": "Consumer Staples", "count": 3, "avg_score": 75.8},
            {"name": "Healthcare", "count": 2, "avg_score": 72.1},
            {"name": "Industrials", "count": 1, "avg_score": 70.5},
        ],
    }


# ---------------------------------------------------------------------------
# Tests: deep_dive_template.md
# ---------------------------------------------------------------------------


class TestDeepDiveTemplate:
    """Validate deep_dive_template.md loads and renders correctly."""

    def test_template_loads(self, jinja_env: jinja2.Environment):
        """Template file loads without Jinja2 syntax errors."""
        tmpl = jinja_env.get_template("deep_dive_template.md")
        assert tmpl is not None

    def test_renders_nonempty(self, jinja_env: jinja2.Environment):
        """Rendering with full mock data produces non-empty output."""
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert len(output) > 100

    def test_title_contains_ticker_and_company(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Apple Inc." in output
        assert "AAPL" in output

    def test_executive_summary_section(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Executive Summary" in output
        assert "82.5" in output  # composite_score
        assert "195.4" in output  # iv_weighted (Jinja2 round drops trailing zeros)
        assert "150.0" in output  # current_price
        assert "BUY" in output
        assert "High" in output

    def test_recommendation_and_confidence_present(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "**BUY**" in output
        assert "10+ years" in output

    def test_moat_section_quantitative(self, jinja_env: jinja2.Environment):
        """When qualitative_enabled=False, shows quantitative indicators."""
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Durable Competitive Advantage" in output
        assert "pricing power" in output

    def test_moat_section_qualitative(self, jinja_env: jinja2.Environment):
        """When qualitative_enabled=True with moat_assessment, shows moat type."""
        ctx = _deep_dive_context()
        ctx["qualitative_enabled"] = True
        ctx["moat_assessment"] = {
            "moat_type": "Brand + Ecosystem",
            "evidence": "Apple's ecosystem creates high switching costs.",
            "threats": ["Regulatory unbundling", "AI commoditization"],
        }
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "Brand + Ecosystem" in output
        assert "switching costs" in output
        assert "Regulatory unbundling" in output

    def test_income_statement_tests_table(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Income Statement Tests" in output
        assert "Earnings Consistency" in output
        assert "Average ROE" in output

    def test_annual_income_detail(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Year-by-Year Income Statement Detail" in output
        assert "2022" in output
        assert "2023" in output
        assert "6.15" in output  # eps_diluted 2022
        assert "6.42" in output  # eps_diluted 2023

    def test_balance_sheet_tests(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Balance Sheet Tests" in output
        assert "Debt Payoff" in output

    def test_negative_equity_flag_hidden(self, jinja_env: jinja2.Environment):
        """No negative equity flag → section hidden."""
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Negative shareholders' equity detected" not in output

    def test_negative_equity_flag_shown(self, jinja_env: jinja2.Environment):
        """With negative equity → flag section shown."""
        ctx = _deep_dive_context()
        ctx["negative_equity_flag"] = True
        ctx["negative_equity_years"] = 2
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "Negative shareholders' equity detected" in output
        assert "2 year(s)" in output

    def test_cashflow_tests(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Cash Flow Tests" in output
        assert "Owner Earnings" in output

    def test_capex_flag_hidden(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "CapEx exceeds 2x D&A" not in output

    def test_capex_flag_shown(self, jinja_env: jinja2.Environment):
        ctx = _deep_dive_context()
        ctx["capex_flag"] = True
        ctx["capex_flag_years"] = 3
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "CapEx exceeds 2x D&A" in output

    def test_valuation_three_scenarios(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Valuation" in output
        assert "Three Scenarios" in output
        assert "Bear" in output
        assert "Base" in output
        assert "Bull" in output
        assert "Weighted Average" in output

    def test_margin_of_safety_section(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Margin of Safety" in output
        assert "Conservative buy below" in output
        assert "Moderate buy below" in output

    def test_earnings_yield_section(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Earnings Yield vs. Bond Yield" in output
        assert "US Treasury 10yr" in output

    def test_sensitivity_analysis_present(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Sensitivity Analysis" in output
        assert "EPS Growth Rate Sensitivity" in output
        assert "Terminal P/E Sensitivity" in output
        assert "Discount Rate Sensitivity" in output

    def test_sensitivity_hidden_when_no_data(self, jinja_env: jinja2.Environment):
        """When sensitivity_data is None/empty, section is omitted."""
        ctx = _deep_dive_context()
        ctx["sensitivity_data"] = None
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "Sensitivity Analysis" not in output

    def test_assumption_log(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Assumption Log" in output
        assert "D&A sourced from income statement" in output
        assert "Owner earnings" in output

    def test_assumption_log_empty(self, jinja_env: jinja2.Environment):
        ctx = _deep_dive_context()
        ctx["assumption_log"] = []
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "No material assumptions" in output

    def test_bear_case_section(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Devil's Advocate" in output
        assert "Smartphone market saturation" in output
        assert "Regulatory risk" in output

    def test_bear_case_empty(self, jinja_env: jinja2.Environment):
        ctx = _deep_dive_context()
        ctx["bear_case_arguments"] = []
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "No bear case arguments generated" in output

    def test_investment_strategy_section(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Investment Strategy" in output
        assert "Entry Price Target" in output
        assert "Position Sizing" in output
        assert "Sell Triggers" in output

    def test_sell_triggers_table(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "roe_deterioration" in output
        assert "leverage_spike" in output

    def test_account_reasoning(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Recommended Account" in output
        assert "Either" in output

    def test_data_quality_section(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Data Quality Notes" in output
        assert "Years of data available" in output

    def test_critical_flags_hidden_when_empty(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "Data Quality Flags" not in output

    def test_critical_flags_shown(self, jinja_env: jinja2.Environment):
        ctx = _deep_dive_context()
        ctx["critical_flags"] = ["Missing net_income for 2018", "SG&A estimated"]
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "Data Quality Flags" in output
        assert "Missing net_income for 2018" in output
        assert "SG&A estimated" in output

    def test_tsx_bond_yield_label(self, jinja_env: jinja2.Environment):
        """TSX exchange → GoC bond yield label."""
        ctx = _deep_dive_context()
        ctx["exchange"] = "TSX"
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "GoC 10yr (TSX security)" in output

    def test_entry_strategy_narrative(self, jinja_env: jinja2.Environment):
        """Entry strategy narrative is rendered in Investment Strategy section."""
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "above the ideal entry" in output
        assert "130.92" in output  # ideal_entry price

    def test_entry_strategy_missing_falls_back(self, jinja_env: jinja2.Environment):
        """When entry_strategy is None/missing, falls back to buy_below_moderate."""
        ctx = _deep_dive_context()
        ctx["entry_strategy"] = None
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "Entry Price Target" in output
        assert "130.92" in output  # buy_below_moderate

    def test_sensitivity_partial_eps_only(self, jinja_env: jinja2.Environment):
        """Only EPS sensitivity populated — others hidden."""
        ctx = _deep_dive_context()
        ctx["sensitivity_data"] = {
            "eps_sensitivity": [
                (0.0861, 120.50, 0.196),
                (0.1230, 195.40, 0.232),
                (0.1599, 298.50, 0.497),
            ],
            "pe_sensitivity": [],
            "discount_sensitivity": [],
        }
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**ctx)
        assert "EPS Growth Rate Sensitivity" in output
        assert "Terminal P/E Sensitivity" not in output
        assert "Discount Rate Sensitivity" not in output

    def test_disclaimer_present(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("deep_dive_template.md")
        output = tmpl.render(**_deep_dive_context())
        assert "not investment advice" in output


# ---------------------------------------------------------------------------
# Tests: summary_table_template.md
# ---------------------------------------------------------------------------


class TestSummaryTableTemplate:
    """Validate summary_table_template.md loads and renders correctly."""

    def test_template_loads(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        assert tmpl is not None

    def test_renders_nonempty(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert len(output) > 100

    def test_title_contains_top_n(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "Top 10 Summary" in output

    def test_generation_date(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "2024-06-15" in output

    def test_macro_context(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "Macro Context" in output
        assert "4.25%" in output  # US Treasury
        assert "3.8%" in output   # GoC (Jinja2 round drops trailing zeros)
        assert "1.365" in output  # USD/CAD

    def test_shortlisted_securities_header(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "Shortlisted Securities" in output

    def test_all_rows_rendered(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "AAPL" in output
        assert "KO" in output
        assert "RY.TO" in output
        assert "Apple Inc." in output
        assert "Coca-Cola Co." in output

    def test_row_data_values(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "82.5" in output  # AAPL score
        assert "BUY" in output
        assert "HOLD" in output
        assert "RRSP" in output
        assert "TFSA" in output

    def test_screener_statistics(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "Screener Statistics" in output
        assert "2500" in output  # universe_size
        assert "1800" in output  # after_exclusions
        assert "120" in output  # passed_hard_filters

    def test_filter_stats(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "Hard filter elimination breakdown" in output
        assert "min_profitable_years" in output
        assert "450" in output

    def test_sector_summary(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "Top Sectors Represented" in output
        assert "Technology" in output
        assert "Consumer Staples" in output
        assert "79.2" in output  # Technology avg_score

    def test_sector_summary_hidden_when_empty(self, jinja_env: jinja2.Environment):
        ctx = _summary_context()
        ctx["sector_summary"] = []
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**ctx)
        assert "Top Sectors Represented" not in output

    def test_disclaimer_present(self, jinja_env: jinja2.Environment):
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "not investment advice" in output

    def test_iv_and_price_columns_present(self, jinja_env: jinja2.Environment):
        """Summary table includes IV and Price columns."""
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**_summary_context())
        assert "| IV |" in output
        assert "| Price |" in output
        assert "$195.4" in output   # AAPL iv_weighted
        assert "$150.0" in output   # AAPL current_price_usd

    def test_filter_stats_hidden_when_empty(self, jinja_env: jinja2.Environment):
        ctx = _summary_context()
        ctx["filter_stats"] = {}
        tmpl = jinja_env.get_template("summary_table_template.md")
        output = tmpl.render(**ctx)
        assert "Hard filter elimination breakdown" not in output

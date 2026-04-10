"""Unit tests for all pure financial formula functions in metrics_engine/ (no I/O, deterministic inputs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import math

from metrics_engine.leverage import (
    compute_debt_payoff,
    compute_debt_to_equity,
    compute_interest_coverage,
)
from metrics_engine.owner_earnings import compute_owner_earnings
from metrics_engine.profitability import (
    compute_gross_margin,
    compute_net_margin,
    compute_roe,
    compute_sga_ratio,
)
from metrics_engine.returns import (
    compute_initial_rate_of_return,
    compute_return_on_retained_earnings,
)
from screener.filter_config_loader import ConfigError, get_threshold, load_config


# ---------------------------------------------------------------------------
# DataFrame builders (single-ticker, minimal columns)
# ---------------------------------------------------------------------------

def _income_df(
    fiscal_years: list[int],
    net_income: list[float],
    total_revenue: list[float] | None = None,
    gross_profit: list[float] | None = None,
    sga: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal income statement DataFrame for formula tests."""
    n = len(fiscal_years)
    return pd.DataFrame({
        "fiscal_year": fiscal_years,
        "net_income": net_income,
        "total_revenue": total_revenue if total_revenue is not None else [1_000.0] * n,
        "gross_profit": gross_profit if gross_profit is not None else [400.0] * n,
        "sga": sga if sga is not None else [100.0] * n,
    })


def _balance_df(
    fiscal_years: list[int],
    shareholders_equity: list[float],
) -> pd.DataFrame:
    """Build a minimal balance sheet DataFrame for formula tests."""
    return pd.DataFrame({
        "fiscal_year": fiscal_years,
        "shareholders_equity": shareholders_equity,
    })


# ===========================================================================
# Tests — get_threshold / load_config
# ===========================================================================

class TestGetThreshold:
    def test_returns_known_roe_floor(self):
        # filter_config.yaml → hard_filters.min_avg_roe = 0.15
        assert get_threshold("hard_filters.min_avg_roe") == pytest.approx(0.15)

    def test_returns_nested_soft_score_weight(self):
        # filter_config.yaml → soft_scores.roe.weight = 0.15
        assert get_threshold("soft_scores.roe.weight") == pytest.approx(0.15)

    def test_returns_three_level_path(self):
        # filter_config.yaml → soft_scores.gross_margin.weight = 0.10
        assert get_threshold("soft_scores.gross_margin.weight") == pytest.approx(0.10)

    def test_missing_top_level_key_raises_config_error(self):
        with pytest.raises(ConfigError, match="no_such_section"):
            get_threshold("no_such_section.some_key")

    def test_missing_nested_key_raises_config_error(self):
        with pytest.raises(ConfigError, match="nonexistent_key"):
            get_threshold("hard_filters.nonexistent_key")

    def test_error_message_includes_full_path(self):
        with pytest.raises(ConfigError, match="hard_filters.missing"):
            get_threshold("hard_filters.missing")

    def test_load_config_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "hard_filters" in cfg
        assert "soft_scores" in cfg


# ===========================================================================
# Tests — compute_roe (F3)
# ===========================================================================

class TestComputeROE:
    def test_first_year_uses_current_equity_only(self):
        """No prior year → denominator = equity_t, not an average."""
        income = _income_df([2020], [100.0])
        balance = _balance_df([2020], [500.0])
        annual, _ = compute_roe(income, balance)
        assert annual["roe"].iloc[0] == pytest.approx(100.0 / 500.0)
        assert annual["negative_equity"].iloc[0] == False  # noqa: E712

    def test_second_year_uses_average_equity(self):
        """Year t: avg = (equity_{t-1} + equity_t) / 2."""
        income = _income_df([2020, 2021], [100.0, 110.0])
        balance = _balance_df([2020, 2021], [500.0, 550.0])
        annual, _ = compute_roe(income, balance)
        # Year 2021: avg = (500 + 550) / 2 = 525
        row = annual.loc[annual["fiscal_year"] == 2021].iloc[0]
        assert row["roe"] == pytest.approx(110.0 / 525.0)

    def test_three_year_known_values(self):
        """Verify all three years of a 3-year series end-to-end."""
        income = _income_df([2020, 2021, 2022], [100.0, 110.0, 120.0])
        balance = _balance_df([2020, 2021, 2022], [500.0, 550.0, 600.0])
        annual, _ = compute_roe(income, balance)
        # 2020: no prior → 100/500 = 0.20
        assert annual.loc[annual["fiscal_year"] == 2020, "roe"].iloc[0] == pytest.approx(0.20)
        # 2021: avg = 525 → 110/525
        assert annual.loc[annual["fiscal_year"] == 2021, "roe"].iloc[0] == pytest.approx(110.0 / 525.0)
        # 2022: avg = 575 → 120/575
        assert annual.loc[annual["fiscal_year"] == 2022, "roe"].iloc[0] == pytest.approx(120.0 / 575.0)

    def test_negative_equity_produces_nan_and_flag(self):
        """Avg equity ≤ 0 → roe = NaN, negative_equity = True."""
        income = _income_df([2020, 2021, 2022], [100.0, 110.0, 120.0])
        # 2020: equity = -500 → avg = -500 ≤ 0 → NaN
        balance = _balance_df([2020, 2021, 2022], [-500.0, 550.0, 600.0])
        annual, _ = compute_roe(income, balance)
        row_2020 = annual.loc[annual["fiscal_year"] == 2020].iloc[0]
        assert pd.isna(row_2020["roe"])
        assert row_2020["negative_equity"] == True  # noqa: E712

    def test_positive_equity_not_flagged(self):
        income = _income_df([2020, 2021], [100.0, 110.0])
        balance = _balance_df([2020, 2021], [500.0, 550.0])
        annual, _ = compute_roe(income, balance)
        assert not annual["negative_equity"].any()

    def test_summary_avg_roe(self):
        income = _income_df([2020, 2021, 2022], [100.0, 110.0, 120.0])
        balance = _balance_df([2020, 2021, 2022], [500.0, 550.0, 600.0])
        _, summary = compute_roe(income, balance)
        expected = (100.0 / 500.0 + 110.0 / 525.0 + 120.0 / 575.0) / 3.0
        assert summary["avg_roe"] == pytest.approx(expected, rel=1e-4)

    def test_summary_years_above_threshold_all_pass(self):
        """All ROEs ~20% exceed the 15% default threshold."""
        income = _income_df([2020, 2021, 2022], [100.0, 110.0, 120.0])
        balance = _balance_df([2020, 2021, 2022], [500.0, 550.0, 600.0])
        _, summary = compute_roe(income, balance)
        assert summary["years_above_threshold"] == 3

    def test_summary_years_above_threshold_none_pass(self):
        """ROE ~1% — well below the 15% threshold."""
        income = _income_df([2020, 2021], [5.0, 5.5])
        balance = _balance_df([2020, 2021], [500.0, 550.0])
        _, summary = compute_roe(income, balance)
        assert summary["years_above_threshold"] == 0

    def test_roe_stdev_single_year_is_nan(self):
        """Standard deviation requires ≥ 2 observations."""
        income = _income_df([2020], [100.0])
        balance = _balance_df([2020], [500.0])
        _, summary = compute_roe(income, balance)
        assert pd.isna(summary["roe_stdev"])

    def test_roe_stdev_multi_year_not_nan(self):
        income = _income_df([2020, 2021, 2022], [100.0, 110.0, 120.0])
        balance = _balance_df([2020, 2021, 2022], [500.0, 550.0, 600.0])
        _, summary = compute_roe(income, balance)
        assert not pd.isna(summary["roe_stdev"])

    def test_annual_df_column_names(self):
        income = _income_df([2020], [100.0])
        balance = _balance_df([2020], [500.0])
        annual, _ = compute_roe(income, balance)
        assert set(annual.columns) == {"fiscal_year", "roe", "negative_equity"}

    def test_nan_equity_produces_nan_roe(self):
        """Missing balance sheet data for a year → NaN ROE."""
        income = _income_df([2020, 2021], [100.0, 110.0])
        balance = _balance_df([2020], [500.0])  # 2021 equity missing
        annual, _ = compute_roe(income, balance)
        assert pd.isna(annual.loc[annual["fiscal_year"] == 2021, "roe"].iloc[0])


# ===========================================================================
# Tests — compute_gross_margin (F7)
# ===========================================================================

class TestComputeGrossMargin:
    def test_basic_known_values(self):
        """gross_profit / total_revenue for three years."""
        income = _income_df(
            [2020, 2021, 2022], [0.0, 0.0, 0.0],
            total_revenue=[1_000.0, 1_000.0, 1_000.0],
            gross_profit=[400.0, 420.0, 440.0],
        )
        annual, _ = compute_gross_margin(income)
        assert list(annual["gross_margin"]) == pytest.approx([0.40, 0.42, 0.44])

    def test_zero_revenue_produces_nan(self):
        income = _income_df(
            [2020, 2021], [0.0, 0.0],
            total_revenue=[0.0, 1_000.0],
            gross_profit=[400.0, 420.0],
        )
        annual, _ = compute_gross_margin(income)
        assert pd.isna(annual.iloc[0]["gross_margin"])
        assert annual.iloc[1]["gross_margin"] == pytest.approx(0.42)

    def test_summary_avg_gross_margin(self):
        income = _income_df(
            [2020, 2021, 2022], [0.0, 0.0, 0.0],
            total_revenue=[1_000.0, 1_000.0, 1_000.0],
            gross_profit=[400.0, 420.0, 440.0],
        )
        _, summary = compute_gross_margin(income)
        assert summary["avg_gross_margin"] == pytest.approx((0.40 + 0.42 + 0.44) / 3.0)

    def test_summary_min_gross_margin(self):
        income = _income_df(
            [2020, 2021, 2022], [0.0, 0.0, 0.0],
            total_revenue=[1_000.0, 1_000.0, 1_000.0],
            gross_profit=[400.0, 420.0, 440.0],
        )
        _, summary = compute_gross_margin(income)
        assert summary["min_gross_margin"] == pytest.approx(0.40)

    def test_all_invalid_revenue_produces_nan_summary(self):
        income = _income_df(
            [2020], [0.0],
            total_revenue=[0.0],
            gross_profit=[400.0],
        )
        _, summary = compute_gross_margin(income)
        assert pd.isna(summary["avg_gross_margin"])
        assert pd.isna(summary["min_gross_margin"])

    def test_annual_df_column_names(self):
        income = _income_df([2020], [0.0])
        annual, _ = compute_gross_margin(income)
        assert set(annual.columns) == {"fiscal_year", "gross_margin"}


# ===========================================================================
# Tests — compute_sga_ratio (F8)
# ===========================================================================

class TestComputeSGARatio:
    def test_zero_gross_profit_produces_nan(self):
        income = _income_df([2020], [0.0], gross_profit=[0.0], sga=[100.0])
        annual, _ = compute_sga_ratio(income)
        assert pd.isna(annual.iloc[0]["sga_ratio"])

    def test_negative_gross_profit_produces_nan(self):
        income = _income_df([2020], [0.0], gross_profit=[-50.0], sga=[100.0])
        annual, _ = compute_sga_ratio(income)
        assert pd.isna(annual.iloc[0]["sga_ratio"])

    def test_basic_known_values(self):
        income = _income_df(
            [2020, 2021], [0.0, 0.0],
            gross_profit=[1_000.0, 1_000.0],
            sga=[200.0, 250.0],
        )
        annual, summary = compute_sga_ratio(income)
        assert list(annual["sga_ratio"]) == pytest.approx([0.20, 0.25])
        assert summary["avg_sga_ratio"] == pytest.approx(0.225)

    def test_summary_avg_ignores_nan_years(self):
        """NaN year (gross_profit=0) excluded from average."""
        income = _income_df(
            [2020, 2021], [0.0, 0.0],
            gross_profit=[0.0, 1_000.0],
            sga=[100.0, 250.0],
        )
        _, summary = compute_sga_ratio(income)
        assert summary["avg_sga_ratio"] == pytest.approx(0.25)

    def test_annual_df_column_names(self):
        income = _income_df([2020], [0.0])
        annual, _ = compute_sga_ratio(income)
        assert set(annual.columns) == {"fiscal_year", "sga_ratio"}

    def test_low_sga_ratio_computation(self):
        """Excellent SGA ratio < 30%."""
        income = _income_df([2020], [0.0], gross_profit=[1_000.0], sga=[250.0])
        annual, summary = compute_sga_ratio(income)
        assert annual.iloc[0]["sga_ratio"] == pytest.approx(0.25)
        assert summary["avg_sga_ratio"] == pytest.approx(0.25)


# ===========================================================================
# Tests — compute_net_margin (F10)
# ===========================================================================

class TestComputeNetMargin:
    def test_basic_known_values(self):
        income = _income_df(
            [2020, 2021], [100.0, 200.0],
            total_revenue=[1_000.0, 1_000.0],
        )
        annual, _ = compute_net_margin(income)
        assert list(annual["net_margin"]) == pytest.approx([0.10, 0.20])

    def test_profitable_years_count_excludes_zero(self):
        """profitable_years counts net_income > 0; zero is NOT profitable."""
        income = _income_df(
            [2020, 2021, 2022, 2023],
            [100.0, -50.0, 0.0, 150.0],
            total_revenue=[1_000.0] * 4,
        )
        _, summary = compute_net_margin(income)
        # 2020 and 2023 are profitable; 2021 (negative) and 2022 (zero) are not.
        assert summary["profitable_years"] == 2

    def test_profitable_years_all_positive(self):
        income = _income_df(
            [2020, 2021, 2022], [100.0, 110.0, 120.0],
            total_revenue=[1_000.0] * 3,
        )
        _, summary = compute_net_margin(income)
        assert summary["profitable_years"] == 3

    def test_zero_revenue_excluded_from_margin(self):
        income = _income_df(
            [2020, 2021], [100.0, 200.0],
            total_revenue=[0.0, 1_000.0],
        )
        annual, _ = compute_net_margin(income)
        assert pd.isna(annual.iloc[0]["net_margin"])
        assert annual.iloc[1]["net_margin"] == pytest.approx(0.20)

    def test_avg_net_margin(self):
        income = _income_df(
            [2020, 2021], [100.0, 200.0],
            total_revenue=[1_000.0, 1_000.0],
        )
        _, summary = compute_net_margin(income)
        assert summary["avg_net_margin"] == pytest.approx(0.15)  # (0.10 + 0.20) / 2

    def test_trend_improving(self):
        """Clear upward trajectory in net margins → 'improving'."""
        income = _income_df(
            list(range(2020, 2030)),
            [100.0 * (1.15 ** i) for i in range(10)],
            total_revenue=[1_000.0] * 10,
        )
        _, summary = compute_net_margin(income)
        assert summary["trend"] == "improving"

    def test_trend_deteriorating(self):
        """Clear downward trajectory in net margins → 'deteriorating'."""
        income = _income_df(
            list(range(2020, 2030)),
            [500.0 - 50.0 * i for i in range(10)],   # 500, 450, 400, ..., 50
            total_revenue=[1_000.0] * 10,
        )
        _, summary = compute_net_margin(income)
        assert summary["trend"] == "deteriorating"

    def test_trend_stable(self):
        """Constant margins → 'stable'."""
        income = _income_df(
            list(range(2020, 2030)),
            [200.0] * 10,
            total_revenue=[1_000.0] * 10,
        )
        _, summary = compute_net_margin(income)
        assert summary["trend"] == "stable"

    def test_trend_single_year_is_stable(self):
        income = _income_df([2020], [100.0], total_revenue=[1_000.0])
        _, summary = compute_net_margin(income)
        assert summary["trend"] == "stable"

    def test_annual_df_column_names(self):
        income = _income_df([2020], [100.0])
        annual, _ = compute_net_margin(income)
        assert set(annual.columns) == {"fiscal_year", "net_margin"}

    def test_all_zero_revenue_produces_nan_avg(self):
        income = _income_df([2020, 2021], [100.0, 200.0], total_revenue=[0.0, 0.0])
        _, summary = compute_net_margin(income)
        assert pd.isna(summary["avg_net_margin"])


# ---------------------------------------------------------------------------
# Leverage builder helpers
# ---------------------------------------------------------------------------

def _balance_leverage_df(
    fiscal_years: list[int],
    long_term_debt: list[float],
    shareholders_equity: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal balance sheet DataFrame for leverage tests."""
    n = len(fiscal_years)
    return pd.DataFrame({
        "fiscal_year": fiscal_years,
        "long_term_debt": long_term_debt,
        "shareholders_equity": shareholders_equity if shareholders_equity is not None else [500.0] * n,
    })


def _coverage_df(
    fiscal_years: list[int],
    interest_expense: list[float],
    operating_income: list[float],
) -> pd.DataFrame:
    """Build a minimal income DataFrame for interest coverage tests."""
    return pd.DataFrame({
        "fiscal_year": fiscal_years,
        "interest_expense": interest_expense,
        "operating_income": operating_income,
    })


# ===========================================================================
# Tests — compute_debt_payoff (F5)
# ===========================================================================

class TestComputeDebtPayoff:
    def test_basic_four_year_payoff_passes(self):
        """debt=200, owner_earnings=50 → 4.0 years, pass=True (threshold=5)."""
        balance = _balance_leverage_df([2022], [200.0])
        oe = pd.Series({2022: 50.0})
        annual, summary = compute_debt_payoff(balance, oe)
        assert annual.iloc[0]["debt_payoff_years"] == pytest.approx(4.0)
        assert summary["debt_payoff_years"] == pytest.approx(4.0)
        assert summary["pass"] == True  # noqa: E712

    def test_zero_owner_earnings_produces_inf_and_fails(self):
        """owner_earnings=0 → debt_payoff=inf → pass=False."""
        balance = _balance_leverage_df([2022], [200.0])
        oe = pd.Series({2022: 0.0})
        annual, summary = compute_debt_payoff(balance, oe)
        assert math.isinf(annual.iloc[0]["debt_payoff_years"])
        assert math.isinf(summary["debt_payoff_years"])
        assert summary["pass"] == False  # noqa: E712

    def test_negative_owner_earnings_produces_inf_and_fails(self):
        """owner_earnings<0 → debt_payoff=inf → pass=False."""
        balance = _balance_leverage_df([2022], [200.0])
        oe = pd.Series({2022: -100.0})
        _, summary = compute_debt_payoff(balance, oe)
        assert math.isinf(summary["debt_payoff_years"])
        assert summary["pass"] == False  # noqa: E712

    def test_zero_debt_produces_zero_years_and_passes(self):
        """debt=0 → debt_payoff=0.0 → automatic pass."""
        balance = _balance_leverage_df([2022], [0.0])
        oe = pd.Series({2022: 50.0})
        annual, summary = compute_debt_payoff(balance, oe)
        assert annual.iloc[0]["debt_payoff_years"] == pytest.approx(0.0)
        assert summary["debt_payoff_years"] == pytest.approx(0.0)
        assert summary["pass"] == True  # noqa: E712

    def test_payoff_just_over_threshold_fails(self):
        """6 years payoff > 5-year threshold → pass=False."""
        balance = _balance_leverage_df([2022], [300.0])
        oe = pd.Series({2022: 50.0})
        _, summary = compute_debt_payoff(balance, oe)
        assert summary["debt_payoff_years"] == pytest.approx(6.0)
        assert summary["pass"] == False  # noqa: E712

    def test_summary_uses_most_recent_year(self):
        """Multi-year series: summary reflects the last fiscal year."""
        balance = _balance_leverage_df([2020, 2021, 2022], [100.0, 150.0, 200.0])
        oe = pd.Series({2020: 50.0, 2021: 50.0, 2022: 50.0})
        annual, summary = compute_debt_payoff(balance, oe)
        # Most recent year: 2022 → 200/50 = 4.0
        assert summary["debt_payoff_years"] == pytest.approx(4.0)
        assert annual.loc[annual["fiscal_year"] == 2020, "debt_payoff_years"].iloc[0] == pytest.approx(2.0)

    def test_missing_owner_earnings_year_produces_nan(self):
        """No owner earnings data for a year → NaN debt_payoff for that row."""
        balance = _balance_leverage_df([2020, 2021], [100.0, 200.0])
        oe = pd.Series({2020: 50.0})  # 2021 has no OE data
        annual, _ = compute_debt_payoff(balance, oe)
        assert annual.loc[annual["fiscal_year"] == 2020, "debt_payoff_years"].iloc[0] == pytest.approx(2.0)
        assert pd.isna(annual.loc[annual["fiscal_year"] == 2021, "debt_payoff_years"].iloc[0])

    def test_annual_df_column_names(self):
        balance = _balance_leverage_df([2022], [100.0])
        oe = pd.Series({2022: 50.0})
        annual, _ = compute_debt_payoff(balance, oe)
        assert set(annual.columns) == {"fiscal_year", "debt_payoff_years"}


# ===========================================================================
# Tests — compute_debt_to_equity (F6)
# ===========================================================================

class TestComputeDebtToEquity:
    def test_basic_known_value(self):
        """D/E = long_term_debt / shareholders_equity."""
        balance = _balance_leverage_df([2022], [80.0], shareholders_equity=[200.0])
        annual, _ = compute_debt_to_equity(balance)
        assert annual.iloc[0]["de_ratio"] == pytest.approx(0.40)

    def test_negative_equity_produces_nan_and_flag(self):
        """Non-positive equity → D/E = NaN, negative_equity = True."""
        balance = _balance_leverage_df(
            [2020, 2021],
            [100.0, 100.0],
            shareholders_equity=[-50.0, 500.0],
        )
        annual, _ = compute_debt_to_equity(balance)
        row_2020 = annual.loc[annual["fiscal_year"] == 2020].iloc[0]
        assert pd.isna(row_2020["de_ratio"])
        assert row_2020["negative_equity"] == True  # noqa: E712

    def test_positive_equity_not_flagged(self):
        balance = _balance_leverage_df([2022], [100.0], shareholders_equity=[500.0])
        annual, _ = compute_debt_to_equity(balance)
        assert annual.iloc[0]["negative_equity"] == False  # noqa: E712

    def test_zero_debt_produces_zero_de(self):
        """Zero debt → D/E = 0.0 regardless of equity."""
        balance = _balance_leverage_df([2022], [0.0], shareholders_equity=[500.0])
        annual, _ = compute_debt_to_equity(balance)
        assert annual.iloc[0]["de_ratio"] == pytest.approx(0.0)

    def test_zero_debt_with_negative_equity_still_zero_de(self):
        """Zero debt → D/E = 0.0 even when equity is negative."""
        balance = _balance_leverage_df([2022], [0.0], shareholders_equity=[-100.0])
        annual, _ = compute_debt_to_equity(balance)
        assert annual.iloc[0]["de_ratio"] == pytest.approx(0.0)
        assert annual.iloc[0]["negative_equity"] == True  # noqa: E712

    def test_summary_avg_de(self):
        balance = _balance_leverage_df(
            [2020, 2021, 2022],
            [100.0, 120.0, 80.0],
            shareholders_equity=[500.0, 400.0, 400.0],
        )
        _, summary = compute_debt_to_equity(balance)
        expected_avg = (100 / 500 + 120 / 400 + 80 / 400) / 3
        assert summary["avg_de_10yr"] == pytest.approx(expected_avg, rel=1e-4)

    def test_summary_max_de(self):
        balance = _balance_leverage_df(
            [2020, 2021],
            [50.0, 200.0],
            shareholders_equity=[500.0, 400.0],
        )
        _, summary = compute_debt_to_equity(balance)
        assert summary["max_de"] == pytest.approx(200.0 / 400.0)

    def test_summary_latest_de_uses_most_recent_year(self):
        balance = _balance_leverage_df(
            [2020, 2021, 2022],
            [100.0, 200.0, 50.0],
            shareholders_equity=[500.0, 500.0, 500.0],
        )
        _, summary = compute_debt_to_equity(balance)
        assert summary["latest_de"] == pytest.approx(50.0 / 500.0)

    def test_summary_nan_when_all_negative_equity(self):
        balance = _balance_leverage_df([2022], [100.0], shareholders_equity=[-50.0])
        _, summary = compute_debt_to_equity(balance)
        assert pd.isna(summary["avg_de_10yr"])
        assert pd.isna(summary["max_de"])

    def test_annual_df_column_names(self):
        balance = _balance_leverage_df([2022], [100.0])
        annual, _ = compute_debt_to_equity(balance)
        assert set(annual.columns) == {"fiscal_year", "de_ratio", "negative_equity"}


# ===========================================================================
# Tests — compute_interest_coverage (F9)
# ===========================================================================

class TestComputeInterestCoverage:
    def test_basic_known_value(self):
        """interest=10, operating_income=100 → interest_pct = 0.10."""
        income = _coverage_df([2022], [10.0], [100.0])
        annual, summary = compute_interest_coverage(income)
        assert annual.iloc[0]["interest_pct_of_ebit"] == pytest.approx(0.10)
        assert summary["avg_interest_pct_10yr"] == pytest.approx(0.10)

    def test_zero_interest_produces_zero_burden(self):
        """No interest expense → burden = 0.0."""
        income = _coverage_df([2022], [0.0], [100.0])
        annual, _ = compute_interest_coverage(income)
        assert annual.iloc[0]["interest_pct_of_ebit"] == pytest.approx(0.0)

    def test_negative_ebit_produces_nan(self):
        """operating_income ≤ 0 → NaN."""
        income = _coverage_df([2022], [10.0], [-50.0])
        annual, _ = compute_interest_coverage(income)
        assert pd.isna(annual.iloc[0]["interest_pct_of_ebit"])

    def test_zero_ebit_produces_nan(self):
        income = _coverage_df([2022], [10.0], [0.0])
        annual, _ = compute_interest_coverage(income)
        assert pd.isna(annual.iloc[0]["interest_pct_of_ebit"])

    def test_summary_avg_excludes_nan_years(self):
        """NaN year (bad ebit) excluded from average."""
        income = _coverage_df([2020, 2021], [10.0, 20.0], [-100.0, 200.0])
        _, summary = compute_interest_coverage(income)
        # Only 2021 is valid: 20/200 = 0.10
        assert summary["avg_interest_pct_10yr"] == pytest.approx(0.10)

    def test_summary_avg_multi_year(self):
        """Average of two valid years."""
        income = _coverage_df([2020, 2021], [10.0, 20.0], [100.0, 200.0])
        _, summary = compute_interest_coverage(income)
        # 2020: 10/100=0.10; 2021: 20/200=0.10 → avg=0.10
        assert summary["avg_interest_pct_10yr"] == pytest.approx(0.10)

    def test_negative_interest_value_uses_abs(self):
        """Negative-stored interest_expense is treated as absolute value."""
        income = _coverage_df([2022], [-10.0], [100.0])
        annual, _ = compute_interest_coverage(income)
        assert annual.iloc[0]["interest_pct_of_ebit"] == pytest.approx(0.10)

    def test_all_bad_ebit_produces_nan_avg(self):
        income = _coverage_df([2020, 2021], [10.0, 10.0], [-50.0, 0.0])
        _, summary = compute_interest_coverage(income)
        assert pd.isna(summary["avg_interest_pct_10yr"])

    def test_annual_df_column_names(self):
        income = _coverage_df([2022], [10.0], [100.0])
        annual, _ = compute_interest_coverage(income)
        assert set(annual.columns) == {"fiscal_year", "interest_pct_of_ebit"}


# ---------------------------------------------------------------------------
# Owner Earnings / Returns builder helpers
# ---------------------------------------------------------------------------

def _oe_income(fiscal_years: list[int], net_income: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"fiscal_year": fiscal_years, "net_income": net_income})


def _oe_cashflow(
    fiscal_years: list[int],
    da: list[float],
    capex: list[float],
    wc_change: list[float] | None = None,
) -> pd.DataFrame:
    data: dict = {
        "fiscal_year": fiscal_years,
        "depreciation_amortization": da,
        "capital_expenditures": capex,
    }
    if wc_change is not None:
        data["working_capital_change"] = wc_change
    return pd.DataFrame(data)


_EMPTY_BALANCE = pd.DataFrame()  # balance_df not used by owner_earnings


# ===========================================================================
# Tests — compute_owner_earnings (F1)
# ===========================================================================

class TestComputeOwnerEarnings:
    def test_basic_known_value_positive_capex(self):
        """net_income=100, da=20, capex=30 (positive, sign-corrected) → OE = 90."""
        income = _oe_income([2022], [100.0])
        cf = _oe_cashflow([2022], [20.0], [30.0])  # capex positive → sign-corrected to -30
        annual, _ = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        assert annual.iloc[0]["owner_earnings"] == pytest.approx(90.0)

    def test_basic_known_value_negative_capex(self):
        """capex already negative (schema convention) → same OE = 90."""
        income = _oe_income([2022], [100.0])
        cf = _oe_cashflow([2022], [20.0], [-30.0])  # capex = -30, no sign correction
        annual, _ = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        assert annual.iloc[0]["owner_earnings"] == pytest.approx(90.0)

    def test_high_capex_flag_triggered(self):
        """capex=50, da=20 → ratio=2.5 > 2.0 → high_capex_flag=True."""
        income = _oe_income([2022], [100.0])
        cf = _oe_cashflow([2022], [20.0], [50.0])
        _, summary = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        assert summary["high_capex_flag"] == True  # noqa: E712

    def test_high_capex_flag_not_triggered(self):
        """capex=30, da=20 → ratio=1.5 ≤ 2.0 → high_capex_flag=False."""
        income = _oe_income([2022], [100.0])
        cf = _oe_cashflow([2022], [20.0], [30.0])
        _, summary = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        assert summary["high_capex_flag"] == False  # noqa: E712

    def test_high_capex_flag_exactly_at_threshold_not_triggered(self):
        """capex = 2× da exactly → ratio=2.0, NOT > 2.0 → flag=False."""
        income = _oe_income([2022], [100.0])
        cf = _oe_cashflow([2022], [20.0], [40.0])  # 40/20 = 2.0, not > 2.0
        _, summary = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        assert summary["high_capex_flag"] == False  # noqa: E712

    def test_missing_da_produces_nan_oe(self):
        """NaN D&A → OE = NaN for that year."""
        income = _oe_income([2022], [100.0])
        cf = pd.DataFrame({
            "fiscal_year": [2022],
            "depreciation_amortization": [float("nan")],
            "capital_expenditures": [-30.0],
        })
        annual, _ = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        assert pd.isna(annual.iloc[0]["owner_earnings"])

    def test_negative_oe_is_valid(self):
        """OE can be negative; it means the business consumed capital."""
        income = _oe_income([2022], [-200.0])
        cf = _oe_cashflow([2022], [20.0], [30.0])
        annual, _ = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        # OE = -200 + 20 - 30 = -210
        assert annual.iloc[0]["owner_earnings"] == pytest.approx(-210.0)

    def test_wc_change_tracked_not_deducted(self):
        """wc_change is recorded in annual_df but does NOT alter owner_earnings."""
        income = _oe_income([2022], [100.0])
        cf = _oe_cashflow([2022], [20.0], [30.0], wc_change=[15.0])
        annual, _ = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        # OE = 100 + 20 - 30 = 90 (wc_change not deducted)
        assert annual.iloc[0]["owner_earnings"] == pytest.approx(90.0)
        assert annual.iloc[0]["wc_change"] == pytest.approx(15.0)

    def test_summary_avg_owner_earnings(self):
        """avg_owner_earnings_10yr is the mean of valid annual OE values."""
        income = _oe_income([2020, 2021, 2022], [100.0, 110.0, 120.0])
        cf = _oe_cashflow([2020, 2021, 2022], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0])
        # OE = [90, 100, 110] → avg = 100
        _, summary = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        assert summary["avg_owner_earnings_10yr"] == pytest.approx(100.0)

    def test_summary_cagr_multi_year(self):
        """owner_earnings_cagr: OE grows from 90 to 110 over 3 years."""
        income = _oe_income([2020, 2021, 2022], [100.0, 110.0, 120.0])
        cf = _oe_cashflow([2020, 2021, 2022], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0])
        # OE: 90, 100, 110 → CAGR = (110/90)^(1/2) - 1
        _, summary = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        expected_cagr = (110.0 / 90.0) ** 0.5 - 1.0
        assert summary["owner_earnings_cagr"] == pytest.approx(expected_cagr, rel=1e-4)

    def test_summary_cagr_single_year_is_nan(self):
        income = _oe_income([2022], [100.0])
        cf = _oe_cashflow([2022], [20.0], [30.0])
        _, summary = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        assert pd.isna(summary["owner_earnings_cagr"])

    def test_annual_df_column_names(self):
        income = _oe_income([2022], [100.0])
        cf = _oe_cashflow([2022], [20.0], [30.0])
        annual, _ = compute_owner_earnings(income, cf, _EMPTY_BALANCE)
        expected = {"fiscal_year", "owner_earnings", "net_income", "da", "capex", "wc_change", "capex_to_da_ratio"}
        assert set(annual.columns) == expected


# ===========================================================================
# Tests — compute_initial_rate_of_return (F2)
# ===========================================================================

class TestComputeInitialRateOfReturn:
    def test_basic_known_value(self):
        """OE_per_share=5, price=50 → initial_return = 0.10."""
        result = compute_initial_rate_of_return(
            owner_earnings_per_share=5.0,
            current_price=50.0,
            risk_free_rate=0.04,
        )
        assert result["initial_return"] == pytest.approx(0.10)

    def test_passes_2x_test_when_above_threshold(self):
        """0.10 ≥ 2 × 0.04 = 0.08 → passes_2x_test = True."""
        result = compute_initial_rate_of_return(5.0, 50.0, risk_free_rate=0.04)
        assert result["passes_2x_test"] == True  # noqa: E712

    def test_fails_2x_test_when_below_threshold(self):
        """0.10 < 2 × 0.06 = 0.12 → passes_2x_test = False."""
        result = compute_initial_rate_of_return(5.0, 50.0, risk_free_rate=0.06)
        assert result["passes_2x_test"] == False  # noqa: E712

    def test_vs_bond_yield_spread(self):
        """vs_bond_yield = initial_return − risk_free_rate."""
        result = compute_initial_rate_of_return(5.0, 50.0, risk_free_rate=0.04)
        assert result["vs_bond_yield"] == pytest.approx(0.10 - 0.04)

    def test_zero_price_returns_nan(self):
        result = compute_initial_rate_of_return(5.0, current_price=0.0, risk_free_rate=0.04)
        assert pd.isna(result["initial_return"])
        assert result["passes_2x_test"] == False  # noqa: E712

    def test_negative_price_returns_nan(self):
        result = compute_initial_rate_of_return(5.0, current_price=-10.0, risk_free_rate=0.04)
        assert pd.isna(result["initial_return"])

    def test_negative_oe_per_share_computes_negative_return(self):
        """Negative OE → negative return → automatic fail (no crash)."""
        result = compute_initial_rate_of_return(-5.0, 50.0, risk_free_rate=0.04)
        assert result["initial_return"] == pytest.approx(-0.10)
        assert result["passes_2x_test"] == False  # noqa: E712

    def test_return_dict_keys_present(self):
        result = compute_initial_rate_of_return(5.0, 50.0, 0.04)
        assert set(result.keys()) == {"initial_return", "vs_bond_yield", "passes_2x_test"}


# ===========================================================================
# Tests — compute_return_on_retained_earnings (F4)
# ===========================================================================

class TestComputeReturnOnRetainedEarnings:
    def test_basic_known_values(self):
        """eps=[2,2.5,3,3.5,4], divs=[0.5]*5 → cumulative=12.5, growth=2, return=0.16."""
        eps = pd.Series([2.0, 2.5, 3.0, 3.5, 4.0])
        dps = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
        result = compute_return_on_retained_earnings(eps, dps)
        assert result["cumulative_retained_per_share"] == pytest.approx(12.5)
        assert result["eps_growth"] == pytest.approx(2.0)
        assert result["return_on_retained"] == pytest.approx(0.16)
        assert result["meaningful"] == True  # noqa: E712

    def test_no_dividends_retains_all_eps(self):
        """Zero dividends → retained = eps for each year."""
        eps = pd.Series([1.0, 2.0, 3.0])
        dps = pd.Series([0.0, 0.0, 0.0])
        result = compute_return_on_retained_earnings(eps, dps)
        assert result["cumulative_retained_per_share"] == pytest.approx(6.0)
        assert result["eps_growth"] == pytest.approx(2.0)
        assert result["return_on_retained"] == pytest.approx(2.0 / 6.0)

    def test_cumulative_retained_zero_is_not_meaningful(self):
        """Paid out exactly as much as earned → cumulative ≤ 0 → NaN."""
        eps = pd.Series([2.0, 2.0])
        dps = pd.Series([2.0, 2.0])  # retained = 0 each year
        result = compute_return_on_retained_earnings(eps, dps)
        assert pd.isna(result["return_on_retained"])
        assert result["meaningful"] == False  # noqa: E712
        assert result["cumulative_retained_per_share"] == pytest.approx(0.0)

    def test_cumulative_retained_negative_is_not_meaningful(self):
        """Paid out more than earned → cumulative < 0 → NaN."""
        eps = pd.Series([1.0, 1.0])
        dps = pd.Series([2.0, 2.0])  # retained = -1 each year
        result = compute_return_on_retained_earnings(eps, dps)
        assert pd.isna(result["return_on_retained"])
        assert result["meaningful"] == False  # noqa: E712

    def test_eps_growth_uses_first_and_last(self):
        """eps_growth = eps_latest - eps_earliest regardless of middle values."""
        eps = pd.Series([2.0, 10.0, 4.0])  # first=2, last=4
        dps = pd.Series([0.0, 0.0, 0.0])
        result = compute_return_on_retained_earnings(eps, dps)
        assert result["eps_growth"] == pytest.approx(2.0)  # 4 - 2

    def test_single_year_eps_growth_is_nan(self):
        """Cannot compute growth with only one year."""
        eps = pd.Series([3.0])
        dps = pd.Series([0.5])
        result = compute_return_on_retained_earnings(eps, dps)
        assert pd.isna(result["eps_growth"])

    def test_return_dict_keys_present(self):
        eps = pd.Series([2.0, 4.0])
        dps = pd.Series([0.5, 0.5])
        result = compute_return_on_retained_earnings(eps, dps)
        assert set(result.keys()) == {
            "return_on_retained", "cumulative_retained_per_share", "eps_growth", "meaningful"
        }

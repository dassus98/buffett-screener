"""Unit tests for all pure financial formula functions in metrics_engine/ (no I/O, deterministic inputs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics_engine.profitability import (
    compute_gross_margin,
    compute_net_margin,
    compute_roe,
    compute_sga_ratio,
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

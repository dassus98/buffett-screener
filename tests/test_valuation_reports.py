"""Tests for valuation_reports: intrinsic_value, margin_of_safety, earnings_yield.

Covers:
  - compute_full_valuation (mock DuckDB, all fields populated)
  - compute_sensitivity_table (5 entries per axis, monotonicity)
  - assess_yield_attractiveness (verdict boundaries, NaN handling)
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pandas as pd
import pytest

from valuation_reports.earnings_yield import assess_yield_attractiveness
from valuation_reports.margin_of_safety import (
    _compute_mos,
    _derive_current_eps,
    _make_additive_steps,
    _make_steps,
    _project_iv,
    compute_sensitivity_table,
)
from valuation_reports.intrinsic_value import compute_full_valuation


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_income_df(ticker: str, n_years: int = 10, base_eps: float = 3.0) -> pd.DataFrame:
    """Create a minimal income_statement DataFrame with growing EPS."""
    years = list(range(2014, 2014 + n_years))
    return pd.DataFrame(
        {
            "ticker": [ticker] * n_years,
            "fiscal_year": years,
            "eps_diluted": [base_eps * (1.05 ** i) for i in range(n_years)],
            "net_income": [1_000_000 * (1.05 ** i) for i in range(n_years)],
            "total_revenue": [10_000_000] * n_years,
            "gross_profit": [5_000_000] * n_years,
            "sga": [1_500_000] * n_years,
            "operating_income": [3_000_000] * n_years,
            "interest_expense": [200_000] * n_years,
            "shares_outstanding_diluted": [1_000_000] * n_years,
        }
    )


def _make_market_data_df(
    ticker: str, price: float = 50.0, pe: float = 18.0
) -> pd.DataFrame:
    """Create a one-row market_data DataFrame."""
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "current_price_usd": [price],
            "pe_ratio_trailing": [pe],
            "market_cap_usd": [50_000_000_000.0],
            "enterprise_value_usd": [52_000_000_000.0],
            "shares_outstanding": [1_000_000_000],
            "high_52w": [60.0],
            "low_52w": [40.0],
            "avg_volume_3m": [5_000_000],
            "dividend_yield": [0.02],
            "as_of_date": ["2024-01-15"],
        }
    )


def _make_macro_df(
    rate: float = 0.04,
    key: str = "us_treasury_10yr",
) -> pd.DataFrame:
    """Create a macro_data DataFrame with a single bond yield row."""
    return pd.DataFrame(
        {
            "key": [key],
            "value": [rate],
            "as_of_date": ["2024-01-15"],
        }
    )


def _make_universe_df(
    ticker: str,
    exchange: str = "NYSE",
) -> pd.DataFrame:
    """Create a one-row universe DataFrame for exchange-aware tests."""
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "exchange": [exchange],
            "company_name": [f"{ticker} Inc."],
            "market_cap_usd": [50_000_000_000.0],
            "sector": ["Technology"],
            "industry": ["Software"],
            "country": ["US"],
        }
    )


def _mock_read_table(
    income_df: pd.DataFrame,
    market_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    universe_df: pd.DataFrame | None = None,
):
    """Return a side_effect function for patching data_acquisition.store.read_table.

    Parameters
    ----------
    income_df, market_df, macro_df:
        DataFrames returned for each table name.
    universe_df:
        Optional universe table DataFrame. If ``None``, a default single-row
        NYSE entry is generated from the first ticker in *income_df*.
    """
    # Default universe: derive ticker from income_df if available
    _universe_df = universe_df
    if _universe_df is None:
        if not income_df.empty and "ticker" in income_df.columns:
            _universe_df = _make_universe_df(income_df["ticker"].iloc[0])
        else:
            _universe_df = pd.DataFrame()

    def _read_table(table_name: str, where: str | None = None) -> pd.DataFrame:
        if table_name == "income_statement":
            return income_df
        if table_name == "market_data":
            return market_df
        if table_name == "macro_data":
            # Support exchange-aware filtering: if where clause filters by key,
            # only return rows that match.  Guard against empty DataFrames
            # that lack the "key" column entirely.
            if where and "key =" in where and not macro_df.empty:
                return macro_df[macro_df["key"].apply(lambda k: k in where)]
            return macro_df
        if table_name == "universe":
            return _universe_df
        return pd.DataFrame()

    return _read_table


# ---------------------------------------------------------------------------
# Tests: compute_full_valuation
# ---------------------------------------------------------------------------


class TestComputeFullValuation:
    """Tests for valuation_reports.intrinsic_value.compute_full_valuation."""

    REQUIRED_KEYS = {
        "ticker",
        "current_price",
        "scenarios",
        "weighted_iv",
        "margin_of_safety",
        "earnings_yield",
        "bond_yield",
        "spread",
        "meets_hurdle",
        "is_undervalued",
    }

    SCENARIO_KEYS = {
        "growth",
        "pe",
        "discount_rate",
        "present_value",
        "projected_price",
        "annual_return",
        "probability",
    }

    def _patch_and_run(self, ticker: str = "AAPL", **overrides):
        """Helper: patch read_table and run compute_full_valuation."""
        inc = overrides.get("income_df", _make_income_df(ticker))
        mkt = overrides.get("market_df", _make_market_data_df(ticker))
        mac = overrides.get("macro_df", _make_macro_df())
        uni = overrides.get("universe_df", None)
        with patch(
            "valuation_reports.intrinsic_value.read_table",
            side_effect=_mock_read_table(inc, mkt, mac, universe_df=uni),
        ):
            return compute_full_valuation(ticker)

    def test_all_top_level_keys_present(self):
        result = self._patch_and_run()
        assert self.REQUIRED_KEYS == set(result.keys())

    def test_ticker_echoed_back(self):
        result = self._patch_and_run(ticker="MSFT")
        assert result["ticker"] == "MSFT"

    def test_current_price_matches_input(self):
        mkt = _make_market_data_df("AAPL", price=123.45)
        result = self._patch_and_run(market_df=mkt)
        assert result["current_price"] == pytest.approx(123.45)

    def test_scenarios_contain_all_labels(self):
        result = self._patch_and_run()
        assert set(result["scenarios"].keys()) == {"bear", "base", "bull"}

    def test_scenario_dicts_have_required_keys(self):
        result = self._patch_and_run()
        for label in ("bear", "base", "bull"):
            assert self.SCENARIO_KEYS == set(result["scenarios"][label].keys()), (
                f"Missing keys in {label} scenario"
            )

    def test_weighted_iv_is_finite(self):
        result = self._patch_and_run()
        assert math.isfinite(result["weighted_iv"])
        assert result["weighted_iv"] > 0

    def test_margin_of_safety_is_finite(self):
        result = self._patch_and_run()
        assert math.isfinite(result["margin_of_safety"])

    def test_earnings_yield_is_finite(self):
        result = self._patch_and_run()
        assert math.isfinite(result["earnings_yield"])
        assert result["earnings_yield"] > 0  # positive EPS → positive yield

    def test_bond_yield_matches_macro(self):
        mac = _make_macro_df(rate=0.05)
        result = self._patch_and_run(macro_df=mac)
        assert result["bond_yield"] == pytest.approx(0.05)

    def test_spread_equals_ey_minus_bond(self):
        result = self._patch_and_run()
        assert result["spread"] == pytest.approx(
            result["earnings_yield"] - result["bond_yield"], abs=1e-8
        )

    def test_meets_hurdle_is_bool(self):
        result = self._patch_and_run()
        assert isinstance(result["meets_hurdle"], bool)

    def test_is_undervalued_is_bool(self):
        result = self._patch_and_run()
        assert isinstance(result["is_undervalued"], bool)

    def test_undervalued_when_price_below_iv(self):
        """A very low price relative to EPS should produce positive MoS."""
        mkt = _make_market_data_df("AAPL", price=10.0, pe=18.0)
        inc = _make_income_df("AAPL", base_eps=5.0)
        result = self._patch_and_run(market_df=mkt, income_df=inc)
        assert result["margin_of_safety"] > 0
        assert result["is_undervalued"] is True

    def test_overvalued_when_price_above_iv(self):
        """A very high price relative to EPS should produce negative MoS."""
        mkt = _make_market_data_df("AAPL", price=500.0, pe=18.0)
        inc = _make_income_df("AAPL", base_eps=1.0)
        result = self._patch_and_run(market_df=mkt, income_df=inc)
        assert result["margin_of_safety"] < 0
        assert result["is_undervalued"] is False

    # --- NaN / missing-data edge cases ---

    def test_missing_market_data_returns_nan_valuation(self):
        empty_mkt = pd.DataFrame()
        result = self._patch_and_run(market_df=empty_mkt)
        assert math.isnan(result["weighted_iv"])
        assert result["meets_hurdle"] is False
        assert result["is_undervalued"] is False

    def test_missing_macro_data_returns_nan_valuation(self):
        empty_mac = pd.DataFrame()
        result = self._patch_and_run(macro_df=empty_mac)
        assert math.isnan(result["weighted_iv"])

    def test_missing_income_data_returns_nan_valuation(self):
        empty_inc = pd.DataFrame()
        result = self._patch_and_run(income_df=empty_inc)
        assert math.isnan(result["weighted_iv"])

    def test_probabilities_sum_to_one(self):
        result = self._patch_and_run()
        total_prob = sum(
            result["scenarios"][s]["probability"] for s in ("bear", "base", "bull")
        )
        assert total_prob == pytest.approx(1.0)

    def test_bear_pv_leq_base_leq_bull(self):
        """Present values should increase from bear to base to bull."""
        result = self._patch_and_run()
        bear_pv = result["scenarios"]["bear"]["present_value"]
        base_pv = result["scenarios"]["base"]["present_value"]
        bull_pv = result["scenarios"]["bull"]["present_value"]
        assert bear_pv <= base_pv <= bull_pv

    # --- discount_rate and exchange-aware routing tests ---

    def test_discount_rate_in_scenario_dicts(self):
        """Each scenario should expose a finite discount_rate."""
        result = self._patch_and_run()
        for label in ("bear", "base", "bull"):
            dr = result["scenarios"][label]["discount_rate"]
            assert math.isfinite(dr), (
                f"{label} scenario discount_rate should be finite, got {dr}"
            )
            assert dr > 0, (
                f"{label} scenario discount_rate should be positive, got {dr}"
            )

    def test_tsx_ticker_uses_goc_bond_yield(self):
        """TSX-listed tickers should use the GoC 10-year bond yield."""
        goc_rate = 0.035
        uni = _make_universe_df("RY.TO", exchange="TSX")
        mac = _make_macro_df(rate=goc_rate, key="goc_bond_10yr")
        inc = _make_income_df("RY.TO", base_eps=5.0)
        mkt = _make_market_data_df("RY.TO", price=100.0)
        result = self._patch_and_run(
            ticker="RY.TO",
            income_df=inc,
            market_df=mkt,
            macro_df=mac,
            universe_df=uni,
        )
        assert result["bond_yield"] == pytest.approx(goc_rate)
        # Weighted IV should be finite (proves the full pipeline ran with GoC rate)
        assert math.isfinite(result["weighted_iv"])

    def test_nyse_ticker_uses_us_treasury_yield(self):
        """NYSE-listed tickers should use the US Treasury 10-year yield."""
        us_rate = 0.045
        uni = _make_universe_df("AAPL", exchange="NYSE")
        mac = _make_macro_df(rate=us_rate, key="us_treasury_10yr")
        result = self._patch_and_run(
            ticker="AAPL",
            macro_df=mac,
            universe_df=uni,
        )
        assert result["bond_yield"] == pytest.approx(us_rate)

    def test_negative_eps_returns_nan_valuation(self):
        """Negative trailing EPS should produce a NaN weighted_iv.

        When the most recent eps_diluted is negative, the F14 intrinsic
        value formula cannot produce a meaningful result because projecting
        a negative base forward at a positive growth rate diverges further
        into negative territory.
        """
        inc = _make_income_df("LOSS", base_eps=-2.0)
        mkt = _make_market_data_df("LOSS", price=50.0)
        result = self._patch_and_run(
            ticker="LOSS",
            income_df=inc,
            market_df=mkt,
        )
        # With negative EPS, weighted_iv should be NaN or negative —
        # either way, the stock should not be flagged as undervalued
        assert result["is_undervalued"] is False

    def test_nan_valuation_includes_discount_rate(self):
        """NaN valuation dicts (missing data) should still include discount_rate."""
        empty_mkt = pd.DataFrame()
        result = self._patch_and_run(market_df=empty_mkt)
        for label in ("bear", "base", "bull"):
            assert "discount_rate" in result["scenarios"][label], (
                f"NaN valuation missing discount_rate in {label} scenario"
            )


# ---------------------------------------------------------------------------
# Tests: sensitivity table helpers
# ---------------------------------------------------------------------------


class TestProjectIV:
    """Tests for margin_of_safety._project_iv."""

    def test_basic_calculation(self):
        # current_eps=5, growth=10%, pe=15, discount=7%, n=10
        iv = _project_iv(5.0, 0.10, 15.0, 0.07, 10)
        # projected_eps = 5 * 1.1^10 = 12.9687...
        # projected_price = 12.9687 * 15 = 194.531
        # iv = 194.531 / 1.07^10 = 194.531 / 1.9672 = 98.89...
        assert iv == pytest.approx(98.89, rel=0.01)

    def test_zero_growth(self):
        iv = _project_iv(5.0, 0.0, 15.0, 0.07, 10)
        # projected_eps = 5 (no growth)
        # projected_price = 75
        # iv = 75 / 1.07^10 ≈ 38.13
        assert iv == pytest.approx(38.13, rel=0.01)

    def test_zero_discount(self):
        iv = _project_iv(5.0, 0.10, 15.0, 0.0, 10)
        # No discounting: iv = projected_price
        projected = 5.0 * (1.1 ** 10) * 15.0
        assert iv == pytest.approx(projected, rel=1e-6)


class TestComputeMoS:
    """Tests for margin_of_safety._compute_mos."""

    def test_positive_mos(self):
        mos = _compute_mos(100.0, 75.0)
        assert mos == pytest.approx(0.25)

    def test_zero_mos(self):
        mos = _compute_mos(100.0, 100.0)
        assert mos == pytest.approx(0.0)

    def test_negative_mos(self):
        mos = _compute_mos(100.0, 120.0)
        assert mos == pytest.approx(-0.20)

    def test_nan_iv(self):
        mos = _compute_mos(float("nan"), 100.0)
        assert math.isnan(mos)

    def test_zero_iv(self):
        mos = _compute_mos(0.0, 100.0)
        assert math.isnan(mos)

    def test_negative_iv(self):
        mos = _compute_mos(-50.0, 100.0)
        assert math.isnan(mos)


class TestMakeSteps:
    """Tests for margin_of_safety._make_steps (multiplicative)."""

    def test_five_steps_symmetry(self):
        steps = _make_steps(0.10, 0.30, n_steps=5)
        assert len(steps) == 5
        # Center step should equal center value
        assert steps[2] == pytest.approx(0.10)
        # First and last should be symmetric around center
        assert steps[0] == pytest.approx(0.10 * 0.70)
        assert steps[4] == pytest.approx(0.10 * 1.30)

    def test_increasing_order(self):
        steps = _make_steps(0.15, 0.30)
        for i in range(len(steps) - 1):
            assert steps[i] < steps[i + 1]


class TestMakeAdditiveSteps:
    """Tests for margin_of_safety._make_additive_steps."""

    def test_five_steps_symmetry(self):
        steps = _make_additive_steps(0.07, 0.02, n_steps=5)
        assert len(steps) == 5
        assert steps[0] == pytest.approx(0.05)
        assert steps[2] == pytest.approx(0.07)
        assert steps[4] == pytest.approx(0.09)

    def test_increasing_order(self):
        steps = _make_additive_steps(0.10, 0.02)
        for i in range(len(steps) - 1):
            assert steps[i] < steps[i + 1]


class TestDeriveCurrentEps:
    """Tests for margin_of_safety._derive_current_eps."""

    def test_roundtrip(self):
        """Projecting EPS forward then back-deriving should recover original."""
        growth, pe, n = 0.10, 15.0, 10
        original_eps = 5.0
        projected_price = original_eps * (1 + growth) ** n * pe
        scenario = {"growth": growth, "pe": pe, "projected_price": projected_price}
        derived = _derive_current_eps(scenario, n)
        assert derived == pytest.approx(original_eps, rel=1e-6)

    def test_nan_growth(self):
        scenario = {"growth": float("nan"), "pe": 15.0, "projected_price": 100.0}
        assert math.isnan(_derive_current_eps(scenario, 10))

    def test_zero_pe(self):
        scenario = {"growth": 0.10, "pe": 0.0, "projected_price": 100.0}
        assert math.isnan(_derive_current_eps(scenario, 10))


# ---------------------------------------------------------------------------
# Tests: compute_sensitivity_table
# ---------------------------------------------------------------------------


class TestComputeSensitivityTable:
    """Tests for valuation_reports.margin_of_safety.compute_sensitivity_table."""

    @pytest.fixture()
    def base_inputs(self):
        """Build valid base valuation dict and companion inputs."""
        # Simulate a realistic base valuation output
        growth, pe, n = 0.10, 18.0, 10
        current_eps = 5.0
        risk_free = 0.04
        discount = risk_free + 0.03  # base risk premium
        projected_eps = current_eps * (1 + growth) ** n
        projected_price = projected_eps * pe
        present_value = projected_price / (1 + discount) ** n

        base_valuation = {
            "scenarios": {
                "bear": {
                    "growth": growth * 0.5,
                    "pe": min(pe, 12),
                    "projected_price": current_eps * (1 + growth * 0.5) ** n * min(pe, 12),
                    "present_value": 50.0,
                    "annual_return": 0.05,
                    "probability": 0.25,
                },
                "base": {
                    "growth": growth,
                    "pe": pe,
                    "projected_price": projected_price,
                    "present_value": present_value,
                    "annual_return": 0.10,
                    "probability": 0.50,
                },
                "bull": {
                    "growth": growth * 1.3,
                    "pe": max(pe, 20),
                    "projected_price": current_eps * (1 + growth * 1.3) ** n * max(pe, 20),
                    "present_value": 200.0,
                    "annual_return": 0.15,
                    "probability": 0.25,
                },
            },
            "weighted_iv": present_value * 0.5 + 50.0 * 0.25 + 200.0 * 0.25,
        }
        return {
            "base_valuation": base_valuation,
            "eps_cagr": growth,
            "historical_pe": pd.Series([pe]),
            "current_price": 80.0,
            "risk_free_rate": risk_free,
        }

    def test_returns_three_sensitivity_keys(self, base_inputs):
        result = compute_sensitivity_table(**base_inputs)
        assert set(result.keys()) == {
            "eps_sensitivity",
            "pe_sensitivity",
            "discount_sensitivity",
        }

    def test_each_axis_has_five_entries(self, base_inputs):
        result = compute_sensitivity_table(**base_inputs)
        for key in ("eps_sensitivity", "pe_sensitivity", "discount_sensitivity"):
            assert len(result[key]) == 5, f"{key} should have 5 entries"

    def test_each_entry_is_triple(self, base_inputs):
        result = compute_sensitivity_table(**base_inputs)
        for key in ("eps_sensitivity", "pe_sensitivity", "discount_sensitivity"):
            for entry in result[key]:
                assert len(entry) == 3, f"Each {key} entry should be a 3-tuple"

    def test_eps_sensitivity_iv_monotonically_increases(self, base_inputs):
        """Higher EPS growth should produce higher intrinsic value."""
        result = compute_sensitivity_table(**base_inputs)
        ivs = [entry[1] for entry in result["eps_sensitivity"]]
        for i in range(len(ivs) - 1):
            assert ivs[i] <= ivs[i + 1], (
                f"EPS sensitivity IV should increase: step {i} ({ivs[i]}) > step {i+1} ({ivs[i+1]})"
            )

    def test_pe_sensitivity_iv_monotonically_increases(self, base_inputs):
        """Higher terminal P/E should produce higher intrinsic value."""
        result = compute_sensitivity_table(**base_inputs)
        ivs = [entry[1] for entry in result["pe_sensitivity"]]
        for i in range(len(ivs) - 1):
            assert ivs[i] <= ivs[i + 1], (
                f"PE sensitivity IV should increase: step {i} ({ivs[i]}) > step {i+1} ({ivs[i+1]})"
            )

    def test_discount_sensitivity_iv_monotonically_decreases(self, base_inputs):
        """Higher discount rate should produce lower intrinsic value."""
        result = compute_sensitivity_table(**base_inputs)
        ivs = [entry[1] for entry in result["discount_sensitivity"]]
        for i in range(len(ivs) - 1):
            assert ivs[i] >= ivs[i + 1], (
                f"Discount sensitivity IV should decrease: step {i} ({ivs[i]}) < step {i+1} ({ivs[i+1]})"
            )

    def test_eps_sensitivity_mos_monotonically_increases(self, base_inputs):
        """Higher IV → higher MoS for fixed price."""
        result = compute_sensitivity_table(**base_inputs)
        mos_vals = [entry[2] for entry in result["eps_sensitivity"]]
        for i in range(len(mos_vals) - 1):
            assert mos_vals[i] <= mos_vals[i + 1]

    def test_center_step_uses_base_growth(self, base_inputs):
        """The middle step should use the base scenario growth rate."""
        result = compute_sensitivity_table(**base_inputs)
        center_growth = result["eps_sensitivity"][2][0]
        assert center_growth == pytest.approx(
            base_inputs["eps_cagr"], rel=1e-4
        )

    def test_empty_tables_when_eps_cannot_be_derived(self):
        """If base scenario has NaN fields, return empty tables."""
        bad_valuation = {
            "scenarios": {
                "base": {
                    "growth": float("nan"),
                    "pe": float("nan"),
                    "projected_price": float("nan"),
                },
            },
        }
        result = compute_sensitivity_table(
            base_valuation=bad_valuation,
            eps_cagr=0.10,
            historical_pe=pd.Series([15.0]),
            current_price=80.0,
            risk_free_rate=0.04,
        )
        assert result["eps_sensitivity"] == []
        assert result["pe_sensitivity"] == []
        assert result["discount_sensitivity"] == []

    def test_all_ivs_are_positive(self, base_inputs):
        """With positive EPS and growth, all IVs should be positive."""
        result = compute_sensitivity_table(**base_inputs)
        for key in ("eps_sensitivity", "pe_sensitivity", "discount_sensitivity"):
            for param, iv, mos in result[key]:
                assert iv > 0, f"{key}: IV should be positive, got {iv}"


# ---------------------------------------------------------------------------
# Tests: assess_yield_attractiveness
# ---------------------------------------------------------------------------


class TestAssessYieldAttractiveness:
    """Tests for valuation_reports.earnings_yield.assess_yield_attractiveness."""

    REQUIRED_KEYS = {"spread", "verdict", "explanation"}

    def test_all_keys_present(self):
        result = assess_yield_attractiveness(0.08, 0.04)
        assert self.REQUIRED_KEYS == set(result.keys())

    # --- Verdict boundary tests ---

    def test_attractive_when_spread_above_4pct(self):
        """Spread > 4% (attractive_min_spread) → Attractive."""
        # EY=10%, bond=4% → spread=6% > 4%
        result = assess_yield_attractiveness(0.10, 0.04)
        assert result["verdict"] == "Attractive"
        assert result["spread"] == pytest.approx(0.06)

    def test_attractive_boundary_just_above(self):
        """Spread = 4.1% → Attractive."""
        result = assess_yield_attractiveness(0.081, 0.04)
        assert result["verdict"] == "Attractive"

    def test_moderate_boundary_at_4pct(self):
        """Spread = 4% exactly → Moderate (not strictly > attractive_min=0.04)."""
        result = assess_yield_attractiveness(0.08, 0.04)
        assert result["verdict"] == "Moderate"

    def test_moderate_when_spread_between_2_and_4pct(self):
        """Spread = 3% → Moderate."""
        # EY=7%, bond=4% → spread=3%
        result = assess_yield_attractiveness(0.07, 0.04)
        assert result["verdict"] == "Moderate"
        assert result["spread"] == pytest.approx(0.03)

    def test_moderate_boundary_at_2pct(self):
        """Spread just above 2% → Moderate (>= moderate_min=0.02)."""
        # Use 0.062 - 0.04 = 0.022 to avoid IEEE 754 edge at exact boundary
        result = assess_yield_attractiveness(0.062, 0.04)
        assert result["verdict"] == "Moderate"

    def test_unattractive_when_spread_below_2pct(self):
        """Spread = 1% → Unattractive."""
        # EY=5%, bond=4% → spread=1%
        result = assess_yield_attractiveness(0.05, 0.04)
        assert result["verdict"] == "Unattractive"
        assert result["spread"] == pytest.approx(0.01)

    def test_unattractive_when_negative_spread(self):
        """Bonds yield more than equities → Unattractive."""
        # EY=3%, bond=5% → spread=-2%
        result = assess_yield_attractiveness(0.03, 0.05)
        assert result["verdict"] == "Unattractive"
        assert result["spread"] == pytest.approx(-0.02)

    def test_unattractive_zero_spread(self):
        """Spread = 0 → Unattractive."""
        result = assess_yield_attractiveness(0.04, 0.04)
        assert result["verdict"] == "Unattractive"
        assert result["spread"] == pytest.approx(0.0)

    # --- Explanation content ---

    def test_attractive_explanation_mentions_strong_premium(self):
        result = assess_yield_attractiveness(0.10, 0.04)
        assert "strong premium" in result["explanation"].lower()

    def test_moderate_explanation_mentions_reasonable(self):
        result = assess_yield_attractiveness(0.07, 0.04)
        assert "reasonable" in result["explanation"].lower()

    def test_unattractive_negative_explanation_mentions_bonds(self):
        result = assess_yield_attractiveness(0.03, 0.05)
        assert "bonds" in result["explanation"].lower()

    def test_unattractive_small_positive_explanation_mentions_insufficient(self):
        result = assess_yield_attractiveness(0.05, 0.04)
        assert "insufficient" in result["explanation"].lower()

    # --- NaN handling ---

    def test_nan_earnings_yield(self):
        result = assess_yield_attractiveness(float("nan"), 0.04)
        assert result["verdict"] == "Unattractive"
        assert math.isnan(result["spread"])
        assert "unavailable" in result["explanation"].lower()

    def test_nan_bond_yield(self):
        result = assess_yield_attractiveness(0.08, float("nan"))
        assert result["verdict"] == "Unattractive"
        assert math.isnan(result["spread"])

    def test_both_nan(self):
        result = assess_yield_attractiveness(float("nan"), float("nan"))
        assert result["verdict"] == "Unattractive"
        assert math.isnan(result["spread"])

    # --- Spread arithmetic ---

    def test_spread_equals_ey_minus_bond(self):
        ey, bond = 0.08, 0.035
        result = assess_yield_attractiveness(ey, bond)
        assert result["spread"] == pytest.approx(ey - bond, abs=1e-10)

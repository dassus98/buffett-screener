"""Tests for valuation_reports.recommendation module.

Covers:
  - generate_recommendation (BUY / HOLD / PASS tiers, confidence levels)
  - recommend_account (RRSP / TFSA / Either logic)
  - generate_sell_signals (OK / WARNING / TRIGGERED classification)
  - generate_entry_strategy (ideal entry, narrative)
"""

from __future__ import annotations

import math

import pytest

from valuation_reports.recommendation import (
    generate_entry_strategy,
    generate_recommendation,
    generate_sell_signals,
    recommend_account,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _quality(
    years: int = 10,
    subs: int = 0,
    drop: bool = False,
    drop_reason: str = "",
) -> dict:
    """Build a minimal data_quality dict."""
    return {
        "years_available": years,
        "substitutions_count": subs,
        "drop": drop,
        "drop_reason": drop_reason,
    }


def _valuation(
    weighted_iv: float = 100.0,
    current_price: float = 75.0,
) -> dict:
    """Build a minimal valuation dict."""
    return {
        "weighted_iv": weighted_iv,
        "current_price": current_price,
    }


# ---------------------------------------------------------------------------
# Tests: generate_recommendation
# ---------------------------------------------------------------------------


class TestGenerateRecommendation:
    """Tests for recommendation tier and confidence classification."""

    def test_buy_when_mos_and_score_sufficient(self):
        """score=75, mos=0.30 → BUY (per REPORT_SPEC: mos≥0.25, score≥70)."""
        result = generate_recommendation(
            ticker="AAPL",
            composite_score=75.0,
            margin_of_safety=0.30,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "BUY"

    def test_hold_when_below_buy_thresholds(self):
        """score=65, mos=0.15 → HOLD (mos≥0.10, score≥60 but < buy)."""
        result = generate_recommendation(
            ticker="MSFT",
            composite_score=65.0,
            margin_of_safety=0.15,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "HOLD"

    def test_pass_when_below_hold_thresholds(self):
        """score=50, mos=0.05 → PASS (below hold minimums)."""
        result = generate_recommendation(
            ticker="XYZ",
            composite_score=50.0,
            margin_of_safety=0.05,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "PASS"

    def test_pass_when_critical_flags(self):
        """Even with strong score/mos, critical flags → PASS."""
        result = generate_recommendation(
            ticker="BAD",
            composite_score=90.0,
            margin_of_safety=0.50,
            data_quality=_quality(drop=True, drop_reason="Missing critical fields"),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "PASS"
        assert "data confidence" in result["reasoning"].lower()

    def test_pass_when_nan_mos(self):
        result = generate_recommendation(
            ticker="NAN1",
            composite_score=80.0,
            margin_of_safety=float("nan"),
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "PASS"

    def test_pass_when_nan_score(self):
        result = generate_recommendation(
            ticker="NAN2",
            composite_score=float("nan"),
            margin_of_safety=0.30,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "PASS"

    def test_buy_at_exact_boundary(self):
        """score=70, mos=0.25 → BUY (at exact thresholds)."""
        result = generate_recommendation(
            ticker="EDGE",
            composite_score=70.0,
            margin_of_safety=0.25,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "BUY"

    def test_hold_at_exact_boundary(self):
        """score=60, mos=0.10 → HOLD."""
        result = generate_recommendation(
            ticker="EDGE2",
            composite_score=60.0,
            margin_of_safety=0.10,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "HOLD"

    def test_hold_high_score_low_mos(self):
        """score=85 but mos=0.15 → HOLD (mos below buy threshold)."""
        result = generate_recommendation(
            ticker="HI",
            composite_score=85.0,
            margin_of_safety=0.15,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "HOLD"

    def test_pass_high_mos_low_score(self):
        """mos=0.40 but score=55 → PASS (score below hold threshold)."""
        result = generate_recommendation(
            ticker="LO",
            composite_score=55.0,
            margin_of_safety=0.40,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert result["recommendation"] == "PASS"

    # --- Return shape ---

    def test_result_has_required_keys(self):
        result = generate_recommendation(
            ticker="KEYS",
            composite_score=75.0,
            margin_of_safety=0.30,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert set(result.keys()) == {"recommendation", "confidence", "reasoning"}

    def test_reasoning_is_nonempty_string(self):
        result = generate_recommendation(
            ticker="RES",
            composite_score=75.0,
            margin_of_safety=0.30,
            data_quality=_quality(),
            valuation=_valuation(),
        )
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0


class TestConfidenceLevel:
    """Tests for confidence classification within generate_recommendation."""

    def test_high_confidence(self):
        """10 years, 0 subs, mos > 0.30 → High."""
        result = generate_recommendation(
            ticker="HC",
            composite_score=75.0,
            margin_of_safety=0.35,
            data_quality=_quality(years=10, subs=0),
            valuation=_valuation(),
        )
        assert result["confidence"] == "High"

    def test_moderate_confidence(self):
        """10 years, 1 sub, mos = 0.20 → Moderate."""
        result = generate_recommendation(
            ticker="MC",
            composite_score=65.0,
            margin_of_safety=0.20,
            data_quality=_quality(years=10, subs=1),
            valuation=_valuation(),
        )
        assert result["confidence"] == "Moderate"

    def test_low_confidence_few_years(self):
        """5 years → Low regardless of subs/mos."""
        result = generate_recommendation(
            ticker="LC",
            composite_score=75.0,
            margin_of_safety=0.35,
            data_quality=_quality(years=5, subs=0),
            valuation=_valuation(),
        )
        assert result["confidence"] == "Low"

    def test_low_confidence_many_subs(self):
        """10 years but 5 substitutions → Low."""
        result = generate_recommendation(
            ticker="LS",
            composite_score=75.0,
            margin_of_safety=0.35,
            data_quality=_quality(years=10, subs=5),
            valuation=_valuation(),
        )
        assert result["confidence"] == "Low"

    def test_low_confidence_thin_mos(self):
        """10 years, 0 subs, but mos = 0.05 → Low."""
        result = generate_recommendation(
            ticker="TM",
            composite_score=50.0,
            margin_of_safety=0.05,
            data_quality=_quality(years=10, subs=0),
            valuation=_valuation(),
        )
        assert result["confidence"] == "Low"

    def test_low_confidence_nan_mos(self):
        result = generate_recommendation(
            ticker="NAN",
            composite_score=75.0,
            margin_of_safety=float("nan"),
            data_quality=_quality(years=10, subs=0),
            valuation=_valuation(),
        )
        assert result["confidence"] == "Low"

    def test_high_confidence_at_boundary(self):
        """8 years, 0 subs, mos = 0.31 → High (> 0.30 required)."""
        result = generate_recommendation(
            ticker="BND",
            composite_score=75.0,
            margin_of_safety=0.31,
            data_quality=_quality(years=8, subs=0),
            valuation=_valuation(),
        )
        assert result["confidence"] == "High"

    def test_moderate_not_high_at_mos_30(self):
        """mos = 0.30 exactly → NOT High (requires > 0.30), should be Moderate."""
        result = generate_recommendation(
            ticker="MOD",
            composite_score=75.0,
            margin_of_safety=0.30,
            data_quality=_quality(years=10, subs=0),
            valuation=_valuation(),
        )
        # High requires MoS > 0.30, so 0.30 falls to Moderate (>= 0.15)
        assert result["confidence"] == "Moderate"


# ---------------------------------------------------------------------------
# Tests: recommend_account
# ---------------------------------------------------------------------------


class TestRecommendAccount:
    """Tests for RRSP vs TFSA account recommendation logic."""

    REQUIRED_KEYS = {"account", "reasoning"}

    def test_us_ticker_high_dividend_rrsp(self):
        """US-listed + dividend yield ≥ 1% → RRSP."""
        result = recommend_account(
            ticker="KO",
            exchange="NYSE",
            dividend_yield=0.03,
            expected_return=0.10,
        )
        assert result["account"] == "RRSP"
        assert "withholding" in result["reasoning"].lower()

    def test_us_ticker_no_dividend_either(self):
        """US-listed + no dividend → Either."""
        result = recommend_account(
            ticker="AMZN",
            exchange="NASDAQ",
            dividend_yield=0.0,
            expected_return=0.15,
        )
        assert result["account"] == "Either"

    def test_canadian_ticker_tfsa(self):
        """TSX-listed → TFSA."""
        result = recommend_account(
            ticker="RY.TO",
            exchange="TSX",
            dividend_yield=0.04,
            expected_return=0.10,
        )
        assert result["account"] == "TFSA"

    def test_to_suffix_detected_as_canadian(self):
        """.TO suffix → Canadian → TFSA, even if exchange not explicitly TSX."""
        result = recommend_account(
            ticker="ENB.TO",
            exchange="",
            dividend_yield=0.07,
            expected_return=0.08,
        )
        assert result["account"] == "TFSA"

    def test_us_dividend_at_boundary(self):
        """Dividend yield = 0.01 exactly → RRSP (≥ threshold)."""
        result = recommend_account(
            ticker="JNJ",
            exchange="NYSE",
            dividend_yield=0.01,
            expected_return=0.10,
        )
        assert result["account"] == "RRSP"

    def test_us_dividend_below_boundary(self):
        """Dividend yield = 0.005 → Either (< threshold)."""
        result = recommend_account(
            ticker="BRK.B",
            exchange="NYSE",
            dividend_yield=0.005,
            expected_return=0.12,
        )
        assert result["account"] == "Either"

    def test_nan_dividend_treated_as_zero(self):
        """NaN dividend yield → no dividend → Either for US stocks."""
        result = recommend_account(
            ticker="GOOG",
            exchange="NASDAQ",
            dividend_yield=float("nan"),
            expected_return=0.15,
        )
        assert result["account"] == "Either"

    def test_result_has_required_keys(self):
        result = recommend_account("AAPL", "NASDAQ", 0.005, 0.12)
        assert self.REQUIRED_KEYS == set(result.keys())

    def test_reasoning_is_nonempty(self):
        result = recommend_account("AAPL", "NYSE", 0.02, 0.10)
        assert len(result["reasoning"]) > 0

    def test_canadian_growth_stock_tfsa(self):
        """Canadian, zero dividend, growth → TFSA."""
        result = recommend_account(
            ticker="SHOP.TO",
            exchange="TSX",
            dividend_yield=0.0,
            expected_return=0.20,
        )
        assert result["account"] == "TFSA"


# ---------------------------------------------------------------------------
# Tests: generate_sell_signals
# ---------------------------------------------------------------------------


class TestGenerateSellSignals:
    """Tests for sell-signal evaluation."""

    def _base_metrics(self, **overrides) -> dict:
        """Build a healthy baseline metrics_summary dict."""
        base = {
            "avg_roe_10yr": 0.20,
            "gross_margin_avg_10yr": 0.45,
            "de_ratio_latest": 0.30,
            "debt_payoff_years": 2.0,
            "return_on_retained_earnings": 0.15,
            "bull_present_value": 150.0,
            "current_price": 80.0,
        }
        base.update(overrides)
        return base

    def test_all_ok_for_healthy_metrics(self):
        """Strong metrics → all signals OK."""
        signals = generate_sell_signals("AAPL", self._base_metrics())
        statuses = {s["signal"]: s["status"] for s in signals}
        assert all(v == "OK" for v in statuses.values()), statuses

    def test_returns_five_signals(self):
        signals = generate_sell_signals("AAPL", self._base_metrics())
        assert len(signals) == 5

    def test_signal_dict_has_required_keys(self):
        signals = generate_sell_signals("AAPL", self._base_metrics())
        required = {"signal", "current_value", "threshold", "status", "description"}
        for sig in signals:
            assert required == set(sig.keys()), f"{sig['signal']} missing keys"

    def test_roe_warning_when_approaching(self):
        """ROE at 0.13 (above 0.12 floor but within 20% = 0.024) → WARNING."""
        signals = generate_sell_signals(
            "WARN", self._base_metrics(avg_roe_10yr=0.13),
        )
        roe_sig = next(s for s in signals if s["signal"] == "roe_deterioration")
        assert roe_sig["status"] == "WARNING"
        assert roe_sig["current_value"] == pytest.approx(0.13)

    def test_roe_triggered_when_below_floor(self):
        """ROE at 0.10 (below 0.12) → TRIGGERED."""
        signals = generate_sell_signals(
            "TRIG", self._base_metrics(avg_roe_10yr=0.10),
        )
        roe_sig = next(s for s in signals if s["signal"] == "roe_deterioration")
        assert roe_sig["status"] == "TRIGGERED"

    def test_roe_ok_when_well_above(self):
        """ROE at 0.25 (well above 0.12) → OK."""
        signals = generate_sell_signals(
            "OK", self._base_metrics(avg_roe_10yr=0.25),
        )
        roe_sig = next(s for s in signals if s["signal"] == "roe_deterioration")
        assert roe_sig["status"] == "OK"

    def test_leverage_triggered_when_de_above_max(self):
        """D/E at 1.5 (above 1.0 max) → TRIGGERED."""
        signals = generate_sell_signals(
            "LEV", self._base_metrics(de_ratio_latest=1.5),
        )
        lev_sig = next(s for s in signals if s["signal"] == "leverage_spike")
        assert lev_sig["status"] == "TRIGGERED"

    def test_leverage_warning_when_approaching(self):
        """D/E at 0.85 (within 20% of 1.0 = within 0.80-1.00) → WARNING."""
        signals = generate_sell_signals(
            "LWARN", self._base_metrics(de_ratio_latest=0.85),
        )
        lev_sig = next(s for s in signals if s["signal"] == "leverage_spike")
        assert lev_sig["status"] == "WARNING"

    def test_overvaluation_triggered(self):
        """Price > bull IV → TRIGGERED."""
        signals = generate_sell_signals(
            "OVER",
            self._base_metrics(bull_present_value=100.0, current_price=120.0),
        )
        over_sig = next(s for s in signals if s["signal"] == "overvaluation")
        assert over_sig["status"] == "TRIGGERED"

    def test_overvaluation_ok_when_well_below(self):
        """Price well below bull IV → OK."""
        signals = generate_sell_signals(
            "UNDER",
            self._base_metrics(bull_present_value=200.0, current_price=80.0),
        )
        over_sig = next(s for s in signals if s["signal"] == "overvaluation")
        assert over_sig["status"] == "OK"

    def test_capital_misallocation_triggered(self):
        """RORE at 0.05 (below 0.08) → TRIGGERED."""
        signals = generate_sell_signals(
            "CAP", self._base_metrics(return_on_retained_earnings=0.05),
        )
        cap_sig = next(s for s in signals if s["signal"] == "capital_misallocation")
        assert cap_sig["status"] == "TRIGGERED"

    def test_capital_misallocation_warning(self):
        """RORE at 0.09 (above 0.08 but within 20% buffer) → WARNING."""
        signals = generate_sell_signals(
            "CAPW", self._base_metrics(return_on_retained_earnings=0.09),
        )
        cap_sig = next(s for s in signals if s["signal"] == "capital_misallocation")
        assert cap_sig["status"] == "WARNING"

    def test_nan_metric_defaults_to_ok(self):
        """NaN value → OK (data unavailable, not triggered)."""
        signals = generate_sell_signals(
            "NAN", self._base_metrics(avg_roe_10yr=float("nan")),
        )
        roe_sig = next(s for s in signals if s["signal"] == "roe_deterioration")
        assert roe_sig["status"] == "OK"

    def test_signal_names_are_expected(self):
        """Verify the five expected signal names."""
        signals = generate_sell_signals("AAPL", self._base_metrics())
        names = {s["signal"] for s in signals}
        assert names == {
            "roe_deterioration",
            "gross_margin_erosion",
            "leverage_spike",
            "overvaluation",
            "capital_misallocation",
        }


# ---------------------------------------------------------------------------
# Tests: generate_entry_strategy
# ---------------------------------------------------------------------------


class TestGenerateEntryStrategy:
    """Tests for entry strategy computation."""

    REQUIRED_KEYS = {"current_price", "ideal_entry", "discount_needed_pct", "strategy"}

    def test_result_has_required_keys(self):
        result = generate_entry_strategy(80.0, 100.0, 0.20)
        assert self.REQUIRED_KEYS == set(result.keys())

    def test_ideal_entry_uses_buy_min_mos(self):
        """ideal_entry = weighted_iv × (1 − 0.25) = 100 × 0.75 = 75."""
        result = generate_entry_strategy(80.0, 100.0, 0.20)
        assert result["ideal_entry"] == pytest.approx(75.0)

    def test_above_ideal_entry_narrative(self):
        """Price $80 > ideal $75 → narrative mentions 'above'."""
        result = generate_entry_strategy(80.0, 100.0, 0.20)
        assert "above" in result["strategy"].lower()

    def test_at_or_below_ideal_entry_narrative(self):
        """Price $60 < ideal $75 → narrative mentions 'buy zone'."""
        result = generate_entry_strategy(60.0, 100.0, 0.40)
        assert "buy zone" in result["strategy"].lower()

    def test_discount_needed_positive_when_above(self):
        """Price above ideal → discount_needed > 0."""
        result = generate_entry_strategy(80.0, 100.0, 0.20)
        assert result["discount_needed_pct"] > 0

    def test_discount_needed_negative_when_below(self):
        """Price below ideal → discount_needed < 0 (already cheap)."""
        result = generate_entry_strategy(60.0, 100.0, 0.40)
        assert result["discount_needed_pct"] < 0

    def test_nan_iv_returns_nan_entry(self):
        result = generate_entry_strategy(80.0, float("nan"), 0.20)
        assert math.isnan(result["ideal_entry"])
        assert "unavailable" in result["strategy"].lower()

    def test_nan_price_returns_nan_discount(self):
        result = generate_entry_strategy(float("nan"), 100.0, 0.20)
        assert math.isnan(result["discount_needed_pct"])

    def test_current_price_echoed(self):
        result = generate_entry_strategy(123.45, 200.0, 0.38)
        assert result["current_price"] == pytest.approx(123.45)

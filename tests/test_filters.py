"""Tests for screener/hard_filters.py and screener/soft_filters.py.

Covers:
- Hard filter ROE threshold (10 % fails, 20 % passes, 15 % boundary)
- Hard filter earnings consistency (6 profitable years fails, 8 passes)
- All 5 filters applied — no filter is skipped
- NaN handling — NaN on any metric -> fail
- Edge cases: EPS CAGR = 0 fails (strictly >), debt = inf fails, debt = 0 passes
- Soft scoring: ranking correct for 5 mock stocks with known composite scores
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from screener.hard_filters import apply_hard_filters
from screener.soft_filters import apply_soft_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passing_row(**overrides: object) -> dict:
    """Return a single-ticker summary dict with defaults that pass ALL filters.

    Override any key to inject a failing value.  Defaults:
      profitable_years=10, avg_roe=0.20, eps_cagr=0.05,
      debt_payoff_years=2.0, years_available=10.
    """
    base: dict = {
        "ticker": "TEST",
        "profitable_years": 10,
        "avg_roe": 0.20,
        "eps_cagr": 0.05,
        "debt_payoff_years": 2.0,
        "years_available": 10,
    }
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    """Build a DataFrame from one or more dicts."""
    return pd.DataFrame(list(rows))


def _filter_result(
    log: pd.DataFrame, ticker: str, filter_name: str,
) -> bool:
    """Extract the pass_fail bool for a specific ticker x filter_name."""
    match = log[(log["ticker"] == ticker) & (log["filter_name"] == filter_name)]
    assert len(match) == 1, (
        f"Expected 1 row for ({ticker!r}, {filter_name!r}), got {len(match)}"
    )
    return bool(match.iloc[0]["pass_fail"])


# ===========================================================================
# Hard filter -- ROE floor
# ===========================================================================


class TestHardFilterRoeFloor:
    """ROE floor: avg_roe >= 0.15 (from config hard_filters.min_avg_roe)."""

    def test_roe_10pct_fails(self):
        """Stock with ROE = 10 % must fail the roe_floor filter."""
        df = _df(_passing_row(ticker="LOW_ROE", avg_roe=0.10))
        survivors, log = apply_hard_filters(df)
        assert len(survivors) == 0
        assert not _filter_result(log, "LOW_ROE", "roe_floor")

    def test_roe_20pct_passes(self):
        """Stock with ROE = 20 % must pass the roe_floor filter."""
        df = _df(_passing_row(ticker="HIGH_ROE", avg_roe=0.20))
        survivors, log = apply_hard_filters(df)
        assert _filter_result(log, "HIGH_ROE", "roe_floor")
        assert "HIGH_ROE" in survivors["ticker"].values

    def test_roe_exactly_15pct_passes(self):
        """ROE exactly at the threshold (15 %) passes (>= comparison)."""
        df = _df(_passing_row(ticker="EXACT_ROE", avg_roe=0.15))
        _, log = apply_hard_filters(df)
        assert _filter_result(log, "EXACT_ROE", "roe_floor")

    def test_roe_nan_fails(self):
        """NaN ROE must fail -- per SCORING.md NaN-handling rule."""
        df = _df(_passing_row(ticker="NAN_ROE", avg_roe=float("nan")))
        survivors, _ = apply_hard_filters(df)
        assert len(survivors) == 0


# ===========================================================================
# Hard filter -- Earnings consistency
# ===========================================================================


class TestHardFilterEarningsConsistency:
    """Earnings consistency: profitable_years >= 8 (from config)."""

    def test_6_profitable_years_fails(self):
        """Stock with 6 profitable years fails (threshold is 8)."""
        df = _df(_passing_row(ticker="LOW_EARN", profitable_years=6))
        survivors, log = apply_hard_filters(df)
        assert len(survivors) == 0
        assert not _filter_result(log, "LOW_EARN", "earnings_consistency")

    def test_8_profitable_years_passes(self):
        """Stock with exactly 8 profitable years passes."""
        df = _df(_passing_row(ticker="OK_EARN", profitable_years=8))
        _, log = apply_hard_filters(df)
        assert _filter_result(log, "OK_EARN", "earnings_consistency")

    def test_10_profitable_years_passes(self):
        """Stock with 10 profitable years passes."""
        df = _df(_passing_row(ticker="FULL_EARN", profitable_years=10))
        _, log = apply_hard_filters(df)
        assert _filter_result(log, "FULL_EARN", "earnings_consistency")


# ===========================================================================
# Hard filter -- All 5 filters applied (nothing skipped)
# ===========================================================================


class TestAllFiveFiltersApplied:
    """Verify that every ticker is evaluated against all 5 filters."""

    EXPECTED_FILTERS = {
        "earnings_consistency",
        "roe_floor",
        "eps_growth",
        "debt_sustainability",
        "data_sufficiency",
    }

    def test_filter_log_contains_all_five_names(self):
        """filter_log must contain entries for all 5 filter names."""
        df = _df(_passing_row(ticker="CHECK"))
        _, log = apply_hard_filters(df)
        assert set(log["filter_name"].unique()) == self.EXPECTED_FILTERS

    def test_each_ticker_gets_five_rows(self):
        """Each ticker must have exactly 5 rows in the filter log."""
        df = _df(
            _passing_row(ticker="A"),
            _passing_row(ticker="B"),
            _passing_row(ticker="C"),
        )
        _, log = apply_hard_filters(df)
        for ticker in ["A", "B", "C"]:
            ticker_log = log[log["ticker"] == ticker]
            assert len(ticker_log) == 5, (
                f"Ticker {ticker!r} has {len(ticker_log)} rows, expected 5"
            )
            assert set(ticker_log["filter_name"]) == self.EXPECTED_FILTERS

    def test_fail_one_filter_still_excluded(self):
        """A stock that fails only eps_growth must be excluded."""
        df = _df(
            _passing_row(ticker="GOOD"),
            _passing_row(ticker="BAD_EPS", eps_cagr=-0.01),
        )
        survivors, log = apply_hard_filters(df)
        assert set(survivors["ticker"]) == {"GOOD"}
        # BAD_EPS should fail only eps_growth
        assert not _filter_result(log, "BAD_EPS", "eps_growth")
        for f in self.EXPECTED_FILTERS - {"eps_growth"}:
            assert _filter_result(log, "BAD_EPS", f), (
                f"BAD_EPS should pass {f}"
            )

    def test_fail_all_five_filters(self):
        """A stock that fails all 5 filters is excluded with 5 False entries."""
        df = _df(_passing_row(
            ticker="TERRIBLE",
            profitable_years=2,
            avg_roe=0.02,
            eps_cagr=-0.10,
            debt_payoff_years=20.0,
            years_available=3,
        ))
        survivors, log = apply_hard_filters(df)
        assert len(survivors) == 0
        terrible_log = log[log["ticker"] == "TERRIBLE"]
        assert not terrible_log["pass_fail"].any(), (
            "All 5 filters should be False for TERRIBLE"
        )


# ===========================================================================
# Hard filter -- EPS CAGR edge cases
# ===========================================================================


class TestHardFilterEpsGrowth:
    """EPS growth: eps_cagr > min_eps_cagr (strictly greater than 0.0)."""

    def test_eps_cagr_zero_fails(self):
        """EPS CAGR exactly 0.0 fails (strictly > 0 required)."""
        df = _df(_passing_row(ticker="ZERO_EPS", eps_cagr=0.0))
        survivors, log = apply_hard_filters(df)
        assert not _filter_result(log, "ZERO_EPS", "eps_growth")
        assert len(survivors) == 0

    def test_eps_cagr_tiny_positive_passes(self):
        """EPS CAGR = 0.001 (barely positive) passes."""
        df = _df(_passing_row(ticker="TINY_EPS", eps_cagr=0.001))
        _, log = apply_hard_filters(df)
        assert _filter_result(log, "TINY_EPS", "eps_growth")

    def test_eps_cagr_negative_fails(self):
        """Negative EPS CAGR fails."""
        df = _df(_passing_row(ticker="NEG_EPS", eps_cagr=-0.05))
        _, log = apply_hard_filters(df)
        assert not _filter_result(log, "NEG_EPS", "eps_growth")


# ===========================================================================
# Hard filter -- Debt sustainability edge cases
# ===========================================================================


class TestHardFilterDebt:
    """Debt sustainability: debt_payoff_years <= 5.0."""

    def test_debt_payoff_infinity_fails(self):
        """Infinite debt payoff (negative OE) fails."""
        df = _df(_passing_row(ticker="INF_DEBT", debt_payoff_years=float("inf")))
        _, log = apply_hard_filters(df)
        assert not _filter_result(log, "INF_DEBT", "debt_sustainability")

    def test_debt_payoff_zero_passes(self):
        """Zero debt payoff (debt-free company) passes."""
        df = _df(_passing_row(ticker="NO_DEBT", debt_payoff_years=0.0))
        _, log = apply_hard_filters(df)
        assert _filter_result(log, "NO_DEBT", "debt_sustainability")

    def test_debt_payoff_exactly_5_passes(self):
        """Debt payoff exactly 5 years passes (<= comparison)."""
        df = _df(_passing_row(ticker="EXACT5", debt_payoff_years=5.0))
        _, log = apply_hard_filters(df)
        assert _filter_result(log, "EXACT5", "debt_sustainability")

    def test_debt_payoff_nan_fails(self):
        """NaN debt payoff fails."""
        df = _df(_passing_row(ticker="NAN_DEBT", debt_payoff_years=float("nan")))
        _, log = apply_hard_filters(df)
        assert not _filter_result(log, "NAN_DEBT", "debt_sustainability")


# ===========================================================================
# Hard filter -- Mixed batch and structural tests
# ===========================================================================


class TestHardFilterMixedBatch:
    """Batch of stocks with varying quality."""

    def test_mixed_batch_correct_survivors(self):
        """Only stocks passing all 5 filters survive."""
        df = _df(
            _passing_row(ticker="PASS_1"),
            _passing_row(ticker="PASS_2"),
            _passing_row(ticker="FAIL_ROE", avg_roe=0.10),
            _passing_row(ticker="FAIL_EPS", eps_cagr=-0.02),
            _passing_row(ticker="FAIL_DEBT", debt_payoff_years=10.0),
        )
        survivors, _ = apply_hard_filters(df)
        assert set(survivors["ticker"]) == {"PASS_1", "PASS_2"}

    def test_filter_log_has_correct_shape(self):
        """5 tickers x 5 filters = 25 rows in filter_log."""
        df = _df(*[_passing_row(ticker=t) for t in "ABCDE"])
        _, log = apply_hard_filters(df)
        assert len(log) == 25

    def test_empty_input_returns_empty(self):
        """Empty DataFrame input returns two empty DataFrames."""
        survivors, log = apply_hard_filters(pd.DataFrame())
        assert survivors.empty
        assert log.empty

    def test_filter_log_columns(self):
        """Filter log must have exactly the expected columns."""
        df = _df(_passing_row(ticker="X"))
        _, log = apply_hard_filters(df)
        expected = {"ticker", "filter_name", "filter_value", "threshold", "pass_fail"}
        assert set(log.columns) == expected

    def test_all_pass_all_survive(self):
        """When every stock passes all filters, survivors = input."""
        df = _df(
            _passing_row(ticker="A"),
            _passing_row(ticker="B"),
            _passing_row(ticker="C"),
        )
        survivors, _ = apply_hard_filters(df)
        assert set(survivors["ticker"]) == {"A", "B", "C"}


# ===========================================================================
# Soft scoring -- ranking
# ===========================================================================


class TestSoftScoringRanking:
    """apply_soft_scores must rank survivors correctly by composite_score."""

    def test_ranking_correct_for_5_stocks(self):
        """5 stocks with known scores: B(90) > C(72) > E(60) > A(45) > D(30)."""
        survivors = pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E"],
            "avg_roe": [0.20, 0.25, 0.30, 0.15, 0.18],
        })
        composite = pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E"],
            "composite_score": [45.0, 90.0, 72.0, 30.0, 60.0],
            "score_roe": [70.0, 100.0, 85.0, 50.0, 65.0],
        })
        result = apply_soft_scores(survivors, composite)
        assert list(result["ticker"]) == ["B", "C", "E", "A", "D"]
        assert list(result["rank"]) == [1, 2, 3, 4, 5]

    def test_composite_scores_preserved(self):
        """Original composite_score values must appear in the output."""
        survivors = pd.DataFrame({"ticker": ["X", "Y"]})
        composite = pd.DataFrame({
            "ticker": ["X", "Y"],
            "composite_score": [88.0, 55.0],
        })
        result = apply_soft_scores(survivors, composite)
        x_score = float(result.loc[result["ticker"] == "X", "composite_score"].iloc[0])
        y_score = float(result.loc[result["ticker"] == "Y", "composite_score"].iloc[0])
        assert x_score == 88.0
        assert y_score == 55.0

    def test_rank_starts_at_1(self):
        """Rank must be 1-based."""
        survivors = pd.DataFrame({"ticker": ["ONLY"]})
        composite = pd.DataFrame({
            "ticker": ["ONLY"],
            "composite_score": [50.0],
        })
        result = apply_soft_scores(survivors, composite)
        assert int(result.iloc[0]["rank"]) == 1

    def test_sorted_descending_by_composite_score(self):
        """Output must be sorted descending by composite_score."""
        survivors = pd.DataFrame({"ticker": ["A", "B", "C"]})
        composite = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "composite_score": [40.0, 80.0, 60.0],
        })
        result = apply_soft_scores(survivors, composite)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_score_columns_carried_through(self):
        """score_* columns from composite_scores_df must appear in output."""
        survivors = pd.DataFrame({"ticker": ["X"]})
        composite = pd.DataFrame({
            "ticker": ["X"],
            "composite_score": [80.0],
            "score_roe": [95.0],
            "score_gross_margin": [70.0],
        })
        result = apply_soft_scores(survivors, composite)
        assert "score_roe" in result.columns
        assert "score_gross_margin" in result.columns
        assert float(result.iloc[0]["score_roe"]) == 95.0

    def test_survivor_missing_from_composite_gets_nan(self):
        """A survivor not in composite_scores_df gets NaN composite_score."""
        survivors = pd.DataFrame({"ticker": ["A", "B"]})
        composite = pd.DataFrame({
            "ticker": ["A"],
            "composite_score": [70.0],
        })
        result = apply_soft_scores(survivors, composite)
        a_score = float(result.loc[result["ticker"] == "A", "composite_score"].iloc[0])
        b_score = result.loc[result["ticker"] == "B", "composite_score"].iloc[0]
        assert a_score == 70.0
        assert math.isnan(float(b_score))

    def test_empty_survivors_returns_empty(self):
        """No survivors -> empty result."""
        result = apply_soft_scores(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_rank_column_present(self):
        """Output must contain a 'rank' column."""
        survivors = pd.DataFrame({"ticker": ["A"]})
        composite = pd.DataFrame({
            "ticker": ["A"],
            "composite_score": [50.0],
        })
        result = apply_soft_scores(survivors, composite)
        assert "rank" in result.columns

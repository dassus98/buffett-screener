"""Tests for screener/hard_filters.py, screener/soft_filters.py,
screener/exclusions.py, and screener/composite_ranker.py.

Covers:
- Hard filter ROE threshold (10 % fails, 20 % passes, 15 % boundary)
- Hard filter earnings consistency (6 profitable years fails, 8 passes)
- All 5 filters applied — no filter is skipped
- NaN handling — NaN on any metric -> fail
- Edge cases: EPS CAGR = 0 fails (strictly >), debt = inf fails, debt = 0 passes
- Soft scoring: ranking correct for 5 mock stocks with known composite scores
- Exclusions: bank vs non-bank, SIC codes, industry patterns, flags
- Shortlist: 100 stocks → top 50, score_category boundaries
- Screener summary: correct keys and values
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from screener.composite_ranker import generate_screener_summary, generate_shortlist
from screener.exclusions import apply_exclusions
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


# ===========================================================================
# Exclusions — apply_exclusions
# ===========================================================================


def _universe_row(**overrides: object) -> dict:
    """Return a single-ticker universe dict with safe defaults.

    Override any key to inject sector/industry/SIC values.
    """
    base: dict = {
        "ticker": "SAFE",
        "exchange": "NYSE",
        "company_name": "Safe Inc.",
        "market_cap_usd": 1_000_000_000.0,
        "sector": "Technology",
        "industry": "Software — Application",
        "country": "US",
    }
    base.update(overrides)
    return base


class TestApplyExclusions:
    """apply_exclusions must remove financial-sector tickers and keep others."""

    def test_bank_excluded_by_industry_pattern(self):
        """A 'Banks' industry in 'Financial Services' sector is excluded."""
        df = _df(
            _universe_row(ticker="BANK", sector="Financial Services",
                          industry="Banks — Regional"),
            _universe_row(ticker="TECH", sector="Technology",
                          industry="Software — Application"),
        )
        filtered, log = apply_exclusions(df)
        assert set(filtered["ticker"]) == {"TECH"}
        assert "BANK" in log["ticker"].values

    def test_insurance_excluded_by_industry_pattern(self):
        """Insurance industry in Financial Services sector is excluded."""
        df = _df(
            _universe_row(ticker="INS", sector="Financial Services",
                          industry="Insurance — Property & Casualty"),
        )
        filtered, _ = apply_exclusions(df)
        assert len(filtered) == 0

    def test_reit_excluded_by_industry_pattern(self):
        """REIT industry in Financial Services sector is excluded."""
        df = _df(
            _universe_row(ticker="REIT_CO", sector="Financial Services",
                          industry="REIT — Diversified"),
        )
        filtered, log = apply_exclusions(df)
        assert len(filtered) == 0
        assert "REIT_CO" in log["ticker"].values

    def test_non_financial_sector_not_excluded_by_industry_keyword(self):
        """Industry keyword match alone (without Financial Services sector)
        does NOT exclude."""
        df = _df(
            _universe_row(ticker="SAFE_BANK", sector="Technology",
                          industry="Data Banks & Storage"),
        )
        filtered, log = apply_exclusions(df)
        assert "SAFE_BANK" in filtered["ticker"].values
        assert len(log) == 0

    def test_sic_code_exclusion(self):
        """Ticker with SIC code 6020 (commercial bank) is excluded."""
        df = _df(
            _universe_row(ticker="SIC_BANK", sic_code=6020),
            _universe_row(ticker="SIC_TECH", sic_code=3674),
        )
        filtered, log = apply_exclusions(df)
        assert set(filtered["ticker"]) == {"SIC_TECH"}
        assert "SIC_BANK" in log["ticker"].values

    def test_sic_code_range_boundary(self):
        """SIC code at range boundary (6029 = end of commercial banks)."""
        df = _df(_universe_row(ticker="BOUNDARY", sic_code=6029))
        filtered, _ = apply_exclusions(df)
        assert len(filtered) == 0

    def test_sic_code_outside_range_passes(self):
        """SIC code just outside excluded range passes."""
        df = _df(_universe_row(ticker="OUTSIDE", sic_code=6030))
        filtered, _ = apply_exclusions(df)
        assert "OUTSIDE" in filtered["ticker"].values

    def test_flag_spac_excluded(self):
        """Ticker flagged as SPAC is excluded."""
        df = _df(_universe_row(ticker="SPAC_CO", is_SPAC=True))
        filtered, log = apply_exclusions(df)
        assert len(filtered) == 0
        assert "SPAC_CO" in log["ticker"].values
        assert "flag_excluded" in log.iloc[0]["reason"]

    def test_flag_shell_company_excluded(self):
        """Ticker flagged as shell company is excluded."""
        df = _df(_universe_row(ticker="SHELL", is_shell_company=True))
        filtered, _ = apply_exclusions(df)
        assert len(filtered) == 0

    def test_no_flag_columns_passes(self):
        """Missing flag columns do not cause exclusion."""
        df = _df(_universe_row(ticker="NORMAL"))
        filtered, _ = apply_exclusions(df)
        assert "NORMAL" in filtered["ticker"].values

    def test_empty_input_returns_empty(self):
        """Empty DataFrame returns two empty DataFrames."""
        filtered, log = apply_exclusions(pd.DataFrame())
        assert filtered.empty
        assert log.empty

    def test_exclusion_log_columns(self):
        """Exclusion log must have ticker and reason columns."""
        df = _df(
            _universe_row(ticker="BANK", sector="Financial Services",
                          industry="Banks — Regional"),
        )
        _, log = apply_exclusions(df)
        assert set(log.columns) == {"ticker", "reason"}

    def test_multiple_reasons_semicolon_separated(self):
        """Ticker matching both SIC and industry gets combined reason."""
        df = _df(
            _universe_row(
                ticker="DOUBLE",
                sector="Financial Services",
                industry="Insurance Brokers",
                sic_code=6311,
            ),
        )
        _, log = apply_exclusions(df)
        reason = log.iloc[0]["reason"]
        assert "sic_code_excluded" in reason
        assert "industry_pattern_excluded" in reason
        assert ";" in reason

    def test_mixed_batch_correct_survivors(self):
        """Mixed universe: only non-financial, non-flagged tickers survive."""
        df = _df(
            _universe_row(ticker="AAPL", sector="Technology",
                          industry="Consumer Electronics"),
            _universe_row(ticker="JPM", sector="Financial Services",
                          industry="Banks — Diversified"),
            _universe_row(ticker="KO", sector="Consumer Defensive",
                          industry="Beverages — Non-Alcoholic"),
            _universe_row(ticker="MET", sector="Financial Services",
                          industry="Insurance — Life"),
            _universe_row(ticker="MSFT", sector="Technology",
                          industry="Software — Infrastructure"),
        )
        filtered, log = apply_exclusions(df)
        assert set(filtered["ticker"]) == {"AAPL", "KO", "MSFT"}
        assert set(log["ticker"]) == {"JPM", "MET"}


# ===========================================================================
# Shortlist — generate_shortlist
# ===========================================================================


def _ranked_df(n: int, base_score: float = 90.0) -> pd.DataFrame:
    """Build a mock ranked DataFrame with *n* tickers.

    Scores range from *base_score* down to *base_score - n + 1*
    (one point apart). Rank column is 1-based.
    """
    tickers = [f"T{i:03d}" for i in range(1, n + 1)]
    scores = [base_score - i + 1 for i in range(1, n + 1)]
    return pd.DataFrame({
        "ticker": tickers,
        "composite_score": scores,
        "rank": list(range(1, n + 1)),
    })


class TestGenerateShortlist:
    """generate_shortlist selects top-N with rank, percentile, score_category."""

    def test_100_stocks_top_50(self):
        """100 ranked stocks → shortlist of 50 (config default)."""
        df = _ranked_df(100)
        shortlist = generate_shortlist(df, top_n=50)
        assert len(shortlist) == 50
        assert list(shortlist["ticker"])[0] == "T001"
        assert list(shortlist["ticker"])[-1] == "T050"

    def test_rank_preserved_from_input(self):
        """Rank values should be preserved from the input ranked_df."""
        df = _ranked_df(20)
        shortlist = generate_shortlist(df, top_n=5)
        assert list(shortlist["rank"]) == [1, 2, 3, 4, 5]

    def test_percentile_rank_1_of_100(self):
        """Rank 1 out of 100 → percentile = 100.0."""
        df = _ranked_df(100)
        shortlist = generate_shortlist(df, top_n=1)
        assert shortlist.iloc[0]["percentile"] == 100.0

    def test_percentile_rank_50_of_100(self):
        """Rank 50 out of 100 → percentile = 51.0."""
        df = _ranked_df(100)
        shortlist = generate_shortlist(df, top_n=50)
        row_50 = shortlist[shortlist["rank"] == 50].iloc[0]
        assert row_50["percentile"] == 51.0

    def test_score_category_strong_buy(self):
        """Composite score >= 80 → 'Strong Buy'."""
        df = pd.DataFrame({
            "ticker": ["HIGH"],
            "composite_score": [85.0],
            "rank": [1],
        })
        shortlist = generate_shortlist(df, top_n=1)
        assert shortlist.iloc[0]["score_category"] == "Strong Buy"

    def test_score_category_buy(self):
        """Composite score 70 <= s < 80 → 'Buy'."""
        df = pd.DataFrame({
            "ticker": ["MID"],
            "composite_score": [75.0],
            "rank": [1],
        })
        shortlist = generate_shortlist(df, top_n=1)
        assert shortlist.iloc[0]["score_category"] == "Buy"

    def test_score_category_hold(self):
        """Composite score 60 <= s < 70 → 'Hold'."""
        df = pd.DataFrame({
            "ticker": ["OK"],
            "composite_score": [65.0],
            "rank": [1],
        })
        shortlist = generate_shortlist(df, top_n=1)
        assert shortlist.iloc[0]["score_category"] == "Hold"

    def test_score_category_weak(self):
        """Composite score < 60 → 'Weak'."""
        df = pd.DataFrame({
            "ticker": ["LOW"],
            "composite_score": [55.0],
            "rank": [1],
        })
        shortlist = generate_shortlist(df, top_n=1)
        assert shortlist.iloc[0]["score_category"] == "Weak"

    def test_score_category_boundary_80(self):
        """Exactly 80 → 'Strong Buy' (>= comparison)."""
        df = pd.DataFrame({
            "ticker": ["EDGE80"],
            "composite_score": [80.0],
            "rank": [1],
        })
        shortlist = generate_shortlist(df, top_n=1)
        assert shortlist.iloc[0]["score_category"] == "Strong Buy"

    def test_score_category_boundary_70(self):
        """Exactly 70 → 'Buy' (>= comparison)."""
        df = pd.DataFrame({
            "ticker": ["EDGE70"],
            "composite_score": [70.0],
            "rank": [1],
        })
        shortlist = generate_shortlist(df, top_n=1)
        assert shortlist.iloc[0]["score_category"] == "Buy"

    def test_score_category_boundary_60(self):
        """Exactly 60 → 'Hold' (>= comparison)."""
        df = pd.DataFrame({
            "ticker": ["EDGE60"],
            "composite_score": [60.0],
            "rank": [1],
        })
        shortlist = generate_shortlist(df, top_n=1)
        assert shortlist.iloc[0]["score_category"] == "Hold"

    def test_score_category_nan_is_weak(self):
        """NaN composite_score → 'Weak'."""
        df = pd.DataFrame({
            "ticker": ["NAN"],
            "composite_score": [float("nan")],
            "rank": [1],
        })
        shortlist = generate_shortlist(df, top_n=1)
        assert shortlist.iloc[0]["score_category"] == "Weak"

    def test_top_n_larger_than_population(self):
        """top_n > population → returns all rows."""
        df = _ranked_df(5)
        shortlist = generate_shortlist(df, top_n=100)
        assert len(shortlist) == 5

    def test_empty_ranked_returns_empty(self):
        """Empty ranked_df → empty result."""
        result = generate_shortlist(pd.DataFrame(), top_n=10)
        assert result.empty

    def test_metadata_columns_present(self):
        """Output must have rank, percentile, score_category columns."""
        df = _ranked_df(3)
        shortlist = generate_shortlist(df, top_n=2)
        assert "rank" in shortlist.columns
        assert "percentile" in shortlist.columns
        assert "score_category" in shortlist.columns

    def test_default_top_n_from_config(self):
        """When top_n=None, reads output.shortlist_size from config (50)."""
        df = _ranked_df(100)
        shortlist = generate_shortlist(df)  # top_n=None
        assert len(shortlist) == 50


# ===========================================================================
# Screener summary — generate_screener_summary
# ===========================================================================


class TestGenerateScreenerSummary:
    """generate_screener_summary builds a stats dict from pipeline outputs."""

    def test_summary_has_required_keys(self):
        """Summary dict must contain all expected keys."""
        full_ranked = _ranked_df(10)
        shortlist = full_ranked.head(5).copy()
        filter_log = pd.DataFrame({
            "ticker": ["A", "A", "B", "B"],
            "filter_name": ["f1", "f2", "f1", "f2"],
            "pass_fail": [True, True, True, False],
        })
        summary = generate_screener_summary(full_ranked, shortlist, filter_log)
        required = {
            "after_exclusions", "after_tier1", "shortlisted",
            "top_score", "median_score", "bottom_score",
            "sector_distribution", "exchange_distribution",
        }
        assert required.issubset(set(summary.keys()))

    def test_after_exclusions_from_filter_log(self):
        """after_exclusions = unique tickers in filter_log."""
        full_ranked = _ranked_df(5)
        shortlist = full_ranked.head(3).copy()
        filter_log = pd.DataFrame({
            "ticker": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "filter_name": ["f1"] * 5 + ["f2"] * 5,
            "pass_fail": [True] * 10,
        })
        summary = generate_screener_summary(full_ranked, shortlist, filter_log)
        assert summary["after_exclusions"] == 5

    def test_after_tier1_count(self):
        """after_tier1 = number of rows in full_ranked_df."""
        full_ranked = _ranked_df(8)
        summary = generate_screener_summary(
            full_ranked, full_ranked.head(3), pd.DataFrame(),
        )
        assert summary["after_tier1"] == 8

    def test_shortlisted_count(self):
        """shortlisted = number of rows in shortlist_df."""
        full_ranked = _ranked_df(20)
        shortlist = full_ranked.head(10).copy()
        summary = generate_screener_summary(
            full_ranked, shortlist, pd.DataFrame(),
        )
        assert summary["shortlisted"] == 10

    def test_score_statistics(self):
        """top/median/bottom score correctly computed."""
        full_ranked = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "composite_score": [90.0, 70.0, 50.0],
        })
        summary = generate_screener_summary(
            full_ranked, full_ranked, pd.DataFrame(),
        )
        assert summary["top_score"] == 90.0
        assert summary["median_score"] == 70.0
        assert summary["bottom_score"] == 50.0

    def test_sector_distribution(self):
        """sector_distribution counts sectors in shortlist."""
        shortlist = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "composite_score": [90.0, 80.0, 70.0],
            "sector": ["Tech", "Tech", "Health"],
        })
        summary = generate_screener_summary(
            shortlist, shortlist, pd.DataFrame(),
        )
        assert summary["sector_distribution"]["Tech"] == 2
        assert summary["sector_distribution"]["Health"] == 1

    def test_exchange_distribution(self):
        """exchange_distribution counts exchanges in shortlist."""
        shortlist = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "composite_score": [90.0, 80.0, 70.0],
            "exchange": ["NYSE", "NASDAQ", "NYSE"],
        })
        summary = generate_screener_summary(
            shortlist, shortlist, pd.DataFrame(),
        )
        assert summary["exchange_distribution"]["NYSE"] == 2
        assert summary["exchange_distribution"]["NASDAQ"] == 1

    def test_empty_inputs(self):
        """All empty inputs → zero counts and None scores."""
        summary = generate_screener_summary(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        )
        assert summary["after_tier1"] == 0
        assert summary["shortlisted"] == 0
        assert summary["top_score"] is None

    def test_nan_scores_excluded_from_stats(self):
        """NaN scores should not affect top/median/bottom."""
        full_ranked = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "composite_score": [80.0, float("nan"), 60.0],
        })
        summary = generate_screener_summary(
            full_ranked, full_ranked, pd.DataFrame(),
        )
        assert summary["top_score"] == 80.0
        assert summary["bottom_score"] == 60.0

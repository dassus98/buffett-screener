"""Integration tests: Module 3 full screening pipeline.

Exercises the complete screening chain with 20 mock tickers:

    apply_exclusions → apply_hard_filters → apply_soft_scores
        → generate_shortlist → generate_screener_summary

Ticker categories
-----------------
* **5 high-quality** — ROE > 20 %, EPS CAGR > 7 %, low debt, high margins.
  Composite scores 78–92.
* **5 medium-quality** — pass all hard filters but mediocre soft scores.
  Composite scores 52–65.
* **5 Tier 1 failures** — each fails a different hard filter
  (low ROE, negative EPS growth, high debt, few profitable years,
  insufficient history).
* **3 financial-sector** — excluded before hard filters via industry-keyword
  matching (Banks, Insurance, REIT).
* **2 below-shortlist** — pass hard filters but composite scores (42–45) are
  the lowest among survivors, ensuring they fall outside the top-10 shortlist.

What is verified
----------------
1. Financial-sector tickers excluded before Tier 1.
2. Tier 1 failures removed for the correct reason.
3. Shortlist size, ordering, and score dominance over non-shortlisted
   survivors.
4. Filter log completeness (every ticker × every filter).
5. Summary statistics match pipeline counts.
6. Idempotency — running the pipeline twice yields identical results.
"""

from __future__ import annotations

import pandas as pd
import pytest

from screener.composite_ranker import generate_screener_summary, generate_shortlist
from screener.exclusions import apply_exclusions
from screener.hard_filters import apply_hard_filters
from screener.soft_filters import apply_soft_scores


# ---------------------------------------------------------------------------
# Test data — 20 tickers with known quality profiles
# ---------------------------------------------------------------------------

_HIGH_QUALITY: list[dict] = [
    {
        "ticker": "HQ1", "exchange": "NYSE",
        "sector": "Technology", "industry": "Software — Application",
        "profitable_years": 10, "avg_roe": 0.25, "eps_cagr": 0.12,
        "debt_payoff_years": 1.5, "years_available": 10,
        "composite_score": 92.0, "score_roe": 95.0,
    },
    {
        "ticker": "HQ2", "exchange": "NYSE",
        "sector": "Healthcare", "industry": "Drug Manufacturers",
        "profitable_years": 10, "avg_roe": 0.22, "eps_cagr": 0.10,
        "debt_payoff_years": 2.0, "years_available": 10,
        "composite_score": 88.0, "score_roe": 85.0,
    },
    {
        "ticker": "HQ3", "exchange": "NASDAQ",
        "sector": "Technology", "industry": "Semiconductors",
        "profitable_years": 9, "avg_roe": 0.20, "eps_cagr": 0.08,
        "debt_payoff_years": 1.0, "years_available": 10,
        "composite_score": 85.0, "score_roe": 80.0,
    },
    {
        "ticker": "HQ4", "exchange": "NASDAQ",
        "sector": "Consumer Defensive", "industry": "Beverages",
        "profitable_years": 10, "avg_roe": 0.18, "eps_cagr": 0.15,
        "debt_payoff_years": 3.0, "years_available": 10,
        "composite_score": 82.0, "score_roe": 70.0,
    },
    {
        "ticker": "HQ5", "exchange": "TSX",
        "sector": "Industrials", "industry": "Aerospace & Defense",
        "profitable_years": 10, "avg_roe": 0.21, "eps_cagr": 0.07,
        "debt_payoff_years": 2.5, "years_available": 10,
        "composite_score": 78.0, "score_roe": 82.0,
    },
]

_MEDIUM_QUALITY: list[dict] = [
    {
        "ticker": "MQ1", "exchange": "NYSE",
        "sector": "Consumer Cyclical", "industry": "Apparel",
        "profitable_years": 8, "avg_roe": 0.16, "eps_cagr": 0.03,
        "debt_payoff_years": 4.0, "years_available": 9,
        "composite_score": 65.0, "score_roe": 55.0,
    },
    {
        "ticker": "MQ2", "exchange": "NASDAQ",
        "sector": "Technology", "industry": "IT Services",
        "profitable_years": 9, "avg_roe": 0.17, "eps_cagr": 0.02,
        "debt_payoff_years": 3.5, "years_available": 10,
        "composite_score": 62.0, "score_roe": 58.0,
    },
    {
        "ticker": "MQ3", "exchange": "NYSE",
        "sector": "Healthcare", "industry": "Medical Devices",
        "profitable_years": 8, "avg_roe": 0.15, "eps_cagr": 0.01,
        "debt_payoff_years": 4.5, "years_available": 8,
        "composite_score": 58.0, "score_roe": 50.0,
    },
    {
        "ticker": "MQ4", "exchange": "TSX",
        "sector": "Industrials", "industry": "Building Products",
        "profitable_years": 8, "avg_roe": 0.16, "eps_cagr": 0.04,
        "debt_payoff_years": 5.0, "years_available": 9,
        "composite_score": 55.0, "score_roe": 55.0,
    },
    {
        "ticker": "MQ5", "exchange": "NYSE",
        "sector": "Consumer Defensive", "industry": "Household Products",
        "profitable_years": 9, "avg_roe": 0.15, "eps_cagr": 0.02,
        "debt_payoff_years": 3.0, "years_available": 10,
        "composite_score": 52.0, "score_roe": 50.0,
    },
]

_FAIL_TIER1: list[dict] = [
    {   # Fails roe_floor (0.10 < 0.15)
        "ticker": "F1_ROE", "exchange": "NYSE",
        "sector": "Technology", "industry": "Software",
        "profitable_years": 10, "avg_roe": 0.10, "eps_cagr": 0.05,
        "debt_payoff_years": 2.0, "years_available": 10,
        "composite_score": 40.0, "score_roe": 30.0,
    },
    {   # Fails eps_growth (-0.02 not > 0.0)
        "ticker": "F2_EPS", "exchange": "NASDAQ",
        "sector": "Healthcare", "industry": "Diagnostics",
        "profitable_years": 10, "avg_roe": 0.20, "eps_cagr": -0.02,
        "debt_payoff_years": 2.0, "years_available": 10,
        "composite_score": 35.0, "score_roe": 80.0,
    },
    {   # Fails debt_sustainability (8.0 > 5.0)
        "ticker": "F3_DEBT", "exchange": "NYSE",
        "sector": "Industrials", "industry": "Construction",
        "profitable_years": 10, "avg_roe": 0.20, "eps_cagr": 0.05,
        "debt_payoff_years": 8.0, "years_available": 10,
        "composite_score": 30.0, "score_roe": 80.0,
    },
    {   # Fails earnings_consistency (5 < 8)
        "ticker": "F4_EARN", "exchange": "TSX",
        "sector": "Energy", "industry": "Oil & Gas",
        "profitable_years": 5, "avg_roe": 0.20, "eps_cagr": 0.05,
        "debt_payoff_years": 2.0, "years_available": 10,
        "composite_score": 25.0, "score_roe": 80.0,
    },
    {   # Fails data_sufficiency (5 < 8)
        "ticker": "F5_DATA", "exchange": "NASDAQ",
        "sector": "Technology", "industry": "Cloud Computing",
        "profitable_years": 10, "avg_roe": 0.20, "eps_cagr": 0.05,
        "debt_payoff_years": 2.0, "years_available": 5,
        "composite_score": 20.0, "score_roe": 80.0,
    },
]

_FINANCIALS: list[dict] = [
    {
        "ticker": "FIN1", "exchange": "NYSE",
        "sector": "Financial Services", "industry": "Banks — Diversified",
        "profitable_years": 10, "avg_roe": 0.15, "eps_cagr": 0.05,
        "debt_payoff_years": 2.0, "years_available": 10,
        "composite_score": 70.0, "score_roe": 50.0,
    },
    {
        "ticker": "FIN2", "exchange": "NYSE",
        "sector": "Financial Services",
        "industry": "Insurance — Property & Casualty",
        "profitable_years": 10, "avg_roe": 0.18, "eps_cagr": 0.06,
        "debt_payoff_years": 3.0, "years_available": 10,
        "composite_score": 68.0, "score_roe": 65.0,
    },
    {
        "ticker": "FIN3", "exchange": "NASDAQ",
        "sector": "Financial Services", "industry": "REIT — Diversified",
        "profitable_years": 10, "avg_roe": 0.16, "eps_cagr": 0.04,
        "debt_payoff_years": 2.5, "years_available": 10,
        "composite_score": 66.0, "score_roe": 55.0,
    },
]

_BELOW_THRESHOLD: list[dict] = [
    {
        "ticker": "BT1", "exchange": "NYSE",
        "sector": "Consumer Cyclical", "industry": "Restaurants",
        "profitable_years": 8, "avg_roe": 0.155, "eps_cagr": 0.005,
        "debt_payoff_years": 4.8, "years_available": 8,
        "composite_score": 45.0, "score_roe": 52.0,
    },
    {
        "ticker": "BT2", "exchange": "TSX",
        "sector": "Industrials", "industry": "Waste Management",
        "profitable_years": 8, "avg_roe": 0.151, "eps_cagr": 0.003,
        "debt_payoff_years": 4.9, "years_available": 8,
        "composite_score": 42.0, "score_roe": 48.0,
    },
]

_ALL_TICKERS = (
    _HIGH_QUALITY + _MEDIUM_QUALITY + _FAIL_TIER1
    + _FINANCIALS + _BELOW_THRESHOLD
)

# Expected ticker sets at each pipeline stage
_EXCLUDED_TICKERS = {"FIN1", "FIN2", "FIN3"}
_TIER1_FAIL_TICKERS = {"F1_ROE", "F2_EPS", "F3_DEBT", "F4_EARN", "F5_DATA"}
_SURVIVOR_TICKERS = {
    "HQ1", "HQ2", "HQ3", "HQ4", "HQ5",
    "MQ1", "MQ2", "MQ3", "MQ4", "MQ5",
    "BT1", "BT2",
}
_SHORTLIST_TICKERS = {
    "HQ1", "HQ2", "HQ3", "HQ4", "HQ5",
    "MQ1", "MQ2", "MQ3", "MQ4", "MQ5",
}
_NON_SHORTLIST_SURVIVORS = {"BT1", "BT2"}

_EXPECTED_FILTERS = {
    "earnings_consistency", "roe_floor", "eps_growth",
    "debt_sustainability", "data_sufficiency",
}


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------


def _build_universe_df() -> pd.DataFrame:
    """Construct the 20-ticker universe DataFrame."""
    rows = []
    for t in _ALL_TICKERS:
        rows.append({
            "ticker": t["ticker"],
            "exchange": t["exchange"],
            "company_name": f"{t['ticker']} Inc.",
            "market_cap_usd": 1_000_000_000.0,
            "sector": t["sector"],
            "industry": t["industry"],
            "country": "US",
        })
    return pd.DataFrame(rows)


def _build_metrics_summary(tickers: set[str]) -> pd.DataFrame:
    """Build a metrics-summary DataFrame for a subset of tickers.

    Includes both hard-filter metric columns and universe metadata
    (sector, exchange) so they propagate to the shortlist.
    """
    lookup = {t["ticker"]: t for t in _ALL_TICKERS}
    rows = []
    for ticker in sorted(tickers):
        d = lookup[ticker]
        rows.append({
            "ticker": d["ticker"],
            "profitable_years": d["profitable_years"],
            "avg_roe": d["avg_roe"],
            "eps_cagr": d["eps_cagr"],
            "debt_payoff_years": d["debt_payoff_years"],
            "years_available": d["years_available"],
            "sector": d["sector"],
            "exchange": d["exchange"],
        })
    return pd.DataFrame(rows)


def _build_composite_scores(tickers: set[str]) -> pd.DataFrame:
    """Build composite-scores DataFrame for a subset of tickers."""
    lookup = {t["ticker"]: t for t in _ALL_TICKERS}
    rows = []
    for ticker in sorted(tickers):
        d = lookup[ticker]
        rows.append({
            "ticker": d["ticker"],
            "composite_score": d["composite_score"],
            "score_roe": d["score_roe"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _run_pipeline() -> dict:
    """Execute the full screening pipeline and return all artefacts.

    Returns a dict containing every intermediate DataFrame and the
    final summary, keyed by stage name.
    """
    # Stage 1: Exclusions
    universe_df = _build_universe_df()
    filtered_df, exclusion_log = apply_exclusions(universe_df)

    # Stage 2: Build metrics summary for surviving tickers
    surviving_tickers = set(filtered_df["ticker"])
    metrics_summary_df = _build_metrics_summary(surviving_tickers)

    # Stage 3: Hard filters (Tier 1)
    survivors_df, filter_log_df = apply_hard_filters(metrics_summary_df)

    # Stage 4: Soft scores (Tier 2) — join pre-computed composite scores
    all_non_excluded = set(metrics_summary_df["ticker"])
    composite_scores_df = _build_composite_scores(all_non_excluded)
    ranked_df = apply_soft_scores(survivors_df, composite_scores_df)

    # Stage 5: Shortlist (top 10)
    shortlist_df = generate_shortlist(ranked_df, top_n=10)

    # Stage 6: Summary
    summary = generate_screener_summary(
        ranked_df, shortlist_df, filter_log_df,
    )

    return {
        "universe_df": universe_df,
        "filtered_df": filtered_df,
        "exclusion_log": exclusion_log,
        "metrics_summary_df": metrics_summary_df,
        "survivors_df": survivors_df,
        "filter_log_df": filter_log_df,
        "composite_scores_df": composite_scores_df,
        "ranked_df": ranked_df,
        "shortlist_df": shortlist_df,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result() -> dict:
    """Run the pipeline once for the entire test module."""
    return _run_pipeline()


# ===========================================================================
# 1. Exclusion stage
# ===========================================================================


class TestExclusionStage:
    """Verify that financial-sector tickers are removed before Tier 1."""

    def test_three_financials_excluded(self, result: dict) -> None:
        """Exactly 3 financial-sector tickers must be excluded."""
        excluded = set(result["exclusion_log"]["ticker"])
        assert excluded == _EXCLUDED_TICKERS

    def test_exclusion_log_has_reasons(self, result: dict) -> None:
        """Every excluded ticker must have a non-empty reason string."""
        log = result["exclusion_log"]
        for _, row in log.iterrows():
            assert len(row["reason"]) > 0, (
                f"Empty reason for excluded ticker {row['ticker']}"
            )

    def test_filtered_df_has_17_tickers(self, result: dict) -> None:
        """After exclusions, 17 of 20 tickers remain."""
        assert len(result["filtered_df"]) == 17

    def test_financials_absent_from_filtered(self, result: dict) -> None:
        """No financial ticker appears in the filtered universe."""
        tickers = set(result["filtered_df"]["ticker"])
        assert tickers.isdisjoint(_EXCLUDED_TICKERS)

    def test_all_non_financials_present(self, result: dict) -> None:
        """Every non-financial ticker survives exclusions."""
        expected = {t["ticker"] for t in _ALL_TICKERS} - _EXCLUDED_TICKERS
        actual = set(result["filtered_df"]["ticker"])
        assert actual == expected


# ===========================================================================
# 2. Hard filter (Tier 1) stage
# ===========================================================================


class TestHardFilterStage:
    """Verify that exactly 5 low-quality stocks fail Tier 1."""

    def test_five_tickers_fail_tier1(self, result: dict) -> None:
        """Exactly 5 tickers must be excluded by hard filters."""
        survivors = set(result["survivors_df"]["ticker"])
        failed = set(result["metrics_summary_df"]["ticker"]) - survivors
        assert failed == _TIER1_FAIL_TICKERS

    def test_twelve_survivors(self, result: dict) -> None:
        """17 after exclusions minus 5 failures = 12 survivors."""
        assert len(result["survivors_df"]) == 12

    def test_survivor_set_correct(self, result: dict) -> None:
        """The surviving tickers match the expected set."""
        actual = set(result["survivors_df"]["ticker"])
        assert actual == _SURVIVOR_TICKERS

    def test_f1_roe_fails_roe_floor(self, result: dict) -> None:
        """F1_ROE (ROE = 10 %) must fail specifically on roe_floor."""
        log = result["filter_log_df"]
        row = log[(log["ticker"] == "F1_ROE") & (log["filter_name"] == "roe_floor")]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])

    def test_f2_eps_fails_eps_growth(self, result: dict) -> None:
        """F2_EPS (CAGR = -2 %) must fail specifically on eps_growth."""
        log = result["filter_log_df"]
        row = log[(log["ticker"] == "F2_EPS") & (log["filter_name"] == "eps_growth")]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])

    def test_f3_debt_fails_debt_sustainability(self, result: dict) -> None:
        """F3_DEBT (8 yr payoff) must fail specifically on debt_sustainability."""
        log = result["filter_log_df"]
        row = log[
            (log["ticker"] == "F3_DEBT")
            & (log["filter_name"] == "debt_sustainability")
        ]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])

    def test_f4_earn_fails_earnings_consistency(self, result: dict) -> None:
        """F4_EARN (5 profitable years) must fail earnings_consistency."""
        log = result["filter_log_df"]
        row = log[
            (log["ticker"] == "F4_EARN")
            & (log["filter_name"] == "earnings_consistency")
        ]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])

    def test_f5_data_fails_data_sufficiency(self, result: dict) -> None:
        """F5_DATA (5 years available) must fail data_sufficiency."""
        log = result["filter_log_df"]
        row = log[
            (log["ticker"] == "F5_DATA")
            & (log["filter_name"] == "data_sufficiency")
        ]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])


# ===========================================================================
# 3. Filter log completeness
# ===========================================================================


class TestFilterLogCompleteness:
    """filter_log_df must have entries for every ticker and every filter."""

    def test_filter_log_row_count(self, result: dict) -> None:
        """17 tickers × 5 filters = 85 rows."""
        assert len(result["filter_log_df"]) == 85

    def test_every_ticker_has_five_entries(self, result: dict) -> None:
        """Each of the 17 post-exclusion tickers gets 5 filter rows."""
        log = result["filter_log_df"]
        for ticker in result["metrics_summary_df"]["ticker"]:
            ticker_rows = log[log["ticker"] == ticker]
            assert len(ticker_rows) == 5, (
                f"{ticker} has {len(ticker_rows)} filter rows, expected 5"
            )

    def test_all_five_filter_names_present(self, result: dict) -> None:
        """All 5 expected filter names appear in the log."""
        actual = set(result["filter_log_df"]["filter_name"].unique())
        assert actual == _EXPECTED_FILTERS

    def test_filter_log_columns(self, result: dict) -> None:
        """Filter log must have the expected column schema."""
        expected_cols = {
            "ticker", "filter_name", "filter_value", "threshold", "pass_fail",
        }
        assert set(result["filter_log_df"].columns) == expected_cols

    def test_survivors_pass_all_five_filters(self, result: dict) -> None:
        """Every surviving ticker must have pass_fail=True on all 5 filters."""
        log = result["filter_log_df"]
        for ticker in _SURVIVOR_TICKERS:
            ticker_log = log[log["ticker"] == ticker]
            assert ticker_log["pass_fail"].all(), (
                f"Survivor {ticker} has at least one False in filter log"
            )


# ===========================================================================
# 4. Shortlist stage
# ===========================================================================


class TestShortlistStage:
    """Shortlist must contain ≤ 10 stocks with correct ordering."""

    def test_shortlist_has_10_stocks(self, result: dict) -> None:
        """With 12 survivors and top_n=10, shortlist has exactly 10."""
        assert len(result["shortlist_df"]) == 10

    def test_shortlist_tickers_correct(self, result: dict) -> None:
        """Shortlisted tickers are the 5 HQ + 5 MQ (highest scores)."""
        actual = set(result["shortlist_df"]["ticker"])
        assert actual == _SHORTLIST_TICKERS

    def test_shortlist_ordered_descending(self, result: dict) -> None:
        """Shortlist must be sorted by composite_score descending."""
        scores = result["shortlist_df"]["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_shortlist_top_ticker_is_hq1(self, result: dict) -> None:
        """HQ1 (composite_score=92) must be rank 1."""
        top = result["shortlist_df"].iloc[0]
        assert top["ticker"] == "HQ1"
        assert top["rank"] == 1

    def test_shortlist_scores_dominate_non_shortlist(self, result: dict) -> None:
        """Every shortlisted score must exceed every non-shortlisted score."""
        shortlist = result["shortlist_df"]
        ranked = result["ranked_df"]
        non_shortlist = ranked[
            ranked["ticker"].isin(_NON_SHORTLIST_SURVIVORS)
        ]
        min_shortlist_score = shortlist["composite_score"].min()
        max_non_shortlist_score = non_shortlist["composite_score"].max()
        assert min_shortlist_score > max_non_shortlist_score, (
            f"Min shortlist score ({min_shortlist_score}) must exceed "
            f"max non-shortlist score ({max_non_shortlist_score})"
        )

    def test_shortlist_has_metadata_columns(self, result: dict) -> None:
        """Shortlist must include rank, percentile, and score_category."""
        cols = set(result["shortlist_df"].columns)
        assert {"rank", "percentile", "score_category"}.issubset(cols)

    def test_shortlist_ranks_are_1_to_10(self, result: dict) -> None:
        """Rank values must be consecutive 1 through 10."""
        ranks = result["shortlist_df"]["rank"].tolist()
        assert ranks == list(range(1, 11))

    def test_score_categories_correct(self, result: dict) -> None:
        """Verify score categories match composite_score values.

        HQ1(92), HQ2(88), HQ3(85), HQ4(82) → "Strong Buy" (≥ 80)
        HQ5(78) → "Buy" (≥ 70, < 80)
        MQ1(65), MQ2(62) → "Hold" (≥ 60, < 70)
        MQ3(58), MQ4(55), MQ5(52) → "Weak" (< 60)
        """
        df = result["shortlist_df"]

        strong_buy = df[df["score_category"] == "Strong Buy"]["ticker"]
        assert set(strong_buy) == {"HQ1", "HQ2", "HQ3", "HQ4"}

        buy = df[df["score_category"] == "Buy"]["ticker"]
        assert set(buy) == {"HQ5"}

        hold = df[df["score_category"] == "Hold"]["ticker"]
        assert set(hold) == {"MQ1", "MQ2"}

        weak = df[df["score_category"] == "Weak"]["ticker"]
        assert set(weak) == {"MQ3", "MQ4", "MQ5"}

    def test_percentile_rank1_is_highest(self, result: dict) -> None:
        """Rank 1 out of 12 total → percentile = 100.0."""
        top = result["shortlist_df"].iloc[0]
        assert top["percentile"] == 100.0

    def test_non_shortlist_survivors_exist_in_ranked(self, result: dict) -> None:
        """BT1 and BT2 must be present in ranked_df but absent from shortlist."""
        ranked_tickers = set(result["ranked_df"]["ticker"])
        shortlist_tickers = set(result["shortlist_df"]["ticker"])
        assert _NON_SHORTLIST_SURVIVORS.issubset(ranked_tickers)
        assert _NON_SHORTLIST_SURVIVORS.isdisjoint(shortlist_tickers)


# ===========================================================================
# 5. Screener summary
# ===========================================================================


class TestScreenerSummary:
    """generate_screener_summary must produce correct aggregate counts."""

    def test_after_exclusions_count(self, result: dict) -> None:
        """after_exclusions = 17 (unique tickers in filter_log)."""
        assert result["summary"]["after_exclusions"] == 17

    def test_after_tier1_count(self, result: dict) -> None:
        """after_tier1 = 12 (survivors of hard filters)."""
        assert result["summary"]["after_tier1"] == 12

    def test_shortlisted_count(self, result: dict) -> None:
        """shortlisted = 10."""
        assert result["summary"]["shortlisted"] == 10

    def test_top_score(self, result: dict) -> None:
        """Top score across all survivors = 92.0 (HQ1)."""
        assert result["summary"]["top_score"] == 92.0

    def test_bottom_score(self, result: dict) -> None:
        """Bottom score across all survivors = 42.0 (BT2)."""
        assert result["summary"]["bottom_score"] == 42.0

    def test_median_score(self, result: dict) -> None:
        """Median of 12 survivor scores.

        Sorted scores: 42, 45, 52, 55, 58, 62, 65, 78, 82, 85, 88, 92
        Median of 12 values = (62 + 65) / 2 = 63.5
        """
        assert result["summary"]["median_score"] == 63.5

    def test_summary_has_all_expected_keys(self, result: dict) -> None:
        """Summary dict must contain all required keys."""
        required = {
            "after_exclusions", "after_tier1", "shortlisted",
            "top_score", "median_score", "bottom_score",
            "sector_distribution", "exchange_distribution",
        }
        assert required.issubset(set(result["summary"].keys()))


# ===========================================================================
# 6. Idempotency
# ===========================================================================


class TestIdempotency:
    """Running the pipeline twice on identical data produces identical output."""

    def test_idempotent_results(self) -> None:
        """Two independent runs must produce identical shortlists."""
        r1 = _run_pipeline()
        r2 = _run_pipeline()

        # Shortlist tickers identical
        assert list(r1["shortlist_df"]["ticker"]) == list(
            r2["shortlist_df"]["ticker"]
        )

        # Composite scores identical
        assert list(r1["shortlist_df"]["composite_score"]) == list(
            r2["shortlist_df"]["composite_score"]
        )

        # Ranks identical
        assert list(r1["shortlist_df"]["rank"]) == list(
            r2["shortlist_df"]["rank"]
        )

        # Score categories identical
        assert list(r1["shortlist_df"]["score_category"]) == list(
            r2["shortlist_df"]["score_category"]
        )

        # Filter log shape identical
        assert len(r1["filter_log_df"]) == len(r2["filter_log_df"])

        # Summary counts identical
        for key in ("after_exclusions", "after_tier1", "shortlisted",
                    "top_score", "median_score", "bottom_score"):
            assert r1["summary"][key] == r2["summary"][key], (
                f"Summary key {key!r} differs: {r1['summary'][key]} "
                f"vs {r2['summary'][key]}"
            )

"""Integration tests: Module 3 screening pipeline.

Exercises the full screening pipeline end-to-end with 20 synthetic tickers
in five quality tiers:

* **5 high-quality (HQ_01–HQ_05)** — strong Buffett characteristics: ROE > 20 %,
  EPS CAGR > 10 %, low debt, high margins.  Expected composite scores 80–97.
* **5 medium-quality (MQ_01–MQ_05)** — pass hard filters but mediocre soft
  scores: moderate ROE, low-ish growth, middling debt and margins.
  Expected composite scores 40–55.
* **5 hard-filter failures (FAIL_ROE, FAIL_EPS, FAIL_DEBT, FAIL_EARN,
  FAIL_DATA)** — each fails exactly one Tier 1 hard filter.
* **3 financial-sector (FIN_BANK, FIN_INS, FIN_SIC)** — excluded by SIC code
  and/or industry keyword before Tier 1.
* **2 low-score (LOW_01, LOW_02)** — pass hard filters but score poorly on
  all soft criteria (~10–20), placing them outside the top-10 shortlist.

Pipeline under test
-------------------
``apply_exclusions → apply_hard_filters → apply_soft_scores → generate_shortlist``

Composite scores are computed via the real ``compute_all_composite_scores``
engine (not pre-baked), so this test also exercises the full soft-scoring
pipeline end-to-end.

What is verified
----------------
1. Financial-sector exclusion removes exactly 3 tickers before Tier 1.
2. Hard filters remove exactly 5 tickers (one per failure reason).
3. Shortlist contains ≤ 10 stocks, ordered by composite_score descending.
4. All shortlist stocks outscore all non-shortlist survivors.
5. filter_log_df has entries for every ticker × every filter.
6. generate_screener_summary produces correct pipeline counts.
7. Idempotency: running the pipeline twice yields identical results.
"""

from __future__ import annotations

import pandas as pd
import pytest

from metrics_engine.composite_score import compute_all_composite_scores
from screener.composite_ranker import generate_screener_summary, generate_shortlist
from screener.exclusions import apply_exclusions
from screener.hard_filters import apply_hard_filters
from screener.soft_filters import apply_soft_scores

# ---------------------------------------------------------------------------
# Ticker group constants
# ---------------------------------------------------------------------------

TICKERS_HQ = ["HQ_01", "HQ_02", "HQ_03", "HQ_04", "HQ_05"]
TICKERS_MQ = ["MQ_01", "MQ_02", "MQ_03", "MQ_04", "MQ_05"]
TICKERS_FAIL = ["FAIL_ROE", "FAIL_EPS", "FAIL_DEBT", "FAIL_EARN", "FAIL_DATA"]
TICKERS_FIN = ["FIN_BANK", "FIN_INS", "FIN_SIC"]
TICKERS_LOW = ["LOW_01", "LOW_02"]

ALL_TICKERS = TICKERS_HQ + TICKERS_MQ + TICKERS_FAIL + TICKERS_FIN + TICKERS_LOW
TICKERS_AFTER_EXCLUSIONS = TICKERS_HQ + TICKERS_MQ + TICKERS_FAIL + TICKERS_LOW
TICKERS_AFTER_HARD = TICKERS_HQ + TICKERS_MQ + TICKERS_LOW

N_HARD_FILTERS = 5  # earnings_consistency, roe_floor, eps_growth, debt, data
EXPECTED_FILTERS = {
    "earnings_consistency", "roe_floor", "eps_growth",
    "debt_sustainability", "data_sufficiency",
}


# ---------------------------------------------------------------------------
# Universe DataFrame builder (input to apply_exclusions)
# ---------------------------------------------------------------------------


def _build_universe_df() -> pd.DataFrame:
    """Build a 20-ticker universe DataFrame with sector/industry/SIC/flags."""
    rows: list[dict] = []

    # --- High-quality: Technology sector, clean non-financial SIC ---
    for t in TICKERS_HQ:
        rows.append({
            "ticker": t, "sector": "Technology", "industry": "Software",
            "sic_code": 7372, "is_SPAC": False, "is_shell_company": False,
        })

    # --- Medium-quality: Consumer Defensive sector ---
    for t in TICKERS_MQ:
        rows.append({
            "ticker": t, "sector": "Consumer Defensive",
            "industry": "Beverages—Non-Alcoholic",
            "sic_code": 2080, "is_SPAC": False, "is_shell_company": False,
        })

    # --- Hard-filter failures: Industrials sector (not excluded) ---
    for t in TICKERS_FAIL:
        rows.append({
            "ticker": t, "sector": "Industrials",
            "industry": "Aerospace & Defense",
            "sic_code": 3720, "is_SPAC": False, "is_shell_company": False,
        })

    # --- Financial-sector tickers (should be excluded) ---
    #     FIN_BANK: SIC 6020 (excluded range) + keyword "Banks" match
    #     FIN_INS:  SIC 6311 (excluded range) + keyword "Insurance" match
    #     FIN_SIC:  SIC 6726 (excluded range), no keyword match
    rows.append({
        "ticker": "FIN_BANK", "sector": "Financial Services",
        "industry": "Banks—Diversified",
        "sic_code": 6020, "is_SPAC": False, "is_shell_company": False,
    })
    rows.append({
        "ticker": "FIN_INS", "sector": "Financial Services",
        "industry": "Insurance—Diversified",
        "sic_code": 6311, "is_SPAC": False, "is_shell_company": False,
    })
    rows.append({
        "ticker": "FIN_SIC", "sector": "Financial Services",
        "industry": "Capital Markets",
        "sic_code": 6726, "is_SPAC": False, "is_shell_company": False,
    })

    # --- Low-score: Healthcare sector (pass hard filters, poor soft scores) ---
    for t in TICKERS_LOW:
        rows.append({
            "ticker": t, "sector": "Healthcare",
            "industry": "Pharmaceuticals",
            "sic_code": 2834, "is_SPAC": False, "is_shell_company": False,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics summary DataFrame builder (input to apply_hard_filters)
#
# Contains both hard-filter columns and sector/exchange metadata so they
# propagate through survivors_df → ranked_df → shortlist_df for downstream
# distribution reporting.
# ---------------------------------------------------------------------------

_PASS_DEFAULTS: dict[str, object] = {
    "profitable_years": 10,
    "avg_roe": 0.20,
    "eps_cagr": 0.05,
    "debt_payoff_years": 2.0,
    "years_available": 10,
}

# Each FAIL ticker overrides exactly ONE metric so it fails that filter.
_FAIL_OVERRIDES: dict[str, dict[str, object]] = {
    "FAIL_ROE":  {"avg_roe": 0.10},            # below min_avg_roe (0.15)
    "FAIL_EPS":  {"eps_cagr": -0.02},           # not strictly > min_eps_cagr (0.0)
    "FAIL_DEBT": {"debt_payoff_years": 8.0},    # above max_debt_payoff_years (5.0)
    "FAIL_EARN": {"profitable_years": 5},        # below min_profitable_years (8)
    "FAIL_DATA": {"years_available": 3},         # below min_history_years (8)
}

_SECTOR_MAP: dict[str, tuple[str, str]] = {
    "HQ":   ("Technology",         "NASDAQ"),
    "MQ":   ("Consumer Defensive", "NYSE"),
    "FAIL": ("Industrials",        "NYSE"),
    "LOW":  ("Healthcare",         "TSX"),
}


def _build_metrics_summary_df(tickers: list[str]) -> pd.DataFrame:
    """Build metrics_summary_df for *tickers* (post-exclusion).

    Each row carries the five hard-filter columns plus ``sector`` and
    ``exchange`` so downstream shortlist/summary can compute distributions.
    """
    rows: list[dict] = []
    for t in tickers:
        row: dict = {"ticker": t, **_PASS_DEFAULTS}
        if t in _FAIL_OVERRIDES:
            row.update(_FAIL_OVERRIDES[t])
        prefix = t.split("_")[0]
        sector, exchange = _SECTOR_MAP.get(prefix, ("Unknown", "OTC"))
        row["sector"] = sector
        row["exchange"] = exchange
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Composite-score metric builders (12 keys consumed by the 10 criterion
# scorers in composite_score.py).  Each builder produces slightly
# different values per index so all 12 survivors have distinct scores.
# ---------------------------------------------------------------------------


def _hq_metrics(idx: int) -> dict[str, float]:
    """High-quality metrics — composite score expected ~80-97.

    Slight per-ticker variation (via *idx* 0-4) so each HQ ticker gets a
    distinct composite score, enabling deterministic rank ordering.
    """
    return {
        "avg_roe":              0.25 - idx * 0.005,   # 0.250 → 0.230
        "roe_stdev":            0.02,                  # below 0.05 penalty threshold
        "avg_gross_margin":     0.60 - idx * 0.01,     # 0.60 → 0.56
        "avg_sga_ratio":        0.30 + idx * 0.01,     # 0.30 → 0.34
        "eps_cagr":             0.20 - idx * 0.01,     # 0.20 → 0.16
        "decline_years":        0,
        "avg_de_10yr":          0.20 + idx * 0.02,     # 0.20 → 0.28
        "owner_earnings_cagr":  0.18 - idx * 0.01,     # 0.18 → 0.14
        "avg_capex_to_ni":      0.25 + idx * 0.02,     # 0.25 → 0.33
        "buyback_pct":          0.15 - idx * 0.01,     # 0.15 → 0.11
        "return_on_retained":   0.15 - idx * 0.005,    # 0.150 → 0.130
        "avg_interest_pct_10yr": 0.08,                 # below excellent (0.10) → 100
    }


def _mq_metrics(idx: int) -> dict[str, float]:
    """Medium-quality metrics — composite score expected ~40-55.

    Moderate ROE, low growth, middling debt and efficiency ratios.
    """
    return {
        "avg_roe":              0.17 + idx * 0.002,    # 0.170 → 0.178
        "roe_stdev":            0.03,                  # below penalty threshold
        "avg_gross_margin":     0.40 - idx * 0.01,     # 0.40 → 0.36
        "avg_sga_ratio":        0.55 + idx * 0.02,     # 0.55 → 0.63
        "eps_cagr":             0.08 - idx * 0.005,    # 0.08 → 0.06
        "decline_years":        1,                     # 0.9× consistency multiplier
        "avg_de_10yr":          0.50 + idx * 0.02,     # 0.50 → 0.58
        "owner_earnings_cagr":  0.06 - idx * 0.005,    # 0.06 → 0.04
        "avg_capex_to_ni":      0.50 + idx * 0.02,     # 0.50 → 0.58
        "buyback_pct":          0.02 - idx * 0.005,    # 0.020 → 0.000
        "return_on_retained":   0.12 - idx * 0.005,    # 0.120 → 0.100
        "avg_interest_pct_10yr": 0.15 + idx * 0.005,   # 0.15 → 0.17
    }


def _low_metrics(idx: int) -> dict[str, float]:
    """Low-score metrics — composite score expected ~10-20.

    Barely passes hard filters but scores poorly on every soft criterion.
    ROE stdev above penalty threshold, 3 decline years (worst multiplier).
    """
    return {
        "avg_roe":              0.16,
        "roe_stdev":            0.06,                  # above 0.05 penalty threshold
        "avg_gross_margin":     0.22 - idx * 0.01,     # 0.22 → 0.21
        "avg_sga_ratio":        0.72 + idx * 0.02,     # 0.72 → 0.74
        "eps_cagr":             0.01,
        "decline_years":        3,                     # 0.6× consistency multiplier
        "avg_de_10yr":          0.75 + idx * 0.02,     # 0.75 → 0.77
        "owner_earnings_cagr":  0.01,
        "avg_capex_to_ni":      0.70 + idx * 0.02,     # 0.70 → 0.72
        "buyback_pct":          -0.03,                 # slight dilution
        "return_on_retained":   0.03 - idx * 0.005,    # 0.030 → 0.025
        "avg_interest_pct_10yr": 0.28 + idx * 0.01,    # 0.28 → 0.29
    }


def _build_all_metrics(tickers: list[str]) -> dict[str, dict[str, float]]:
    """Build ``{ticker: metrics_dict}`` for ``compute_all_composite_scores``."""
    result: dict[str, dict[str, float]] = {}
    for t in tickers:
        prefix, num = t.split("_", 1)
        idx = int(num) - 1
        if prefix == "HQ":
            result[t] = _hq_metrics(idx)
        elif prefix == "MQ":
            result[t] = _mq_metrics(idx)
        elif prefix == "LOW":
            result[t] = _low_metrics(idx)
    return result


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------


def _run_pipeline() -> dict:
    """Execute the full screening pipeline and return all intermediate outputs.

    Steps: exclusions → hard_filters → composite_scores → soft_scores →
    shortlist → summary.
    """
    # --- Step 1: Exclusions — remove financial-sector tickers ---
    universe_df = _build_universe_df()
    filtered_df, exclusion_log_df = apply_exclusions(universe_df)

    # --- Step 2: Build metrics summary for surviving tickers ---
    surviving_tickers = sorted(filtered_df["ticker"].tolist())
    metrics_summary_df = _build_metrics_summary_df(surviving_tickers)

    # --- Step 3: Hard filters — remove tickers failing any Tier 1 check ---
    survivors_df, filter_log_df = apply_hard_filters(metrics_summary_df)

    # --- Step 4: Composite scores for hard-filter survivors ---
    #     Uses the real compute_all_composite_scores engine (not pre-baked).
    survivor_list = survivors_df["ticker"].tolist()
    all_metrics = _build_all_metrics(survivor_list)
    composite_scores_df = compute_all_composite_scores(all_metrics)

    # --- Step 5: Soft scoring — join composite scores, add rank ---
    ranked_df = apply_soft_scores(survivors_df, composite_scores_df)

    # --- Step 6: Shortlist — top 10 with percentile and score_category ---
    shortlist_df = generate_shortlist(ranked_df, top_n=10)

    # --- Step 7: Summary statistics ---
    summary = generate_screener_summary(
        ranked_df, shortlist_df, filter_log_df,
        total_universe=len(universe_df),
    )

    return {
        "universe_df": universe_df,
        "filtered_df": filtered_df,
        "exclusion_log_df": exclusion_log_df,
        "metrics_summary_df": metrics_summary_df,
        "survivors_df": survivors_df,
        "filter_log_df": filter_log_df,
        "composite_scores_df": composite_scores_df,
        "ranked_df": ranked_df,
        "shortlist_df": shortlist_df,
        "summary": summary,
    }


@pytest.fixture(scope="module")
def pipeline() -> dict:
    """Run the full pipeline once and cache results for all tests."""
    return _run_pipeline()


# ===========================================================================
# 1. Exclusion stage — 3 financial-sector tickers removed
# ===========================================================================


class TestExclusionStage:
    """Financial-sector tickers removed before Tier 1 hard filtering."""

    def test_three_financial_tickers_excluded(self, pipeline: dict) -> None:
        """Exactly 3 financial-sector tickers are excluded."""
        excluded = set(pipeline["exclusion_log_df"]["ticker"])
        assert excluded == set(TICKERS_FIN)

    def test_exclusion_log_has_reasons(self, pipeline: dict) -> None:
        """Every excluded ticker has a non-empty reason string."""
        log = pipeline["exclusion_log_df"]
        for _, row in log.iterrows():
            assert len(str(row["reason"])) > 0, (
                f"Empty reason for excluded ticker {row['ticker']}"
            )

    def test_seventeen_tickers_survive_exclusions(self, pipeline: dict) -> None:
        """17 of 20 tickers survive the exclusion stage."""
        assert len(pipeline["filtered_df"]) == 17

    def test_no_financial_tickers_in_filtered(self, pipeline: dict) -> None:
        """No financial-sector ticker appears in the filtered output."""
        surviving = set(pipeline["filtered_df"]["ticker"])
        assert surviving.isdisjoint(set(TICKERS_FIN))

    def test_all_non_financial_tickers_survive(self, pipeline: dict) -> None:
        """All 17 non-financial tickers survive exclusions."""
        surviving = set(pipeline["filtered_df"]["ticker"])
        assert surviving == set(TICKERS_AFTER_EXCLUSIONS)


# ===========================================================================
# 2. Hard filter (Tier 1) stage — 5 tickers fail
# ===========================================================================


class TestHardFilterStage:
    """Five hard-filter-failure tickers removed by Tier 1."""

    def test_five_tickers_fail_hard_filters(self, pipeline: dict) -> None:
        """Exactly 5 tickers are removed by hard filters."""
        n_in = len(pipeline["metrics_summary_df"])
        n_out = len(pipeline["survivors_df"])
        assert n_in - n_out == 5

    def test_correct_tickers_fail(self, pipeline: dict) -> None:
        """The 5 FAIL tickers are not in the survivors."""
        surviving = set(pipeline["survivors_df"]["ticker"])
        assert surviving.isdisjoint(set(TICKERS_FAIL))

    def test_twelve_survivors(self, pipeline: dict) -> None:
        """12 tickers survive hard filters (5 HQ + 5 MQ + 2 LOW)."""
        assert len(pipeline["survivors_df"]) == 12

    def test_all_hq_mq_low_survive(self, pipeline: dict) -> None:
        """All HQ, MQ, and LOW tickers survive hard filters."""
        surviving = set(pipeline["survivors_df"]["ticker"])
        assert surviving == set(TICKERS_AFTER_HARD)

    def test_fail_roe_fails_roe_filter(self, pipeline: dict) -> None:
        """FAIL_ROE (ROE = 10 %) fails the roe_floor filter specifically."""
        log = pipeline["filter_log_df"]
        row = log[(log["ticker"] == "FAIL_ROE") & (log["filter_name"] == "roe_floor")]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])

    def test_fail_eps_fails_eps_growth_filter(self, pipeline: dict) -> None:
        """FAIL_EPS (CAGR = −2 %) fails the eps_growth filter specifically."""
        log = pipeline["filter_log_df"]
        row = log[(log["ticker"] == "FAIL_EPS") & (log["filter_name"] == "eps_growth")]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])

    def test_fail_debt_fails_debt_filter(self, pipeline: dict) -> None:
        """FAIL_DEBT (8 yr payoff) fails the debt_sustainability filter."""
        log = pipeline["filter_log_df"]
        row = log[
            (log["ticker"] == "FAIL_DEBT")
            & (log["filter_name"] == "debt_sustainability")
        ]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])

    def test_fail_earn_fails_earnings_filter(self, pipeline: dict) -> None:
        """FAIL_EARN (5 profitable years) fails earnings_consistency."""
        log = pipeline["filter_log_df"]
        row = log[
            (log["ticker"] == "FAIL_EARN")
            & (log["filter_name"] == "earnings_consistency")
        ]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])

    def test_fail_data_fails_data_sufficiency_filter(self, pipeline: dict) -> None:
        """FAIL_DATA (3 years available) fails data_sufficiency."""
        log = pipeline["filter_log_df"]
        row = log[
            (log["ticker"] == "FAIL_DATA")
            & (log["filter_name"] == "data_sufficiency")
        ]
        assert len(row) == 1
        assert not bool(row.iloc[0]["pass_fail"])


# ===========================================================================
# 3. Filter log completeness
# ===========================================================================


class TestFilterLogCompleteness:
    """filter_log_df has entries for every ticker × every filter."""

    def test_every_ticker_has_entry(self, pipeline: dict) -> None:
        """Every post-exclusion ticker appears in the filter log."""
        log_tickers = set(pipeline["filter_log_df"]["ticker"])
        expected = set(TICKERS_AFTER_EXCLUSIONS)
        assert log_tickers == expected

    def test_every_filter_applied(self, pipeline: dict) -> None:
        """All 5 hard filters are present in the filter log."""
        filters = set(pipeline["filter_log_df"]["filter_name"])
        assert filters == EXPECTED_FILTERS

    def test_rows_per_ticker(self, pipeline: dict) -> None:
        """Each ticker has exactly 5 rows (one per hard filter)."""
        log = pipeline["filter_log_df"]
        counts = log.groupby("ticker").size()
        assert (counts == N_HARD_FILTERS).all()

    def test_total_filter_log_rows(self, pipeline: dict) -> None:
        """Total rows = 17 tickers × 5 filters = 85."""
        assert len(pipeline["filter_log_df"]) == 17 * N_HARD_FILTERS

    def test_filter_log_columns(self, pipeline: dict) -> None:
        """Filter log must have the expected column schema."""
        expected_cols = {
            "ticker", "filter_name", "filter_value", "threshold", "pass_fail",
        }
        assert set(pipeline["filter_log_df"].columns) == expected_cols

    def test_survivors_pass_all_five_filters(self, pipeline: dict) -> None:
        """Every surviving ticker must have pass_fail=True on all 5 filters."""
        log = pipeline["filter_log_df"]
        for ticker in TICKERS_AFTER_HARD:
            ticker_log = log[log["ticker"] == ticker]
            assert ticker_log["pass_fail"].all(), (
                f"Survivor {ticker} has at least one False in filter log"
            )


# ===========================================================================
# 4. Shortlist properties
# ===========================================================================


class TestShortlistProperties:
    """Shortlist contains ≤ 10 stocks with correct ordering and metadata."""

    def test_shortlist_size_at_most_10(self, pipeline: dict) -> None:
        """Shortlist contains ≤ 10 stocks (12 survivors, top_n=10)."""
        assert len(pipeline["shortlist_df"]) <= 10

    def test_shortlist_size_exactly_10(self, pipeline: dict) -> None:
        """With 12 survivors and top_n=10, shortlist has exactly 10 rows."""
        assert len(pipeline["shortlist_df"]) == 10

    def test_ordered_by_composite_score_descending(self, pipeline: dict) -> None:
        """Shortlist is sorted by composite_score descending."""
        scores = pipeline["shortlist_df"]["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_no_adjacent_ties(self, pipeline: dict) -> None:
        """No two adjacent shortlist entries share the same composite_score.

        Mock data is designed with per-ticker variation so all 12 survivors
        have distinct composite scores, ensuring deterministic ordering.
        """
        scores = pipeline["shortlist_df"]["composite_score"].tolist()
        for i in range(len(scores) - 1):
            assert scores[i] != scores[i + 1], (
                f"Tie at positions {i} and {i + 1}: {scores[i]}"
            )

    def test_shortlist_outscores_non_shortlist(self, pipeline: dict) -> None:
        """Every shortlist stock has composite_score > every non-shortlist survivor."""
        ranked = pipeline["ranked_df"]
        shortlist = pipeline["shortlist_df"]
        shortlist_tickers = set(shortlist["ticker"])

        non_shortlist = ranked[~ranked["ticker"].isin(shortlist_tickers)]
        assert not non_shortlist.empty, "Expected 2 non-shortlist survivors."

        min_shortlist_score = shortlist["composite_score"].min()
        max_non_shortlist_score = non_shortlist["composite_score"].max()
        assert min_shortlist_score > max_non_shortlist_score, (
            f"Min shortlist ({min_shortlist_score:.1f}) must exceed "
            f"max non-shortlist ({max_non_shortlist_score:.1f})"
        )

    def test_hq_tickers_all_in_shortlist(self, pipeline: dict) -> None:
        """All 5 high-quality tickers make the shortlist."""
        shortlisted = set(pipeline["shortlist_df"]["ticker"])
        assert set(TICKERS_HQ).issubset(shortlisted)

    def test_mq_tickers_all_in_shortlist(self, pipeline: dict) -> None:
        """All 5 medium-quality tickers make the shortlist (ranked 6-10)."""
        shortlisted = set(pipeline["shortlist_df"]["ticker"])
        assert set(TICKERS_MQ).issubset(shortlisted)

    def test_low_tickers_not_in_shortlist(self, pipeline: dict) -> None:
        """Both LOW tickers are ranked below the top-10 cutoff."""
        shortlisted = set(pipeline["shortlist_df"]["ticker"])
        assert shortlisted.isdisjoint(set(TICKERS_LOW))

    def test_rank_column_present_and_sequential(self, pipeline: dict) -> None:
        """Shortlist has a rank column starting at 1 with sequential values."""
        ranks = pipeline["shortlist_df"]["rank"].tolist()
        assert ranks == sorted(ranks)
        assert ranks[0] == 1

    def test_percentile_column_in_valid_range(self, pipeline: dict) -> None:
        """Shortlist has a percentile column with values in (0, 100]."""
        pctiles = pipeline["shortlist_df"]["percentile"].tolist()
        assert all(0 < p <= 100 for p in pctiles)

    def test_score_category_column_valid(self, pipeline: dict) -> None:
        """Shortlist has a score_category column with valid values only."""
        valid = {"Strong Buy", "Buy", "Hold", "Weak"}
        categories = set(pipeline["shortlist_df"]["score_category"])
        assert categories.issubset(valid)

    def test_metadata_columns_present(self, pipeline: dict) -> None:
        """Shortlist must include rank, percentile, and score_category."""
        cols = set(pipeline["shortlist_df"].columns)
        assert {"rank", "percentile", "score_category"}.issubset(cols)


# ===========================================================================
# 5. Composite score tier ranking
# ===========================================================================


class TestCompositeScoreRanking:
    """Composite scores reflect quality tiers: HQ > MQ > LOW."""

    def test_hq_scores_above_mq(self, pipeline: dict) -> None:
        """Every HQ ticker scores higher than every MQ ticker."""
        ranked = pipeline["ranked_df"]
        hq = ranked[ranked["ticker"].isin(TICKERS_HQ)]["composite_score"]
        mq = ranked[ranked["ticker"].isin(TICKERS_MQ)]["composite_score"]
        assert hq.min() > mq.max(), (
            f"Worst HQ ({hq.min():.1f}) must beat best MQ ({mq.max():.1f})"
        )

    def test_mq_scores_above_low(self, pipeline: dict) -> None:
        """Every MQ ticker scores higher than every LOW ticker."""
        ranked = pipeline["ranked_df"]
        mq = ranked[ranked["ticker"].isin(TICKERS_MQ)]["composite_score"]
        low = ranked[ranked["ticker"].isin(TICKERS_LOW)]["composite_score"]
        assert mq.min() > low.max(), (
            f"Worst MQ ({mq.min():.1f}) must beat best LOW ({low.max():.1f})"
        )

    def test_all_composite_scores_in_range(self, pipeline: dict) -> None:
        """All composite scores lie in [0, 100]."""
        scores = pipeline["ranked_df"]["composite_score"]
        assert (scores >= 0).all()
        assert (scores <= 100).all()

    def test_no_nan_composite_scores(self, pipeline: dict) -> None:
        """No NaN values in composite_score for any survivor."""
        assert pipeline["ranked_df"]["composite_score"].notna().all()

    def test_hq_01_is_rank_1(self, pipeline: dict) -> None:
        """HQ_01 (best HQ metrics across all criteria) must be ranked first."""
        top = pipeline["shortlist_df"].iloc[0]
        assert top["ticker"] == "HQ_01"
        assert top["rank"] == 1

    def test_hq_tickers_score_above_70(self, pipeline: dict) -> None:
        """All HQ tickers score above 70 (at least 'Buy' territory)."""
        ranked = pipeline["ranked_df"]
        hq = ranked[ranked["ticker"].isin(TICKERS_HQ)]["composite_score"]
        assert (hq > 70).all(), (
            f"HQ scores: {hq.tolist()} — expected all > 70"
        )

    def test_low_tickers_score_below_30(self, pipeline: dict) -> None:
        """LOW tickers score below 30 (deeply in 'Weak' territory)."""
        ranked = pipeline["ranked_df"]
        low = ranked[ranked["ticker"].isin(TICKERS_LOW)]["composite_score"]
        assert (low < 30).all(), (
            f"LOW scores: {low.tolist()} — expected all < 30"
        )

    def test_twelve_distinct_scores(self, pipeline: dict) -> None:
        """All 12 survivors have distinct composite scores (no ties)."""
        scores = pipeline["ranked_df"]["composite_score"].tolist()
        assert len(set(scores)) == 12


# ===========================================================================
# 6. Screener summary
# ===========================================================================


class TestScreenerSummary:
    """generate_screener_summary produces correct pipeline counts."""

    def test_total_universe_count(self, pipeline: dict) -> None:
        """total_universe equals 20 (all tickers before any filtering)."""
        assert pipeline["summary"]["total_universe"] == 20

    def test_after_exclusions_count(self, pipeline: dict) -> None:
        """after_exclusions = 17 unique tickers in filter_log_df."""
        assert pipeline["summary"]["after_exclusions"] == 17

    def test_after_tier1_count(self, pipeline: dict) -> None:
        """after_tier1 = 12 (survivors of hard filters)."""
        assert pipeline["summary"]["after_tier1"] == 12

    def test_shortlisted_count(self, pipeline: dict) -> None:
        """shortlisted = 10 (top_n=10 of 12 survivors)."""
        assert pipeline["summary"]["shortlisted"] == 10

    def test_score_statistics_present(self, pipeline: dict) -> None:
        """Summary contains top_score, median_score, bottom_score — all numeric."""
        s = pipeline["summary"]
        for key in ("top_score", "median_score", "bottom_score"):
            assert key in s
            assert isinstance(s[key], (int, float))

    def test_top_score_equals_max_ranked(self, pipeline: dict) -> None:
        """top_score equals the highest composite_score in ranked_df."""
        expected = float(pipeline["ranked_df"]["composite_score"].max())
        assert abs(pipeline["summary"]["top_score"] - expected) < 1e-6

    def test_bottom_score_equals_min_ranked(self, pipeline: dict) -> None:
        """bottom_score equals the lowest composite_score in ranked_df."""
        expected = float(pipeline["ranked_df"]["composite_score"].min())
        assert abs(pipeline["summary"]["bottom_score"] - expected) < 1e-6

    def test_sector_distribution_sums_to_shortlist_size(self, pipeline: dict) -> None:
        """sector_distribution values sum to 10 (total shortlisted)."""
        dist = pipeline["summary"]["sector_distribution"]
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 10

    def test_exchange_distribution_sums_to_shortlist_size(self, pipeline: dict) -> None:
        """exchange_distribution values sum to 10 (total shortlisted)."""
        dist = pipeline["summary"]["exchange_distribution"]
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 10

    def test_summary_has_all_required_keys(self, pipeline: dict) -> None:
        """Summary dict contains every key defined in the data contract."""
        required = {
            "total_universe", "after_exclusions", "after_tier1", "shortlisted",
            "top_score", "median_score", "bottom_score",
            "sector_distribution", "exchange_distribution",
        }
        assert required.issubset(set(pipeline["summary"].keys()))


# ===========================================================================
# 7. Idempotency — pipeline produces identical results on repeat
# ===========================================================================


class TestIdempotency:
    """Running the same pipeline twice produces identical results."""

    @pytest.fixture(scope="class")
    def second_run(self) -> dict:
        """Run the pipeline a second time independently."""
        return _run_pipeline()

    def test_exclusion_log_identical(
        self, pipeline: dict, second_run: dict,
    ) -> None:
        """Exclusion logs match across two independent runs."""
        pd.testing.assert_frame_equal(
            pipeline["exclusion_log_df"].reset_index(drop=True),
            second_run["exclusion_log_df"].reset_index(drop=True),
        )

    def test_filter_log_identical(
        self, pipeline: dict, second_run: dict,
    ) -> None:
        """Hard-filter logs match across two independent runs."""
        a = pipeline["filter_log_df"].sort_values(
            ["ticker", "filter_name"],
        ).reset_index(drop=True)
        b = second_run["filter_log_df"].sort_values(
            ["ticker", "filter_name"],
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_shortlist_tickers_identical(
        self, pipeline: dict, second_run: dict,
    ) -> None:
        """Shortlist tickers and ordering match across runs."""
        assert list(pipeline["shortlist_df"]["ticker"]) == list(
            second_run["shortlist_df"]["ticker"]
        )

    def test_shortlist_scores_identical(
        self, pipeline: dict, second_run: dict,
    ) -> None:
        """Shortlist composite scores are numerically identical across runs."""
        assert list(pipeline["shortlist_df"]["composite_score"]) == list(
            second_run["shortlist_df"]["composite_score"]
        )

    def test_summary_counts_identical(
        self, pipeline: dict, second_run: dict,
    ) -> None:
        """Summary counts match across two independent runs."""
        for key in ("total_universe", "after_exclusions", "after_tier1",
                    "shortlisted", "top_score", "median_score", "bottom_score"):
            assert pipeline["summary"][key] == second_run["summary"][key], (
                f"Summary {key!r} differs: "
                f"{pipeline['summary'][key]} vs {second_run['summary'][key]}"
            )

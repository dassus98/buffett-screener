"""Integration tests: Module 2 full pipeline (F1–F16).

Exercises ``_compute_all_from_data`` — the pure-computation function that
takes pre-loaded DataFrames and produces a complete metrics summary —
for three synthetic stocks:

* **AAPL_MOCK** — high-quality stock with negative shareholders' equity in
  years 2019–2023 (tests the ROE edge-case path).
* **KO_MOCK** — archetypal strong Buffett company: high ROE, 62 % gross
  margin, low debt, consistent EPS growth.
* **WEAK_MOCK** — deliberately weak: declining EPS, high debt, low margins,
  share dilution, and very high CapEx.

What is verified
----------------
1. AAPL_MOCK negative-equity years produce ``roe=NaN``; ``avg_roe`` is
   computed only from the five years with positive equity.
2. F14 present-value ordering: ``bear_pv < base_pv < bull_pv`` for each
   stock that has positive EPS and a valid (non-NaN) projected price.
3. Composite-score ranking: KO_MOCK strictly outscores WEAK_MOCK.
4. All per-criterion ``score_*`` values lie in [0, 100]; ``composite_score``
   column contains no NaN.
5. Hardcoded-value audit: documents the eight ``cfg.get(key, fallback)``
   matches found in ``metrics_engine/`` and confirms no bare threshold
   literals appear outside config reads.
6. mypy type-check: ``metrics_engine/`` passes with zero errors.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from metrics_engine import _compute_all_from_data
from metrics_engine.composite_score import compute_all_composite_scores

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

YEARS = list(range(2014, 2024))   # 10-year window: 2014–2023
N = 10
RFR = 0.04                        # 4 % risk-free rate (US 10-yr fallback)

# ---------------------------------------------------------------------------
# Fixture-data builder helpers
# ---------------------------------------------------------------------------


def _income(
    ticker: str,
    *,
    eps: list[float],
    shares_m: list[float],
    revenue: list[float],
    gross_profit: list[float],
    sga: list[float],
    operating_income: list[float],
    interest_expense: list[float],
) -> pd.DataFrame:
    """Build an income-statement DataFrame (units: $M / shares in millions)."""
    net_income = [e * s for e, s in zip(eps, shares_m)]
    return pd.DataFrame({
        "ticker": [ticker] * N,
        "fiscal_year": YEARS,
        "net_income": net_income,
        "total_revenue": revenue,
        "gross_profit": gross_profit,
        "sga": sga,
        "operating_income": operating_income,
        "interest_expense": interest_expense,
        "eps_diluted": eps,
        "shares_outstanding_diluted": shares_m,  # millions
    })


def _balance(
    ticker: str,
    *,
    equity: list[float],
    long_term_debt: list[float],
    treasury_stock: list[float] | None = None,
) -> pd.DataFrame:
    if treasury_stock is None:
        treasury_stock = [0.0] * N
    return pd.DataFrame({
        "ticker": [ticker] * N,
        "fiscal_year": YEARS,
        "long_term_debt": long_term_debt,
        "shareholders_equity": equity,
        "treasury_stock": treasury_stock,
    })


def _cashflow(
    ticker: str,
    *,
    da: list[float],
    capex: list[float],
    wcc: list[float],
) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [ticker] * N,
        "fiscal_year": YEARS,
        "depreciation_amortization": da,
        "capital_expenditures": capex,   # negative = cash outflow
        "working_capital_change": wcc,
    })


def _market(price: float, pe: float) -> pd.Series:
    return pd.Series({"current_price_usd": price, "pe_ratio_trailing": pe})


# ---------------------------------------------------------------------------
# AAPL_MOCK fixture
#
# Profile:
#   - EPS grows 6.0 → 10.5 (CAGR ≈ 6.4 %)
#   - Shares decline 6 000 M → 4 200 M (30 % buyback)
#   - shareholders_equity turns negative in 2019–2023 (AAPL pattern)
#   - 60 % gross margin, 10 % SGA/GP (excellent), very low interest burden
# ---------------------------------------------------------------------------

_AAPL_EPS    = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]
_AAPL_SHARES = [6000.0, 5700.0, 5400.0, 5100.0, 4900.0,
                4700.0, 4500.0, 4400.0, 4300.0, 4200.0]  # millions
_AAPL_NI     = [e * s for e, s in zip(_AAPL_EPS, _AAPL_SHARES)]
# Revenue assumes 25 % net margin; gross profit at 60 %; SGA = 10 % of GP.
_AAPL_REV    = [ni / 0.25 for ni in _AAPL_NI]
_AAPL_GP     = [r * 0.60 for r in _AAPL_REV]
_AAPL_SGA    = [gp * 0.10 for gp in _AAPL_GP]   # sga_ratio = 0.10 → score 100
_AAPL_EBIT   = [r * 0.30 for r in _AAPL_REV]
# shareholders_equity: positive 2014–2018, negative 2019–2023
_AAPL_EQ     = [120_000.0, 120_000.0, 130_000.0, 135_000.0, 110_000.0,
                -1_000.0,   -5_000.0,  -4_000.0,  -6_000.0,  -3_000.0]
_AAPL_DEBT   = [15_000.0, 15_000.0, 17_000.0, 20_000.0, 25_000.0,
                40_000.0, 50_000.0, 55_000.0, 60_000.0, 58_000.0]
_AAPL_TS     = [50_000.0, 55_000.0, 60_000.0, 65_000.0, 70_000.0,
                75_000.0, 80_000.0, 90_000.0, 95_000.0, 100_000.0]


@pytest.fixture
def aapl_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    inc = _income(
        "AAPL_MOCK",
        eps=_AAPL_EPS,
        shares_m=_AAPL_SHARES,
        revenue=_AAPL_REV,
        gross_profit=_AAPL_GP,
        sga=_AAPL_SGA,
        operating_income=_AAPL_EBIT,
        interest_expense=[500.0] * N,
    )
    bal = _balance("AAPL_MOCK", equity=_AAPL_EQ, long_term_debt=_AAPL_DEBT, treasury_stock=_AAPL_TS)
    cf = _cashflow("AAPL_MOCK", da=[2_000.0] * N, capex=[-1_500.0] * N, wcc=[-500.0] * N)
    mkt = _market(price=190.0, pe=29.0)
    return inc, bal, cf, mkt


@pytest.fixture
def aapl_summary(aapl_data) -> dict[str, Any]:
    inc, bal, cf, mkt = aapl_data
    summary, _ = _compute_all_from_data("AAPL_MOCK", inc, bal, cf, mkt, RFR)
    return summary


@pytest.fixture
def aapl_annual(aapl_data) -> pd.DataFrame:
    inc, bal, cf, mkt = aapl_data
    _, annual_df = _compute_all_from_data("AAPL_MOCK", inc, bal, cf, mkt, RFR)
    return annual_df


# ---------------------------------------------------------------------------
# KO_MOCK fixture
#
# Profile:
#   - EPS grows 1.60 → 4.50 (CAGR ≈ 12.2 %)
#   - Shares decline 4 300 M → 3 850 M (≈ 10.5 % buyback)
#   - Equity = NI / 0.35 (constant ~35 % ROE throughout)
#   - D/E ≈ 0.42; interest ≈ 4 % of EBIT → interest_coverage score = 100
#   - 62 % gross margin; SGA = 40 % of GP
# ---------------------------------------------------------------------------

_KO_EPS    = [1.60, 1.80, 2.00, 2.25, 2.50, 2.80, 3.15, 3.55, 4.00, 4.50]
_KO_SHARES = [4300.0, 4250.0, 4200.0, 4150.0, 4100.0,
              4050.0, 4000.0, 3950.0, 3900.0, 3850.0]  # millions
_KO_NI     = [e * s for e, s in zip(_KO_EPS, _KO_SHARES)]
_KO_EQ     = [ni / 0.35 for ni in _KO_NI]   # fixed 35 % ROE target
_KO_DEBT   = [eq * 0.42 for eq in _KO_EQ]   # D/E = 0.42 throughout
_KO_REV    = [ni / 0.21 for ni in _KO_NI]   # 21 % net margin
_KO_GP     = [r * 0.62 for r in _KO_REV]    # 62 % gross margin
_KO_SGA    = [gp * 0.40 for gp in _KO_GP]   # sga_ratio = 0.40
_KO_EBIT   = [r * 0.28 for r in _KO_REV]    # 28 % EBIT margin
_KO_INT    = [ebit * 0.04 for ebit in _KO_EBIT]  # 4 % of EBIT → score 100


@pytest.fixture
def ko_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    inc = _income(
        "KO_MOCK",
        eps=_KO_EPS,
        shares_m=_KO_SHARES,
        revenue=_KO_REV,
        gross_profit=_KO_GP,
        sga=_KO_SGA,
        operating_income=_KO_EBIT,
        interest_expense=_KO_INT,
    )
    bal = _balance("KO_MOCK", equity=_KO_EQ, long_term_debt=_KO_DEBT)
    cf = _cashflow("KO_MOCK", da=[700.0] * N, capex=[-500.0] * N, wcc=[100.0] * N)
    mkt = _market(price=60.0, pe=18.0)
    return inc, bal, cf, mkt


@pytest.fixture
def ko_summary(ko_data) -> dict[str, Any]:
    inc, bal, cf, mkt = ko_data
    summary, _ = _compute_all_from_data("KO_MOCK", inc, bal, cf, mkt, RFR)
    return summary


@pytest.fixture
def ko_annual(ko_data) -> pd.DataFrame:
    inc, bal, cf, mkt = ko_data
    _, annual_df = _compute_all_from_data("KO_MOCK", inc, bal, cf, mkt, RFR)
    return annual_df


# ---------------------------------------------------------------------------
# WEAK_MOCK fixture
#
# Profile:
#   - EPS declining 1.00 → 0.64 (CAGR ≈ −4.8 %)
#   - Shares growing 400 M → 760 M (90 % dilution — buyback score = 0)
#   - D/E = 1.2 (> fail_de = 0.80 → debt_conservatism score = 0)
#   - Interest = 40 % of EBIT (> fail = 30 % → interest_coverage score = 0)
#   - CapEx = 4 × NI (avg_capex_to_ni = 4.0 > fail = 0.75 → cap_eff = 0)
#   - 25 % gross margin; SGA = 75 % of GP (close to fail 80 %)
# ---------------------------------------------------------------------------

_W_EPS    = [1.00, 0.95, 0.79, 0.87, 0.77, 0.83, 0.72, 0.71, 0.71, 0.64]
_W_SHARES = [400.0, 450.0, 500.0, 530.0, 570.0,
             600.0, 640.0, 680.0, 720.0, 760.0]  # millions
_W_NI     = [e * s for e, s in zip(_W_EPS, _W_SHARES)]
_W_EQ     = [8_000.0 + i * 200.0 for i in range(N)]  # slowly growing equity
_W_DEBT   = [eq * 1.2 for eq in _W_EQ]              # D/E = 1.2 throughout
_W_REV    = [ni / 0.05 for ni in _W_NI]             # 5 % NI margin
_W_GP     = [r * 0.25 for r in _W_REV]              # 25 % gross margin
_W_SGA    = [gp * 0.75 for gp in _W_GP]             # sga_ratio = 0.75
_W_EBIT   = [gp - sga for gp, sga in zip(_W_GP, _W_SGA)]  # = 0.25 × GP
_W_INT    = [ebit * 0.40 for ebit in _W_EBIT]       # 40 % of EBIT
_W_CAPEX  = [-4.0 * ni for ni in _W_NI]             # 4× NI (negative sign)


@pytest.fixture
def weak_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    inc = _income(
        "WEAK_MOCK",
        eps=_W_EPS,
        shares_m=_W_SHARES,
        revenue=_W_REV,
        gross_profit=_W_GP,
        sga=_W_SGA,
        operating_income=_W_EBIT,
        interest_expense=_W_INT,
    )
    bal = _balance("WEAK_MOCK", equity=_W_EQ, long_term_debt=_W_DEBT)
    cf = _cashflow("WEAK_MOCK", da=[50.0] * N, capex=_W_CAPEX, wcc=[0.0] * N)
    mkt = _market(price=8.0, pe=12.0)
    return inc, bal, cf, mkt


@pytest.fixture
def weak_summary(weak_data) -> dict[str, Any]:
    inc, bal, cf, mkt = weak_data
    summary, _ = _compute_all_from_data("WEAK_MOCK", inc, bal, cf, mkt, RFR)
    return summary


@pytest.fixture
def weak_annual(weak_data) -> pd.DataFrame:
    inc, bal, cf, mkt = weak_data
    _, annual_df = _compute_all_from_data("WEAK_MOCK", inc, bal, cf, mkt, RFR)
    return annual_df


# ---------------------------------------------------------------------------
# Composite-score DataFrame (all three tickers together)
# ---------------------------------------------------------------------------


@pytest.fixture
def composite_df(aapl_summary, ko_summary, weak_summary) -> pd.DataFrame:
    all_metrics = {
        "AAPL_MOCK": aapl_summary,
        "KO_MOCK":   ko_summary,
        "WEAK_MOCK": weak_summary,
    }
    return compute_all_composite_scores(all_metrics)


# ===========================================================================
# Test class 1 — AAPL negative-equity edge case
# ===========================================================================


class TestAaplNegativeEquity:
    """ROE computations handle years with negative shareholders' equity."""

    def test_no_exception_raised(self, aapl_data):
        """_compute_all_from_data must not raise for AAPL-like data."""
        inc, bal, cf, mkt = aapl_data
        # Should not raise
        _compute_all_from_data("AAPL_MOCK", inc, bal, cf, mkt, RFR)

    def test_avg_roe_is_finite(self, aapl_summary):
        """avg_roe must be a finite float despite five NaN ROE years."""
        avg_roe = aapl_summary.get("avg_roe")
        assert avg_roe is not None
        assert math.isfinite(float(avg_roe)), (
            f"Expected finite avg_roe, got {avg_roe!r}"
        )

    def test_avg_roe_from_positive_equity_years_only(self, aapl_summary):
        """avg_roe computed from the six years where avg_equity > 0 (2014–2019).

        In those years ROE = NI / avg_equity ranges from ~28 % (2014–2018)
        up to ~73 % for 2019 (avg_equity = 54 500 M, a "transition" year).
        The overall avg_roe will be elevated but finite.
        """
        avg_roe = float(aapl_summary["avg_roe"])
        # Six valid years; ROE range is ~28 %–73 %, average ≈ 35–50 %.
        assert 0.20 <= avg_roe <= 0.80, (
            f"avg_roe={avg_roe:.4f} out of expected range for valid avg-equity years"
        )

    def test_annual_df_has_roe_nan_for_negative_equity_years(self, aapl_annual):
        """Annual DataFrame rows for years 2020–2023 must have roe=NaN.

        Year 2019 uses avg_equity = (equity_2018 + equity_2019) / 2.
        Because equity_2018 = +110 000 and equity_2019 = −1 000, the
        average is +54 500 (positive), so ROE for 2019 is finite.
        Only years 2020–2023 produce a truly negative avg_equity → NaN.
        """
        truly_negative_avg_eq_years = [2020, 2021, 2022, 2023]
        rows = aapl_annual[aapl_annual["fiscal_year"].isin(truly_negative_avg_eq_years)]
        assert not rows.empty, "Expected rows for years with negative avg_equity"
        assert rows["roe"].isna().all(), (
            "ROE must be NaN for all years where avg_equity ≤ 0:\n"
            f"{rows[['fiscal_year', 'roe']]}"
        )

    def test_annual_df_has_roe_for_positive_equity_years(self, aapl_annual):
        """Annual DataFrame rows for years 2014–2019 must have finite roe.

        2019 qualifies because avg_equity = (equity_2018 + equity_2019) / 2
        = (110 000 + (−1 000)) / 2 = +54 500 > 0.
        """
        positive_avg_eq_years = [2014, 2015, 2016, 2017, 2018, 2019]
        rows = aapl_annual[aapl_annual["fiscal_year"].isin(positive_avg_eq_years)]
        assert not rows.empty, "Expected rows for years with positive avg_equity"
        assert rows["roe"].notna().all(), (
            "ROE must be finite for all years where avg_equity > 0"
        )

    def test_buyback_detected(self, aapl_summary):
        """Shares declined 30 %, so buyback_pct ≈ 0.30 (score should → 100)."""
        bp = float(aapl_summary.get("buyback_pct", float("nan")))
        assert math.isfinite(bp), "buyback_pct must be finite"
        assert 0.25 <= bp <= 0.35, (
            f"Expected buyback_pct ≈ 0.30, got {bp:.4f}"
        )


# ===========================================================================
# Test class 2 — KO strong profile
# ===========================================================================


class TestKoProfile:
    """KO_MOCK has the metrics expected of a high-quality Buffett stock."""

    def test_no_exception_raised(self, ko_data):
        inc, bal, cf, mkt = ko_data
        _compute_all_from_data("KO_MOCK", inc, bal, cf, mkt, RFR)

    def test_avg_roe_above_floor(self, ko_summary):
        """avg_roe must exceed the 15 % hard-filter floor."""
        avg_roe = float(ko_summary["avg_roe"])
        assert avg_roe >= 0.15, f"avg_roe={avg_roe:.4f} below 15 % floor"

    def test_gross_margin_above_60_pct(self, ko_summary):
        """Fixture sets 62 % gross margin; avg_gross_margin should be ≥ 0.60."""
        gm = float(ko_summary["avg_gross_margin"])
        assert gm >= 0.60, f"avg_gross_margin={gm:.4f} below 60 %"

    def test_eps_cagr_positive(self, ko_summary):
        """EPS grows monotonically from 1.60 to 4.50 → CAGR must be positive."""
        cagr = float(ko_summary["eps_cagr"])
        assert math.isfinite(cagr) and cagr > 0, (
            f"eps_cagr should be positive, got {cagr}"
        )

    def test_avg_de_below_fail_threshold(self, ko_summary):
        """D/E ≈ 0.42 must be below the fail threshold of 0.80."""
        de = float(ko_summary["avg_de_10yr"])
        assert de < 0.80, f"avg_de_10yr={de:.4f} is above the fail threshold"

    def test_interest_coverage_excellent(self, ko_summary):
        """Interest is 4 % of EBIT → avg_interest_pct must be < excellent=0.10."""
        ic = float(ko_summary["avg_interest_pct_10yr"])
        assert ic < 0.10, (
            f"avg_interest_pct_10yr={ic:.4f} should be < 0.10 (excellent)"
        )


# ===========================================================================
# Test class 3 — WEAK profile (deliberately poor metrics)
# ===========================================================================


class TestWeakProfile:
    """WEAK_MOCK exhibits the metrics that should score near-zero."""

    def test_no_exception_raised(self, weak_data):
        inc, bal, cf, mkt = weak_data
        _compute_all_from_data("WEAK_MOCK", inc, bal, cf, mkt, RFR)

    def test_eps_cagr_negative(self, weak_summary):
        """EPS declines from 1.00 to 0.64 → CAGR must be negative."""
        cagr = float(weak_summary["eps_cagr"])
        assert math.isfinite(cagr) and cagr < 0, (
            f"eps_cagr should be negative, got {cagr}"
        )

    def test_dilution_detected(self, weak_summary):
        """Shares grow from 400 M to 760 M → buyback_pct must be negative."""
        bp = float(weak_summary["buyback_pct"])
        assert bp < 0, f"Expected negative buyback_pct (dilution), got {bp:.4f}"

    def test_high_debt_to_equity(self, weak_summary):
        """D/E = 1.2 throughout → avg_de_10yr must be above the fail threshold (0.80)."""
        de = float(weak_summary["avg_de_10yr"])
        assert de > 0.80, f"avg_de_10yr={de:.4f} should exceed fail_de=0.80"

    def test_high_interest_burden(self, weak_summary):
        """Interest = 40 % of EBIT → avg_interest_pct must exceed fail=0.30."""
        ic = float(weak_summary["avg_interest_pct_10yr"])
        assert ic > 0.30, (
            f"avg_interest_pct_10yr={ic:.4f} should exceed fail=0.30"
        )


# ===========================================================================
# Test class 4 — F14 bear < base < bull present-value ordering
# ===========================================================================


class TestF14PvOrdering:
    """For stocks with positive EPS the three present values must be ordered."""

    def _pv(self, summary: dict, scenario: str) -> float:
        key = f"f14_{scenario}_present_value"
        val = summary.get(key)
        if val is None:
            pytest.fail(f"Key {key!r} not found in summary")
        return float(val)

    def test_aapl_bear_lt_base(self, aapl_summary):
        bear = self._pv(aapl_summary, "bear")
        base = self._pv(aapl_summary, "base")
        assert math.isfinite(bear) and math.isfinite(base), (
            f"bear={bear}, base={base}"
        )
        assert bear < base, (
            f"Expected bear_pv ({bear:.2f}) < base_pv ({base:.2f})"
        )

    def test_aapl_base_lt_bull(self, aapl_summary):
        base = self._pv(aapl_summary, "base")
        bull = self._pv(aapl_summary, "bull")
        assert math.isfinite(base) and math.isfinite(bull), (
            f"base={base}, bull={bull}"
        )
        assert base < bull, (
            f"Expected base_pv ({base:.2f}) < bull_pv ({bull:.2f})"
        )

    def test_ko_bear_lt_base(self, ko_summary):
        bear = self._pv(ko_summary, "bear")
        base = self._pv(ko_summary, "base")
        assert math.isfinite(bear) and math.isfinite(base)
        assert bear < base, (
            f"Expected bear_pv ({bear:.2f}) < base_pv ({base:.2f})"
        )

    def test_ko_base_lt_bull(self, ko_summary):
        base = self._pv(ko_summary, "base")
        bull = self._pv(ko_summary, "bull")
        assert math.isfinite(base) and math.isfinite(bull)
        assert base < bull, (
            f"Expected base_pv ({base:.2f}) < bull_pv ({bull:.2f})"
        )

    def test_weak_bear_lt_bull(self, weak_summary):
        """Negative EPS CAGR floors growth rates (bear=0, bull=3%) — bull PV > bear PV."""
        bear = self._pv(weak_summary, "bear")
        bull = self._pv(weak_summary, "bull")
        assert math.isfinite(bear) and math.isfinite(bull), (
            f"bear={bear}, bull={bull}"
        )
        assert bear < bull, (
            f"Expected bear_pv ({bear:.2f}) < bull_pv ({bull:.2f})"
        )


# ===========================================================================
# Test class 5 — Composite score ranking
# ===========================================================================


class TestCompositeScoreRanking:
    """Verify composite-score ordering and validity across all three stocks."""

    def _score(self, df: pd.DataFrame, ticker: str) -> float:
        row = df.loc[df["ticker"] == ticker, "composite_score"]
        assert not row.empty, f"Ticker {ticker!r} missing from composite_df"
        return float(row.iloc[0])

    def test_composite_df_has_three_rows(self, composite_df):
        assert len(composite_df) == 3, (
            f"Expected 3 rows, got {len(composite_df)}"
        )

    def test_all_three_tickers_present(self, composite_df):
        tickers = set(composite_df["ticker"])
        assert "AAPL_MOCK" in tickers
        assert "KO_MOCK" in tickers
        assert "WEAK_MOCK" in tickers

    def test_ko_outscores_weak(self, composite_df):
        """KO_MOCK composite score must strictly exceed WEAK_MOCK."""
        ko_score   = self._score(composite_df, "KO_MOCK")
        weak_score = self._score(composite_df, "WEAK_MOCK")
        assert ko_score > weak_score, (
            f"Expected KO_MOCK ({ko_score:.1f}) > WEAK_MOCK ({weak_score:.1f})"
        )

    def test_composite_scores_no_nan(self, composite_df):
        assert composite_df["composite_score"].notna().all(), (
            "composite_score column must contain no NaN values"
        )

    def test_all_criterion_scores_in_range(self, composite_df):
        """Every score_* column must be in [0, 100] for all rows."""
        score_cols = [c for c in composite_df.columns if c.startswith("score_")]
        assert score_cols, "No score_* columns found in composite_df"
        for col in score_cols:
            lo = float(composite_df[col].min())
            hi = float(composite_df[col].max())
            assert lo >= 0.0 and hi <= 100.0, (
                f"Column {col!r} out of [0, 100]: min={lo:.2f}, max={hi:.2f}"
            )

    def test_composite_scores_in_range(self, composite_df):
        lo = float(composite_df["composite_score"].min())
        hi = float(composite_df["composite_score"].max())
        assert 0.0 <= lo and hi <= 100.0, (
            f"composite_score out of [0, 100]: min={lo:.1f}, max={hi:.1f}"
        )

    def test_sorted_descending(self, composite_df):
        """compute_all_composite_scores returns rows sorted descending."""
        scores = composite_df["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True), (
            f"composite_df not sorted descending: {scores}"
        )

    def test_weak_composite_score_low(self, composite_df):
        """WEAK_MOCK has many zero-scoring criteria; composite should be below 30."""
        weak_score = self._score(composite_df, "WEAK_MOCK")
        assert weak_score < 30.0, (
            f"WEAK_MOCK composite score={weak_score:.1f} expected < 30"
        )


# ===========================================================================
# Test class 6 — Annual DataFrame structure
# ===========================================================================


class TestAnnualDfStructure:
    """Annual per-year DataFrame returned by _compute_all_from_data."""

    EXPECTED_COLS = {
        "ticker", "fiscal_year", "owner_earnings",
        "roe", "gross_margin", "sga_ratio", "net_margin",
        "de_ratio", "interest_pct_of_ebit",
    }

    def test_aapl_annual_has_expected_columns(self, aapl_annual):
        for col in self.EXPECTED_COLS:
            assert col in aapl_annual.columns, (
                f"Column {col!r} missing from AAPL_MOCK annual DataFrame"
            )

    def test_aapl_annual_has_ten_rows(self, aapl_annual):
        assert len(aapl_annual) == N, (
            f"Expected {N} rows, got {len(aapl_annual)}"
        )

    def test_aapl_annual_sorted_by_fiscal_year(self, aapl_annual):
        years = aapl_annual["fiscal_year"].tolist()
        assert years == sorted(years), "Annual DataFrame must be sorted by fiscal_year"

    def test_summary_ticker_matches(self, aapl_summary):
        assert aapl_summary["ticker"] == "AAPL_MOCK"


# ===========================================================================
# Test class 7 — Owner Earnings computed for at least 8 years
# ===========================================================================


class TestOwnerEarnings:
    """Owner Earnings (F1) must be non-NaN for at least 8 of 10 years."""

    def test_aapl_owner_earnings_at_least_8_years(self, aapl_annual):
        valid = aapl_annual["owner_earnings"].notna().sum()
        assert valid >= 8, (
            f"AAPL_MOCK: expected ≥8 non-NaN owner_earnings years, got {valid}"
        )

    def test_ko_owner_earnings_at_least_8_years(self, ko_annual):
        valid = ko_annual["owner_earnings"].notna().sum()
        assert valid >= 8, (
            f"KO_MOCK: expected ≥8 non-NaN owner_earnings years, got {valid}"
        )

    def test_weak_owner_earnings_at_least_8_years(self, weak_annual):
        valid = weak_annual["owner_earnings"].notna().sum()
        assert valid >= 8, (
            f"WEAK_MOCK: expected ≥8 non-NaN owner_earnings years, got {valid}"
        )

    def test_aapl_owner_earnings_all_positive(self, aapl_annual):
        """AAPL_MOCK has NI ~36k–44k, DA=2000, maint_capex≤DA → OE positive."""
        valid_oe = aapl_annual["owner_earnings"].dropna()
        assert (valid_oe > 0).all(), (
            "AAPL_MOCK: all non-NaN owner_earnings should be positive"
        )

    def test_ko_owner_earnings_all_positive(self, ko_annual):
        """KO_MOCK has consistent NI growth and low capex → OE positive."""
        valid_oe = ko_annual["owner_earnings"].dropna()
        assert (valid_oe > 0).all(), (
            "KO_MOCK: all non-NaN owner_earnings should be positive"
        )


# ===========================================================================
# Test class 8 — Margin of Safety is computed
# ===========================================================================


class TestMarginOfSafety:
    """Margin of Safety (F15) must be present in summary for each stock."""

    def test_aapl_margin_of_safety_computed(self, aapl_summary):
        """AAPL_MOCK has valid price and weighted_iv → margin_of_safety finite."""
        mos = aapl_summary.get("margin_of_safety")
        assert mos is not None, "margin_of_safety key missing from AAPL_MOCK summary"
        assert math.isfinite(float(mos)), (
            f"AAPL_MOCK margin_of_safety should be finite, got {mos}"
        )

    def test_ko_margin_of_safety_computed(self, ko_summary):
        mos = ko_summary.get("margin_of_safety")
        assert mos is not None, "margin_of_safety key missing from KO_MOCK summary"
        assert math.isfinite(float(mos)), (
            f"KO_MOCK margin_of_safety should be finite, got {mos}"
        )

    def test_weak_margin_of_safety_computed(self, weak_summary):
        mos = weak_summary.get("margin_of_safety")
        assert mos is not None, "margin_of_safety key missing from WEAK_MOCK summary"
        assert math.isfinite(float(mos)), (
            f"WEAK_MOCK margin_of_safety should be finite, got {mos}"
        )

    def test_aapl_weighted_iv_is_finite(self, aapl_summary):
        """weighted_iv (F14) must be finite — it feeds margin_of_safety."""
        wiv = aapl_summary.get("weighted_iv")
        assert wiv is not None and math.isfinite(float(wiv)), (
            f"AAPL_MOCK weighted_iv should be finite, got {wiv}"
        )

    def test_ko_weighted_iv_is_finite(self, ko_summary):
        wiv = ko_summary.get("weighted_iv")
        assert wiv is not None and math.isfinite(float(wiv)), (
            f"KO_MOCK weighted_iv should be finite, got {wiv}"
        )


# ===========================================================================
# Test class 9 — Hardcoded-value audit (documentary)
# ===========================================================================


class TestHardcodedValueAudit:
    """Verify that no bare threshold literals appear in metrics_engine/ source.

    The eight matches found by the canonical grep command are all of the form
    ``cfg.get("key", fallback_default)`` — they are safe fallback defaults,
    not hardcoded comparisons.  This test documents those findings and fails
    if any *new* bare literal is introduced outside a ``cfg.get`` call.
    """

    # The eight expected matches from:
    #   grep -rn "0\\.15\\|0\\.40\\|0\\.25\\|0\\.30" metrics_engine/ --include="*.py"
    #     | grep -v "test_\\|__pycache__"
    # All are ``cfg.get(key, fallback)`` patterns.
    KNOWN_ACCEPTABLE_PATTERNS = [
        "profitability.py",      # cfg.get("hard_filters", ...).get("min_avg_roe", 0.15)
        "valuation.py",          # cfg.get("hurdle_rate", 0.15)
        "valuation.py",          # cfg.get("buy_min_mos", 0.25)
        "composite_score.py",    # cfg.get("excellent_threshold", 0.30)
        "composite_score.py",    # cfg.get("excellent_capex_ratio", 0.25)
        "composite_score.py",    # cfg.get("excellent", 0.15)
        "composite_score.py",    # cfg.get("good", 0.15)
        "composite_score.py",    # cfg.get("fail", 0.30)
    ]

    def test_no_new_bare_literals(self):
        """No hardcoded comparisons may appear outside cfg.get() calls.

        Scans for the four sentinel values (0.15, 0.25, 0.30, 0.40) that are
        threshold candidates in the scoring system.  Every match must contain
        'cfg.get' on the same line, confirming it is a fallback default.
        """
        proj = Path(__file__).parent.parent / "metrics_engine"
        pattern_values = ["0.15", "0.40", "0.25", "0.30"]
        violations: list[str] = []
        for py_file in sorted(proj.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
                for val in pattern_values:
                    if val in line:
                        # Acceptable if the value appears in a cfg.get() call or comment/docstring
                        stripped = line.strip()
                        is_comment = stripped.startswith("#")
                        is_docstring_fragment = stripped.startswith(('"""', "'''", "*"))
                        is_config_read = "cfg.get" in line or "get_config" in line
                        if not (is_comment or is_docstring_fragment or is_config_read):
                            violations.append(f"{py_file.name}:{lineno}: {stripped!r}")
        assert not violations, (
            "Bare threshold literals found outside cfg.get() calls "
            f"in metrics_engine/:\n" + "\n".join(violations)
        )


# ===========================================================================
# Test class 10 — mypy type checking
# ===========================================================================


class TestMypy:
    """mypy must report zero errors across the ten metrics_engine source files."""

    def test_mypy_passes(self):
        proj = Path(__file__).parent.parent
        result = subprocess.run(
            [
                sys.executable, "-m", "mypy",
                "metrics_engine/",
                "--ignore-missing-imports",
            ],
            cwd=str(proj),
            capture_output=True,
            text=True,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"mypy reported errors in metrics_engine/:\n{combined}"
        )
        assert "Success" in combined, (
            f"mypy did not report 'Success':\n{combined}"
        )

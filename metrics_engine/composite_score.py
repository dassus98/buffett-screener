"""Weighted composite score (0-100) across the 10 Tier-2 soft criteria.

Computes an absolute-scale score for each soft criterion using breakpoints
defined in ``config/filter_config.yaml``, then produces a weighted sum
(the composite score) used to rank all tickers that passed the Tier 1 hard
filters.  Scores are NOT percentile-normalised; they are comparable across
different run dates and universe sizes.

Data Lineage Contract
---------------------
Upstream producers:
    - ``metrics_engine.__init__._merge_all_summaries``
      -> provides a flat ``metrics_summary`` dict for each ticker, containing
         all keys consumed by the 10 per-criterion scorers:
           F3  (profitability.compute_roe)                    -> avg_roe, roe_stdev
           F7  (profitability.compute_gross_margin)           -> avg_gross_margin
           F8  (profitability.compute_sga_ratio)              -> avg_sga_ratio
           F11 (growth.compute_eps_cagr)                      -> eps_cagr, decline_years
           F6  (leverage.compute_debt_to_equity)              -> avg_de_10yr
           F1  (owner_earnings.compute_owner_earnings)        -> owner_earnings_cagr
           F12 (capex.compute_capex_to_earnings)              -> avg_capex_to_ni
           F13 (growth.compute_buyback_indicator)             -> buyback_pct
           F4  (returns.compute_return_on_retained_earnings)  -> return_on_retained
           F9  (leverage.compute_interest_coverage)           -> avg_interest_pct_10yr

Downstream consumers:
    - ``metrics_engine.__init__.run_metrics_engine``
      -> calls ``compute_all_composite_scores`` to produce the ranked DataFrame,
         writes it to the ``composite_scores`` DuckDB table.
    - ``metrics_engine.__init__._write_summary_with_scores``
      -> reads ``composite_score`` from the returned DataFrame to augment the
         per-ticker summary table in DuckDB.
    - ``screener`` (recommendation engine)
      -> reads ``composite_score`` for buy/hold/sell ranking thresholds
         (see ``config/filter_config.yaml -> recommendations``).

Config dependencies (all loaded via ``get_threshold()``):
    - ``soft_scores.{criterion}.weight`` (10 criteria; must sum to 1.0)
    - ``soft_scores.roe.score_floor_value``, ``score_ceiling_value``,
      ``variance_penalty_threshold``, ``variance_penalty_points``
    - ``soft_scores.gross_margin.breakpoints``
    - ``soft_scores.sga_ratio.excellent_threshold``, ``fail_threshold``
    - ``soft_scores.eps_growth.cagr_floor``, ``cagr_ceiling``,
      ``consistency_multipliers``
    - ``soft_scores.debt_conservatism.excellent_de``, ``fail_de``
    - ``soft_scores.owner_earnings_growth`` (uses ``eps_growth`` breakpoints
      at runtime per ``scoring: same_as_eps_growth`` sentinel)
    - ``soft_scores.capital_efficiency.excellent_capex_ratio``,
      ``fail_capex_ratio``
    - ``soft_scores.buyback.breakpoints``
    - ``soft_scores.retained_earnings_return.excellent``, ``good``,
      ``good_score``
    - ``soft_scores.interest_coverage.excellent``, ``good``, ``fail``

Authoritative spec: docs/SCORING.md.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level interpolation helper
# ---------------------------------------------------------------------------

def _linear_interp(value: float, breakpoints: dict[float, float]) -> float:
    """Linearly interpolate a score from sorted (metric_value -> score) breakpoints.

    Parameters
    ----------
    value:
        Pre-validated, non-NaN metric value.
    breakpoints:
        Mapping from metric value to score.  Values outside the range are
        clamped to the boundary score.

    Returns
    -------
    float
        Interpolated score (not yet clamped to [0, 100]).
    """
    # Sort breakpoints by metric value for ordered traversal.
    pts = sorted(breakpoints.items())

    # Clamp below: value at or below the lowest breakpoint returns its score.
    if value <= pts[0][0]:
        return float(pts[0][1])

    # Clamp above: value at or above the highest breakpoint returns its score.
    if value >= pts[-1][0]:
        return float(pts[-1][1])

    # Walk adjacent pairs to find the enclosing segment, then interpolate.
    for i in range(len(pts) - 1):
        lo_v, lo_s = pts[i]
        hi_v, hi_s = pts[i + 1]
        if lo_v <= value <= hi_v:
            span = hi_v - lo_v
            # Guard: zero-width segment maps to the high-end score.
            ratio = (value - lo_v) / span if span > 0 else 1.0
            return float(lo_s + ratio * (hi_s - lo_s))

    # Fallback (should not be reached with well-formed breakpoints).
    return float(pts[-1][1])


# ---------------------------------------------------------------------------
# Private per-criterion scorers
# ---------------------------------------------------------------------------

def _score_roe(summary: dict, cfg: dict) -> float:
    """ROE: floor/ceiling score minus optional variance penalty.

    Per SCORING.md criterion 1:
    - Base score from 10-year average ROE via floor/ceiling interpolation
      (0% -> 0, floor -> 50, ceiling -> 100).
    - Variance penalty: subtract ``variance_penalty_points`` if the standard
      deviation of annual ROE exceeds ``variance_penalty_threshold``.
    - Final score floored at 0.
    """
    # --- Step 1: Extract raw metric values from the summary dict ---
    value = summary.get("avg_roe", float("nan"))
    roe_stdev = summary.get("roe_stdev", float("nan"))

    # --- Step 2: Compute base score via floor/ceiling interpolation ---
    # score_criterion handles NaN -> 0 automatically.
    base = score_criterion(value, cfg)

    # --- Step 3: Apply variance penalty if ROE is too volatile ---
    # A high stdev signals cyclical earnings, not a durable moat.
    pen_thresh = float(cfg["variance_penalty_threshold"])
    pen_pts = float(cfg["variance_penalty_points"])
    if not pd.isna(roe_stdev) and float(roe_stdev) > pen_thresh:
        base = max(0.0, base - pen_pts)
    return base


def _score_sga(summary: dict, cfg: dict) -> float:
    """SGA ratio: lower-is-better linear from excellent_threshold->100 to fail_threshold->0.

    Per SCORING.md criterion 3:
    - SGA as % of gross profit. < excellent -> 100; linear decay to fail -> 0.
    """
    # --- Step 1: Extract raw metric value ---
    value = summary.get("avg_sga_ratio", float("nan"))
    if pd.isna(value):
        return 0.0

    # --- Step 2: Build inverted breakpoints (lower is better) and interpolate ---
    excellent = float(cfg["excellent_threshold"])
    fail = float(cfg["fail_threshold"])
    return float(max(0.0, min(100.0, _linear_interp(float(value), {excellent: 100.0, fail: 0.0}))))


def _score_eps_growth(summary: dict, cfg: dict) -> float:
    """EPS growth: CAGR score multiplied by consistency factor.

    Per SCORING.md criterion 4:
    - CAGR component: linear from 0% -> 0, cagr_floor -> 50, cagr_ceiling -> 100.
    - Consistency multiplier: penalises year-over-year EPS declines.
    - Final score = CAGR score x multiplier.
    """
    # --- Step 1: Extract raw metrics ---
    cagr = summary.get("eps_cagr", float("nan"))
    decline_raw = summary.get("decline_years", 0)
    # Guard: decline_years may be NaN or None when data is incomplete.
    decline = 0 if pd.isna(decline_raw) else int(decline_raw or 0)
    if pd.isna(cagr):
        return 0.0

    # --- Step 2: Compute CAGR component score via linear interpolation ---
    floor_v = float(cfg["cagr_floor"])
    ceil_v = float(cfg["cagr_ceiling"])
    cagr_score = max(0.0, min(100.0, _linear_interp(float(cagr), {0.0: 0.0, floor_v: 50.0, ceil_v: 100.0})))

    # --- Step 3: Apply consistency multiplier ---
    # Per config: {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.6}.
    # 3+ decline years all use the key=3 entry (capped via min()).
    multipliers: dict = cfg["consistency_multipliers"]
    mult = float(multipliers[min(decline, 3)])
    return float(cagr_score * mult)


def _score_debt(summary: dict, cfg: dict) -> float:
    """Debt conservatism: lower-is-better linear from excellent_de->100 to fail_de->0.

    Per SCORING.md criterion 5:
    - D/E < excellent -> 100; D/E >= fail -> 0; linear in between.
    - Negative equity -> avg_de_10yr is NaN upstream -> score 0.
    """
    # --- Step 1: Extract raw metric value ---
    value = summary.get("avg_de_10yr", float("nan"))
    if pd.isna(value):
        return 0.0

    # --- Step 2: Build inverted breakpoints (lower D/E is better) ---
    excellent = float(cfg["excellent_de"])
    fail = float(cfg["fail_de"])
    return float(max(0.0, min(100.0, _linear_interp(float(value), {excellent: 100.0, fail: 0.0}))))


def _score_oe_growth(summary: dict, eps_cfg: dict) -> float:
    """Owner earnings CAGR: same breakpoints as EPS growth CAGR.

    Per config ``owner_earnings_growth.scoring: same_as_eps_growth``, this
    criterion reuses the ``eps_growth`` section's ``cagr_floor`` and
    ``cagr_ceiling`` breakpoints at runtime.

    Note: ``owner_earnings_growth.min_computable_years: 5`` is enforced
    upstream — when OE is computable for < 5 years, the CAGR is NaN and
    this scorer returns 0.
    """
    # --- Step 1: Extract raw metric value ---
    value = summary.get("owner_earnings_cagr", float("nan"))
    if pd.isna(value):
        return 0.0

    # --- Step 2: Interpolate using shared EPS growth breakpoints ---
    floor_v = float(eps_cfg["cagr_floor"])
    ceil_v = float(eps_cfg["cagr_ceiling"])
    return float(max(0.0, min(100.0, _linear_interp(float(value), {0.0: 0.0, floor_v: 50.0, ceil_v: 100.0}))))


def _score_capex(summary: dict, cfg: dict) -> float:
    """Capital efficiency: lower-is-better linear from excellent->100 to fail->0.

    Per SCORING.md criterion 7:
    - CapEx/NI < excellent -> 100; >= fail -> 0; linear in between.
    """
    # --- Step 1: Extract raw metric value ---
    value = summary.get("avg_capex_to_ni", float("nan"))
    if pd.isna(value):
        return 0.0

    # --- Step 2: Build inverted breakpoints (lower CapEx ratio is better) ---
    excellent = float(cfg["excellent_capex_ratio"])
    fail = float(cfg["fail_capex_ratio"])
    return float(max(0.0, min(100.0, _linear_interp(float(value), {excellent: 100.0, fail: 0.0}))))


def _score_retained(summary: dict, cfg: dict) -> float:
    """Return on retained earnings: multi-breakpoint linear (F4 Dollar Test).

    Per SCORING.md criterion 9:
    - 0% -> 0, good% -> good_score, excellent% -> 100.
    - Negative or zero return -> score 0 (company destroyed value).
    """
    # --- Step 1: Extract raw metric value ---
    value = summary.get("return_on_retained", float("nan"))
    # Non-positive return on retained earnings is economically meaningless —
    # the company failed to create value from retained capital.
    if pd.isna(value) or float(value) <= 0:
        return 0.0

    # --- Step 2: Build multi-breakpoint curve from config ---
    good = float(cfg["good"])
    good_score = float(cfg["good_score"])
    excellent = float(cfg["excellent"])
    return float(max(0.0, min(100.0, _linear_interp(float(value), {0.0: 0.0, good: good_score, excellent: 100.0}))))


def _score_interest(summary: dict, cfg: dict) -> float:
    """Interest coverage: stepped zones with linear decay in the middle range.

    Per SCORING.md criterion 10:
    - Interest < excellent% of EBIT -> 100 (negligible burden).
    - excellent <= interest < good -> 70 (flat zone — acceptable).
    - good <= interest < fail -> linear decay from 70 to 0.
    - interest >= fail -> 0 (excessive burden).
    """
    # --- Step 1: Extract raw metric value ---
    value = summary.get("avg_interest_pct_10yr", float("nan"))
    if pd.isna(value):
        return 0.0
    v = float(value)

    # --- Step 2: Load zone boundaries from config ---
    excellent = float(cfg["excellent"])
    good = float(cfg["good"])
    fail = float(cfg["fail"])

    # --- Step 3: Determine score by zone ---
    if v < excellent:
        # Best zone: interest expense is negligible relative to EBIT.
        return 100.0
    if v < good:
        # Acceptable zone: flat score of 70 (per SCORING.md).
        return 70.0
    if v < fail:
        # Decay zone: linear from 70 (at good threshold) to 0 (at fail).
        return float(70.0 * (1.0 - (v - good) / (fail - good)))
    # Worst zone: interest consumes too much of EBIT.
    return 0.0


# ---------------------------------------------------------------------------
# Public formula functions
# ---------------------------------------------------------------------------

def score_criterion(value: float, criterion_config: dict) -> float:
    """Score a single metric 0-100 using the criterion's config dict.

    Supports two scoring patterns detected by config key presence:

    - **Breakpoints** (``"breakpoints"`` key): ``{metric_value: score}`` dict
      interpreted as a piecewise-linear function; values outside the range
      are clamped to the boundary score.
    - **Floor/ceiling** (``"score_floor_value"`` + ``"score_ceiling_value"``):
      constructs implicit breakpoints ``{0: 0, floor: 50, ceiling: 100}``.

    Parameters
    ----------
    value:
        Raw metric value.  ``NaN`` or ``None`` -> returns **0** immediately
        (per SCORING.md: missing data is penalised, not excluded).
    criterion_config:
        Config sub-dict for this criterion (e.g. the ``gross_margin`` section
        from ``get_threshold("soft_scores")``).

    Returns
    -------
    float
        Score in ``[0, 100]``.
    """
    # --- Step 1: Handle missing data ---
    # Per SCORING.md: "If the underlying metric is NaN, the criterion
    # score = 0 (not NaN).  The criterion's weight still applies."
    if pd.isna(value):
        return 0.0
    v = float(value)

    # --- Step 2: Detect config pattern and interpolate ---
    if "breakpoints" in criterion_config:
        # Explicit breakpoint dict: convert string keys to float (YAML may
        # produce string keys for negative numbers like "-0.05").
        bpts = {float(k): float(sv) for k, sv in criterion_config["breakpoints"].items()}
        raw = _linear_interp(v, bpts)
    elif "score_floor_value" in criterion_config and "score_ceiling_value" in criterion_config:
        # Implicit three-point curve: 0->0, floor->50, ceiling->100.
        floor_v = float(criterion_config["score_floor_value"])
        ceil_v = float(criterion_config["score_ceiling_value"])
        raw = _linear_interp(v, {0.0: 0.0, floor_v: 50.0, ceil_v: 100.0})
    else:
        # Config format not recognized — defensive return.
        logger.warning("score_criterion: unrecognized config format; returning 0.")
        return 0.0

    # --- Step 3: Clamp to [0, 100] ---
    return float(max(0.0, min(100.0, raw)))


def compute_composite_score(metrics_summary: dict) -> dict[str, Any]:
    """Compute the 10-criterion weighted composite score (0-100) for one ticker.

    Parameters
    ----------
    metrics_summary:
        Flat dict merging all per-formula summary outputs for one ticker.
        Missing keys are treated as NaN -> criterion score = 0.  Expected
        keys: ``avg_roe``, ``roe_stdev``, ``avg_gross_margin``, ``avg_sga_ratio``,
        ``eps_cagr``, ``decline_years``, ``avg_de_10yr``, ``owner_earnings_cagr``,
        ``avg_capex_to_ni``, ``buyback_pct``, ``return_on_retained``,
        ``avg_interest_pct_10yr``.

    Returns
    -------
    dict
        ``composite_score`` (float 0-100) and ``scores_detail`` mapping each
        criterion name to ``{"raw_value": ..., "score": float, "weight": float}``.
    """
    # --- Step 1: Load the full soft_scores config section ---
    # All 10 criterion sub-dicts and their weights live under this key.
    # Config validation (filter_config_loader) guarantees weights sum to 1.0.
    ss: dict[str, Any] = get_threshold("soft_scores")

    # eps_growth config is shared with owner_earnings_growth
    # (per config sentinel: ``scoring: same_as_eps_growth``).
    eps_cfg: dict[str, Any] = ss["eps_growth"]
    s = metrics_summary

    # --- Step 2: Helper to build per-criterion detail entry ---
    def _e(name: str, raw: Any, score: float) -> dict[str, Any]:
        """Bundle raw value, score, and weight into a detail dict."""
        return {
            "raw_value": raw,
            "score": float(score),
            "weight": float(ss[name]["weight"]),
        }

    # --- Step 3: Compute each criterion's score ---
    # Each private scorer handles NaN -> 0 internally.
    gm_val = s.get("avg_gross_margin", float("nan"))
    buy_val = s.get("buyback_pct", float("nan"))
    detail: dict[str, dict[str, Any]] = {
        "roe":                      _e("roe",                      s.get("avg_roe"),              _score_roe(s, ss["roe"])),
        "gross_margin":             _e("gross_margin",             gm_val,                         score_criterion(gm_val, ss["gross_margin"])),
        "sga_ratio":                _e("sga_ratio",                s.get("avg_sga_ratio"),         _score_sga(s, ss["sga_ratio"])),
        "eps_growth":               _e("eps_growth",               s.get("eps_cagr"),              _score_eps_growth(s, eps_cfg)),
        "debt_conservatism":        _e("debt_conservatism",        s.get("avg_de_10yr"),           _score_debt(s, ss["debt_conservatism"])),
        "owner_earnings_growth":    _e("owner_earnings_growth",    s.get("owner_earnings_cagr"),   _score_oe_growth(s, eps_cfg)),
        "capital_efficiency":       _e("capital_efficiency",       s.get("avg_capex_to_ni"),       _score_capex(s, ss["capital_efficiency"])),
        "buyback":                  _e("buyback",                  buy_val,                        score_criterion(buy_val, ss["buyback"])),
        "retained_earnings_return": _e("retained_earnings_return", s.get("return_on_retained"),    _score_retained(s, ss["retained_earnings_return"])),
        "interest_coverage":        _e("interest_coverage",        s.get("avg_interest_pct_10yr"), _score_interest(s, ss["interest_coverage"])),
    }

    # --- Step 4: Compute weighted sum ---
    # composite = sum(score_i * weight_i) for all 10 criteria.
    composite = sum(d["score"] * d["weight"] for d in detail.values())
    return {"composite_score": float(composite), "scores_detail": detail}


def compute_all_composite_scores(
    all_metrics: dict[str, dict],
) -> pd.DataFrame:
    """Compute composite scores for all tickers and return a ranked DataFrame.

    Parameters
    ----------
    all_metrics:
        ``{ticker: metrics_summary_dict}`` for every ticker that passed the
        Tier 1 hard filters.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``composite_score``, and ``score_{criterion}``
        for each of the 10 soft criteria.  Sorted descending by
        ``composite_score``.  Empty DataFrame when *all_metrics* is empty.
    """
    # --- Step 1: Score each ticker independently ---
    rows: list[dict[str, Any]] = []
    for ticker, metrics in all_metrics.items():
        result = compute_composite_score(metrics)
        row: dict[str, Any] = {"ticker": ticker, "composite_score": result["composite_score"]}
        # Flatten per-criterion scores into score_{name} columns.
        for criterion, d in result["scores_detail"].items():
            row[f"score_{criterion}"] = d["score"]
        rows.append(row)

    # --- Step 2: Handle empty input ---
    if not rows:
        return pd.DataFrame()

    # --- Step 3: Assemble DataFrame, sort descending, reset index ---
    return (
        pd.DataFrame(rows)
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )

"""Weighted composite score (0–100) across the 10 Tier-2 soft criteria."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level interpolation helper
# ---------------------------------------------------------------------------

def _linear_interp(value: float, breakpoints: dict[float, float]) -> float:
    """Linearly interpolate a score from sorted (metric_value → score) breakpoints.

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
    pts = sorted(breakpoints.items())
    if value <= pts[0][0]:
        return float(pts[0][1])
    if value >= pts[-1][0]:
        return float(pts[-1][1])
    for i in range(len(pts) - 1):
        lo_v, lo_s = pts[i]
        hi_v, hi_s = pts[i + 1]
        if lo_v <= value <= hi_v:
            span = hi_v - lo_v
            ratio = (value - lo_v) / span if span > 0 else 1.0
            return float(lo_s + ratio * (hi_s - lo_s))
    return float(pts[-1][1])


# ---------------------------------------------------------------------------
# Private per-criterion scorers
# ---------------------------------------------------------------------------

def _score_roe(summary: dict, cfg: dict) -> float:
    """ROE: floor/ceiling score minus optional variance penalty."""
    value = summary.get("avg_roe", float("nan"))
    roe_stdev = summary.get("roe_stdev", float("nan"))
    base = score_criterion(value, cfg)
    pen_thresh = float(cfg.get("variance_penalty_threshold", 0.05))
    pen_pts = float(cfg.get("variance_penalty_points", 15))
    if not pd.isna(roe_stdev) and float(roe_stdev) > pen_thresh:
        base = max(0.0, base - pen_pts)
    return base


def _score_sga(summary: dict, cfg: dict) -> float:
    """SGA ratio: lower-is-better linear from excellent_threshold→100 to fail_threshold→0."""
    value = summary.get("avg_sga_ratio", float("nan"))
    if pd.isna(value):
        return 0.0
    excellent = float(cfg.get("excellent_threshold", 0.30))
    fail = float(cfg.get("fail_threshold", 0.80))
    return float(max(0.0, min(100.0, _linear_interp(float(value), {excellent: 100.0, fail: 0.0}))))


def _score_eps_growth(summary: dict, cfg: dict) -> float:
    """EPS growth: CAGR score multiplied by consistency factor."""
    cagr = summary.get("eps_cagr", float("nan"))
    decline_raw = summary.get("decline_years", 0)
    decline = 0 if pd.isna(decline_raw) else int(decline_raw or 0)
    if pd.isna(cagr):
        return 0.0
    floor_v = float(cfg.get("cagr_floor", 0.10))
    ceil_v = float(cfg.get("cagr_ceiling", 0.20))
    cagr_score = max(0.0, min(100.0, _linear_interp(float(cagr), {0.0: 0.0, floor_v: 50.0, ceil_v: 100.0})))
    multipliers: dict = cfg.get("consistency_multipliers", {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.6})
    mult = float(multipliers.get(min(decline, 3), 0.6))
    return float(cagr_score * mult)


def _score_debt(summary: dict, cfg: dict) -> float:
    """Debt conservatism: lower-is-better linear from excellent_de→100 to fail_de→0."""
    value = summary.get("avg_de_10yr", float("nan"))
    if pd.isna(value):
        return 0.0
    excellent = float(cfg.get("excellent_de", 0.20))
    fail = float(cfg.get("fail_de", 0.80))
    return float(max(0.0, min(100.0, _linear_interp(float(value), {excellent: 100.0, fail: 0.0}))))


def _score_oe_growth(summary: dict, eps_cfg: dict) -> float:
    """Owner earnings CAGR: same breakpoints as EPS growth CAGR."""
    value = summary.get("owner_earnings_cagr", float("nan"))
    if pd.isna(value):
        return 0.0
    floor_v = float(eps_cfg.get("cagr_floor", 0.10))
    ceil_v = float(eps_cfg.get("cagr_ceiling", 0.20))
    return float(max(0.0, min(100.0, _linear_interp(float(value), {0.0: 0.0, floor_v: 50.0, ceil_v: 100.0}))))


def _score_capex(summary: dict, cfg: dict) -> float:
    """Capital efficiency: lower-is-better linear from excellent_capex_ratio→100 to fail→0."""
    value = summary.get("avg_capex_to_ni", float("nan"))
    if pd.isna(value):
        return 0.0
    excellent = float(cfg.get("excellent_capex_ratio", 0.25))
    fail = float(cfg.get("fail_capex_ratio", 0.75))
    return float(max(0.0, min(100.0, _linear_interp(float(value), {excellent: 100.0, fail: 0.0}))))


def _score_retained(summary: dict, cfg: dict) -> float:
    """Return on retained earnings: multi-breakpoint linear (F4 $1 test)."""
    value = summary.get("return_on_retained", float("nan"))
    if pd.isna(value) or float(value) <= 0:
        return 0.0
    good = float(cfg.get("good", 0.12))
    good_score = float(cfg.get("good_score", 70))
    excellent = float(cfg.get("excellent", 0.15))
    return float(max(0.0, min(100.0, _linear_interp(float(value), {0.0: 0.0, good: good_score, excellent: 100.0}))))


def _score_interest(summary: dict, cfg: dict) -> float:
    """Interest coverage: stepped zones (flat at 70 between excellent and good thresholds)."""
    value = summary.get("avg_interest_pct_10yr", float("nan"))
    if pd.isna(value):
        return 0.0
    v = float(value)
    excellent = float(cfg.get("excellent", 0.10))
    good = float(cfg.get("good", 0.15))
    fail = float(cfg.get("fail", 0.30))
    if v < excellent:
        return 100.0
    if v < good:
        return 70.0
    if v < fail:
        return float(70.0 * (1.0 - (v - good) / (fail - good)))
    return 0.0


# ---------------------------------------------------------------------------
# Public formula functions
# ---------------------------------------------------------------------------

def score_criterion(value: float, criterion_config: dict) -> float:
    """Score a single metric 0–100 using the criterion's config dict.

    Supports two scoring patterns detected by config key presence:

    - **Breakpoints** (``"breakpoints"`` key): ``{metric_value: score}`` dict
      interpreted as a piecewise-linear function; values outside the range
      are clamped to the boundary score.
    - **Floor/ceiling** (``"score_floor_value"`` + ``"score_ceiling_value"``):
      constructs implicit breakpoints ``{0: 0, floor: 50, ceiling: 100}``.

    Parameters
    ----------
    value:
        Raw metric value.  ``NaN`` or ``None`` → returns **0** immediately.
    criterion_config:
        Config sub-dict for this criterion (e.g. the ``gross_margin`` section
        from ``get_config()["soft_scores"]``).

    Returns
    -------
    float
        Score in ``[0, 100]``.
    """
    if pd.isna(value):
        return 0.0
    v = float(value)
    if "breakpoints" in criterion_config:
        bpts = {float(k): float(sv) for k, sv in criterion_config["breakpoints"].items()}
        raw = _linear_interp(v, bpts)
    elif "score_floor_value" in criterion_config and "score_ceiling_value" in criterion_config:
        floor_v = float(criterion_config["score_floor_value"])
        ceil_v = float(criterion_config["score_ceiling_value"])
        raw = _linear_interp(v, {0.0: 0.0, floor_v: 50.0, ceil_v: 100.0})
    else:
        logger.warning("score_criterion: unrecognized config format; returning 0.")
        return 0.0
    return float(max(0.0, min(100.0, raw)))


def compute_composite_score(metrics_summary: dict) -> dict[str, Any]:
    """Compute the 10-criterion weighted composite score (0–100) for one ticker.

    Parameters
    ----------
    metrics_summary:
        Flat dict merging all per-formula summary outputs for one ticker.
        Missing keys are treated as NaN → criterion score = 0.  Expected
        keys: ``avg_roe``, ``roe_stdev``, ``avg_gross_margin``, ``avg_sga_ratio``,
        ``eps_cagr``, ``decline_years``, ``avg_de_10yr``, ``owner_earnings_cagr``,
        ``avg_capex_to_ni``, ``buyback_pct``, ``return_on_retained``,
        ``avg_interest_pct_10yr``.

    Returns
    -------
    dict
        ``composite_score`` (float 0–100) and ``scores_detail`` mapping each
        criterion name to ``{"raw_value": …, "score": float, "weight": float}``.
    """
    cfg = get_config()
    ss = cfg.get("soft_scores", {})
    eps_cfg = ss.get("eps_growth", {})
    s = metrics_summary

    def _e(name: str, raw: Any, score: float) -> dict[str, Any]:
        return {"raw_value": raw, "score": float(score), "weight": float(ss.get(name, {}).get("weight", 0.0))}

    gm_val = s.get("avg_gross_margin", float("nan"))
    buy_val = s.get("buyback_pct", float("nan"))
    detail: dict[str, dict[str, Any]] = {
        "roe":                      _e("roe",                      s.get("avg_roe"),              _score_roe(s, ss.get("roe", {}))),
        "gross_margin":             _e("gross_margin",             gm_val,                         score_criterion(gm_val, ss.get("gross_margin", {}))),
        "sga_ratio":                _e("sga_ratio",                s.get("avg_sga_ratio"),         _score_sga(s, ss.get("sga_ratio", {}))),
        "eps_growth":               _e("eps_growth",               s.get("eps_cagr"),              _score_eps_growth(s, eps_cfg)),
        "debt_conservatism":        _e("debt_conservatism",        s.get("avg_de_10yr"),           _score_debt(s, ss.get("debt_conservatism", {}))),
        "owner_earnings_growth":    _e("owner_earnings_growth",    s.get("owner_earnings_cagr"),   _score_oe_growth(s, eps_cfg)),
        "capital_efficiency":       _e("capital_efficiency",       s.get("avg_capex_to_ni"),       _score_capex(s, ss.get("capital_efficiency", {}))),
        "buyback":                  _e("buyback",                  buy_val,                        score_criterion(buy_val, ss.get("buyback", {}))),
        "retained_earnings_return": _e("retained_earnings_return", s.get("return_on_retained"),    _score_retained(s, ss.get("retained_earnings_return", {}))),
        "interest_coverage":        _e("interest_coverage",        s.get("avg_interest_pct_10yr"), _score_interest(s, ss.get("interest_coverage", {}))),
    }
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
    rows: list[dict[str, Any]] = []
    for ticker, metrics in all_metrics.items():
        result = compute_composite_score(metrics)
        row: dict[str, Any] = {"ticker": ticker, "composite_score": result["composite_score"]}
        for criterion, d in result["scores_detail"].items():
            row[f"score_{criterion}"] = d["score"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )

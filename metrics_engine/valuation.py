"""Three-scenario intrinsic value (F14), margin of safety (F15), and earnings yield (F16)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_NAN_SCENARIO: dict[str, Any] = {
    "growth": float("nan"),
    "pe": float("nan"),
    "projected_price": float("nan"),
    "present_value": float("nan"),
    "annual_return": float("nan"),
    "probability": float("nan"),
}


def _nan_iv_result() -> dict[str, Any]:
    """Return an all-NaN result dict for invalid-input early exits (F14)."""
    return {
        "bear": dict(_NAN_SCENARIO),
        "base": dict(_NAN_SCENARIO),
        "bull": dict(_NAN_SCENARIO),
        "weighted_iv": float("nan"),
        "meets_hurdle": False,
    }


def _resolve_pe_estimates(
    pe_series: pd.Series,
    scen_cfg: dict[str, Any],
    fallback_pe: float,
) -> dict[str, float]:
    """Derive bear / base / bull terminal P/E from the historical series.

    Parameters
    ----------
    pe_series:
        10-year historical P/E ratios (NaN values are dropped).
    scen_cfg:
        ``valuation.scenarios`` sub-dict from config.
    fallback_pe:
        P/E to use when *pe_series* is empty or all-NaN.

    Returns
    -------
    dict
        Keys ``"bear"``, ``"base"``, ``"bull"`` each mapping to a float P/E.
    """
    valid = pe_series.dropna() if len(pe_series) > 0 else pd.Series(dtype=float)
    if valid.empty:
        logger.warning(
            "Historical P/E series is empty or all-NaN; using fallback P/E=%.1f.", fallback_pe
        )
        mean_pe = median_pe = fallback_pe
    else:
        mean_pe = float(valid.mean())
        median_pe = float(valid.median())

    bear_cap = float(scen_cfg.get("bear", {}).get("pe_cap", 12))
    bull_floor = float(scen_cfg.get("bull", {}).get("pe_floor", 20))
    return {
        "bear": min(mean_pe, bear_cap),
        "base": median_pe,
        "bull": max(mean_pe, bull_floor),
    }


def _resolve_growth_rates(
    eps_cagr: float,
    scen_cfg: dict[str, Any],
) -> dict[str, float]:
    """Map *eps_cagr* to bear / base / bull growth rates, flooring negatives.

    Edge case — NaN or negative *eps_cagr*:
      bear = 0, base = max(0, eps_cagr), bull = max(eps_cagr × 1.3, 0.03)

    Parameters
    ----------
    eps_cagr:
        Historical 10-year EPS CAGR (may be NaN or negative).
    scen_cfg:
        ``valuation.scenarios`` sub-dict from config.

    Returns
    -------
    dict
        Keys ``"bear"``, ``"base"``, ``"bull"`` each mapping to a float growth rate.
    """
    if pd.isna(eps_cagr) or float(eps_cagr) < 0:
        if not pd.isna(eps_cagr):
            logger.warning("EPS CAGR=%.4f < 0; flooring growth rates.", float(eps_cagr))
        cagr = 0.0 if pd.isna(eps_cagr) else float(eps_cagr)
        return {
            "bear": 0.0,
            "base": max(0.0, cagr),
            "bull": max(cagr * float(scen_cfg.get("bull", {}).get("growth_multiplier", 1.3)), 0.03),
        }
    cagr = float(eps_cagr)
    return {
        "bear": cagr * float(scen_cfg.get("bear", {}).get("growth_multiplier", 0.5)),
        "base": cagr * float(scen_cfg.get("base", {}).get("growth_multiplier", 1.0)),
        "bull": cagr * float(scen_cfg.get("bull", {}).get("growth_multiplier", 1.3)),
    }


def _compute_one_scenario(
    growth_rate: float,
    terminal_pe: float,
    discount_rate: float,
    current_eps: float,
    current_price: float,
    n_years: int,
    probability: float,
) -> dict[str, Any]:
    """Compute a single valuation scenario for F14.

    Parameters
    ----------
    growth_rate:
        Annual EPS growth rate for this scenario.
    terminal_pe:
        Terminal (year-10) P/E multiple.
    discount_rate:
        Annual discount rate (risk-free rate + risk premium).
    current_eps, current_price, n_years, probability:
        Forward projection inputs and scenario weighting.

    Returns
    -------
    dict
        Keys: ``growth``, ``pe``, ``projected_price``, ``present_value``,
        ``annual_return``, ``probability``.
    """
    projected_eps = current_eps * (1 + growth_rate) ** n_years
    projected_price = projected_eps * terminal_pe
    present_value = projected_price / (1 + discount_rate) ** n_years
    if current_price > 0:
        annual_return = (projected_price / current_price) ** (1 / n_years) - 1
    else:
        annual_return = float("nan")
    return {
        "growth": float(growth_rate),
        "pe": float(terminal_pe),
        "projected_price": float(projected_price),
        "present_value": float(present_value),
        "annual_return": float(annual_return),
        "probability": float(probability),
    }


# ---------------------------------------------------------------------------
# Public formula functions
# ---------------------------------------------------------------------------

def compute_intrinsic_value(
    current_eps: float,
    eps_cagr: float,
    historical_pe_series: pd.Series,
    current_price: float,
    risk_free_rate: float,
) -> dict[str, Any]:
    """Compute Three-Scenario Intrinsic Value (F14).

    Bear / base / bull scenarios weighted 25 / 50 / 25 per config.

    Parameters
    ----------
    current_eps:
        Most-recent diluted EPS.  Must be > 0 to project forward.
    eps_cagr:
        10-year historical EPS CAGR.  NaN or negative triggers floor logic.
    historical_pe_series:
        10-year historical trailing P/E series.  Empty → fallback P/E used.
    current_price:
        Current market price per share.
    risk_free_rate:
        10-year government bond yield (decimal, e.g. 0.04 for 4%).

    Returns
    -------
    dict
        Keys: ``"bear"``, ``"base"``, ``"bull"`` scenario dicts each
        containing ``growth``, ``pe``, ``projected_price``, ``present_value``,
        ``annual_return``, ``probability``; plus ``weighted_iv`` (float) and
        ``meets_hurdle`` (bool).
    """
    cfg = get_config()
    val_cfg = cfg.get("valuation", {})
    if pd.isna(current_eps) or float(current_eps) <= 0:
        logger.warning(
            "Intrinsic value: current_eps=%.4f ≤ 0. Cannot project forward.", current_eps
        )
        return _nan_iv_result()
    n = int(val_cfg.get("projection_years", 10))
    hurdle = float(val_cfg.get("hurdle_rate", 0.15))
    scen_cfg = val_cfg.get("scenarios", {})
    fallback_pe = float(val_cfg.get("fallback_historical_pe", 15.0))
    pe = _resolve_pe_estimates(historical_pe_series, scen_cfg, fallback_pe)
    gr = _resolve_growth_rates(eps_cagr, scen_cfg)
    result: dict[str, Any] = {}
    for label in ("bear", "base", "bull"):
        sc = scen_cfg.get(label, {})
        dr = float(risk_free_rate) + float(sc.get("risk_premium", 0.03))
        prob = float(sc.get("probability", 1 / 3))
        result[label] = _compute_one_scenario(
            gr[label], pe[label], dr, float(current_eps), float(current_price), n, prob
        )
    w_iv = sum(
        result[s]["present_value"] * result[s]["probability"] for s in ("bear", "base", "bull")
    )
    result["weighted_iv"] = float(w_iv)
    w_ret = (float(w_iv) / float(current_price)) ** (1 / n) - 1 if not pd.isna(w_iv) else float("nan")
    result["meets_hurdle"] = bool(not pd.isna(w_ret) and w_ret >= hurdle)
    return result


def compute_margin_of_safety(
    intrinsic_value: float,
    current_price: float,
) -> dict[str, Any]:
    """Compute Margin of Safety (F15).

    ``MoS = (Intrinsic Value − Current Price) / Intrinsic Value``

    Parameters
    ----------
    intrinsic_value:
        Weighted intrinsic value from F14 (``weighted_iv``).
    current_price:
        Current market price per share.

    Returns
    -------
    dict
        Keys: ``margin_of_safety`` (float), ``is_undervalued`` (bool),
        ``meets_threshold`` (bool — MoS ≥ ``recommendations.buy_min_mos``).

    Notes
    -----
    ``intrinsic_value`` NaN or ≤ 0 → ``margin_of_safety = NaN``, both bool
    flags ``False`` (NaN case logged at WARNING; ≤ 0 case logged at ERROR).
    Negative MoS (overvalued) is reported as-is, never clamped to zero.
    """
    cfg = get_config()
    buy_min_mos = float(cfg.get("recommendations", {}).get("buy_min_mos", 0.25))
    _nan_result: dict[str, Any] = {
        "margin_of_safety": float("nan"),
        "is_undervalued": False,
        "meets_threshold": False,
    }
    if pd.isna(intrinsic_value):
        logger.warning("Margin of safety: intrinsic_value is NaN. Cannot compute.")
        return _nan_result
    if float(intrinsic_value) <= 0:
        logger.error(
            "Margin of safety: intrinsic_value=%.4f ≤ 0 (pathological). Returning NaN.",
            intrinsic_value,
        )
        return _nan_result
    mos = (float(intrinsic_value) - float(current_price)) / float(intrinsic_value)
    return {
        "margin_of_safety": float(mos),
        "is_undervalued": bool(mos > 0),
        "meets_threshold": bool(mos >= buy_min_mos),
    }


def compute_earnings_yield(
    eps: float,
    price: float,
    risk_free_rate: float,
) -> dict[str, Any]:
    """Compute Earnings Yield vs Bond Yield spread (F16).

    ``Earnings Yield = EPS / Price``
    ``Spread = Earnings Yield − Risk-Free Rate``

    Parameters
    ----------
    eps:
        Trailing twelve-month diluted EPS.  May be negative (computed as-is).
    price:
        Current market price per share.  Must be > 0.
    risk_free_rate:
        10-year government bond yield (decimal, e.g. 0.04 for 4%).

    Returns
    -------
    dict
        Keys: ``earnings_yield`` (float), ``bond_yield`` (float),
        ``spread`` (float), ``equities_attractive`` (bool —
        spread > ``valuation.earnings_yield.min_spread_over_rfr_pct``).

    Notes
    -----
    ``price ≤ 0`` → all float fields NaN, ``equities_attractive = False``
    (logged at ERROR).  Negative EPS produces a negative yield — computed
    and returned without modification per F16 edge-case spec.
    """
    cfg = get_config()
    min_spread = float(
        cfg.get("valuation", {})
           .get("earnings_yield", {})
           .get("min_spread_over_rfr_pct", 0.02)
    )
    if pd.isna(price) or float(price) <= 0:
        logger.error(
            "Earnings yield: price=%.4f ≤ 0 or NaN. Cannot compute.", price
        )
        return {
            "earnings_yield": float("nan"),
            "bond_yield": float(risk_free_rate),
            "spread": float("nan"),
            "equities_attractive": False,
        }
    ey = float(eps) / float(price)
    spread = ey - float(risk_free_rate)
    return {
        "earnings_yield": float(ey),
        "bond_yield": float(risk_free_rate),
        "spread": float(spread),
        "equities_attractive": bool(spread > min_spread),
    }

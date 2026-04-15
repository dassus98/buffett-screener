"""Valuation formula functions: Intrinsic Value (F14), Margin of Safety (F15),
Earnings Yield vs Bond Yield (F16).

F14 projects intrinsic value using three weighted scenarios (bear / base / bull)
based on the Buffettology method. F15 computes the margin of safety as the
discount of current price to intrinsic value. F16 compares the equity earnings
yield to the risk-free rate to gauge relative attractiveness.

Data Lineage Contract
---------------------
Upstream producers:
    - ``metrics_engine.growth.compute_eps_cagr`` (F11)
      → provides ``eps_cagr`` used as the base growth rate for F14 projections.
    - ``data_acquisition.store.read_table("income_statement")``
      → provides ``eps_diluted`` (current EPS for F14 and F16).
    - ``data_acquisition.store.read_table("market_data")``
      → provides ``current_price_usd`` (all three functions) and
        ``pe_ratio_trailing`` (historical P/E proxy for F14).
    - ``data_acquisition.store.read_table("macro_data")``
      → provides risk-free rate (10-year government bond yield) for F14
        discount rate and F16 spread calculation.
    - ``metrics_engine.__init__._compute_valuation_formulas``
      → orchestrates the dependency chain: F11 → F14 → F15.

Downstream consumers:
    - ``metrics_engine.__init__._merge_all_summaries``
      → flattens F14 scenario dicts with ``f14_{scenario}_`` key prefixes;
        exposes ``weighted_iv``, ``meets_hurdle``, ``margin_of_safety``,
        ``is_undervalued``, ``meets_threshold``, ``earnings_yield``,
        ``spread``, ``equities_attractive``.
    - ``screener.hard_filters``
      → may read ``meets_hurdle`` for Tier 1 pass/fail (F14).
    - ``screener.soft_filters`` / recommendation engine
      → reads ``margin_of_safety``, ``meets_threshold``, ``equities_attractive``
        for ranking and buy/hold/sell decisions.

Config dependencies:
    - ``config/filter_config.yaml → valuation.projection_years``
    - ``config/filter_config.yaml → valuation.hurdle_rate``
    - ``config/filter_config.yaml → valuation.terminal_growth_rate``
    - ``config/filter_config.yaml → valuation.fallback_historical_pe``
    - ``config/filter_config.yaml → valuation.scenarios.{bear,base,bull}.*``
    - ``config/filter_config.yaml → valuation.earnings_yield.min_spread_over_rfr_pct``
    - ``config/filter_config.yaml → recommendations.buy_min_mos``
    - All thresholds loaded via ``screener.filter_config_loader.get_threshold()``.

Authoritative spec: docs/FORMULAS.md §F14, §F15, §F16.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_threshold

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
        ``valuation.scenarios`` sub-dict from config (loaded via
        ``get_threshold("valuation.scenarios")`` by the caller).
    fallback_pe:
        P/E to use when *pe_series* is empty or all-NaN.  Loaded from
        ``valuation.fallback_historical_pe`` in config.

    Returns
    -------
    dict
        Keys ``"bear"``, ``"base"``, ``"bull"`` each mapping to a float P/E.
    """
    # Drop NaN entries from the historical P/E series.
    valid = pe_series.dropna() if len(pe_series) > 0 else pd.Series(dtype=float)

    # If no valid P/E data is available, fall back to the industry median P/E
    # from config (per FORMULAS.md F14 edge case: "Use industry median P/E
    # from config as fallback").
    if valid.empty:
        logger.warning(
            "Historical P/E series is empty or all-NaN; using fallback P/E=%.1f.", fallback_pe
        )
        mean_pe = median_pe = fallback_pe
    else:
        mean_pe = float(valid.mean())
        median_pe = float(valid.median())

    # Per FORMULAS.md F14 three-scenario table:
    #   Bear P/E  = min(historical_mean_pe, pe_cap)   — caps optimism
    #   Base P/E  = median(historical_pe)             — central estimate
    #   Bull P/E  = max(historical_mean_pe, pe_floor) — floors pessimism
    bear_cap = float(scen_cfg["bear"]["pe_cap"])
    bull_floor = float(scen_cfg["bull"]["pe_floor"])
    return {
        "bear": min(mean_pe, bear_cap),
        "base": median_pe,
        "bull": max(mean_pe, bull_floor),
    }


def _resolve_growth_rates(
    eps_cagr: float,
    scen_cfg: dict[str, Any],
    terminal_growth: float,
) -> dict[str, float]:
    """Map *eps_cagr* to bear / base / bull growth rates, flooring negatives.

    Edge case — NaN or negative *eps_cagr*:
      bear = 0, base = max(0, eps_cagr),
      bull = max(eps_cagr × bull_multiplier, terminal_growth_rate)

    Parameters
    ----------
    eps_cagr:
        Historical 10-year EPS CAGR (may be NaN or negative).
    scen_cfg:
        ``valuation.scenarios`` sub-dict from config (loaded via
        ``get_threshold("valuation.scenarios")`` by the caller).
    terminal_growth:
        Perpetuity growth rate from ``valuation.terminal_growth_rate`` in
        config.  Used as the minimum bull-case growth when CAGR is negative.

    Returns
    -------
    dict
        Keys ``"bear"``, ``"base"``, ``"bull"`` each mapping to a float growth rate.
    """
    # Extract per-scenario growth multipliers from validated config.
    bear_mult = float(scen_cfg["bear"]["growth_multiplier"])
    base_mult = float(scen_cfg["base"]["growth_multiplier"])
    bull_mult = float(scen_cfg["bull"]["growth_multiplier"])

    # Edge case: NaN or negative CAGR → floor growth rates to prevent
    # pathological projections. Bear is always 0 (no growth in worst case).
    # Bull is floored at terminal_growth_rate so the bull scenario always
    # projects at least perpetuity-level growth.
    if pd.isna(eps_cagr) or float(eps_cagr) < 0:
        if not pd.isna(eps_cagr):
            logger.warning("EPS CAGR=%.4f < 0; flooring growth rates.", float(eps_cagr))
        cagr = 0.0 if pd.isna(eps_cagr) else float(eps_cagr)
        return {
            "bear": 0.0,
            "base": max(0.0, cagr),
            "bull": max(cagr * bull_mult, terminal_growth),
        }

    # Normal case: apply scenario multipliers to historical CAGR.
    # Bear = conservative (half growth), Base = historical, Bull = optimistic (1.3×).
    cagr = float(eps_cagr)
    return {
        "bear": cagr * bear_mult,
        "base": cagr * base_mult,
        "bull": cagr * bull_mult,
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
    # Step 1: Project EPS forward — Buffettology Ch. 11
    projected_eps = current_eps * (1 + growth_rate) ** n_years

    # Step 2: Project price = EPS × terminal P/E multiple
    projected_price = projected_eps * terminal_pe

    # Step 3: Discount projected price back to present value
    present_value = projected_price / (1 + discount_rate) ** n_years

    # Step 4: Compute projected annual return (what you'd earn buying at
    # current_price and selling at projected_price in n_years)
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
    # --- Step 1: Guard — current EPS must be positive to project forward ---
    # Per FORMULAS.md F14: "Current EPS ≤ 0 → Cannot project forward →
    # set all scenario IVs to NaN."
    if pd.isna(current_eps) or float(current_eps) <= 0:
        logger.warning(
            "Intrinsic value: current_eps=%.4f ≤ 0. Cannot project forward.", current_eps
        )
        return _nan_iv_result()

    # --- Step 2: Load all valuation parameters from config ---
    n = int(get_threshold("valuation.projection_years"))
    hurdle = float(get_threshold("valuation.hurdle_rate"))
    terminal_growth = float(get_threshold("valuation.terminal_growth_rate"))
    scen_cfg: dict[str, Any] = get_threshold("valuation.scenarios")
    fallback_pe = float(get_threshold("valuation.fallback_historical_pe"))

    # --- Step 3: Resolve terminal P/E and growth rates per scenario ---
    pe = _resolve_pe_estimates(historical_pe_series, scen_cfg, fallback_pe)
    gr = _resolve_growth_rates(eps_cagr, scen_cfg, terminal_growth)

    # --- Step 4: Compute each scenario (bear / base / bull) ---
    result: dict[str, Any] = {}
    for label in ("bear", "base", "bull"):
        sc = scen_cfg[label]
        # Discount rate = risk-free rate + scenario-specific risk premium
        dr = float(risk_free_rate) + float(sc["risk_premium"])
        # Scenario probability (weights must sum to 1.0: 0.25 + 0.50 + 0.25)
        prob = float(sc["probability"])
        result[label] = _compute_one_scenario(
            gr[label], pe[label], dr, float(current_eps), float(current_price), n, prob
        )

    # --- Step 5: Compute probability-weighted intrinsic value ---
    # Per FORMULAS.md F14: IV_weighted = Σ(IV_scenario × probability)
    w_iv = sum(
        result[s]["present_value"] * result[s]["probability"] for s in ("bear", "base", "bull")
    )
    result["weighted_iv"] = float(w_iv)

    # --- Step 6: Check hurdle rate ---
    # Projected weighted return = (weighted_IV / current_price)^(1/n) − 1
    # Per FORMULAS.md F14: must meet or exceed hurdle_rate (default 15%).
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
    # --- Step 1: Load the buy-minimum MoS threshold from config ---
    buy_min_mos = float(get_threshold("recommendations.buy_min_mos"))

    _nan_result: dict[str, Any] = {
        "margin_of_safety": float("nan"),
        "is_undervalued": False,
        "meets_threshold": False,
    }

    # --- Step 2: Guard — NaN intrinsic value → cannot compute MoS ---
    if pd.isna(intrinsic_value):
        logger.warning("Margin of safety: intrinsic_value is NaN. Cannot compute.")
        return _nan_result

    # --- Step 3: Guard — non-positive IV is pathological (negative IV = error) ---
    if float(intrinsic_value) <= 0:
        logger.error(
            "Margin of safety: intrinsic_value=%.4f ≤ 0 (pathological). Returning NaN.",
            intrinsic_value,
        )
        return _nan_result

    # --- Step 4: Compute MoS ---
    # Per FORMULAS.md F15: MoS = (IV − Price) / IV
    # Negative MoS = overvalued; reported as-is, never clamped to zero.
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
    # --- Step 1: Load minimum spread threshold from config ---
    min_spread = float(get_threshold("valuation.earnings_yield.min_spread_over_rfr_pct"))

    # --- Step 2: Guard — price must be positive ---
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

    # --- Step 3: Compute earnings yield and spread ---
    # Per FORMULAS.md F16: Earnings Yield = EPS / Price = 1 / P/E
    # Negative EPS produces a negative yield — computed and returned as-is.
    ey = float(eps) / float(price)

    # Spread = how much more the equity yields vs risk-free bonds.
    # Positive = equity premium; negative = bonds yield more.
    spread = ey - float(risk_free_rate)
    return {
        "earnings_yield": float(ey),
        "bond_yield": float(risk_free_rate),
        "spread": float(spread),
        "equities_attractive": bool(spread > min_spread),
    }

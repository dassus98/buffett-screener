"""Sensitivity analysis for the Margin of Safety (F15) section of the report.

Varies valuation inputs one axis at a time to show how intrinsic value and
margin of safety respond to changes in EPS growth, terminal P/E, and
discount rate.  This supplements the point-estimate from F14/F15 with a
range of outcomes to help the investor understand estimation risk.

Data Lineage Contract
---------------------
Upstream producers:
    - ``valuation_reports.intrinsic_value.compute_full_valuation``
      → provides the base valuation dict (``scenarios``, ``weighted_iv``).
    - ``metrics_engine.valuation.compute_intrinsic_value`` (F14)
      → provides the three-scenario result used as the base case.

Downstream consumers:
    - ``valuation_reports.report_generator``
      → reads the sensitivity tables for the Sensitivity Analysis section.

Config dependencies (all via ``get_threshold``):
    - ``valuation.projection_years``
    - ``valuation.scenarios.base.risk_premium``
    - ``output.sensitivity_eps_range``     (default 0.30)
    - ``output.sensitivity_pe_range``      (default 0.25)
    - ``output.sensitivity_discount_bps``  (default 200)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _project_iv(
    current_eps: float,
    growth: float,
    pe: float,
    discount: float,
    n_years: int,
) -> float:
    """Compute a single-scenario intrinsic value (present value of projected price).

    ``IV = current_eps × (1 + growth)^n × pe / (1 + discount)^n``

    Parameters
    ----------
    current_eps:
        Most-recent diluted EPS.
    growth:
        Annual EPS growth rate (decimal).
    pe:
        Terminal P/E multiple.
    discount:
        Annual discount rate (decimal).
    n_years:
        Projection horizon (years).

    Returns
    -------
    float
        Intrinsic value per share (present value).
    """
    projected_eps = current_eps * (1 + growth) ** n_years
    projected_price = projected_eps * pe
    return projected_price / (1 + discount) ** n_years


def _compute_mos(iv: float, current_price: float) -> float:
    """Compute margin of safety: ``(IV − price) / IV``.

    Returns ``NaN`` if IV ≤ 0.
    """
    if iv <= 0 or math.isnan(iv):
        return float("nan")
    return (iv - current_price) / iv


def _derive_current_eps(base_scenario: dict, n_years: int) -> float:
    """Back-derive current EPS from the base scenario's projected price.

    ``current_eps = projected_price / ((1 + growth)^n × pe)``

    Parameters
    ----------
    base_scenario:
        The ``"base"`` scenario dict from ``compute_intrinsic_value`` output.
    n_years:
        Projection horizon (years), read from config.

    Returns
    -------
    float
        Current EPS.  ``NaN`` if derivation fails.
    """
    growth = base_scenario.get("growth", float("nan"))
    pe = base_scenario.get("pe", float("nan"))
    proj_price = base_scenario.get("projected_price", float("nan"))
    if any(math.isnan(v) for v in (growth, pe, proj_price)) or pe == 0:
        return float("nan")
    return proj_price / ((1 + growth) ** n_years * pe)


def _make_steps(
    center: float,
    half_range: float,
    n_steps: int = 5,
) -> list[float]:
    """Generate *n_steps* evenly-spaced multiplicative adjustments.

    For *half_range* = 0.30 and *n_steps* = 5, produces
    ``[center×0.70, center×0.85, center×1.00, center×1.15, center×1.30]``.
    """
    fracs = [
        -half_range + i * (2 * half_range / (n_steps - 1))
        for i in range(n_steps)
    ]
    return [center * (1 + f) for f in fracs]


def _make_additive_steps(
    center: float,
    half_range: float,
    n_steps: int = 5,
) -> list[float]:
    """Generate *n_steps* evenly-spaced additive adjustments around *center*.

    For *half_range* = 0.02 and *n_steps* = 5, produces
    ``[center−0.02, center−0.01, center, center+0.01, center+0.02]``.
    """
    return [
        center - half_range + i * (2 * half_range / (n_steps - 1))
        for i in range(n_steps)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_sensitivity_table(
    base_valuation: dict[str, Any],
    eps_cagr: float,
    historical_pe: pd.Series,
    current_price: float,
    risk_free_rate: float,
) -> dict[str, list[tuple[float, float, float]]]:
    """Compute one-axis-at-a-time sensitivity tables for the report.

    Varies EPS growth, terminal P/E, and discount rate independently across
    5 evenly-spaced steps.  All other parameters are held at the base-
    scenario values from *base_valuation*.

    Parameters
    ----------
    base_valuation:
        Output of :func:`~valuation_reports.intrinsic_value.compute_full_valuation`
        (or equivalently, the raw ``compute_intrinsic_value`` result — must
        contain ``scenarios.base`` with ``growth``, ``pe``, ``projected_price``).
    eps_cagr:
        Historical 10-year EPS CAGR (used as the base growth rate).
    historical_pe:
        Historical P/E series (unused here; kept for interface symmetry with
        ``compute_intrinsic_value``).
    current_price:
        Current market price per share.
    risk_free_rate:
        10-year government bond yield (decimal).

    Returns
    -------
    dict
        Keys: ``eps_sensitivity``, ``pe_sensitivity``, ``discount_sensitivity``.
        Each value is a list of 5 ``(parameter_value, weighted_iv, mos)``
        tuples ordered from the most conservative to most optimistic step.
    """
    # --- Step 1: Read sensitivity ranges and projection years from config ---
    n_years = int(get_threshold("valuation.projection_years"))
    eps_range = float(get_threshold("output.sensitivity_eps_range"))
    pe_range = float(get_threshold("output.sensitivity_pe_range"))
    disc_bps = float(get_threshold("output.sensitivity_discount_bps"))
    base_risk_premium = float(
        get_threshold("valuation.scenarios.base.risk_premium"),
    )

    # --- Step 2: Extract base-scenario parameters ---
    #     The "scenarios" key may be nested (from compute_full_valuation) or
    #     flat (from compute_intrinsic_value).
    scenarios = base_valuation.get("scenarios", base_valuation)
    base_scen = scenarios.get("base", base_valuation.get("base", {}))

    base_growth = base_scen.get("growth", eps_cagr)
    base_pe = base_scen.get("pe", float("nan"))
    base_discount = risk_free_rate + base_risk_premium

    # --- Step 3: Derive current EPS from base scenario ---
    current_eps = _derive_current_eps(base_scen, n_years)
    if math.isnan(current_eps):
        logger.warning(
            "Cannot derive current_eps for sensitivity analysis; "
            "returning empty tables.",
        )
        return {
            "eps_sensitivity": [],
            "pe_sensitivity": [],
            "discount_sensitivity": [],
        }

    # --- Step 4: EPS growth sensitivity (vary growth, fix P/E and discount) ---
    growth_steps = _make_steps(base_growth, eps_range)
    eps_sens: list[tuple[float, float, float]] = []
    for g in growth_steps:
        iv = _project_iv(current_eps, g, base_pe, base_discount, n_years)
        mos = _compute_mos(iv, current_price)
        eps_sens.append((round(g, 6), round(iv, 4), round(mos, 4)))

    # --- Step 5: Terminal P/E sensitivity (vary P/E, fix growth and discount) ---
    pe_steps = _make_steps(base_pe, pe_range)
    pe_sens: list[tuple[float, float, float]] = []
    for p in pe_steps:
        iv = _project_iv(current_eps, base_growth, p, base_discount, n_years)
        mos = _compute_mos(iv, current_price)
        pe_sens.append((round(p, 4), round(iv, 4), round(mos, 4)))

    # --- Step 6: Discount rate sensitivity (vary discount, fix growth and P/E) ---
    disc_half = disc_bps / 10_000  # 200 bps → 0.02
    disc_steps = _make_additive_steps(base_discount, disc_half)
    disc_sens: list[tuple[float, float, float]] = []
    for d in disc_steps:
        iv = _project_iv(current_eps, base_growth, base_pe, d, n_years)
        mos = _compute_mos(iv, current_price)
        disc_sens.append((round(d, 6), round(iv, 4), round(mos, 4)))

    return {
        "eps_sensitivity": eps_sens,
        "pe_sensitivity": pe_sens,
        "discount_sensitivity": disc_sens,
    }

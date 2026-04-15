"""Wraps the F16 earnings-yield-vs-bond-yield result with interpretive context.

Provides a human-readable verdict and explanation for the Deep-Dive Report
Template's "Earnings Yield vs. Bond Yield" section.

Data Lineage Contract
---------------------
Upstream producers:
    - ``metrics_engine.valuation.compute_earnings_yield`` (F16)
      → computes the raw earnings yield, bond yield, and spread.

Downstream consumers:
    - ``valuation_reports.report_generator``
      → reads ``verdict`` and ``explanation`` for the report's F16 section.

Config dependencies (all via ``get_threshold``):
    - ``reports.yield_verdict.attractive_min_spread``  (default 0.04)
    - ``reports.yield_verdict.moderate_min_spread``    (default 0.02)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_yield_attractiveness(
    earnings_yield: float,
    bond_yield: float,
) -> dict[str, Any]:
    """Assess the attractiveness of earnings yield relative to bond yield.

    Wraps the raw F16 spread with a categorical verdict and a one-to-two
    sentence explanation suitable for the report.

    Parameters
    ----------
    earnings_yield:
        Equity earnings yield (EPS / price), decimal (e.g. 0.06 for 6 %).
    bond_yield:
        10-year government bond yield, decimal (e.g. 0.04 for 4 %).

    Returns
    -------
    dict
        Keys:

        * ``spread`` — earnings yield minus bond yield (decimal).
        * ``verdict`` — ``"Attractive"``, ``"Moderate"``, or
          ``"Unattractive"``.
        * ``explanation`` — one-to-two sentence human-readable
          interpretation of the spread.
    """
    # --- Step 1: Read verdict thresholds from config ---
    attractive_min = float(
        get_threshold("reports.yield_verdict.attractive_min_spread"),
    )
    moderate_min = float(
        get_threshold("reports.yield_verdict.moderate_min_spread"),
    )

    # --- Step 2: Compute spread ---
    if math.isnan(earnings_yield) or math.isnan(bond_yield):
        return {
            "spread": float("nan"),
            "verdict": "Unattractive",
            "explanation": (
                "Earnings yield or bond yield is unavailable; "
                "cannot assess relative attractiveness."
            ),
        }

    spread = earnings_yield - bond_yield

    # --- Step 3: Classify verdict using config thresholds ---
    if spread > attractive_min:
        verdict = "Attractive"
        explanation = (
            f"Earnings yield ({earnings_yield:.1%}) exceeds the bond yield "
            f"({bond_yield:.1%}) by {spread:.1%}, offering a strong premium "
            f"over the risk-free alternative."
        )
    elif spread >= moderate_min:
        verdict = "Moderate"
        explanation = (
            f"Earnings yield ({earnings_yield:.1%}) provides a {spread:.1%} "
            f"spread over the bond yield ({bond_yield:.1%}). "
            f"Equities offer a reasonable but not compelling premium."
        )
    else:
        verdict = "Unattractive"
        if spread < 0:
            explanation = (
                f"Earnings yield ({earnings_yield:.1%}) is below the bond "
                f"yield ({bond_yield:.1%}), resulting in a negative spread "
                f"of {spread:.1%}. Bonds are currently more attractive than "
                f"this equity on a yield basis."
            )
        else:
            explanation = (
                f"Earnings yield ({earnings_yield:.1%}) exceeds the bond "
                f"yield ({bond_yield:.1%}) by only {spread:.1%}. The premium "
                f"is insufficient to compensate for equity risk."
            )

    return {
        "spread": float(spread),
        "verdict": verdict,
        "explanation": explanation,
    }

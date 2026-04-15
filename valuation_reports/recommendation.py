"""Synthesises quality score and margin of safety into a structured
BUY / HOLD / PASS recommendation with justification.

Provides recommendation tier, confidence level, RRSP/TFSA account guidance,
sell-signal evaluation, and entry-strategy narrative for the Deep-Dive Report.

Data Lineage Contract
---------------------
Upstream producers:
    - ``screener.composite_score``  → ``composite_score`` (float 0–100)
    - ``valuation_reports.intrinsic_value.compute_full_valuation``
      → ``valuation`` dict with ``weighted_iv``, ``margin_of_safety``,
        ``scenarios``, ``current_price``, etc.
    - ``data_acquisition.store.read_table("data_quality_log")``
      → ``data_quality`` dict with ``years_available``,
        ``substitutions_count``, ``drop_reason``, etc.
    - ``metrics_engine`` summary dicts → ``metrics_summary`` for sell signals
      (``avg_roe_10yr``, ``gross_margin_avg_10yr``, ``de_ratio_latest``,
       ``debt_payoff_years``, ``return_on_retained_earnings``).

Downstream consumers:
    - ``valuation_reports.report_generator``
      → reads ``recommendation``, ``confidence``, ``reasoning``,
        ``account_recommendation``, ``sell_triggers``, ``entry_strategy``.

Config dependencies (all via ``get_threshold``):
    - ``recommendations.buy_min_mos``          (default 0.25)
    - ``recommendations.buy_min_score``        (default 70)
    - ``recommendations.hold_min_mos``         (default 0.10)
    - ``recommendations.hold_min_score``       (default 60)
    - ``recommendations.confidence.*``
    - ``recommendations.rrsp_us_dividend_yield_threshold``  (default 0.01)
    - ``recommendations.tfsa_preference_min_expected_return`` (default 0.12)
    - ``sell_signals.*``
"""

from __future__ import annotations

import logging
import math
from typing import Any

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)

# US exchanges for RRSP/TFSA logic
_US_EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX"})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _has_critical_flags(data_quality: dict[str, Any]) -> bool:
    """Return True if the data-quality dict indicates critical issues.

    A ticker has critical flags when:
      - ``drop`` is True (data quality forced exclusion), or
      - ``drop_reason`` is non-empty.
    """
    if data_quality.get("drop", False):
        return True
    reason = data_quality.get("drop_reason", "")
    return bool(reason and str(reason).strip())


def _classify_recommendation(
    composite_score: float,
    margin_of_safety: float,
    has_flags: bool,
    buy_min_mos: float,
    buy_min_score: float,
    hold_min_mos: float,
    hold_min_score: float,
) -> tuple[str, str]:
    """Apply the decision matrix and return ``(recommendation, reasoning)``.

    Per REPORT_SPEC.md §4:
      BUY:  MoS ≥ buy_min AND score ≥ buy_min AND no critical flags
      HOLD: MoS ≥ hold_min AND score ≥ hold_min
      PASS: everything else
    """
    if has_flags:
        return (
            "PASS",
            "Insufficient data confidence — critical data-quality flags "
            "prevent a reliable recommendation.",
        )

    if math.isnan(margin_of_safety) or math.isnan(composite_score):
        return (
            "PASS",
            "Unable to compute margin of safety or composite score; "
            "required inputs are unavailable.",
        )

    if margin_of_safety >= buy_min_mos and composite_score >= buy_min_score:
        return (
            "BUY",
            f"Margin of safety ({margin_of_safety:.1%}) meets the "
            f"{buy_min_mos:.0%} threshold and composite score "
            f"({composite_score:.1f}) exceeds the minimum of "
            f"{buy_min_score:.0f}.",
        )

    if margin_of_safety >= hold_min_mos and composite_score >= hold_min_score:
        return (
            "HOLD",
            f"Margin of safety ({margin_of_safety:.1%}) and composite "
            f"score ({composite_score:.1f}) meet HOLD thresholds but fall "
            f"short of BUY requirements. Consider accumulating on pullbacks.",
        )

    return (
        "PASS",
        f"Margin of safety ({margin_of_safety:.1%}) or composite score "
        f"({composite_score:.1f}) is below minimum HOLD thresholds.",
    )


def _classify_confidence(
    years_available: int,
    substitutions_count: int,
    margin_of_safety: float,
    high_min_years: int,
    high_max_subs: int,
    high_min_mos: float,
    mod_max_subs: int,
    mod_min_mos: float,
) -> str:
    """Assign confidence level per REPORT_SPEC.md §4.

    High:     years ≥ threshold AND subs = 0 AND MoS > high_min_mos
    Moderate: years ≥ threshold AND subs ≤ 2 AND MoS ≥ mod_min_mos
    Low:      everything else
    """
    if math.isnan(margin_of_safety):
        return "Low"

    if (
        years_available >= high_min_years
        and substitutions_count <= high_max_subs
        and margin_of_safety > high_min_mos
    ):
        return "High"

    if (
        years_available >= high_min_years
        and substitutions_count <= mod_max_subs
        and margin_of_safety >= mod_min_mos
    ):
        return "Moderate"

    return "Low"


def _is_approaching(
    current: float,
    threshold: float,
    *,
    lower_is_bad: bool = True,
) -> bool:
    """Check if *current* is within 20 % of *threshold*.

    For ``lower_is_bad=True`` (e.g. ROE floor): value is above threshold
    but within 20 % of it.
    For ``lower_is_bad=False`` (e.g. D/E max): value is below threshold
    but within 20 % of it.
    """
    if math.isnan(current) or math.isnan(threshold) or threshold == 0:
        return False

    buffer = abs(threshold) * 0.20

    if lower_is_bad:
        # e.g. ROE 0.13 approaching floor 0.12: above but close
        return current >= threshold and current <= threshold + buffer
    # e.g. D/E 0.85 approaching cap 1.0: below but close
    return current <= threshold and current >= threshold - buffer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_recommendation(
    ticker: str,
    composite_score: float,
    margin_of_safety: float,
    data_quality: dict[str, Any],
    valuation: dict[str, Any],
) -> dict[str, Any]:
    """Generate a BUY / HOLD / PASS recommendation with confidence level.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.
    composite_score:
        Buffett composite quality score (0–100).
    margin_of_safety:
        Margin of safety as decimal (0.25 = 25 %).
    data_quality:
        Dict with at minimum ``years_available`` (int),
        ``substitutions_count`` (int), ``drop`` (bool),
        ``drop_reason`` (str).
    valuation:
        Dict from ``compute_full_valuation`` — used for context only
        (weighted_iv, current_price).

    Returns
    -------
    dict
        Keys: ``recommendation``, ``confidence``, ``reasoning``.
    """
    # --- Step 1: Load thresholds from config ---
    buy_min_mos = float(get_threshold("recommendations.buy_min_mos"))
    buy_min_score = float(get_threshold("recommendations.buy_min_score"))
    hold_min_mos = float(get_threshold("recommendations.hold_min_mos"))
    hold_min_score = float(get_threshold("recommendations.hold_min_score"))

    # --- Step 2: Check for critical data-quality flags ---
    has_flags = _has_critical_flags(data_quality)

    # --- Step 3: Classify recommendation ---
    recommendation, reasoning = _classify_recommendation(
        composite_score,
        margin_of_safety,
        has_flags,
        buy_min_mos,
        buy_min_score,
        hold_min_mos,
        hold_min_score,
    )

    # --- Step 4: Classify confidence (independent of recommendation) ---
    high_min_years = int(get_threshold("recommendations.confidence.high_min_years"))
    high_max_subs = int(get_threshold("recommendations.confidence.high_max_substitutions"))
    high_min_mos = float(get_threshold("recommendations.confidence.high_min_mos"))
    mod_max_subs = int(get_threshold("recommendations.confidence.moderate_max_substitutions"))
    mod_min_mos = float(get_threshold("recommendations.confidence.moderate_min_mos"))

    years = int(data_quality.get("years_available", 0))
    subs = int(data_quality.get("substitutions_count", 0))

    confidence = _classify_confidence(
        years, subs, margin_of_safety,
        high_min_years, high_max_subs, high_min_mos,
        mod_max_subs, mod_min_mos,
    )

    logger.info(
        "%s → %s (confidence=%s, MoS=%.1f%%, score=%.1f)",
        ticker, recommendation, confidence,
        margin_of_safety * 100 if not math.isnan(margin_of_safety) else 0.0,
        composite_score,
    )

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def recommend_account(
    ticker: str,
    exchange: str,
    dividend_yield: float,
    expected_return: float,
) -> dict[str, str]:
    """Recommend RRSP, TFSA, or Either based on exchange and dividend yield.

    Implements the RRSP vs TFSA logic from REPORT_SPEC.md §3.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (used for .TO suffix detection).
    exchange:
        Listing exchange (``"TSX"``, ``"NYSE"``, ``"NASDAQ"``, etc.).
    dividend_yield:
        Trailing 12-month dividend yield as decimal (0.02 = 2 %).
    expected_return:
        Expected annualised return (decimal).

    Returns
    -------
    dict
        Keys: ``account`` (``"RRSP"`` | ``"TFSA"`` | ``"Either"``),
        ``reasoning`` (str).
    """
    # --- Step 1: Load thresholds from config ---
    rrsp_div_threshold = float(
        get_threshold("recommendations.rrsp_us_dividend_yield_threshold"),
    )
    tfsa_return_threshold = float(
        get_threshold("recommendations.tfsa_preference_min_expected_return"),
    )

    # --- Step 2: Determine whether the security is Canadian-listed ---
    is_canadian = (
        exchange == "TSX"
        or ticker.upper().endswith(".TO")
    )

    # --- Step 3: Apply decision logic ---
    if is_canadian:
        return {
            "account": "TFSA",
            "reasoning": (
                "Canadian dividends and capital gains are fully tax-free "
                "in TFSA. RRSP loses the dividend tax credit and withdrawals "
                "are taxed as ordinary income."
            ),
        }

    # US-listed securities
    if exchange in _US_EXCHANGES or not is_canadian:
        div = dividend_yield if not math.isnan(dividend_yield) else 0.0
        if div >= rrsp_div_threshold:
            return {
                "account": "RRSP",
                "reasoning": (
                    "US dividends are exempt from the 15% IRS withholding "
                    "tax in RRSP under the Canada-US tax treaty. TFSA is "
                    "not recognised as a pension account under the treaty."
                ),
            }
        # No significant dividend — slight TFSA preference
        return {
            "account": "Either",
            "reasoning": (
                "No withholding tax issue (minimal or no dividend). "
                "TFSA gains are fully tax-free vs. taxed as ordinary "
                "income on RRSP withdrawal. Slight TFSA preference for "
                "growth holdings."
            ),
        }

    # Fallback — should not be reached with valid data
    return {
        "account": "Either",
        "reasoning": (
            f"Exchange '{exchange}' is not explicitly handled. Defaulting "
            "to Either. Consult a tax professional."
        ),
    }


def generate_sell_signals(
    ticker: str,
    metrics_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate current metrics against sell-signal thresholds.

    Checks five signal categories from REPORT_SPEC.md §5 and classifies
    each as OK, WARNING (within 20 % of threshold), or TRIGGERED.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (for logging).
    metrics_summary:
        Dict of current metric values.  Expected keys:

        - ``avg_roe_10yr`` (float)
        - ``gross_margin_decline_3yr`` (float, optional — 3yr decline in pp;
          positive = decline)
        - ``de_ratio_latest`` (float)
        - ``debt_payoff_years`` (float)
        - ``return_on_retained_earnings`` (float)
        - ``bull_present_value`` (float, optional — bull IV)
        - ``current_price`` (float, optional)

    Returns
    -------
    list[dict]
        Each dict has: ``signal``, ``current_value``, ``threshold``,
        ``status`` (``"OK"`` | ``"WARNING"`` | ``"TRIGGERED"``),
        ``description``.
    """
    # --- Step 1: Load all sell-signal thresholds from config ---
    roe_floor = float(get_threshold("sell_signals.roe_floor"))
    gm_decline_pp = float(
        get_threshold("sell_signals.gross_margin_decline_pct_points"),
    )
    max_de = float(get_threshold("sell_signals.max_de_ratio"))
    max_debt_yrs = float(get_threshold("sell_signals.max_debt_payoff_years"))
    min_rore = float(get_threshold("sell_signals.min_retained_earnings_return"))

    signals: list[dict[str, Any]] = []

    # --- Step 2: ROE deterioration ---
    roe = metrics_summary.get("avg_roe_10yr", float("nan"))
    signals.append(
        _evaluate_floor_signal(
            signal="roe_deterioration",
            current=roe,
            threshold=roe_floor,
            description_ok="ROE remains above sell-signal floor.",
            description_warn=(
                f"ROE ({roe:.1%}) is approaching the sell-signal "
                f"floor of {roe_floor:.0%}. Monitor closely."
            ),
            description_triggered=(
                f"ROE ({roe:.1%}) has fallen below the sell-signal "
                f"floor of {roe_floor:.0%}. Consider exiting."
            ),
        ),
    )

    # --- Step 3: Gross margin erosion ---
    # Per REPORT_SPEC §5: "Gross margin declines > 5pp over any 3-year
    # rolling window."  This requires a historical decline figure.  If the
    # caller provides ``gross_margin_decline_3yr`` (positive = decline in
    # pp), use it.  Otherwise default to OK (cannot detect trend from a
    # single average).
    gm_decline = metrics_summary.get("gross_margin_decline_3yr", float("nan"))
    if math.isnan(gm_decline):
        signals.append({
            "signal": "gross_margin_erosion",
            "current_value": None,
            "threshold": float(gm_decline_pp),
            "status": "OK",
            "description": (
                "Gross margin erosion signal requires year-over-year data; "
                "no decline detected from available summary."
            ),
        })
    else:
        signals.append(
            _evaluate_ceiling_signal(
                signal="gross_margin_erosion",
                current=gm_decline,
                threshold=gm_decline_pp,
                description_ok="Gross margin stable; no erosion signal.",
                description_warn=(
                    f"Gross margin decline ({gm_decline:.1%}) is "
                    f"approaching the {gm_decline_pp:.0%} threshold."
                ),
                description_triggered=(
                    f"Gross margin has declined by {gm_decline:.1%} over "
                    f"the trailing 3-year window, exceeding the "
                    f"{gm_decline_pp:.0%} threshold. Review pricing "
                    f"power and competitive position."
                ),
            ),
        )

    # --- Step 4: Leverage spike (D/E) ---
    de = metrics_summary.get("de_ratio_latest", float("nan"))
    signals.append(
        _evaluate_ceiling_signal(
            signal="leverage_spike",
            current=de,
            threshold=max_de,
            description_ok="Debt-to-equity ratio is within acceptable range.",
            description_warn=(
                f"D/E ratio ({de:.2f}) is approaching the sell-signal "
                f"ceiling of {max_de:.1f}. Monitor leverage trend."
            ),
            description_triggered=(
                f"D/E ratio ({de:.2f}) exceeds the sell-signal ceiling "
                f"of {max_de:.1f}. Leverage is excessive."
            ),
        ),
    )

    # --- Step 5: Overvaluation ---
    bull_iv = metrics_summary.get("bull_present_value", float("nan"))
    price = metrics_summary.get("current_price", float("nan"))
    if not math.isnan(bull_iv) and not math.isnan(price) and bull_iv > 0:
        if price > bull_iv:
            status = "TRIGGERED"
            desc = (
                f"Current price (${price:.2f}) exceeds bull-case IV "
                f"(${bull_iv:.2f}). Stock appears overvalued even in "
                f"the most optimistic scenario."
            )
        elif _is_approaching(price, bull_iv, lower_is_bad=False):
            status = "WARNING"
            desc = (
                f"Current price (${price:.2f}) is approaching the "
                f"bull-case IV (${bull_iv:.2f})."
            )
        else:
            status = "OK"
            desc = "Price is below the bull-case intrinsic value."
    else:
        status = "OK"
        desc = "Overvaluation signal unavailable (missing bull IV or price)."

    signals.append({
        "signal": "overvaluation",
        "current_value": float(price) if not math.isnan(price) else None,
        "threshold": float(bull_iv) if not math.isnan(bull_iv) else None,
        "status": status,
        "description": desc,
    })

    # --- Step 6: Capital misallocation ---
    rore = metrics_summary.get("return_on_retained_earnings", float("nan"))
    signals.append(
        _evaluate_floor_signal(
            signal="capital_misallocation",
            current=rore,
            threshold=min_rore,
            description_ok=(
                "Return on retained earnings is above sell-signal floor."
            ),
            description_warn=(
                f"Return on retained earnings ({rore:.1%}) is approaching "
                f"the sell-signal floor of {min_rore:.0%}."
            ),
            description_triggered=(
                f"Return on retained earnings ({rore:.1%}) has fallen "
                f"below {min_rore:.0%}. Capital allocation may be "
                f"destroying shareholder value."
            ),
        ),
    )

    # Log any triggered signals
    for sig in signals:
        if sig["status"] == "TRIGGERED":
            logger.warning(
                "[%s] sell signal triggered: %s", ticker, sig["signal"],
            )

    return signals


def generate_entry_strategy(
    current_price: float,
    weighted_iv: float,
    margin_of_safety: float,
) -> dict[str, Any]:
    """Compute ideal entry price and a narrative entry strategy.

    Parameters
    ----------
    current_price:
        Current market price per share.
    weighted_iv:
        Probability-weighted intrinsic value from F14.
    margin_of_safety:
        Current MoS as decimal.

    Returns
    -------
    dict
        Keys: ``current_price``, ``ideal_entry``, ``discount_needed_pct``,
        ``strategy``.
    """
    # --- Step 1: Load buy_min_mos from config ---
    buy_min_mos = float(get_threshold("recommendations.buy_min_mos"))

    # --- Step 2: Guard NaN inputs ---
    if math.isnan(weighted_iv) or math.isnan(current_price):
        return {
            "current_price": float(current_price),
            "ideal_entry": float("nan"),
            "discount_needed_pct": float("nan"),
            "strategy": (
                "Cannot compute entry strategy — intrinsic value or "
                "current price is unavailable."
            ),
        }

    # --- Step 3: Compute ideal entry = IV × (1 − buy_min_mos) ---
    ideal_entry = weighted_iv * (1 - buy_min_mos)

    # --- Step 4: Compute discount needed (how much price needs to drop) ---
    if current_price > 0:
        discount_needed = (current_price - ideal_entry) / current_price
    else:
        discount_needed = float("nan")

    # --- Step 5: Build narrative ---
    if current_price <= ideal_entry:
        strategy = (
            f"Current price of ${current_price:.2f} is at or below the "
            f"ideal entry of ${ideal_entry:.2f} "
            f"({buy_min_mos:.0%} margin of safety). "
            f"The stock is within the buy zone."
        )
    else:
        pct_above = ((current_price / ideal_entry) - 1) * 100
        strategy = (
            f"Current price of ${current_price:.2f} is {pct_above:.1f}% "
            f"above the ideal entry of ${ideal_entry:.2f}. "
            f"Wait for a pullback or accumulate in tranches."
        )

    return {
        "current_price": float(current_price),
        "ideal_entry": float(ideal_entry),
        "discount_needed_pct": float(discount_needed),
        "strategy": strategy,
    }


# ---------------------------------------------------------------------------
# Private signal evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_floor_signal(
    signal: str,
    current: float,
    threshold: float,
    description_ok: str,
    description_warn: str,
    description_triggered: str,
) -> dict[str, Any]:
    """Evaluate a metric where falling *below* threshold is bad.

    Returns a signal dict with status OK / WARNING / TRIGGERED.
    """
    if math.isnan(current) or math.isnan(threshold):
        return {
            "signal": signal,
            "current_value": None,
            "threshold": float(threshold) if not math.isnan(threshold) else None,
            "status": "OK",
            "description": f"{signal}: data unavailable.",
        }

    if current < threshold:
        status = "TRIGGERED"
        desc = description_triggered
    elif _is_approaching(current, threshold, lower_is_bad=True):
        status = "WARNING"
        desc = description_warn
    else:
        status = "OK"
        desc = description_ok

    return {
        "signal": signal,
        "current_value": float(current),
        "threshold": float(threshold),
        "status": status,
        "description": desc,
    }


def _evaluate_ceiling_signal(
    signal: str,
    current: float,
    threshold: float,
    description_ok: str,
    description_warn: str,
    description_triggered: str,
) -> dict[str, Any]:
    """Evaluate a metric where *exceeding* threshold is bad.

    Returns a signal dict with status OK / WARNING / TRIGGERED.
    """
    if math.isnan(current) or math.isnan(threshold):
        return {
            "signal": signal,
            "current_value": None,
            "threshold": float(threshold) if not math.isnan(threshold) else None,
            "status": "OK",
            "description": f"{signal}: data unavailable.",
        }

    if current > threshold:
        status = "TRIGGERED"
        desc = description_triggered
    elif _is_approaching(current, threshold, lower_is_bad=False):
        status = "WARNING"
        desc = description_warn
    else:
        status = "OK"
        desc = description_ok

    return {
        "signal": signal,
        "current_value": float(current),
        "threshold": float(threshold),
        "status": status,
        "description": desc,
    }

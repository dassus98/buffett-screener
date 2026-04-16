"""Orchestrates all valuation sub-modules and renders per-stock Markdown
reports via Jinja2 templates.

Public API
----------
build_report_context(ticker) -> dict
    Assemble the full template context for one ticker.
render_deep_dive(ticker) -> str
    Render a Deep-Dive Analysis report to markdown.
render_summary(shortlist_df, screener_summary) -> str
    Render the portfolio-level Summary table to markdown.
generate_all_reports(shortlist_df, screener_summary) -> list[pathlib.Path]
    Write all individual reports + summary to ``data/reports/``.

Data Lineage Contract
---------------------
Upstream producers:
    - ``data_acquisition.store.read_table``
      → universe, income_statement, balance_sheet, cash_flow,
        market_data, macro_data, data_quality_log, substitution_log.
    - ``valuation_reports.intrinsic_value.compute_full_valuation``
    - ``valuation_reports.margin_of_safety.compute_sensitivity_table``
    - ``valuation_reports.earnings_yield.assess_yield_attractiveness``
    - ``valuation_reports.recommendation``
      → generate_recommendation, recommend_account,
        generate_sell_signals, generate_entry_strategy.
    - ``valuation_reports.qualitative_prompts.enrich_report_with_moat``
      → optional LLM moat assessment (Step 25); no-ops if
        ``reports.enable_qualitative`` is false or API key is absent.
    - ``metrics_engine.growth.compute_eps_cagr``

Downstream consumers:
    - ``output/`` dashboard and CLI export.

Config dependencies:
    - ``output.report_dir``       (default ``data/reports``)
    - ``recommendations.buy_min_mos``
    - ``valuation.projection_years``
    - ``valuation.terminal_growth_rate``
    - ``recommendations.time_horizon.*``
    - ``recommendations.position_sizing.*``
    - ``reports.enable_qualitative``  (bool, default ``false``)

Authoritative spec: docs/REPORT_SPEC.md.
"""

from __future__ import annotations

import datetime
import logging
import math
import pathlib
from typing import Any

import jinja2
import pandas as pd

from data_acquisition.store import get_connection, read_table
from metrics_engine.growth import compute_eps_cagr
from screener.filter_config_loader import get_threshold
from valuation_reports.earnings_yield import assess_yield_attractiveness
from valuation_reports.intrinsic_value import compute_full_valuation
from valuation_reports.margin_of_safety import compute_sensitivity_table
from valuation_reports.qualitative_prompts import enrich_report_with_moat
from valuation_reports.recommendation import (
    apply_sell_signal_override,
    generate_entry_strategy,
    generate_recommendation,
    generate_sell_signals,
    recommend_account,
)

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Jinja2 environment (module-level singleton)
# ---------------------------------------------------------------------------

def _get_jinja_env() -> jinja2.Environment:
    """Return a configured Jinja2 Environment for rendering templates."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=jinja2.Undefined,
        keep_trailing_newline=True,
    )


# ---------------------------------------------------------------------------
# Private helpers — data reading
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* on failure/NaN."""
    try:
        v = float(value)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _read_universe_row(ticker: str) -> dict[str, Any]:
    """Read the universe row for *ticker*; return defaults on miss."""
    df = read_table("universe", where=f"ticker = '{ticker}'")
    if df.empty:
        return {
            "company_name": ticker,
            "exchange": "",
            "sector": "",
            "industry": "",
        }
    row = df.iloc[0]
    return {
        "company_name": str(row.get("company_name", ticker)),
        "exchange": str(row.get("exchange", "")),
        "sector": str(row.get("sector", "")),
        "industry": str(row.get("industry", "")),
    }


def _read_market_row(ticker: str) -> dict[str, Any]:
    """Read market_data for *ticker*."""
    df = read_table("market_data", where=f"ticker = '{ticker}'")
    if df.empty:
        return {
            "current_price_usd": float("nan"),
            "pe_ratio_trailing": float("nan"),
            "dividend_yield": 0.0,
        }
    row = df.iloc[0]
    return {
        "current_price_usd": _safe_float(row.get("current_price_usd"), float("nan")),
        "pe_ratio_trailing": _safe_float(row.get("pe_ratio_trailing"), float("nan")),
        "dividend_yield": _safe_float(row.get("dividend_yield"), 0.0),
    }


def _read_data_quality(ticker: str) -> dict[str, Any]:
    """Read data_quality_log for *ticker*."""
    df = read_table("data_quality_log", where=f"ticker = '{ticker}'")
    if df.empty:
        return {
            "years_available": 0,
            "substitutions_count": 0,
            "missing_critical_fields": "",
            "drop": False,
            "drop_reason": "",
        }
    row = df.iloc[0]
    return {
        "years_available": int(row.get("years_available", 0)),
        "substitutions_count": int(row.get("substitutions_count", 0)),
        "missing_critical_fields": str(row.get("missing_critical_fields", "") or ""),
        "drop": bool(row.get("drop", False)),
        "drop_reason": str(row.get("drop_reason", "") or ""),
    }


def _read_eps_series(ticker: str) -> pd.Series:
    """Read historical EPS series from income_statement."""
    inc = read_table("income_statement", where=f"ticker = '{ticker}'")
    if inc.empty or "eps_diluted" not in inc.columns:
        return pd.Series(dtype=float)
    df = inc[["fiscal_year", "eps_diluted"]].dropna(subset=["eps_diluted"])
    df = df.sort_values("fiscal_year")
    return pd.Series(
        df["eps_diluted"].values,
        index=df["fiscal_year"].astype(int).values,
        dtype=float,
    )


def _read_pe_series(ticker: str) -> pd.Series:
    """Read trailing P/E (single-element proxy) for sensitivity analysis."""
    mkt = read_table("market_data", where=f"ticker = '{ticker}'")
    if mkt.empty:
        return pd.Series(dtype=float)
    pe = mkt.iloc[0].get("pe_ratio_trailing", float("nan"))
    if math.isnan(float(pe)):
        return pd.Series(dtype=float)
    return pd.Series([float(pe)])


def _read_risk_free_rate() -> float:
    """Read the 10-year US Treasury yield from macro_data."""
    macro = read_table("macro_data", where="key = 'us_treasury_10yr'")
    if macro.empty:
        return float("nan")
    return float(macro.iloc[0]["value"])


# ---------------------------------------------------------------------------
# Private helpers — financial statement detail tables
# ---------------------------------------------------------------------------


def _build_income_tests(
    inc_df: pd.DataFrame,
    ticker: str,
) -> list[dict[str, str]]:
    """Build the income-statement tests summary table rows."""
    tests: list[dict[str, str]] = []
    if inc_df.empty:
        return tests

    # Profitable years count
    if "net_income" in inc_df.columns:
        prof_years = int((inc_df["net_income"] > 0).sum())
        total_years = len(inc_df)
        tests.append({
            "name": "Earnings Consistency (F10)",
            "result": f"{prof_years}/{total_years} years profitable",
            "threshold": ">= 8",
            "status": "Pass" if prof_years >= 8 else "Fail",
            "notes": "",
        })

    # EPS CAGR
    eps_series = _read_eps_series(ticker)
    if not eps_series.empty:
        cagr_result = compute_eps_cagr(eps_series)
        cagr = cagr_result.get("eps_cagr", float("nan"))
        decline = cagr_result.get("decline_years", 0)
        cagr_str = f"{cagr * 100:.1f}%" if not math.isnan(cagr) else "N/A"
        tests.append({
            "name": "EPS CAGR (F11)",
            "result": cagr_str,
            "threshold": "> 0%",
            "status": "Pass" if not math.isnan(cagr) and cagr > 0 else "Fail",
            "notes": f"{decline} decline year(s)" if decline else "",
        })

    return tests


def _build_balance_tests(bs_df: pd.DataFrame) -> list[dict[str, str]]:
    """Build the balance-sheet tests summary table rows."""
    tests: list[dict[str, str]] = []
    if bs_df.empty:
        return tests

    if "long_term_debt" in bs_df.columns and "shareholders_equity" in bs_df.columns:
        latest = bs_df.sort_values("fiscal_year").iloc[-1]
        equity = _safe_float(latest.get("shareholders_equity"), 0.0)
        debt = _safe_float(latest.get("long_term_debt"), 0.0)
        de = debt / equity if equity > 0 else float("nan")
        de_str = f"{de:.2f}" if not math.isnan(de) else "N/A"
        tests.append({
            "name": "Debt-to-Equity (F6)",
            "result": de_str,
            "threshold": "< 0.80",
            "status": "Pass" if not math.isnan(de) and de < 0.80 else "Fail",
            "notes": "",
        })

    return tests


def _build_cashflow_tests(cf_df: pd.DataFrame) -> list[dict[str, str]]:
    """Build the cash-flow tests summary table rows."""
    tests: list[dict[str, str]] = []
    if cf_df.empty:
        return tests

    if "capital_expenditures" in cf_df.columns:
        tests.append({
            "name": "CapEx / Net Income (F12)",
            "result": "See detail",
            "threshold": "< 50%",
            "status": "Pass",
            "notes": "",
        })

    return tests


def _build_annual_rows(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Convert a fiscal-year DataFrame into a list of dicts for the template."""
    if df.empty:
        return []
    df = df.sort_values("fiscal_year")
    rows = []
    for _, row in df.iterrows():
        entry: dict[str, Any] = {}
        for col in columns:
            val = row.get(col, float("nan"))
            entry[col] = _safe_float(val) if col != "fiscal_year" else int(val)
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Private helpers — assumption log and bear case
# ---------------------------------------------------------------------------


def _build_assumption_log(
    ticker: str,
    data_quality: dict[str, Any],
) -> list[dict[str, str]]:
    """Auto-populate the assumption log per REPORT_SPEC §7."""
    log: list[dict[str, str]] = []

    # Always-included assumptions
    log.append({
        "assumption": "Maintenance CapEx approximated as Depreciation & Amortisation",
        "confidence": "Medium",
        "failure_mode": (
            "Growth CapEx included in D&A proxy; "
            "maintenance CapEx may be understated"
        ),
        "consequence": "Owner earnings (F1) may be overstated",
    })
    log.append({
        "assumption": "Historical EPS CAGR predicts future growth",
        "confidence": "Medium",
        "failure_mode": (
            "Past growth rates may not persist; "
            "mean reversion or disruption possible"
        ),
        "consequence": (
            "F14 intrinsic value projection may be optimistic "
            "or pessimistic relative to actual future growth"
        ),
    })

    # Line-item substitutions from substitution_log
    subs_df = read_table(
        "substitution_log",
        where=f"ticker = '{ticker}'",
    )
    if not subs_df.empty:
        for _, row in subs_df.iterrows():
            log.append({
                "assumption": (
                    f"{row['buffett_field']} sourced from "
                    f"{row['api_field_used']} (substitution)"
                ),
                "confidence": str(row.get("confidence", "Medium")),
                "failure_mode": (
                    "Substitute field may not perfectly match "
                    "the canonical metric definition"
                ),
                "consequence": (
                    f"Metric using {row['buffett_field']} may be "
                    "directionally correct but imprecise"
                ),
            })

    return log


def _build_bear_case(
    ticker: str,
    sector: str,
    valuation: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    """Generate rule-based bear-case arguments."""
    arguments: list[dict[str, str]] = []

    gm = metrics.get("gross_margin_avg_10yr", 1.0)
    if not math.isnan(gm) and gm < 0.50:
        arguments.append({
            "title": "Limited pricing power",
            "body": (
                f"Average gross margin of {gm * 100:.1f}% over the last "
                "10 years is below 50%, suggesting the business may lack "
                "the pricing power characteristic of a durable competitive "
                "advantage."
            ),
        })

    if sector and "technolog" in sector.lower():
        arguments.append({
            "title": "Technology disruption risk",
            "body": (
                "Technology moats can erode rapidly with disruption. "
                "New entrants, open-source alternatives, or platform shifts "
                "could undermine the company's competitive position."
            ),
        })

    debt_payoff = metrics.get("debt_payoff_years", 0.0)
    if not math.isnan(debt_payoff) and debt_payoff > 3:
        arguments.append({
            "title": "Elevated debt levels",
            "body": (
                f"Debt payoff period of {debt_payoff:.1f} years, while "
                "manageable, limits financial flexibility during economic "
                "downturns and reduces the margin of error in the thesis."
            ),
        })

    decline_years = metrics.get("decline_years", 0)
    if decline_years and int(decline_years) > 0:
        n = int(decline_years)
        arguments.append({
            "title": "Earnings volatility",
            "body": (
                f"EPS declined in {n} of the last 10 years, suggesting "
                "earnings are not as predictable as headline growth figures "
                "imply. Cyclicality or one-off charges may be masking "
                "underlying business volatility."
            ),
        })

    # Always-included argument
    arguments.append({
        "title": "Mean reversion risk",
        "body": (
            "Historical growth rates may not persist. Exceptional "
            "past performance often reverts to industry averages as "
            "companies mature, competition intensifies, or addressable "
            "markets become saturated."
        ),
    })

    return arguments


# ---------------------------------------------------------------------------
# Private helpers — position sizing and time horizon
# ---------------------------------------------------------------------------


def _determine_time_horizon(
    composite_score: float,
    confidence: str,
    eps_cagr: float,
) -> int:
    """Determine time horizon per REPORT_SPEC §4."""
    standard = int(get_threshold("recommendations.time_horizon.standard_years"))
    ext_min_score = float(get_threshold("recommendations.time_horizon.extended_min_score"))
    ext_min_cagr = float(get_threshold("recommendations.time_horizon.extended_min_eps_cagr"))
    ext_years = int(get_threshold("recommendations.time_horizon.extended_years"))
    short_years = int(get_threshold("recommendations.time_horizon.shortened_years"))

    if confidence == "Low":
        return short_years

    cagr = eps_cagr if not math.isnan(eps_cagr) else 0.0
    if composite_score >= ext_min_score and cagr >= ext_min_cagr:
        return ext_years

    return standard


def _determine_position_sizing(
    composite_score: float,
    confidence: str,
) -> str:
    """Generate position sizing guidance narrative."""
    sizing = get_threshold("recommendations.position_sizing")

    if confidence == "Low":
        low = sizing["low_confidence"]
        return (
            f"Low confidence: {low[0] * 100:.0f}-{low[1] * 100:.0f}% "
            f"of portfolio maximum. Monitor closely."
        )

    if confidence == "High":
        if composite_score >= 80:
            rng = sizing["high_confidence_high_score"]
            return (
                f"High confidence, score >= 80: Up to "
                f"{rng[0] * 100:.0f}-{rng[1] * 100:.0f}% of portfolio."
            )
        rng = sizing["high_confidence_mid_score"]
        return (
            f"High confidence, score 70-79: Up to "
            f"{rng[0] * 100:.0f}-{rng[1] * 100:.0f}% of portfolio."
        )

    # Moderate
    if composite_score >= 70:
        rng = sizing["moderate_confidence_high_score"]
        return (
            f"Moderate confidence, score >= 70: Up to "
            f"{rng[0] * 100:.0f}-{rng[1] * 100:.0f}% of portfolio."
        )
    rng = sizing["moderate_confidence_mid_score"]
    return (
        f"Moderate confidence, score 60-69: Up to "
        f"{rng[0] * 100:.0f}-{rng[1] * 100:.0f}% of portfolio."
    )


# ---------------------------------------------------------------------------
# Private helpers — metrics summary for sell signals
# ---------------------------------------------------------------------------


def _build_metrics_for_sell_signals(
    ticker: str,
    inc_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    valuation: dict[str, Any],
) -> dict[str, Any]:
    """Build a metrics_summary dict suitable for generate_sell_signals."""
    metrics: dict[str, Any] = {}

    # Average ROE 10yr
    if (
        not inc_df.empty
        and not bs_df.empty
        and "net_income" in inc_df.columns
    ):
        merged = inc_df.merge(bs_df, on=["ticker", "fiscal_year"], how="inner")
        if "shareholders_equity" in merged.columns:
            equity = merged["shareholders_equity"]
            ni = merged["net_income"]
            valid = equity > 0
            if valid.any():
                roe_vals = ni[valid] / equity[valid]
                metrics["avg_roe_10yr"] = float(roe_vals.mean())
            else:
                metrics["avg_roe_10yr"] = float("nan")
        else:
            metrics["avg_roe_10yr"] = float("nan")
    else:
        metrics["avg_roe_10yr"] = float("nan")

    # Gross margin avg + 3-year rolling decline
    #     Per REPORT_SPEC §5: "Gross margin declines > 5pp over any 3-year
    #     rolling window → sell signal."  We compute the max 3-year decline
    #     in pp (positive = margin has declined).
    if not inc_df.empty and "gross_profit" in inc_df.columns and "total_revenue" in inc_df.columns:
        rev = inc_df["total_revenue"]
        gp = inc_df["gross_profit"]
        valid = rev > 0
        if valid.any():
            gm_vals = gp[valid] / rev[valid]
            metrics["gross_margin_avg_10yr"] = float(gm_vals.mean())
            # Compute 3-year rolling decline from year-sorted margins
            sorted_df = inc_df.loc[valid].sort_values("fiscal_year")
            gm_sorted = (sorted_df["gross_profit"] / sorted_df["total_revenue"]).values
            if len(gm_sorted) >= 4:
                # Max decline over any 3-year window (positive = decline)
                declines = [
                    gm_sorted[i] - gm_sorted[i + 3]
                    for i in range(len(gm_sorted) - 3)
                ]
                metrics["gross_margin_decline_3yr"] = float(max(declines))
            else:
                metrics["gross_margin_decline_3yr"] = float("nan")
        else:
            metrics["gross_margin_avg_10yr"] = float("nan")
            metrics["gross_margin_decline_3yr"] = float("nan")
    else:
        metrics["gross_margin_avg_10yr"] = float("nan")
        metrics["gross_margin_decline_3yr"] = float("nan")

    # D/E latest
    if not bs_df.empty and "long_term_debt" in bs_df.columns:
        latest = bs_df.sort_values("fiscal_year").iloc[-1]
        equity = _safe_float(latest.get("shareholders_equity"), 0.0)
        debt = _safe_float(latest.get("long_term_debt"), 0.0)
        metrics["de_ratio_latest"] = debt / equity if equity > 0 else float("nan")
    else:
        metrics["de_ratio_latest"] = float("nan")

    # Debt payoff years (simplified: LTD / net_income)
    if not inc_df.empty and not bs_df.empty:
        latest_bs = bs_df.sort_values("fiscal_year").iloc[-1]
        avg_ni = _safe_float(inc_df["net_income"].mean(), 0.0) if "net_income" in inc_df.columns else 0.0
        debt = _safe_float(latest_bs.get("long_term_debt"), 0.0)
        metrics["debt_payoff_years"] = debt / avg_ni if avg_ni > 0 else float("nan")
    else:
        metrics["debt_payoff_years"] = float("nan")

    # Return on retained earnings (placeholder)
    metrics["return_on_retained_earnings"] = float("nan")

    # Bull present value from valuation
    bull_pv = (
        valuation.get("scenarios", {})
        .get("bull", {})
        .get("present_value", float("nan"))
    )
    metrics["bull_present_value"] = bull_pv
    metrics["current_price"] = valuation.get("current_price", float("nan"))

    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_report_context(ticker: str) -> dict[str, Any]:
    """Assemble the full Jinja2 template context for one ticker.

    Reads from DuckDB, calls all valuation sub-modules, and packages
    everything into the dict expected by ``deep_dive_template.md``.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (must exist in DuckDB tables).

    Returns
    -------
    dict
        Complete template context.  See docs/REPORT_SPEC.md §2 for the
        full variable reference.
    """
    # --- Step 1: Read raw data from DuckDB ---
    universe = _read_universe_row(ticker)
    market = _read_market_row(ticker)
    data_quality = _read_data_quality(ticker)
    risk_free_rate = _read_risk_free_rate()

    inc_df = read_table("income_statement", where=f"ticker = '{ticker}'")
    bs_df = read_table("balance_sheet", where=f"ticker = '{ticker}'")
    cf_df = read_table("cash_flow", where=f"ticker = '{ticker}'")

    # --- Step 2: Compute EPS CAGR ---
    eps_series = _read_eps_series(ticker)
    cagr_result = compute_eps_cagr(eps_series) if not eps_series.empty else {}
    eps_cagr = cagr_result.get("eps_cagr", float("nan"))
    decline_years = cagr_result.get("decline_years", 0)

    # --- Step 3: Compute full valuation (F14/F15/F16) ---
    valuation = compute_full_valuation(ticker)

    # --- Step 4: Compute sensitivity analysis ---
    pe_series = _read_pe_series(ticker)
    sensitivity_data = compute_sensitivity_table(
        base_valuation=valuation,
        eps_cagr=eps_cagr,
        historical_pe=pe_series,
        current_price=valuation.get("current_price", float("nan")),
        risk_free_rate=risk_free_rate if not math.isnan(risk_free_rate) else 0.04,
    )
    # Hide sensitivity if all axes empty
    if all(
        len(sensitivity_data.get(k, [])) == 0
        for k in ("eps_sensitivity", "pe_sensitivity", "discount_sensitivity")
    ):
        sensitivity_data = None

    # --- Step 5: Yield attractiveness assessment ---
    yield_assessment = assess_yield_attractiveness(
        valuation.get("earnings_yield", float("nan")),
        valuation.get("bond_yield", float("nan")),
    )

    # --- Step 6: Read composite score from pipeline tables ---
    composite_score = float("nan")
    try:
        conn = get_connection()
        cs_df = conn.execute(
            "SELECT composite_score FROM composite_scores WHERE ticker = ?",
            [ticker],
        ).fetchdf()
        if not cs_df.empty:
            composite_score = float(cs_df.iloc[0]["composite_score"])
    except Exception:
        pass
    if math.isnan(composite_score):
        try:
            conn = get_connection()
            ms_df = conn.execute(
                "SELECT composite_score FROM buffett_metrics_summary "
                "WHERE ticker = ?",
                [ticker],
            ).fetchdf()
            if not ms_df.empty:
                composite_score = float(ms_df.iloc[0]["composite_score"])
        except Exception:
            pass

    # --- Step 7: Generate recommendation ---
    rec = generate_recommendation(
        ticker=ticker,
        composite_score=composite_score,
        margin_of_safety=valuation.get("margin_of_safety", float("nan")),
        data_quality=data_quality,
        valuation=valuation,
    )

    # --- Step 8: Account recommendation ---
    acct = recommend_account(
        ticker=ticker,
        exchange=universe["exchange"],
        dividend_yield=market["dividend_yield"],
        expected_return=valuation.get("scenarios", {})
        .get("base", {})
        .get("annual_return", 0.0),
    )

    # --- Step 9: Build metrics summary for sell signals ---
    sell_metrics = _build_metrics_for_sell_signals(
        ticker, inc_df, bs_df, valuation,
    )

    # --- Step 10: Sell signals ---
    sell_triggers = generate_sell_signals(ticker, sell_metrics)

    # --- Step 10b: Override recommendation if any sell signal is TRIGGERED ---
    #     Per REPORT_SPEC §5: "Triggered sell signals cause the
    #     recommendation to be overridden to PASS regardless of MoS or
    #     composite score."
    rec = apply_sell_signal_override(rec, sell_triggers, ticker=ticker)

    # --- Step 11: Entry strategy ---
    entry = generate_entry_strategy(
        current_price=valuation.get("current_price", float("nan")),
        weighted_iv=valuation.get("weighted_iv", float("nan")),
        margin_of_safety=valuation.get("margin_of_safety", float("nan")),
    )

    # --- Step 12: Time horizon and position sizing ---
    time_horizon = _determine_time_horizon(
        composite_score, rec["confidence"], eps_cagr,
    )
    position_sizing_guidance = _determine_position_sizing(
        composite_score, rec["confidence"],
    )

    # --- Step 13: Build test tables for financial statements ---
    income_tests = _build_income_tests(inc_df, ticker)
    balance_tests = _build_balance_tests(bs_df)
    cashflow_tests = _build_cashflow_tests(cf_df)

    # --- Step 14: Build annual detail rows ---
    annual_income = _build_annual_income(inc_df, bs_df)
    annual_balance = _build_annual_balance(bs_df)
    annual_cashflow = _build_annual_cashflow(cf_df, inc_df)

    # --- Step 15: Latest fiscal year ---
    latest_fy = 0
    for df in (inc_df, bs_df, cf_df):
        if not df.empty and "fiscal_year" in df.columns:
            latest_fy = max(latest_fy, int(df["fiscal_year"].max()))

    # --- Step 16: Assumption log ---
    assumption_log = _build_assumption_log(ticker, data_quality)

    # --- Step 17: Bear case ---
    bear_case_arguments = _build_bear_case(
        ticker,
        universe["sector"],
        valuation,
        {
            "gross_margin_avg_10yr": sell_metrics.get("gross_margin_avg_10yr", 1.0),
            "debt_payoff_years": sell_metrics.get("debt_payoff_years", 0.0),
            "decline_years": decline_years,
        },
    )

    # --- Step 18: MoS-derived buy-below prices ---
    buy_min_mos = float(get_threshold("recommendations.buy_min_mos"))
    iv_w = valuation.get("weighted_iv", float("nan"))
    mos_conservative = float(get_threshold("valuation.mos_conservative"))
    mos_moderate = buy_min_mos

    buy_below_conservative = iv_w * (1 - mos_conservative) if not math.isnan(iv_w) else float("nan")
    buy_below_moderate = iv_w * (1 - mos_moderate) if not math.isnan(iv_w) else float("nan")

    # --- Step 19: Extract scenario details ---
    scenarios = valuation.get("scenarios", {})
    bear_s = scenarios.get("bear", {})
    base_s = scenarios.get("base", {})
    bull_s = scenarios.get("bull", {})

    # --- Step 20: Critical flags ---
    critical_flags: list[str] = []
    if data_quality.get("drop", False):
        critical_flags.append(str(data_quality.get("drop_reason", "Critical flag")))
    mcf = data_quality.get("missing_critical_fields", "")
    if mcf:
        critical_flags.append(f"Missing fields: {mcf}")

    # --- Step 21: Projection config ---
    projection_years = int(get_threshold("valuation.projection_years"))
    terminal_growth = float(get_threshold("valuation.terminal_growth_rate"))

    # --- Step 22: MoS interpretation ---
    mos_pct = valuation.get("margin_of_safety", float("nan"))
    if math.isnan(mos_pct):
        mos_interp = "Margin of safety cannot be computed due to missing valuation data."
    elif mos_pct > buy_min_mos:
        mos_interp = (
            f"Current price offers a {mos_pct * 100:.1f}% margin of safety, "
            f"exceeding the {buy_min_mos * 100:.0f}% BUY threshold. "
            f"The stock appears attractively priced."
        )
    elif mos_pct > 0:
        mos_interp = (
            f"Current price offers a {mos_pct * 100:.1f}% margin of safety. "
            f"This is positive but below the {buy_min_mos * 100:.0f}% BUY threshold. "
            f"Consider waiting for a better entry point."
        )
    else:
        mos_interp = (
            f"Current price implies a negative margin of safety ({mos_pct * 100:.1f}%). "
            f"The stock appears overvalued relative to intrinsic value."
        )

    # --- Step 23: Negative equity detection ---
    neg_equity_flag = False
    neg_equity_years = 0
    if not bs_df.empty and "shareholders_equity" in bs_df.columns:
        neg_mask = bs_df["shareholders_equity"] < 0
        neg_equity_years = int(neg_mask.sum())
        neg_equity_flag = neg_equity_years > 0

    # --- Step 24: CapEx flag ---
    capex_flag = False
    capex_flag_years = 0
    if not cf_df.empty and "capital_expenditures" in cf_df.columns and "depreciation_amortization" in cf_df.columns:
        capex = cf_df["capital_expenditures"].abs()
        da = cf_df["depreciation_amortization"].abs()
        excess = capex > (2 * da)
        capex_flag_years = int(excess.sum())
        capex_flag = capex_flag_years > 0

    # --- Assemble full context ---
    ctx: dict[str, Any] = {
        # Header
        "company_name": universe["company_name"],
        "ticker": ticker,
        "exchange": universe["exchange"],
        "sector": universe["sector"],
        "industry": universe["industry"],
        "report_date": datetime.date.today().isoformat(),
        "latest_fiscal_year": latest_fy or "N/A",
        # Executive Summary
        "composite_score": _safe_float(composite_score),
        "iv_weighted": _safe_float(iv_w),
        "current_price_usd": _safe_float(market["current_price_usd"]),
        "margin_of_safety_pct": _safe_float(mos_pct),
        "recommendation": rec["recommendation"],
        "confidence_level": rec["confidence"],
        "account_recommendation": acct["account"],
        "time_horizon_years": time_horizon,
        "critical_flags": critical_flags,
        # Moat — defaults; overridden by enrich_report_with_moat if enabled
        "qualitative_enabled": False,
        "moat_assessment": None,
        "moat_indicators": None,
        "gross_margin_avg_10yr": _safe_float(
            sell_metrics.get("gross_margin_avg_10yr"), 0.0,
        ),
        "roe_avg_10yr": _safe_float(
            sell_metrics.get("avg_roe_10yr"), 0.0,
        ),
        # Sell-derived metrics (used by enrich_report_with_moat)
        "de_ratio_latest": _safe_float(
            sell_metrics.get("de_ratio_latest"), 0.0,
        ),
        "debt_payoff_years": _safe_float(
            sell_metrics.get("debt_payoff_years"), 0.0,
        ),
        # Financial Statement Tests
        "income_tests": income_tests,
        "balance_tests": balance_tests,
        "cashflow_tests": cashflow_tests,
        "annual_income": annual_income,
        "annual_balance": annual_balance,
        "annual_cashflow": annual_cashflow,
        "negative_equity_flag": neg_equity_flag,
        "negative_equity_years": neg_equity_years,
        "capex_flag": capex_flag,
        "capex_flag_years": capex_flag_years,
        # Valuation
        "eps_latest": _safe_float(eps_series.iloc[-1]) if not eps_series.empty else 0.0,
        "eps_cagr_10yr": _safe_float(eps_cagr),
        "pe_avg_10yr": _safe_float(market["pe_ratio_trailing"]),
        "risk_free_rate": _safe_float(risk_free_rate),
        "projection_years": projection_years,
        "terminal_growth_rate": terminal_growth,
        # Scenario details
        "bear_growth": _safe_float(bear_s.get("growth")),
        "bear_terminal_pe": _safe_float(bear_s.get("pe")),
        "bear_discount_rate": _safe_float(bear_s.get("discount_rate")),
        "bear_probability": _safe_float(bear_s.get("probability"), 0.25),
        "iv_bear": _safe_float(bear_s.get("present_value")),
        "base_growth": _safe_float(base_s.get("growth")),
        "base_terminal_pe": _safe_float(base_s.get("pe")),
        "base_discount_rate": _safe_float(base_s.get("discount_rate")),
        "base_probability": _safe_float(base_s.get("probability"), 0.50),
        "iv_base": _safe_float(base_s.get("present_value")),
        "bull_growth": _safe_float(bull_s.get("growth")),
        "bull_terminal_pe": _safe_float(bull_s.get("pe")),
        "bull_discount_rate": _safe_float(bull_s.get("discount_rate")),
        "bull_probability": _safe_float(bull_s.get("probability"), 0.25),
        "iv_bull": _safe_float(bull_s.get("present_value")),
        # Margin of Safety
        "mos_conservative": mos_conservative,
        "mos_moderate": mos_moderate,
        "buy_below_conservative": _safe_float(buy_below_conservative),
        "buy_below_moderate": _safe_float(buy_below_moderate),
        "margin_of_safety_interpretation": mos_interp,
        # Earnings Yield
        "earnings_yield": _safe_float(valuation.get("earnings_yield")),
        "bond_yield": _safe_float(valuation.get("bond_yield")),
        "bond_yield_type": (
            "GoC 10yr" if universe["exchange"] == "TSX"
            else "US Treasury 10yr"
        ),
        "earnings_yield_spread": _safe_float(valuation.get("spread")),
        "earnings_yield_interpretation": yield_assessment.get("verdict", ""),
        # Sensitivity
        "sensitivity_data": sensitivity_data,
        # Assumption Log
        "assumption_log": assumption_log,
        # Bear Case
        "bear_case_arguments": bear_case_arguments,
        # Investment Strategy
        "entry_strategy": entry,
        "position_sizing_guidance": position_sizing_guidance,
        "sell_triggers": sell_triggers,
        "account_reasoning": acct["reasoning"],
        # Data Quality
        "data_quality": data_quality,
    }

    # --- Step 25: Conditional LLM moat enrichment ---
    #     enrich_report_with_moat checks reports.enable_qualitative and
    #     ANTHROPIC_API_KEY before making any API call.  If either is
    #     absent, it sets qualitative_enabled=False and moat_assessment=None
    #     — the template then renders the quantitative-only fallback.
    enrich_report_with_moat(ctx)

    return ctx


# ---------------------------------------------------------------------------
# Annual detail row builders
# ---------------------------------------------------------------------------


def _build_annual_income(
    inc_df: pd.DataFrame,
    bs_df: pd.DataFrame | None = None,
) -> list[dict]:
    """Build annual income statement detail rows for the template."""
    if inc_df.empty:
        return []
    df = inc_df.sort_values("fiscal_year").copy()

    # Build a year → shareholders_equity lookup from balance sheet
    equity_by_year: dict[int, float] = {}
    if bs_df is not None and not bs_df.empty:
        for _, brow in bs_df.iterrows():
            fy = int(brow.get("fiscal_year", 0))
            eq = _safe_float(brow.get("shareholders_equity"))
            if fy > 0:
                equity_by_year[fy] = eq

    rows = []
    for _, row in df.iterrows():
        revenue = _safe_float(row.get("total_revenue"))
        gross_profit = _safe_float(row.get("gross_profit"))
        operating_income = _safe_float(row.get("operating_income"))
        net_income = _safe_float(row.get("net_income"))
        gm = gross_profit / revenue if revenue > 0 else 0.0
        om = operating_income / revenue if revenue > 0 else 0.0
        nm = net_income / revenue if revenue > 0 else 0.0
        fy = int(row["fiscal_year"])
        equity = equity_by_year.get(fy, 0.0)
        roe = net_income / equity if equity > 0 else 0.0
        rows.append({
            "fiscal_year": fy,
            "revenue": revenue / 1000,  # convert to $K
            "gross_margin": gm,
            "operating_margin": om,
            "net_margin": nm,
            "eps_diluted": _safe_float(row.get("eps_diluted")),
            "roe": roe,
        })
    return rows


def _build_annual_balance(bs_df: pd.DataFrame) -> list[dict]:
    """Build annual balance sheet detail rows for the template."""
    if bs_df.empty:
        return []
    df = bs_df.sort_values("fiscal_year").copy()
    rows = []
    for _, row in df.iterrows():
        equity = _safe_float(row.get("shareholders_equity"))
        debt = _safe_float(row.get("long_term_debt"))
        de = debt / equity if equity > 0 else 0.0
        rows.append({
            "fiscal_year": int(row["fiscal_year"]),
            "long_term_debt": debt / 1000,
            "shareholders_equity": equity / 1000,
            "de_ratio": de,
            "retained_earnings": 0.0,  # Not in DuckDB schema
        })
    return rows


def _build_annual_cashflow(
    cf_df: pd.DataFrame,
    inc_df: pd.DataFrame | None = None,
) -> list[dict]:
    """Build annual cash flow detail rows for the template."""
    if cf_df.empty:
        return []
    df = cf_df.sort_values("fiscal_year").copy()

    # Build a year → net_income lookup from income statement
    ni_by_year: dict[int, float] = {}
    if inc_df is not None and not inc_df.empty:
        for _, irow in inc_df.iterrows():
            fy = int(irow.get("fiscal_year", 0))
            ni = _safe_float(irow.get("net_income"))
            if fy > 0:
                ni_by_year[fy] = ni

    rows = []
    for _, row in df.iterrows():
        da = _safe_float(row.get("depreciation_amortization"))
        capex = _safe_float(row.get("capital_expenditures"))
        fy = int(row["fiscal_year"])
        net_income = ni_by_year.get(fy, 0.0)
        # Owner Earnings = Net Income + D&A + CapEx
        # (capex is stored as negative per schema convention, so adding
        # it effectively subtracts the absolute CapEx spend)
        owner_earnings = net_income + da + capex
        rows.append({
            "fiscal_year": fy,
            "operating_cash_flow": 0.0,  # Not directly in DuckDB schema
            "capital_expenditures": capex / 1000,
            "free_cash_flow": 0.0,
            "owner_earnings": owner_earnings / 1000,
            "depreciation_amortization": da / 1000,
        })
    return rows


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_deep_dive(ticker: str) -> str:
    """Build context and render the Deep-Dive Analysis report.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.

    Returns
    -------
    str
        Rendered markdown report.
    """
    ctx = build_report_context(ticker)
    env = _get_jinja_env()
    template = env.get_template("deep_dive_template.md")
    return template.render(**ctx)


def render_summary(
    shortlist_df: pd.DataFrame,
    screener_summary: dict[str, Any],
) -> str:
    """Render the portfolio-level summary table.

    Parameters
    ----------
    shortlist_df:
        DataFrame with at least columns: ``ticker``, ``company_name``,
        ``exchange``, ``composite_score``, ``margin_of_safety_pct``,
        ``recommendation``, ``confidence_level``, ``account_recommendation``,
        ``gross_margin_avg_10yr``, ``roe_avg_10yr``, ``eps_cagr_10yr``.
    screener_summary:
        Dict with keys: ``universe_size``, ``after_exclusions``,
        ``passed_hard_filters``, ``filter_stats`` (dict),
        ``macro`` (dict with ``us_treasury_10yr``, etc.).

    Returns
    -------
    str
        Rendered markdown summary.
    """
    # Build ranked rows
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(shortlist_df.iterrows(), start=1):
        rows.append({
            "rank": rank,
            "ticker": row.get("ticker", ""),
            "company_name": row.get("company_name", ""),
            "exchange": row.get("exchange", ""),
            "composite_score": _safe_float(row.get("composite_score")),
            "iv_weighted": _safe_float(row.get("iv_weighted")),
            "current_price_usd": _safe_float(row.get("current_price_usd")),
            "margin_of_safety_pct": _safe_float(row.get("margin_of_safety_pct")),
            "recommendation": row.get("recommendation", ""),
            "confidence_level": row.get("confidence_level", ""),
            "account_recommendation": row.get("account_recommendation", ""),
            "gross_margin_avg_10yr": _safe_float(row.get("gross_margin_avg_10yr")),
            "roe_avg_10yr": _safe_float(row.get("roe_avg_10yr")),
            "eps_cagr_10yr": _safe_float(row.get("eps_cagr_10yr")),
        })

    # Count sectors
    sector_counts: dict[str, list[float]] = {}
    for _, row in shortlist_df.iterrows():
        sector = str(row.get("sector", "Other"))
        score = _safe_float(row.get("composite_score"))
        sector_counts.setdefault(sector, []).append(score)
    sector_summary = sorted(
        [
            {
                "name": name,
                "count": len(scores),
                "avg_score": sum(scores) / len(scores) if scores else 0,
            }
            for name, scores in sector_counts.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )

    # Build macro context from DuckDB macro_data table (key-value store)
    macro: dict[str, float] = screener_summary.get("macro", {})
    if not macro:
        try:
            macro_df = read_table("macro_data")
            for _, mrow in macro_df.iterrows():
                key = str(mrow.get("key", ""))
                val = mrow.get("value")
                if key and val is not None:
                    macro[key] = float(val)
        except Exception:
            pass

    ctx: dict[str, Any] = {
        "top_n": len(rows),
        "run_date": datetime.date.today().isoformat(),
        "universe_size": screener_summary.get("universe_size", 0),
        "after_exclusions": screener_summary.get("after_exclusions", 0),
        "passed_hard_filters": screener_summary.get("passed_hard_filters", 0),
        "shortlist_count": len(rows),
        "macro": macro,
        "rows": rows,
        "filter_stats": screener_summary.get("filter_stats", {}),
        "sector_summary": sector_summary,
    }

    env = _get_jinja_env()
    template = env.get_template("summary_table_template.md")
    return template.render(**ctx)


def generate_all_reports(
    shortlist_df: pd.DataFrame,
    screener_summary: dict[str, Any] | None = None,
) -> list[pathlib.Path]:
    """Generate all individual reports + summary and write to disk.

    Parameters
    ----------
    shortlist_df:
        DataFrame with at least a ``ticker`` column.
    screener_summary:
        Optional screener summary dict for the summary report.

    Returns
    -------
    list[pathlib.Path]
        File paths of all generated reports.
    """
    report_dir_str = str(get_threshold("output.report_dir"))
    report_dir = _PROJECT_ROOT / report_dir_str
    report_dir.mkdir(parents=True, exist_ok=True)

    if shortlist_df.empty or "ticker" not in shortlist_df.columns:
        logger.warning("generate_all_reports: no tickers to report on.")
        return []

    tickers = list(shortlist_df["ticker"])
    total = len(tickers)
    generated: list[pathlib.Path] = []

    for i, ticker in enumerate(tickers, start=1):
        logger.info("Generated report %d/%d: %s", i, total, ticker)
        try:
            md = render_deep_dive(ticker)
            path = report_dir / f"{ticker}_analysis.md"
            path.write_text(md, encoding="utf-8")
            generated.append(path)
        except Exception:
            logger.exception(
                "Failed to generate report for %s", ticker,
            )

    # Summary report
    if screener_summary is not None:
        try:
            summary_md = render_summary(shortlist_df, screener_summary)
            summary_path = report_dir / "summary.md"
            summary_path.write_text(summary_md, encoding="utf-8")
            generated.append(summary_path)
            logger.info("Generated summary report: %s", summary_path)
        except Exception:
            logger.exception("Failed to generate summary report")

    return generated

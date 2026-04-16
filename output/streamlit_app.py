"""Interactive Streamlit dashboard for exploring screener results,
deep-dive valuations, and DCF sensitivity analysis.

Launch with::

    streamlit run output/streamlit_app.py
    # or via pipeline:
    python -m output.pipeline_runner --mode dashboard

Data Lineage Contract
---------------------
Upstream producers:
    - ``data_acquisition.store`` → all DuckDB tables (universe, income_statement,
      balance_sheet, cash_flow, market_data, macro_data, data_quality_log).
    - ``metrics_engine`` → ``buffett_metrics_summary``, ``composite_scores``.
    - ``valuation_reports.report_generator`` → ``build_report_context(ticker)``.

Config dependencies (all via ``get_threshold``):
    - ``recommendations.buy_min_mos`` — MoS threshold for BUY.
    - ``recommendations.buy_min_score`` — composite score threshold for BUY.
    - ``recommendations.hold_min_mos`` — MoS threshold for HOLD.
    - ``recommendations.hold_min_score`` — composite score threshold for HOLD.
    - ``recommendations.rrsp_us_dividend_yield_threshold`` — dividend yield
      threshold for RRSP account recommendation.
    - ``reports.yield_verdict.attractive_min_spread`` — spread for Attractive.
    - ``reports.yield_verdict.moderate_min_spread`` — spread for Moderate.
    - ``output.report_dir`` — report directory for pre-rendered Markdown.
    - ``output.shortlist_size`` — default top-N display count.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration — MUST be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Buffett Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_DB_PATH = _PROJECT_ROOT / "data" / "processed" / "buffett.duckdb"

_RECOMMENDATION_COLORS: dict[str, str] = {
    "BUY": "#16a34a",
    "HOLD": "#ca8a04",
    "PASS": "#dc2626",
}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


def _db_exists() -> bool:
    """Check whether the DuckDB file exists on disk."""
    return _DB_PATH.exists()


@st.cache_data(ttl=300)
def _load_table(table_name: str) -> pd.DataFrame:
    """Load a full DuckDB table via the store module.

    Parameters
    ----------
    table_name:
        DuckDB table name.

    Returns
    -------
    pd.DataFrame
        Full table contents (may be empty).
    """
    from data_acquisition.store import read_table

    try:
        return read_table(table_name)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_report_context(ticker: str) -> dict[str, Any] | None:
    """Load the full report context for a single ticker.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.

    Returns
    -------
    dict | None
        Full template context from ``build_report_context``, or ``None``
        on any failure.
    """
    try:
        from valuation_reports.report_generator import build_report_context

        return build_report_context(ticker)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper: safe numeric formatting
# ---------------------------------------------------------------------------


def _fmt_pct(value: Any, decimals: int = 1) -> str:
    """Format a ratio as a percentage string.

    Parameters
    ----------
    value:
        Numeric value (0.25 → ``"25.0%"``).
    decimals:
        Number of decimal places.

    Returns
    -------
    str
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "—"
        return f"{v * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_usd(value: Any, decimals: int = 2) -> str:
    """Format a dollar amount.

    Parameters
    ----------
    value:
        Numeric USD value.
    decimals:
        Decimal places.

    Returns
    -------
    str
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "—"
        return f"${v:,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(value: Any, decimals: int = 1) -> str:
    """Format a general number.

    Parameters
    ----------
    value:
        Numeric value.
    decimals:
        Decimal places.

    Returns
    -------
    str
    """
    try:
        v = float(value)
        if math.isnan(v):
            return "—"
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert to float safely; return *default* on failure or NaN."""
    try:
        v = float(value)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Recommendation / account helpers (config-driven, no hardcoding)
# ---------------------------------------------------------------------------


def _get_recommendation_thresholds() -> dict[str, float]:
    """Load recommendation thresholds from config with safe fallbacks.

    Returns
    -------
    dict[str, float]
        Keys: ``buy_min_mos``, ``buy_min_score``, ``hold_min_mos``,
        ``hold_min_score``.
    """
    defaults = {
        "buy_min_mos": 0.25,
        "buy_min_score": 70.0,
        "hold_min_mos": 0.10,
        "hold_min_score": 60.0,
    }
    result: dict[str, float] = {}
    for key, fallback in defaults.items():
        try:
            result[key] = float(get_threshold(f"recommendations.{key}"))
        except Exception:
            result[key] = fallback
    return result


def _compute_recommendation(mos: float, score: float) -> str:
    """Derive BUY / HOLD / PASS from margin of safety and composite score.

    Uses thresholds from ``config/filter_config.yaml`` under
    ``recommendations``.  Falls back to spec defaults if config is
    unavailable.

    Parameters
    ----------
    mos:
        Margin of safety as a decimal (e.g. 0.25 for 25 %).
    score:
        Composite score (0–100 scale).

    Returns
    -------
    str
        ``"BUY"``, ``"HOLD"``, or ``"PASS"``.
    """
    if math.isnan(mos) or math.isnan(score):
        return "PASS"
    thresholds = _get_recommendation_thresholds()
    if mos >= thresholds["buy_min_mos"] and score >= thresholds["buy_min_score"]:
        return "BUY"
    if mos >= thresholds["hold_min_mos"] and score >= thresholds["hold_min_score"]:
        return "HOLD"
    return "PASS"


def _compute_account(exchange: str, dividend_yield: float) -> str:
    """Derive RRSP / TFSA account recommendation.

    Logic (from REPORT_SPEC.md §Account Recommendation):
    - US-listed with dividend yield ≥ threshold → RRSP (treaty benefit).
    - All other cases → TFSA (tax-free growth).

    Parameters
    ----------
    exchange:
        Listing exchange (``"TSX"``, ``"NYSE"``, ``"NASDAQ"``, etc.).
    dividend_yield:
        Trailing dividend yield as a decimal.

    Returns
    -------
    str
        ``"RRSP"`` or ``"TFSA"``.
    """
    try:
        rrsp_threshold = float(
            get_threshold("recommendations.rrsp_us_dividend_yield_threshold"),
        )
    except Exception:
        rrsp_threshold = 0.01

    is_us = exchange in {"NYSE", "NASDAQ"}
    div_y = _safe_float(dividend_yield)

    if is_us and div_y >= rrsp_threshold:
        return "RRSP"
    return "TFSA"


# ---------------------------------------------------------------------------
# Build the ranked results table from raw DuckDB data
# ---------------------------------------------------------------------------


def _list_available_tables() -> set[str]:
    """Return names of all DuckDB tables in the ``main`` schema.

    Returns
    -------
    set[str]
        Table names, or empty set on failure.
    """
    try:
        from data_acquisition.store import get_connection

        conn = get_connection()
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'",
        ).fetchdf()
        return set(tables["table_name"].tolist())
    except Exception:
        return set()


def _join_scores_and_metrics(
    df: pd.DataFrame,
    available_tables: set[str],
) -> pd.DataFrame:
    """Join composite scores and metrics summary onto *df*."""
    # Load metrics summary if available
    metrics = (
        _load_table("buffett_metrics_summary")
        if "buffett_metrics_summary" in available_tables
        else pd.DataFrame()
    )
    scores = (
        _load_table("composite_scores")
        if "composite_scores" in available_tables
        else pd.DataFrame()
    )

    # Join composite scores
    if not scores.empty and "ticker" in scores.columns:
        score_cols = [c for c in scores.columns if c in
                      {"ticker", "composite_score"}]
        df = df.merge(scores[score_cols], on="ticker", how="left")
    elif not metrics.empty and "composite_score" in metrics.columns:
        df = df.merge(
            metrics[["ticker", "composite_score"]],
            on="ticker", how="left",
        )

    # Join key metrics from metrics_summary
    if not metrics.empty and "ticker" in metrics.columns:
        metric_cols = ["ticker"]
        for col in ("avg_roe", "eps_cagr", "gross_margin_avg",
                     "debt_payoff_years", "margin_of_safety",
                     "weighted_iv", "owner_earnings_yield"):
            if col in metrics.columns:
                metric_cols.append(col)
        if len(metric_cols) > 1:
            df = df.merge(metrics[metric_cols], on="ticker", how="left")

    return df


def _add_recommendation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``recommendation`` and ``account`` columns to *df*.

    Derived from margin_of_safety, composite_score, exchange, and
    dividend_yield using config-driven thresholds.

    Parameters
    ----------
    df:
        Ranked securities DataFrame.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with two new columns appended.
    """
    mos_col = "margin_of_safety" if "margin_of_safety" in df.columns else None
    score_col = "composite_score" if "composite_score" in df.columns else None
    div_col = "dividend_yield" if "dividend_yield" in df.columns else None

    rec_values: list[str] = []
    acct_values: list[str] = []
    for _, row in df.iterrows():
        mos_val = _safe_float(
            row.get(mos_col) if mos_col else None, float("nan"),
        )
        score_val = _safe_float(
            row.get(score_col) if score_col else None, float("nan"),
        )
        rec_values.append(_compute_recommendation(mos_val, score_val))
        acct_values.append(_compute_account(
            str(row.get("exchange", "")),
            _safe_float(row.get(div_col) if div_col else None),
        ))

    df["recommendation"] = rec_values
    df["account"] = acct_values
    return df


@st.cache_data(ttl=300)
def _build_ranked_table() -> pd.DataFrame:
    """Assemble the ranked securities table for Tab 1.

    Joins universe, market_data, composite_scores, and metrics_summary
    tables to produce a single display-ready DataFrame with
    recommendation and account columns.

    Returns
    -------
    pd.DataFrame
        Columns include: rank, ticker, company_name, exchange, sector,
        composite_score, weighted_iv, current_price_usd,
        margin_of_safety, recommendation, account.
    """
    universe = _load_table("universe")
    if universe.empty:
        return pd.DataFrame()

    df = universe[["ticker", "company_name", "exchange", "sector"]].copy()

    # Join market data (price, P/E, dividend yield)
    market = _load_table("market_data")
    if not market.empty and "ticker" in market.columns:
        market_cols = ["ticker", "current_price_usd", "pe_ratio_trailing",
                       "dividend_yield"]
        available = [c for c in market_cols if c in market.columns]
        df = df.merge(market[available], on="ticker", how="left")

    # Join scores + metrics from pipeline tables
    df = _join_scores_and_metrics(df, _list_available_tables())

    # Filter to survivors (non-dropped tickers)
    quality = _load_table("data_quality_log")
    if not quality.empty and "ticker" in quality.columns:
        survivors = quality[quality["drop"] == False]["ticker"].tolist()  # noqa: E712
        if survivors:
            df = df[df["ticker"].isin(survivors)]

    # Sort, rank, and derive recommendation/account
    if "composite_score" in df.columns:
        df = df.sort_values("composite_score", ascending=False)
    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    return _add_recommendation_columns(df)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------


def _apply_range_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render numeric range sliders and filter *df* accordingly.

    Sliders: Composite Score (0–100), Margin of Safety (-50 % to 100 %).

    Parameters
    ----------
    df:
        Input DataFrame (modified copy returned).

    Returns
    -------
    pd.DataFrame
    """
    if "composite_score" in df.columns:
        lo, hi = st.sidebar.slider(
            "Composite Score Range", 0, 100, (0, 100), step=1,
        )
        df = df[
            df["composite_score"].between(lo, hi)
            | df["composite_score"].isna()
        ]

    if "margin_of_safety" in df.columns:
        lo, hi = st.sidebar.slider(
            "Margin of Safety (%)", -50, 100, (-50, 100), step=1,
        )
        df = df[
            df["margin_of_safety"].between(lo / 100, hi / 100)
            | df["margin_of_safety"].isna()
        ]
    return df


def _apply_category_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render multiselect widgets for exchange, sector, recommendation.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    for col, label in [
        ("exchange", "Exchange"),
        ("sector", "Sector"),
    ]:
        if col in df.columns:
            opts = sorted(df[col].dropna().unique().tolist())
            if opts:
                sel = st.sidebar.multiselect(label, opts, default=opts)
                if sel:
                    df = df[df[col].isin(sel)]

    # Recommendation filter (fixed options)
    if "recommendation" in df.columns:
        rec_opts = ["BUY", "HOLD", "PASS"]
        sel_recs = st.sidebar.multiselect(
            "Recommendation", rec_opts, default=rec_opts,
        )
        if sel_recs:
            df = df[df["recommendation"].isin(sel_recs)]

    return df


def _render_sidebar(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filter widgets and return the filtered DataFrame.

    Parameters
    ----------
    ranked_df:
        Full ranked securities DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered subset based on user selections.
    """
    st.sidebar.markdown("## Filters")

    filtered = ranked_df.copy()
    filtered = _apply_range_filters(filtered)
    filtered = _apply_category_filters(filtered)

    # Show top N slider
    top_n = st.sidebar.slider("Show Top N", 10, 100, 50, step=5)
    filtered = filtered.head(top_n)

    # Re-rank after filtering
    filtered = filtered.reset_index(drop=True)
    filtered["rank"] = range(1, len(filtered) + 1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Showing {len(filtered)}** of {len(ranked_df)} securities",
    )
    return filtered


# ---------------------------------------------------------------------------
# Tab 1: Ranked Results
# ---------------------------------------------------------------------------


_COLUMN_RENAME: dict[str, str] = {
    "rank": "Rank",
    "ticker": "Ticker",
    "company_name": "Company",
    "exchange": "Exchange",
    "sector": "Sector",
    "composite_score": "Score",
    "weighted_iv": "Weighted IV",
    "current_price_usd": "Price (USD)",
    "margin_of_safety": "MoS %",
    "recommendation": "Rec.",
    "account": "Account",
}

# Ordered list of optional columns to include after the base set.
_OPTIONAL_DISPLAY_COLS: tuple[str, ...] = (
    "composite_score", "weighted_iv", "current_price_usd",
    "margin_of_safety", "recommendation", "account",
)


def _style_rec(val: str) -> str:
    """Return CSS colour for a recommendation label.

    BUY → green, HOLD → amber, PASS → red.
    """
    color = _RECOMMENDATION_COLORS.get(str(val), "")
    if color:
        return f"color: {color}; font-weight: 700"
    return ""


def _prepare_display_df(source_df: pd.DataFrame) -> pd.DataFrame:
    """Select, rename, and format columns for the ranked results table.

    Parameters
    ----------
    source_df:
        Raw filtered DataFrame.

    Returns
    -------
    pd.DataFrame
        Display-ready DataFrame with friendly column names and
        formatted numeric values.
    """
    cols = ["rank", "ticker", "company_name", "exchange", "sector"]
    for c in _OPTIONAL_DISPLAY_COLS:
        if c in source_df.columns:
            cols.append(c)

    df = source_df[[c for c in cols if c in source_df.columns]].copy()
    df = df.rename(
        columns={k: v for k, v in _COLUMN_RENAME.items() if k in df.columns},
    )

    # Format helpers (NaN-safe)
    _dollar = lambda x: f"${x:,.2f}" if pd.notna(x) else "—"  # noqa: E731
    _pct = lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "—"  # noqa: E731
    _score = lambda x: f"{x:.1f}" if pd.notna(x) else "—"  # noqa: E731

    for col, fn in [
        ("MoS %", _pct),
        ("Price (USD)", _dollar),
        ("Weighted IV", _dollar),
        ("Score", _score),
    ]:
        if col in df.columns:
            df[col] = df[col].apply(fn)

    return df


def _render_tab_ranked(filtered_df: pd.DataFrame) -> str | None:
    """Render the ranked results table and return selected ticker.

    Parameters
    ----------
    filtered_df:
        Filtered and ranked DataFrame.

    Returns
    -------
    str | None
        Selected ticker for deep dive, or ``None``.
    """
    st.subheader("Ranked Securities")

    if filtered_df.empty:
        st.info("No securities match the current filters.")
        return None

    display_df = _prepare_display_df(filtered_df)

    # Apply recommendation colour via pandas Styler
    styled = display_df.style
    if "Rec." in display_df.columns:
        styled = styled.map(_style_rec, subset=["Rec."])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(len(display_df) * 35 + 38, 600),
    )

    # Ticker selector for deep dive
    st.markdown("---")
    tickers = filtered_df["ticker"].tolist()
    if tickers:
        return st.selectbox(
            "Select a ticker for Deep Dive analysis:",
            options=tickers,
            index=0,
            key="ranked_ticker_select",
        )
    return None


# ---------------------------------------------------------------------------
# Tab 2: Deep Dive
# ---------------------------------------------------------------------------


def _render_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
) -> None:
    """Render a single metric card using st.metric.

    Parameters
    ----------
    label:
        Metric label.
    value:
        Formatted display value.
    delta:
        Optional delta value string.
    """
    st.metric(label=label, value=value, delta=delta)


def _render_key_metrics_cards(ctx: dict[str, Any]) -> None:
    """Display the five top-level metric cards for a deep dive.

    Cards: Score, IV, Price, MoS, Recommendation.

    Parameters
    ----------
    ctx:
        Report context dict from ``build_report_context``.
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        _render_metric_card("Score", _fmt_num(ctx.get("composite_score"), 1))
    with col2:
        _render_metric_card("Intrinsic Value", _fmt_usd(ctx.get("iv_weighted")))
    with col3:
        _render_metric_card("Price", _fmt_usd(ctx.get("current_price_usd")))
    with col4:
        _render_metric_card("MoS", _fmt_pct(ctx.get("margin_of_safety_pct", 0.0)))
    with col5:
        rec = ctx.get("recommendation", "—")
        color = _RECOMMENDATION_COLORS.get(rec, "#6b7280")
        st.markdown(
            f"**Recommendation**<br>"
            f'<span style="color:{color}; font-size:1.8rem; '
            f'font-weight:700;">{rec}</span>',
            unsafe_allow_html=True,
        )


def _render_financial_trends(ctx: dict[str, Any]) -> None:
    """Render the 2×2 grid of 10-year financial trend charts.

    Charts: ROE, Gross Margin, EPS, Owner Earnings (per spec).

    Parameters
    ----------
    ctx:
        Report context dict.
    """
    st.markdown("### 10-Year Financial Trends")
    annual_income = ctx.get("annual_income", [])

    if not annual_income:
        st.info("No annual income data available for charting.")
        return

    inc_df = pd.DataFrame(annual_income).set_index("fiscal_year")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        if "roe" in inc_df.columns:
            st.markdown("**ROE (Return on Equity)**")
            st.line_chart(inc_df[["roe"]].rename(columns={"roe": "ROE"}))
    with r1c2:
        if "gross_margin" in inc_df.columns:
            st.markdown("**Gross Margin**")
            st.line_chart(inc_df[["gross_margin"]].rename(
                columns={"gross_margin": "Gross Margin"},
            ))

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        if "eps_diluted" in inc_df.columns:
            st.markdown("**EPS (Diluted)**")
            st.line_chart(inc_df[["eps_diluted"]].rename(
                columns={"eps_diluted": "EPS"},
            ))
    # 4th chart: Owner Earnings (from cash flow data, per spec)
    with r2c2:
        _render_owner_earnings_chart(ctx)


def _render_owner_earnings_chart(ctx: dict[str, Any]) -> None:
    """Render the Owner Earnings line chart (4th trend panel).

    Falls back to Free Cash Flow if owner_earnings is unavailable.

    Parameters
    ----------
    ctx:
        Report context dict.
    """
    annual_cf = ctx.get("annual_cashflow", [])
    if not annual_cf:
        st.info("No cash flow data available for charting.")
        return

    cf_df = pd.DataFrame(annual_cf)
    if "fiscal_year" in cf_df.columns:
        cf_df = cf_df.set_index("fiscal_year")

    if "owner_earnings" in cf_df.columns:
        st.markdown("**Owner Earnings**")
        st.line_chart(cf_df[["owner_earnings"]].rename(
            columns={"owner_earnings": "Owner Earnings"},
        ))
    elif "free_cash_flow" in cf_df.columns:
        st.markdown("**Free Cash Flow**")
        st.line_chart(cf_df[["free_cash_flow"]].rename(
            columns={"free_cash_flow": "Free Cash Flow"},
        ))
    else:
        st.info("No owner earnings data available.")


def _render_valuation_scenarios(ctx: dict[str, Any]) -> None:
    """Render the three-scenario bar chart and summary table.

    Parameters
    ----------
    ctx:
        Report context dict.
    """
    st.markdown("### Valuation — Three Scenarios")
    vals = {k: _safe_float(ctx.get(k)) for k in
            ("iv_bear", "iv_base", "iv_bull", "iv_weighted", "current_price_usd")}

    scenario_df = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull", "Weighted", "Current Price"],
        "Value (USD)": [vals["iv_bear"], vals["iv_base"], vals["iv_bull"],
                        vals["iv_weighted"], vals["current_price_usd"]],
    }).set_index("Scenario")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.bar_chart(scenario_df)
    with c2:
        st.markdown("| Scenario | Value |\n|---|---|")
        for label, key in [("Bear", "iv_bear"), ("Base", "iv_base"),
                           ("Bull", "iv_bull")]:
            st.markdown(f"| {label} | {_fmt_usd(vals[key])} |")
        st.markdown(f"| **Weighted** | **{_fmt_usd(vals['iv_weighted'])}** |")
        st.markdown(f"| Current Price | {_fmt_usd(vals['current_price_usd'])} |")
        st.markdown(
            f"| **MoS** | **{_fmt_pct(ctx.get('margin_of_safety_pct'))}** |",
        )


def _render_investment_strategy(ctx: dict[str, Any]) -> None:
    """Render the investment strategy section (entries, sizing, account).

    Parameters
    ----------
    ctx:
        Report context dict.
    """
    st.markdown("### Investment Strategy")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Entry Target:** {_fmt_usd(ctx.get('buy_below_moderate'))}")
        st.markdown(
            f"**Conservative Entry:** "
            f"{_fmt_usd(ctx.get('buy_below_conservative'))}",
        )
    with c2:
        st.markdown(
            f"**Time Horizon:** {ctx.get('time_horizon_years', '—')}+ years",
        )
        st.markdown(f"**Account:** {ctx.get('account_recommendation', '—')}")
    with c3:
        st.markdown(f"**Confidence:** {ctx.get('confidence_level', '—')}")
        st.markdown(
            f"**Position Sizing:** {ctx.get('position_sizing_guidance', '—')}",
        )


def _render_supplementary_sections(ctx: dict[str, Any]) -> None:
    """Render assumption log, bear case, and investment strategy.

    Parameters
    ----------
    ctx:
        Report context dict.
    """
    # Assumption Log
    st.markdown("### Assumption Log")
    assumption_log = ctx.get("assumption_log", [])
    if assumption_log:
        st.dataframe(
            pd.DataFrame(assumption_log),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No assumptions logged.")

    # Bear Case
    bear_case = ctx.get("bear_case_arguments", [])
    if bear_case:
        st.markdown("---")
        st.markdown("### Devil's Advocate — Bear Case")
        for arg in bear_case:
            with st.expander(arg.get("title", "Argument")):
                st.write(arg.get("body", ""))

    st.markdown("---")
    _render_investment_strategy(ctx)


def _render_tab_deep_dive(selected_ticker: str | None) -> None:
    """Render the deep-dive analysis for one ticker.

    Parameters
    ----------
    selected_ticker:
        Ticker symbol to analyse, or ``None`` to show placeholder.
    """
    if selected_ticker is None:
        st.info("Select a ticker from the Ranked Results tab.")
        return

    st.subheader(f"Deep Dive: {selected_ticker}")

    with st.spinner(f"Loading analysis for {selected_ticker}..."):
        ctx = _load_report_context(selected_ticker)

    if ctx is None:
        st.error(f"Could not load analysis for {selected_ticker}.")
        return

    _render_key_metrics_cards(ctx)
    st.markdown("---")
    _render_financial_trends(ctx)
    st.markdown("---")
    _render_valuation_scenarios(ctx)
    st.markdown("---")

    # Sensitivity Analysis
    sensitivity = ctx.get("sensitivity_data")
    if sensitivity:
        st.markdown("### Sensitivity Analysis")
        _render_sensitivity(sensitivity)
        st.markdown("---")

    # Sell Signal Monitor
    st.markdown("### Sell Signal Monitor")
    triggers = ctx.get("sell_triggers", [])
    if triggers:
        _render_sell_signals(triggers)
    else:
        st.info("No sell signal data available.")
    st.markdown("---")

    _render_supplementary_sections(ctx)


def _render_sensitivity(sensitivity: dict[str, Any]) -> None:
    """Render sensitivity analysis tables.

    Parameters
    ----------
    sensitivity:
        Sensitivity data dict from ``compute_sensitivity_table``.
    """
    eps_sens = sensitivity.get("eps_sensitivity", [])
    pe_sens = sensitivity.get("pe_sensitivity", [])
    discount_sens = sensitivity.get("discount_sensitivity", [])

    if eps_sens or pe_sens:
        col1, col2 = st.columns(2)
        with col1:
            if eps_sens:
                st.markdown("**EPS Growth Sensitivity**")
                eps_df = pd.DataFrame(eps_sens)
                st.dataframe(eps_df, use_container_width=True, hide_index=True)
        with col2:
            if pe_sens:
                st.markdown("**Terminal P/E Sensitivity**")
                pe_df = pd.DataFrame(pe_sens)
                st.dataframe(pe_df, use_container_width=True, hide_index=True)

    if discount_sens:
        st.markdown("**Discount Rate Sensitivity**")
        disc_df = pd.DataFrame(discount_sens)
        st.dataframe(disc_df, use_container_width=True, hide_index=True)


def _render_sell_signals(triggers: list[dict[str, Any]]) -> None:
    """Render sell signal status indicators.

    Parameters
    ----------
    triggers:
        List of sell-signal dicts with keys ``signal``, ``status``,
        ``current_value``, ``threshold``, ``description``.
    """
    cols = st.columns(min(len(triggers), 3))
    for i, trigger in enumerate(triggers):
        with cols[i % len(cols)]:
            status = trigger.get("status", "OK")
            signal = trigger.get("signal", "Unknown")

            if status == "TRIGGERED":
                icon = "🔴"
            elif status == "WARNING":
                icon = "🟡"
            else:
                icon = "🟢"

            st.markdown(
                f"{icon} **{signal}** — {status}",
            )
            desc = trigger.get("description", "")
            if desc:
                st.caption(desc)


# ---------------------------------------------------------------------------
# Tab 3: Screener Summary
# ---------------------------------------------------------------------------


def _load_run_log() -> dict[str, Any]:
    """Load the latest run_log.json for pipeline funnel statistics.

    Returns
    -------
    dict[str, Any]
        Run log contents, or empty dict if the file is unavailable.
    """
    try:
        report_dir_str = str(get_threshold("output.report_dir"))
    except Exception:
        report_dir_str = "data/reports"

    log_path = _PROJECT_ROOT / report_dir_str / "run_log.json"
    if log_path.exists():
        try:
            return json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _render_pipeline_funnel(ranked_df: pd.DataFrame) -> None:
    """Render the pipeline funnel metrics row.

    Reads run_log.json and data_quality_log for counts.

    Parameters
    ----------
    ranked_df:
        Full ranked DataFrame (used as fallback shortlist count).
    """
    universe = _load_table("universe")
    quality = _load_table("data_quality_log")
    run_log = _load_run_log()

    total_universe = run_log.get("universe_size", len(universe))
    tier1_survivors = run_log.get("tier1_survivors", 0)
    shortlisted = run_log.get("shortlisted", len(ranked_df))
    reports_generated = run_log.get("reports_generated", 0)

    dropped = 0
    if not quality.empty and "drop" in quality.columns:
        survivors = int((quality["drop"] == False).sum())  # noqa: E712
        dropped = int((quality["drop"] == True).sum())  # noqa: E712
        if tier1_survivors == 0 and survivors > 0:
            tier1_survivors = survivors

    st.markdown("#### Pipeline Funnel")
    cols = st.columns(5)
    with cols[0]:
        st.metric("Universe", total_universe)
    with cols[1]:
        st.metric("After Exclusions",
                   total_universe - dropped if dropped else total_universe)
    with cols[2]:
        st.metric("Tier 1 Survivors", tier1_survivors)
    with cols[3]:
        st.metric("Shortlisted", shortlisted)
    with cols[4]:
        st.metric("Reports Generated", reports_generated)


def _render_sector_pie(ranked_df: pd.DataFrame) -> None:
    """Render a matplotlib pie chart of sector distribution.

    Parameters
    ----------
    ranked_df:
        Ranked securities DataFrame with a ``sector`` column.
    """
    st.markdown("#### Sector Distribution")
    if "sector" not in ranked_df.columns or ranked_df.empty:
        st.info("No sector data available.")
        return

    counts = ranked_df["sector"].dropna().value_counts()
    if counts.empty:
        st.info("No sector data available.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 9})
    ax.set_title("Sectors Among Ranked Securities")
    st.pyplot(fig)
    plt.close(fig)


def _render_score_histogram(ranked_df: pd.DataFrame) -> None:
    """Render a composite-score histogram.

    Parameters
    ----------
    ranked_df:
        Ranked securities DataFrame.
    """
    if "composite_score" not in ranked_df.columns:
        return

    st.markdown("---")
    st.markdown("#### Composite Score Distribution")
    scores = ranked_df["composite_score"].dropna()
    if scores.empty:
        st.info("No score data available for histogram.")
        return

    import numpy as np

    hist_vals, edges = np.histogram(scores, bins=20, range=(0, 100))
    labels = [f"{edges[i]:.0f}–{edges[i + 1]:.0f}" for i in range(len(hist_vals))]
    hist_df = pd.DataFrame({"Score Range": labels, "Count": hist_vals})
    st.bar_chart(hist_df.set_index("Score Range"))


def _render_tab_summary(ranked_df: pd.DataFrame) -> None:
    """Render pipeline statistics and distribution charts.

    Parameters
    ----------
    ranked_df:
        Full (unfiltered) ranked DataFrame.
    """
    st.subheader("Screener Summary")

    _render_pipeline_funnel(ranked_df)
    st.markdown("---")

    # Distribution charts — sector (pie) and exchange (bar)
    c1, c2 = st.columns(2)
    with c1:
        _render_sector_pie(ranked_df)
    with c2:
        st.markdown("#### Exchange Distribution")
        if "exchange" in ranked_df.columns and not ranked_df.empty:
            exc = ranked_df["exchange"].dropna().value_counts().reset_index()
            exc.columns = ["Exchange", "Count"]
            if not exc.empty:
                st.bar_chart(exc.set_index("Exchange"))
            else:
                st.info("No exchange data available.")
        else:
            st.info("No exchange data available.")

    _render_score_histogram(ranked_df)


# ---------------------------------------------------------------------------
# Tab 4: Macro Context
# ---------------------------------------------------------------------------


def _extract_macro_dict(macro_df: pd.DataFrame) -> tuple[dict[str, float], str]:
    """Parse the macro_data table into a key→value dict.

    Parameters
    ----------
    macro_df:
        Raw macro_data DataFrame with columns ``key``, ``value``,
        ``as_of_date``.

    Returns
    -------
    tuple[dict[str, float], str]
        ``(macro_dict, as_of_date)`` — the as-of date string for display.
    """
    macro_dict: dict[str, float] = {}
    as_of_date = "—"
    for _, row in macro_df.iterrows():
        macro_dict[str(row.get("key", ""))] = _safe_float(row.get("value"))
        date_val = row.get("as_of_date", "")
        if date_val:
            as_of_date = str(date_val)
    return macro_dict, as_of_date


def _render_bond_yields(macro_dict: dict[str, float]) -> None:
    """Display bond-yield and exchange-rate metric cards.

    Parameters
    ----------
    macro_dict:
        Key-value map of macro indicators.
    """
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("US 10-Year Treasury",
                   _fmt_pct(macro_dict.get("us_treasury_10yr", 0.0)))
    with c2:
        goc = macro_dict.get("goc_bond_10yr", 0.0)
        st.metric("GoC 10-Year Bond", _fmt_pct(goc) if goc > 0 else "—")
    with c3:
        usd_cad = macro_dict.get("usd_cad_rate", 0.0)
        st.metric("USD/CAD Rate",
                   _fmt_num(usd_cad, 4) if usd_cad > 0 else "—")


def _render_yield_verdict(spread: float) -> None:
    """Display the Attractive / Moderate / Unattractive yield verdict.

    Thresholds from ``reports.yield_verdict`` config.

    Parameters
    ----------
    spread:
        Earnings yield minus bond yield (decimal).
    """
    try:
        attractive_min = float(get_threshold(
            "reports.yield_verdict.attractive_min_spread",
        ))
    except Exception:
        attractive_min = 0.04
    try:
        moderate_min = float(get_threshold(
            "reports.yield_verdict.moderate_min_spread",
        ))
    except Exception:
        moderate_min = 0.02

    pct = f"{spread * 100:.1f}%"
    if spread > attractive_min:
        st.success(f"**Attractive:** {pct} spread over bonds.")
    elif spread > moderate_min:
        st.warning(f"**Moderate:** {pct} spread. Selectivity is key.")
    else:
        st.error(f"**Unattractive:** {pct} spread. Consider fixed income.")


def _render_market_attractiveness(treasury: float) -> None:
    """Render the earnings-yield-vs-bonds assessment.

    Parameters
    ----------
    treasury:
        US 10-year treasury yield as a decimal.
    """
    if treasury <= 0:
        st.info("Treasury yield data unavailable for assessment.")
        return

    market = _load_table("market_data")
    if market.empty or "pe_ratio_trailing" not in market.columns:
        st.info("Market data unavailable for yield calculation.")
        return

    valid_pe = market["pe_ratio_trailing"].dropna()
    valid_pe = valid_pe[valid_pe > 0]
    if valid_pe.empty:
        st.info("Insufficient P/E data for yield comparison.")
        return

    median_pe = float(valid_pe.median())
    avg_ey = 1.0 / median_pe if median_pe > 0 else 0.0
    spread = avg_ey - treasury

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Median Earnings Yield (screened)", _fmt_pct(avg_ey))
    with c2:
        st.metric("Bond Yield", _fmt_pct(treasury))
    with c3:
        st.metric("Spread (EY − Bond)", _fmt_pct(spread))
    st.markdown("---")

    _render_yield_verdict(spread)


def _render_tab_macro() -> None:
    """Render macro context: bond yields, earnings yield spread."""
    st.subheader("Macro Context")

    macro_df = _load_table("macro_data")
    if macro_df.empty:
        st.warning(
            "No macro data available. Run the pipeline to fetch "
            "treasury yields and exchange rates.",
        )
        return

    macro_dict, as_of_date = _extract_macro_dict(macro_df)
    st.markdown(f"*Data as of: {as_of_date}*")
    st.markdown("---")

    _render_bond_yields(macro_dict)
    st.markdown("---")

    st.markdown("### Market Attractiveness Assessment")
    _render_market_attractiveness(macro_dict.get("us_treasury_10yr", 0.0))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def _check_data_ready() -> pd.DataFrame | None:
    """Check prerequisites and return ranked DataFrame or ``None``.

    Shows user-facing error/warning messages if the database is
    missing or empty.

    Returns
    -------
    pd.DataFrame | None
        Ranked table, or ``None`` if the dashboard cannot proceed.
    """
    if not _db_exists():
        st.error(
            "**Database not found.**\n\n"
            f"Expected: `{_DB_PATH}`\n\n"
            "Run the pipeline first:\n\n"
            "```bash\n"
            "python -m output.pipeline_runner --mode reports\n"
            "```",
        )
        return None

    ranked_df = _build_ranked_table()
    if ranked_df.empty:
        st.warning(
            "The database exists but contains no ranked securities. "
            "Run the full pipeline first.",
        )
        return None
    return ranked_df


def main() -> None:
    """Render the full Streamlit dashboard."""
    st.markdown(
        '<h1 style="margin-bottom:0;">📊 Buffett Screener</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Warren Buffett-style equity screening and valuation dashboard")

    ranked_df = _check_data_ready()
    if ranked_df is None:
        return

    filtered_df = _render_sidebar(ranked_df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Ranked Results", "🔍 Deep Dive",
        "📊 Screener Summary", "🌐 Macro Context",
    ])

    with tab1:
        _render_tab_ranked(filtered_df)
    with tab2:
        tickers = filtered_df["ticker"].tolist()
        ticker = st.selectbox(
            "Select ticker for analysis:",
            options=tickers if tickers else [""],
            index=0, key="deep_dive_ticker_select",
        )
        _render_tab_deep_dive(ticker if ticker else None)
    with tab3:
        _render_tab_summary(ranked_df)
    with tab4:
        _render_tab_macro()


if __name__ == "__main__":
    main()

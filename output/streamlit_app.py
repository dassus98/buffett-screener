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

Config dependencies:
    - ``output.report_dir`` — report directory for pre-rendered Markdown.
    - ``output.shortlist_size`` — default top-N display count.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import pandas as pd
import streamlit as st

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
# Build the ranked results table from raw DuckDB data
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def _build_ranked_table() -> pd.DataFrame:
    """Assemble the ranked securities table for Tab 1.

    Joins universe, market_data, composite_scores, and metrics_summary
    tables to produce a single display-ready DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: Rank, Ticker, Company, Exchange, Sector, Composite Score,
        and available valuation fields.
    """
    universe = _load_table("universe")
    market = _load_table("market_data")
    quality = _load_table("data_quality_log")

    # Try to load composite scores — may not exist if pipeline hasn't run
    try:
        from data_acquisition.store import get_connection

        conn = get_connection()
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchdf()
        available_tables = set(tables["table_name"].tolist())
    except Exception:
        available_tables = set()

    # Load metrics summary if available
    if "buffett_metrics_summary" in available_tables:
        metrics = _load_table("buffett_metrics_summary")
    else:
        metrics = pd.DataFrame()

    if "composite_scores" in available_tables:
        scores = _load_table("composite_scores")
    else:
        scores = pd.DataFrame()

    # Start with universe
    if universe.empty:
        return pd.DataFrame()

    df = universe[["ticker", "company_name", "exchange", "sector"]].copy()

    # Join market data
    if not market.empty and "ticker" in market.columns:
        market_cols = ["ticker", "current_price_usd", "pe_ratio_trailing",
                       "dividend_yield"]
        available_cols = [c for c in market_cols if c in market.columns]
        df = df.merge(market[available_cols], on="ticker", how="left")

    # Join composite scores
    if not scores.empty and "ticker" in scores.columns:
        score_cols = [c for c in scores.columns if c in
                      {"ticker", "composite_score"}]
        df = df.merge(scores[score_cols], on="ticker", how="left")
    elif not metrics.empty and "composite_score" in metrics.columns:
        df = df.merge(
            metrics[["ticker", "composite_score"]],
            on="ticker",
            how="left",
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

    # Filter to survivors (non-dropped tickers)
    if not quality.empty and "ticker" in quality.columns:
        survivors = quality[quality["drop"] == False]["ticker"].tolist()  # noqa: E712
        if survivors:
            df = df[df["ticker"].isin(survivors)]

    # Sort by composite score descending
    if "composite_score" in df.columns:
        df = df.sort_values("composite_score", ascending=False)
    df = df.reset_index(drop=True)

    # Add rank column
    df.insert(0, "rank", range(1, len(df) + 1))

    return df


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------


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

    # Score range slider
    if "composite_score" in filtered.columns:
        score_min, score_max = st.sidebar.slider(
            "Composite Score Range",
            min_value=0,
            max_value=100,
            value=(0, 100),
            step=1,
        )
        filtered = filtered[
            filtered["composite_score"].between(score_min, score_max)
            | filtered["composite_score"].isna()
        ]

    # Margin of safety slider
    if "margin_of_safety" in filtered.columns:
        mos_min, mos_max = st.sidebar.slider(
            "Margin of Safety (%)",
            min_value=-50,
            max_value=100,
            value=(-50, 100),
            step=1,
        )
        mos_min_dec = mos_min / 100.0
        mos_max_dec = mos_max / 100.0
        filtered = filtered[
            filtered["margin_of_safety"].between(mos_min_dec, mos_max_dec)
            | filtered["margin_of_safety"].isna()
        ]

    # Exchange multiselect
    if "exchange" in filtered.columns:
        exchanges = sorted(filtered["exchange"].dropna().unique().tolist())
        if exchanges:
            selected_exchanges = st.sidebar.multiselect(
                "Exchange",
                options=exchanges,
                default=exchanges,
            )
            if selected_exchanges:
                filtered = filtered[
                    filtered["exchange"].isin(selected_exchanges)
                ]

    # Sector multiselect
    if "sector" in filtered.columns:
        sectors = sorted(filtered["sector"].dropna().unique().tolist())
        if sectors:
            selected_sectors = st.sidebar.multiselect(
                "Sector",
                options=sectors,
                default=sectors,
            )
            if selected_sectors:
                filtered = filtered[
                    filtered["sector"].isin(selected_sectors)
                ]

    # Show top N
    top_n = st.sidebar.slider(
        "Show Top N",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
    )
    filtered = filtered.head(top_n)

    # Re-rank after filtering
    filtered = filtered.reset_index(drop=True)
    filtered["rank"] = range(1, len(filtered) + 1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Showing {len(filtered)}** of {len(ranked_df)} securities"
    )

    return filtered


# ---------------------------------------------------------------------------
# Tab 1: Ranked Results
# ---------------------------------------------------------------------------


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

    # Prepare display columns
    display_cols = ["rank", "ticker", "company_name", "exchange", "sector"]
    if "composite_score" in filtered_df.columns:
        display_cols.append("composite_score")
    if "weighted_iv" in filtered_df.columns:
        display_cols.append("weighted_iv")
    if "current_price_usd" in filtered_df.columns:
        display_cols.append("current_price_usd")
    if "margin_of_safety" in filtered_df.columns:
        display_cols.append("margin_of_safety")

    available_cols = [c for c in display_cols if c in filtered_df.columns]
    display_df = filtered_df[available_cols].copy()

    # Rename for display
    rename_map = {
        "rank": "Rank",
        "ticker": "Ticker",
        "company_name": "Company",
        "exchange": "Exchange",
        "sector": "Sector",
        "composite_score": "Score",
        "weighted_iv": "Weighted IV",
        "current_price_usd": "Price (USD)",
        "margin_of_safety": "MoS %",
    }
    display_df = display_df.rename(
        columns={k: v for k, v in rename_map.items() if k in display_df.columns}
    )

    # Format MoS as percentage for display
    if "MoS %" in display_df.columns:
        display_df["MoS %"] = display_df["MoS %"].apply(
            lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "—"
        )

    # Format price
    if "Price (USD)" in display_df.columns:
        display_df["Price (USD)"] = display_df["Price (USD)"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
        )

    # Format Weighted IV
    if "Weighted IV" in display_df.columns:
        display_df["Weighted IV"] = display_df["Weighted IV"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
        )

    # Format Score
    if "Score" in display_df.columns:
        display_df["Score"] = display_df["Score"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(len(display_df) * 35 + 38, 600),
    )

    # Ticker selector for deep dive
    st.markdown("---")
    tickers = filtered_df["ticker"].tolist()
    if tickers:
        selected = st.selectbox(
            "Select a ticker for Deep Dive analysis:",
            options=tickers,
            index=0,
            key="ranked_ticker_select",
        )
        return selected
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


def _render_tab_deep_dive(selected_ticker: str | None) -> None:
    """Render the deep-dive analysis for one ticker.

    Parameters
    ----------
    selected_ticker:
        Ticker symbol to analyse, or ``None`` to show placeholder.
    """
    if selected_ticker is None:
        st.info("Select a ticker from the Ranked Results tab to view "
                "its deep-dive analysis.")
        return

    st.subheader(f"Deep Dive: {selected_ticker}")

    with st.spinner(f"Loading analysis for {selected_ticker}..."):
        ctx = _load_report_context(selected_ticker)

    if ctx is None:
        st.error(
            f"Could not load analysis for {selected_ticker}. "
            "Ensure the pipeline has been run and data is available."
        )
        return

    # --- Key metrics cards ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        _render_metric_card(
            "Composite Score",
            _fmt_num(ctx.get("composite_score"), 1),
        )
    with col2:
        _render_metric_card(
            "Intrinsic Value",
            _fmt_usd(ctx.get("iv_weighted")),
        )
    with col3:
        _render_metric_card(
            "Current Price",
            _fmt_usd(ctx.get("current_price_usd")),
        )
    with col4:
        mos = ctx.get("margin_of_safety_pct", 0.0)
        _render_metric_card(
            "Margin of Safety",
            _fmt_pct(mos),
        )
    with col5:
        rec = ctx.get("recommendation", "—")
        color = _RECOMMENDATION_COLORS.get(rec, "#6b7280")
        st.markdown(
            f"**Recommendation**<br>"
            f'<span style="color:{color}; font-size:1.8rem; '
            f'font-weight:700;">{rec}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- 10-Year Financial Trends (2×2 grid) ---
    st.markdown("### 10-Year Financial Trends")

    annual_income = ctx.get("annual_income", [])
    annual_balance = ctx.get("annual_balance", [])

    if annual_income:
        inc_df = pd.DataFrame(annual_income)
        inc_df = inc_df.set_index("fiscal_year")

        chart_row1_col1, chart_row1_col2 = st.columns(2)

        with chart_row1_col1:
            if "roe" in inc_df.columns:
                st.markdown("**ROE (Return on Equity)**")
                roe_data = inc_df[["roe"]].rename(columns={"roe": "ROE"})
                st.line_chart(roe_data)

        with chart_row1_col2:
            if "gross_margin" in inc_df.columns:
                st.markdown("**Gross Margin**")
                gm_data = inc_df[["gross_margin"]].rename(
                    columns={"gross_margin": "Gross Margin"},
                )
                st.line_chart(gm_data)

        chart_row2_col1, chart_row2_col2 = st.columns(2)

        with chart_row2_col1:
            if "eps_diluted" in inc_df.columns:
                st.markdown("**EPS (Diluted)**")
                eps_data = inc_df[["eps_diluted"]].rename(
                    columns={"eps_diluted": "EPS"},
                )
                st.line_chart(eps_data)

        with chart_row2_col2:
            if "net_margin" in inc_df.columns:
                st.markdown("**Net Margin**")
                nm_data = inc_df[["net_margin"]].rename(
                    columns={"net_margin": "Net Margin"},
                )
                st.line_chart(nm_data)
    else:
        st.info("No annual income data available for charting.")

    st.markdown("---")

    # --- Valuation Scenario Comparison ---
    st.markdown("### Valuation — Three Scenarios")

    iv_bear = _safe_float(ctx.get("iv_bear"))
    iv_base = _safe_float(ctx.get("iv_base"))
    iv_bull = _safe_float(ctx.get("iv_bull"))
    iv_weighted = _safe_float(ctx.get("iv_weighted"))
    current_price = _safe_float(ctx.get("current_price_usd"))

    scenario_data = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull", "Weighted", "Current Price"],
        "Value (USD)": [iv_bear, iv_base, iv_bull, iv_weighted, current_price],
    })
    scenario_data = scenario_data.set_index("Scenario")

    val_col1, val_col2 = st.columns([2, 1])
    with val_col1:
        st.bar_chart(scenario_data)
    with val_col2:
        st.markdown("| Scenario | Value |")
        st.markdown("|---|---|")
        st.markdown(f"| Bear | {_fmt_usd(iv_bear)} |")
        st.markdown(f"| Base | {_fmt_usd(iv_base)} |")
        st.markdown(f"| Bull | {_fmt_usd(iv_bull)} |")
        st.markdown(f"| **Weighted** | **{_fmt_usd(iv_weighted)}** |")
        st.markdown(f"| Current Price | {_fmt_usd(current_price)} |")
        st.markdown(
            f"| **Margin of Safety** | "
            f"**{_fmt_pct(ctx.get('margin_of_safety_pct'))}** |"
        )

    st.markdown("---")

    # --- Sensitivity Analysis ---
    sensitivity = ctx.get("sensitivity_data")
    if sensitivity:
        st.markdown("### Sensitivity Analysis")
        _render_sensitivity(sensitivity)
        st.markdown("---")

    # --- Sell Signal Status ---
    st.markdown("### Sell Signal Monitor")
    sell_triggers = ctx.get("sell_triggers", [])
    if sell_triggers:
        _render_sell_signals(sell_triggers)
    else:
        st.info("No sell signal data available.")

    st.markdown("---")

    # --- Assumption Log ---
    st.markdown("### Assumption Log")
    assumption_log = ctx.get("assumption_log", [])
    if assumption_log:
        log_df = pd.DataFrame(assumption_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No assumptions logged.")

    # --- Bear Case ---
    bear_case = ctx.get("bear_case_arguments", [])
    if bear_case:
        st.markdown("---")
        st.markdown("### Devil's Advocate — Bear Case")
        for arg in bear_case:
            with st.expander(arg.get("title", "Argument")):
                st.write(arg.get("body", ""))

    # --- Investment Strategy ---
    st.markdown("---")
    st.markdown("### Investment Strategy")

    strat_col1, strat_col2, strat_col3 = st.columns(3)
    with strat_col1:
        st.markdown(f"**Entry Target:** {_fmt_usd(ctx.get('buy_below_moderate'))}")
        st.markdown(
            f"**Conservative Entry:** "
            f"{_fmt_usd(ctx.get('buy_below_conservative'))}"
        )
    with strat_col2:
        st.markdown(
            f"**Time Horizon:** {ctx.get('time_horizon_years', '—')}+ years"
        )
        st.markdown(
            f"**Account:** {ctx.get('account_recommendation', '—')}"
        )
    with strat_col3:
        st.markdown(
            f"**Confidence:** {ctx.get('confidence_level', '—')}"
        )
        sizing = ctx.get("position_sizing_guidance", "—")
        st.markdown(f"**Position Sizing:** {sizing}")


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


def _render_tab_summary(ranked_df: pd.DataFrame) -> None:
    """Render pipeline statistics and distribution charts.

    Parameters
    ----------
    ranked_df:
        Full (unfiltered) ranked DataFrame.
    """
    st.subheader("Screener Summary")

    universe = _load_table("universe")
    quality = _load_table("data_quality_log")

    # Pipeline statistics
    total_universe = len(universe)
    survivors = 0
    dropped = 0
    if not quality.empty and "drop" in quality.columns:
        survivors = int((quality["drop"] == False).sum())  # noqa: E712
        dropped = int((quality["drop"] == True).sum())  # noqa: E712

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Total Universe", total_universe)
    with stat_col2:
        st.metric("Quality Survivors", survivors)
    with stat_col3:
        st.metric("Dropped", dropped)
    with stat_col4:
        st.metric("Ranked", len(ranked_df))

    st.markdown("---")

    # Distribution charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Sector Distribution")
        if "sector" in ranked_df.columns and not ranked_df.empty:
            sector_counts = (
                ranked_df["sector"]
                .dropna()
                .value_counts()
                .reset_index()
            )
            sector_counts.columns = ["Sector", "Count"]
            if not sector_counts.empty:
                sector_chart = sector_counts.set_index("Sector")
                st.bar_chart(sector_chart)
            else:
                st.info("No sector data available.")
        else:
            st.info("No sector data available.")

    with chart_col2:
        st.markdown("#### Exchange Distribution")
        if "exchange" in ranked_df.columns and not ranked_df.empty:
            exchange_counts = (
                ranked_df["exchange"]
                .dropna()
                .value_counts()
                .reset_index()
            )
            exchange_counts.columns = ["Exchange", "Count"]
            if not exchange_counts.empty:
                exchange_chart = exchange_counts.set_index("Exchange")
                st.bar_chart(exchange_chart)
            else:
                st.info("No exchange data available.")
        else:
            st.info("No exchange data available.")

    # Score distribution histogram
    if "composite_score" in ranked_df.columns:
        st.markdown("---")
        st.markdown("#### Composite Score Distribution")
        score_vals = ranked_df["composite_score"].dropna()
        if not score_vals.empty:
            import numpy as np

            hist_values, bin_edges = np.histogram(
                score_vals, bins=20, range=(0, 100),
            )
            bin_labels = [
                f"{bin_edges[i]:.0f}–{bin_edges[i + 1]:.0f}"
                for i in range(len(hist_values))
            ]
            hist_df = pd.DataFrame(
                {"Score Range": bin_labels, "Count": hist_values},
            )
            hist_df = hist_df.set_index("Score Range")
            st.bar_chart(hist_df)
        else:
            st.info("No score data available for histogram.")


# ---------------------------------------------------------------------------
# Tab 4: Macro Context
# ---------------------------------------------------------------------------


def _render_tab_macro() -> None:
    """Render macro context: bond yields, earnings yield spread."""
    st.subheader("Macro Context")

    macro_df = _load_table("macro_data")

    if macro_df.empty:
        st.warning(
            "No macro data available. Run the pipeline to fetch "
            "treasury yields and exchange rates."
        )
        return

    # Extract macro values
    macro_dict: dict[str, float] = {}
    as_of_date = "—"
    for _, row in macro_df.iterrows():
        key = str(row.get("key", ""))
        val = _safe_float(row.get("value"))
        macro_dict[key] = val
        date_val = row.get("as_of_date", "")
        if date_val:
            as_of_date = str(date_val)

    st.markdown(f"*Data as of: {as_of_date}*")
    st.markdown("---")

    # Bond yield display
    macro_col1, macro_col2, macro_col3 = st.columns(3)
    with macro_col1:
        treasury_10yr = macro_dict.get("us_treasury_10yr", 0.0)
        st.metric(
            "US 10-Year Treasury",
            _fmt_pct(treasury_10yr),
        )
    with macro_col2:
        goc_10yr = macro_dict.get("goc_bond_10yr", 0.0)
        if goc_10yr > 0:
            st.metric("GoC 10-Year Bond", _fmt_pct(goc_10yr))
        else:
            st.metric("GoC 10-Year Bond", "—")
    with macro_col3:
        usd_cad = macro_dict.get("usd_cad_rate", 0.0)
        if usd_cad > 0:
            st.metric("USD/CAD Rate", _fmt_num(usd_cad, 4))
        else:
            st.metric("USD/CAD Rate", "—")

    st.markdown("---")

    # Market attractiveness assessment
    st.markdown("### Market Attractiveness Assessment")

    treasury = macro_dict.get("us_treasury_10yr", 0.0)
    if treasury > 0:
        # Compare average earnings yield of shortlisted stocks to bond yield
        ranked = _build_ranked_table()
        if not ranked.empty and "current_price_usd" in ranked.columns:
            market = _load_table("market_data")
            if not market.empty and "pe_ratio_trailing" in market.columns:
                pe_series = market["pe_ratio_trailing"].dropna()
                valid_pe = pe_series[pe_series > 0]
                if not valid_pe.empty:
                    median_pe = float(valid_pe.median())
                    avg_ey = 1.0 / median_pe if median_pe > 0 else 0.0
                    spread = avg_ey - treasury

                    ey_col1, ey_col2, ey_col3 = st.columns(3)
                    with ey_col1:
                        st.metric(
                            "Median Earnings Yield (screened)",
                            _fmt_pct(avg_ey),
                        )
                    with ey_col2:
                        st.metric(
                            "Bond Yield",
                            _fmt_pct(treasury),
                        )
                    with ey_col3:
                        st.metric(
                            "Spread (EY − Bond)",
                            _fmt_pct(spread),
                        )

                    st.markdown("---")

                    if spread > 0.04:
                        st.success(
                            "**Attractive:** Equity earnings yield offers "
                            f"a {spread * 100:.1f}% spread over bonds. "
                            "Equities appear attractively priced relative "
                            "to fixed income."
                        )
                    elif spread > 0.02:
                        st.warning(
                            "**Moderate:** Equity earnings yield offers "
                            f"a {spread * 100:.1f}% spread over bonds. "
                            "Equities offer a slim premium. Selectivity "
                            "is key."
                        )
                    else:
                        st.error(
                            "**Unattractive:** Equity earnings yield "
                            f"spread is only {spread * 100:.1f}% over "
                            "bonds. Fixed income may offer better "
                            "risk-adjusted returns for new capital."
                        )
                else:
                    st.info("Insufficient P/E data for yield comparison.")
            else:
                st.info("Market data unavailable for yield calculation.")
        else:
            st.info("No ranked securities available for yield comparison.")
    else:
        st.info("Treasury yield data unavailable for assessment.")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the full Streamlit dashboard."""
    # Header
    st.markdown(
        '<h1 style="margin-bottom:0;">📊 Buffett Screener</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Warren Buffett-style equity screening and valuation dashboard")

    # Check for database
    if not _db_exists():
        st.error(
            "**Database not found.**\n\n"
            f"Expected: `{_DB_PATH}`\n\n"
            "Run the pipeline first to populate the database:\n\n"
            "```bash\n"
            "python -m output.pipeline_runner --mode reports\n"
            "```"
        )
        return

    # Load ranked data
    ranked_df = _build_ranked_table()

    if ranked_df.empty:
        st.warning(
            "The database exists but contains no ranked securities. "
            "Ensure the pipeline has completed all stages:\n\n"
            "```bash\n"
            "python -m output.pipeline_runner --mode reports\n"
            "```"
        )
        return

    # Apply sidebar filters
    filtered_df = _render_sidebar(ranked_df)

    # Tabs
    tab_ranked, tab_deep_dive, tab_summary, tab_macro = st.tabs([
        "📋 Ranked Results",
        "🔍 Deep Dive",
        "📊 Screener Summary",
        "🌐 Macro Context",
    ])

    with tab_ranked:
        selected_ticker = _render_tab_ranked(filtered_df)

    with tab_deep_dive:
        # Use ticker from ranked tab or let user pick from selectbox
        all_tickers = filtered_df["ticker"].tolist()
        deep_dive_ticker = st.selectbox(
            "Select ticker for analysis:",
            options=all_tickers if all_tickers else [""],
            index=0,
            key="deep_dive_ticker_select",
        )
        if deep_dive_ticker:
            _render_tab_deep_dive(deep_dive_ticker)
        else:
            _render_tab_deep_dive(None)

    with tab_summary:
        _render_tab_summary(ranked_df)

    with tab_macro:
        _render_tab_macro()


if __name__ == "__main__":
    main()

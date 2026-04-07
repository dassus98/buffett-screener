"""
metrics_engine
==============
Computes all quantitative metrics from a TickerDataBundle.

The metrics engine is stateless and pure — it takes data in, computes numbers,
and returns a MetricsBundle. It does not read from or write to any data store;
that is the responsibility of the pipeline runner.

Public surface:
    - compute_all_metrics(bundle) → MetricsBundle
    Individual metric modules are importable for testing and ad-hoc use.

Metric categories:
    owner_earnings   → Buffett's owner earnings (FCF proxy)
    returns          → ROIC, ROE, ROCE
    profitability    → Gross/operating/net margins and their consistency
    leverage         → D/E, net debt/EBITDA, interest coverage
    growth           → Revenue, EPS, book value CAGR over 5 and 10 years
    capex            → CapEx intensity, maintenance vs. growth CapEx
    valuation        → P/E, P/B, EV/EBITDA, earnings yield, FCF yield
    composite_score  → Weighted composite quality + value score
"""

from metrics_engine.owner_earnings import compute_owner_earnings
from metrics_engine.returns import compute_returns
from metrics_engine.profitability import compute_profitability
from metrics_engine.leverage import compute_leverage
from metrics_engine.growth import compute_growth
from metrics_engine.capex import compute_capex_metrics
from metrics_engine.valuation import compute_valuation_multiples
from metrics_engine.composite_score import compute_composite_score


def compute_all_metrics(bundle, config: dict) -> dict:
    """
    Orchestrate computation of all metric categories for a TickerDataBundle.

    Args:
        bundle: TickerDataBundle with financial + market + macro data populated.
        config: Full filter_config.yaml dict (used for weights and parameters).

    Returns:
        Dict of all computed metrics, suitable for passing to screener filters
        and valuation reports. Keys mirror the section names of the metric modules.

    Logic:
        Call each compute_* function in dependency order (leverage before
        composite, valuation before composite, etc.) and merge their results
        into a single flat dict keyed by metric name.
    """
    ...


__all__ = [
    "compute_all_metrics",
    "compute_owner_earnings",
    "compute_returns",
    "compute_profitability",
    "compute_leverage",
    "compute_growth",
    "compute_capex_metrics",
    "compute_valuation_multiples",
    "compute_composite_score",
]

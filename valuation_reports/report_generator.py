"""
valuation_reports.report_generator
=====================================
Orchestrates all valuation sub-modules and renders the final Markdown reports
using Jinja2 templates.

Per-stock pipeline:
    1. compute_intrinsic_value()       → IntrinsicValueEstimate
    2. compute_margin_of_safety()      → MarginOfSafetyResult
    3. compute_earnings_yield_comparison() → EarningsYieldComparison
    4. generate_qualitative_prompts()  → dict of prompts
    5. generate_recommendation()       → RecommendationResult
    6. Render to Markdown via Jinja2 deep_dive_template.md
    7. Write to data/reports/{ticker}_report.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from data_acquisition.schema import TickerDataBundle
from valuation_reports.intrinsic_value import IntrinsicValueEstimate
from valuation_reports.margin_of_safety import MarginOfSafetyResult
from valuation_reports.earnings_yield import EarningsYieldComparison
from valuation_reports.recommendation import RecommendationResult


TEMPLATE_DIR = Path(__file__).parent / "templates"
REPORTS_DIR = Path("data/reports")


@dataclass
class ValuationReport:
    """Aggregated valuation analysis for a single stock."""

    ticker: str
    bundle: TickerDataBundle
    metrics: dict
    intrinsic_value: IntrinsicValueEstimate
    margin_of_safety: MarginOfSafetyResult
    earnings_yield: EarningsYieldComparison
    qualitative_prompts: dict[str, list[str]]
    recommendation: RecommendationResult
    report_markdown: str = ""           # rendered Markdown text


def generate_report(
    bundle: TickerDataBundle,
    metrics: dict,
    config: dict,
    write_to_disk: bool = True,
) -> ValuationReport:
    """
    Run the full valuation pipeline for a single stock and generate a report.

    Args:
        bundle:        TickerDataBundle with all financial and market data.
        metrics:       Pre-computed metrics dict (output of compute_all_metrics()).
        config:        Full filter_config.yaml dict.
        write_to_disk: If True, write the rendered Markdown to data/reports/.

    Returns:
        ValuationReport dataclass with all computed valuation fields and the
        rendered Markdown report string.

    Logic:
        1. Run compute_intrinsic_value()
        2. Run compute_margin_of_safety()
        3. Run compute_earnings_yield_comparison()
        4. Run generate_qualitative_prompts()
        5. Run generate_recommendation()
        6. Render Markdown via _render_deep_dive()
        7. Optionally write to data/reports/{ticker}_report.md
        8. Return ValuationReport
    """
    ...


def generate_all_reports(
    ranked_df,
    bundles: dict[str, TickerDataBundle],
    all_metrics: dict[str, dict],
    config: dict,
    top_n: Optional[int] = None,
    write_to_disk: bool = True,
) -> list[ValuationReport]:
    """
    Generate valuation reports for all stocks in the ranked universe.

    Args:
        ranked_df:    DataFrame from rank_universe() with composite scores and ranks.
        bundles:      Dict mapping ticker → TickerDataBundle.
        all_metrics:  Dict mapping ticker → metrics dict.
        config:       Full filter_config.yaml dict.
        top_n:        If provided, only generate reports for the top N stocks.
                      Defaults to config["output"]["top_n_stocks"].
        write_to_disk: Whether to write individual reports to data/reports/.

    Returns:
        List of ValuationReport objects in rank order.

    Logic:
        1. Slice ranked_df to top_n rows
        2. For each ticker, call generate_report()
        3. Return list of ValuationReports
        4. Log progress (report X of N)
    """
    ...


def _render_deep_dive(
    report: ValuationReport,
    template_name: str = "deep_dive_template.md",
) -> str:
    """
    Render the deep-dive Markdown report using the Jinja2 template.

    Args:
        report:        ValuationReport with all computed fields.
        template_name: Filename of the Jinja2 template in templates/.

    Returns:
        Rendered Markdown string.

    Logic:
        1. Load template from TEMPLATE_DIR / template_name using jinja2.Environment
        2. Pass all report fields as template context
        3. Render and return the string
    """
    ...


def _write_report(ticker: str, markdown: str) -> Path:
    """
    Write a rendered Markdown report to the data/reports/ directory.

    Args:
        ticker:   Ticker symbol (used as the filename base).
        markdown: Rendered Markdown string.

    Returns:
        Path to the written file.

    Logic:
        1. Ensure REPORTS_DIR exists (mkdir -p)
        2. Write markdown to REPORTS_DIR / f"{ticker}_report.md"
        3. Return the file path
    """
    ...

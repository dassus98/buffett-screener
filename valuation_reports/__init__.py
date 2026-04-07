"""
valuation_reports
=================
Generates per-stock valuation analyses and formatted reports.

For each stock that passes the screener, this module produces:
    1. Intrinsic value estimate via owner-earnings DCF
    2. Margin of safety calculation
    3. Earnings yield vs. bond yield comparison
    4. Qualitative analysis prompts (for human review)
    5. Formatted Markdown report (individual deep-dive)
    6. Buy / Hold / Pass recommendation with justification

Public surface:
    generate_report(bundle, metrics, config) → ValuationReport
    generate_all_reports(ranked_df, bundles, config) → list[ValuationReport]
"""

from valuation_reports.intrinsic_value import compute_intrinsic_value
from valuation_reports.margin_of_safety import compute_margin_of_safety
from valuation_reports.earnings_yield import compute_earnings_yield_comparison
from valuation_reports.recommendation import generate_recommendation
from valuation_reports.report_generator import generate_report, generate_all_reports

__all__ = [
    "compute_intrinsic_value",
    "compute_margin_of_safety",
    "compute_earnings_yield_comparison",
    "generate_recommendation",
    "generate_report",
    "generate_all_reports",
]

"""
output.markdown_export
=======================
Exports valuation reports and the summary table to Markdown files
in the data/reports/ directory.

This module is a thin wrapper around the report rendering logic in
valuation_reports.report_generator, handling file system operations
and batch export coordination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

REPORTS_DIR = Path("data/reports")


def export_all_reports(
    reports: list,
    output_dir: Path = REPORTS_DIR,
    overwrite: bool = True,
) -> list[Path]:
    """
    Write all ValuationReport objects to individual Markdown files.

    Args:
        reports:    List of ValuationReport objects (from generate_all_reports()).
        output_dir: Directory to write reports to (default: data/reports/).
        overwrite:  If False, skip tickers that already have a report file.

    Returns:
        List of Paths to the written Markdown files.

    Logic:
        1. Ensure output_dir exists
        2. For each report, write report.report_markdown to
           output_dir / f"{report.ticker}_report.md"
        3. Skip existing files if overwrite=False
        4. Return list of written paths
    """
    ...


def export_summary_table(
    summary_markdown: str,
    output_dir: Path = REPORTS_DIR,
    filename: str = "summary.md",
) -> Path:
    """
    Write the summary table Markdown string to a file.

    Args:
        summary_markdown: Rendered Markdown string (from generate_summary_table()).
        output_dir:       Directory to write to.
        filename:         Output filename.

    Returns:
        Path to the written file.
    """
    ...


def collect_existing_reports(output_dir: Path = REPORTS_DIR) -> dict[str, Path]:
    """
    Scan the output directory and return existing report files.

    Args:
        output_dir: Directory to scan.

    Returns:
        Dict mapping ticker symbol → Path for each existing *_report.md file.

    Logic:
        Glob for "*_report.md" in output_dir.
        Extract ticker from filename: "AAPL_report.md" → "AAPL".
        Return {ticker: path} dict.
    """
    ...

"""Writes rendered ValuationReport Markdown files and the summary table
to the data/reports/ directory.

Provides a post-processing step that combines individually generated
per-ticker reports and the summary report into a single concatenated
``all_reports.md`` file with a navigable table of contents.

Data Lineage Contract
---------------------
Upstream producers:
    - ``valuation_reports.report_generator.generate_all_reports``
      → writes ``{TICKER}_analysis.md`` and ``summary.md`` to
        ``data/reports/``.

Downstream consumers:
    - End user reads ``all_reports.md`` as a combined portfolio document.

Config dependencies (all via ``get_threshold``):
    - ``output.report_dir``  (default ``data/reports``)
"""

from __future__ import annotations

import datetime
import logging
import pathlib

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_report_dir() -> pathlib.Path:
    """Resolve the report output directory from config.

    Returns
    -------
    pathlib.Path
        Absolute path to the report directory.
    """
    try:
        report_dir_str = str(get_threshold("output.report_dir"))
    except (KeyError, ValueError):
        report_dir_str = "data/reports"
    return _PROJECT_ROOT / report_dir_str


def _read_report_files(
    report_dir: pathlib.Path,
) -> tuple[list[pathlib.Path], pathlib.Path | None]:
    """Discover individual analysis and summary reports.

    Parameters
    ----------
    report_dir:
        Directory containing generated Markdown reports.

    Returns
    -------
    tuple[list[pathlib.Path], pathlib.Path | None]
        ``(analysis_files, summary_file)`` — analysis files sorted
        alphabetically by ticker, and the summary file (or ``None``).
    """
    analysis_files: list[pathlib.Path] = sorted(
        report_dir.glob("*_analysis.md"),
    )
    summary_path = report_dir / "summary.md"
    summary_file = summary_path if summary_path.exists() else None
    return analysis_files, summary_file


def _extract_ticker(path: pathlib.Path) -> str:
    """Extract the ticker symbol from an analysis report filename.

    Parameters
    ----------
    path:
        Path like ``data/reports/AAPL_analysis.md``.

    Returns
    -------
    str
        Ticker symbol (e.g. ``"AAPL"``).
    """
    stem = path.stem  # e.g. "AAPL_analysis"
    return stem.replace("_analysis", "")


def _build_table_of_contents(
    analysis_files: list[pathlib.Path],
    has_summary: bool,
) -> str:
    """Build a Markdown table of contents for the combined report.

    Parameters
    ----------
    analysis_files:
        List of per-ticker analysis report paths.
    has_summary:
        Whether a summary report is included.

    Returns
    -------
    str
        Markdown-formatted table of contents.
    """
    lines: list[str] = [
        "# Buffett Screener — Combined Report",
        "",
        f"*Generated: {datetime.date.today().isoformat()}*",
        "",
        "## Table of Contents",
        "",
    ]

    if has_summary:
        lines.append("1. [Portfolio Summary](#portfolio-summary)")

    for i, path in enumerate(analysis_files, start=2 if has_summary else 1):
        ticker = _extract_ticker(path)
        anchor = f"deep-dive-{ticker.lower()}"
        lines.append(f"{i}. [{ticker} — Deep Dive Analysis](#{anchor})")

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _format_section(
    content: str,
    ticker: str | None = None,
) -> str:
    """Wrap a report section with an anchor heading and separator.

    Parameters
    ----------
    content:
        Raw Markdown content of the report.
    ticker:
        Ticker symbol for the anchor (``None`` for summary).

    Returns
    -------
    str
        Content prefixed with an anchor tag and followed by a separator.
    """
    if ticker is not None:
        anchor = f"deep-dive-{ticker.lower()}"
        header = f'<a id="{anchor}"></a>\n\n'
    else:
        header = '<a id="portfolio-summary"></a>\n\n'

    return f"{header}{content}\n\n---\n\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_reports(
    report_dir: pathlib.Path | str | None = None,
) -> pathlib.Path | None:
    """Combine all individual reports into a single ``all_reports.md`` file.

    Reads all ``*_analysis.md`` files and ``summary.md`` from the report
    directory, builds a table of contents, and writes the combined
    output to ``all_reports.md`` in the same directory.

    Parameters
    ----------
    report_dir:
        Path to the report directory.  If ``None``, reads from config
        ``output.report_dir``.

    Returns
    -------
    pathlib.Path | None
        Path to the generated ``all_reports.md`` file, or ``None`` if
        no reports were found.
    """
    if report_dir is None:
        resolved_dir = _get_report_dir()
    else:
        resolved_dir = pathlib.Path(report_dir)

    if not resolved_dir.exists():
        logger.warning(
            "Report directory does not exist: %s", resolved_dir,
        )
        return None

    analysis_files, summary_file = _read_report_files(resolved_dir)

    if not analysis_files and summary_file is None:
        logger.warning(
            "No reports found in %s; nothing to combine.", resolved_dir,
        )
        return None

    # Build table of contents
    toc = _build_table_of_contents(
        analysis_files, has_summary=summary_file is not None,
    )

    # Assemble combined content
    parts: list[str] = [toc]

    # Summary first (if present)
    if summary_file is not None:
        summary_content = summary_file.read_text(encoding="utf-8")
        parts.append(_format_section(summary_content, ticker=None))
        logger.info("Included summary report.")

    # Individual analyses
    for path in analysis_files:
        ticker = _extract_ticker(path)
        content = path.read_text(encoding="utf-8")
        parts.append(_format_section(content, ticker=ticker))
        logger.info("Included analysis report: %s", ticker)

    combined = "".join(parts)
    output_path = resolved_dir / "all_reports.md"
    output_path.write_text(combined, encoding="utf-8")

    logger.info(
        "Combined report written to %s (%d individual + %s summary).",
        output_path,
        len(analysis_files),
        "1" if summary_file else "0",
    )

    return output_path

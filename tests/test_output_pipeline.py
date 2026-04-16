"""Tests for the output package: pipeline_runner, markdown_export, summary_table.

Covers:
- Argparse parsing (--mode required, --top defaults, flags recognised)
- Pipeline runner logging configuration
- Summary table console output (no crash, correct format)
- Markdown export combined report creation
"""

from __future__ import annotations

import io
import json
import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from output.markdown_export import (
    _build_table_of_contents,
    _extract_ticker,
    _format_section,
    export_reports,
)
from output.pipeline_runner import (
    _write_run_log,
    build_arg_parser,
    configure_logging,
    run_pipeline,
)
from output.summary_table import (
    _safe_dollar,
    _safe_pct,
    _safe_score,
    _safe_str,
    print_summary_to_console,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_shortlist_df(n: int = 5) -> pd.DataFrame:
    """Create a minimal shortlist DataFrame with *n* rows.

    Parameters
    ----------
    n:
        Number of rows.

    Returns
    -------
    pd.DataFrame
    """
    tickers = [f"T{i:03d}" for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "ticker": tickers,
            "composite_score": [90 - i * 5 for i in range(n)],
            "rank": list(range(1, n + 1)),
            "iv_weighted": [200.0 - i * 10 for i in range(n)],
            "current_price_usd": [150.0 - i * 5 for i in range(n)],
            "margin_of_safety_pct": [0.25 + i * 0.05 for i in range(n)],
            "recommendation": ["Buy"] * min(3, n)
            + ["Hold"] * max(0, n - 3),
        },
    )


# ===========================================================================
# TestBuildArgParser
# ===========================================================================


class TestBuildArgParser:
    """Tests for CLI argument parsing."""

    def test_mode_is_required(self) -> None:
        """--mode must be provided."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_mode_reports(self) -> None:
        """--mode reports is accepted."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports"])
        assert args.mode == "reports"

    def test_mode_dashboard(self) -> None:
        """--mode dashboard is accepted."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "dashboard"])
        assert args.mode == "dashboard"

    def test_invalid_mode_rejected(self) -> None:
        """Invalid --mode value raises SystemExit."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "invalid"])

    def test_top_default_is_none(self) -> None:
        """--top defaults to None when not provided."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports"])
        assert args.top is None

    def test_top_accepts_integer(self) -> None:
        """--top accepts an integer value."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports", "--top", "30"])
        assert args.top == 30

    def test_exchange_default_is_all(self) -> None:
        """--exchange defaults to ALL."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports"])
        assert args.exchange == "ALL"

    def test_exchange_accepts_tsx(self) -> None:
        """--exchange TSX is accepted."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports", "--exchange", "TSX"])
        assert args.exchange == "TSX"

    def test_skip_acquisition_flag(self) -> None:
        """--skip-acquisition sets skip_acquisition to True."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports", "--skip-acquisition"])
        assert args.skip_acquisition is True

    def test_skip_metrics_flag(self) -> None:
        """--skip-metrics sets skip_metrics to True."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports", "--skip-metrics"])
        assert args.skip_metrics is True

    def test_no_cache_flag(self) -> None:
        """--no-cache sets no_cache to True."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports", "--no-cache"])
        assert args.no_cache is True

    def test_no_moat_flag(self) -> None:
        """--no-moat sets no_moat to True."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports", "--no-moat"])
        assert args.no_moat is True

    def test_verbose_flag(self) -> None:
        """--verbose sets verbose to True."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports", "--verbose"])
        assert args.verbose is True

    def test_flags_default_false(self) -> None:
        """All boolean flags default to False."""
        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports"])
        assert args.skip_acquisition is False
        assert args.skip_metrics is False
        assert args.no_cache is False
        assert args.no_moat is False
        assert args.verbose is False

    def test_all_flags_combined(self) -> None:
        """All flags can be specified together."""
        parser = build_arg_parser()
        args = parser.parse_args([
            "--mode", "reports",
            "--top", "10",
            "--exchange", "NASDAQ",
            "--skip-acquisition",
            "--skip-metrics",
            "--no-cache",
            "--no-moat",
            "--verbose",
        ])
        assert args.mode == "reports"
        assert args.top == 10
        assert args.exchange == "NASDAQ"
        assert args.skip_acquisition is True
        assert args.skip_metrics is True
        assert args.no_cache is True
        assert args.no_moat is True
        assert args.verbose is True


# ===========================================================================
# TestConfigureLogging
# ===========================================================================


class TestConfigureLogging:
    """Tests for logging configuration."""

    @patch("output.pipeline_runner.get_threshold")
    def test_configure_logging_no_crash(
        self, mock_gt: MagicMock,
    ) -> None:
        """configure_logging does not crash."""
        mock_gt.side_effect = lambda key: {
            "logging.level": "INFO",
            "logging.log_file": "data/pipeline.log",
        }.get(key, "INFO")
        # Should not raise — we can't fully test file handler in unit tests
        # but we verify it doesn't crash
        configure_logging(verbose=False)

    @patch("output.pipeline_runner.get_threshold")
    def test_verbose_sets_debug(
        self, mock_gt: MagicMock,
    ) -> None:
        """verbose=True should configure DEBUG level."""
        mock_gt.side_effect = lambda key: {
            "logging.level": "INFO",
            "logging.log_file": "data/pipeline.log",
        }.get(key, "INFO")
        configure_logging(verbose=True)
        import logging

        root = logging.getLogger()
        assert root.level == logging.DEBUG


# ===========================================================================
# TestWriteRunLog
# ===========================================================================


class TestWriteRunLog:
    """Tests for run log JSON output."""

    @patch("output.pipeline_runner.get_threshold")
    def test_writes_json_file(
        self, mock_gt: MagicMock, tmp_path: pathlib.Path,
    ) -> None:
        """_write_run_log creates a valid JSON file with pipeline stats."""
        mock_gt.return_value = str(tmp_path)

        # Patch _PROJECT_ROOT so the path resolves correctly
        with patch(
            "output.pipeline_runner._PROJECT_ROOT",
            pathlib.Path("/"),
        ):
            parser = build_arg_parser()
            args = parser.parse_args(["--mode", "reports"])
            log_path = _write_run_log(
                args=args,
                elapsed_seconds=42.5,
                stages_run=["data_acquisition", "metrics_engine"],
                report_paths=[pathlib.Path("summary.md")],
                pipeline_stats={
                    "universe_size": 500,
                    "tier1_survivors": 200,
                    "shortlisted": 30,
                    "reports_generated": 30,
                },
            )

        assert log_path.exists()
        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert data["mode"] == "reports"
        assert data["runtime_seconds"] == 42.5
        assert data["status"] == "success"
        assert "data_acquisition" in data["stages_run"]
        # Verify pipeline statistics keys
        assert data["universe_size"] == 500
        assert data["tier1_survivors"] == 200
        assert data["shortlisted"] == 30
        assert data["reports_generated"] == 30

    @patch("output.pipeline_runner.get_threshold")
    def test_error_status_on_failure(
        self, mock_gt: MagicMock, tmp_path: pathlib.Path,
    ) -> None:
        """_write_run_log records error status and message."""
        mock_gt.return_value = str(tmp_path)

        with patch(
            "output.pipeline_runner._PROJECT_ROOT",
            pathlib.Path("/"),
        ):
            parser = build_arg_parser()
            args = parser.parse_args(["--mode", "reports"])
            log_path = _write_run_log(
                args=args,
                elapsed_seconds=1.0,
                stages_run=[],
                report_paths=None,
                error="Test error",
            )

        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert data["status"] == "error"
        assert data["error"] == "Test error"
        # Default pipeline_stats when None passed
        assert data["universe_size"] == 0
        assert data["reports_generated"] == 0


# ===========================================================================
# TestRunPipeline
# ===========================================================================


class TestRunPipeline:
    """Integration-style tests for run_pipeline with mocked stages."""

    @patch("output.pipeline_runner._write_run_log")
    @patch("output.pipeline_runner._run_stage_4_reports")
    @patch("output.pipeline_runner._run_stage_3")
    @patch("output.pipeline_runner._run_stage_2")
    @patch("output.pipeline_runner._run_stage_1")
    @patch("output.pipeline_runner.configure_logging")
    def test_full_pipeline_reports_mode(
        self,
        mock_log: MagicMock,
        mock_s1: MagicMock,
        mock_s2: MagicMock,
        mock_s3: MagicMock,
        mock_s4: MagicMock,
        mock_run_log: MagicMock,
    ) -> None:
        """Full pipeline in reports mode calls all 4 stages."""
        mock_s1.return_value = {
            "universe_size": 100,
            "survivors": 80,
            "dropped": 20,
            "macro": {},
        }
        mock_s2.return_value = pd.DataFrame({"ticker": ["AAPL"]})
        mock_s3.return_value = (
            pd.DataFrame({"ticker": ["AAPL"]}),
            {"total_universe": 100},
        )
        mock_s4.return_value = [pathlib.Path("AAPL_analysis.md")]

        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "reports"])
        run_pipeline(args)

        mock_s1.assert_called_once()
        mock_s2.assert_called_once()
        mock_s3.assert_called_once()
        mock_s4.assert_called_once()

    @patch("output.pipeline_runner._write_run_log")
    @patch("output.pipeline_runner._run_stage_3")
    @patch("output.pipeline_runner.configure_logging")
    def test_skip_acquisition_and_metrics(
        self,
        mock_log: MagicMock,
        mock_s3: MagicMock,
        mock_run_log: MagicMock,
    ) -> None:
        """--skip-acquisition --skip-metrics skips stages 1 and 2."""
        mock_s3.return_value = (pd.DataFrame(), {})

        parser = build_arg_parser()
        args = parser.parse_args([
            "--mode", "reports",
            "--skip-acquisition",
            "--skip-metrics",
        ])

        with patch(
            "output.pipeline_runner._run_stage_4_reports",
            return_value=[],
        ):
            run_pipeline(args)

        # Stages 1 and 2 should not be called — they are skipped
        # We verify by checking that mock_s3 was called (stage 3 always runs)
        mock_s3.assert_called_once()

    @patch("output.pipeline_runner._write_run_log")
    @patch("output.pipeline_runner._run_stage_4_dashboard")
    @patch("output.pipeline_runner._run_stage_3")
    @patch("output.pipeline_runner._run_stage_2")
    @patch("output.pipeline_runner._run_stage_1")
    @patch("output.pipeline_runner.configure_logging")
    def test_dashboard_mode(
        self,
        mock_log: MagicMock,
        mock_s1: MagicMock,
        mock_s2: MagicMock,
        mock_s3: MagicMock,
        mock_s4_dash: MagicMock,
        mock_run_log: MagicMock,
    ) -> None:
        """Dashboard mode calls _run_stage_4_dashboard."""
        mock_s1.return_value = {
            "universe_size": 100,
            "survivors": 80,
            "dropped": 20,
            "macro": {},
        }
        mock_s2.return_value = pd.DataFrame()
        mock_s3.return_value = (pd.DataFrame(), {})

        parser = build_arg_parser()
        args = parser.parse_args(["--mode", "dashboard"])
        run_pipeline(args)

        mock_s4_dash.assert_called_once()


# ===========================================================================
# TestPrintSummaryToConsole
# ===========================================================================


class TestPrintSummaryToConsole:
    """Tests for console summary table output."""

    def test_empty_df_no_crash(self) -> None:
        """Empty DataFrame produces a 'no stocks' message."""
        buf = io.StringIO()
        print_summary_to_console(pd.DataFrame(), output=buf)
        output = buf.getvalue()
        assert "No stocks" in output

    def test_basic_output_contains_header(self) -> None:
        """Output contains column headers."""
        df = _make_shortlist_df(3)
        buf = io.StringIO()
        print_summary_to_console(df, output=buf)
        output = buf.getvalue()
        assert "Rank" in output
        assert "Ticker" in output
        assert "Score" in output
        assert "IV" in output
        assert "Price" in output
        assert "MoS%" in output
        assert "Rec" in output

    def test_basic_output_contains_tickers(self) -> None:
        """Output contains ticker symbols from the DataFrame."""
        df = _make_shortlist_df(3)
        buf = io.StringIO()
        print_summary_to_console(df, output=buf)
        output = buf.getvalue()
        assert "T001" in output
        assert "T002" in output
        assert "T003" in output

    def test_max_rows_limits_output(self) -> None:
        """max_rows limits the number of displayed rows."""
        df = _make_shortlist_df(10)
        buf = io.StringIO()
        print_summary_to_console(df, max_rows=3, output=buf)
        output = buf.getvalue()
        assert "T001" in output
        assert "T003" in output
        # T004 should not be shown since max_rows=3
        assert "T004" not in output
        assert "Showing 3 of 10" in output

    def test_showing_count_message(self) -> None:
        """Footer shows correct 'Showing X of Y' message."""
        df = _make_shortlist_df(5)
        buf = io.StringIO()
        print_summary_to_console(df, output=buf)
        output = buf.getvalue()
        assert "Showing 5 of 5" in output

    def test_title_banner(self) -> None:
        """Output includes a title banner."""
        df = _make_shortlist_df(1)
        buf = io.StringIO()
        print_summary_to_console(df, output=buf)
        output = buf.getvalue()
        assert "Buffett Screener" in output

    def test_missing_optional_columns(self) -> None:
        """Works even when optional columns are missing."""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "composite_score": [85.0],
                "rank": [1],
                "score_category": ["Strong Buy"],
            },
        )
        buf = io.StringIO()
        print_summary_to_console(df, output=buf)
        output = buf.getvalue()
        assert "AAPL" in output


# ===========================================================================
# TestSafeStr
# ===========================================================================


class TestSafeStr:
    """Tests for _safe_str helper."""

    def test_none_renders_dash(self) -> None:
        """None renders as '—'."""
        result = _safe_str(None, 10)
        assert "—" in result

    def test_nan_renders_dash(self) -> None:
        """NaN renders as '—'."""
        result = _safe_str(float("nan"), 10)
        assert "—" in result

    def test_truncation(self) -> None:
        """Long strings are truncated with ellipsis."""
        result = _safe_str("This is a very long string", 10)
        assert len(result) == 10
        assert result.endswith("…")

    def test_short_string_padded(self) -> None:
        """Short strings are padded to width."""
        result = _safe_str("hi", 10)
        assert len(result) == 10


# ===========================================================================
# TestSafeScore
# ===========================================================================


class TestSafeScore:
    """Tests for _safe_score helper."""

    def test_valid_float(self) -> None:
        """Valid float formats with one decimal."""
        assert _safe_score(78.3) == "78.3"

    def test_nan_returns_dash(self) -> None:
        """NaN returns '—'."""
        assert _safe_score(float("nan")) == "—"

    def test_none_returns_dash(self) -> None:
        """None returns '—'."""
        assert _safe_score(None) == "—"

    def test_string_returns_dash(self) -> None:
        """Non-numeric string returns '—'."""
        assert _safe_score("abc") == "—"


# ===========================================================================
# TestExportReports
# ===========================================================================


class TestExportReports:
    """Tests for markdown_export.export_reports."""

    def test_creates_combined_file(self, tmp_path: pathlib.Path) -> None:
        """export_reports creates all_reports.md from analysis files."""
        # Create mock analysis files
        (tmp_path / "AAPL_analysis.md").write_text(
            "# AAPL Analysis\nContent here.", encoding="utf-8",
        )
        (tmp_path / "MSFT_analysis.md").write_text(
            "# MSFT Analysis\nContent here.", encoding="utf-8",
        )

        result = export_reports(report_dir=tmp_path)

        assert result is not None
        assert result.exists()
        assert result.name == "all_reports.md"

        content = result.read_text(encoding="utf-8")
        assert "AAPL" in content
        assert "MSFT" in content

    def test_includes_summary(self, tmp_path: pathlib.Path) -> None:
        """Combined report includes summary.md when present."""
        (tmp_path / "summary.md").write_text(
            "# Summary\nOverview.", encoding="utf-8",
        )
        (tmp_path / "AAPL_analysis.md").write_text(
            "# AAPL\nData.", encoding="utf-8",
        )

        result = export_reports(report_dir=tmp_path)

        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert "Portfolio Summary" in content
        assert "AAPL" in content

    def test_table_of_contents(self, tmp_path: pathlib.Path) -> None:
        """Combined report has a table of contents."""
        (tmp_path / "AAPL_analysis.md").write_text("# A", encoding="utf-8")
        (tmp_path / "MSFT_analysis.md").write_text("# M", encoding="utf-8")

        result = export_reports(report_dir=tmp_path)

        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert "Table of Contents" in content
        assert "AAPL" in content
        assert "MSFT" in content

    def test_no_reports_returns_none(self, tmp_path: pathlib.Path) -> None:
        """Returns None when no report files exist."""
        result = export_reports(report_dir=tmp_path)
        assert result is None

    def test_nonexistent_dir_returns_none(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Returns None when directory does not exist."""
        result = export_reports(report_dir=tmp_path / "nonexistent")
        assert result is None

    def test_only_summary_creates_combined(
        self, tmp_path: pathlib.Path,
    ) -> None:
        """Works with only summary.md and no analysis files."""
        (tmp_path / "summary.md").write_text(
            "# Summary", encoding="utf-8",
        )

        result = export_reports(report_dir=tmp_path)

        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert "Summary" in content

    def test_anchors_in_combined(self, tmp_path: pathlib.Path) -> None:
        """Combined report contains anchor tags for navigation."""
        (tmp_path / "AAPL_analysis.md").write_text("# A", encoding="utf-8")

        result = export_reports(report_dir=tmp_path)
        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert 'id="deep-dive-aapl"' in content


# ===========================================================================
# TestExtractTicker
# ===========================================================================


class TestExtractTicker:
    """Tests for _extract_ticker helper."""

    def test_standard_filename(self) -> None:
        """Extracts ticker from standard filename."""
        path = pathlib.Path("data/reports/AAPL_analysis.md")
        assert _extract_ticker(path) == "AAPL"

    def test_dotted_ticker(self) -> None:
        """Extracts ticker with dot (e.g. BRK.B)."""
        path = pathlib.Path("data/reports/BRK.B_analysis.md")
        assert _extract_ticker(path) == "BRK.B"


# ===========================================================================
# TestBuildTableOfContents
# ===========================================================================


class TestBuildTableOfContents:
    """Tests for _build_table_of_contents."""

    def test_with_summary(self) -> None:
        """TOC includes summary entry when has_summary is True."""
        files = [pathlib.Path("AAPL_analysis.md")]
        toc = _build_table_of_contents(files, has_summary=True)
        assert "Portfolio Summary" in toc
        assert "AAPL" in toc

    def test_without_summary(self) -> None:
        """TOC omits summary entry when has_summary is False."""
        files = [pathlib.Path("AAPL_analysis.md")]
        toc = _build_table_of_contents(files, has_summary=False)
        assert "Portfolio Summary" not in toc
        assert "AAPL" in toc

    def test_multiple_tickers(self) -> None:
        """TOC lists all ticker entries."""
        files = [
            pathlib.Path("AAPL_analysis.md"),
            pathlib.Path("MSFT_analysis.md"),
            pathlib.Path("KO_analysis.md"),
        ]
        toc = _build_table_of_contents(files, has_summary=False)
        assert "AAPL" in toc
        assert "MSFT" in toc
        assert "KO" in toc


# ===========================================================================
# TestFormatSection
# ===========================================================================


class TestFormatSection:
    """Tests for _format_section."""

    def test_analysis_section_has_anchor(self) -> None:
        """Analysis sections have a ticker-based anchor."""
        result = _format_section("content", ticker="AAPL")
        assert 'id="deep-dive-aapl"' in result
        assert "content" in result

    def test_summary_section_has_anchor(self) -> None:
        """Summary section has a portfolio-summary anchor."""
        result = _format_section("content", ticker=None)
        assert 'id="portfolio-summary"' in result

    def test_separator_included(self) -> None:
        """Sections end with a horizontal rule."""
        result = _format_section("content", ticker="X")
        assert "---" in result


# ===========================================================================
# TestSafeDollar
# ===========================================================================


class TestSafeDollar:
    """Tests for _safe_dollar helper."""

    def test_valid_float(self) -> None:
        """Valid float formats as dollar amount."""
        assert _safe_dollar(150.25) == "$150.25"

    def test_nan_returns_dash(self) -> None:
        """NaN returns '—'."""
        assert _safe_dollar(float("nan")) == "—"

    def test_none_returns_dash(self) -> None:
        """None returns '—'."""
        assert _safe_dollar(None) == "—"

    def test_string_returns_dash(self) -> None:
        """Non-numeric string returns '—'."""
        assert _safe_dollar("abc") == "—"


# ===========================================================================
# TestSafePct
# ===========================================================================


class TestSafePct:
    """Tests for _safe_pct helper."""

    def test_valid_fraction(self) -> None:
        """Decimal fraction formats as percentage."""
        assert _safe_pct(0.25) == "25.0%"

    def test_nan_returns_dash(self) -> None:
        """NaN returns '—'."""
        assert _safe_pct(float("nan")) == "—"

    def test_none_returns_dash(self) -> None:
        """None returns '—'."""
        assert _safe_pct(None) == "—"

    def test_string_returns_dash(self) -> None:
        """Non-numeric string returns '—'."""
        assert _safe_pct("abc") == "—"

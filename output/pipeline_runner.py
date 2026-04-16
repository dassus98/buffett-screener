"""End-to-end pipeline orchestrator: loads config, runs all stages from
universe fetch to report generation, and persists results.

Callable as a module::

    python -m output.pipeline_runner --mode reports --top 30
    python -m output.pipeline_runner --mode dashboard --exchange TSX

Data Lineage Contract
---------------------
Upstream modules invoked (in pipeline order):
    Stage 1 — ``data_acquisition.run_data_acquisition``
        → fetches universe, financials, market data, macro data, runs
          data quality checks, persists to DuckDB.
    Stage 2 — ``metrics_engine.run_metrics_engine``
        → computes all Buffett formula metrics (F1–F16) and composite
          scores for surviving tickers.
    Stage 3 — Screening pipeline:
        a) ``screener.exclusions.apply_exclusions``
        b) ``screener.hard_filters.apply_hard_filters``
        c) ``screener.soft_filters.apply_soft_scores``
        d) ``screener.composite_ranker.generate_shortlist``
        e) ``screener.composite_ranker.generate_screener_summary``
    Stage 4 — Output:
        a) ``valuation_reports.report_generator.generate_all_reports``
           (mode ``reports``)
        b) ``output.streamlit_app`` (mode ``dashboard``) — placeholder.

Downstream consumers:
    - ``data/reports/`` — individual ``{TICKER}_analysis.md`` and
      ``summary.md`` files.
    - ``data/reports/run_log.json`` — execution metadata.

Config dependencies (all via ``get_threshold``):
    - ``output.shortlist_size`` (default 50) — fallback ``--top`` value.
    - ``output.report_dir`` (default ``data/reports``) — report output dir.
    - ``logging.level`` — configures root logger level.
    - ``logging.log_file`` — file handler destination.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys
import time
from typing import Any

import pandas as pd

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all supported CLI flags.
    """
    parser = argparse.ArgumentParser(
        prog="buffett-screener",
        description=(
            "Buffett Screener — end-to-end pipeline from universe fetch "
            "to investment reports."
        ),
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["reports", "dashboard"],
        help="Output mode: 'reports' generates Markdown files; "
             "'dashboard' launches the Streamlit app.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Number of top-ranked stocks to include in the shortlist. "
             "Defaults to output.shortlist_size from config.",
    )
    parser.add_argument(
        "--exchange",
        choices=["TSX", "NYSE", "NASDAQ", "ALL"],
        default="ALL",
        help="Filter universe to a single exchange (default: ALL).",
    )
    parser.add_argument(
        "--skip-acquisition",
        action="store_true",
        help="Skip Stage 1 (data acquisition). Use existing DuckDB data.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip Stage 2 (metrics engine). Use existing metrics tables.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-fetch from all APIs (ignore cached data).",
    )
    parser.add_argument(
        "--no-moat",
        action="store_true",
        help="Disable LLM-assisted qualitative moat assessment.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to DEBUG for all modules.",
    )

    return parser


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool = False) -> None:
    """Configure the root logger with console and file handlers.

    Parameters
    ----------
    verbose:
        If ``True``, override the config log level to ``DEBUG``.
    """
    try:
        level_name = str(get_threshold("logging.level"))
    except (KeyError, ValueError):
        level_name = "INFO"

    if verbose:
        level_name = "DEBUG"

    level = getattr(logging, level_name.upper(), logging.INFO)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    # File handler
    try:
        log_file_str = str(get_threshold("logging.log_file"))
    except (KeyError, ValueError):
        log_file_str = "data/pipeline.log"

    log_path = _PROJECT_ROOT / log_file_str
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on repeated calls
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _run_stage_1(use_cache: bool) -> dict[str, Any]:
    """Stage 1: Data acquisition.

    Parameters
    ----------
    use_cache:
        Forwarded to ``run_data_acquisition``.

    Returns
    -------
    dict
        Acquisition summary with keys ``universe_size``, ``survivors``,
        ``dropped``, ``macro``.
    """
    from data_acquisition import run_data_acquisition

    logger.info("=== Stage 1: Data Acquisition ===")
    result = run_data_acquisition(use_cache=use_cache)
    logger.info(
        "Stage 1 complete: %d tickers, %d survivors.",
        result["universe_size"],
        result["survivors"],
    )
    return result


def _run_stage_2() -> pd.DataFrame:
    """Stage 2: Metrics engine.

    Returns
    -------
    pd.DataFrame
        Composite-score ranking table from ``run_metrics_engine``.
    """
    from metrics_engine import run_metrics_engine

    logger.info("=== Stage 2: Metrics Engine ===")
    composite_df = run_metrics_engine()
    logger.info(
        "Stage 2 complete: %d tickers scored.",
        len(composite_df),
    )
    return composite_df


def _run_stage_3(
    composite_df: pd.DataFrame,
    top_n: int | None,
    exchange_filter: str,
    total_universe: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stage 3: Screening pipeline (exclusions → hard → soft → shortlist).

    Parameters
    ----------
    composite_df:
        Composite-score DataFrame from Stage 2.
    top_n:
        Number of top stocks for the shortlist (``None`` → config default).
    exchange_filter:
        Exchange code to filter on, or ``"ALL"`` for no filtering.
    total_universe:
        Universe size before exclusions, for screener summary.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        ``(shortlist_df, screener_summary)``.
    """
    from data_acquisition.store import read_table
    from screener.composite_ranker import (
        generate_screener_summary,
        generate_shortlist,
    )
    from screener.exclusions import apply_exclusions
    from screener.hard_filters import apply_hard_filters
    from screener.soft_filters import apply_soft_scores

    logger.info("=== Stage 3: Screening Pipeline ===")

    # Step 3a: Read universe and apply exclusions
    universe_df = read_table("universe")
    if exchange_filter != "ALL" and not universe_df.empty:
        universe_df = universe_df[
            universe_df["exchange"] == exchange_filter
        ].reset_index(drop=True)
        logger.info(
            "Exchange filter applied: %s → %d tickers.",
            exchange_filter,
            len(universe_df),
        )

    filtered_df, exclusion_log = apply_exclusions(universe_df)

    # Step 3b: Read metrics summary and apply hard filters
    #   buffett_metrics_summary is a Module 2 table created via
    #   _write_table_full_replace, not in Module 1's _TABLE_DDL.
    #   Read directly via get_connection().
    from data_acquisition.store import get_connection as _get_conn

    conn = _get_conn()
    try:
        metrics_summary_df = conn.execute(
            "SELECT * FROM buffett_metrics_summary",
        ).fetchdf()
    except Exception:
        logger.warning("buffett_metrics_summary table not found.")
        metrics_summary_df = pd.DataFrame()

    # Enrich with years_available from data_quality_log
    # (hard filters need this for the data_sufficiency check)
    if not metrics_summary_df.empty:
        quality_df = read_table("data_quality_log")
        if not quality_df.empty and "years_available" in quality_df.columns:
            metrics_summary_df = metrics_summary_df.merge(
                quality_df[["ticker", "years_available"]],
                on="ticker",
                how="left",
            )

    # Keep only tickers that survived exclusions
    if not filtered_df.empty and not metrics_summary_df.empty:
        surviving_tickers = set(filtered_df["ticker"].tolist())
        metrics_summary_df = metrics_summary_df[
            metrics_summary_df["ticker"].isin(surviving_tickers)
        ].reset_index(drop=True)

    survivors_df, filter_log_df = apply_hard_filters(metrics_summary_df)

    # Step 3c: Apply soft scores
    ranked_df = apply_soft_scores(survivors_df, composite_df)

    # Step 3d: Generate shortlist
    shortlist_df = generate_shortlist(ranked_df, top_n=top_n)

    # Step 3e: Generate screener summary
    screener_summary = generate_screener_summary(
        full_ranked_df=ranked_df,
        shortlist_df=shortlist_df,
        filter_log_df=filter_log_df,
        total_universe=total_universe,
    )

    logger.info(
        "Stage 3 complete: %d shortlisted from %d survivors.",
        len(shortlist_df),
        len(ranked_df),
    )

    return shortlist_df, screener_summary


def _run_stage_4_reports(
    shortlist_df: pd.DataFrame,
    screener_summary: dict[str, Any],
) -> list[pathlib.Path]:
    """Stage 4a: Generate Markdown reports.

    Parameters
    ----------
    shortlist_df:
        Shortlisted stocks DataFrame.
    screener_summary:
        Pipeline statistics dict.

    Returns
    -------
    list[pathlib.Path]
        Paths of generated report files.
    """
    from valuation_reports.report_generator import generate_all_reports

    logger.info("=== Stage 4: Report Generation ===")
    paths = generate_all_reports(shortlist_df, screener_summary)
    logger.info("Stage 4 complete: %d reports generated.", len(paths))
    return paths


def _run_stage_4_dashboard() -> None:
    """Stage 4b: Launch the Streamlit dashboard (placeholder)."""
    logger.info("=== Stage 4: Streamlit Dashboard ===")
    logger.warning(
        "Streamlit dashboard not yet implemented. "
        "Use '--mode reports' for Markdown output.",
    )


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------


def _write_run_log(
    args: argparse.Namespace,
    elapsed_seconds: float,
    stages_run: list[str],
    report_paths: list[pathlib.Path] | None,
    pipeline_stats: dict[str, int] | None = None,
    error: str | None = None,
) -> pathlib.Path:
    """Write a JSON run log to the report directory.

    The log captures both CLI arguments and pipeline statistics for
    downstream auditing and idempotency checks.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    elapsed_seconds:
        Total pipeline wall-clock time.
    stages_run:
        List of stage names that were executed.
    report_paths:
        Paths to generated reports (``None`` if dashboard mode).
    pipeline_stats:
        Dict with ``universe_size``, ``tier1_survivors``, ``shortlisted``,
        ``reports_generated`` (int).  Populated by ``run_pipeline``.
    error:
        Error message if the pipeline failed (``None`` on success).

    Returns
    -------
    pathlib.Path
        Path to the written run log file.
    """
    try:
        report_dir_str = str(get_threshold("output.report_dir"))
    except (KeyError, ValueError):
        report_dir_str = "data/reports"

    report_dir = _PROJECT_ROOT / report_dir_str
    report_dir.mkdir(parents=True, exist_ok=True)

    stats = pipeline_stats or {}

    log_data: dict[str, Any] = {
        "timestamp": datetime.datetime.now(
            tz=datetime.timezone.utc,
        ).isoformat(),
        "mode": args.mode,
        # Pipeline statistics (per spec)
        "universe_size": stats.get("universe_size", 0),
        "tier1_survivors": stats.get("tier1_survivors", 0),
        "shortlisted": stats.get("shortlisted", 0),
        "reports_generated": stats.get("reports_generated", 0),
        "runtime_seconds": round(elapsed_seconds, 2),
        # CLI parameters
        "top_n": args.top,
        "exchange": args.exchange,
        "skip_acquisition": args.skip_acquisition,
        "skip_metrics": args.skip_metrics,
        "no_cache": args.no_cache,
        "no_moat": args.no_moat,
        "verbose": args.verbose,
        "stages_run": stages_run,
        "report_files": (
            [str(p) for p in report_paths] if report_paths else []
        ),
        "status": "error" if error else "success",
    }
    if error:
        log_data["error"] = error

    log_path = report_dir / "run_log.json"
    log_path.write_text(
        json.dumps(log_data, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Run log written to %s", log_path)
    return log_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full pipeline based on parsed CLI arguments.

    Parameters
    ----------
    args:
        Parsed CLI namespace from :func:`build_arg_parser`.
    """
    start_time = time.monotonic()
    stages_run: list[str] = []
    report_paths: list[pathlib.Path] | None = None
    error_msg: str | None = None

    # Pipeline statistics for the run log (per spec)
    pipeline_stats: dict[str, int] = {
        "universe_size": 0,
        "tier1_survivors": 0,
        "shortlisted": 0,
        "reports_generated": 0,
    }

    configure_logging(verbose=args.verbose)
    logger.info("Pipeline starting (mode=%s)", args.mode)

    # Disable qualitative analysis if --no-moat
    if args.no_moat:
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        logger.info("Moat assessment disabled via --no-moat flag.")

    try:
        # --- Stage 1: Data Acquisition ---
        total_universe = 0
        if not args.skip_acquisition:
            acq_result = _run_stage_1(use_cache=not args.no_cache)
            total_universe = acq_result["universe_size"]
            pipeline_stats["universe_size"] = total_universe
            stages_run.append("data_acquisition")
        else:
            logger.info("Stage 1 skipped (--skip-acquisition).")
            # Read universe size from existing DuckDB data
            try:
                from data_acquisition.store import read_table as _read
                universe_df = _read("universe")
                total_universe = len(universe_df)
                pipeline_stats["universe_size"] = total_universe
            except Exception:
                pass

        # --- Stage 2: Metrics Engine ---
        composite_df = pd.DataFrame()
        if not args.skip_metrics:
            composite_df = _run_stage_2()
            pipeline_stats["tier1_survivors"] = len(composite_df)
            stages_run.append("metrics_engine")
        else:
            logger.info("Stage 2 skipped (--skip-metrics).")

        # --- Stage 3: Screening Pipeline ---
        shortlist_df, screener_summary = _run_stage_3(
            composite_df=composite_df,
            top_n=args.top,
            exchange_filter=args.exchange,
            total_universe=total_universe,
        )
        pipeline_stats["tier1_survivors"] = screener_summary.get(
            "after_tier1", pipeline_stats.get("tier1_survivors", 0),
        )
        pipeline_stats["shortlisted"] = len(shortlist_df)
        stages_run.append("screening")

        # --- Stage 4: Output ---
        if args.mode == "reports" and not shortlist_df.empty:
            report_paths = _run_stage_4_reports(
                shortlist_df, screener_summary,
            )
            pipeline_stats["reports_generated"] = (
                len(report_paths) if report_paths else 0
            )
            stages_run.append("reports")

            # Explicit print for CLI visibility (per spec)
            try:
                report_dir_str = str(get_threshold("output.report_dir"))
            except (KeyError, ValueError):
                report_dir_str = "data/reports"
            logger.info("Reports generated in %s/", report_dir_str)

        elif args.mode == "reports" and shortlist_df.empty:
            logger.warning(
                "No stocks in shortlist — skipping report generation. "
                "Check that API keys in .env are valid and not placeholders.",
            )
            stages_run.append("reports")

        elif args.mode == "dashboard":
            _run_stage_4_dashboard()
            stages_run.append("dashboard")

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("Pipeline failed: %s", exc)

    elapsed = time.monotonic() - start_time
    _write_run_log(
        args=args,
        elapsed_seconds=elapsed,
        stages_run=stages_run,
        report_paths=report_paths,
        pipeline_stats=pipeline_stats,
        error=error_msg,
    )

    if error_msg:
        logger.error("Pipeline finished with errors (%.1fs).", elapsed)
        sys.exit(1)
    else:
        logger.info("Pipeline finished successfully (%.1fs).", elapsed)


def main() -> None:
    """CLI entry point — parse args and run the pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

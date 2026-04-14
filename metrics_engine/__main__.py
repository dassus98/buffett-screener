"""Entry point for ``python -m metrics_engine``.

Runs the full Module 2 pipeline:
- Reads surviving tickers from DuckDB (written by Module 1).
- Computes all F1–F16 metrics for each ticker.
- Writes ``buffett_metrics``, ``buffett_metrics_summary``, and
  ``composite_scores`` tables back to DuckDB.
- Prints the top-20 ranked tickers to stdout.

Usage::

    python -m metrics_engine
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

from metrics_engine import run_metrics_engine  # noqa: E402 — logging must be configured first

if __name__ == "__main__":
    df = run_metrics_engine()
    if df.empty:
        print("\nMetrics engine produced no results. "
              "Ensure Module 1 (data_acquisition) has been run first.")
        sys.exit(1)

    print(f"\nMetrics engine complete — {len(df)} tickers scored.\n")
    cols = ["ticker", "composite_score"]
    print(df[cols].head(20).to_string(index=False))

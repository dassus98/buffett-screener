"""Entry point for ``python -m data_acquisition``.

Runs the full data acquisition pipeline end-to-end.

Usage
-----
    python -m data_acquisition               # Use cached universe/macro if fresh
    python -m data_acquisition --no-cache    # Force fresh fetch regardless of cache
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the data acquisition CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with the ``--no-cache`` flag.
    """
    parser = argparse.ArgumentParser(
        prog="python -m data_acquisition",
        description=(
            "Run the Buffett screener data acquisition pipeline. "
            "Fetches universe, financials, market data, and macro indicators, "
            "then runs data quality checks and persists all results to DuckDB."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Bypass cached universe (Parquet) and macro data (JSON); "
            "fetch fresh data from FMP, FRED, and yfinance."
        ),
    )
    return parser


def main() -> None:
    """Parse CLI arguments, configure logging, and run the pipeline.

    Exits with status code 1 if the pipeline raises an unhandled exception.
    """
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Import here so the module-level API key / config loading happens
    # after logging is configured, keeping startup messages visible.
    from data_acquisition import run_data_acquisition  # noqa: PLC0415

    try:
        result = run_data_acquisition(use_cache=not args.no_cache)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)

    print()
    print("Pipeline complete.")
    print(f"  Universe:  {result['universe_size']} tickers")
    print(f"  Survivors: {result['survivors']}")
    print(f"  Dropped:   {result['dropped']}")


if __name__ == "__main__":
    main()

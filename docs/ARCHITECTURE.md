# Architecture

> Module dependency map and data flow for buffett-screener.

## Module dependency order

```
schema.py
  └─► api_config.py
        └─► store.py
              └─► universe.py
              └─► financials.py
              └─► market_data.py
              └─► macro_data.py
  └─► data_quality.py

metrics_engine/*       ← consumes TickerDataBundle from data_acquisition
screener/*             ← consumes metrics DataFrame from metrics_engine
valuation_reports/*    ← consumes ranked DataFrame + bundles
output/*               ← orchestrates all of the above
```

No circular imports. Dependencies flow strictly downward.

## Data flow

```
[Universe fetch]
      │ list[str] tickers
      ▼
[Financial fetch]      [Market data fetch]      [Macro fetch]
      │                       │                       │
      └───────────────────────┴───────────────────────┘
                              │ TickerDataBundle
                              ▼
                    [Data quality checks]
                              │ valid bundles
                              ▼
                    [Metrics engine]
                              │ metrics DataFrame
                              ▼
                    [Screener]
                      exclusions → hard filters → soft scores → rank
                              │ ranked DataFrame (top N)
                              ▼
                    [Valuation reports]
                              │ ValuationReport per ticker
                              ▼
                    [Output: Streamlit / Markdown]
```

## Storage layer

DuckDB (`data/processed/buffett_screener.duckdb`) is the single persistence layer.
Tables: `universe_snapshots`, `income_statements`, `balance_sheets`,
`cash_flow_statements`, `market_data`, `macro_snapshots`, `data_quality_reports`, `metrics`.

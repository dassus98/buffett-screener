# Buffett Principles Stock Screener

An end-to-end system that screens North American public equities (TSX, NYSE, NASDAQ; ≥$500M USD market cap) using Warren Buffett's investment principles. Produces ranked shortlists and deep-dive analytical reports with intrinsic value estimates, margin of safety calculations, and buy/hold/pass recommendations.

## Canonical Sources
Principles and formulas are derived from:
- Berkshire Hathaway Shareholder Letters (1965–2025) — Warren Buffett
- *The Essays of Warren Buffett* (6th ed.) — Buffett, compiled by Cunningham
- *Buffettology* — Mary Buffett & David Clark
- *Warren Buffett and the Interpretation of Financial Statements* — Mary Buffett & David Clark
- *The Warren Buffett Way* (3rd ed.) — Robert Hagstrom
- *The Intelligent Investor* (rev. ed.) — Benjamin Graham
- *Security Analysis* (6th ed.) — Graham & Dodd

## Project Structure

```
buffett-screener/
├── CLAUDE.md                  # Persistent instructions — read on every task
├── README.md                  # This file — project overview and routing
├── config/
│   └── filter_config.yaml     # ALL thresholds, weights, API endpoints
├── docs/                      # Detailed reference — load per task
│   ├── FORMULAS.md            # F1–F16: exact formulas, line items, thresholds
│   ├── ARCHITECTURE.md        # Module breakdown, data flow, dependencies
│   ├── SCORING.md             # Composite score weights and justification
│   ├── DATA_SOURCES.md        # API strategy, rate limits, substitution rules
│   └── REPORT_SPEC.md         # Report template, account logic, sell signals
├── data_acquisition/          # Module 1: data fetching, normalization, storage
├── metrics_engine/            # Module 2: compute F1–F16 from financial data
├── screener/                  # Module 3: tiered filtering and ranking
├── valuation_reports/         # Module 4: intrinsic value, reports, recommendations
├── output/                    # Module 5: Streamlit dashboard + CLI runner
├── tests/                     # Unit, integration, data quality tests
└── data/                      # Local data store (gitignored)
```

## Reference Docs — Load by Task

| Document | Contains | Load when working on |
|---|---|---|
| docs/FORMULAS.md | All 16 Buffett formulas with exact definitions, inputs, thresholds | metrics_engine/, valuation_reports/ |
| docs/ARCHITECTURE.md | Module breakdown, data flow, interface contracts | Structural changes, new modules, debugging cross-module issues |
| docs/SCORING.md | Composite score weights with justification | screener/ |
| docs/DATA_SOURCES.md | API selection, rate limits, line-item mapping table, substitution rules | data_acquisition/ |
| docs/REPORT_SPEC.md | Deep-dive report template, RRSP/TFSA logic, sell signals | valuation_reports/, output/ |

## Build Order

| Phase | Module | Depends On | Key Docs |
|---|---|---|---|
| 1 | config/filter_config.yaml | Nothing | — |
| 2 | data_acquisition/schema.py | Config | DATA_SOURCES.md |
| 3 | data_acquisition/api_config.py | Config | DATA_SOURCES.md |
| 4 | data_acquisition/universe.py | Schema, API config | DATA_SOURCES.md |
| 5 | data_acquisition/financials.py | Schema, API config | DATA_SOURCES.md, FORMULAS.md |
| 6 | data_acquisition/market_data.py | Schema | DATA_SOURCES.md |
| 7 | data_acquisition/macro_data.py | API config | DATA_SOURCES.md |
| 8 | data_acquisition/data_quality.py | Schema | DATA_SOURCES.md |
| 9 | data_acquisition/store.py | All of Module 1 | ARCHITECTURE.md |
| 10 | metrics_engine/* | Module 1 complete | FORMULAS.md |
| 11 | screener/* | Module 2 complete | SCORING.md |
| 12 | valuation_reports/* | Modules 2–3 | FORMULAS.md, REPORT_SPEC.md |
| 13 | output/* | All modules | ARCHITECTURE.md, REPORT_SPEC.md |
| 14 | tests/* | All modules | FORMULAS.md |

## Quick Start
```bash
cp .env.example .env
# Add your API keys to .env
pip install -e .
python -m output.pipeline_runner --mode reports  # Generate markdown reports
python -m output.pipeline_runner --mode dashboard  # Launch Streamlit dashboard
```

## Key Design Decisions

- All thresholds in config/filter_config.yaml — zero hardcoded values
- All line-item mappings in data_acquisition/schema.py — single source of truth
- Missing data → NaN with logged reason — never silently imputed
- Banks, insurers, REITs excluded from screening (different financial structure)
- Currency normalized to USD for screening; final recommendations converted to CAD
- Maintenance CapEx approximated as Depreciation (known limitation, flagged in reports)

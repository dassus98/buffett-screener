# Buffett Principles Stock Screener

An end-to-end system that screens North American public equities (TSX, NYSE, NASDAQ; ≥ $500 M USD market cap) using Warren Buffett's investment principles. Produces ranked shortlists and deep-dive analytical reports with intrinsic value estimates, margin of safety calculations, and buy/hold/pass recommendations.

**Data flow:** APIs → DuckDB → 16 Buffett formulas → Tier 1/2 screening → 3-scenario valuation → Markdown reports + Streamlit dashboard

## Canonical Sources

Principles and formulas are derived from:

- Berkshire Hathaway Shareholder Letters (1965-2025) — Warren Buffett
- *The Essays of Warren Buffett* (6th ed.) — Buffett, compiled by Cunningham
- *Buffettology* — Mary Buffett & David Clark
- *Warren Buffett and the Interpretation of Financial Statements* — Mary Buffett & David Clark
- *The Warren Buffett Way* (3rd ed.) — Robert Hagstrom
- *The Intelligent Investor* (rev. ed.) — Benjamin Graham
- *Security Analysis* (6th ed.) — Graham & Dodd

## Installation

### Prerequisites

- Python 3.11 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/buffett-screener.git
cd buffett-screener

# Install in editable mode (includes all dependencies)
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys (see below)
```

### Required API Keys

Add these to your `.env` file:

| Key | Required | Source | Purpose |
|---|---|---|---|
| `FMP_API_KEY` | Yes | [financialmodelingprep.com](https://financialmodelingprep.com/) | Financial statements, market data, universe listing |
| `FRED_API_KEY` | Yes | [fred.stlouisfed.org](https://fred.stlouisfed.org/) | US Treasury yields, GoC bond yields, USD/CAD rates, CPI |
| `ANTHROPIC_API_KEY` | No | [anthropic.com](https://www.anthropic.com/) | LLM-assisted qualitative moat assessment (optional) |

## Usage

```bash
# Full pipeline — fetch data, compute metrics, screen, generate reports
python -m output.pipeline_runner --mode reports

# TSX-only analysis, top 20
python -m output.pipeline_runner --mode reports --exchange TSX --top 20

# Skip data acquisition (reuse existing data), regenerate reports
python -m output.pipeline_runner --mode reports --skip-acquisition --skip-metrics

# Launch interactive dashboard
python -m output.pipeline_runner --mode dashboard

# Full pipeline without LLM moat assessment
python -m output.pipeline_runner --mode reports --no-moat
```

### CLI Options

| Flag | Description |
|---|---|
| `--mode reports` | Generate Markdown analysis reports |
| `--mode dashboard` | Launch the Streamlit interactive dashboard |
| `--top N` | Number of top-ranked stocks in the shortlist (default: 50) |
| `--exchange {TSX,NYSE,NASDAQ,ALL}` | Filter universe to a single exchange (default: ALL) |
| `--skip-acquisition` | Skip Stage 1 (data fetching). Reuse existing DuckDB data |
| `--skip-metrics` | Skip Stage 2 (metrics engine). Reuse existing metrics tables |
| `--no-cache` | Force re-fetch from all APIs, ignoring cached data |
| `--no-moat` | Disable LLM-assisted qualitative moat assessment |
| `--verbose` | Set log level to DEBUG for all modules |

## Configuration

All thresholds, weights, and parameters live in a single file:

```
config/filter_config.yaml
```

This is the sole source of truth for every numeric constant in the codebase. To adjust the screener's behavior, edit this file — no code changes needed. Changes take effect on the next pipeline run.

### Key configurable sections

| Section | What it controls | Example parameters |
|---|---|---|
| `universe` | Screening scope | Exchanges, min market cap, history window |
| `exclusions` | Sector/industry exclusions | SIC codes, industry keywords, flags |
| `hard_filters` | Tier 1 pass/fail gates | Min profitable years, min ROE, max debt payoff |
| `soft_scores` | Tier 2 weighted scoring | 10 criteria with weights (must sum to 1.0) |
| `valuation` | 3-scenario DCF parameters | Growth multipliers, P/E caps, risk premiums, probabilities |
| `recommendations` | BUY/HOLD/PASS thresholds | Min MoS, min score, RRSP/TFSA logic |
| `sell_signals` | Per-security sell triggers | ROE floor, margin decline, max D/E |
| `output` | Report generation | Shortlist size, sensitivity grid, display currency |

## Output

A pipeline run produces the following artifacts:

| Path | Description |
|---|---|
| `data/reports/{TICKER}_analysis.md` | Individual deep-dive report per shortlisted ticker |
| `data/reports/summary.md` | Ranked summary table of all shortlisted securities |
| `data/reports/all_reports.md` | Combined export of all reports in a single file |
| `data/reports/run_log.json` | Execution metadata (runtime, stages, ticker counts) |
| `data/processed/buffett.duckdb` | Full database — queryable with DuckDB CLI or Python |
| `data/processed/data_quality_report.csv` | Data quality audit trail (coverage, substitutions, drops) |
| `data/pipeline.log` | Timestamped execution log |

## Architecture

```
buffett-screener/
├── CLAUDE.md                  # Persistent instructions — read on every task
├── README.md                  # This file
├── config/
│   └── filter_config.yaml     # ALL thresholds, weights, API endpoints
├── docs/                      # Detailed reference
│   ├── FORMULAS.md            # F1-F16: exact formulas, line items, thresholds
│   ├── ARCHITECTURE.md        # Module breakdown, data flow, dependencies
│   ├── SCORING.md             # Composite score weights and justification
│   ├── DATA_SOURCES.md        # API strategy, rate limits, substitution rules
│   └── REPORT_SPEC.md         # Report template, account logic, sell signals
├── data_acquisition/          # Module 1: data fetching, normalization, storage
├── metrics_engine/            # Module 2: compute F1-F16 from financial data
├── screener/                  # Module 3: tiered filtering and ranking
├── valuation_reports/         # Module 4: intrinsic value, reports, recommendations
├── output/                    # Module 5: Streamlit dashboard + CLI runner
├── tests/                     # Unit, integration, data quality tests
└── data/                      # Local data store (gitignored)
```

### Module overview

| Module | Stage | Responsibility |
|---|---|---|
| `data_acquisition` | 1 | Fetch universe, financials, market data, macro data from APIs. Normalize to canonical schema. Persist to DuckDB. Run data quality checks. |
| `metrics_engine` | 2 | Compute all 16 Buffett formulas (F1-F16) — profitability, leverage, returns, growth, owner earnings, capital efficiency, valuation. Produce composite scores. |
| `screener` | 3 | Apply sector/industry exclusions. Tier 1 hard filters (binary pass/fail). Tier 2 soft scores (weighted 0-100). Rank and shortlist. |
| `valuation_reports` | 4 | 3-scenario intrinsic value (bear/base/bull). Margin of safety. Sensitivity analysis. BUY/HOLD/PASS recommendations. RRSP/TFSA account logic. Sell signals. Markdown report generation. |
| `output` | 5 | CLI pipeline orchestrator. Streamlit interactive dashboard with filters, charts, and deep-dive views. |

### Reference docs

| Document | Contains | Load when working on |
|---|---|---|
| `docs/FORMULAS.md` | All 16 Buffett formulas with exact definitions, inputs, thresholds | `metrics_engine/`, `valuation_reports/` |
| `docs/ARCHITECTURE.md` | Module breakdown, data flow, interface contracts | Structural changes, new modules, debugging |
| `docs/SCORING.md` | Composite score weights with justification | `screener/` |
| `docs/DATA_SOURCES.md` | API selection, rate limits, line-item mapping, substitution rules | `data_acquisition/` |
| `docs/REPORT_SPEC.md` | Deep-dive report template, RRSP/TFSA logic, sell signals | `valuation_reports/`, `output/` |

## Known Limitations

- **Maintenance CapEx approximation.** Owner earnings use total CapEx as a proxy for maintenance CapEx. This is conservative — it overstates maintenance spending for growth companies, understating their true owner earnings.
- **Survivorship bias.** The screener only evaluates currently-listed companies. Delisted, bankrupt, or acquired firms are absent from the universe, which may overstate historical success rates.
- **Historical growth extrapolation.** EPS CAGR and growth projections extend historical trends forward. Mean reversion is not modeled — companies with unsustainably high growth rates may receive overly optimistic valuations.
- **Financial sector excluded.** Banks, insurers, and REITs are excluded from screening because their financial structures (leverage-dependent, reserve-based) require a fundamentally different analytical framework.
- **Qualitative moat assessment is supplementary.** When enabled (`--no-moat` omitted and `ANTHROPIC_API_KEY` set), the LLM-generated moat narrative is supplementary context. It should not be the sole basis for investment decisions.

## Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_integration_full.py -v

# Run with short tracebacks
pytest tests/ -v --tb=short
```

## Key Design Decisions

- All thresholds in `config/filter_config.yaml` — zero hardcoded values in source code
- All line-item mappings in `data_acquisition/schema.py` — single source of truth
- Missing data produces NaN with a logged reason — never silently imputed
- Banks, insurers, REITs excluded from screening (different financial structure)
- Currency normalized to USD for screening; final recommendations converted to CAD
- Maintenance CapEx approximated as Depreciation (known limitation, flagged in reports)

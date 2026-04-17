# Buffett Principles Stock Screener

An end-to-end system that screens North American public equities (NYSE, NASDAQ, TSX; market cap >= $500M USD) using Warren Buffett's investment principles. Produces ranked shortlists and deep-dive analytical reports with intrinsic value estimates, margin of safety calculations, and BUY/HOLD/PASS recommendations with RRSP/TFSA account guidance.

**Data flow:** yfinance/FRED APIs --> DuckDB --> 16 Buffett formulas --> Tier 1/2 screening --> 3-scenario valuation --> Markdown reports + Streamlit dashboard

## Canonical Sources

Principles and formulas are derived from:

- Berkshire Hathaway Shareholder Letters (1965-2025) -- Warren Buffett
- *The Essays of Warren Buffett* (6th ed.) -- Buffett, compiled by Cunningham
- *Buffettology* -- Mary Buffett & David Clark
- *Warren Buffett and the Interpretation of Financial Statements* -- Mary Buffett & David Clark
- *The Warren Buffett Way* (3rd ed.) -- Robert Hagstrom
- *The Intelligent Investor* (rev. ed.) -- Benjamin Graham
- *Security Analysis* (6th ed.) -- Graham & Dodd

## Installation

### Prerequisites

- Python 3.11 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/buffett-screener.git
cd buffett-screener

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in editable mode (includes all dependencies)
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys (see below)
```

### API Keys

Add these to your `.env` file:

| Key | Required | Source | Purpose |
|---|---|---|---|
| `FMP_API_KEY` | No | [financialmodelingprep.com](https://financialmodelingprep.com/) | Primary financial data API (free tier is rate-limited; yfinance fallback handles this automatically) |
| `FRED_API_KEY` | Recommended | [fred.stlouisfed.org](https://fred.stlouisfed.org/) | US Treasury yields, GoC bond yields, USD/CAD rates |
| `ANTHROPIC_API_KEY` | No | [anthropic.com](https://www.anthropic.com/) | LLM-assisted qualitative moat assessment (optional) |

> **Note:** The pipeline runs fully without any API keys. yfinance (no key required) serves as the primary data source for financial statements, market data, and the ticker universe. FRED provides macro data (bond yields, exchange rates); if unavailable, yfinance fallbacks and hardcoded defaults are used.

## Usage

### Full Pipeline

```bash
# Full pipeline -- fetch data, compute metrics, screen, generate reports
python -m output.pipeline_runner --mode reports

# Top 20 stocks only
python -m output.pipeline_runner --mode reports --top 20

# Full pipeline without LLM moat assessment
python -m output.pipeline_runner --mode reports --no-moat
```

### Incremental Runs

```bash
# Skip data acquisition (reuse existing DuckDB data), re-run metrics + screening + reports
python -m output.pipeline_runner --mode reports --skip-acquisition

# Skip both acquisition and metrics, re-run screening + reports only
python -m output.pipeline_runner --mode reports --skip-acquisition --skip-metrics

# Force re-fetch all data (ignore caches)
python -m output.pipeline_runner --mode reports --no-cache
```

### Streamlit Dashboard

```bash
# Launch the interactive dashboard (requires pipeline to have run at least once)
streamlit run output/streamlit_app.py
```

The dashboard has four tabs:

| Tab | Contents |
|---|---|
| **Ranked Results** | Sortable table of all shortlisted securities with scores, IV, price, MoS, recommendation, and account guidance |
| **Deep Dive** | Per-ticker analysis: score, IV, recommendation, 10-year financial charts (ROE, Gross Margin, EPS, Owner Earnings), 3-scenario valuation, sensitivity analysis |
| **Screener Summary** | Pipeline funnel (Universe --> Exclusions --> Tier 1 --> Shortlist --> Reports), sector/exchange distributions, composite score histogram |
| **Macro Context** | US 10-Year Treasury yield, GoC 10-Year bond yield, USD/CAD rate, market attractiveness assessment (earnings yield vs. bond yield spread) |

> **Important:** DuckDB allows only one write connection at a time. If the dashboard is running, stop it before running the pipeline, and vice versa.

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

This is the sole source of truth for every numeric constant in the codebase. To adjust the screener's behavior, edit this file -- no code changes needed. Changes take effect on the next pipeline run.

### Key configurable sections

| Section | What it controls | Example parameters |
|---|---|---|
| `universe` | Screening scope | Exchanges, min market cap ($500M), min history years (4) |
| `data_quality` | Data completeness gates | Min field coverage years (3) |
| `exclusions` | Sector/industry exclusions | Financial Services, Utilities, Real Estate, Banks, Insurance, REITs |
| `hard_filters` | Tier 1 pass/fail gates | Min profitable years (3), min ROE (15%), max debt payoff (5 yrs) |
| `soft_scores` | Tier 2 weighted scoring | 10 criteria with weights summing to 1.0 |
| `valuation` | 3-scenario DCF parameters | Max growth rate cap (25%), projection years (10), growth multipliers, P/E caps, risk premiums, scenario probabilities |
| `recommendations` | BUY/HOLD/PASS thresholds | Min MoS (25%), min score (70), RRSP/TFSA dividend threshold |
| `sell_signals` | Per-security sell triggers | ROE floor (12%), margin decline (5pp), max D/E (1.0) |
| `output` | Report generation | Shortlist size, sensitivity grid, display currency |

## Output

A pipeline run produces the following artifacts:

| Path | Description |
|---|---|
| `data/reports/{TICKER}_analysis.md` | Individual deep-dive report per shortlisted ticker |
| `data/reports/summary.md` | Ranked summary table with macro context and sector breakdown |
| `data/reports/all_reports.md` | Combined export of all reports in a single file |
| `data/reports/run_log.json` | Execution metadata: runtime, stages, ticker counts, report paths |
| `data/processed/buffett.duckdb` | Full DuckDB database -- queryable directly |
| `data/processed/data_quality_report.csv` | Data quality audit trail (coverage, substitutions, drops) |
| `data/pipeline.log` | Timestamped execution log |

## Architecture

```
buffett-screener/
├── CLAUDE.md                  # Persistent instructions -- read on every task
├── README.md                  # This file
├── config/
│   └── filter_config.yaml     # ALL thresholds, weights, API endpoints (single source of truth)
├── docs/                      # Detailed reference
│   ├── FORMULAS.md            # F1-F16: exact formulas, line items, thresholds
│   ├── ARCHITECTURE.md        # Module breakdown, data flow, dependencies
│   ├── SCORING.md             # Composite score weights and justification
│   ├── DATA_SOURCES.md        # API strategy, rate limits, substitution rules
│   └── REPORT_SPEC.md         # Report template, account logic, sell signals
├── data_acquisition/          # Stage 1: data fetching, normalization, storage
│   ├── universe.py            #   Ticker universe (FMP screener or Wikipedia + yfinance)
│   ├── financials.py          #   Financial statements (FMP or yfinance fallback)
│   ├── market_data.py         #   Current market data (price, P/E, dividend yield)
│   ├── macro_data.py          #   Macro indicators (FRED or yfinance fallback)
│   ├── schema.py              #   Canonical line-item mapping (14 fields, 3 statements)
│   ├── store.py               #   DuckDB persistence layer (11 tables)
│   ├── data_quality.py        #   Data completeness checks, drop decisions
│   └── api_config.py          #   Rate limiting, retry logic, API key management
├── metrics_engine/            # Stage 2: compute F1-F16 from financial data
│   ├── __init__.py            #   Orchestrator: runs all 16 formulas per ticker
│   ├── profitability.py       #   F7 (Gross Margin), F8 (SGA Ratio), F10 (Net Margin)
│   ├── returns.py             #   F2 (Initial Return), F3 (ROE), F4 (Retained Earnings Return)
│   ├── leverage.py            #   F5 (Debt Payoff), F6 (D/E Ratio), F9 (Interest Coverage)
│   ├── growth.py              #   F11 (EPS CAGR), F13 (Buyback Indicator)
│   ├── owner_earnings.py      #   F1 (Owner Earnings)
│   ├── capex.py               #   F12 (CapEx-to-Net-Earnings)
│   ├── valuation.py           #   F14 (Intrinsic Value), F15 (MoS), F16 (Yield Spread)
│   └── composite_score.py     #   10-criterion weighted composite scoring
├── screener/                  # Stage 3: tiered filtering and ranking
│   ├── exclusions.py          #   Sector/industry exclusion logic
│   ├── hard_filters.py        #   5 binary pass/fail Tier 1 gates
│   ├── soft_filters.py        #   Join composite scores onto survivors
│   ├── composite_ranker.py    #   Shortlist generation, summary statistics
│   └── filter_config_loader.py#   Config reader (get_threshold / get_config)
├── valuation_reports/         # Stage 4: intrinsic value, reports, recommendations
│   ├── intrinsic_value.py     #   Three-scenario valuation orchestrator
│   ├── margin_of_safety.py    #   MoS calculation and threshold checks
│   ├── earnings_yield.py      #   Earnings yield vs. bond yield assessment
│   ├── recommendation.py      #   BUY/HOLD/PASS, RRSP/TFSA, sell signals, entry strategy
│   ├── report_generator.py    #   Deep-dive context builder, Jinja2 rendering
│   ├── qualitative_prompts.py #   LLM moat assessment (optional)
│   └── templates/             #   Jinja2 Markdown templates
├── output/                    # Stage 5: CLI runner + interactive dashboard
│   ├── pipeline_runner.py     #   CLI orchestrator (argparse, stage sequencing)
│   ├── streamlit_app.py       #   4-tab interactive dashboard
│   ├── markdown_export.py     #   Combined report export
│   └── summary_table.py       #   Summary table generation
├── tests/                     # 1,234 tests: unit, integration, data quality
└── data/                      # Local data store (gitignored)
    ├── raw/                   #   Cached API responses (JSON, Parquet)
    ├── processed/             #   DuckDB database
    └── reports/               #   Generated reports and run logs
```

## Pipeline Stages

### Stage 1: Data Acquisition

Fetches all raw data from external APIs and persists to DuckDB.

```
Ticker Universe (FMP or Wikipedia + yfinance)
    --> Financial Statements (FMP or yfinance, 3 statements x N years)
    --> Market Data (yfinance: price, P/E, market cap, dividend yield)
    --> Macro Data (FRED: Treasury yield, GoC bond yield, USD/CAD rate)
    --> Data Quality Check (coverage, substitutions, drop decisions)
```

**Data source strategy:**
- **Primary:** FMP API (Financial Modeling Prep) for financial statements and universe listing.
- **Fallback:** yfinance (no API key required). Triggered automatically when FMP returns errors or is rate-limited. For the free-tier FMP key, yfinance handles most data fetching in practice.
- **Macro data:** FRED API for bond yields and exchange rates. yfinance fallbacks for US Treasury (`^TNX`) and USD/CAD (`CADUSD=X`).

**yfinance data characteristics:**
- Returns approximately 4-5 years of annual financial data.
- Oldest year is often a sparse stub (< 25% of fields populated) -- automatically dropped by the sparse row filter.
- Exchange codes are normalized: NMS/NGM/NCM --> NASDAQ, NYQ/NYS --> NYSE, TOR --> TSX.

**DuckDB tables written (Stage 1):**

| Table | Primary Key | Description |
|---|---|---|
| `universe` | ticker | Ticker, exchange, company name, market cap, sector, industry |
| `income_statement` | (ticker, fiscal_year) | Revenue, gross profit, operating income, net income, EPS, SGA, interest expense |
| `balance_sheet` | (ticker, fiscal_year) | Long-term debt, shareholders' equity, treasury stock |
| `cash_flow` | (ticker, fiscal_year) | D&A, capital expenditures, working capital change |
| `market_data` | ticker | Current price, trailing P/E, market cap, EV, dividend yield |
| `macro_data` | key | Key-value store: us_treasury_10yr, goc_bond_10yr, usd_cad_rate |
| `data_quality_log` | ticker | Years available, missing fields, substitution count, drop flag |
| `substitution_log` | (ticker, fiscal_year, field) | Field-level substitution audit trail with confidence |

### Stage 2: Metrics Engine

Computes all 16 Buffett formulas (F1-F16) for every surviving ticker.

- Reads Stage 1 tables from DuckDB.
- Selects GoC bond yield for TSX tickers, US Treasury for NYSE/NASDAQ.
- Runs formulas in dependency order (F1 --> F5, F11 --> F14 --> F15).
- Produces per-year detail and 10-year aggregates.
- Computes the 10-criterion weighted composite score (0-100).

**DuckDB tables written (Stage 2):**

| Table | Primary Key | Description |
|---|---|---|
| `buffett_metrics` | (ticker, fiscal_year) | Per-year: owner earnings, ROE, margins, leverage ratios |
| `buffett_metrics_summary` | ticker | 10-year aggregates + composite score (one row per ticker) |
| `composite_scores` | ticker | Ranked composite scores for all evaluated tickers |

### Stage 3: Screening Pipeline

Filters and ranks tickers through two tiers.

```
Universe (e.g. 403 tickers)
    --> Sector/Industry Exclusions (remove financials, utilities, REITs)
    --> Tier 1: 5 Hard Filters (binary pass/fail)
    --> Tier 2: 10-Criterion Composite Score (weighted 0-100)
    --> Shortlist: Top N by composite score (default 50)
```

### Stage 4: Report Generation

For each shortlisted ticker:
1. Reads composite score from DuckDB.
2. Computes 3-scenario intrinsic value (Bear/Base/Bull).
3. Calculates margin of safety.
4. Generates BUY/HOLD/PASS recommendation.
5. Determines RRSP vs. TFSA account suitability.
6. Evaluates 5 sell signal categories.
7. Renders a Markdown deep-dive report.

Also generates a portfolio summary table with macro context.

## Data Lineage

```
config/filter_config.yaml (single source of truth for all thresholds)
        |
        v
  +------------------+     +------------------+     +------------------+
  |   universe.py    |     |  financials.py   |     |  macro_data.py   |
  | FMP or Wiki+yf   |     | FMP or yfinance  |     |  FRED or yf      |
  +--------+---------+     +--------+---------+     +--------+---------+
           |                        |                        |
           v                        v                        v
     DuckDB: universe         DuckDB: income_stmt      DuckDB: macro_data
                              balance_sheet, cash_flow
           |                        |                        |
           +----------+-------------+------------------------+
                      |
                      v
            data_quality.py
            (coverage checks, drop decisions)
                      |
                      v
            DuckDB: data_quality_log
                      |
                      v
            metrics_engine/__init__.py
            (F1-F16 in dependency order)
                      |
                      v
            DuckDB: buffett_metrics, buffett_metrics_summary, composite_scores
                      |
                      v
            screener/ (exclusions --> hard filters --> soft scores --> shortlist)
                      |
                      v
            valuation_reports/ (3-scenario IV, MoS, recommendations, sell signals)
                      |
                      v
            output/ (Markdown reports, Streamlit dashboard, run_log.json)
```

## Screening Criteria

### Tier 1: Hard Filters (all must pass)

Binary pass/fail gates. Any single failure permanently excludes the ticker.

| Filter | Metric | Operator | Default Threshold | Rationale |
|---|---|---|---|---|
| **Earnings Consistency** | profitable_years | >= | 3 | Buffett requires a track record of profitability |
| **ROE Floor** | avg_roe (10-year average) | >= | 15% | Demonstrates durable competitive advantage |
| **EPS Growth** | eps_cagr | > | 0% (strictly positive) | Company must show real earnings growth |
| **Debt Sustainability** | debt_payoff_years | <= | 5 years | Debt repayable from owner earnings within 5 years |
| **Data Sufficiency** | years_available | >= | 4 years | Minimum data window for reliable analysis |

### Tier 2: Composite Score (10 weighted criteria)

Each criterion scores 0-100, then composited via weighted average. All weights sum to 1.0.

| # | Criterion | Weight | Scoring Logic |
|---|---|---|---|
| 1 | **ROE** | 0.15 | Linear 0% --> 0, 15% --> 50, 25%+ --> 100. Variance penalty: stdev > 5% subtracts 15 points. |
| 2 | **Gross Margin** | 0.10 | Breakpoints: 20% --> 0, 40% --> 70, 60%+ --> 100. |
| 3 | **SGA Ratio** | 0.08 | Lower is better. SGA < 30% of gross profit --> 100; >= 80% --> 0. |
| 4 | **EPS Growth** | 0.15 | CAGR breakpoints: 0% --> 0, 10% --> 50, 20%+ --> 100. Consistency multiplier penalizes decline years. |
| 5 | **Debt Conservatism** | 0.10 | Lower D/E is better. D/E < 0.2 --> 100; >= 0.8 --> 0. |
| 6 | **Owner Earnings Growth** | 0.12 | Same CAGR breakpoints as EPS Growth. Min 5 computable years required. |
| 7 | **Capital Efficiency** | 0.08 | CapEx/NI < 25% --> 100; >= 75% --> 0. |
| 8 | **Buyback** | 0.05 | Share dilution penalized; buybacks rewarded up to 15%+ --> 100. |
| 9 | **Retained Earnings Return** | 0.10 | 0% --> 0, 12% --> 70, 15%+ --> 100. Buffett's "$1 test." |
| 10 | **Interest Coverage** | 0.07 | Interest < 10% of EBIT --> 100; >= 30% --> 0. Debt-free companies score 100. |

Missing data (NaN) scores 0 for that criterion but its weight still applies, dragging the composite down. This penalizes companies with incomplete data.

## Valuation Model

### Three-Scenario Intrinsic Value (F14)

Projects current EPS forward 10 years under three scenarios, then discounts back to present value.

| Scenario | Growth Rate | Terminal P/E | Discount Rate | Probability |
|---|---|---|---|---|
| **Bear** | EPS CAGR x 0.5 | min(historical avg P/E, 12) | Risk-free + 5% | 25% |
| **Base** | EPS CAGR x 1.0 | median(historical P/E) | Risk-free + 3% | 50% |
| **Bull** | EPS CAGR x 1.3 | max(historical avg P/E, 20) | Risk-free + 2% | 25% |

**Growth rate cap:** Raw EPS CAGR is capped at 25% before scenario multipliers are applied. This prevents absurd intrinsic values when short data windows (4-5 years from yfinance) produce extreme CAGRs that compound unrealistically over a 10-year projection. Configurable via `valuation.max_growth_rate` in the YAML.

**Per-scenario computation:**
1. `projected_eps = current_eps x (1 + growth_rate)^10`
2. `projected_price = projected_eps x terminal_pe`
3. `present_value = projected_price / (1 + discount_rate)^10`

**Weighted IV** = Bear PV x 0.25 + Base PV x 0.50 + Bull PV x 0.25

### Margin of Safety (F15)

```
MoS = (Intrinsic Value - Current Price) / Intrinsic Value
```

- MoS >= 25% required for BUY recommendation.
- MoS >= 10% required for HOLD.
- Negative MoS (overvalued) is preserved as-is, not clamped.

### Earnings Yield vs. Bond Yield (F16)

```
Earnings Yield = EPS / Price
Spread = Earnings Yield - Risk-Free Rate
```

Equities are "attractive" when the spread exceeds 2% over the risk-free rate.

## Recommendation Logic

### Purchase Decision: BUY / HOLD / PASS

Evaluated in cascade order (first match wins):

| Recommendation | Conditions | Meaning |
|---|---|---|
| **BUY** | MoS >= 25% AND composite score >= 70 | Stock is undervalued with strong fundamentals |
| **HOLD** | MoS >= 10% AND composite score >= 60 | Fairly valued, worth holding if already owned |
| **PASS** | Everything else, OR any data quality flags, OR any sell signal triggered | Do not buy; either overvalued, weak fundamentals, or insufficient data |

**Sell signal override:** If ANY of the 5 sell signals is TRIGGERED, the recommendation is forcibly overridden to PASS regardless of MoS or composite score.

### Confidence Level

| Level | Conditions |
|---|---|
| **High** | >= 8 years of data, 0 substituted fields, MoS > 30% |
| **Moderate** | >= 8 years of data, <= 2 substitutions, MoS >= 15% |
| **Low** | Everything else |

Confidence affects position sizing guidance and time horizon:
- **High confidence + score >= 80:** 8-10% position, 15-year extended horizon.
- **Low confidence:** 1-2% position, 5-year shortened horizon.

### Sell Signals (5 categories)

Each signal has three statuses: **OK**, **WARNING** (within 20% of threshold), **TRIGGERED**.

| Signal | What It Monitors | Triggered When |
|---|---|---|
| **ROE Deterioration** | 10-year average ROE | ROE < 12% |
| **Gross Margin Erosion** | 3-year gross margin trend | Decline > 5 percentage points |
| **Leverage Spike** | D/E ratio and debt payoff years | D/E > 1.0 OR debt payoff > 5 years |
| **Overvaluation** | Current price vs. bull-case IV | Price exceeds the most optimistic scenario |
| **Capital Misallocation** | Return on retained earnings | RORE < 8% (the "$1 test" failing) |

## Account Decision: RRSP vs. TFSA

The screener recommends the optimal registered account for Canadian investors based on tax treaty considerations and expected return profiles.

### Decision Rules

| Condition | Recommendation | Rationale |
|---|---|---|
| **Canadian-listed** (TSX, .TO suffix) | **TFSA** | Canadian dividends and capital gains are fully tax-free in TFSA. RRSP offers no additional benefit and loses the dividend tax credit. |
| **US-listed, dividend yield >= 1%** | **RRSP** | The Canada-US tax treaty waives the 15% IRS withholding tax on US dividends in RRSP accounts. TFSA is not recognized as a pension account under the treaty, so US dividends are taxed at 15%. |
| **US-listed, dividend yield < 1%, expected return >= 12%** | **TFSA** | With minimal dividend exposure, the withholding tax impact is negligible. High-growth compounding makes TFSA's fully tax-free treatment on capital gains more valuable than RRSP's tax deferral. |
| **US-listed, dividend yield < 1%, expected return < 12%** | **Either** | Marginal benefit either way. Slight TFSA preference for growth orientation, but both accounts are suitable. |

### Why This Matters

- **RRSP advantage for US dividends:** Under Article XXI of the Canada-US Tax Treaty, US-source dividends in RRSP/RRIF are exempt from the standard 15% IRS withholding tax. TFSA is not a recognized pension vehicle, so US dividends are taxed regardless.
- **TFSA advantage for Canadian stocks and growth:** Canadian dividends qualify for the dividend tax credit (lost in RRSP). Capital gains in TFSA are permanently tax-free (RRSP only defers tax until withdrawal). For high-growth, low-dividend US stocks, the small withholding tax loss is outweighed by permanent tax-free compounding.

## The 16 Buffett Formulas

| ID | Formula | Module | Purpose |
|---|---|---|---|
| F1 | Owner Earnings | `owner_earnings.py` | Net Income + D&A - CapEx. Buffett's preferred cash flow measure. |
| F2 | Initial Rate of Return | `returns.py` | Owner earnings yield must exceed 2x the bond yield. |
| F3 | Return on Equity (ROE) | `returns.py` | 10-year average >= 15%. Measures management effectiveness. |
| F4 | Return on Retained Earnings | `returns.py` | Buffett's "$1 test" -- every $1 retained must create > $1 of market value. |
| F5 | Debt Payoff Test | `leverage.py` | Long-term debt repayable from owner earnings within 5 years. |
| F6 | Debt-to-Equity Ratio | `leverage.py` | Conservative balance sheet: D/E < 0.80. |
| F7 | Gross Margin | `profitability.py` | Durable competitive advantage indicator. Threshold: > 20%. |
| F8 | SGA-to-Gross-Profit Ratio | `profitability.py` | Operating efficiency. Lower is better; < 80% acceptable. |
| F9 | Interest Expense Coverage | `leverage.py` | Interest < 15% of EBIT. Tests debt servicing capacity. |
| F10 | Net Margin Consistency | `profitability.py` | Positive net margin in >= 80% of years. Earnings predictability. |
| F11 | EPS Growth Rate (CAGR) | `growth.py` | Compound annual EPS growth. Must be positive. Capped at 25% for projections. |
| F12 | CapEx-to-Net-Earnings | `capex.py` | Capital intensity. < 50% indicates a capital-light business. |
| F13 | Share Buyback Indicator | `growth.py` | Management returning capital via buybacks (not diluting). |
| F14 | Intrinsic Value (3-Scenario) | `intrinsic_value.py` | Bear/Base/Bull DCF with probability weighting. |
| F15 | Margin of Safety | `margin_of_safety.py` | (IV - Price) / IV. Requires >= 25% for BUY. |
| F16 | Earnings Yield vs. Bond Yield | `valuation.py` | Equities must offer a spread over risk-free bonds. |

## Testing

```bash
# Run the full test suite (1,234 tests)
pytest tests/ -v

# Run a specific test module
pytest tests/test_formulas.py -v

# Run with short tracebacks
pytest tests/ -q --tb=short
```

## Key Design Decisions

- **Single config file:** All thresholds live in `config/filter_config.yaml` -- zero hardcoded values in source code.
- **Canonical schema:** All line-item mappings defined in `data_acquisition/schema.py` with a prioritized substitution chain.
- **Missing data transparency:** Missing data produces NaN with a logged reason -- never silently imputed.
- **Financial sector exclusion:** Banks, insurers, and REITs are excluded (leverage-dependent structures require a different analytical framework).
- **Currency normalization:** All screening done in USD. Final recommendations convert to CAD where appropriate.
- **Growth rate cap:** EPS CAGR is capped at 25% for IV projections to prevent unrealistic extrapolation from short data windows.
- **Graceful fallbacks:** FMP --> yfinance for financials/universe, FRED --> yfinance for macro data. Each ticker's failure is isolated -- one error does not abort the batch.
- **Maintenance CapEx approximation:** Owner earnings use total D&A as a proxy for maintenance CapEx. This is conservative -- it overstates maintenance spending for growth companies, understating their true owner earnings.

## Known Limitations

- **4-5 year data window (yfinance).** yfinance provides approximately 4-5 years of annual data, compared to the 10+ years available from premium API providers. Thresholds have been adjusted accordingly (min history years = 4, min profitable years = 3). The growth rate cap mitigates the risk of extreme CAGR extrapolation from short windows.
- **Survivorship bias.** The screener only evaluates currently-listed companies. Delisted, bankrupt, or acquired firms are absent from the universe.
- **Maintenance CapEx approximation.** Owner earnings use total CapEx as a proxy for maintenance CapEx. This is conservative for growth companies.
- **No TSX coverage without FMP.** The Wikipedia + yfinance fallback for universe construction currently scrapes S&P 500 tickers reliably, but TSX coverage depends on successful Wikipedia scraping which may be fragile.
- **GoC bond yield requires FRED API key.** Without a FRED key, TSX valuations use a hardcoded 4% fallback risk-free rate instead of the actual Government of Canada 10-year bond yield.
- **Qualitative moat assessment is supplementary.** When enabled (`ANTHROPIC_API_KEY` set and `--no-moat` not passed), the LLM-generated moat narrative provides context but should not be the sole basis for investment decisions.
- **DuckDB single-writer constraint.** Only one process can write to DuckDB at a time. Stop Streamlit before running the pipeline.

## Disclaimer

This tool is for educational and research purposes only. It is not financial advice. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

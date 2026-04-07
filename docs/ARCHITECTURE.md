# ARCHITECTURE.md — System Architecture Reference

This is the structural reference document for buffett-screener. Load this when making structural
changes, adding new modules, or debugging cross-module issues.

---

## 1. System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATION LAYER                                │
│                     output/pipeline_runner.py (CLI)                         │
│           python -m output.pipeline_runner --mode reports|dashboard         │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ invokes in sequence
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  MODULE 1 — DATA ACQUISITION         data_acquisition/                    │
│                                                                           │
│  universe.py ──► financials.py ──► market_data.py ──► macro_data.py      │
│                          │                                                │
│                  data_quality.py                                          │
│                          │                                                │
│                      store.py                                             │
│                          │                                                │
│             ┌────────────▼────────────┐                                   │
│             │   DuckDB + Parquet      │  ◄── config/filter_config.yaml   │
│             │  data/processed/        │  ◄── data_acquisition/schema.py  │
│             │  buffett.duckdb         │                                   │
│             └────────────┬────────────┘                                   │
└──────────────────────────│────────────────────────────────────────────────┘
                           │ reads universe, income_statement,
                           │ balance_sheet, cash_flow, market_data,
                           │ macro_data, data_quality_log
                           ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  MODULE 2 — METRICS ENGINE           metrics_engine/                      │
│                                                                           │
│  owner_earnings.py (F1)    returns.py (F2, F4)                            │
│  profitability.py (F3,F7,F8,F10)     leverage.py (F5, F6, F9)            │
│  growth.py (F11, F13)      capex.py (F12)                                 │
│  valuation.py (F14, F15, F16)        composite_score.py                   │
│                          │                                                │
│             ┌────────────▼────────────┐                                   │
│             │   DuckDB                │                                   │
│             │  buffett_metrics        │                                   │
│             │  buffett_metrics_summary│                                   │
│             └────────────┬────────────┘                                   │
└──────────────────────────│────────────────────────────────────────────────┘
                           │ reads buffett_metrics_summary
                           ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  MODULE 3 — SCREENER & RANKING       screener/                            │
│                                                                           │
│  exclusions.py ──► hard_filters.py ──► soft_filters.py                   │
│                                              │                            │
│                                    composite_ranker.py                    │
│                          │                                                │
│             ┌────────────▼────────────┐                                   │
│             │   DuckDB                │                                   │
│             │  screener_results       │                                   │
│             │  shortlist              │                                   │
│             └────────────┬────────────┘                                   │
└──────────────────────────│────────────────────────────────────────────────┘
                           │ reads shortlist + full metrics
                           ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  MODULE 4 — VALUATION & REPORTS      valuation_reports/                   │
│                                                                           │
│  intrinsic_value.py (F14)  margin_of_safety.py (F15)                     │
│  earnings_yield.py (F16)   qualitative_prompts.py                         │
│  recommendation.py         report_generator.py                            │
│                          │                                                │
│             ┌────────────▼────────────┐                                   │
│             │  data/reports/          │                                   │
│             │  {TICKER}_analysis.md   │                                   │
│             │  summary.md             │                                   │
│             └────────────┬────────────┘                                   │
└──────────────────────────│────────────────────────────────────────────────┘
                           │ reads DuckDB + data/reports/
                           ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  MODULE 5 — OUTPUT LAYER             output/                              │
│                                                                           │
│  streamlit_app.py (interactive dashboard)                                 │
│  markdown_export.py (batch report generation)                             │
│  summary_table.py (portfolio-level summary)                               │
└───────────────────────────────────────────────────────────────────────────┘
```

**Shared infrastructure (accessed by all modules):**
- `config/filter_config.yaml` — all thresholds, weights, parameters (read via `filter_config_loader`)
- `data_acquisition/schema.py` — canonical column names and line-item mapping (single source of truth)
- `data/processed/buffett.duckdb` — single persistence layer for all structured data

---

## 2. Module 1 — Data Acquisition (`data_acquisition/`)

**Purpose:** Pull the full eligible universe and 10 years of annual financial statements, normalize
to canonical schema, store in DuckDB + Parquet.

### Files and Responsibilities

#### `schema.py`
The single source of truth for all field name references downstream. Defines:
- Canonical dataclass schemas: `IncomeStatement`, `BalanceSheet`, `CashFlowStatement`,
  `MarketData`, `MacroSnapshot`, `CompanyProfile`, `TickerDataBundle`
- Line-item mapping table: ideal field → acceptable substitutes → substitution confidence
  (High/Medium/Low) → drop-if-missing flag
- All monetary values stored in USD thousands (exception: `market_cap` and `enterprise_value`
  in full USD dollars — documented in schema)
- Sign convention: `capital_expenditures` always stored negative; negate + warn if source
  returns positive

No downstream module may reference a financial field name that is not defined here. If an API
returns a field not in the mapping, add it to `schema.py` with a documented substitution
confidence level before using it.

#### `api_config.py`
- Load API keys from `.env` via `python-dotenv`
- Rate limiter: configurable requests/minute (read from `filter_config.yaml`)
- Retry decorator: exponential backoff on all API calls (max retries configurable)
- Request logging: log every outbound API call at DEBUG level

#### `universe.py`
Fetch all tickers meeting eligibility criteria.

**Output:** `DataFrame` with columns:

| Column | Type | Notes |
|---|---|---|
| `ticker` | str | Exchange-native symbol (e.g., `SHOP.TO` for TSX) |
| `exchange` | str | `NYSE`, `NASDAQ`, `AMEX`, `TSX` |
| `company_name` | str | |
| `market_cap_usd` | float | Full USD dollars |
| `sector` | str | GICS sector |
| `industry` | str | GICS industry |
| `sic_code` | str | Used for sector exclusion checks |

**Eligibility filters applied here:**
- Market cap ≥ $500M USD (threshold from config)
- Exchanges: configurable list in `filter_config.yaml` under `universe.exchanges`
- TSX tickers: `.TO` suffix appended for yfinance compatibility
- Sector exclusions (Financials, Utilities, Real Estate) applied downstream in `exclusions.py`

#### `financials.py`
For each ticker in universe, pull annual income statement, balance sheet, and cash flow statement
for the most recent 10 fiscal years.

**Output:** 3 DataFrames normalized to canonical schema (field names from `schema.py`):
- `income_statement`: one row per (ticker, fiscal_year)
- `balance_sheet`: one row per (ticker, fiscal_year)
- `cash_flow_statement`: one row per (ticker, fiscal_year)

**Key behaviors:**
- All values in USD thousands (divide yfinance full-dollar values by 1000)
- Missing line items → `NaN` with logged reason (never zero-filled)
- Label instability: `INCOME_LABELS`, `BALANCE_LABELS`, `CASHFLOW_LABELS` dicts with
  prioritized candidate lists; `_get_field()` helper tries each in order, logs which was used
- Every substitution logged: original field, substitute used, confidence level, ticker

#### `market_data.py`
Current price and trading data for each universe ticker.

**Output:** `DataFrame` keyed by ticker:

| Column | Type | Notes |
|---|---|---|
| `ticker` | str | |
| `current_price_usd` | float | |
| `shares_outstanding` | float | Full count |
| `market_cap_usd` | float | Full USD dollars |
| `enterprise_value_usd` | float | Full USD dollars |
| `52w_high` | float | |
| `52w_low` | float | |
| `avg_daily_volume` | float | |
| `as_of_date` | date | Fetch date |

#### `macro_data.py`
Current macroeconomic values used in F16 (earnings yield vs bond yield).

**Output:** Dictionary of current macro values:

| Key | Source | Used in |
|---|---|---|
| `us_treasury_10yr` | FRED (`GS10`) | F16 for US securities |
| `goc_bond_10yr` | FRED (`IRLTLT01CAM156N`) or Bank of Canada | F16 for TSX securities |
| `cpi_yoy` | FRED (`CPIAUCSL`) | Informational |
| `usd_cad_rate` | yfinance (`CADUSD=X`) | CAD conversion in recommendations |

#### `data_quality.py`
After data ingestion, run completeness and integrity checks.

**Checks performed:**

| Check | Severity | Action |
|---|---|---|
| Ticker has < 8 of 10 fiscal years | ERROR | Add to drop list |
| Required line item is null for all years | ERROR | Add to drop list |
| Required line item null for some years | WARNING | Flag; NaN propagates |
| Line-item substitution used | WARNING | Log substitution details |
| CapEx returned as positive by source | WARNING | Negate and log |
| Sector in exclusion list | INFO | Route to exclusions.py |

**Output:** `data_quality_log` table with columns:
`ticker`, `check_name`, `severity`, `detail`, `action_taken`, `run_timestamp`

**Drop list:** Tickers failing ERROR-level checks are written to `data_quality_log` with
`action_taken = 'dropped'`. `store.py` excludes these from the processed tables.

#### `store.py`
Write processed DataFrames to DuckDB and optionally to Parquet.

**DuckDB path:** `data/processed/buffett.duckdb`
**Parquet path:** `data/processed/parquet/{table_name}/` (optional, controlled by config)

**Upsert pattern:** Delete-then-insert (DuckDB `ON CONFLICT` syntax is version-sensitive).
Delete rows matching conflict columns, then insert from staging DataFrame.

### Module 1 Interface Contract

**Output:** DuckDB database at `data/processed/buffett.duckdb` with the following tables:

| Table | Primary Key | Description |
|---|---|---|
| `universe` | `ticker` | All eligible tickers after eligibility filters |
| `income_statement` | `(ticker, fiscal_year)` | 10yr annual income statements |
| `balance_sheet` | `(ticker, fiscal_year)` | 10yr annual balance sheets |
| `cash_flow` | `(ticker, fiscal_year)` | 10yr annual cash flow statements |
| `market_data` | `ticker` | Current price and trading data |
| `macro_data` | `as_of_date` | Current macro values |
| `data_quality_log` | `(ticker, check_name, run_timestamp)` | All quality check results |

All column names in these tables correspond exactly to canonical field names in `schema.py`.

---

## 3. Module 2 — Metrics Engine (`metrics_engine/`)

**Purpose:** Compute all 16 Buffett formulas from canonical financial data. Output one row per
ticker per year, plus a 10-year summary row per ticker.

### Files and Responsibilities

| File | Formulas | Key outputs |
|---|---|---|
| `owner_earnings.py` | F1 | `f1_owner_earnings` (absolute $), `f1_owner_earnings_yield` |
| `returns.py` | F2, F4 | `f2_initial_return`, `f4_return_on_retained_earnings` |
| `profitability.py` | F3, F7, F8, F10 | `f3_roe`, `f7_gross_margin`, `f8_sga_ratio`, `f10_net_margin_consistency` |
| `leverage.py` | F5, F6, F9 | `f5_debt_payoff_years`, `f6_debt_equity`, `f9_interest_coverage` |
| `growth.py` | F11, F13 | `f11_eps_cagr`, `f13_buyback_indicator` |
| `capex.py` | F12 | `f12_capex_to_earnings` |
| `valuation.py` | F14, F15, F16 | `f14_iv_bear`, `f14_iv_base`, `f14_iv_bull`, `f14_iv_weighted`, `f15_margin_of_safety`, `f16_earnings_yield_spread` |
| `composite_score.py` | — | `composite_score` (0–100 percentile-normalised weighted sum) |

### Key Behaviors

- All formulas read thresholds from `config/filter_config.yaml` via `filter_config_loader`
- Missing inputs → metric = `NaN`; log at WARNING with ticker and formula ID
- Edge cases documented in `docs/FORMULAS.md` (e.g., negative equity → F3/F6 = NaN, don't drop)
- F14 uses 3 scenarios: Bear (25% weight), Base (50% weight), Bull (25% weight) — weights from config
- F16 uses `goc_bond_10yr` for TSX tickers; `us_treasury_10yr` for US tickers
- `composite_score.py` percentile-ranks each metric across all passing-universe tickers, then
  applies weights from config — never hardcoded

### Module 2 Interface Contract

**Reads from:** DuckDB tables produced by Module 1 (all 7 tables)

**Writes to:** DuckDB:

| Table | Primary Key | Description |
|---|---|---|
| `buffett_metrics` | `(ticker, fiscal_year)` | F1–F16 values per ticker per year |
| `buffett_metrics_summary` | `ticker` | 10-year aggregates: means, medians, CAGRs, pass/fail per hard filter, composite score |

**`buffett_metrics` columns (partial):**
`ticker`, `fiscal_year`, `f1_owner_earnings`, `f2_initial_return`, `f3_roe`,
`f4_return_on_retained_earnings`, `f5_debt_payoff_years`, `f6_debt_equity`,
`f7_gross_margin`, `f8_sga_ratio`, `f9_interest_coverage`, `f10_net_margin_consistency`,
`f11_eps_cagr`, `f12_capex_to_earnings`, `f13_buyback_indicator`, `f14_iv_bear`,
`f14_iv_base`, `f14_iv_bull`, `f14_iv_weighted`, `f15_margin_of_safety`,
`f16_earnings_yield_spread`, `composite_score`

---

## 4. Module 3 — Screener & Ranking (`screener/`)

**Purpose:** Apply Buffett criteria as tiered filters, rank survivors by composite score.

### Files and Responsibilities

#### `exclusions.py`
First gate. Remove sectors/industries that Buffett's framework does not apply to.

Excluded sectors (from config `screener.excluded_sectors`):
- Financials (banks, insurers — different capital structure)
- Utilities (regulated returns, not moat-driven)
- Real Estate / REITs (FFO-based, not EPS-based)

Also excludes: SPACs, shell companies (flagged via SIC code or company name patterns in config).

#### `hard_filters.py`
Tier 1: Binary pass/fail. A ticker failing any hard filter is dropped from ranking entirely.

Hard filters correspond to minimum quality thresholds in `filter_config.yaml`:
- Profitability: ≥ N of 10 years profitable, gross margin ≥ X%, operating margin ≥ Y%
- Returns: ROIC ≥ X%, ROE ≥ Y%
- Leverage: D/E ≤ X, net debt/EBITDA ≤ Y, interest coverage ≥ Z
- Cash flow: ≥ N years positive FCF, FCF margin ≥ X%
- Liquidity: current ratio ≥ X

All thresholds from config. No hardcoded numbers.

#### `soft_filters.py`
Tier 2: Scored 0–100 per criterion. All tickers that passed Tier 1 are scored here.

Produces per-ticker, per-criterion scores that feed `composite_ranker.py`. Scores are
percentile-normalised across the passing universe (not absolute).

#### `composite_ranker.py`
Applies weights from config to per-criterion scores, produces final `composite_score` and rank.
Outputs the top N tickers (N from config `screener.shortlist_size`) to the `shortlist` table.

#### `filter_config_loader.py`
Loads and validates `config/filter_config.yaml`. Raises `ValueError` on missing required keys.
All other modules import config exclusively through this loader — never via direct `yaml.safe_load`.

### Module 3 Interface Contract

**Reads from:** `buffett_metrics_summary` table in DuckDB

**Writes to:** DuckDB:

| Table | Primary Key | Description |
|---|---|---|
| `screener_results` | `ticker` | All post-exclusion tickers with tier1_pass flag, tier2 score per criterion, composite_score, rank |
| `shortlist` | `ticker` | Top N tickers selected for deep-dive valuation reports |

**`screener_results` columns:**
`ticker`, `exchange`, `company_name`, `sector`, `tier1_pass`, `tier2_roic_score`,
`tier2_owner_earnings_score`, `tier2_roe_score`, `tier2_gross_margin_score`,
`tier2_revenue_cagr_score`, `tier2_fcf_margin_score`, `tier2_debt_safety_score`,
`tier2_earnings_yield_score`, `composite_score`, `rank`

---

## 5. Module 4 — Valuation & Reports (`valuation_reports/`)

**Purpose:** For each shortlisted security, compute 3-scenario intrinsic value, generate
structured analytical reports with buy/hold/pass recommendations.

### Files and Responsibilities

#### `intrinsic_value.py`
Implements F14: two-stage DCF with three scenarios.

| Scenario | Weight | Growth mult. | Terminal P/E | Discount rate |
|---|---|---|---|---|
| Bear | 25% | 0.5× historical CAGR | min(avg P/E, 12) | risk-free + 5% |
| Base | 50% | 1.0× historical CAGR | median P/E | risk-free + 3% |
| Bull | 25% | 1.3× historical CAGR | max(avg P/E, 20) | risk-free + 2% |

Scenario weights and multipliers from config. Risk-free rate from `macro_data` table.

#### `margin_of_safety.py`
Implements F15. Computes MoS for each scenario and for the weighted average IV.

Negative MoS reported as negative (security is overvalued) — never floor at zero.

#### `earnings_yield.py`
Implements F16. Computes earnings yield spread vs. appropriate bond yield.
TSX tickers → GoC 10yr; US tickers → US Treasury 10yr. Exchange determined from `universe` table.

#### `qualitative_prompts.py`
Optional LLM-assisted moat assessment. Generates structured prompts for:
- Competitive moat type (brand, switching cost, network effect, cost advantage, efficient scale)
- Management quality indicators
- Industry tailwinds/headwinds

Disabled by default (config `reports.enable_qualitative: false`). No API calls made if disabled.

#### `recommendation.py`
Produces final buy/hold/pass recommendation based on:
- F15 Margin of Safety vs. thresholds in config (conservative/moderate/aggressive)
- F16 earnings yield spread vs. bonds
- Composite score rank
- Account type suitability: RRSP (US dividend stocks), TFSA (Canadian growth stocks)
- Entry price range (based on MoS thresholds)
- Sell signals: MoS < 0%, composite score drops below config threshold, dividend cut

#### `report_generator.py`
Renders Jinja2 templates to markdown.
- Template: `valuation_reports/templates/deep_dive_template.md`
- Output: `data/reports/{TICKER}_analysis.md`
- Summary template: `valuation_reports/templates/summary_table_template.md`
- Summary output: `data/reports/summary.md`

### Module 4 Interface Contract

**Reads from:**
- DuckDB: `shortlist`, `buffett_metrics`, `buffett_metrics_summary`, `market_data`, `macro_data`,
  `universe`, `screener_results`
- Templates: `valuation_reports/templates/`

**Writes to:**
- `data/reports/{TICKER}_analysis.md` — one deep-dive report per shortlisted ticker
- `data/reports/summary.md` — ranked summary table of all shortlisted tickers

---

## 6. Module 5 — Output Layer (`output/`)

**Purpose:** Deliver results in chosen format via CLI or interactive dashboard.

### Files and Responsibilities

#### `pipeline_runner.py`
CLI entry point. Orchestrates the full pipeline in sequence:
1. `data_acquisition` — fetch + store
2. `metrics_engine` — compute F1–F16
3. `screener` — filter + rank
4. `valuation_reports` — IV + reports (for shortlist only)

**CLI usage:**
```bash
python -m output.pipeline_runner --mode reports|dashboard [--top N] [--exchange TSX|NYSE|NASDAQ|ALL]
```

Options:
- `--mode reports`: Run full pipeline and generate markdown reports
- `--mode dashboard`: Run full pipeline and launch Streamlit
- `--top N`: Override shortlist size (default from config)
- `--exchange`: Filter universe to one exchange (default: ALL configured exchanges)

#### `streamlit_app.py`
Interactive dashboard. Reads DuckDB directly for all filtering and display.
Does not re-run pipeline computations — display only.

Features:
- Universe browser with sortable metrics
- Hard filter toggle (show all vs. passing only)
- Composite score ranking table
- Per-ticker metric detail panel
- Valuation summary with IV scenarios and MoS gauge

#### `markdown_export.py`
Batch generation of all reports in `data/reports/`. Called by `pipeline_runner.py` in
`--mode reports`. Wraps `report_generator.py` for each ticker in the shortlist.

#### `summary_table.py`
Generates `data/reports/summary.md`: a ranked table of the top N shortlisted tickers with
key metrics inline (composite score, MoS, F16 spread, recommendation).

### Module 5 Interface Contract

**Reads from:**
- DuckDB: all tables
- `data/reports/`: all markdown files

**Writes to:**
- `data/reports/` (via `markdown_export.py`)
- Terminal / Streamlit UI (no DuckDB writes in Module 5)

---

## 7. Data Flow Summary

```
External APIs (yfinance, FRED, SEC EDGAR)
        │
        ▼
data_acquisition/          Fetch → normalize → quality-check → store
        │
        ▼
DuckDB: universe, income_statement, balance_sheet, cash_flow,
        market_data, macro_data, data_quality_log
        │
        ▼
metrics_engine/            Compute F1–F16 per ticker per year
        │
        ▼
DuckDB: buffett_metrics, buffett_metrics_summary
        │
        ▼
screener/                  Exclude → hard filter → soft score → rank
        │
        ▼
DuckDB: screener_results, shortlist
        │
        ▼
valuation_reports/         IV scenarios → MoS → recommendation → render
        │
        ▼
data/reports/{TICKER}_analysis.md, data/reports/summary.md
        │
        ▼
output/                    CLI reports or Streamlit dashboard
```

---

## 8. Error Propagation

Errors flow downstream as `NaN` values, never as silent defaults or zero-fills.

| Stage | Error type | Propagation |
|---|---|---|
| Module 1 — fetch | API failure | Log ERROR; ticker's rows absent from DuckDB |
| Module 1 — parse | Missing line item | Field = `NaN`; log WARNING with substitution details |
| Module 1 — quality | < 8 years data | Ticker added to drop list; excluded from all downstream tables |
| Module 2 — formula | NaN input | Metric = `NaN`; log WARNING with formula ID and ticker |
| Module 2 — formula | Edge case (e.g. negative equity) | Metric = `NaN`; flag set; ticker not dropped (see FORMULAS.md) |
| Module 3 — hard filter | NaN metric required for filter | Ticker fails that filter; dropped from shortlist |
| Module 3 — soft score | NaN metric used in scoring | Score = `NaN` for that criterion; weighted sum excludes it |
| Module 4 — IV | NaN EPS CAGR | IV = `NaN`; report section flagged "insufficient data" |

**No module silently swallows upstream errors.** If a module receives a `NaN` where it needs a
real value, it propagates `NaN` and logs at WARNING. If it receives an absent ticker (not in
DuckDB), it logs at ERROR.

---

## 9. Import Dependency Rules

Dependencies flow strictly downward. No circular imports.

```
schema.py                    ← no internal imports
api_config.py                ← schema
universe.py                  ← schema, api_config
financials.py                ← schema, api_config
market_data.py               ← schema, api_config
macro_data.py                ← api_config
data_quality.py              ← schema
store.py                     ← schema, all fetchers above

metrics_engine/*             ← schema, store (reads DuckDB), filter_config_loader
screener/*                   ← filter_config_loader (reads DuckDB)
valuation_reports/*          ← filter_config_loader, schema (reads DuckDB)
output/*                     ← all modules above
```

`filter_config_loader` is the only permitted path to `config/filter_config.yaml`. Direct
`yaml.safe_load` calls in business logic are prohibited (see CLAUDE.md).

---

## 10. DuckDB Table Reference

All tables reside in `data/processed/buffett.duckdb`.

| Table | Written by | Read by | Notes |
|---|---|---|---|
| `universe` | Module 1 `store.py` | Modules 2, 3, 4, 5 | Post-eligibility-filter tickers |
| `income_statement` | Module 1 `store.py` | Module 2 | 10yr annual, USD thousands |
| `balance_sheet` | Module 1 `store.py` | Module 2 | 10yr annual, USD thousands |
| `cash_flow` | Module 1 `store.py` | Module 2 | 10yr annual, USD thousands |
| `market_data` | Module 1 `store.py` | Modules 2, 4, 5 | Current snapshot, USD full dollars |
| `macro_data` | Module 1 `store.py` | Modules 2, 4 | Current macro values |
| `data_quality_log` | Module 1 `data_quality.py` | Module 5 (dashboard) | All check results + drop list |
| `buffett_metrics` | Module 2 | Modules 3, 4, 5 | F1–F16 per ticker per year |
| `buffett_metrics_summary` | Module 2 | Modules 3, 4, 5 | 10-year aggregates per ticker |
| `screener_results` | Module 3 | Modules 4, 5 | All post-exclusion tickers + scores |
| `shortlist` | Module 3 | Modules 4, 5 | Top N tickers for deep-dive |

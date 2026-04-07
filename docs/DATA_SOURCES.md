# DATA_SOURCES.md — Data Source Strategy, API Configuration, and Line-Item Mapping

This document specifies every external data source used by the pipeline, how to configure
API credentials, rate limits, the canonical line-item mapping table, substitution rules,
currency normalization, and data quality protocol.

Load this file when working on `data_acquisition/`.

---

## 1. Data Source Selection

| Data Need | Primary Source | Fallback | Notes |
|---|---|---|---|
| Universe listing | Financial Modeling Prep (FMP) `/stock-screener` | yfinance screener | FMP provides clean ticker lists with market cap, sector, SIC code, and exchange in a single call. Recommended tier: FMP Starter ($29/mo) for build phase — free tier (250 req/day) is impractical for 8,000+ tickers. |
| Financial statements (10yr annual) | FMP `/income-statement`, `/balance-sheet-statement`, `/cash-flow-statement` | yfinance `Ticker.financials` / `Ticker.balance_sheet` / `Ticker.cashflow` | FMP returns standardized JSON with consistent field names. Store raw API responses in `data/raw/{ticker}/` before transformation. yfinance is lower quality for Canadian equities (TSX coverage gaps). |
| Current market data | yfinance `fast_info` + `info` (free, no hard rate limit) | FMP `/quote` | yfinance is sufficient for current price, shares outstanding, 52-week high/low, average volume. No authentication required. |
| Macro rates | FRED REST API (free, key required) | Bank of Canada API for GoC yields | FRED series: `DGS10` (10yr US Treasury), `DEXCAUS` (USD/CAD exchange rate), `CPIAUCSL` (CPI). API key from `FRED_API_KEY` env var. |
| TSX-specific coverage | FMP (covers TSX with `.TO` suffix) | Verify before committing | Run coverage audit before build: fetch TSX universe from FMP, confirm >200 tickers with $500M+ market cap. TSX tickers stored with `.TO` suffix for yfinance fallback compatibility. |

### API Key Configuration

All API keys loaded from `.env` via `python-dotenv`. Never hardcoded.

```
# .env
FMP_API_KEY=your_key_here
FRED_API_KEY=your_key_here
SEC_USER_AGENT=YourName your@email.com   # Required by SEC EDGAR ToS
```

Config keys controlling which sources are active:
```yaml
# config/filter_config.yaml
api:
  primary_financial_source: "fmp"   # "fmp" or "yfinance"
  enable_sec_edgar_fallback: false
  enable_fred: true
```

---

## 2. Rate Limiting Configuration

All limits configurable in `config/filter_config.yaml` under `api.rate_limits`. No hardcoded
values in `api_config.py`.

| Source | Limit | Implementation | Config Key |
|---|---|---|---|
| FMP Starter | 300 req/min | Token bucket rate limiter | `api.rate_limits.fmp_requests_per_minute` |
| yfinance | ~2 req/sec (informal) | `time.sleep` between calls | `api.rate_limits.yfinance_requests_per_second` |
| FRED | 120 req/min | Token bucket rate limiter | `api.rate_limits.fred_requests_per_minute` |
| SEC EDGAR | 10 req/sec | Token bucket rate limiter | `api.rate_limits.sec_requests_per_second` |

**Token bucket implementation** (in `api_config.py`):
- Burst capacity = 1 full minute of requests
- Refill rate = limit / 60 tokens per second
- All outbound calls pass through the rate limiter before executing

**Retry logic** (applied to all sources):
- Max retries: configurable via `api.retry.max_attempts` (default: 3)
- Backoff: exponential with jitter — `wait = base × 2^attempt + random(0, 1)`
- Base wait: `api.retry.base_wait_seconds` (default: 1.0)
- Retry on: HTTP 429, 500, 502, 503, 504
- Do NOT retry on: HTTP 400, 401, 403, 404

---

## 3. Financial Statement Line-Item Mapping Table

This is the canonical mapping used by `data_acquisition/schema.py`. The schema defines the
ideal field name, acceptable substitutes in priority order, substitution confidence, and whether
the security should be dropped if the field is unavailable for ≥ 8 of 10 required annual periods.

All column names in this table are the schema canonical names — the names stored in DuckDB and
referenced by all downstream modules. Downstream modules never reference source API field names.

### Income Statement

| Schema Field | Ideal API Field (FMP) | Acceptable Substitute | Substitution Confidence | Drop if Missing? |
|---|---|---|---|---|
| `net_income` | `netIncome` | `netIncomeFromContinuingOperations` | High | Yes |
| `revenue` | `totalRevenue` | `revenue`, `salesRevenue` | High | Yes |
| `gross_profit` | `grossProfit` | Derived: `totalRevenue − costOfRevenue` | High | Yes |
| `cost_of_revenue` | `costOfRevenue` | `costAndExpenses` (only if gross profit is also missing) | Medium | No — derive gross profit if revenue available |
| `operating_income` | `operatingIncome` | `ebit` | High | Yes |
| `interest_expense` | `interestExpense` | `interestAndDebtExpense` | High | No — assume 0 if missing AND long_term_debt = 0 |
| `income_tax_expense` | `incomeTaxExpense` | `taxProvision` | High | No — informational only |
| `pretax_income` | `incomeBeforeTax` | Derived: `operatingIncome − interestExpense` | Medium | No |
| `selling_general_admin` | `sellingGeneralAndAdministrativeExpenses` | Derived: `operatingExpenses − costOfRevenue − depreciationAndAmortization` | Medium | No — flag; omit F8 for affected tickers |
| `eps_diluted` | `epsdiluted` | `eps` | High | Yes |
| `shares_diluted` | `weightedAverageSharesDiluted` | `sharesOutstanding` | Medium | Yes |
| `depreciation_amortization_is` | `depreciationAndAmortization` (IS) | — | Low | No — prefer CF source |

### Balance Sheet

| Schema Field | Ideal API Field (FMP) | Acceptable Substitute | Substitution Confidence | Drop if Missing? |
|---|---|---|---|---|
| `long_term_debt` | `longTermDebt` | `totalLongTermDebt`, `longTermDebtNoncurrent` | High | Yes |
| `shareholders_equity` | `totalStockholdersEquity` | `totalEquity` | High | Yes |
| `retained_earnings` | `retainedEarnings` | Derived: cumulative net income − dividends (low confidence, use only if direct field unavailable for all years) | Low | No — skip F4 if unavailable |
| `total_current_assets` | `totalCurrentAssets` | `currentAssets` | High | No |
| `total_current_liabilities` | `totalCurrentLiabilities` | `currentLiabilities` | High | No |
| `cash_and_equivalents` | `cashAndCashEquivalents` | `cashAndShortTermInvestments` | High | No |
| `total_assets` | `totalAssets` | — | High | No |
| `total_liabilities` | `totalLiabilities` | Derived: `totalAssets − totalStockholdersEquity` | High | No |
| `shares_outstanding` | `commonStock` (share count) | `sharesOutstanding` | Medium | No — use `shares_diluted` from IS if absent |
| `treasury_stock` | `treasuryStock` | Derived from shares outstanding delta year-over-year | Medium | No — omit buyback test |
| `short_term_debt` | `shortTermDebt` | `currentPortionOfLongTermDebt` | Medium | No |

### Cash Flow Statement

| Schema Field | Ideal API Field (FMP) | Acceptable Substitute | Substitution Confidence | Drop if Missing? | Sign Note |
|---|---|---|---|---|---|
| `operating_cash_flow` | `operatingCashFlow` | `netCashProvidedByOperatingActivities` | High | Yes | Positive = inflow |
| `capital_expenditures` | `capitalExpenditure` | `purchasesOfPropertyPlantAndEquipment`, `purchaseOfPPE` | High | Yes | **Always stored negative. Negate + log WARNING if source returns positive.** |
| `depreciation_amortization_cf` | `depreciationAndAmortization` (CF) | `depreciationAmortizationDepletion` | High | Yes — preferred source for F1 | Positive = add-back |
| `free_cash_flow` | `freeCashFlow` | Derived: `operatingCashFlow + capitalExpenditure` (using stored negative CapEx) | High | No — derive if absent |
| `dividends_paid` | `dividendsPaid` | `paymentOfDividends`, `cashDividendsPaid` | Medium | No |
| `stock_repurchases` | `commonStockRepurchased` | `repurchaseOfCapitalStock` | Medium | No |
| `net_debt_issuance` | `netIssuancePaymentsOfDebt` | `longTermDebtIssuance` + `longTermDebtRepayment` | Medium | No |

### D&A Source Priority

Depreciation & Amortization is used in F1 (Owner Earnings) and must be sourced carefully:

1. **First choice:** `depreciationAndAmortization` from the **cash flow statement** (CF)
   — this is the non-cash add-back figure, which is what F1 requires.
2. **Second choice:** `depreciationAndAmortization` from the **income statement** (IS)
   — acceptable substitute, confidence = Medium.
3. **Drop rule:** If D&A is unavailable from both CF and IS for a given year, that year's
   F1 computation returns NaN. If D&A is unavailable for ≥ 5 of 10 years, the security is
   excluded (D&A is required for owner earnings, the primary quality metric).

### SG&A Derivation

When `sellingGeneralAndAdministrativeExpenses` is unavailable directly:

```
sga_derived = operatingExpenses − costOfRevenue − depreciationAndAmortization
```

Log as WARNING with confidence = Medium. If the derived value is negative (indicating the
decomposition is inconsistent), set SG&A = NaN and skip F8 for that ticker. Do not drop.

### Interest Expense Zero-Fill Rule

If `interest_expense` is missing AND `long_term_debt = 0` for that period, set
`interest_expense = 0` and log at INFO (not WARNING — this is an expected case for
debt-free companies). If `long_term_debt > 0` and interest expense is still missing, set
`interest_expense = NaN` and log at WARNING.

---

## 4. Drop Rule

If any line item marked **"Drop if Missing? = Yes"** cannot be sourced (neither ideal field nor
any acceptable substitute) for **≥ 8 of 10 required annual periods**, the security is excluded
from all downstream processing. Reason logged in `data_quality_log` table:

```
check_name: "required_field_coverage"
severity: "ERROR"
detail: "Field {schema_field} available for only {n}/10 years"
action_taken: "dropped"
```

The drop threshold (8 of 10 years) is read from `config/filter_config.yaml` under
`data_quality.min_field_coverage_years`.

---

## 5. Currency Normalization

- **TSX stocks** report financial statements in **CAD**. FMP indicates reporting currency in the
  response metadata (`reportedCurrency` field).
- **US stocks** report in **USD**.
- **Normalization rule:** Convert all CAD-denominated financial statement values to USD at the
  time of ingestion using the FRED `DEXCAUS` rate (USD per CAD, most recent monthly average).
- Store the exchange rate used in `data_quality_log` for auditability.
- `market_data.current_price_usd` is always stored in USD regardless of exchange.
- **Final reports:** Recommendation module converts entry price and intrinsic value back to CAD
  for TSX-listed securities using the same FRED rate. Clearly labeled "CAD" in output.
- Exchange rate is NOT applied to `market_cap_usd` or `enterprise_value_usd` — these are sourced
  directly in USD from FMP/yfinance.

---

## 6. yfinance Field Candidate Labels

When FMP is unavailable or for market data, yfinance is used. Field names in yfinance are
unstable across versions — use the candidate list approach: try each label in priority order,
use the first one found, log which was selected.

### Income Statement (`Ticker.financials`)

| Schema Field | Candidate Labels (try in order) |
|---|---|
| `net_income` | `Net Income`, `Net Income From Continuing Operations` |
| `revenue` | `Total Revenue`, `Revenue` |
| `gross_profit` | `Gross Profit` |
| `operating_income` | `Operating Income`, `Total Operating Income As Reported`, `EBIT` |
| `interest_expense` | `Interest Expense`, `Interest Expense Non Operating` |
| `eps_diluted` | `Diluted EPS` |
| `shares_diluted` | `Diluted Average Shares`, `Average Dilution Earnings` |
| `selling_general_admin` | `Selling General Administrative`, `Selling And Marketing Expense` |
| `depreciation_amortization_is` | `Reconciled Depreciation`, `Depreciation And Amortization` |

### Balance Sheet (`Ticker.balance_sheet`)

| Schema Field | Candidate Labels (try in order) |
|---|---|
| `long_term_debt` | `Long Term Debt`, `Long Term Debt And Capital Lease Obligation` |
| `shareholders_equity` | `Stockholders Equity`, `Common Stock Equity` |
| `retained_earnings` | `Retained Earnings` |
| `total_current_assets` | `Current Assets`, `Total Current Assets` |
| `total_current_liabilities` | `Current Liabilities`, `Total Current Liabilities` |
| `cash_and_equivalents` | `Cash And Cash Equivalents`, `Cash` |
| `total_assets` | `Total Assets` |
| `shares_outstanding` | `Ordinary Shares Number`, `Share Issued` |
| `treasury_stock` | `Treasury Stock`, `Treasury Shares Number` |

### Cash Flow (`Ticker.cashflow`)

| Schema Field | Candidate Labels (try in order) | Sign Note |
|---|---|---|
| `operating_cash_flow` | `Operating Cash Flow`, `Cash Flow From Continuing Operating Activities` | |
| `capital_expenditures` | `Capital Expenditure`, `Purchase Of Ppe` | Negate if positive |
| `depreciation_amortization_cf` | `Reconciled Depreciation`, `Depreciation And Amortization` | |
| `free_cash_flow` | `Free Cash Flow` | Derive if absent |
| `dividends_paid` | `Cash Dividends Paid`, `Payment Of Dividends` | |
| `stock_repurchases` | `Repurchase Of Capital Stock`, `Common Stock Repurchase` | |

---

## 7. FRED Series Reference

| Metric | FRED Series ID | Units | Schema Transformation |
|---|---|---|---|
| 10yr US Treasury yield | `DGS10` | % (e.g., 4.25) | Divide by 100 → store as decimal (0.0425) |
| USD/CAD exchange rate | `DEXCAUS` | USD per 1 CAD | Use directly |
| CPI (US) | `CPIAUCSL` | Index (1982-84=100) | Compute YoY: `(current − prior_yr) / prior_yr` |

FRED missing value marker is `"."` — replace with `float("nan")` during parsing.
Always fetch the most recent observation. Store `as_of_date` alongside macro values.

**GoC 10yr yield:** FRED series `IRLTLT01CAM156N` (IMF-sourced, monthly). If unavailable,
fall back to Bank of Canada published rate (manual entry or scrape from bankofcanada.ca).
Config key: `api.goc_yield_source` — `"fred"` or `"bankofcanada"`.

---

## 8. Raw Response Storage

Before transforming any API response, write the raw JSON to disk:

```
data/raw/{ticker}/fmp_income_{fiscal_year}.json
data/raw/{ticker}/fmp_balance_{fiscal_year}.json
data/raw/{ticker}/fmp_cashflow_{fiscal_year}.json
data/raw/{ticker}/yfinance_info_{as_of_date}.json
```

This allows re-running transformations without re-fetching. Controlled by config:
```yaml
api:
  store_raw_responses: true    # Set false to save disk space in production
  raw_data_dir: "data/raw"
```

---

## 9. Data Quality Protocol

After ingestion of each batch:

1. **Cross-validation sample:** For a random sample of 20 tickers (or all tickers if universe
   < 100), compare FMP and yfinance values for `net_income`, `revenue`, `operating_cash_flow`.
   Flag any ticker where any field diverges by > 5%. Log at WARNING.

2. **Substitution density check:** Flag any security where > 2 line items required substitution.
   These are candidates for manual review. Log at WARNING: `"[TICKER] used substitutions for
   {n} fields: {list}"`.

3. **CapEx sign audit:** After ingestion, assert that all stored `capital_expenditures` values
   are ≤ 0. If any are positive, log at ERROR and re-negate.

4. **Data quality report output:**
   Written to `data/processed/data_quality_report.csv` after each full pipeline run.

   Columns:
   | Column | Description |
   |---|---|
   | `ticker` | Ticker symbol |
   | `exchange` | Exchange |
   | `years_available` | Count of fiscal years with data |
   | `missing_fields` | Comma-separated list of schema fields with NaN for all years |
   | `substitutions_used` | Comma-separated list of `{field}:{substitute}:{confidence}` |
   | `cross_validation_flags` | Fields where FMP vs yfinance diverge > 5% |
   | `drop_reason` | Reason for exclusion, or empty if included |

---

## 10. Sector Exclusion — Financial Companies

Companies with the following SIC codes are excluded before any metric computation. Their
financial statements do not conform to Buffett's framework: gross margin is not meaningful,
leverage is a business input not a risk factor, and interest expense is revenue not a cost.

| Sector | SIC Range | Reason |
|---|---|---|
| Commercial Banks | 6020–6029 | Leverage is core business model; D/E and interest coverage meaningless |
| Savings Institutions | 6035–6036 | Same as commercial banks |
| Insurance | 6311–6399 | Float-funded; loss ratios replace gross margin |
| Real Estate Investment Trusts | 6798 | FFO-based valuation; no meaningful EPS CAGR |
| Investment Firms | 6726 | Holding companies; consolidated financials obscure underlying economics |

SIC ranges configurable in `filter_config.yaml`:
```yaml
exclusions:
  sic_ranges:
    - [6020, 6029]   # Commercial banks
    - [6035, 6036]   # Savings institutions
    - [6311, 6399]   # Insurance
    - [6726, 6726]   # Investment firms
    - [6798, 6798]   # REITs
  sectors:           # GICS sector-level backup if SIC unavailable
    - "Financials"
    - "Utilities"
    - "Real Estate"
```

If SIC code is unavailable for a ticker, fall back to GICS sector string from the universe
fetch. If neither is available, log at WARNING and include the ticker (do not silently exclude).

---

## 11. Unit Convention Summary

| Context | Unit | Notes |
|---|---|---|
| All financial statement fields in DuckDB | USD thousands | Divide FMP/yfinance full-dollar values by 1,000 on ingest |
| `market_cap_usd` | Full USD dollars | Exception — not divided by 1,000 |
| `enterprise_value_usd` | Full USD dollars | Exception — not divided by 1,000 |
| `current_price_usd` | USD per share | Not divided |
| FRED rates (`us_treasury_10yr`, `goc_bond_10yr`) | Decimal (e.g., 0.0425) | Divide raw FRED percent by 100 on ingest |
| `usd_cad_rate` | USD per 1 CAD | Use directly from FRED `DEXCAUS` |
| EPS | USD per share | Not divided — yfinance and FMP already return per-share |
| Shares outstanding | Full count | Not divided |

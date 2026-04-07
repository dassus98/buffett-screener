# Data Sources

> API sources, field mappings, rate limits, and caching strategy.

---

## Source matrix

| Data need | Primary source | Fallback | Auth |
|-----------|---------------|----------|------|
| Universe (US) | Wikipedia S&P 500 scrape + iShares IWB CSV | — | None |
| Universe (CA) | Wikipedia S&P/TSX 60 scrape | FMP `.TO` symbols | None / FMP key |
| Financial statements (10yr) | yfinance (`Ticker.financials`) | SEC EDGAR XBRL | None / `SEC_USER_AGENT` |
| Current price & market cap | yfinance `fast_info` | — | None |
| Beta, 52wk hi/lo | yfinance `info` dict | — | None |
| Macro rates | FRED REST API | — | `FRED_API_KEY` |
| Company profiles | yfinance `info` dict | FMP `/profile` | None / FMP key |

---

## yfinance field mappings

### Income statement (`Ticker.financials`)

| Schema field | Candidate labels (tried in order) |
|---|---|
| `revenue` | `Total Revenue`, `Revenue` |
| `cost_of_revenue` | `Cost Of Revenue`, `Cost of Revenue` |
| `gross_profit` | `Gross Profit` |
| `operating_income` | `Operating Income`, `Total Operating Income As Reported`, `EBIT` |
| `interest_expense` | `Interest Expense`, `Interest Expense Non Operating` |
| `pretax_income` | `Pretax Income`, `Income Before Tax` |
| `income_tax` | `Tax Provision`, `Income Tax Expense` |
| `net_income` | `Net Income`, `Net Income From Continuing Operations` |
| `depreciation_amortization` | `Reconciled Depreciation`, `Depreciation And Amortization` |
| `shares_diluted` | `Diluted Average Shares`, `Average Dilution Earnings` |
| `eps_diluted` | `Diluted EPS` |

### Balance sheet (`Ticker.balance_sheet`)

| Schema field | Candidate labels |
|---|---|
| `cash_and_equivalents` | `Cash And Cash Equivalents`, `Cash` |
| `short_term_investments` | `Short Term Investments`, `Available For Sale Securities` |
| `total_current_assets` | `Current Assets`, `Total Current Assets` |
| `total_assets` | `Total Assets` |
| `total_current_liabilities` | `Current Liabilities`, `Total Current Liabilities` |
| `short_term_debt` | `Current Debt`, `Short Long Term Debt`, `Short Term Debt` |
| `long_term_debt` | `Long Term Debt`, `Long Term Debt And Capital Lease Obligation` |
| `total_liabilities` | `Total Liabilities Net Minority Interest`, `Total Liabilities` |
| `shareholders_equity` | `Stockholders Equity`, `Common Stock Equity` |
| `retained_earnings` | `Retained Earnings` |
| `shares_outstanding` | `Ordinary Shares Number`, `Share Issued` |

### Cash flow (`Ticker.cashflow`)

| Schema field | Candidate labels | Sign note |
|---|---|---|
| `operating_cash_flow` | `Operating Cash Flow`, `Cash Flow From Continuing Operating Activities` | |
| `capital_expenditures` | `Capital Expenditure`, `Purchase Of Ppe` | **Negate if positive** |
| `free_cash_flow` | `Free Cash Flow` | Compute if absent |
| `dividends_paid` | `Cash Dividends Paid`, `Payment Of Dividends` | |
| `stock_buybacks` | `Repurchase Of Capital Stock`, `Common Stock Repurchase` | |
| `net_debt_issuance` | `Net Issuance Payments Of Debt`, `Long Term Debt Issuance` | |

---

## FRED series IDs

| Metric | FRED series | Units → schema |
|--------|------------|---------------|
| 10Y Treasury | `DGS10` | % → ÷100 → decimal |
| 2Y Treasury | `DGS2` | % → ÷100 → decimal |
| CPI | `CPIAUCSL` | index → compute YoY |
| Fed funds rate | `FEDFUNDS` | % → ÷100 → decimal |
| Real GDP growth | `A191RL1Q225SBEA` | % → ÷100 → decimal |

FRED uses `"."` as the missing value marker. Replace with `float("nan")`.

---

## SEC EDGAR XBRL

- Company facts endpoint: `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`
- Company tickers lookup: `https://www.sec.gov/files/company_tickers.json`
- Rate limit: 10 requests/second. Stay at ≤ 8 req/s.
- User-Agent: must be `"Name email@example.com"` — set via `SEC_USER_AGENT` env var.
- Filter to: `form="10-K"`, `unit="USD"`, annual frequency.

---

## Cache TTL

| Data type | DuckDB TTL |
|-----------|-----------|
| Universe snapshot | 7 days |
| Financial statements | 24 hours |
| Market data | 1 day |
| Macro snapshot | 1 day |
| Data quality reports | per run |

---

## Unit convention reminder

| Context | Unit |
|---------|------|
| All financial statement fields | USD thousands |
| `MarketData.market_cap` | Full USD dollars |
| `MarketData.enterprise_value` | Full USD dollars |
| yfinance raw output | Full USD dollars → **divide by 1000** when storing |
| Rates (FRED) | Percent → **divide by 100** when storing |

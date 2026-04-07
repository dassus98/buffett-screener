# FORMULAS.md — Buffett Screener Formula Registry

**Authoritative reference for every quantitative test in this system.**
`metrics_engine/` and `valuation_reports/` must implement these definitions exactly.
Any deviation requires an explicit comment citing the reason and confidence level.

---

## Registry Index

| ID | Name | Module | Hard Filter? | Threshold |
|----|------|--------|-------------|-----------|
| F1 | Owner Earnings | `metrics_engine/owner_earnings.py` | No (input to F2, F5) | — |
| F2 | Initial Rate of Return | `metrics_engine/returns.py` | Yes | > 2× bond yield |
| F3 | Return on Equity | `metrics_engine/returns.py` | Yes | 10yr avg ≥ 15% |
| F4 | Return on Retained Earnings ($1 Test) | `metrics_engine/returns.py` | Yes | ≥ 12% |
| F5 | Debt Payoff Test | `metrics_engine/leverage.py` | Yes | < 5 years |
| F6 | Debt-to-Equity Ratio | `metrics_engine/leverage.py` | Yes | < 0.80 |
| F7 | Gross Margin | `metrics_engine/profitability.py` | Yes | > 20% |
| F8 | SGA-to-Gross-Profit Ratio | `metrics_engine/profitability.py` | No (scoring) | < 80% |
| F9 | Interest Expense Coverage | `metrics_engine/leverage.py` | Yes | < 15% of EBIT |
| F10 | Net Margin Consistency | `metrics_engine/profitability.py` | Yes | Positive ≥ 8/10 yrs |
| F11 | EPS Growth Rate (10yr CAGR) | `metrics_engine/growth.py` | Yes | ≥ 10% CAGR |
| F12 | CapEx-to-Net-Earnings Ratio | `metrics_engine/capex.py` | No (scoring) | < 50% |
| F13 | Share Buyback Indicator | `metrics_engine/growth.py` | No (scoring) | Positive = good |
| F14 | Projected Intrinsic Value | `valuation_reports/intrinsic_value.py` | No (valuation) | Return ≥ 15% |
| F15 | Margin of Safety | `valuation_reports/margin_of_safety.py` | No (valuation) | ≥ 25–40% |
| F16 | Earnings Yield vs. Bond Yield | `metrics_engine/valuation.py` | No (scoring) | Spread ≥ 2% |

All thresholds listed here are defaults. Runtime values are read from
`config/filter_config.yaml`. These definitions are the **semantic contract**;
the config file holds the **tunable parameters**.

---

## Formula Definitions

---

### F1 — Owner Earnings

**Buffett's preferred cash flow measure.**

#### Source
1986 Berkshire Hathaway Shareholder Letter; Hagstrom *The Warren Buffett Way* Ch. 8

#### Definition

```
Owner Earnings = Net Income
               + Depreciation & Amortization
               + Other Non-Cash Charges
               − Average Annual Maintenance Capital Expenditures
               − Change in Working Capital (if required to maintain competitive position)
```

**Working Capital Change:**
```
ΔWC = (Current Assets_t − Cash_t) − (Current Liabilities_t − Short-Term Debt_t)
     − [(Current Assets_{t-1} − Cash_{t-1}) − (Current Liabilities_{t-1} − Short-Term Debt_{t-1})]
```

A positive ΔWC means working capital increased (cash consumed); subtract from Owner Earnings.
A negative ΔWC means working capital decreased (cash released); add to Owner Earnings.

#### Maintenance CapEx Proxy

Maintenance CapEx is not separately disclosed in public filings.
**Proxy used: Depreciation & Amortization.**

This is a **known low-confidence approximation** (Buffett himself acknowledged the
difficulty of estimating this in the 1986 letter). Implement as:

```
Maintenance CapEx ≈ Depreciation & Amortization
```

**Distortion flag:** If `Total CapEx > 2 × Depreciation`, the approximation is likely
understating true maintenance CapEx (the company is in a heavy growth investment phase).
Log a WARNING and set `owner_earnings_capex_flag = True` in the output record.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Net Income | Income Statement | `net_income` |
| Depreciation & Amortization | Cash Flow (preferred) or Income Statement | `depreciation_amortization` |
| Capital Expenditures | Cash Flow | `capital_expenditures` (stored negative) |
| Total Current Assets | Balance Sheet | `total_current_assets` |
| Cash & Equivalents | Balance Sheet | `cash_and_equivalents` |
| Total Current Liabilities | Balance Sheet | `total_current_liabilities` |
| Short-Term Debt | Balance Sheet | `short_term_debt` |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| D&A missing from Cash Flow Statement | Check Income Statement for D&A line. Log substitution at WARNING level with confidence = Medium. |
| D&A missing from both statements | **Drop security.** Cannot compute F1 without D&A. Log at ERROR. |
| Net Income negative | Compute Owner Earnings anyway. A negative result is valid — it means the business destroyed value this year. Do not drop. |
| CapEx reported as positive by data source | Negate before use. Log at WARNING: "CapEx sign corrected for [TICKER]." |
| Single year of data | Owner Earnings computable for one year. Multi-year average requires ≥ 3 years. |

#### Interpretation

Owner Earnings is the cash a business could distribute to owners without impairing
future earning power. It is more conservative than FCF when growth CapEx is high,
and more liberal than FCF when D&A substantially understates true maintenance costs.

---

### F2 — Initial Rate of Return

**Buffett's first-cut value check.**

#### Source
*Buffettology* Ch. 10 (Mary Buffett & David Clark)

#### Definition

```
Initial Rate of Return = Owner Earnings Per Share / Current Market Price Per Share

Owner Earnings Per Share = Owner Earnings (F1) / Shares Outstanding (diluted)
```

#### Threshold

```
Must exceed: 2 × current 10-year government bond yield
```

*Example: If the 10-year Treasury yields 4.5%, the Initial Rate of Return
must exceed 9.0% to pass.*

This threshold is read from `config/filter_config.yaml → valuation.earnings_yield.min_multiplier_of_rfr`.

#### Required Line Items

| Line Item | Source |
|-----------|--------|
| Owner Earnings | F1 output |
| Shares Outstanding (diluted) | `shares_diluted` (Income Statement) or `shares_outstanding` (Balance Sheet) |
| Current Market Price | `market_data.price` |
| 10-year Government Bond Yield | `macro_data.treasury_10y_yield` (US) or `macro_data.goc_10y_yield` (CA) |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Owner Earnings ≤ 0 | Rate of return is negative or zero → automatic fail. Log at WARNING. |
| Market price unavailable | Cannot compute → skip security, log at ERROR. |
| Bond yield unavailable | Use last cached value. If cache > 7 days old, log WARNING and use config default. |

#### Interpretation

A rate of return above 2× the bond yield means the equity is offering more than
twice the "risk-free" alternative — a rough margin that compensates for equity risk
and business uncertainty. This is an entry-point screen, not a precision valuation.

---

### F3 — Return on Equity (ROE)

**Primary measure of management's capital allocation skill.**

#### Source
*Buffettology* Ch. 6; Hagstrom *The Warren Buffett Way* Ch. 8

#### Definition

```
ROE = Net Income / Average Shareholders' Equity

Average Shareholders' Equity = (Equity_beginning_of_year + Equity_end_of_year) / 2
                              = (Equity_{t-1} + Equity_t) / 2
```

Compute for each available year. The **hard filter** applies to the **10-year average**.
Also flag if fewer than 3 of the last 10 years individually exceed 15%.

#### Threshold

| Tier | Value | Action |
|------|-------|--------|
| Excellent | ≥ 20% (10yr avg) | Strong positive signal |
| Pass | ≥ 15% (10yr avg) | Passes hard filter |
| Fail | < 15% (10yr avg) | Eliminated |

Thresholds from `config/filter_config.yaml → hard_filters.returns.min_roe_10yr_avg_pct`.

#### Leverage Cross-Check

**ROE inflated by debt is a false positive.** Always report `debt_to_equity` (F6)
alongside ROE. If F6 > 1.0 and ROE > 20%, log a WARNING:
"High ROE may be leverage-driven — verify via ROIC."

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Net Income | Income Statement | `net_income` |
| Shareholders' Equity (beginning) | Balance Sheet, prior year | `shareholders_equity` |
| Shareholders' Equity (ending) | Balance Sheet, current year | `shareholders_equity` |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Negative equity (e.g., McDonald's, large buyback programs) | ROE is **mathematically meaningless** (negative denominator produces a misleadingly positive ratio). **Do not compute. Do not drop.** Set `roe = NaN`, flag `negative_equity = True`. Assess the company via F5, F7, F10, F11. |
| Equity = 0 in one year | Use single-year equity for that year's calculation, not average. Log at WARNING. |
| Fewer than 10 years of data | Compute average over available years (minimum 5). Adjust threshold note in output. |

---

### F4 — Return on Retained Earnings (The $1 Test)

**Does management create ≥ $1 of market value for every $1 retained?**

#### Source
Hagstrom *The Warren Buffett Way* Ch. 8; 1983 Berkshire Hathaway Shareholder Letter

#### Definition

```
Retained Earnings Per Share (year N) = EPS_N − Dividends Per Share_N

Cumulative Retained EPS (10yr) = Σ (EPS_N − DPS_N) for N = year_0 to year_9

EPS Change over 10yr = EPS_current − EPS_10yr_ago

Return on Retained Earnings = EPS Change / Cumulative Retained EPS Per Share
```

#### Threshold

```
≥ 12% — operationalizes Buffett's principle that a retained dollar must
          produce at least $1 of market value (approximated via EPS growth)
```

Threshold from `config/filter_config.yaml → hard_filters.returns.min_return_on_retained_earnings_pct`.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Diluted EPS (each year, 10yr) | Income Statement | `eps_diluted` |
| Dividends Per Share (each year) | Derived: `dividends_paid / shares_outstanding` or direct if available |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Cumulative retained earnings ≤ 0 | Company paid out more than it earned → metric not meaningful. Flag, skip (do not drop security — assess via other metrics). |
| EPS_10yr_ago negative or zero | Cannot compute meaningful EPS change. Use earliest year with positive EPS as base; document adjusted window in output. |
| No dividends paid | Retained Earnings Per Share = EPS for each year. Valid case — compute normally. |
| Fewer than 10 years of EPS data | Use available years (minimum 5). Reduce the denominator accordingly. |

---

### F5 — Debt Payoff Test

**How many years of Owner Earnings does it take to retire all long-term debt?**

#### Source
*Warren Buffett and the Interpretation of Financial Statements* Ch. 31

#### Definition

```
Debt Payoff Years = Total Long-Term Debt / Owner Earnings (F1, most recent year)
```

#### Threshold

| Years to Pay Off | Rating | Action |
|-----------------|--------|--------|
| < 4 years | Excellent | Strong pass |
| 4–5 years | Acceptable | Passes hard filter |
| ≥ 5 years | Fail | Eliminated |

Threshold from `config/filter_config.yaml → hard_filters.leverage.max_debt_payoff_years`.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Long-Term Debt | Balance Sheet | `long_term_debt` |
| Owner Earnings | F1 output | — |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Owner Earnings ≤ 0 | Payoff time is infinite → **automatic fail**. Log at WARNING. |
| Zero long-term debt | Payoff time = 0 years → **automatic pass**. Set `debt_payoff_years = 0`. |
| Owner Earnings positive but very small (< $1M) | Compute normally. May produce a very large number → will fail threshold. |

---

### F6 — Debt-to-Equity Ratio

**Balance sheet conservatism test.**

#### Source
*Buffettology* Ch. 8; Graham's defensive investor criteria (*The Intelligent Investor* Ch. 14)

#### Definition

```
Debt-to-Equity = Total Long-Term Debt / Total Shareholders' Equity
```

#### Threshold

| D/E Ratio | Action |
|-----------|--------|
| < 0.50 | Preferred — durable advantage indicator |
| 0.50–0.80 | Acceptable — passes hard filter |
| > 0.80 | Fails hard filter (unless extraordinary earnings predictability documented) |
| > 1.0 | Generally disqualifying |

Threshold from `config/filter_config.yaml → hard_filters.leverage.max_debt_to_equity`.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Long-Term Debt | Balance Sheet | `long_term_debt` |
| Shareholders' Equity | Balance Sheet | `shareholders_equity` |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Negative shareholders' equity | Ratio is mathematically meaningless (negative denominator). **Do not compute.** Flag `negative_equity = True`. Assess debt burden via F5 instead. |
| Zero shareholders' equity | Same as negative equity treatment. |
| Zero long-term debt | D/E = 0 → automatic pass. |

---

### F7 — Gross Margin

**Primary moat indicator — pricing power and cost structure.**

#### Source
*Warren Buffett and the Interpretation of Financial Statements* Ch. 6

#### Definition

```
Gross Margin = Gross Profit / Total Revenue

Gross Profit = Total Revenue − Cost of Revenue (COGS)
```

Compute for each available year. The hard filter applies to the **10-year average**.
Also report the trend (improving / stable / deteriorating) over the 10-year window.

#### Threshold

| Gross Margin | Interpretation | Action |
|-------------|---------------|--------|
| > 40% | Durable competitive advantage | Strong pass |
| 20–40% | Competitive but potentially vulnerable | Passes hard filter |
| < 20% | Commodity economics | **Fails hard filter** |

Threshold from `config/filter_config.yaml → hard_filters.profitability.min_gross_margin_pct`.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Gross Profit | Income Statement | `gross_profit` |
| Total Revenue | Income Statement | `revenue` |

If Gross Profit is not reported directly, derive as: `revenue − cost_of_revenue`.
Log this substitution at INFO level.

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Revenue = 0 | Cannot compute → skip year, log at WARNING. |
| Financial companies (banks, insurers) | Gross margin is not a meaningful concept → already excluded by sector filter. |
| Gross profit reported as negative | Valid (selling below cost). Include in average. Do not drop. |

---

### F8 — SGA-to-Gross-Profit Ratio

**Overhead efficiency indicator.**

#### Source
*Warren Buffett and the Interpretation of Financial Statements* Ch. 7

#### Definition

```
SGA-to-Gross-Profit = SG&A Expenses / Gross Profit
```

Where SG&A = Selling, General & Administrative Expenses (operating overhead).

#### Threshold

| SGA/GP Ratio | Interpretation |
|-------------|---------------|
| < 30% | Excellent — minimal overhead relative to gross economics |
| 30–80% | Industry-dependent — acceptable |
| > 80% | Red flag — overhead consuming most of gross profit |

This is a **scoring factor** (contributes to composite rank), not a hard filter.
Threshold from `config/filter_config.yaml → soft_filters.profitability.max_sga_to_gross_profit_pct`.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| SG&A Expenses | Income Statement | `selling_general_administrative` |
| Gross Profit | Income Statement | `gross_profit` |

#### Implementation Note — D&A Normalization

Some data sources include Depreciation & Amortization within SG&A. When both
are separately reported, normalize:
```
Adjusted SG&A = Reported SG&A − D&A (if D&A is included within SG&A)
```
If D&A inclusion cannot be determined from the source, flag the substitution:
`sga_da_separated = False`, log at WARNING.

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Gross profit ≤ 0 | Ratio is meaningless → set to `NaN`, log at WARNING. |
| SG&A not separately reported | Set to `NaN`, log at WARNING. Do not impute. |

---

### F9 — Interest Expense Coverage

**Debt burden relative to operating earnings.**

#### Source
*Warren Buffett and the Interpretation of Financial Statements* Ch. 11

#### Definition

```
Interest Burden = Interest Expense / Operating Income (EBIT)
```

Note: This is the **inverse** of the traditional interest coverage ratio.
A lower value means interest expense is a smaller fraction of operating income → better.

```
Traditional Coverage = EBIT / Interest Expense  (higher = better)
Buffett's framing   = Interest Expense / EBIT   (lower = better, threshold < 15%)
```

Both are computed and stored. The hard filter uses the Buffett framing.

#### Threshold

```
Interest Expense / EBIT < 15%  →  passes
```

Threshold from `config/filter_config.yaml → hard_filters.leverage.max_interest_to_ebit_pct`.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Interest Expense | Income Statement | `interest_expense` |
| Operating Income (EBIT) | Income Statement | `operating_income` |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Zero interest expense + zero debt | **Automatic pass.** Set `interest_burden = 0.0`. |
| Negative operating income (EBIT) | Ratio is economically undefined (or infinite burden). **Do not compute.** Flag, automatic fail. Log at WARNING. |
| Interest expense = 0 but debt exists | Possible reporting gap. Log at WARNING. Treat as 0% burden for calculation purposes. |

---

### F10 — Net Margin Consistency

**Earnings quality and business durability.**

#### Source
Hagstrom *The Warren Buffett Way* Ch. 8; *Interpretation of Financial Statements* Ch. 14

#### Definition

```
Net Margin (year N) = Net Income_N / Total Revenue_N
```

Compute for each of the 10 available years. Report:
- Count of years with positive net margin
- 10-year average net margin
- Trend direction (improving / stable / deteriorating, via linear regression slope)
- Whether upward trend exists

#### Threshold

| Criterion | Threshold | Source in Config |
|-----------|-----------|-----------------|
| Positive margin years | ≥ 8 of last 10 years | `hard_filters.profitability.min_profitable_years` |
| 10-year average | Above industry median (informational, not a hard filter) | — |
| Excellent indicator | > 20% average | Scoring bonus |

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Net Income (10yr) | Income Statement | `net_income` |
| Total Revenue (10yr) | Income Statement | `revenue` |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Revenue = 0 in a year | Skip that year in the average. Log at WARNING. |
| Fewer than 10 years of data | Compute over available years (minimum 5). Adjust the "≥ 8 of 10" threshold proportionally: `min_positive = round(0.8 × available_years)`. |

#### Interpretation

Buffett seeks businesses with a "castle and moat" — competitors cannot easily
erode margins. Consistent positive margins (even if moderate) are more predictive
of durable advantage than high but volatile margins.

---

### F11 — EPS Growth Rate (10-Year CAGR)

**Earnings power trend over a full business cycle.**

#### Source
*Buffettology* Ch. 6

#### Definition

```
EPS CAGR = (EPS_current / EPS_10yr_ago)^(1/10) − 1
```

Also compute a **3-year CAGR** and a **5-year CAGR** as supporting data points.

Report the number of years in the 10-year window where EPS declined year-over-year.

#### Threshold

| Criterion | Threshold |
|-----------|-----------|
| 10-year CAGR | ≥ 10% (hard filter) |
| Max EPS decline years | ≤ 2 of 10 years (hard filter) |

Thresholds from `config/filter_config.yaml → hard_filters.growth`.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Diluted EPS (each year, 10yr) | Income Statement | `eps_diluted` |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| EPS_10yr_ago ≤ 0 | CAGR undefined (log or negative base). Use earliest year with positive EPS as base; adjust the exponent to `1 / (current_year − base_year)`. Document the adjusted window. |
| Fewer than 5 positive EPS years in the window | **Drop security.** Cannot establish an earnings trend. Log at ERROR. |
| EPS_current ≤ 0 | CAGR is negative or undefined. Security fails automatically on this metric. |
| Exactly one year of data | CAGR not computable. Require ≥ 3 years minimum. |

---

### F12 — CapEx-to-Net-Earnings Ratio

**Capital intensity indicator — how much must be reinvested to sustain earnings?**

#### Source
*Warren Buffett and the Interpretation of Financial Statements* Ch. 24

#### Definition

```
CapEx-to-Earnings = |Capital Expenditures| / Net Income
```

Compute for each year, then take the **10-year average** of the annual ratios.
Use absolute value of CapEx (it is stored as negative per schema convention).

#### Threshold

| CapEx/Earnings | Interpretation | Action |
|---------------|---------------|--------|
| < 25% | Excellent — low reinvestment need | Strong scoring signal |
| 25–50% | Acceptable — moderate capital intensity | Neutral |
| > 50% | Capital-intensive — flag for review | Negative scoring signal |

This is a **scoring factor**, not a hard filter.
Threshold from `config/filter_config.yaml → soft_filters.capex.max_capex_to_earnings_pct`.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Capital Expenditures | Cash Flow | `capital_expenditures` (stored negative) |
| Net Income | Income Statement | `net_income` |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Net income ≤ 0 in a year | **Exclude that year from the average.** Do not divide by zero or a negative number. Log at DEBUG. |
| All 10 years have negative net income | Cannot compute → set to `NaN`, do not score. Security should already have failed F10. |
| CapEx = 0 (e.g., pure software/service) | Ratio = 0% → excellent score. Valid. |

---

### F13 — Share Buyback Indicator

**Does management return excess capital by retiring shares?**

#### Source
*Warren Buffett and the Interpretation of Financial Statements* Ch. 28

#### Definition

```
Buyback Rate = (Shares_10yr_ago − Shares_current) / Shares_10yr_ago
```

A positive value means shares were retired (good — increases per-share value).
A negative value means shares were issued / diluted (bad — per-share value eroded).

Also compute the **dilution-adjusted EPS CAGR**:
```
Dilution Adjustment = Shares_10yr_ago / Shares_current
Dilution-Adjusted EPS = Reported EPS × Dilution Adjustment
```

#### Threshold

This is a **scoring factor**, not a hard filter. Scoring:
- Buyback rate > 5% → strong positive signal (+2 scoring points)
- Buyback rate 0–5% → neutral
- Buyback rate < 0% (dilution) → negative signal (−1 scoring point)

Scoring weights from `config/filter_config.yaml → composite_weights`.

#### Required Line Items

| Line Item | Statement | Schema Field |
|-----------|-----------|-------------|
| Shares Outstanding / Diluted (current) | Income Statement | `shares_diluted` |
| Shares Outstanding / Diluted (10yr ago) | Income Statement | `shares_diluted` (historical) |

Use `weightedAverageSharesDiluted` where available; fall back to period-end
`sharesOutstanding`. Log the substitution if fallback is used.

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Shares data unavailable 10yr ago | Use earliest available year. Adjust the denominator label in output. |
| Stock splits in the period | Ensure share counts are split-adjusted before computing. yfinance returns split-adjusted values by default — verify and log. |

---

### F14 — Projected Intrinsic Value (Three-Scenario Buffettology Method)

**Forward-looking value estimate using EPS projection.**

#### Source
*Buffettology* Ch. 11–14 (Mary Buffett & David Clark)

#### Definition

**Step 1 — Project EPS:**
```
Projected EPS = Current EPS × (1 + Growth Rate)^10
```

**Step 2 — Project Price:**
```
Projected Price = Projected EPS × Terminal P/E
```

**Step 3 — Projected Annual Return:**
```
Projected Return = (Projected Price / Current Price)^(1/10) − 1
```

**Step 4 — Weighted Intrinsic Value (present value approach):**
```
Intrinsic Value = Projected Price / (1 + Discount Rate)^10
```

#### Three Scenarios

| Parameter | Bear (weight 25%) | Base (weight 50%) | Bull (weight 25%) |
|-----------|-------------------|-------------------|-------------------|
| Growth Rate | Historical 10yr EPS CAGR × 0.5 | Historical 10yr EPS CAGR | Historical 10yr EPS CAGR × 1.3 |
| Terminal P/E | `min(historical avg P/E, 12)` | Median historical P/E (10yr) | `max(historical avg P/E, 20)` |
| Discount Rate | Risk-free rate + 5% | Risk-free rate + 3% | Risk-free rate + 2% |

**Weighted Intrinsic Value:**
```
IV_weighted = (IV_bear × 0.25) + (IV_base × 0.50) + (IV_bull × 0.25)
```

Scenario weights from `config/filter_config.yaml → valuation.scenario_weights`.

#### Pass Criterion

```
Projected Return (base scenario) ≥ 15% annualized
```

Threshold from `config/filter_config.yaml → valuation.min_projected_return_pct`.

#### Required Line Items

| Line Item | Source |
|-----------|--------|
| Current EPS (diluted) | `income_statements[-1].eps_diluted` |
| Historical EPS (10yr, for CAGR) | `income_statements[*].eps_diluted` |
| Current Price | `market_data.price` |
| Historical P/E (10yr) | Derived: `price / eps_diluted` per year |
| 10-year government bond yield | `macro_data.treasury_10y_yield` or `macro_data.goc_10y_yield` |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| Current EPS ≤ 0 | Cannot project forward → set all scenario IVs to `NaN`. Log at WARNING. |
| Historical EPS CAGR < 0 | Bear/base/bull growth rates will be negative/low → will likely fail. Compute anyway. |
| Historical P/E unavailable (missing price history) | Use industry median P/E from config as fallback. Log substitution at WARNING. |
| Discount rate unavailable | Use config default. Log at WARNING. |

---

### F15 — Margin of Safety

**The single most important concept in value investing.**

#### Source
*The Intelligent Investor* Ch. 20 (Benjamin Graham); Buffett's shareholder letters (referenced throughout)

#### Definition

```
Margin of Safety = (Intrinsic Value − Current Price) / Intrinsic Value
```

Uses `IV_weighted` from F14.

```
MoS = (IV_weighted − Current Price) / IV_weighted
```

#### Threshold

| Business Quality | Required MoS |
|-----------------|-------------|
| High-quality (F3 ≥ 20%, F7 ≥ 40%, F10 excellent) | ≥ 25% |
| Standard quality (passes all hard filters) | ≥ 33% (Graham's original threshold) |
| Uncertain / lower-quality (borderline pass) | ≥ 40% |

The applicable threshold is determined by the composite quality score from `screener/composite_ranker.py`.
Threshold from `config/filter_config.yaml → valuation.margin_of_safety`.

#### Buy-Below Price

```
Buy-Below Price = IV_weighted × (1 − required_MoS)
```

Report three buy-below prices: one for each MoS tier (25%, 33%, 40%).

#### Required Line Items

| Line Item | Source |
|-----------|--------|
| Intrinsic Value (weighted) | F14 output |
| Current Price | `market_data.price` |

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| IV_weighted = NaN | Cannot compute MoS → set to `NaN`. |
| Current price > IV_weighted | MoS is negative → stock is overvalued relative to this estimate. Report as negative MoS, not zero. |
| IV_weighted ≤ 0 | Pathological case (negative intrinsic value) → set MoS to `NaN`, flag at ERROR. |

#### Interpretation

Graham's margin of safety exists because intrinsic value estimates are inherently
imprecise. The MoS is not a prediction — it is a buffer against estimation error
and unforeseen business deterioration. Buffett requires a large MoS for uncertain
businesses and accepts a smaller MoS for businesses he understands extremely well.

---

### F16 — Earnings Yield vs. Bond Yield

**Is the equity offering meaningful compensation over risk-free alternatives?**

#### Source
1992 Berkshire Hathaway Shareholder Letter; Hagstrom *The Warren Buffett Way* Ch. 9

#### Definition

```
Earnings Yield = EPS (diluted) / Current Market Price
               = 1 / P/E ratio

Earnings Yield Spread = Earnings Yield − 10-year Government Bond Yield
```

#### Threshold

| Spread | Interpretation |
|--------|---------------|
| ≥ 5% | Equity offers strong premium over bonds |
| 2–5% | Reasonable premium |
| < 2% | Equities expensive relative to bonds → negative scoring signal |
| ≤ 0% | Bonds are yielding more than equities → strong negative signal |

Minimum spread threshold from `config/filter_config.yaml → valuation.earnings_yield.min_spread_over_rfr_pct`.

This is a **scoring factor**, not a hard filter. A very negative spread (e.g., < 0%)
may demote a security from BUY to WATCHLIST in the recommendation engine.

#### Required Line Items

| Line Item | Source |
|-----------|--------|
| Diluted EPS | `income_statements[-1].eps_diluted` (TTM preferred) |
| Current Price | `market_data.price` |
| 10yr Government Bond Yield (US) | `macro_data.treasury_10y_yield` |
| 10yr Government Bond Yield (CA) | `macro_data.goc_10y_yield` (for TSX securities) |

#### Currency Note

For TSX-listed securities, use the **Government of Canada 10-year bond yield** (sourced
from the Bank of Canada API), not the US Treasury yield.
The appropriate yield series is selected based on `company_profile.exchange`.

#### Edge Cases

| Condition | Handling |
|-----------|----------|
| EPS ≤ 0 | Earnings yield is zero or negative → spread is negative. Report as-is. |
| Bond yield unavailable | Use last cached value (≤ 7 days). If stale, use config default and log WARNING. |
| P/E ratio > 50× | Earnings yield < 2% → very low. Valid — compute and score accordingly. |

---

## Implementation Checklist

Before closing any PR touching `metrics_engine/` or `valuation_reports/`, verify:

- [ ] Every formula reads thresholds from `config` dict, not hardcoded values
- [ ] Every missing field returns `NaN` with a `logger.warning(...)` — never `0`
- [ ] Every line-item substitution is logged with ticker, original field, substitute, confidence level
- [ ] CapEx sign is verified to be negative before use in F1 and F12
- [ ] Negative equity cases are handled explicitly in F3 and F6 (flag, don't compute)
- [ ] F14 uses three scenarios with correct weights (25/50/25)
- [ ] F15 reports negative MoS for overvalued securities (not zero)
- [ ] F16 uses GoC yield for TSX stocks, US Treasury for NYSE/NASDAQ stocks
- [ ] Unit tests exist for each formula with at least one hand-calculated expected value

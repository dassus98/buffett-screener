# REPORT_SPEC.md — Deep-Dive Report Structure, Account Logic, and Recommendation Framework

This document specifies the deep-dive analytical report produced by `valuation_reports/` for
each shortlisted security. Load this file when working on `valuation_reports/` or `output/`.

---

## 1. Deep-Dive Report Template

**File:** `valuation_reports/templates/deep_dive_template.md` (Jinja2)
**Output:** `data/reports/{TICKER}_analysis.md`
**Rendered by:** `valuation_reports/report_generator.py`

The full template structure is shown below. All `{{ }}` blocks are Jinja2 variables. All
thresholds referenced in conditional logic must come from config — do not hardcode numbers
in the template.

---

```
# {{ company_name }} ({{ ticker }}) — Buffett Analysis Report

Generated: {{ report_date }} | Data as of: {{ latest_fiscal_year }}

---

## Executive Summary

| | |
|---|---|
| Composite Buffett Score | {{ composite_score }}/100 |
| Weighted Intrinsic Value | ${{ iv_weighted | round(2) }} USD |
| Current Price | ${{ current_price_usd | round(2) }} USD |
| Margin of Safety | {{ margin_of_safety_pct | round(1) }}% |
| Recommendation | **{{ recommendation }}** |
| Confidence Level | {{ confidence_level }} |
| Recommended Account | {{ account_recommendation }} |
| Time Horizon | {{ time_horizon_years }}+ years |

{% if critical_flags %}
> **Data Quality Flags:** {{ critical_flags | join('; ') }}
{% endif %}

---

## Durable Competitive Advantage Assessment

{% if qualitative_enabled %}
**Moat Type:** {{ moat_type }}

{{ moat_evidence }}

**Key Threats to Moat:**
{% for threat in moat_threats %}
- {{ threat }}
{% endfor %}
{% else %}
*Qualitative moat assessment disabled. Set `reports.enable_qualitative: true` in
config/filter_config.yaml and provide an LLM API key to enable.*

**Indicators from quantitative data:**
- 10-year average gross margin: {{ gross_margin_avg_10yr | pct }} —
  {{ "Consistent with pricing power" if gross_margin_avg_10yr > config.moat_gross_margin_threshold else "Below moat threshold" }}
- 10-year average ROE: {{ roe_avg_10yr | pct }} —
  {{ "Consistent with durable advantage" if roe_avg_10yr > config.moat_roe_threshold else "Below moat threshold" }}
- Gross margin standard deviation: {{ gross_margin_std | pct }} —
  {{ "Stable" if gross_margin_std < config.moat_margin_stability_threshold else "Variable — moat may be weaker than headline numbers suggest" }}
{% endif %}

---

## Financial Statement Analysis (10-Year)

### Income Statement Tests

| Test | 10-Year Result | Threshold | Pass/Fail | Notes |
|---|---|---|---|---|
| Earnings Consistency (F10) | {{ profitable_years }}/10 years profitable | ≥ {{ config.hard_filters.min_profitable_years }} | {{ "✓" if f10_pass else "✗" }} | {{ f10_notes }} |
| Average ROE (F3) | {{ roe_avg_10yr | pct }} | ≥ {{ config.hard_filters.min_roe_avg | pct }} | {{ "✓" if f3_pass else "✗" }} | {{ f3_notes }} |
| Gross Margin (F7) | {{ gross_margin_avg_10yr | pct }} | ≥ {{ config.hard_filters.min_gross_margin | pct }} | {{ "✓" if f7_pass else "✗" }} | Median: {{ gross_margin_median_10yr | pct }} |
| SG&A / Gross Profit (F8) | {{ sga_ratio_median_10yr | pct }} | < {{ config.hard_filters.max_sga_ratio | pct }} | {{ "✓" if f8_pass else "✗" }} | {{ f8_notes }} |
| EPS CAGR (F11) | {{ eps_cagr_10yr | pct }} | > {{ config.hard_filters.min_eps_cagr | pct }} | {{ "✓" if f11_pass else "✗" }} | {{ eps_decline_years }} decline years |
| Net Margin Trend (F10) | {{ net_margin_avg_10yr | pct }} | Positive trend | {{ "✓" if net_margin_trend_positive else "✗" }} | {{ net_margin_notes }} |

**Year-by-Year Income Statement Detail:**

| Fiscal Year | Revenue ($K) | Gross Margin | Operating Margin | Net Margin | EPS (Diluted) | ROE |
|---|---|---|---|---|---|---|
{% for row in annual_income %}
| {{ row.fiscal_year }} | {{ row.revenue | thousands }} | {{ row.gross_margin | pct }} | {{ row.operating_margin | pct }} | {{ row.net_margin | pct }} | {{ row.eps_diluted }} | {{ row.roe | pct }} |
{% endfor %}

### Balance Sheet Tests

| Test | 10-Year Result | Threshold | Pass/Fail | Notes |
|---|---|---|---|---|
| Debt Payoff Years (F5) | {{ debt_payoff_years | round(1) }} years | ≤ {{ config.hard_filters.max_debt_payoff_years }} | {{ "✓" if f5_pass else "✗" }} | Based on owner earnings |
| Debt-to-Equity (F6) | {{ de_ratio_latest | round(2) }} | < {{ config.hard_filters.max_de_ratio }} | {{ "✓" if f6_pass else "✗" }} | 10-yr avg: {{ de_ratio_avg_10yr | round(2) }} |
| Return on Retained Earnings (F4) | {{ return_on_retained_earnings | pct }} | ≥ {{ config.hard_filters.min_return_on_retained | pct }} | {{ "✓" if f4_pass else "✗" }} | {{ f4_notes }} |
| Equity Growth | {{ equity_cagr_10yr | pct }} | Positive | {{ "✓" if equity_cagr_10yr > 0 else "✗" }} | |

{% if negative_equity_flag %}
> **Flag:** Negative shareholders' equity detected in {{ negative_equity_years }} year(s).
> F3 (ROE) and F6 (D/E) are unreliable for those periods. Assessment relies on F5, F7, F10, F11.
{% endif %}

**Year-by-Year Balance Sheet Detail:**

| Fiscal Year | Long-Term Debt ($K) | Shareholders' Equity ($K) | D/E Ratio | Retained Earnings ($K) |
|---|---|---|---|---|
{% for row in annual_balance %}
| {{ row.fiscal_year }} | {{ row.long_term_debt | thousands }} | {{ row.shareholders_equity | thousands }} | {{ row.de_ratio | round(2) }} | {{ row.retained_earnings | thousands }} |
{% endfor %}

### Cash Flow Tests

| Test | 10-Year Result | Threshold | Pass/Fail | Notes |
|---|---|---|---|---|
| Owner Earnings (F1, latest yr) | ${{ f1_owner_earnings_latest | thousands }}K | Positive | {{ "✓" if f1_owner_earnings_latest > 0 else "✗" }} | |
| Owner Earnings Yield (F2) | {{ f2_initial_return | pct }} | ≥ {{ config.hard_filters.min_owner_earnings_yield | pct }} | {{ "✓" if f2_pass else "✗" }} | |
| FCF Positive Years | {{ fcf_positive_years }}/10 | ≥ {{ config.hard_filters.min_fcf_positive_years }} | {{ "✓" if fcf_consistency_pass else "✗" }} | |
| CapEx / Net Income (F12) | {{ capex_to_ni_median | pct }} | < {{ config.hard_filters.max_capex_to_ni | pct }} | {{ "✓" if f12_pass else "✗" }} | {{ f12_notes }} |
| Interest Coverage (F9) | {{ interest_coverage_avg | round(1) }}× | ≥ {{ config.hard_filters.min_interest_coverage }}× | {{ "✓" if f9_pass else "✗" }} | |
| Buyback Activity (F13) | {{ buyback_summary }} | Net reducer preferred | {{ "✓" if f13_positive else "—" }} | {{ net_share_change_pct | pct }} over 10yr |

{% if capex_flag %}
> **Flag:** Total CapEx exceeds 2× D&A in {{ capex_flag_years }} year(s). Maintenance CapEx
> estimate (≈ D&A) may be materially understated. Owner earnings figures should be interpreted
> conservatively. See FORMULAS.md F1 implementation notes.
{% endif %}

**Year-by-Year Cash Flow Detail:**

| Fiscal Year | Operating CF ($K) | CapEx ($K) | Free CF ($K) | Owner Earnings ($K) | D&A ($K) |
|---|---|---|---|---|---|
{% for row in annual_cashflow %}
| {{ row.fiscal_year }} | {{ row.operating_cash_flow | thousands }} | {{ row.capital_expenditures | thousands }} | {{ row.free_cash_flow | thousands }} | {{ row.owner_earnings | thousands }} | {{ row.depreciation_amortization | thousands }} |
{% endfor %}

---

## Valuation — Three Scenarios

**Base inputs:** EPS (latest): ${{ eps_latest }} | 10-yr EPS CAGR: {{ eps_cagr_10yr | pct }} |
Avg P/E (10yr): {{ pe_avg_10yr | round(1) }} | Risk-free rate: {{ risk_free_rate | pct }}

| Scenario | EPS Growth | Terminal P/E | Discount Rate | Intrinsic Value | Probability Weight |
|---|---|---|---|---|---|
| Bear | {{ bear_growth | pct }} | {{ bear_terminal_pe | round(1) }}× | {{ bear_discount_rate | pct }} | ${{ iv_bear | round(2) }} | 25% |
| Base | {{ base_growth | pct }} | {{ base_terminal_pe | round(1) }}× | {{ base_discount_rate | pct }} | ${{ iv_base | round(2) }} | 50% |
| Bull | {{ bull_growth | pct }} | {{ bull_terminal_pe | round(1) }}× | {{ bull_discount_rate | pct }} | ${{ iv_bull | round(2) }} | 25% |
| **Weighted Average** | | | | **${{ iv_weighted | round(2) }}** | **100%** |

*Projection period: {{ config.valuation.projection_years }} years.
Terminal growth rate: {{ config.valuation.terminal_growth_rate | pct }}.
Scenario weights and growth multipliers from config/filter_config.yaml.*

### Margin of Safety

| | Value |
|---|---|
| Weighted Intrinsic Value | ${{ iv_weighted | round(2) }} |
| Current Price | ${{ current_price_usd | round(2) }} |
| Margin of Safety | **{{ margin_of_safety_pct | round(1) }}%** |
| Conservative buy below ({{ config.valuation.mos_conservative | pct }} MoS) | ${{ buy_below_conservative | round(2) }} |
| Moderate buy below ({{ config.valuation.mos_moderate | pct }} MoS) | ${{ buy_below_moderate | round(2) }} |

{{ margin_of_safety_interpretation }}

### Earnings Yield vs. Bond Yield (F16)

| | Value |
|---|---|
| Earnings Yield (EPS / Price) | {{ earnings_yield | pct }} |
| 10-Year Bond Yield ({{ bond_yield_type }}) | {{ bond_yield | pct }} |
| Spread | {{ earnings_yield_spread | pct }} |
| Interpretation | {{ earnings_yield_interpretation }} |

*Bond yield used: {{ "GoC 10yr (TSX security)" if exchange == "TSX" else "US Treasury 10yr (US security)" }}.*

---

## Sensitivity Analysis

Intrinsic value under alternative assumptions (Base scenario, USD):

**EPS Growth Rate vs. Terminal P/E:**

| EPS Growth → | {{ sens_growth_low | pct }} | {{ sens_growth_mid | pct }} | {{ sens_growth_high | pct }} |
|---|---|---|---|
| P/E {{ sens_pe_low }}× | ${{ sens_grid[0][0] | round(0) }} | ${{ sens_grid[0][1] | round(0) }} | ${{ sens_grid[0][2] | round(0) }} |
| P/E {{ sens_pe_mid }}× | ${{ sens_grid[1][0] | round(0) }} | ${{ sens_grid[1][1] | round(0) }} | ${{ sens_grid[1][2] | round(0) }} |
| P/E {{ sens_pe_high }}× | ${{ sens_grid[2][0] | round(0) }} | ${{ sens_grid[2][1] | round(0) }} | ${{ sens_grid[2][2] | round(0) }} |

*EPS growth varied ±30% from base; Terminal P/E varied ±25% from base.*

**Discount Rate Sensitivity (Base growth, Base P/E):**

| Discount Rate | {{ sens_rate_low | pct }} | {{ sens_rate_mid | pct }} | {{ sens_rate_high | pct }} |
|---|---|---|---|
| Intrinsic Value | ${{ sens_rate_grid[0] | round(0) }} | ${{ sens_rate_grid[1] | round(0) }} | ${{ sens_rate_grid[2] | round(0) }} |

*Discount rate varied ±200 bps from base.*

---

## Assumption Log

| Assumption | Confidence | Failure Mode | Consequence to Thesis |
|---|---|---|---|
{% for row in assumption_log %}
| {{ row.assumption }} | {{ row.confidence }} | {{ row.failure_mode }} | {{ row.consequence }} |
{% endfor %}

*Assumptions are auto-populated based on data quality checks and substitutions used.
See data/processed/data_quality_report.csv for full substitution log.*

---

## Devil's Advocate — Bear Case

{% for argument in bear_case_arguments %}
**{{ loop.index }}. {{ argument.title }}**

{{ argument.body }}

{% endfor %}

*Bear case arguments are generated from quantitative flags (margin pressure, debt trends,
CapEx intensity) supplemented by LLM qualitative analysis if enabled.*

---

## Investment Strategy

**Entry Price Target:** ${{ buy_below_moderate | round(2) }} USD ({{ config.valuation.mos_moderate | pct }} margin of safety)
{% if exchange == "TSX" %}
**Entry Price (CAD equivalent):** ${{ buy_below_moderate_cad | round(2) }} CAD (at {{ usd_cad_rate }} USD/CAD)
{% endif %}

**Time Horizon:** {{ time_horizon_years }}+ years

**Position Sizing Guidance:**
{{ position_sizing_guidance }}

**Signs to Reconsider / Sell Triggers:**

| Trigger | Threshold | Current Value | Status |
|---|---|---|---|
{% for trigger in sell_triggers %}
| {{ trigger.name }} | {{ trigger.threshold }} | {{ trigger.current_value }} | {{ trigger.status }} |
{% endfor %}

**Recommended Account:** {{ account_recommendation }}

{{ account_reasoning }}
```

---

## 2. Template Context Variables Reference

All variables passed to the Jinja2 template from `report_generator.py`:

| Variable | Type | Source module |
|---|---|---|
| `company_name` | str | `universe` table |
| `ticker` | str | `universe` table |
| `exchange` | str | `universe` table |
| `report_date` | str | `datetime.date.today()` |
| `latest_fiscal_year` | int | Max fiscal year in `buffett_metrics` |
| `composite_score` | float | `buffett_metrics_summary` |
| `iv_bear`, `iv_base`, `iv_bull`, `iv_weighted` | float | `intrinsic_value.py` |
| `current_price_usd` | float | `market_data` table |
| `margin_of_safety_pct` | float | `margin_of_safety.py` |
| `recommendation` | str | `recommendation.py` |
| `confidence_level` | str | `recommendation.py` |
| `account_recommendation` | str | `recommendation.py` |
| `account_reasoning` | str | `recommendation.py` |
| `time_horizon_years` | int | `recommendation.py` |
| `critical_flags` | list[str] | `data_quality_log` table |
| `annual_income` | list[dict] | `income_statement` + `buffett_metrics` tables |
| `annual_balance` | list[dict] | `balance_sheet` + `buffett_metrics` tables |
| `annual_cashflow` | list[dict] | `cash_flow` + `buffett_metrics` tables |
| `assumption_log` | list[dict] | Auto-generated from substitutions + edge cases |
| `bear_case_arguments` | list[dict] | `qualitative_prompts.py` |
| `sell_triggers` | list[dict] | `recommendation.py` |
| `sens_grid` | list[list[float]] | `intrinsic_value.py` sensitivity computation |
| `sens_rate_grid` | list[float] | `intrinsic_value.py` sensitivity computation |
| `config` | dict | `filter_config_loader` |
| `qualitative_enabled` | bool | `config.reports.enable_qualitative` |

**Custom Jinja2 filters** (registered in `report_generator.py`):
- `pct`: Multiply by 100, format as `"12.3%"`
- `thousands`: Format as `"1,234"` (values already in USD thousands)
- `round(n)`: Standard rounding

---

## 3. Account Recommendation Logic (RRSP vs TFSA)

Implemented in `recommendation.py`. Takes `ticker`, `exchange`, `dividend_yield`,
`expected_return` as inputs. All thresholds from config.

| Security Type | Recommended Account | Reasoning |
|---|---|---|
| US-listed, dividend-paying (yield ≥ threshold) | **RRSP** | US dividends are exempt from the 15% IRS withholding tax under the Canada-US tax treaty — but only in RRSP, not TFSA. TFSA is not recognized as a pension account under the treaty. |
| US-listed, non-dividend / growth | **Either** (slight TFSA preference) | No withholding tax issue. TFSA gains are fully tax-free; RRSP gains are taxed as ordinary income on withdrawal. For high-growth holdings, TFSA advantage compounds over time. |
| Canadian-listed, dividend-paying | **TFSA** | Canadian dividends in TFSA are fully tax-free including the dividend tax credit benefit. In RRSP, Canadian dividends lose the dividend tax credit and are taxed as ordinary income on withdrawal. |
| Canadian-listed, growth | **TFSA** | Tax-free compounding is optimal. No dividend tax credit issue. |
| High expected return, long horizon | **TFSA** if room available | Larger terminal value = larger absolute tax savings in TFSA. Prioritize highest-return holdings in TFSA when contribution room is constrained. |

**Config keys:**
```yaml
reports:
  rrsp_us_dividend_yield_threshold: 0.01   # ≥ 1% yield triggers RRSP recommendation for US stocks
  tfsa_preference_min_expected_return: 0.12  # ≥ 12% expected return → prioritize TFSA
```

**Decision logic in `recommendation.py`:**
```
if exchange in ("NYSE", "NASDAQ", "AMEX"):
    if dividend_yield >= config.rrsp_us_dividend_yield_threshold:
        account = "RRSP"
        reason = "US dividends exempt from 15% withholding tax in RRSP under Canada-US treaty."
    else:
        account = "Either (TFSA preferred)"
        reason = "No withholding tax issue. TFSA gains are fully tax-free vs. income on RRSP withdrawal."
elif exchange == "TSX":
    account = "TFSA"
    reason = "Canadian dividends and capital gains fully tax-free in TFSA. RRSP loses dividend tax credit."
```

**Note:** This is general guidance based on common Canadian tax rules, not personalized tax
advice. Users should consult a tax professional for their specific situation.

---

## 4. Recommendation Decision Logic

Implemented in `recommendation.py`. All thresholds from `config/filter_config.yaml` under
`reports.recommendation_thresholds`.

| Condition | Recommendation |
|---|---|
| MoS ≥ 25% AND composite score ≥ 70 AND no critical data quality flags | **BUY** |
| MoS 10–25% AND composite score ≥ 60 | **HOLD** (wait for better entry price) |
| MoS < 10% OR composite score < 60 | **PASS** |
| Any critical data quality flag (ERROR-level in `data_quality_log`) | **PASS** with note: *"Insufficient data confidence"* |

**Config keys:**
```yaml
reports:
  recommendation_thresholds:
    buy_min_mos: 0.25
    buy_min_composite_score: 70
    hold_min_mos: 0.10
    hold_min_composite_score: 60
```

### Confidence Levels

Assigned independently of recommendation tier. A HOLD can be High confidence; a BUY can be Low.

| Level | Criteria |
|---|---|
| **High** | ≥ 8 of 10 years complete data AND 0 line-item substitutions AND MoS > 30% |
| **Moderate** | ≥ 8 years complete data AND ≤ 2 line-item substitutions AND MoS 15–30% |
| **Low** | Any of: < 8 years data, > 2 substitutions, MoS < 15%, or any edge case flag (negative equity, D&A estimated, EPS CAGR from partial window) |

**Config keys:**
```yaml
reports:
  confidence_thresholds:
    high_min_mos: 0.30
    high_max_substitutions: 0
    moderate_max_substitutions: 2
    moderate_min_mos: 0.15
```

### Position Sizing Guidance

Position size is expressed as a fraction of a model portfolio. Driven by confidence level
and composite score. Thresholds from config.

| Confidence + Score | Suggested Allocation |
|---|---|
| High + score ≥ 80 | Up to 8–10% of portfolio |
| High + score 70–79 | Up to 5–7% |
| Moderate + score ≥ 70 | Up to 3–5% |
| Moderate + score 60–69 | Up to 2–3% |
| Low (any score) | 1–2% maximum; monitor closely |

These ranges are presented as guidance, not instructions. Always displayed with the
disclaimer: *"Position sizing should reflect your personal risk tolerance, portfolio
concentration, and tax situation."*

### Time Horizon

Determined by the expected compounding period implied by the intrinsic value model:
- Standard: 10 years (matches the F14 projection window)
- Extended (15+): If composite score ≥ 80 and EPS CAGR > 15% — business quality justifies
  a longer hold
- Shortened (5): If MoS is thin or confidence is Low — shorter hold reduces exposure to
  model error

---

## 5. Sell Signal Framework

Generated per security by `recommendation.py`. Default thresholds from config under
`reports.sell_signals`. All are observable from subsequent annual data pulls.

| Trigger | Default Threshold | Config Key | Notes |
|---|---|---|---|
| Fundamental deterioration | ROE drops below 12% for 2 consecutive years | `sell_signals.roe_floor`, `sell_signals.roe_consecutive_years` | Does not trigger on a single bad year |
| Moat erosion | Gross margin declines > 5 percentage points over any 3-year window | `sell_signals.gross_margin_decline_pp`, `sell_signals.gross_margin_window_years` | Compares 3-year rolling average |
| Leverage spike | D/E exceeds 1.0, OR debt payoff years exceed 5 | `sell_signals.max_de_ratio`, `sell_signals.max_debt_payoff_years` | Either condition sufficient |
| Overvaluation | Price exceeds bull-case intrinsic value (MoS becomes negative vs. bull IV) | `sell_signals.overvaluation_vs_scenario` | Uses bull IV, not weighted IV |
| Capital misallocation | Return on retained earnings (F4) drops below 8% | `sell_signals.min_return_on_retained` | Computed on trailing 5-year basis |

**Sell signal output in reports:**
Each trigger is listed in the Investment Strategy section with:
1. Trigger name and threshold
2. Current value (at report generation date)
3. Status: `Watch` (approaching threshold), `Triggered` (threshold breached), `Clear`

**Triggered sell signals** cause the recommendation to be overridden to **PASS** regardless
of MoS or composite score. Log at WARNING: `"[TICKER] sell signal triggered: [TRIGGER NAME]"`.

---

## 6. Summary Report Template

**File:** `valuation_reports/templates/summary_table_template.md` (Jinja2)
**Output:** `data/reports/summary.md`

```
# Buffett Screener — Top {{ top_n }} Summary

Generated: {{ run_date }} | Universe: {{ universe_size }} tickers screened |
Passed hard filters: {{ passed_hard_filters }}

**Macro context:** US 10yr: {{ macro.us_treasury_10yr | pct }} |
GoC 10yr: {{ macro.goc_bond_10yr | pct }} | USD/CAD: {{ macro.usd_cad_rate }}

| Rank | Ticker | Company | Exchange | Score | MoS | Rec. | Confidence | Account | Gross Margin | ROE (10yr) | EPS CAGR |
|---|---|---|---|---|---|---|---|---|---|---|---|
{% for row in rows %}
| {{ row.rank }} | {{ row.ticker }} | {{ row.company_name }} | {{ row.exchange }} | {{ row.composite_score | round(1) }} | {{ row.margin_of_safety_pct | round(1) }}% | {{ row.recommendation }} | {{ row.confidence_level }} | {{ row.account_recommendation }} | {{ row.gross_margin_avg_10yr | pct }} | {{ row.roe_avg_10yr | pct }} | {{ row.eps_cagr_10yr | pct }} |
{% endfor %}

---

## Screener Statistics

| Stage | Tickers Remaining |
|---|---|
| Universe (post-eligibility) | {{ universe_size }} |
| After sector exclusions | {{ after_exclusions }} |
| After hard filters | {{ passed_hard_filters }} |
| Shortlist (top {{ top_n }}) | {{ top_n }} |

**Hard filter elimination breakdown:**
{% for filter_name, count in filter_stats.items() %}
- {{ filter_name }}: {{ count }} tickers eliminated
{% endfor %}
```

---

## 7. Assumption Log Auto-Generation

`report_generator.py` auto-populates the Assumption Log from pipeline outputs. Each row
is generated when any of the following conditions are detected for the ticker:

| Condition | Assumption | Confidence | Failure Mode | Consequence |
|---|---|---|---|---|
| D&A sourced from IS (not CF) | "Depreciation approximated from income statement" | Medium | IS D&A may include non-cash charges unrelated to operating asset wear | Owner earnings (F1) may be overstated |
| CapEx > 2× D&A in any year | "Maintenance CapEx approximated as D&A" | Low | Growth CapEx included in D&A proxy; maintenance CapEx understated | Owner earnings overstated; real yield lower than reported |
| SG&A derived (not direct) | "SG&A estimated as residual of operating expenses" | Medium | Residual may include items not properly SG&A | F8 score directionally correct but imprecise |
| EPS CAGR from < 10-year window | "EPS CAGR computed from {{ eps_cagr_window }}-year window (partial history)" | Low | Shorter window may not capture full cycle | F11 and F14 growth projection based on fewer data points |
| GoC yield from FRED IRLTLT01CAM156N | "GoC 10yr yield sourced from FRED monthly series" | High | Monthly series lags daily market rate by up to 30 days | F16 spread may be slightly stale |
| Currency conversion applied (TSX) | "Financial statements converted from CAD to USD at {{ usd_cad_rate }}" | High | Exchange rate fluctuation affects USD-equivalent metrics | All USD metrics are rate-sensitive for TSX securities |
| Negative equity in any year | "Shareholders' equity negative in {{ negative_equity_years }} year(s)" | Low | F3 (ROE) and F6 (D/E) unreliable for affected years | Quality assessment relies more heavily on F5, F7, F10, F11 |
| Retained earnings unavailable | "F4 (return on retained earnings) estimated from cumulative net income minus dividends" | Low | Cumulative approximation drifts from reported retained earnings over time | F4 score directionally correct; magnitude uncertain |

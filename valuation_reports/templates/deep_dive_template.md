# {{ company_name }} ({{ ticker }}) — Buffett Analysis Report

Generated: {{ report_date }} | Data as of: {{ latest_fiscal_year }}

---

## Executive Summary

| | |
|---|---|
| Composite Buffett Score | {{ composite_score | round(1) }}/100 |
| Weighted Intrinsic Value | ${{ iv_weighted | round(2) }} USD |
| Current Price | ${{ current_price_usd | round(2) }} USD |
| Margin of Safety | {{ (margin_of_safety_pct * 100) | round(1) }}% |
| Recommendation | **{{ recommendation }}** |
| Confidence Level | {{ confidence_level }} |
| Recommended Account | {{ account_recommendation }} |
| Time Horizon | {{ time_horizon_years }}+ years |

{% if critical_flags %}
> **Data Quality Flags:** {{ critical_flags | join('; ') }}
{% endif %}

---

## Durable Competitive Advantage Assessment

{% if qualitative_enabled and moat_assessment %}
**Moat Type:** {{ moat_assessment.moat_type }}

{{ moat_assessment.evidence }}

**Key Threats to Moat:**
{% for threat in moat_assessment.threats %}
- {{ threat }}
{% endfor %}
{% else %}
*Qualitative moat assessment disabled. Set `reports.enable_qualitative: true` in
config/filter_config.yaml and provide an LLM API key to enable.*

**Indicators from quantitative data:**
{% if moat_indicators %}
{% for indicator in moat_indicators %}
- {{ indicator }}
{% endfor %}
{% else %}
- 10-year average gross margin: {{ (gross_margin_avg_10yr * 100) | round(1) }}%
- 10-year average ROE: {{ (roe_avg_10yr * 100) | round(1) }}%
{% endif %}
{% endif %}

---

## Financial Statement Analysis (10-Year)

### Income Statement Tests

| Test | 10-Year Result | Threshold | Pass/Fail | Notes |
|---|---|---|---|---|
{% for test in income_tests %}
| {{ test.name }} | {{ test.result }} | {{ test.threshold }} | {{ test.status }} | {{ test.notes }} |
{% endfor %}

{% if annual_income %}
**Year-by-Year Income Statement Detail:**

| Fiscal Year | Revenue ($K) | Gross Margin | Operating Margin | Net Margin | EPS (Diluted) | ROE |
|---|---|---|---|---|---|---|
{% for row in annual_income %}
| {{ row.fiscal_year }} | {{ row.revenue | round(0) }} | {{ (row.gross_margin * 100) | round(1) }}% | {{ (row.operating_margin * 100) | round(1) }}% | {{ (row.net_margin * 100) | round(1) }}% | {{ row.eps_diluted | round(2) }} | {{ (row.roe * 100) | round(1) }}% |
{% endfor %}
{% endif %}

### Balance Sheet Tests

| Test | 10-Year Result | Threshold | Pass/Fail | Notes |
|---|---|---|---|---|
{% for test in balance_tests %}
| {{ test.name }} | {{ test.result }} | {{ test.threshold }} | {{ test.status }} | {{ test.notes }} |
{% endfor %}

{% if negative_equity_flag %}
> **Flag:** Negative shareholders' equity detected in {{ negative_equity_years }} year(s).
> F3 (ROE) and F6 (D/E) are unreliable for those periods. Assessment relies on F5, F7, F10, F11.
{% endif %}

{% if annual_balance %}
**Year-by-Year Balance Sheet Detail:**

| Fiscal Year | Long-Term Debt ($K) | Shareholders' Equity ($K) | D/E Ratio | Retained Earnings ($K) |
|---|---|---|---|---|
{% for row in annual_balance %}
| {{ row.fiscal_year }} | {{ row.long_term_debt | round(0) }} | {{ row.shareholders_equity | round(0) }} | {{ row.de_ratio | round(2) }} | {{ row.retained_earnings | round(0) }} |
{% endfor %}
{% endif %}

### Cash Flow Tests

| Test | 10-Year Result | Threshold | Pass/Fail | Notes |
|---|---|---|---|---|
{% for test in cashflow_tests %}
| {{ test.name }} | {{ test.result }} | {{ test.threshold }} | {{ test.status }} | {{ test.notes }} |
{% endfor %}

{% if capex_flag %}
> **Flag:** Total CapEx exceeds 2x D&A in {{ capex_flag_years }} year(s). Maintenance CapEx
> estimate (approximately D&A) may be materially understated. Owner earnings figures should be
> interpreted conservatively. See FORMULAS.md F1 implementation notes.
{% endif %}

{% if annual_cashflow %}
**Year-by-Year Cash Flow Detail:**

| Fiscal Year | Operating CF ($K) | CapEx ($K) | Free CF ($K) | Owner Earnings ($K) | D&A ($K) |
|---|---|---|---|---|---|
{% for row in annual_cashflow %}
| {{ row.fiscal_year }} | {{ row.operating_cash_flow | round(0) }} | {{ row.capital_expenditures | round(0) }} | {{ row.free_cash_flow | round(0) }} | {{ row.owner_earnings | round(0) }} | {{ row.depreciation_amortization | round(0) }} |
{% endfor %}
{% endif %}

---

## Valuation — Three Scenarios

**Base inputs:** EPS (latest): ${{ eps_latest | round(2) }} | 10-yr EPS CAGR: {{ (eps_cagr_10yr * 100) | round(1) }}% |
Avg P/E (10yr): {{ pe_avg_10yr | round(1) }} | Risk-free rate: {{ (risk_free_rate * 100) | round(2) }}%

| Scenario | EPS Growth | Terminal P/E | Discount Rate | Intrinsic Value | Probability Weight |
|---|---|---|---|---|---|
| Bear | {{ (bear_growth * 100) | round(1) }}% | {{ bear_terminal_pe | round(1) }}x | {{ (bear_discount_rate * 100) | round(1) }}% | ${{ iv_bear | round(2) }} | {{ (bear_probability * 100) | round(0) }}% |
| Base | {{ (base_growth * 100) | round(1) }}% | {{ base_terminal_pe | round(1) }}x | {{ (base_discount_rate * 100) | round(1) }}% | ${{ iv_base | round(2) }} | {{ (base_probability * 100) | round(0) }}% |
| Bull | {{ (bull_growth * 100) | round(1) }}% | {{ bull_terminal_pe | round(1) }}x | {{ (bull_discount_rate * 100) | round(1) }}% | ${{ iv_bull | round(2) }} | {{ (bull_probability * 100) | round(0) }}% |
| **Weighted Average** | | | | **${{ iv_weighted | round(2) }}** | **100%** |

*Projection period: {{ projection_years }} years.
Terminal growth rate: {{ (terminal_growth_rate * 100) | round(1) }}%.
Scenario weights and growth multipliers from config/filter_config.yaml.*

### Margin of Safety

| | Value |
|---|---|
| Weighted Intrinsic Value | ${{ iv_weighted | round(2) }} |
| Current Price | ${{ current_price_usd | round(2) }} |
| Margin of Safety | **{{ (margin_of_safety_pct * 100) | round(1) }}%** |
| Conservative buy below ({{ (mos_conservative * 100) | round(0) }}% MoS) | ${{ buy_below_conservative | round(2) }} |
| Moderate buy below ({{ (mos_moderate * 100) | round(0) }}% MoS) | ${{ buy_below_moderate | round(2) }} |

{{ margin_of_safety_interpretation }}

### Earnings Yield vs. Bond Yield (F16)

| | Value |
|---|---|
| Earnings Yield (EPS / Price) | {{ (earnings_yield * 100) | round(2) }}% |
| 10-Year Bond Yield ({{ bond_yield_type }}) | {{ (bond_yield * 100) | round(2) }}% |
| Spread | {{ (earnings_yield_spread * 100) | round(2) }}% |
| Interpretation | {{ earnings_yield_interpretation }} |

*Bond yield used: {{ "GoC 10yr (TSX security)" if exchange == "TSX" else "US Treasury 10yr (US security)" }}.*

---

{% if sensitivity_data %}
## Sensitivity Analysis

Intrinsic value under alternative assumptions (Base scenario, USD):

{% if sensitivity_data.eps_sensitivity %}
**EPS Growth Rate Sensitivity:**

| EPS Growth | Intrinsic Value | Margin of Safety |
|---|---|---|
{% for step in sensitivity_data.eps_sensitivity %}
| {{ (step[0] * 100) | round(1) }}% | ${{ step[1] | round(2) }} | {{ (step[2] * 100) | round(1) }}% |
{% endfor %}

*EPS growth varied +/-30% from base.*
{% endif %}

{% if sensitivity_data.pe_sensitivity %}
**Terminal P/E Sensitivity:**

| Terminal P/E | Intrinsic Value | Margin of Safety |
|---|---|---|
{% for step in sensitivity_data.pe_sensitivity %}
| {{ step[0] | round(1) }}x | ${{ step[1] | round(2) }} | {{ (step[2] * 100) | round(1) }}% |
{% endfor %}

*Terminal P/E varied +/-25% from base.*
{% endif %}

{% if sensitivity_data.discount_sensitivity %}
**Discount Rate Sensitivity:**

| Discount Rate | Intrinsic Value | Margin of Safety |
|---|---|---|
{% for step in sensitivity_data.discount_sensitivity %}
| {{ (step[0] * 100) | round(2) }}% | ${{ step[1] | round(2) }} | {{ (step[2] * 100) | round(1) }}% |
{% endfor %}

*Discount rate varied +/-200 bps from base.*
{% endif %}

---
{% endif %}

## Assumption Log

{% if assumption_log %}
| Assumption | Confidence | Failure Mode | Consequence to Thesis |
|---|---|---|---|
{% for row in assumption_log %}
| {{ row.assumption }} | {{ row.confidence }} | {{ row.failure_mode }} | {{ row.consequence }} |
{% endfor %}

*Assumptions are auto-populated based on data quality checks and substitutions used.
See data/processed/data_quality_report.csv for full substitution log.*
{% else %}
*No material assumptions or substitutions flagged for this security.*
{% endif %}

---

## Devil's Advocate — Bear Case

{% if bear_case_arguments %}
{% for argument in bear_case_arguments %}
**{{ loop.index }}. {{ argument.title }}**

{{ argument.body }}

{% endfor %}
{% else %}
*No bear case arguments generated. Enable qualitative analysis or review quantitative flags manually.*
{% endif %}

*Bear case arguments are generated from quantitative flags (margin pressure, debt trends,
CapEx intensity) supplemented by LLM qualitative analysis if enabled.*

---

## Investment Strategy

**Entry Price Target:** ${{ buy_below_moderate | round(2) }} USD ({{ (mos_moderate * 100) | round(0) }}% margin of safety)
{% if exchange == "TSX" and buy_below_moderate_cad is defined %}
**Entry Price (CAD equivalent):** ${{ buy_below_moderate_cad | round(2) }} CAD (at {{ usd_cad_rate }} USD/CAD)
{% endif %}

**Time Horizon:** {{ time_horizon_years }}+ years

**Position Sizing Guidance:**
{{ position_sizing_guidance }}

*Position sizing should reflect your personal risk tolerance, portfolio concentration, and tax situation.*

**Signs to Reconsider / Sell Triggers:**

{% if sell_triggers %}
| Trigger | Threshold | Current Value | Status |
|---|---|---|---|
{% for trigger in sell_triggers %}
| {{ trigger.signal }} | {{ trigger.threshold }} | {{ trigger.current_value }} | {{ trigger.status }} |
{% endfor %}
{% else %}
*No sell triggers evaluated.*
{% endif %}

**Recommended Account:** {{ account_recommendation }}

{{ account_reasoning }}

---

## Data Quality Notes

{% if data_quality %}
| Metric | Value |
|---|---|
| Years of data available | {{ data_quality.years_available }} |
| Line-item substitutions | {{ data_quality.substitutions_count }} |
{% if data_quality.missing_critical_fields %}
| Missing critical fields | {{ data_quality.missing_critical_fields }} |
{% endif %}
{% if data_quality.drop_reason %}
| Quality flag | {{ data_quality.drop_reason }} |
{% endif %}
{% else %}
*Data quality information not available.*
{% endif %}

---

*This report was generated by the Buffett Screener pipeline. All thresholds and parameters
are sourced from config/filter_config.yaml. This is not investment advice. Past performance
does not guarantee future results. Consult a qualified financial advisor before making
investment decisions.*

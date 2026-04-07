# {{ ticker }} — Deep Dive Valuation Report

**Company:** {{ bundle.profile.name }}
**Sector / Industry:** {{ bundle.profile.sector }} / {{ bundle.profile.industry }}
**Report Date:** {{ report_date }}

---

## Recommendation: {{ recommendation.recommendation.value }}

> **Composite Score:** {{ recommendation.composite_score | round(1) }} / 100
> **Quality Score:** {{ recommendation.quality_score | round(1) }} / 100
> **Value Score:** {{ recommendation.value_score | round(1) }} / 100

{{ recommendation.justification }}

---

## Intrinsic Value Analysis

| Scenario | Intrinsic Value | Current Price | Margin of Safety |
|----------|----------------|---------------|-----------------|
| Bear     | ${{ intrinsic_value.bear_case | round(2) }} | ${{ margin_of_safety.current_price | round(2) }} | {{ (margin_of_safety.mos_vs_bear * 100) | round(1) }}% |
| Base     | ${{ intrinsic_value.base_case | round(2) }} | ${{ margin_of_safety.current_price | round(2) }} | {{ (margin_of_safety.mos_vs_base * 100) | round(1) }}% |
| Bull     | ${{ intrinsic_value.bull_case | round(2) }} | ${{ margin_of_safety.current_price | round(2) }} | {{ (margin_of_safety.mos_vs_bull * 100) | round(1) }}% |

**Buy Below (50% MoS):** ${{ margin_of_safety.buy_below_conservative | round(2) }}
**Buy Below (33% MoS):** ${{ margin_of_safety.buy_below_moderate | round(2) }}

**DCF Assumptions (base case):**
- Owner Earnings (start): ${{ (intrinsic_value.owner_earnings_used / 1000) | round(1) }}M
- High-growth rate (years 1–{{ intrinsic_value.projection_years // 2 }}): {{ (intrinsic_value.high_growth_rate_used * 100) | round(1) }}%
- Discount rate: {{ (intrinsic_value.discount_rate_used * 100) | round(1) }}%
- Terminal growth rate: {{ (intrinsic_value.terminal_growth_rate_used * 100) | round(1) }}%

---

## Earnings Yield vs. Bonds

| Metric | Value |
|--------|-------|
| Earnings Yield | {{ (earnings_yield.earnings_yield * 100) | round(2) }}% |
| Owner Earnings Yield | {{ (earnings_yield.owner_earnings_yield * 100) | round(2) }}% |
| 10Y Treasury Yield | {{ (earnings_yield.risk_free_rate * 100) | round(2) }}% |
| Spread (Earnings Yield − RFR) | {{ (earnings_yield.earnings_yield_spread * 100) | round(2) }}pp |
| Graham Number | ${{ earnings_yield.graham_number | round(2) }} |
| Price / Graham Number | {{ earnings_yield.price_vs_graham | round(2) }}× |

**Verdict:** {{ "✓ Attractive vs. bonds" if earnings_yield.is_attractive_vs_bonds else "✗ Not attractive vs. bonds" }}

---

## Key Financial Metrics

### Quality

| Metric | Latest | 5Y Avg |
|--------|--------|--------|
| ROIC | {{ (metrics.roic_latest * 100) | round(1) }}% | {{ (metrics.roic_avg_5yr * 100) | round(1) }}% |
| ROE | {{ (metrics.roe_latest * 100) | round(1) }}% | {{ (metrics.roe_avg_5yr * 100) | round(1) }}% |
| Gross Margin | {{ (metrics.gross_margin_latest * 100) | round(1) }}% | {{ (metrics.gross_margin_avg_5yr * 100) | round(1) }}% |
| Operating Margin | {{ (metrics.operating_margin_latest * 100) | round(1) }}% | {{ (metrics.operating_margin_avg_5yr * 100) | round(1) }}% |
| Net Margin | {{ (metrics.net_margin_latest * 100) | round(1) }}% | {{ (metrics.net_margin_avg_5yr * 100) | round(1) }}% |
| FCF Conversion | {{ (metrics.fcf_conversion_latest * 100) | round(1) }}% | {{ (metrics.fcf_conversion_avg_5yr * 100) | round(1) }}% |

### Growth

| Metric | 5Y CAGR | 10Y CAGR |
|--------|---------|----------|
| Revenue | {{ (metrics.revenue_cagr_5yr * 100) | round(1) }}% | {{ (metrics.revenue_cagr_10yr * 100) | round(1) }}% |
| EPS | {{ (metrics.eps_cagr_5yr * 100) | round(1) }}% | {{ (metrics.eps_cagr_10yr * 100) | round(1) }}% |
| Book Value / Share | {{ (metrics.bvps_cagr_5yr * 100) | round(1) }}% | {{ (metrics.bvps_cagr_10yr * 100) | round(1) }}% |

### Leverage & Safety

| Metric | Value |
|--------|-------|
| Debt / Equity | {{ metrics.debt_to_equity_latest | round(2) }}× |
| Net Debt / EBITDA | {{ metrics.net_debt_to_ebitda_latest | round(2) }}× |
| Interest Coverage | {{ metrics.interest_coverage_latest | round(1) }}× |
| Current Ratio | {{ metrics.current_ratio_latest | round(2) }} |
| Net Cash Position | {{ "Yes" if metrics.has_net_cash else "No" }} |

### Valuation

| Multiple | Value |
|----------|-------|
| P/E (TTM) | {{ metrics.pe_ttm | round(1) }}× |
| P/E (Normalised) | {{ metrics.pe_normalised | round(1) }}× |
| P/B | {{ metrics.pb_ratio | round(2) }}× |
| EV / EBITDA | {{ metrics.ev_to_ebitda | round(1) }}× |
| EV / FCF | {{ metrics.ev_to_fcf | round(1) }}× |

---

## Key Strengths

{% for strength in recommendation.key_strengths %}
- {{ strength }}
{% endfor %}

## Key Risks & Watchpoints

{% for risk in recommendation.key_risks %}
- {{ risk }}
{% endfor %}

---

## Qualitative Analysis Checklist

*The following questions require human judgment. Mark each as ✓ confirmed, ✗ concern, or ? needs research.*

### Business Understanding
{% for prompt in qualitative_prompts.business_understanding %}
- [ ] {{ prompt }}
{% endfor %}

### Competitive Moat
{% for prompt in qualitative_prompts.competitive_moat %}
- [ ] {{ prompt }}
{% endfor %}

### Management Quality
{% for prompt in qualitative_prompts.management_quality %}
- [ ] {{ prompt }}
{% endfor %}

### Long-Term Economics
{% for prompt in qualitative_prompts.long_term_economics %}
- [ ] {{ prompt }}
{% endfor %}

### Price Rationality
{% for prompt in qualitative_prompts.price_rationality %}
- [ ] {{ prompt }}
{% endfor %}

---

*Generated by Buffett Screener on {{ report_date }}. Not financial advice.*

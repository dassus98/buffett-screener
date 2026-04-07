# Report Spec

> Template field reference for `valuation_reports/templates/`.

---

## `deep_dive_template.md` — context variables

| Variable | Type | Source |
|----------|------|--------|
| `ticker` | str | `bundle.profile.ticker` |
| `report_date` | str | today formatted per config |
| `bundle` | TickerDataBundle | data_acquisition |
| `metrics` | dict | metrics_engine output |
| `intrinsic_value` | IntrinsicValueEstimate | `intrinsic_value.py` |
| `margin_of_safety` | MarginOfSafetyResult | `margin_of_safety.py` |
| `earnings_yield` | EarningsYieldComparison | `earnings_yield.py` |
| `qualitative_prompts` | dict[str, list[str]] | `qualitative_prompts.py` |
| `recommendation` | RecommendationResult | `recommendation.py` |

### `metrics` keys expected by template

Margins and return rates are displayed as percentages (multiply by 100 in template):
`roic_latest`, `roic_avg_5yr`, `roe_latest`, `roe_avg_5yr`,
`gross_margin_latest`, `gross_margin_avg_5yr`, `operating_margin_latest`, `operating_margin_avg_5yr`,
`net_margin_latest`, `net_margin_avg_5yr`, `fcf_conversion_latest`, `fcf_conversion_avg_5yr`,
`revenue_cagr_5yr`, `revenue_cagr_10yr`, `eps_cagr_5yr`, `eps_cagr_10yr`,
`bvps_cagr_5yr`, `bvps_cagr_10yr`,
`debt_to_equity_latest`, `net_debt_to_ebitda_latest`, `interest_coverage_latest`,
`current_ratio_latest`, `has_net_cash`,
`pe_ttm`, `pe_normalised`, `pb_ratio`, `ev_to_ebitda`, `ev_to_fcf`,
`owner_earnings_yield`

---

## `summary_table_template.md` — context variables

| Variable | Type |
|----------|------|
| `top_n` | int |
| `run_date` | str |
| `universe_size` | int |
| `passed_hard_filters` | int |
| `macro` | MacroSnapshot |
| `rows` | list[dict] — one per ranked ticker |
| `weights` | dict — from config composite_weights |
| `hard_filters` | dict — from config hard_filters |
| `filter_stats` | dict[str, int] — tickers eliminated per filter |

### `rows` dict keys

`rank`, `ticker`, `name`, `sector`, `composite_score`, `quality_score`,
`value_score`, `recommendation`, `roic_avg_5yr`, `gross_margin_avg_5yr`,
`pe_ttm`, `ev_to_ebitda`, `mos_vs_base`

---

## Recommendation tiers

| Tier | Condition |
|------|-----------|
| `STRONG_BUY` | quality_score ≥ 75 AND price ≤ buy_below_conservative |
| `BUY` | quality_score ≥ 60 AND price ≤ buy_below_moderate |
| `WATCHLIST` | quality_score ≥ 60 OR price ≤ buy_below_aggressive |
| `PASS` | does not meet quality or valuation bar |

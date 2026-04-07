# Scoring

> How the composite score is computed in `screener/` and `metrics_engine/composite_score.py`.

---

## Pipeline

```
1. Exclusions          remove disqualified sectors / flags
2. Hard filters        binary pass/fail — any failure eliminates the ticker
3. Soft filters        continuous 0–1 scores per category
4. Composite ranking   percentile-normalised weighted sum → 0–100 score
```

---

## Hard filter thresholds

All thresholds are read from `config/filter_config.yaml → hard_filters`.
See `docs/FORMULAS.md` for the formula behind each metric.

A ticker is eliminated if ANY hard filter fails.

---

## Soft filter scores (0–1)

Each category uses linear interpolation between a minimum (score=0) and a
target (score=1), then clips to [0, 1].

| Category | Formula |
|----------|---------|
| Margin consistency | `max(0, 1 − gross_margin_std / max_std)` averaged with operating margin equivalent |
| Revenue growth | `clip((cagr − min_cagr) / (target_cagr − min_cagr), 0, 1)` |
| CapEx intensity | `clip((max_ratio − actual) / (max_ratio − target_ratio), 0, 1)` |
| Owner earnings yield | `clip((yield − min_yield) / (target_yield − min_yield), 0, 1)` |

---

## Composite score weights

Read from `config/filter_config.yaml → composite_weights`. Must sum to 1.0.

| Component | Default weight | Direction |
|-----------|---------------|-----------|
| ROIC (5yr avg) | 0.25 | higher = better |
| ROE (5yr avg) | 0.10 | higher = better |
| Owner Earnings Yield | 0.20 | higher = better |
| Gross Margin Consistency | 0.10 | higher = better |
| Revenue CAGR (5yr) | 0.10 | higher = better |
| FCF Margin | 0.10 | higher = better |
| Debt Safety | 0.10 | lower D/E = better |
| Earnings Yield vs Bond | 0.05 | higher spread = better |

---

## Percentile normalisation

Each metric is converted to its percentile rank across the **passing universe**
before weighting. This makes the score robust to outliers and scale-invariant.

```python
score_col = df[metric].rank(pct=True)          # 0.0 – 1.0
# For inverse metrics (lower is better):
score_col = 1 − df[metric].rank(pct=True)
```

**Inverse metrics** (lower value = higher score):
`debt_to_equity`, `net_debt_to_ebitda`, `capex_to_revenue`,
`pe_ttm`, `pe_normalised`, `pb_ratio`, `ev_to_ebitda`, `ev_to_ebit`, `ev_to_fcf`

---

## Sub-scores

| Sub-score | Metrics included |
|-----------|-----------------|
| `quality_score` | ROIC, ROE, gross_margin, operating_margin, revenue_cagr, fcf_conversion |
| `value_score` | earnings_yield, owner_earnings_yield, fcf_yield, pe_ttm (inv), ev_to_ebitda (inv), pb_ratio (inv) |

Final `composite_score = weighted_sum(quality_score, value_score, debt_safety, ...)`.

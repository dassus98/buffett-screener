# SCORING.md — Composite Scoring System Specification

This document specifies the two-tier scoring system implemented in `screener/` and
`metrics_engine/composite_score.py`. Load this file when working on any screener or
ranking logic.

**Important:** This document specifies DEFAULT values. All thresholds and weights must be
read from `config/filter_config.yaml` at runtime via `filter_config_loader`. The code must
not reference numbers from this document directly.

---

## Scoring Pipeline

```
Universe tickers
      │
      ▼
1. Exclusions          Remove Financials, Utilities, Real Estate, SPACs, shell companies
      │ remaining tickers
      ▼
2. Tier 1 — Hard Filters    Binary pass/fail — any single failure = permanent exclusion
      │ passing tickers
      ▼
3. Tier 2 — Soft Filters    Scored 0–100 per criterion across passing universe
      │ per-criterion scores
      ▼
4. Composite Ranking        Weighted sum → final composite score (0–100) → rank
      │
      ▼
Shortlist (top N)
```

---

## Tier 1 — Hard Filters (Binary Pass/Fail)

A single failure on any hard filter permanently excludes the ticker from ranking and
from the shortlist. Hard filters are intentionally strict — they encode Buffett's
minimum quality requirements, not aspirational targets.

All thresholds read from `config/filter_config.yaml → hard_filters`.

| Filter | Criterion | Source Formula | Config Key |
|---|---|---|---|
| Earnings Consistency | Net income positive in ≥ 8 of last 10 years | F10 | `hard_filters.min_profitable_years` |
| ROE Floor | 10-year average ROE ≥ 15% | F3 | `hard_filters.min_roe_avg` |
| EPS Growth | 10-year EPS CAGR > 0% | F11 | `hard_filters.min_eps_cagr` |
| Debt Sustainability | Long-term debt payable in ≤ 5 years of owner earnings | F5 | `hard_filters.max_debt_payoff_years` |
| Data Sufficiency | ≥ 8 years of complete financial data | Data quality check | `hard_filters.min_data_years` |

### Hard Filter Behavior Rules

- **NaN on a required field:** Ticker fails that hard filter. A missing metric cannot be
  assumed to pass. Log at WARNING: `"[TICKER] failed [FILTER]: metric is NaN"`.
- **Negative equity:** F3 (ROE) = NaN → ticker fails the ROE Floor hard filter unless
  the config key `hard_filters.allow_negative_equity_pass` is set to true (default: false).
- **Zero earnings years:** Counted as unprofitable for the Earnings Consistency filter.
- **Partial history (8–9 years):** Data Sufficiency requires ≥ 8 complete years. Tickers with
  exactly 8 years pass this filter but may have NaN on metrics requiring 10 years (e.g., F11
  EPS CAGR falls back to available window — see FORMULAS.md).

---

## Tier 2 — Soft Filters (Scored 0–100 per Criterion)

All tickers that pass Tier 1 are scored on each criterion below. Scores are computed on
an absolute scale (0–100) using the breakpoints defined here, then summed with weights
to produce the final composite score.

Scores are NOT percentile-normalised across the passing universe — they are absolute
transformations of the metric value. This means the composite score is comparable
across different run dates and universe sizes.

All breakpoints read from `config/filter_config.yaml → soft_filters`.

### Scoring Functions

**Linear interpolation between breakpoints:**
```
score = clip((value - low_breakpoint) / (high_breakpoint - low_breakpoint) × 100, 0, 100)
```

**NaN handling:** If the underlying metric is NaN, the criterion score = 0 (not NaN).
The criterion's weight still applies — a security with missing data is penalised, not
excluded. Log at WARNING: `"[TICKER] criterion [NAME] scored 0: metric is NaN"`.

---

### Criterion Specifications

#### 1. ROE Consistency & Level (Weight: 15%)
*Config key: `soft_filters.roe`*

Base score from 10-year average ROE:
- 15% → 50 points
- 25%+ → 100 points
- Below 15% → linear from 0 (0% ROE) to 50 (15% ROE)
- Linear interpolation between all breakpoints

Variance penalty: If standard deviation of annual ROE across 10 years > 5%, subtract 15 points.
Floor at 0 after penalty.

```
base_score = clip(linear_interp(avg_roe, [(0%, 0), (15%, 50), (25%, 100)]), 0, 100)
penalty    = 15 if roe_stdev > 5% else 0
final      = max(0, base_score - penalty)
```

*Config keys: `soft_filters.roe.breakpoints`, `soft_filters.roe.variance_penalty_threshold`,
`soft_filters.roe.variance_penalty_points`*

#### 2. Gross Margin Level (Weight: 10%)
*Config key: `soft_filters.gross_margin`*

Scored on 10-year median gross margin:
- < 20% → 0
- 40% → 70
- > 60% → 100
- Linear interpolation between breakpoints

```
score = clip(linear_interp(median_gross_margin, [(<20%, 0), (40%, 70), (>60%, 100)]), 0, 100)
```

*Config keys: `soft_filters.gross_margin.breakpoints`*

#### 3. SG&A Discipline (Weight: 8%)
*Config key: `soft_filters.sga`*

Scored on 10-year median SG&A as a percentage of gross profit:
- < 30% of gross profit → 100
- Linear decay from 30% → 0 at 80%
- ≥ 80% → 0

```
score = clip(linear_interp(median_sga_ratio, [(30%, 100), (80%, 0)]), 0, 100)
```

Note: SG&A ratio = SG&A / Gross Profit (per F8 definition in FORMULAS.md).
If gross profit = 0 or NaN, score = 0.

*Config keys: `soft_filters.sga.breakpoints`*

#### 4. EPS Growth Rate & Consistency (Weight: 15%)
*Config key: `soft_filters.eps_growth`*

Two-component score:

**CAGR component** (10-year EPS CAGR, per F11):
- 10% CAGR → 50
- 20%+ → 100
- Below 0% → 0
- Linear interpolation

**Consistency multiplier** (count of years with EPS decline over 10-year window):
- 0 decline years → 1.0×
- 1 decline year → 0.9×
- 2 decline years → 0.8×
- 3+ decline years → 0.6×

```
cagr_score   = clip(linear_interp(eps_cagr, [(0%, 0), (10%, 50), (20%, 100)]), 0, 100)
multiplier   = {0: 1.0, 1: 0.9, 2: 0.8}.get(decline_years, 0.6)
final        = cagr_score × multiplier
```

*Config keys: `soft_filters.eps_growth.cagr_breakpoints`,
`soft_filters.eps_growth.consistency_multipliers`*

#### 5. Debt Conservatism (Weight: 10%)
*Config key: `soft_filters.debt`*

Scored on 10-year average debt-to-equity ratio (per F6):
- D/E < 0.2 → 100
- D/E = 0.8 → 0
- Linear interpolation

```
score = clip(linear_interp(avg_de_ratio, [(0.2, 100), (0.8, 0)]), 0, 100)
```

Negative equity → score = 0 (treat as maximum debt risk).

*Config keys: `soft_filters.debt.breakpoints`*

#### 6. Owner Earnings Growth (Weight: 12%)
*Config key: `soft_filters.owner_earnings_growth`*

Scored on 10-year owner earnings CAGR (per F1), using the same breakpoints as EPS growth:
- 10% CAGR → 50
- 20%+ → 100
- Below 0% → 0

```
score = clip(linear_interp(oe_cagr, [(0%, 0), (10%, 50), (20%, 100)]), 0, 100)
```

If fewer than 5 years of owner earnings are computable (e.g., D&A missing in early years),
score = 0 and log WARNING.

*Config keys: `soft_filters.owner_earnings_growth.breakpoints`*

#### 7. Capital Efficiency (Weight: 8%)
*Config key: `soft_filters.capex`*

Scored on 10-year median CapEx as a percentage of net income (per F12):
- CapEx/NI < 25% → 100
- Linear decay to CapEx/NI = 75% → 0
- ≥ 75% → 0

```
score = clip(linear_interp(median_capex_ratio, [(25%, 100), (75%, 0)]), 0, 100)
```

If net income ≤ 0 (owner earnings ≤ 0 is a Tier 1 hard filter violation, but if NI is
marginally positive and CapEx is very high), score = 0.

*Config keys: `soft_filters.capex.breakpoints`*

#### 8. Buyback Activity (Weight: 5%)
*Config key: `soft_filters.buybacks`*

Scored on net share count change over 10 years (per F13):
- Shares reduced by > 10% → 100
- Shares reduced by 5–10% → 70
- Shares reduced by 0–5% → 40
- Share count increased by 0–5% → 20
- Share count increased by > 5% (dilution) → 0

```
net_change = (shares_10yr_ago - shares_current) / shares_10yr_ago
score = {
    net_change > 10%:   100,
    5% < net_change ≤ 10%:  70,
    0% < net_change ≤ 5%:   40,
    -5% ≤ net_change ≤ 0%:  20,
    net_change < -5%:   0
}
```

*Config keys: `soft_filters.buybacks.thresholds`*

#### 9. Return on Retained Earnings (Weight: 10%)
*Config key: `soft_filters.retained_earnings_return`*

Scored on F4 (return on retained earnings — Buffett's "$1 test"):
- ≥ 15% → 100
- 12% → 70
- Linear from 0% → 0
- < 0% → 0

```
score = clip(linear_interp(roe_retained, [(0%, 0), (12%, 70), (15%, 100)]), 0, 100)
```

If cumulative retained earnings ≤ 0 (F4 is not computable), score = 0. Do not drop.

*Config keys: `soft_filters.retained_earnings_return.breakpoints`*

#### 10. Interest Expense Discipline (Weight: 7%)
*Config key: `soft_filters.interest_coverage`*

Scored on interest expense as a percentage of EBIT (inverse of coverage ratio):
- Interest < 10% of EBIT → 100
- 10–15% → 70
- Linear decay from 15% → 0 at 30%
- ≥ 30% → 0

```
interest_pct = interest_expense / ebit
score = {
    interest_pct < 10%:          100,
    10% ≤ interest_pct < 15%:    70,
    15% ≤ interest_pct < 30%:    linear from 70 to 0,
    interest_pct ≥ 30%:          0
}
```

If EBIT ≤ 0, score = 0 (negative operating income is disqualifying). If interest = 0 and
debt = 0, score = 100 (automatic pass, debt-free company).

*Config keys: `soft_filters.interest_coverage.breakpoints`*

---

## Weight Summary

| Criterion | Weight | Config Key |
|---|---|---|
| ROE consistency & level | 15% | `composite_weights.roe` |
| EPS growth rate & consistency | 15% | `composite_weights.eps_growth` |
| Owner earnings growth | 12% | `composite_weights.owner_earnings_growth` |
| Gross margin level | 10% | `composite_weights.gross_margin` |
| Debt conservatism | 10% | `composite_weights.debt` |
| Return on retained earnings | 10% | `composite_weights.retained_earnings_return` |
| SG&A discipline | 8% | `composite_weights.sga` |
| Capital efficiency | 8% | `composite_weights.capex` |
| Interest expense discipline | 7% | `composite_weights.interest_coverage` |
| Buyback activity | 5% | `composite_weights.buybacks` |
| **Total** | **100%** | |

Weights must sum to 1.0. `filter_config_loader` raises `ValueError` if they do not.

---

## Weights Justification

**ROE consistency & level (15%)** — ROE is the single most cited quantitative metric across
Buffett's shareholder letters and all major secondary sources (*Buffettology*, *The Warren Buffett
Way*, *Essays of Warren Buffett*). Buffett's 15% ROE threshold appears repeatedly as the minimum
mark of a durable business. Consistency is weighted alongside level because a business that earns
25% ROE in two years and 5% in the others is not a moat business — it is a cyclical. This
criterion receives the joint-highest weight alongside EPS growth.

**EPS growth rate & consistency (15%)** — EPS CAGR over a decade is Buffett's preferred measure
of compounding quality (*Buffettology*, Ch. 8). He emphasizes that earnings must grow *and* that
the trajectory must be consistent — a single high-growth year masking five stagnant ones fails
the test. The consistency multiplier operationalises this. Joint-highest weight with ROE because
together they define whether a business is genuinely compounding.

**Owner earnings growth (12%)** — Owner earnings (F1) is Buffett's own preferred metric over
reported net income, introduced in the 1986 shareholder letter. Growth in owner earnings — not
just their level — reflects the true rate at which the business is building owner wealth. Slightly
below ROE/EPS because it is harder to compute reliably (D&A proxies introduce noise) and
because its level is already partially captured by the initial rate of return (F2).

**Gross margin level (10%)** — Consistently high gross margins are Buffett's clearest proxy for
pricing power and moat (*Warren Buffett and the Interpretation of Financial Statements*, Ch. 4).
A company that can charge more than its cost of goods, year after year, has something competitors
cannot easily replicate. 10% weight because gross margin is a strong signal but is industry-
relative — some moat businesses (e.g., distribution) run on thin gross margins with wide operating
margins.

**Debt conservatism (10%)** — Buffett has consistently avoided financially leveraged businesses
(*Essays*, Part III). Excessive debt destroys optionality and amplifies cyclical damage. D/E is a
simple, robust proxy for balance sheet health and earns 10% weight because it interacts with the
Debt Sustainability hard filter — businesses that passed the hard filter still have meaningfully
different risk profiles along the D/E spectrum.

**Return on retained earnings (10%)** — This is Buffett's "$1 test" from the 1984 letter: every
dollar of retained earnings should generate at least $1 of market value over time, implying a
return on retained capital ≥ the cost of equity. A business that retains earnings but earns only
a savings-account return on them is destroying owner value. 10% weight because it measures the
compounding engine directly — the ability to reinvest at high rates is what separates a 15-year
holding from a trading position.

**SG&A discipline (8%)** — Buffett is hostile to management that allows selling, general, and
administrative costs to creep up relative to gross profit (*Warren Buffett and the Interpretation
of Financial Statements*, Ch. 9). SG&A bloat is a leading indicator of moat erosion — it often
signals that a company must spend heavily to maintain revenue. 8% weight because it is a
supporting signal rather than a primary one; a company with extraordinary ROE and EPS growth
may legitimately run high SG&A (e.g., fast-compounders in early scaling phase).

**Capital efficiency (8%)** — Low maintenance CapEx relative to earnings is a hallmark of
Buffett's preferred businesses — he explicitly favors companies that do not need to constantly
reinvest capital to stay competitive (*Essays*, Part IV). 8% weight because CapEx intensity is
already partially captured by owner earnings (F1 deducts maintenance CapEx) and by gross margin.
It serves as a cross-check on whether the business is truly asset-light.

**Interest expense discipline (7%)** — The interest coverage hard filter already eliminates the
most dangerous cases. This soft criterion scores the gradient within the passing universe. Buffett
views interest expense as a fixed cost that compounds the difficulty of bad years (*Intelligent
Investor*, Ch. 11). 7% weight — lower than debt conservatism because EBIT-based coverage is
more volatile year-to-year and the D/E criterion captures the structural leverage picture better.

**Buyback activity (5%)** — Share repurchases at prices below intrinsic value are, in Buffett's
words, "the highest and best use of cash" when the business is sound and price is right (1984
letter, 2011 letter). However, buybacks can also mask dilution from options grants, making the
signal noisy. 5% weight — supporting signal, not a primary quality indicator. A great business
that reinvests all earnings in high-ROIC internal projects may legitimately have no buybacks.

---

## Sensitivity Note

The weights above are interpretive priors derived from the relative emphasis Buffett places on
each metric across his letters and the secondary literature cited in this project's README.md.
They are not derived from a regression or backtested optimization.

**These weights should be treated as configurable starting points.** Users with different
investment philosophies — for example, those who prioritize capital-light businesses above all
else, or those who apply the framework to a single sector — may adjust weights in
`config/filter_config.yaml` without touching any code.

Reasonable alternative configurations:
- **Capital-light focus:** Raise `capex` to 15%, reduce `debt` to 5%
- **Value-only focus:** Raise `retained_earnings_return` to 18%, reduce `buybacks` to 2%
- **Growth-oriented:** Raise `eps_growth` to 20%, reduce `interest_coverage` to 4%

Any adjustment must ensure weights still sum to 1.0; `filter_config_loader` enforces this.

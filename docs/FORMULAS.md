# Formulas

> Every financial formula used in `metrics_engine/`. All inputs come from schema types.
> Monetary values are in USD thousands unless noted. Rates are decimals (0.04 = 4%).

---

## Owner Earnings  (`owner_earnings.py`)

```
Owner Earnings = Net Income
               + Depreciation & Amortisation
               − Maintenance CapEx (estimated)
               ± Change in Working Capital

Working Capital Change = ΔNWC = NWC_t − NWC_{t−1}
  where NWC = (CurrentAssets − Cash) − (CurrentLiabilities − ShortTermDebt)

Owner Earnings Yield = Owner Earnings / Market Cap
  (market_cap in full USD dollars; owner_earnings in USD thousands → convert)
```

**Maintenance CapEx methods (configurable):**
- `conservative`: maintenance_capex = abs(capital_expenditures)
- `graham`: maintenance_capex = depreciation_amortization
- `regression`: maintenance_capex = total_capex × (1 − revenue_growth / target_roic)

---

## Returns  (`returns.py`)

```
Effective Tax Rate = income_tax / pretax_income   [clipped to 0–0.50]

NOPAT = operating_income × (1 − effective_tax_rate)

Invested Capital = shareholders_equity + total_debt − cash_and_equivalents

ROIC = NOPAT / Invested Capital

ROE  = net_income / avg(shareholders_equity_t, shareholders_equity_{t−1})

Capital Employed = total_assets − total_current_liabilities

ROCE = operating_income / Capital Employed
```

---

## Profitability  (`profitability.py`)

```
Gross Margin     = gross_profit / revenue
Operating Margin = operating_income / revenue
Net Margin       = net_income / revenue
EBITDA Margin    = ebitda / revenue

Margin Trend: fit linear regression over last 5 years of margin values.
  slope > +0.02/yr → "expanding"
  slope < −0.02/yr → "contracting"
  else             → "stable"
```

---

## Leverage  (`leverage.py`)

```
Debt / Equity         = total_debt / shareholders_equity
Net Debt              = total_debt − cash_and_equivalents
Net Debt / EBITDA     = net_debt / ebitda
Interest Coverage     = operating_income / interest_expense
Current Ratio         = total_current_assets / total_current_liabilities
Debt / Assets         = total_debt / total_assets
```

---

## Growth  (`growth.py`)

```
CAGR(start, end, years) = (end / start)^(1/years) − 1

YoY Growth_t = (value_t / value_{t−1}) − 1

BVPS = shareholders_equity / shares_outstanding

Growth Consistency Score:
  neg_years = count(yoy_growth < 0)
  score = 1 − (neg_years / total_years)
  penalty: −0.1 per year where yoy_growth < −0.10  [clipped at 0]
```

---

## CapEx  (`capex.py`)

```
CapEx / Revenue    = abs(capital_expenditures) / revenue
CapEx / OCF        = abs(capital_expenditures) / operating_cash_flow
CapEx / D&A        = abs(capital_expenditures) / depreciation_amortization
FCF Conversion     = free_cash_flow / ebitda   [clipped −1 to 2]

is_asset_light = (CapEx/Revenue 5yr avg) < 0.05
```

---

## Valuation  (`valuation.py`)

```
P/E          = price / eps_diluted
P/B          = price / book_value_per_share
EV/EBITDA    = enterprise_value_usd / (ebitda_thousands × 1000)
EV/EBIT      = enterprise_value_usd / (operating_income_thousands × 1000)
EV/FCF       = enterprise_value_usd / (free_cash_flow_thousands × 1000)

Earnings Yield       = eps_diluted / price
FCF Yield            = (free_cash_flow / shares_outstanding) / price
Owner Earnings Yield = (owner_earnings_thousands × 1000) / market_cap_usd

Earnings Yield Spread = earnings_yield − risk_free_rate

Graham Number = sqrt(22.5 × eps_diluted × book_value_per_share)
```

---

## DCF / Intrinsic Value  (`intrinsic_value.py`)

```
Two-stage owner-earnings DCF:

Stage 1 (years 1 to high_growth_years):
  PV_t = OE_base × (1 + g_high)^t / (1 + r)^t

Stage 2 (years high_growth_years+1 to projection_years):
  PV_t = OE_stage2 × (1 + g_fade)^t / (1 + r)^t

Terminal Value:
  TV  = OE_final × (1 + g_terminal) / (r − g_terminal)
  PV_TV = TV / (1 + r)^projection_years

Intrinsic Value (total, USD thousands) = Σ Stage1 PV + Σ Stage2 PV + PV_TV
Intrinsic Value per Share = total_USD_thousands × 1000 / (shares_outstanding × 1000)
                          = total_USD_thousands / shares_outstanding
```

---

## Margin of Safety  (`margin_of_safety.py`)

```
MoS = (intrinsic_value_per_share − current_price) / intrinsic_value_per_share

Buy Below (conservative, 50%) = intrinsic_value × 0.50
Buy Below (moderate,     33%) = intrinsic_value × 0.67
Buy Below (aggressive,   20%) = intrinsic_value × 0.80
```

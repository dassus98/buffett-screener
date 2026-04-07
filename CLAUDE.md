# CLAUDE.md — Buffett Screener Project Guide

This file is read automatically by Claude Code at the start of every session.
Follow every instruction here without being asked.

---

## Project purpose

A Warren Buffett-style fundamental stock screener that ranks equities by quality
(ROIC, margins, moat proxies) and value (owner-earnings DCF, margin of safety).
It is NOT a trading system — it surfaces candidates for deeper human review.

---

## Before touching any code

1. Read `config/filter_config.yaml` — every threshold lives there. Never hardcode one.
2. Read `data_acquisition/schema.py` — it defines the canonical types that flow between
   all modules. Never bypass these types with raw DataFrames across module boundaries.
3. Read the relevant `docs/` file for the module you are editing:
   - `docs/FORMULAS.md` before touching `metrics_engine/`
   - `docs/SCORING.md` before touching `screener/`
   - `docs/DATA_SOURCES.md` before touching `data_acquisition/`
   - `docs/REPORT_SPEC.md` before touching `valuation_reports/`

---

## Non-negotiable implementation rules

| Rule | Detail |
|------|--------|
| No silent imputation | Missing data → `float("nan")` + `logger.warning(...)`. Never substitute 0. |
| No hardcoded thresholds | All business thresholds come from `config/filter_config.yaml` via the `config` dict parameter. |
| Type hints on all public functions | Every `def` in a public module must have full type annotations. |
| Logging | `logger = logging.getLogger(__name__)` in every module. Log DEBUG for field mappings/cache hits, INFO for major ops, WARNING for data gaps or fallbacks. |
| Unit convention | Monetary values → USD thousands (schema standard). `MarketData.market_cap` / `.enterprise_value` → full USD dollars (exception, documented in schema.py). |
| Sign convention | `capital_expenditures` is always stored **negative**. If a source returns it positive, negate and log a warning. |
| Config-driven exchange list | Check `config["universe"]["exchanges"]` before fetching TSX or any non-US exchange. Never hardcode exchange membership. |

---

## Module dependency order

```
schema.py → api_config.py → store.py
         → universe.py → financials.py → market_data.py → macro_data.py
         → data_quality.py

metrics_engine/* depends on data_acquisition schemas only
screener/* depends on metrics_engine output DataFrames
valuation_reports/* depends on screener output + metrics_engine
output/* orchestrates everything
```

No circular imports. If you need something from a higher layer, pass it as a parameter.

---

## Testing requirements

- All pure formula functions must have unit tests in `tests/test_formulas.py`.
- All filter logic must have unit tests in `tests/test_filters.py`.
- Tests use synthetic in-memory data only — no network calls, no disk I/O.
- Run tests with: `pytest --tb=short`
- Run linting with: `ruff check . && mypy .`

---

## Data sources and costs

| Source | What it provides | Cost |
|--------|-----------------|------|
| yfinance | Price, market cap, basic financials | Free |
| FRED API | Risk-free rate, CPI, GDP growth | Free (key required) |
| SEC EDGAR | Full XBRL financial statements | Free (User-Agent required) |
| FMP | Universe listing, TSX coverage | Free tier: 250 req/day |

API keys are loaded from `.env` only. See `.env.example`. Never commit `.env`.

---

## What NOT to do

- Do not add features beyond what the task specifies.
- Do not add docstrings or comments to code you didn't change.
- Do not create abstractions for one-off operations.
- Do not use `pd.fillna(0)` anywhere in this codebase — use NaN and log it.
- Do not use `pass` as a stub — use `raise NotImplementedError` if truly incomplete.

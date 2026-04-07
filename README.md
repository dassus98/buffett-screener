# buffett-screener

A Warren Buffett-style fundamental stock screener and intrinsic value engine.

## Philosophy

Operationalises the core tenets from Buffett's letters to shareholders,
*The Intelligent Investor* (Graham), and *Security Analysis*:

- **Owner Earnings** over GAAP net income
- **ROIC > WACC** as the primary quality gate
- **Moat proxies** via consistent margins and low capital intensity
- **Margin of Safety** before every buy decision

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# edit .env

# Run the full pipeline
python -m output.pipeline_runner

# Launch dashboard
streamlit run output/streamlit_app.py
```

## Architecture

```
data_acquisition/   → Fetch, validate, and cache raw financial + market data
metrics_engine/     → Compute owner earnings, ROIC, margins, growth, multiples
screener/           → Hard filters → soft scoring → composite ranking
valuation_reports/  → DCF intrinsic value, margin of safety, Markdown reports
output/             → Streamlit dashboard, export, pipeline orchestration
```

## Configuration

All thresholds and weights live in `config/filter_config.yaml`.
No code changes needed to adjust screening criteria.

## Documentation

| Doc | Contents |
|-----|----------|
| `docs/FORMULAS.md` | Every financial formula used in `metrics_engine/` |
| `docs/ARCHITECTURE.md` | Module dependency map and data flow |
| `docs/SCORING.md` | Composite score weights and normalisation method |
| `docs/DATA_SOURCES.md` | API sources, field mappings, rate limits |
| `docs/REPORT_SPEC.md` | Report template field reference |

## Testing

```bash
pytest --tb=short
ruff check .
mypy .
```

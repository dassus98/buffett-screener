# Buffett Screener

A Warren Buffett-style fundamental stock screener and intrinsic value engine.

## Philosophy

This screener operationalises the core tenets from Buffett's letters to shareholders,
*The Intelligent Investor* (Graham), and *Security Analysis*:

- **Owner Earnings** over GAAP earnings
- **ROIC > WACC** as the primary quality gate
- **Moat proxies** via consistent margins and low capital intensity
- **Margin of Safety** before every buy decision

## Quickstart

```bash
# Install dependencies
poetry install

# Copy env template and add API keys
cp .env.example .env

# Run the full screening pipeline
python -m output.pipeline_runner

# Launch the Streamlit dashboard
streamlit run output/streamlit_app.py
```

## Architecture

```
data_acquisition/   → Fetch & validate raw financial + market data
metrics_engine/     → Compute owner earnings, ROIC, margins, growth, valuation multiples
screener/           → Apply hard filters, soft scores, composite ranking
valuation_reports/  → DCF, margin-of-safety, earnings yield, PDF/Markdown reports
output/             → Streamlit dashboard, Markdown export, pipeline orchestration
```

## Configuration

All thresholds and weights live in `config/filter_config.yaml`. No code changes are
needed to adjust screening criteria — edit the YAML and re-run.

## Data Sources

| Source      | Used for                              |
|-------------|---------------------------------------|
| yfinance    | Price, market cap, basic financials   |
| FRED (API)  | Risk-free rate, CPI, macro indicators |
| SEC EDGAR   | Full XBRL financial statements        |

API keys are loaded from `.env` (see `.env.example`).

## Testing

```bash
pytest --cov
```

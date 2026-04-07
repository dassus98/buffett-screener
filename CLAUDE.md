# CLAUDE.md — Persistent Project Instructions

These rules apply to EVERY task in this project. No exceptions.

## Before Starting Any Task
1. Read this file (CLAUDE.md)
2. Read README.md for project structure and module dependencies
3. Read the specific docs/ file(s) referenced in the task prompt
4. Read config/filter_config.yaml for current thresholds and parameters
5. Read data_acquisition/schema.py for canonical column names and line-item mapping

## Absolute Constraints
- **No hardcoded thresholds.** Every numeric threshold, weight, or parameter must be read from config/filter_config.yaml at runtime. If you find yourself typing a threshold number in any file other than filter_config.yaml, stop and refactor.
- **No silent data imputation.** If a required financial data field is missing or null, the computed metric must be NaN/None with a logged reason. Never fill missing data with zeros, averages, or estimates without explicit instruction.
- **No assumed line item names.** All financial statement field names must come from the canonical mapping in data_acquisition/schema.py. If an API returns a field name not in the mapping, add it to schema.py with a documented substitution confidence level before using it.
- **Every public function must have:** type hints on all parameters and return value, a docstring describing inputs/outputs/side effects, and explicit handling of missing/null data.
- **Every line-item substitution must be logged** with: the original (ideal) line item, the substitute used, the confidence level (High/Medium/Low), and the security ticker it applies to.
- **Run tests after every implementation:** `pytest tests/ -v --tb=short`

## Logging Standards
Use Python's logging module (not print statements). Log levels:
- INFO: Pipeline progress (e.g., "Fetched financials for 500 tickers")
- WARNING: Data substitutions, missing fields, dropped securities
- ERROR: API failures, schema mismatches, data integrity violations
- DEBUG: Raw API responses (only when troubleshooting)

## Code Style
- Type hints everywhere
- Functions should do one thing
- No function longer than 50 lines — extract helpers
- Use pathlib for file paths, not os.path
- Config loading: always via filter_config_loader, never direct yaml.safe_load in business logic

## Testing Requirements
- Every formula (F1–F16) must have at least one unit test with a hand-calculated expected value
- Integration tests must validate against known Buffett stocks (KO, AAPL, AXP, BRK.B)
- Edge case tests: negative equity, zero-earnings years, missing data periods, single-year history

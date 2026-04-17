.PHONY: help install test lint pipeline dashboard clean docker-build docker-run docker-dashboard

# Default Python — override with: make PYTHON=python3.12 install
PYTHON ?= python3
VENV   ?= .venv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

install: ## Create venv and install all dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -e .
	@echo "\n  Activate with:  source $(VENV)/bin/activate"

test: ## Run the full test suite (1,234 tests)
	$(VENV)/bin/pytest tests/ -v --tb=short

lint: ## Run ruff linter and mypy type checker
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy data_acquisition metrics_engine screener valuation_reports output

pipeline: ## Run the full pipeline (fetch + metrics + screen + reports)
	$(VENV)/bin/python -m output.pipeline_runner --mode reports --no-moat

pipeline-fast: ## Re-run metrics + screening + reports (skip data fetch)
	$(VENV)/bin/python -m output.pipeline_runner --mode reports --skip-acquisition --no-moat

dashboard: ## Launch the Streamlit dashboard
	$(VENV)/bin/streamlit run output/streamlit_app.py

clean: ## Remove caches, compiled files, and generated data
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/

clean-data: ## Remove all generated data (DuckDB, reports, caches)
	rm -rf data/raw/ data/processed/ data/reports/ data/pipeline.log

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build the Docker image
	docker build -t buffett-screener .

docker-run: ## Run the pipeline in Docker
	docker run --env-file .env -v ./data:/app/data buffett-screener

docker-dashboard: ## Launch Streamlit dashboard in Docker
	docker run --env-file .env -v ./data:/app/data -p 8501:8501 \
		buffett-screener streamlit run output/streamlit_app.py \
		--server.headless true --server.port 8501 --server.address 0.0.0.0

docker-test: ## Run tests in Docker
	docker run buffett-screener pytest tests/ -v --tb=short

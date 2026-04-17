# Multi-stage Dockerfile for buffett-screener
# Produces a lean image (~400MB) with all dependencies pre-installed.
#
# Build:  docker build -t buffett-screener .
# Run:    docker run --env-file .env buffett-screener
# Shell:  docker run --env-file .env -it buffett-screener bash

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by numpy/pandas/matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pyproject.toml .
COPY config/ config/
COPY data_acquisition/ data_acquisition/
COPY metrics_engine/ metrics_engine/
COPY screener/ screener/
COPY valuation_reports/ valuation_reports/
COPY output/ output/
COPY .streamlit/ .streamlit/
COPY tests/ tests/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p data/raw data/processed data/reports

# Default: run the full pipeline
CMD ["python", "-m", "output.pipeline_runner", "--mode", "reports", "--no-moat"]

# Expose Streamlit port for dashboard mode
EXPOSE 8501

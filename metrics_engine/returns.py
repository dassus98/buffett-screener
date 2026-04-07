"""
metrics_engine.returns
=======================
Computes return-on-capital metrics: ROIC, ROE, ROCE.

Buffett's primary quality filter is consistently high return on invested capital
(ROIC > 15% is excellent; > 10% is the hard-filter floor in this screener).
A business that earns high returns without leverage is the gold standard.

Formulas used:
    ROIC  = NOPAT / Invested Capital
    ROE   = Net Income / Average Shareholders' Equity
    ROCE  = EBIT / Capital Employed

Where:
    NOPAT          = Operating Income × (1 - effective_tax_rate)
    Invested Capital = Total Equity + Total Debt - Cash & Equivalents
    Capital Employed = Total Assets - Current Liabilities
"""

from __future__ import annotations

from data_acquisition.schema import TickerDataBundle


def compute_returns(bundle: TickerDataBundle) -> dict[str, float]:
    """
    Compute ROIC, ROE, and ROCE for the most recent period and trailing averages.

    Args:
        bundle: TickerDataBundle with financial statements and balance sheets.

    Returns:
        Dict with keys:
            "roic_latest"          — ROIC for the most recent fiscal year (decimal)
            "roic_avg_5yr"         — 5-year average ROIC (decimal)
            "roic_avg_10yr"        — 10-year average ROIC (decimal)
            "roe_latest"           — ROE for the most recent fiscal year (decimal)
            "roe_avg_5yr"          — 5-year average ROE (decimal)
            "roce_latest"          — ROCE for the most recent fiscal year (decimal)
            "roce_avg_5yr"         — 5-year average ROCE (decimal)
            "effective_tax_rate"   — Effective tax rate used for NOPAT (decimal)

    Logic:
        1. For each period, call compute_roic(), compute_roe(), compute_roce()
        2. Compute trailing averages over 5 and 10 years
        3. Return all metrics in a flat dict
    """
    ...


def compute_roic(
    operating_income: float,
    effective_tax_rate: float,
    total_equity: float,
    total_debt: float,
    cash: float,
) -> float:
    """
    Compute Return on Invested Capital for a single period.

    Args:
        operating_income:  EBIT from the income statement (USD thousands).
        effective_tax_rate: Effective income tax rate as a decimal (e.g. 0.21).
        total_equity:      Shareholders' equity from the balance sheet (USD thousands).
        total_debt:        Total interest-bearing debt (USD thousands).
        cash:              Cash and cash equivalents (USD thousands).

    Returns:
        ROIC as a decimal. Returns NaN if invested capital is zero or negative.

    Formula:
        NOPAT = operating_income × (1 - effective_tax_rate)
        Invested Capital = total_equity + total_debt - cash
        ROIC = NOPAT / Invested Capital
    """
    ...


def compute_roe(
    net_income: float,
    equity_start: float,
    equity_end: float,
) -> float:
    """
    Compute Return on Equity using average equity for the period.

    Args:
        net_income:   Net income for the period (USD thousands).
        equity_start: Shareholders' equity at the start of the period (prior year end).
        equity_end:   Shareholders' equity at the end of the period.

    Returns:
        ROE as a decimal. Returns NaN if average equity is zero or negative.

    Formula:
        ROE = net_income / ((equity_start + equity_end) / 2)
    """
    ...


def compute_roce(
    ebit: float,
    total_assets: float,
    current_liabilities: float,
) -> float:
    """
    Compute Return on Capital Employed.

    Args:
        ebit:                 Earnings before interest and tax (USD thousands).
        total_assets:         Total assets from the balance sheet (USD thousands).
        current_liabilities:  Total current liabilities (USD thousands).

    Returns:
        ROCE as a decimal. Returns NaN if capital employed is zero or negative.

    Formula:
        Capital Employed = total_assets - current_liabilities
        ROCE = ebit / capital_employed
    """
    ...


def effective_tax_rate(
    income_tax: float,
    pretax_income: float,
) -> float:
    """
    Compute the effective income tax rate from reported figures.

    Args:
        income_tax:    Income tax expense (USD thousands).
        pretax_income: Pre-tax income (USD thousands).

    Returns:
        Effective tax rate as a decimal. Clipped to [0, 0.50] to handle
        unusual periods (tax benefits, loss years). Returns 0.21 (US statutory
        rate) if pretax_income is zero or negative.

    Formula:
        rate = income_tax / pretax_income
    """
    ...


def roic_consistency_score(roic_series: list[float]) -> float:
    """
    Score the consistency of ROIC over time on a 0–1 scale.

    Args:
        roic_series: List of annual ROIC values (decimal), sorted ascending by year.
                     Should contain at least 5 values for a meaningful score.

    Returns:
        Score from 0.0 (highly inconsistent) to 1.0 (perfectly consistent).

    Logic:
        1. Compute the coefficient of variation: std(roic) / mean(roic)
        2. Map CV to [0, 1]: score = max(0, 1 - CV)
           (CV of 0 → score=1.0; CV of 1 → score=0.0; CV>1 → score=0.0)
        3. Additionally, penalise any year where ROIC < 0 (below-hurdle year)
           by subtracting 0.1 per such year, clipped at 0
    """
    ...

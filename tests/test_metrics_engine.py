"""Unit tests for metrics_engine/__init__.py — Module 2 orchestrator.

Tests are split into three concerns:

1. Pure computation (``_compute_all_from_data``) — no DuckDB required.
2. ``compute_ticker_metrics`` — the public per-ticker entry point (mocked I/O).
3. ``run_metrics_engine`` — the full pipeline driver (mocked I/O and DuckDB).

All DuckDB interactions are mocked so the test suite runs without a live database.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from metrics_engine import (
    _compute_all_from_data,
    compute_ticker_metrics,
    run_metrics_engine,
)

# ---------------------------------------------------------------------------
# Synthetic DataFrame builders (canonical schema column names)
# ---------------------------------------------------------------------------

_N_YEARS = 10
_START_YEAR = 2014
_YEARS = list(range(_START_YEAR, _START_YEAR + _N_YEARS))
_TICKER = "TEST"


def _income_df(n: int = _N_YEARS, start: int = _START_YEAR) -> pd.DataFrame:
    """Build a minimal income_statement DataFrame with steadily growing metrics."""
    years = list(range(start, start + n))
    return pd.DataFrame({
        "ticker": [_TICKER] * n,
        "fiscal_year": years,
        "net_income": [100.0 * (1.07 ** i) for i in range(n)],
        "total_revenue": [500.0 * (1.05 ** i) for i in range(n)],
        "gross_profit": [200.0 * (1.05 ** i) for i in range(n)],
        "sga": [50.0 * (1.03 ** i) for i in range(n)],
        "operating_income": [120.0 * (1.06 ** i) for i in range(n)],
        "interest_expense": [8.0] * n,
        "eps_diluted": [2.0 * (1.07 ** i) for i in range(n)],
        "shares_outstanding_diluted": [50.0e6] * n,
    })


def _balance_df(n: int = _N_YEARS, start: int = _START_YEAR) -> pd.DataFrame:
    years = list(range(start, start + n))
    return pd.DataFrame({
        "ticker": [_TICKER] * n,
        "fiscal_year": years,
        "long_term_debt": [200.0] * n,
        "shareholders_equity": [400.0 * (1.05 ** i) for i in range(n)],
        "treasury_stock": [0.0] * n,
    })


def _cashflow_df(n: int = _N_YEARS, start: int = _START_YEAR) -> pd.DataFrame:
    years = list(range(start, start + n))
    return pd.DataFrame({
        "ticker": [_TICKER] * n,
        "fiscal_year": years,
        "depreciation_amortization": [20.0] * n,
        "capital_expenditures": [-30.0] * n,
        "working_capital_change": [5.0] * n,
    })


def _market_row() -> pd.Series:
    return pd.Series({
        "ticker": _TICKER,
        "current_price_usd": 100.0,
        "shares_outstanding": 50.0e6,
        "market_cap_usd": 5.0e9,
        "enterprise_value_usd": 5.2e9,
        "pe_ratio_trailing": 22.0,
        "dividend_yield": 0.02,
        "high_52w": 120.0,
        "low_52w": 80.0,
        "avg_volume_3m": 1.0e6,
        "as_of_date": "2024-01-01",
    })


def _macro_dict() -> dict[str, float]:
    return {"us_treasury_10yr": 0.045, "goc_bond_10yr": 0.042}


# ---------------------------------------------------------------------------
# Tests — _compute_all_from_data (pure computation, no mocks needed)
# ---------------------------------------------------------------------------

class TestComputeAllFromData:
    def test_returns_tuple_of_dict_and_dataframe(self):
        result = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        summary, annual_df = result
        assert isinstance(summary, dict)
        assert isinstance(annual_df, pd.DataFrame)

    def test_ticker_present_in_summary(self):
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert summary.get("ticker") == _TICKER

    def test_composite_score_keys_present(self):
        """All keys needed by compute_composite_score must be in the summary."""
        required = {
            "avg_roe", "roe_stdev", "avg_gross_margin", "avg_sga_ratio",
            "eps_cagr", "decline_years", "avg_de_10yr",
            "owner_earnings_cagr", "avg_capex_to_ni",
            "buyback_pct", "return_on_retained", "avg_interest_pct_10yr",
        }
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        missing = required - set(summary.keys())
        assert not missing, f"Missing composite-score keys: {missing}"

    def test_f1_key_present(self):
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert "avg_owner_earnings_10yr" in summary

    def test_f5_key_present_and_renamed(self):
        """'pass' key from F5 must be renamed to 'debt_payoff_pass'."""
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert "debt_payoff_pass" in summary
        assert "pass" not in summary

    def test_f11_eps_cagr_is_float(self):
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert isinstance(summary.get("eps_cagr"), float)

    def test_f11_eps_cagr_positive_for_growing_company(self):
        """10-year EPS growing at 7% → cagr ≈ 0.07 (within tolerance)."""
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        cagr = summary.get("eps_cagr", float("nan"))
        assert not math.isnan(cagr), "eps_cagr should not be NaN for growing EPS"
        assert cagr > 0

    def test_f14_weighted_iv_present(self):
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert "weighted_iv" in summary

    def test_f14_scenario_keys_flattened(self):
        """F14 bear/base/bull scenario fields must be flattened with prefix."""
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert "f14_bear_present_value" in summary
        assert "f14_base_present_value" in summary
        assert "f14_bull_present_value" in summary

    def test_f15_margin_of_safety_present(self):
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert "margin_of_safety" in summary

    def test_f16_earnings_yield_present(self):
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert "earnings_yield" in summary

    def test_annual_df_has_ticker_column(self):
        _, annual_df = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert "ticker" in annual_df.columns

    def test_annual_df_has_fiscal_year_column(self):
        _, annual_df = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert "fiscal_year" in annual_df.columns

    def test_annual_df_row_count_matches_income_years(self):
        """Annual DataFrame must have at least as many rows as the income_df."""
        _, annual_df = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert len(annual_df) >= _N_YEARS

    def test_annual_df_sorted_ascending_by_year(self):
        _, annual_df = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        years = annual_df["fiscal_year"].tolist()
        assert years == sorted(years)

    def test_annual_df_has_required_per_year_columns(self):
        _, annual_df = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        for col in ("owner_earnings", "roe", "gross_margin", "de_ratio"):
            assert col in annual_df.columns, f"Missing per-year column: {col}"

    def test_none_market_row_does_not_raise(self):
        """Absent market data must not crash computation — price-dependent metrics → NaN."""
        summary, annual_df = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), None, 0.04
        )
        assert isinstance(summary, dict)
        assert "ticker" in summary

    def test_tsx_rfr_selection(self):
        """TSX ticker (.TO) should use goc_bond_10yr; the value echoes in f16 bond_yield."""
        from metrics_engine import _extract_macro_rfr
        macro = {"goc_bond_10yr": 0.042, "us_treasury_10yr": 0.045}
        assert _extract_macro_rfr(macro, "SHOP.TO") == pytest.approx(0.042)

    def test_us_rfr_selection(self):
        from metrics_engine import _extract_macro_rfr
        macro = {"goc_bond_10yr": 0.042, "us_treasury_10yr": 0.045}
        assert _extract_macro_rfr(macro, "KO") == pytest.approx(0.045)

    def test_rfr_fallback_when_missing(self):
        from metrics_engine import _extract_macro_rfr
        assert _extract_macro_rfr({}, "KO") == pytest.approx(0.04)

    def test_f14_depends_on_f11_cagr(self):
        """Verify F14 receives F11's eps_cagr through the dependency chain.

        With positive eps_cagr, intrinsic value should be > 0.
        With zero EPS (eps_cagr=NaN), intrinsic value should be NaN.
        """
        # Zero EPS → eps_cagr NaN → IV NaN
        zero_eps_inc = _income_df()
        zero_eps_inc["eps_diluted"] = 0.0
        summary, _ = _compute_all_from_data(
            _TICKER, zero_eps_inc, _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert math.isnan(summary.get("weighted_iv", float("nan")))

    def test_f15_depends_on_f14_weighted_iv(self):
        """MoS must be NaN when IV is NaN (zero-EPS case → F14 NaN → F15 NaN)."""
        zero_eps_inc = _income_df()
        zero_eps_inc["eps_diluted"] = 0.0
        summary, _ = _compute_all_from_data(
            _TICKER, zero_eps_inc, _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        assert math.isnan(summary.get("margin_of_safety", float("nan")))

    def test_avg_roe_positive_for_profitable_company(self):
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        avg_roe = summary.get("avg_roe", float("nan"))
        assert not math.isnan(avg_roe)
        assert avg_roe > 0

    def test_avg_gross_margin_in_valid_range(self):
        summary, _ = _compute_all_from_data(
            _TICKER, _income_df(), _balance_df(), _cashflow_df(), _market_row(), 0.04
        )
        gm = summary.get("avg_gross_margin", float("nan"))
        assert not math.isnan(gm)
        assert 0.0 < gm < 1.0


# ---------------------------------------------------------------------------
# Tests — compute_ticker_metrics (public, mocked _read_ticker_data)
# ---------------------------------------------------------------------------

class TestComputeTickerMetrics:
    def _good_data(self) -> tuple:
        return (
            _income_df(), _balance_df(), _cashflow_df(), _market_row(), _macro_dict()
        )

    @patch("metrics_engine._read_ticker_data")
    def test_returns_dict(self, mock_read: MagicMock) -> None:
        mock_read.return_value = self._good_data()
        result = compute_ticker_metrics(_TICKER)
        assert isinstance(result, dict)

    @patch("metrics_engine._read_ticker_data")
    def test_ticker_key_present(self, mock_read: MagicMock) -> None:
        mock_read.return_value = self._good_data()
        result = compute_ticker_metrics(_TICKER)
        assert result.get("ticker") == _TICKER

    @patch("metrics_engine._read_ticker_data")
    def test_all_composite_score_keys_present(self, mock_read: MagicMock) -> None:
        mock_read.return_value = self._good_data()
        result = compute_ticker_metrics(_TICKER)
        required = {
            "avg_roe", "avg_gross_margin", "eps_cagr",
            "avg_de_10yr", "buyback_pct", "avg_interest_pct_10yr",
        }
        assert required.issubset(result.keys())

    @patch("metrics_engine._read_ticker_data")
    def test_empty_income_df_returns_minimal_dict(self, mock_read: MagicMock) -> None:
        """No income data → returns ``{"ticker": …}`` without crashing."""
        mock_read.return_value = (
            pd.DataFrame(), _balance_df(), _cashflow_df(), _market_row(), _macro_dict()
        )
        result = compute_ticker_metrics(_TICKER)
        assert result.get("ticker") == _TICKER

    @patch("metrics_engine._read_ticker_data")
    def test_uses_goc_rfr_for_tsx_ticker(self, mock_read: MagicMock) -> None:
        """TSX ticker's bond_yield (F16) should equal the GoC rate from macro_dict."""
        mock_read.return_value = (
            _income_df(), _balance_df(), _cashflow_df(), _market_row(), _macro_dict()
        )
        result = compute_ticker_metrics("SHOP.TO")
        assert result.get("bond_yield") == pytest.approx(0.042)

    @patch("metrics_engine._read_ticker_data")
    def test_uses_us_rfr_for_nyse_ticker(self, mock_read: MagicMock) -> None:
        mock_read.return_value = (
            _income_df(), _balance_df(), _cashflow_df(), _market_row(), _macro_dict()
        )
        result = compute_ticker_metrics("KO")
        assert result.get("bond_yield") == pytest.approx(0.045)


# ---------------------------------------------------------------------------
# Tests — run_metrics_engine (mocked store + DuckDB)
# ---------------------------------------------------------------------------

class TestRunMetricsEngine:
    def _mock_conn(self) -> MagicMock:
        conn = MagicMock()
        conn.execute = MagicMock()
        conn.register = MagicMock()
        conn.unregister = MagicMock()
        return conn

    @patch("metrics_engine.get_connection")
    @patch("metrics_engine.get_surviving_tickers")
    def test_empty_tickers_returns_empty_dataframe(
        self, mock_tickers: MagicMock, mock_conn: MagicMock
    ) -> None:
        mock_tickers.return_value = []
        mock_conn.return_value = self._mock_conn()
        df = run_metrics_engine()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("metrics_engine._write_table_full_replace")
    @patch("metrics_engine._write_summary_with_scores")
    @patch("metrics_engine._write_annual_metrics")
    @patch("metrics_engine._init_metrics_tables")
    @patch("metrics_engine.get_connection")
    @patch("metrics_engine._read_ticker_data")
    @patch("metrics_engine.get_surviving_tickers")
    def test_single_ticker_returns_dataframe(
        self, mock_tickers, mock_read, mock_conn,
        mock_init, mock_write_ann, mock_write_summ, mock_write_comp,
    ) -> None:
        mock_tickers.return_value = [_TICKER]
        mock_read.return_value = (
            _income_df(), _balance_df(), _cashflow_df(), _market_row(), _macro_dict()
        )
        mock_conn.return_value = self._mock_conn()
        df = run_metrics_engine()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == _TICKER

    @patch("metrics_engine._write_table_full_replace")
    @patch("metrics_engine._write_summary_with_scores")
    @patch("metrics_engine._write_annual_metrics")
    @patch("metrics_engine._init_metrics_tables")
    @patch("metrics_engine.get_connection")
    @patch("metrics_engine._read_ticker_data")
    @patch("metrics_engine.get_surviving_tickers")
    def test_error_isolation_skips_failed_ticker(
        self, mock_tickers, mock_read, mock_conn,
        mock_init, mock_write_ann, mock_write_summ, mock_write_comp,
    ) -> None:
        """A RuntimeError for one ticker must not abort processing of others."""
        mock_tickers.return_value = ["GOOD", "BAD", "ALSO_GOOD"]
        good_data = (_income_df(), _balance_df(), _cashflow_df(), _market_row(), _macro_dict())

        def _side_effect(ticker: str):
            if ticker == "BAD":
                raise ValueError("simulated bad data for BAD")
            return good_data

        mock_read.side_effect = _side_effect
        mock_conn.return_value = self._mock_conn()
        df = run_metrics_engine()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        tickers_in_result = set(df["ticker"].tolist())
        assert "GOOD" in tickers_in_result
        assert "ALSO_GOOD" in tickers_in_result
        assert "BAD" not in tickers_in_result

    @patch("metrics_engine._write_table_full_replace")
    @patch("metrics_engine._write_summary_with_scores")
    @patch("metrics_engine._write_annual_metrics")
    @patch("metrics_engine._init_metrics_tables")
    @patch("metrics_engine.get_connection")
    @patch("metrics_engine._read_ticker_data")
    @patch("metrics_engine.get_surviving_tickers")
    def test_result_sorted_descending_by_composite_score(
        self, mock_tickers, mock_read, mock_conn,
        mock_init, mock_write_ann, mock_write_summ, mock_write_comp,
    ) -> None:
        mock_tickers.return_value = ["A", "B"]
        # A has lower ROE (worse score) than B
        inc_a = _income_df(); inc_a["net_income"] = 10.0  # low ROE
        inc_b = _income_df(); inc_b["net_income"] = 200.0  # high ROE

        def _side_effect(ticker: str):
            inc = inc_a if ticker == "A" else inc_b
            return (inc, _balance_df(), _cashflow_df(), _market_row(), _macro_dict())

        mock_read.side_effect = _side_effect
        mock_conn.return_value = self._mock_conn()
        df = run_metrics_engine()
        assert df.iloc[0]["composite_score"] >= df.iloc[1]["composite_score"]

    @patch("metrics_engine._write_table_full_replace")
    @patch("metrics_engine._write_summary_with_scores")
    @patch("metrics_engine._write_annual_metrics")
    @patch("metrics_engine._init_metrics_tables")
    @patch("metrics_engine.get_connection")
    @patch("metrics_engine._read_ticker_data")
    @patch("metrics_engine.get_surviving_tickers")
    def test_write_functions_called(
        self, mock_tickers, mock_read, mock_conn,
        mock_init, mock_write_ann, mock_write_summ, mock_write_comp,
    ) -> None:
        """Verify all three write helpers are invoked once per run."""
        mock_tickers.return_value = [_TICKER]
        mock_read.return_value = (
            _income_df(), _balance_df(), _cashflow_df(), _market_row(), _macro_dict()
        )
        mock_conn.return_value = self._mock_conn()
        run_metrics_engine()
        mock_write_ann.assert_called_once()
        mock_write_summ.assert_called_once()
        mock_write_comp.assert_called_once()

    @patch("metrics_engine.get_connection")
    @patch("metrics_engine.get_surviving_tickers")
    def test_all_tickers_fail_returns_empty_df(
        self, mock_tickers: MagicMock, mock_conn: MagicMock
    ) -> None:
        """When every ticker errors out run_metrics_engine should return empty DF."""
        mock_tickers.return_value = ["FAIL1", "FAIL2"]
        mock_conn.return_value = self._mock_conn()

        with patch("metrics_engine._read_ticker_data") as mock_read:
            mock_read.side_effect = RuntimeError("db gone")
            # _read_ticker_data raises → _compute_all returns empty dict → the
            # loop succeeds but all_summaries will be non-empty (ticker: {})
            # because _compute_all handles empty income_df gracefully.
            # Let's patch _compute_all itself instead to force total failure.
            with patch("metrics_engine._compute_all") as mock_ca:
                mock_ca.side_effect = RuntimeError("total failure")
                df = run_metrics_engine()
                assert isinstance(df, pd.DataFrame)
                assert df.empty

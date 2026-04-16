"""Unit tests for data_acquisition/market_data.py and data_acquisition/macro_data.py.

All external I/O (yfinance, FRED API, file system, config) is mocked.
No real network requests, disk writes, or API keys are required.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from data_acquisition.market_data import (
    MARKET_DATA_COLUMNS,
    _NUMERIC_MARKET_COLS,
    _assemble_market_df,
    _empty_market_row,
    _extract_extended_info,
    _extract_fast_info,
    _fetch_single_market_row,
    fetch_historical_pe,
    fetch_market_data,
)
from data_acquisition.macro_data import (
    _FRED_MISSING,
    _MACRO_CACHE_TTL_SECONDS,
    _fetch_all_macro_series,
    _fetch_fred_latest,
    _fetch_yf_treasury_fallback,
    _fetch_yf_usdcad_fallback,
    _load_macro_cache,
    _macro_cache_is_fresh,
    _parse_fred_latest,
    _save_macro_cache,
    fetch_macro_data,
    get_risk_free_rate,
    get_usd_cad_rate,
)


# ===========================================================================
# Fixtures / helpers
# ===========================================================================

def _make_mock_ticker(
    last_price: float = 150.0,
    market_cap: float = 2_000_000_000_000.0,
    shares: float = 15_000_000_000.0,
    year_high: float = 200.0,
    year_low: float = 120.0,
    three_month_average_volume: float = 50_000_000.0,
    enterprise_value: float = 2_100_000_000_000.0,
    trailing_pe: float = 28.5,
    dividend_yield: float = 0.005,
) -> MagicMock:
    """Return a MagicMock yfinance Ticker with preconfigured fast_info and info."""
    mock_t = MagicMock()
    fi = MagicMock()
    fi.last_price = last_price
    fi.market_cap = market_cap
    fi.shares = shares
    fi.year_high = year_high
    fi.year_low = year_low
    fi.three_month_average_volume = three_month_average_volume
    mock_t.fast_info = fi
    mock_t.info = {
        "enterpriseValue": enterprise_value,
        "trailingPE": trailing_pe,
        "dividendYield": dividend_yield,
    }
    return mock_t


def _make_fred_response(value: str = "4.25") -> dict[str, Any]:
    """Return a minimal FRED observations response dict."""
    return {
        "observations": [
            {"date": "2024-01-02", "value": value},
            {"date": "2024-01-01", "value": value},
        ]
    }


# ===========================================================================
# Tests: MARKET_DATA_COLUMNS
# ===========================================================================

class TestMarketDataColumns:
    def test_columns_tuple_is_not_empty(self) -> None:
        assert len(MARKET_DATA_COLUMNS) > 0

    def test_ticker_and_as_of_date_present(self) -> None:
        assert "ticker" in MARKET_DATA_COLUMNS
        assert "as_of_date" in MARKET_DATA_COLUMNS

    def test_numeric_cols_excluded_from_string_cols(self) -> None:
        non_numeric = {"ticker", "as_of_date"}
        for col in _NUMERIC_MARKET_COLS:
            assert col not in non_numeric


# ===========================================================================
# Tests: _empty_market_row
# ===========================================================================

class TestEmptyMarketRow:
    def test_ticker_and_date_populated(self) -> None:
        row = _empty_market_row("AAPL", "2024-01-01")
        assert row["ticker"] == "AAPL"
        assert row["as_of_date"] == "2024-01-01"

    def test_numeric_fields_are_nan(self) -> None:
        row = _empty_market_row("AAPL", "2024-01-01")
        for col in _NUMERIC_MARKET_COLS:
            assert math.isnan(row[col]), f"{col} should be NaN"

    def test_all_market_columns_present(self) -> None:
        row = _empty_market_row("XYZ", "2024-01-01")
        for col in MARKET_DATA_COLUMNS:
            assert col in row


# ===========================================================================
# Tests: _assemble_market_df
# ===========================================================================

class TestAssembleMarketDf:
    def test_empty_rows_returns_correct_columns(self) -> None:
        df = _assemble_market_df([])
        assert list(df.columns) == list(MARKET_DATA_COLUMNS)
        assert len(df) == 0

    def test_numeric_columns_are_float64(self) -> None:
        rows = [_empty_market_row("AAPL", "2024-01-01")]
        df = _assemble_market_df(rows)
        for col in _NUMERIC_MARKET_COLS:
            assert df[col].dtype == "float64", f"{col} should be float64"

    def test_rows_preserved(self) -> None:
        rows = [
            {"ticker": "AAPL", "as_of_date": "2024-01-01",
             **{c: 1.0 for c in _NUMERIC_MARKET_COLS}},
            {"ticker": "MSFT", "as_of_date": "2024-01-01",
             **{c: 2.0 for c in _NUMERIC_MARKET_COLS}},
        ]
        df = _assemble_market_df(rows)
        assert len(df) == 2
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[1]["ticker"] == "MSFT"


# ===========================================================================
# Tests: _extract_fast_info
# ===========================================================================

class TestExtractFastInfo:
    def test_returns_expected_keys(self) -> None:
        mock_t = _make_mock_ticker()
        result = _extract_fast_info(mock_t, "AAPL")
        expected_keys = {
            "current_price_usd", "market_cap_usd", "shares_outstanding",
            "high_52w", "low_52w", "avg_volume_3m",
        }
        assert set(result.keys()) == expected_keys

    def test_values_from_fast_info(self) -> None:
        mock_t = _make_mock_ticker(last_price=155.0, year_high=210.0, year_low=100.0)
        result = _extract_fast_info(mock_t, "AAPL")
        assert result["current_price_usd"] == pytest.approx(155.0)
        assert result["high_52w"] == pytest.approx(210.0)
        assert result["low_52w"] == pytest.approx(100.0)

    def test_none_attribute_returns_nan(self) -> None:
        mock_t = MagicMock()
        fi = MagicMock()
        fi.last_price = None
        mock_t.fast_info = fi
        result = _extract_fast_info(mock_t, "AAPL")
        assert math.isnan(result["current_price_usd"])

    def test_missing_attribute_returns_nan(self) -> None:
        mock_t = MagicMock()
        fi = MagicMock(spec=[])  # no attributes at all
        mock_t.fast_info = fi
        result = _extract_fast_info(mock_t, "AAPL")
        assert math.isnan(result["current_price_usd"])


# ===========================================================================
# Tests: _extract_extended_info
# ===========================================================================

class TestExtractExtendedInfo:
    def test_returns_expected_keys(self) -> None:
        mock_t = _make_mock_ticker()
        result = _extract_extended_info(mock_t, "AAPL")
        assert set(result.keys()) == {"enterprise_value_usd", "pe_ratio_trailing", "dividend_yield"}

    def test_values_from_info_dict(self) -> None:
        mock_t = _make_mock_ticker(enterprise_value=2_100_000_000_000.0, trailing_pe=28.5)
        result = _extract_extended_info(mock_t, "AAPL")
        assert result["enterprise_value_usd"] == pytest.approx(2_100_000_000_000.0)
        assert result["pe_ratio_trailing"] == pytest.approx(28.5)

    def test_missing_key_returns_nan(self) -> None:
        mock_t = MagicMock()
        mock_t.info = {}  # empty info
        result = _extract_extended_info(mock_t, "AAPL")
        assert math.isnan(result["pe_ratio_trailing"])
        assert math.isnan(result["enterprise_value_usd"])

    def test_info_raises_returns_nan_fields(self) -> None:
        mock_t = MagicMock()
        mock_t.info = MagicMock(side_effect=Exception("network error"))
        # info is a property, make it raise
        type(mock_t).info = property(lambda self: (_ for _ in ()).throw(Exception("network error")))
        result = _extract_extended_info(mock_t, "AAPL")
        assert math.isnan(result["pe_ratio_trailing"])


# ===========================================================================
# Tests: fetch_market_data
# ===========================================================================

class TestFetchMarketData:
    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_returns_dataframe_with_correct_columns(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.return_value = _make_mock_ticker()
        df = fetch_market_data(["AAPL"])
        assert list(df.columns) == list(MARKET_DATA_COLUMNS)

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_populates_values_correctly(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.return_value = _make_mock_ticker(
            last_price=150.0, market_cap=2e12, trailing_pe=28.5
        )
        df = fetch_market_data(["AAPL"])
        assert df.iloc[0]["current_price_usd"] == pytest.approx(150.0)
        assert df.iloc[0]["market_cap_usd"] == pytest.approx(2e12)
        assert df.iloc[0]["pe_ratio_trailing"] == pytest.approx(28.5)

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_error_isolation_returns_nan_row(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.side_effect = Exception("network failure")
        df = fetch_market_data(["FAIL"])
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "FAIL"
        assert math.isnan(df.iloc[0]["current_price_usd"])

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_mixed_success_and_failure(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.side_effect = [
            _make_mock_ticker(last_price=100.0),
            Exception("fail"),
        ]
        df = fetch_market_data(["AAPL", "FAIL"])
        assert len(df) == 2
        assert df.iloc[0]["current_price_usd"] == pytest.approx(100.0)
        assert math.isnan(df.iloc[1]["current_price_usd"])

    @patch("data_acquisition.market_data._yf_limiter")
    def test_empty_ticker_list_returns_empty_df(self, mock_limiter: MagicMock) -> None:
        df = fetch_market_data([])
        assert len(df) == 0
        assert list(df.columns) == list(MARKET_DATA_COLUMNS)

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_all_numeric_cols_are_float64(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.return_value = _make_mock_ticker()
        df = fetch_market_data(["AAPL"])
        for col in _NUMERIC_MARKET_COLS:
            assert df[col].dtype == "float64", f"{col} dtype should be float64"

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_tsx_ticker_passed_as_is(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        """TSX tickers with .TO suffix must be passed to yfinance unchanged."""
        mock_ticker_cls.return_value = _make_mock_ticker()
        fetch_market_data(["SHOP.TO"])
        mock_ticker_cls.assert_called_once_with("SHOP.TO")

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_as_of_date_is_string(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.return_value = _make_mock_ticker()
        df = fetch_market_data(["AAPL"])
        assert isinstance(df.iloc[0]["as_of_date"], str)
        # ISO 8601 format: YYYY-MM-DD
        assert len(df.iloc[0]["as_of_date"]) == 10


# ===========================================================================
# Tests: fetch_historical_pe
# ===========================================================================

class TestFetchHistoricalPe:
    def _make_hist_df(self) -> pd.DataFrame:
        idx = pd.DatetimeIndex(["2021-12-31", "2022-12-30", "2023-12-29"])
        return pd.DataFrame({"Close": [100.0, 110.0, 120.0]}, index=idx)

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_returns_expected_columns(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = self._make_hist_df()
        df = fetch_historical_pe("AAPL", years=3)
        assert "ticker" in df.columns
        assert "calendar_year" in df.columns
        assert "close_price" in df.columns

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_ticker_column_populated(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = self._make_hist_df()
        df = fetch_historical_pe("AAPL", years=3)
        assert all(df["ticker"] == "AAPL")

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_sorted_ascending_by_year(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = self._make_hist_df()
        df = fetch_historical_pe("AAPL", years=3)
        years = df["calendar_year"].tolist()
        assert years == sorted(years)

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_empty_history_returns_empty_df(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()
        df = fetch_historical_pe("AAPL")
        assert len(df) == 0
        assert "ticker" in df.columns

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_yfinance_error_returns_empty_df(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.side_effect = Exception("connection refused")
        df = fetch_historical_pe("AAPL")
        assert len(df) == 0
        assert "close_price" in df.columns

    @patch("data_acquisition.market_data._yf_limiter")
    @patch("data_acquisition.market_data.yf.Ticker")
    def test_history_called_with_period_param(
        self, mock_ticker_cls: MagicMock, mock_limiter: MagicMock
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()
        fetch_historical_pe("AAPL", years=5)
        mock_ticker_cls.return_value.history.assert_called_once_with(
            period="5y", interval="1d", auto_adjust=True
        )


# ===========================================================================
# Tests: _parse_fred_latest
# ===========================================================================

class TestParseFredLatest:
    def test_returns_float_from_valid_response(self) -> None:
        data = _make_fred_response("4.25")
        result = _parse_fred_latest(data, "DGS10")
        assert result == pytest.approx(4.25)

    def test_skips_dot_missing_values(self) -> None:
        data = {
            "observations": [
                {"date": "2024-01-02", "value": "."},
                {"date": "2024-01-01", "value": "4.20"},
            ]
        }
        result = _parse_fred_latest(data, "DGS10")
        assert result == pytest.approx(4.20)

    def test_all_missing_returns_none(self) -> None:
        data = {
            "observations": [
                {"date": "2024-01-01", "value": "."},
                {"date": "2024-01-02", "value": "."},
            ]
        }
        result = _parse_fred_latest(data, "DGS10")
        assert result is None

    def test_empty_observations_returns_none(self) -> None:
        data = {"observations": []}
        result = _parse_fred_latest(data, "DGS10")
        assert result is None

    def test_non_dict_response_returns_none(self) -> None:
        result = _parse_fred_latest(["unexpected", "list"], "DGS10")
        assert result is None

    def test_non_numeric_value_skipped(self) -> None:
        data = {
            "observations": [
                {"date": "2024-01-02", "value": "N/A"},
                {"date": "2024-01-01", "value": "3.50"},
            ]
        }
        result = _parse_fred_latest(data, "DGS10")
        assert result == pytest.approx(3.50)


# ===========================================================================
# Tests: _fetch_fred_latest
# ===========================================================================

class TestFetchFredLatest:
    @patch("data_acquisition.macro_data.resilient_request")
    @patch("data_acquisition.macro_data.get_fred_key", return_value="test_key")
    def test_returns_float_on_success(
        self, mock_key: MagicMock, mock_request: MagicMock
    ) -> None:
        mock_request.return_value = _make_fred_response("4.50")
        result = _fetch_fred_latest("DGS10")
        assert result == pytest.approx(4.50)

    @patch("data_acquisition.macro_data.resilient_request")
    @patch("data_acquisition.macro_data.get_fred_key", side_effect=EnvironmentError("no key"))
    def test_returns_none_when_key_missing(
        self, mock_key: MagicMock, mock_request: MagicMock
    ) -> None:
        result = _fetch_fred_latest("DGS10")
        assert result is None
        mock_request.assert_not_called()

    @patch("data_acquisition.macro_data.resilient_request", side_effect=Exception("timeout"))
    @patch("data_acquisition.macro_data.get_fred_key", return_value="test_key")
    def test_returns_none_on_request_error(
        self, mock_key: MagicMock, mock_request: MagicMock
    ) -> None:
        result = _fetch_fred_latest("DGS10")
        assert result is None


# ===========================================================================
# Tests: yfinance fallbacks
# ===========================================================================

class TestYFinanceFallbacks:
    @patch("data_acquisition.macro_data.yf.Ticker")
    def test_treasury_fallback_divides_by_100(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value.fast_info.last_price = 4.25
        result = _fetch_yf_treasury_fallback()
        assert result == pytest.approx(0.0425)

    @patch("data_acquisition.macro_data.yf.Ticker")
    def test_treasury_fallback_none_returns_nan(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value.fast_info.last_price = None
        result = _fetch_yf_treasury_fallback()
        assert math.isnan(result)

    @patch("data_acquisition.macro_data.yf.Ticker")
    def test_treasury_fallback_exception_returns_nan(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.side_effect = Exception("fail")
        result = _fetch_yf_treasury_fallback()
        assert math.isnan(result)

    @patch("data_acquisition.macro_data.yf.Ticker")
    def test_usdcad_fallback_returns_rate_directly(self, mock_ticker_cls: MagicMock) -> None:
        """CADUSD=X already gives USD per 1 CAD — no inversion needed."""
        mock_ticker_cls.return_value.fast_info.last_price = 0.74
        result = _fetch_yf_usdcad_fallback()
        assert result == pytest.approx(0.74)

    @patch("data_acquisition.macro_data.yf.Ticker")
    def test_usdcad_fallback_none_returns_nan(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.return_value.fast_info.last_price = None
        result = _fetch_yf_usdcad_fallback()
        assert math.isnan(result)

    @patch("data_acquisition.macro_data.yf.Ticker")
    def test_usdcad_fallback_exception_returns_nan(self, mock_ticker_cls: MagicMock) -> None:
        mock_ticker_cls.side_effect = RuntimeError("network")
        result = _fetch_yf_usdcad_fallback()
        assert math.isnan(result)


# ===========================================================================
# Tests: _fetch_all_macro_series
# ===========================================================================

class TestFetchAllMacroSeries:
    @patch("data_acquisition.macro_data._fetch_yf_treasury_fallback")
    @patch("data_acquisition.macro_data._fetch_yf_usdcad_fallback")
    @patch("data_acquisition.macro_data._fetch_fred_latest")
    def test_converts_treasury_percent_to_decimal(
        self,
        mock_fred: MagicMock,
        mock_usd_fallback: MagicMock,
        mock_tsy_fallback: MagicMock,
    ) -> None:
        mock_fred.side_effect = [4.25, 0.74, 3.80]  # DGS10, DEXCAUS, GoC
        result = _fetch_all_macro_series()
        assert result["us_treasury_10yr"] == pytest.approx(0.0425)

    @patch("data_acquisition.macro_data._fetch_yf_treasury_fallback")
    @patch("data_acquisition.macro_data._fetch_yf_usdcad_fallback")
    @patch("data_acquisition.macro_data._fetch_fred_latest")
    def test_usd_cad_stored_directly(
        self,
        mock_fred: MagicMock,
        mock_usd_fallback: MagicMock,
        mock_tsy_fallback: MagicMock,
    ) -> None:
        mock_fred.side_effect = [4.25, 0.74, 3.80]  # DGS10, DEXCAUS, GoC
        result = _fetch_all_macro_series()
        assert result["usd_cad_rate"] == pytest.approx(0.74)

    @patch("data_acquisition.macro_data._fetch_yf_treasury_fallback", return_value=0.042)
    @patch("data_acquisition.macro_data._fetch_yf_usdcad_fallback")
    @patch("data_acquisition.macro_data._fetch_fred_latest", return_value=None)
    def test_fred_failure_triggers_yf_fallback_for_treasury(
        self,
        mock_fred: MagicMock,
        mock_usd_fallback: MagicMock,
        mock_tsy_fallback: MagicMock,
    ) -> None:
        result = _fetch_all_macro_series()
        assert result["us_treasury_10yr"] == pytest.approx(0.042)

    @patch("data_acquisition.macro_data._fetch_yf_treasury_fallback")
    @patch("data_acquisition.macro_data._fetch_yf_usdcad_fallback", return_value=0.73)
    @patch("data_acquisition.macro_data._fetch_fred_latest")
    def test_fred_failure_triggers_yf_fallback_for_usdcad(
        self,
        mock_fred: MagicMock,
        mock_usd_fallback: MagicMock,
        mock_tsy_fallback: MagicMock,
    ) -> None:
        mock_fred.side_effect = [4.25, None, 3.80]  # treasury ok, DEXCAUS fails, GoC ok
        result = _fetch_all_macro_series()
        assert result["usd_cad_rate"] == pytest.approx(0.73)

    @patch("data_acquisition.macro_data._fetch_yf_treasury_fallback", return_value=float("nan"))
    @patch("data_acquisition.macro_data._fetch_yf_usdcad_fallback", return_value=float("nan"))
    @patch("data_acquisition.macro_data._fetch_fred_latest", return_value=None)
    def test_both_sources_fail_returns_nan(
        self,
        mock_fred: MagicMock,
        mock_usd_fallback: MagicMock,
        mock_tsy_fallback: MagicMock,
    ) -> None:
        result = _fetch_all_macro_series()
        assert math.isnan(result["us_treasury_10yr"])
        assert math.isnan(result["usd_cad_rate"])

    @patch("data_acquisition.macro_data._fetch_yf_treasury_fallback")
    @patch("data_acquisition.macro_data._fetch_yf_usdcad_fallback")
    @patch("data_acquisition.macro_data._fetch_fred_latest")
    def test_as_of_date_is_iso_string(
        self,
        mock_fred: MagicMock,
        mock_usd_fallback: MagicMock,
        mock_tsy_fallback: MagicMock,
    ) -> None:
        mock_fred.side_effect = [4.25, 0.74, 3.80]  # DGS10, DEXCAUS, GoC
        result = _fetch_all_macro_series()
        assert isinstance(result["as_of_date"], str)
        assert len(result["as_of_date"]) == 10  # YYYY-MM-DD


# ===========================================================================
# Tests: macro data cache helpers
# ===========================================================================

class TestMacroCacheHelpers:
    def test_macro_cache_is_fresh_file_not_exist(self, tmp_path: pathlib.Path) -> None:
        cache_path = tmp_path / "macro_data.json"
        assert _macro_cache_is_fresh(cache_path) is False

    def test_macro_cache_is_fresh_new_file(self, tmp_path: pathlib.Path) -> None:
        cache_path = tmp_path / "macro_data.json"
        cache_path.write_text("{}")
        assert _macro_cache_is_fresh(cache_path) is True

    def test_macro_cache_is_stale_old_file(self, tmp_path: pathlib.Path) -> None:
        cache_path = tmp_path / "macro_data.json"
        cache_path.write_text("{}")
        import time as _time
        old_mtime = _time.time() - (_MACRO_CACHE_TTL_SECONDS + 100)
        import os
        os.utime(cache_path, (old_mtime, old_mtime))
        assert _macro_cache_is_fresh(cache_path) is False

    def test_save_and_load_roundtrip(self, tmp_path: pathlib.Path) -> None:
        cache_path = tmp_path / "macro_data.json"
        data = {"us_treasury_10yr": 0.0425, "usd_cad_rate": 0.74, "as_of_date": "2024-01-01"}
        _save_macro_cache(data, cache_path)
        loaded = _load_macro_cache(cache_path)
        assert loaded["us_treasury_10yr"] == pytest.approx(0.0425)
        assert loaded["usd_cad_rate"] == pytest.approx(0.74)

    def test_save_nan_serializes_as_null(self, tmp_path: pathlib.Path) -> None:
        cache_path = tmp_path / "macro_data.json"
        data = {"us_treasury_10yr": float("nan"), "usd_cad_rate": 0.74, "as_of_date": "2024-01-01"}
        _save_macro_cache(data, cache_path)
        raw = json.loads(cache_path.read_text())
        assert raw["us_treasury_10yr"] is None

    def test_load_null_restores_as_nan(self, tmp_path: pathlib.Path) -> None:
        cache_path = tmp_path / "macro_data.json"
        cache_path.write_text(
            '{"us_treasury_10yr": null, "usd_cad_rate": 0.74, "as_of_date": "2024-01-01"}'
        )
        loaded = _load_macro_cache(cache_path)
        assert math.isnan(loaded["us_treasury_10yr"])

    def test_save_write_error_does_not_raise(self, tmp_path: pathlib.Path) -> None:
        """_save_macro_cache must not propagate filesystem errors."""
        bad_path = tmp_path / "nonexistent_dir" / "macro_data.json"
        data = {"us_treasury_10yr": 0.04, "usd_cad_rate": 0.74, "as_of_date": "2024-01-01"}
        # Should log ERROR but not raise.
        _save_macro_cache(data, bad_path)


# ===========================================================================
# Tests: fetch_macro_data (orchestrator)
# ===========================================================================

class TestFetchMacroData:
    @patch("data_acquisition.macro_data._resolve_macro_cache_path")
    @patch("data_acquisition.macro_data._macro_cache_is_fresh", return_value=True)
    @patch("data_acquisition.macro_data._load_macro_cache")
    @patch("data_acquisition.macro_data._fetch_all_macro_series")
    def test_cache_hit_returns_cached_data(
        self,
        mock_fetch: MagicMock,
        mock_load: MagicMock,
        mock_fresh: MagicMock,
        mock_path: MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        expected = {"us_treasury_10yr": 0.04, "usd_cad_rate": 0.74, "as_of_date": "2024-01-01"}
        mock_load.return_value = expected
        mock_path.return_value = tmp_path / "macro_data.json"

        result = fetch_macro_data()

        mock_fetch.assert_not_called()
        assert result == expected

    @patch("data_acquisition.macro_data._save_macro_cache")
    @patch("data_acquisition.macro_data._resolve_macro_cache_path")
    @patch("data_acquisition.macro_data._macro_cache_is_fresh", return_value=False)
    @patch("data_acquisition.macro_data._fetch_all_macro_series")
    def test_cache_miss_fetches_and_saves(
        self,
        mock_fetch: MagicMock,
        mock_fresh: MagicMock,
        mock_path: MagicMock,
        mock_save: MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        fresh_data = {"us_treasury_10yr": 0.043, "usd_cad_rate": 0.73, "as_of_date": "2024-01-02"}
        mock_fetch.return_value = fresh_data
        mock_path.return_value = tmp_path / "macro_data.json"

        result = fetch_macro_data()

        mock_fetch.assert_called_once()
        mock_save.assert_called_once()
        assert result == fresh_data

    @patch("data_acquisition.macro_data._save_macro_cache")
    @patch("data_acquisition.macro_data._resolve_macro_cache_path")
    @patch("data_acquisition.macro_data._macro_cache_is_fresh", return_value=False)
    @patch("data_acquisition.macro_data._fetch_all_macro_series")
    def test_returns_dict_with_expected_keys(
        self,
        mock_fetch: MagicMock,
        mock_fresh: MagicMock,
        mock_path: MagicMock,
        mock_save: MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        mock_fetch.return_value = {
            "us_treasury_10yr": 0.04,
            "usd_cad_rate": 0.74,
            "as_of_date": "2024-01-01",
        }
        mock_path.return_value = tmp_path / "macro_data.json"

        result = fetch_macro_data()
        assert "us_treasury_10yr" in result
        assert "usd_cad_rate" in result
        assert "as_of_date" in result


# ===========================================================================
# Tests: get_risk_free_rate
# ===========================================================================

class TestGetRiskFreeRate:
    @patch("data_acquisition.macro_data.fetch_macro_data")
    def test_returns_decimal_from_macro_data(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "us_treasury_10yr": 0.0425,
            "usd_cad_rate": 0.74,
            "as_of_date": "2024-01-01",
        }
        rate = get_risk_free_rate()
        assert rate == pytest.approx(0.0425)

    @patch("data_acquisition.macro_data.fetch_macro_data")
    def test_returns_nan_when_unavailable(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "us_treasury_10yr": float("nan"),
            "usd_cad_rate": 0.74,
            "as_of_date": "2024-01-01",
        }
        rate = get_risk_free_rate()
        assert math.isnan(rate)

    @patch("data_acquisition.macro_data.fetch_macro_data")
    def test_returns_nan_when_key_missing(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"usd_cad_rate": 0.74, "as_of_date": "2024-01-01"}
        rate = get_risk_free_rate()
        assert math.isnan(rate)

    @patch("data_acquisition.macro_data.fetch_macro_data")
    def test_returns_nan_when_none_value(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"us_treasury_10yr": None, "as_of_date": "2024-01-01"}
        rate = get_risk_free_rate()
        assert math.isnan(rate)


# ===========================================================================
# Tests: get_usd_cad_rate
# ===========================================================================

class TestGetUsdCadRate:
    @patch("data_acquisition.macro_data.fetch_macro_data")
    def test_returns_rate_from_macro_data(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "us_treasury_10yr": 0.0425,
            "usd_cad_rate": 0.74,
            "as_of_date": "2024-01-01",
        }
        rate = get_usd_cad_rate()
        assert rate == pytest.approx(0.74)

    @patch("data_acquisition.macro_data.fetch_macro_data")
    def test_returns_nan_when_unavailable(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "us_treasury_10yr": 0.04,
            "usd_cad_rate": float("nan"),
            "as_of_date": "2024-01-01",
        }
        rate = get_usd_cad_rate()
        assert math.isnan(rate)

    @patch("data_acquisition.macro_data.fetch_macro_data")
    def test_returns_nan_when_key_missing(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"us_treasury_10yr": 0.04, "as_of_date": "2024-01-01"}
        rate = get_usd_cad_rate()
        assert math.isnan(rate)

    @patch("data_acquisition.macro_data.fetch_macro_data")
    def test_returns_nan_when_none_value(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"usd_cad_rate": None, "as_of_date": "2024-01-01"}
        rate = get_usd_cad_rate()
        assert math.isnan(rate)


# ===========================================================================
# Tests: fetch_macro_data use_cache parameter
# ===========================================================================

class TestFetchMacroDataUseCache:
    """fetch_macro_data(use_cache=False) must bypass the cache entirely."""

    @patch("data_acquisition.macro_data._save_macro_cache")
    @patch("data_acquisition.macro_data._resolve_macro_cache_path")
    @patch("data_acquisition.macro_data._macro_cache_is_fresh", return_value=True)
    @patch("data_acquisition.macro_data._load_macro_cache")
    @patch("data_acquisition.macro_data._fetch_all_macro_series")
    def test_use_cache_false_skips_fresh_cache(
        self,
        mock_fetch: MagicMock,
        mock_load: MagicMock,
        mock_fresh: MagicMock,
        mock_path: MagicMock,
        mock_save: MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        """Even if cache is fresh, use_cache=False must trigger a fresh fetch."""
        mock_path.return_value = tmp_path / "macro_data.json"
        fresh_data = {
            "us_treasury_10yr": 0.045,
            "usd_cad_rate": 0.73,
            "as_of_date": "2024-02-01",
        }
        mock_fetch.return_value = fresh_data

        result = fetch_macro_data(use_cache=False)

        # _load_macro_cache should NOT be called when use_cache=False.
        mock_load.assert_not_called()
        # _fetch_all_macro_series should be called instead.
        mock_fetch.assert_called_once()
        assert result == fresh_data

    @patch("data_acquisition.macro_data._save_macro_cache")
    @patch("data_acquisition.macro_data._resolve_macro_cache_path")
    @patch("data_acquisition.macro_data._macro_cache_is_fresh", return_value=True)
    @patch("data_acquisition.macro_data._load_macro_cache")
    @patch("data_acquisition.macro_data._fetch_all_macro_series")
    def test_use_cache_true_uses_fresh_cache(
        self,
        mock_fetch: MagicMock,
        mock_load: MagicMock,
        mock_fresh: MagicMock,
        mock_path: MagicMock,
        mock_save: MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        """use_cache=True (default) should load from cache when fresh."""
        mock_path.return_value = tmp_path / "macro_data.json"
        cached_data = {
            "us_treasury_10yr": 0.042,
            "usd_cad_rate": 0.74,
            "as_of_date": "2024-01-01",
        }
        mock_load.return_value = cached_data

        result = fetch_macro_data(use_cache=True)

        mock_load.assert_called_once()
        mock_fetch.assert_not_called()
        assert result == cached_data

    @patch("data_acquisition.macro_data._save_macro_cache")
    @patch("data_acquisition.macro_data._resolve_macro_cache_path")
    @patch("data_acquisition.macro_data._macro_cache_is_fresh", return_value=False)
    @patch("data_acquisition.macro_data._fetch_all_macro_series")
    def test_use_cache_true_stale_cache_fetches(
        self,
        mock_fetch: MagicMock,
        mock_fresh: MagicMock,
        mock_path: MagicMock,
        mock_save: MagicMock,
        tmp_path: pathlib.Path,
    ) -> None:
        """use_cache=True with a stale cache still triggers a fresh fetch."""
        mock_path.return_value = tmp_path / "macro_data.json"
        fresh_data = {
            "us_treasury_10yr": 0.044,
            "usd_cad_rate": 0.72,
            "as_of_date": "2024-02-02",
        }
        mock_fetch.return_value = fresh_data

        result = fetch_macro_data(use_cache=True)

        mock_fetch.assert_called_once()
        assert result == fresh_data


# ===========================================================================
# Tests: config-driven FRED series IDs
# ===========================================================================

class TestFredConfigDriven:
    """Verify FRED series IDs and base URL are read from config, not hardcoded."""

    @patch("data_acquisition.macro_data._fetch_yf_treasury_fallback")
    @patch("data_acquisition.macro_data._fetch_yf_usdcad_fallback")
    @patch("data_acquisition.macro_data._fetch_fred_latest")
    @patch("data_acquisition.macro_data.get_config")
    def test_series_ids_read_from_config(
        self,
        mock_config: MagicMock,
        mock_fred: MagicMock,
        mock_usd_fallback: MagicMock,
        mock_tsy_fallback: MagicMock,
    ) -> None:
        """_fetch_all_macro_series must use series IDs from config, not hardcoded."""
        mock_config.return_value = {
            "data_sources": {
                "fred": {
                    "series": {
                        "treasury_10yr": "CUSTOM_TSY",
                        "usd_cad": "CUSTOM_FX",
                        "goc_10yr": "CUSTOM_GOC",
                    },
                },
            },
        }
        mock_fred.side_effect = [4.25, 0.74, 3.80]  # TSY, FX, GoC

        _fetch_all_macro_series()

        # Verify the custom series IDs were passed to _fetch_fred_latest.
        calls = [c[0][0] for c in mock_fred.call_args_list]
        assert calls[0] == "CUSTOM_TSY"
        assert calls[1] == "CUSTOM_FX"
        assert calls[2] == "CUSTOM_GOC"

    @patch("data_acquisition.macro_data._fetch_yf_treasury_fallback")
    @patch("data_acquisition.macro_data._fetch_yf_usdcad_fallback")
    @patch("data_acquisition.macro_data._fetch_fred_latest")
    @patch("data_acquisition.macro_data.get_config")
    def test_defaults_to_standard_series_when_config_absent(
        self,
        mock_config: MagicMock,
        mock_fred: MagicMock,
        mock_usd_fallback: MagicMock,
        mock_tsy_fallback: MagicMock,
    ) -> None:
        """When config lacks series keys, fall back to DGS10, DEXCAUS, IRLTLT01CAM156N."""
        mock_config.return_value = {"data_sources": {"fred": {}}}
        mock_fred.side_effect = [4.25, 0.74, 3.80]  # TSY, FX, GoC

        _fetch_all_macro_series()

        calls = [c[0][0] for c in mock_fred.call_args_list]
        assert calls[0] == "DGS10"
        assert calls[1] == "DEXCAUS"
        assert calls[2] == "IRLTLT01CAM156N"

    @patch("data_acquisition.macro_data.resilient_request")
    @patch("data_acquisition.macro_data.get_fred_key", return_value="test_key")
    @patch("data_acquisition.macro_data.get_config")
    def test_base_url_read_from_config(
        self,
        mock_config: MagicMock,
        mock_key: MagicMock,
        mock_request: MagicMock,
    ) -> None:
        """_fetch_fred_latest must use base_url from config, not hardcoded."""
        mock_config.return_value = {
            "data_sources": {
                "fred": {
                    "base_url": "https://custom-fred.example.com/api",
                },
            },
        }
        mock_request.return_value = _make_fred_response("4.10")

        _fetch_fred_latest("DGS10")

        # The first positional arg to resilient_request should be the custom URL.
        actual_url = mock_request.call_args[0][0]
        assert actual_url == "https://custom-fred.example.com/api"

    @patch("data_acquisition.macro_data.resilient_request")
    @patch("data_acquisition.macro_data.get_fred_key", return_value="test_key")
    @patch("data_acquisition.macro_data.get_config")
    def test_default_base_url_when_config_absent(
        self,
        mock_config: MagicMock,
        mock_key: MagicMock,
        mock_request: MagicMock,
    ) -> None:
        """Without base_url in config, fall back to the standard FRED endpoint."""
        mock_config.return_value = {"data_sources": {"fred": {}}}
        mock_request.return_value = _make_fred_response("3.95")

        _fetch_fred_latest("DGS10")

        actual_url = mock_request.call_args[0][0]
        assert "api.stlouisfed.org" in actual_url

    @patch("data_acquisition.macro_data.resilient_request")
    @patch("data_acquisition.macro_data.get_fred_key", return_value="test_key")
    @patch("data_acquisition.macro_data.get_config")
    def test_fred_limiter_passed_to_resilient_request(
        self,
        mock_config: MagicMock,
        mock_key: MagicMock,
        mock_request: MagicMock,
    ) -> None:
        """_fetch_fred_latest must pass fred_limiter to resilient_request."""
        mock_config.return_value = {
            "data_sources": {
                "fred": {
                    "base_url": "https://api.stlouisfed.org/fred/series/observations",
                },
            },
        }
        mock_request.return_value = _make_fred_response("4.00")

        _fetch_fred_latest("DGS10")

        # Verify rate_limiter kwarg was passed.
        _, kwargs = mock_request.call_args
        assert "rate_limiter" in kwargs
        # fred_limiter should be an instance of RateLimiter.
        from data_acquisition.api_config import RateLimiter
        assert isinstance(kwargs["rate_limiter"], RateLimiter)

"""Unit tests for data_acquisition/universe.py.

All external I/O (API calls, filesystem access, time) is mocked.
No real network requests or disk writes are made.
"""

from __future__ import annotations

import pathlib
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from data_acquisition.universe import (
    UNIVERSE_COLUMNS,
    _CACHE_TTL_SECONDS,
    _FMP_SCREENER_LIMIT,
    _cache_is_fresh,
    _empty_universe_df,
    _normalise_fmp_rows,
    fetch_universe,
    filter_universe,
    get_universe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fmp_rows(
    tickers: list[str],
    sector: str = "Technology",
    industry: str = "Software",
    exchange: str = "NYSE",
    market_cap: float = 1_000_000_000.0,
) -> list[dict]:
    """Build minimal FMP screener response rows for testing."""
    return [
        {
            "symbol": t,
            "companyName": f"{t} Inc",
            "marketCap": market_cap,
            "sector": sector,
            "industry": industry,
            "country": "US",
            "exchangeShortName": exchange,
        }
        for t in tickers
    ]


def _make_universe_df(
    tickers: list[str],
    sector: str = "Technology",
    industry: str = "Software",
    exchange: str = "NYSE",
) -> pd.DataFrame:
    """Return a pre-normalised universe DataFrame."""
    rows = _make_fmp_rows(tickers, sector=sector, industry=industry, exchange=exchange)
    return _normalise_fmp_rows(rows, canonical_exchange=exchange)


# ---------------------------------------------------------------------------
# _normalise_fmp_rows tests
# ---------------------------------------------------------------------------

class TestNormaliseFmpRows:
    """Row normalisation and column mapping."""

    def test_output_has_all_canonical_columns(self) -> None:
        rows = _make_fmp_rows(["AAPL"])
        df = _normalise_fmp_rows(rows, canonical_exchange="NYSE")
        assert set(df.columns) == set(UNIVERSE_COLUMNS)

    def test_symbol_mapped_to_ticker(self) -> None:
        rows = _make_fmp_rows(["MSFT"])
        df = _normalise_fmp_rows(rows, canonical_exchange="NYSE")
        assert df.iloc[0]["ticker"] == "MSFT"

    def test_canonical_exchange_overrides_fmp_exchange(self) -> None:
        rows = _make_fmp_rows(["SHOP"], exchange="TSX")
        df = _normalise_fmp_rows(rows, canonical_exchange="TSX")
        assert df.iloc[0]["exchange"] == "TSX"

    def test_market_cap_is_float(self) -> None:
        rows = _make_fmp_rows(["KO"], market_cap=200_000_000_000)
        df = _normalise_fmp_rows(rows, canonical_exchange="NYSE")
        assert df.dtypes["market_cap_usd"] == float

    def test_market_cap_string_coerced(self) -> None:
        row = {"symbol": "X", "companyName": "X Corp", "marketCap": "5000000",
               "sector": "Tech", "industry": "SW", "country": "US",
               "exchangeShortName": "NYSE"}
        df = _normalise_fmp_rows([row], canonical_exchange="NYSE")
        assert df.iloc[0]["market_cap_usd"] == 5_000_000.0

    def test_missing_columns_filled_with_none(self) -> None:
        """FMP may omit 'country' — column must still be present."""
        row = {"symbol": "Y", "companyName": "Y Ltd", "marketCap": 1e9,
               "sector": "Energy", "industry": "Oil", "exchangeShortName": "NYSE"}
        df = _normalise_fmp_rows([row], canonical_exchange="NYSE")
        assert "country" in df.columns

    def test_whitespace_stripped_from_ticker(self) -> None:
        row = {"symbol": "  BRK.B  ", "companyName": "Berkshire",
               "marketCap": 5e11, "sector": "Financials", "industry": "Insurance",
               "country": "US", "exchangeShortName": "NYSE"}
        df = _normalise_fmp_rows([row], canonical_exchange="NYSE")
        assert df.iloc[0]["ticker"] == "BRK.B"

    def test_empty_rows_returns_empty_df(self) -> None:
        df = _normalise_fmp_rows([], canonical_exchange="NYSE")
        assert len(df) == 0
        assert set(df.columns) == set(UNIVERSE_COLUMNS)


# ---------------------------------------------------------------------------
# filter_universe tests
# ---------------------------------------------------------------------------

class TestFilterUniverseExcludeFinancials:
    """Sector and industry-based exclusion logic."""

    def test_tech_sector_not_excluded(self) -> None:
        df = _make_universe_df(["AAPL", "MSFT"], sector="Technology")
        result = filter_universe(df)
        assert len(result) == 2
        assert set(result["ticker"]) == {"AAPL", "MSFT"}

    def test_financial_services_sector_excluded(self) -> None:
        df = _make_universe_df(["JPM", "BAC"], sector="Financial Services")
        result = filter_universe(df)
        assert len(result) == 0

    def test_financials_sector_excluded(self) -> None:
        """Alternate sector name 'Financials' must also be excluded."""
        df = _make_universe_df(["MS", "GS"], sector="Financials")
        result = filter_universe(df)
        assert len(result) == 0

    def test_bank_industry_excluded(self) -> None:
        df = _make_universe_df(["RY", "TD"], sector="Technology", industry="Banks")
        result = filter_universe(df)
        assert len(result) == 0

    def test_insurance_industry_excluded(self) -> None:
        df = _make_universe_df(["MFC"], sector="Technology", industry="Life Insurance")
        result = filter_universe(df)
        assert len(result) == 0

    def test_reit_industry_excluded(self) -> None:
        df = _make_universe_df(["SPG"], sector="Real Estate", industry="REIT - Retail")
        result = filter_universe(df)
        assert len(result) == 0

    def test_mortgage_industry_excluded(self) -> None:
        df = _make_universe_df(["AGNC"], sector="Finance", industry="Mortgage Finance")
        result = filter_universe(df)
        assert len(result) == 0

    def test_savings_industry_excluded(self) -> None:
        df = _make_universe_df(["WFC"], sector="Finance", industry="Savings Institutions")
        result = filter_universe(df)
        assert len(result) == 0

    def test_investment_trust_industry_excluded(self) -> None:
        df = _make_universe_df(["BX"], sector="Finance", industry="Investment Trust")
        result = filter_universe(df)
        assert len(result) == 0

    def test_mixed_df_only_financials_removed(self) -> None:
        tech = _make_universe_df(["AAPL", "GOOG"], sector="Technology")
        fin = _make_universe_df(["JPM"], sector="Financial Services", industry="Banks")
        mixed = pd.concat([tech, fin], ignore_index=True)
        result = filter_universe(mixed)
        assert set(result["ticker"]) == {"AAPL", "GOOG"}

    def test_exclusion_is_case_insensitive(self) -> None:
        """Sector matching must be case-insensitive."""
        df = _make_universe_df(["XYZ"], sector="financial services")
        result = filter_universe(df)
        assert len(result) == 0

    def test_partial_industry_keyword_match(self) -> None:
        """'Banks' should match 'Regional Banks' and 'Banks — Diversified'."""
        df = _make_universe_df(["USB"], sector="Finance", industry="Regional Banks")
        result = filter_universe(df)
        assert len(result) == 0

    def test_missing_sector_column_does_not_raise(self) -> None:
        """If sector column is absent, skip sector filter and continue."""
        df = _make_universe_df(["AAPL"])
        df = df.drop(columns=["sector"])
        result = filter_universe(df)  # should not raise
        assert "AAPL" in result["ticker"].values

    def test_missing_industry_column_does_not_raise(self) -> None:
        df = _make_universe_df(["AAPL"])
        df = df.drop(columns=["industry"])
        result = filter_universe(df)  # should not raise
        assert "AAPL" in result["ticker"].values

    def test_empty_df_returns_empty_df(self) -> None:
        result = filter_universe(_empty_universe_df())
        assert len(result) == 0
        assert set(result.columns) == set(UNIVERSE_COLUMNS)

    def test_index_reset_after_filtering(self) -> None:
        """Returned DataFrame must have a clean 0-based integer index."""
        df = pd.concat([
            _make_universe_df(["AAPL"]),
            _make_universe_df(["JPM"], sector="Financial Services"),
            _make_universe_df(["MSFT"]),
        ], ignore_index=True)
        result = filter_universe(df)
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# fetch_universe tests (mocked resilient_request)
# ---------------------------------------------------------------------------

class TestFetchUniverse:
    """fetch_universe assembles results across exchanges."""

    def _fmp_side_effect(self, exchange_rows: dict[str, list[dict]]):
        """Return a side_effect function that dispatches by exchange param."""
        def side_effect(url, params=None, rate_limiter=None, **kw):
            ex = (params or {}).get("exchange", "")
            return exchange_rows.get(ex, [])
        return side_effect

    @patch("data_acquisition.universe.resilient_request")
    @patch("data_acquisition.universe.get_config")
    def test_returns_dataframe_with_canonical_columns(
        self, mock_cfg, mock_req
    ) -> None:
        mock_cfg.return_value = {
            "universe": {"exchanges": ["NYSE"], "min_market_cap_usd": 500_000_000},
            "exclusions": {"sectors": []},
        }
        mock_req.return_value = _make_fmp_rows(["AAPL", "MSFT"])
        df = fetch_universe()
        assert set(df.columns) == set(UNIVERSE_COLUMNS)

    @patch("data_acquisition.universe.resilient_request")
    @patch("data_acquisition.universe.get_config")
    def test_each_exchange_queried_once(self, mock_cfg, mock_req) -> None:
        mock_cfg.return_value = {
            "universe": {
                "exchanges": ["NYSE", "NASDAQ"],
                "min_market_cap_usd": 500_000_000,
            },
            "exclusions": {"sectors": []},
        }
        mock_req.return_value = _make_fmp_rows(["A"])
        fetch_universe()
        assert mock_req.call_count == 2

    @patch("data_acquisition.universe.resilient_request")
    @patch("data_acquisition.universe.get_config")
    def test_results_from_multiple_exchanges_merged(
        self, mock_cfg, mock_req
    ) -> None:
        mock_cfg.return_value = {
            "universe": {
                "exchanges": ["NYSE", "NASDAQ"],
                "min_market_cap_usd": 500_000_000,
            },
            "exclusions": {"sectors": []},
        }
        mock_req.side_effect = self._fmp_side_effect({
            "NYSE": _make_fmp_rows(["KO", "JNJ"]),
            "NASDAQ": _make_fmp_rows(["AAPL", "MSFT"]),
        })
        df = fetch_universe()
        assert set(df["ticker"]) == {"KO", "JNJ", "AAPL", "MSFT"}

    @patch("data_acquisition.universe.resilient_request")
    @patch("data_acquisition.universe.get_config")
    def test_duplicate_tickers_deduplicated(self, mock_cfg, mock_req) -> None:
        """If the same ticker appears in two exchange results, keep only one."""
        mock_cfg.return_value = {
            "universe": {
                "exchanges": ["NYSE", "NASDAQ"],
                "min_market_cap_usd": 500_000_000,
            },
            "exclusions": {"sectors": []},
        }
        mock_req.return_value = _make_fmp_rows(["AAPL"])
        df = fetch_universe()
        assert df["ticker"].duplicated().sum() == 0

    @patch("data_acquisition.universe.resilient_request")
    @patch("data_acquisition.universe.get_config")
    def test_empty_exchange_returns_empty_and_logs_warning(
        self, mock_cfg, mock_req, caplog
    ) -> None:
        import logging
        mock_cfg.return_value = {
            "universe": {"exchanges": ["TSX"], "min_market_cap_usd": 500_000_000},
            "exclusions": {"sectors": []},
        }
        mock_req.return_value = []
        with caplog.at_level(logging.WARNING, logger="data_acquisition.universe"):
            df = fetch_universe()
        assert len(df) == 0
        assert any("0 tickers" in r.message for r in caplog.records)

    @patch("data_acquisition.universe.resilient_request")
    @patch("data_acquisition.universe.get_config")
    def test_exactly_limit_rows_triggers_warning(
        self, mock_cfg, mock_req, caplog
    ) -> None:
        import logging
        mock_cfg.return_value = {
            "universe": {"exchanges": ["NYSE"], "min_market_cap_usd": 500_000_000},
            "exclusions": {"sectors": []},
        }
        mock_req.return_value = _make_fmp_rows(
            [f"T{i}" for i in range(_FMP_SCREENER_LIMIT)]
        )
        with caplog.at_level(logging.WARNING, logger="data_acquisition.universe"):
            fetch_universe()
        assert any("truncated" in r.message for r in caplog.records)

    @patch("data_acquisition.universe.resilient_request")
    @patch("data_acquisition.universe.get_config")
    def test_api_error_returns_empty_df(self, mock_cfg, mock_req) -> None:
        mock_cfg.return_value = {
            "universe": {"exchanges": ["NYSE"], "min_market_cap_usd": 500_000_000},
            "exclusions": {"sectors": []},
        }
        mock_req.side_effect = Exception("connection refused")
        df = fetch_universe()
        assert len(df) == 0

    @patch("data_acquisition.universe.resilient_request")
    @patch("data_acquisition.universe.get_config")
    def test_min_cap_passed_to_api(self, mock_cfg, mock_req) -> None:
        """marketCapMoreThan must equal the config value."""
        mock_cfg.return_value = {
            "universe": {"exchanges": ["NYSE"], "min_market_cap_usd": 999_000_000},
            "exclusions": {"sectors": []},
        }
        mock_req.return_value = []
        fetch_universe()
        _, call_kwargs = mock_req.call_args
        params = call_kwargs.get("params", {})
        assert params.get("marketCapMoreThan") == 999_000_000


# ---------------------------------------------------------------------------
# Cache logic tests
# ---------------------------------------------------------------------------

class TestCacheIsFresh:
    """_cache_is_fresh checks file existence and age."""

    def test_returns_false_when_file_missing(self, tmp_path) -> None:
        assert _cache_is_fresh(tmp_path / "nonexistent.parquet") is False

    def test_returns_true_for_brand_new_file(self, tmp_path) -> None:
        cache = tmp_path / "universe.parquet"
        cache.write_bytes(b"dummy")
        assert _cache_is_fresh(cache) is True

    def test_returns_false_when_file_older_than_ttl(self, tmp_path) -> None:
        cache = tmp_path / "universe.parquet"
        cache.write_bytes(b"dummy")
        # Backdate the mtime by TTL + 1 second.
        stale_time = datetime.now(tz=timezone.utc).timestamp() - _CACHE_TTL_SECONDS - 1
        import os
        os.utime(cache, (stale_time, stale_time))
        assert _cache_is_fresh(cache) is False

    def test_returns_true_for_file_just_within_ttl(self, tmp_path) -> None:
        cache = tmp_path / "universe.parquet"
        cache.write_bytes(b"dummy")
        # Set mtime to 10 s before TTL expiry — still fresh.
        fresh_time = datetime.now(tz=timezone.utc).timestamp() - _CACHE_TTL_SECONDS + 10
        import os
        os.utime(cache, (fresh_time, fresh_time))
        assert _cache_is_fresh(cache) is True


class TestGetUniverseCacheHit:
    """get_universe returns cached data when cache is fresh."""

    @patch("data_acquisition.universe._cache_is_fresh", return_value=True)
    @patch("data_acquisition.universe._resolve_cache_path")
    @patch("data_acquisition.universe.pd.read_parquet")
    def test_reads_from_parquet_on_cache_hit(
        self, mock_read, mock_path, mock_fresh
    ) -> None:
        cached_df = _make_universe_df(["AAPL", "MSFT"])
        mock_read.return_value = cached_df
        mock_path.return_value = pathlib.Path("/fake/universe.parquet")

        result = get_universe(use_cache=True)
        mock_read.assert_called_once()
        assert len(result) == 2

    @patch("data_acquisition.universe._cache_is_fresh", return_value=True)
    @patch("data_acquisition.universe._resolve_cache_path")
    @patch("data_acquisition.universe.pd.read_parquet")
    @patch("data_acquisition.universe.fetch_universe")
    def test_fetch_not_called_on_cache_hit(
        self, mock_fetch, mock_read, mock_path, mock_fresh
    ) -> None:
        mock_read.return_value = _make_universe_df(["AAPL"])
        mock_path.return_value = pathlib.Path("/fake/universe.parquet")
        get_universe(use_cache=True)
        mock_fetch.assert_not_called()


class TestGetUniverseCacheMiss:
    """get_universe fetches fresh data on cache miss or use_cache=False."""

    @patch("data_acquisition.universe._save_cache")
    @patch("data_acquisition.universe.filter_universe")
    @patch("data_acquisition.universe.fetch_universe")
    @patch("data_acquisition.universe._cache_is_fresh", return_value=False)
    @patch("data_acquisition.universe._resolve_cache_path")
    def test_fetch_called_on_cache_miss(
        self, mock_path, mock_fresh, mock_fetch, mock_filter, mock_save
    ) -> None:
        mock_path.return_value = pathlib.Path("/fake/universe.parquet")
        raw_df = _make_universe_df(["AAPL"])
        mock_fetch.return_value = raw_df
        mock_filter.return_value = raw_df
        get_universe(use_cache=True)
        mock_fetch.assert_called_once()

    @patch("data_acquisition.universe._save_cache")
    @patch("data_acquisition.universe.filter_universe")
    @patch("data_acquisition.universe.fetch_universe")
    @patch("data_acquisition.universe._cache_is_fresh", return_value=True)
    @patch("data_acquisition.universe._resolve_cache_path")
    def test_fetch_called_when_use_cache_false(
        self, mock_path, mock_fresh, mock_fetch, mock_filter, mock_save
    ) -> None:
        """use_cache=False must always fetch, even if cache is fresh."""
        mock_path.return_value = pathlib.Path("/fake/universe.parquet")
        raw_df = _make_universe_df(["AAPL"])
        mock_fetch.return_value = raw_df
        mock_filter.return_value = raw_df
        get_universe(use_cache=False)
        mock_fetch.assert_called_once()

    @patch("data_acquisition.universe._save_cache")
    @patch("data_acquisition.universe.filter_universe")
    @patch("data_acquisition.universe.fetch_universe")
    @patch("data_acquisition.universe._cache_is_fresh", return_value=False)
    @patch("data_acquisition.universe._resolve_cache_path")
    def test_result_is_filtered(
        self, mock_path, mock_fresh, mock_fetch, mock_filter, mock_save
    ) -> None:
        mock_path.return_value = pathlib.Path("/fake/universe.parquet")
        raw_df = _make_universe_df(["AAPL", "JPM"])
        filtered_df = _make_universe_df(["AAPL"])
        mock_fetch.return_value = raw_df
        mock_filter.return_value = filtered_df
        result = get_universe(use_cache=False)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"

    @patch("data_acquisition.universe._save_cache")
    @patch("data_acquisition.universe.filter_universe")
    @patch("data_acquisition.universe.fetch_universe")
    @patch("data_acquisition.universe._cache_is_fresh", return_value=False)
    @patch("data_acquisition.universe._resolve_cache_path")
    def test_cache_saved_after_fresh_fetch(
        self, mock_path, mock_fresh, mock_fetch, mock_filter, mock_save
    ) -> None:
        mock_path.return_value = pathlib.Path("/fake/universe.parquet")
        df = _make_universe_df(["AAPL"])
        mock_fetch.return_value = df
        mock_filter.return_value = df
        get_universe(use_cache=False)
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# _empty_universe_df tests
# ---------------------------------------------------------------------------

class TestEmptyUniverseDf:
    def test_has_correct_columns(self) -> None:
        df = _empty_universe_df()
        assert set(df.columns) == set(UNIVERSE_COLUMNS)

    def test_has_zero_rows(self) -> None:
        assert len(_empty_universe_df()) == 0

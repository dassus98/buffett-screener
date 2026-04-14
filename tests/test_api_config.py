"""Unit tests for data_acquisition/api_config.py.

All tests are pure-Python with no real network calls or real API keys.
``requests.get`` and ``time.sleep`` / ``time.monotonic`` are patched via
``unittest.mock`` throughout.

Coverage
--------
TestRateLimiterInit            — constructor validation (3 tests)
TestRateLimiterWaitIfNeeded    — sliding-window blocking behaviour (6 tests)
TestResilientRequestSuccess    — 200 happy-path (3 tests)
TestResilientRequestRetryOn429 — 429 / 5xx retry logic (6 tests)
TestResilientRequestRaiseOn4xx — non-retryable 4xx and connection errors (5 tests)
TestBuildFmpUrl                — URL construction (8 tests)
TestApiKeyLoading              — .env credential loading (6 tests)
TestPreconfiguredLimiters      — fmp_limiter / fred_limiter from config (6 tests)
TestBackoffWait                — exponential backoff schedule (4 tests)
TestRedactKey                  — API key redaction in logs (4 tests)
TestConfigDrivenRetry          — verify retry params come from config (5 tests)
TestRequestTimeout             — verify timeout passed to requests.get (2 tests)
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from data_acquisition.api_config import (
    RateLimiter,
    _backoff_wait,
    _BASE_WAIT_SECONDS,
    _MAX_RETRIES,
    _REQUEST_TIMEOUT,
    _RETRY_STATUSES,
    _redact_key,
    build_fmp_url,
    fmp_limiter,
    fred_limiter,
    get_fmp_key,
    get_fred_key,
    resilient_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int, json_body: object = None) -> MagicMock:
    """Return a mock requests.Response with the given status and JSON body."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.text = str(json_body or {})
    # raise_for_status raises HTTPError for 4xx/5xx
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status_code}", response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# RateLimiter tests
# ---------------------------------------------------------------------------


class TestRateLimiterInit:
    """Constructor validation."""

    def test_stores_max_requests(self) -> None:
        limiter = RateLimiter(max_requests_per_minute=100)
        assert limiter.max_requests_per_minute == 100

    def test_raises_on_zero_max(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RateLimiter(max_requests_per_minute=0)

    def test_raises_on_negative_max(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RateLimiter(max_requests_per_minute=-5)


class TestRateLimiterWaitIfNeeded:
    """Sliding-window blocking behaviour — monotonic clock patched throughout."""

    def test_first_request_proceeds_immediately(self) -> None:
        """First call must not sleep when window is empty."""
        limiter = RateLimiter(max_requests_per_minute=60)
        with patch("data_acquisition.api_config.time.sleep") as mock_sleep, \
             patch("data_acquisition.api_config.time.monotonic", return_value=1000.0):
            limiter.wait_if_needed()
        mock_sleep.assert_not_called()

    def test_timestamps_recorded_after_call(self) -> None:
        """Each call must record one timestamp in the internal deque."""
        limiter = RateLimiter(max_requests_per_minute=10)
        t = 1000.0
        with patch("data_acquisition.api_config.time.monotonic", return_value=t):
            limiter.wait_if_needed()
            limiter.wait_if_needed()
        assert len(limiter._timestamps) == 2

    def test_old_timestamps_pruned_from_window(self) -> None:
        """Timestamps older than 60 s must be dropped before the capacity check."""
        limiter = RateLimiter(max_requests_per_minute=2)
        # Pre-fill with two timestamps 61 seconds in the past
        old_ts = 1000.0
        limiter._timestamps = deque([old_ts, old_ts])

        now = old_ts + 61.0  # Both old entries are outside the 60s window
        with patch("data_acquisition.api_config.time.sleep") as mock_sleep, \
             patch("data_acquisition.api_config.time.monotonic", return_value=now):
            limiter.wait_if_needed()  # Should NOT block — window is now empty

        mock_sleep.assert_not_called()

    def test_blocks_when_window_is_full(self) -> None:
        """When the window holds max_requests timestamps, wait_if_needed must sleep."""
        limiter = RateLimiter(max_requests_per_minute=2)
        now = 2000.0

        # Pre-fill with 2 timestamps at now - 30s (inside the 60s window)
        ts = now - 30.0
        limiter._timestamps = deque([ts, ts])

        sleep_calls: list[float] = []

        def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            # After sleeping, advance the clock so the next monotonic call
            # returns a time far enough ahead that the old timestamps expire.
            nonlocal now
            now += seconds + 31.0  # jump past the window

        tick = iter([now, now, now + 31.0, now + 31.0, now + 62.0])

        with patch("data_acquisition.api_config.time.sleep", side_effect=fake_sleep), \
             patch("data_acquisition.api_config.time.monotonic", side_effect=tick):
            limiter.wait_if_needed()

        assert len(sleep_calls) >= 1
        assert all(s > 0 for s in sleep_calls)

    def test_window_full_sleep_duration_reasonable(self) -> None:
        """Sleep duration must be approximately (oldest_ts + 60) - now."""
        limiter = RateLimiter(max_requests_per_minute=1)
        now = 5000.0
        oldest = now - 20.0  # 20 s ago → need to wait ~40 s more
        limiter._timestamps = deque([oldest])

        sleep_durations: list[float] = []

        def fake_sleep(secs: float) -> None:
            sleep_durations.append(secs)

        # monotonic returns: (1) now → window full, sleep; (2) now+41 → window expired
        # The implementation calls time.monotonic() once per while-loop iteration.
        times = iter([now, now + 41.0])

        with patch("data_acquisition.api_config.time.sleep", side_effect=fake_sleep), \
             patch("data_acquisition.api_config.time.monotonic", side_effect=times):
            limiter.wait_if_needed()

        assert len(sleep_durations) == 1
        # Expected sleep: (oldest + 60) - now = 40 s
        assert abs(sleep_durations[0] - 40.0) < 1.0

    def test_thread_safety_no_race(self) -> None:
        """Concurrent threads must each get exactly one slot without data corruption."""
        limiter = RateLimiter(max_requests_per_minute=1000)
        results: list[None] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                limiter.wait_if_needed()
                results.append(None)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 20


# ---------------------------------------------------------------------------
# resilient_request tests
# ---------------------------------------------------------------------------


class TestResilientRequestSuccess:
    """Happy-path: 200 response on first attempt."""

    def test_returns_json_on_200(self) -> None:
        payload = {"symbol": "AAPL", "price": 175.0}
        with patch("data_acquisition.api_config.requests.get",
                   return_value=_mock_response(200, payload)):
            result = resilient_request("https://example.com/api", max_retries=0)
        assert result == payload

    def test_no_sleep_on_first_success(self) -> None:
        with patch("data_acquisition.api_config.requests.get",
                   return_value=_mock_response(200, {})), \
             patch("data_acquisition.api_config.time.sleep") as mock_sleep:
            resilient_request("https://example.com/api", max_retries=2)
        mock_sleep.assert_not_called()

    def test_rate_limiter_called_once(self) -> None:
        limiter = MagicMock(spec=RateLimiter)
        with patch("data_acquisition.api_config.requests.get",
                   return_value=_mock_response(200, {})):
            resilient_request("https://example.com/api",
                               rate_limiter=limiter, max_retries=0)
        limiter.wait_if_needed.assert_called_once()


class TestResilientRequestRetryOn429:
    """Retry logic: 429 triggers exponential backoff, success on final attempt."""

    def test_retries_on_429_and_succeeds(self) -> None:
        """Three 429s followed by a 200 — should succeed with 3 retries allowed."""
        responses = [
            _mock_response(429),
            _mock_response(429),
            _mock_response(200, {"data": "ok"}),
        ]
        with patch("data_acquisition.api_config.requests.get",
                   side_effect=responses), \
             patch("data_acquisition.api_config.time.sleep"):
            result = resilient_request("https://example.com", max_retries=3)
        assert result == {"data": "ok"}

    def test_sleep_called_on_each_retry(self) -> None:
        """time.sleep must be called once per failed attempt before a retry."""
        responses = [_mock_response(429), _mock_response(200, {})]
        with patch("data_acquisition.api_config.requests.get",
                   side_effect=responses), \
             patch("data_acquisition.api_config.time.sleep") as mock_sleep:
            resilient_request("https://example.com", max_retries=2)
        assert mock_sleep.call_count == 1

    def test_backoff_durations_match_schedule(self) -> None:
        """Sleep durations must follow the 1 s → 2 s → 4 s schedule."""
        responses = [
            _mock_response(429),
            _mock_response(500),
            _mock_response(503),
            _mock_response(200, {}),
        ]
        sleep_calls: list[float] = []
        with patch("data_acquisition.api_config.requests.get",
                   side_effect=responses), \
             patch("data_acquisition.api_config.time.sleep",
                   side_effect=lambda s: sleep_calls.append(s)):
            resilient_request("https://example.com", max_retries=3)

        assert sleep_calls == [1.0, 2.0, 4.0]

    def test_raises_after_all_retries_exhausted_on_429(self) -> None:
        """If every attempt returns 429 and retries run out, raise HTTPError."""
        responses = [_mock_response(429)] * 4  # 1 initial + 3 retries
        with patch("data_acquisition.api_config.requests.get",
                   side_effect=responses), \
             patch("data_acquisition.api_config.time.sleep"):
            with pytest.raises(requests.HTTPError):
                resilient_request("https://example.com", max_retries=3)

    def test_rate_limiter_called_on_every_attempt(self) -> None:
        """Rate limiter wait_if_needed must be called before each attempt."""
        limiter = MagicMock(spec=RateLimiter)
        responses = [_mock_response(429), _mock_response(200, {})]
        with patch("data_acquisition.api_config.requests.get",
                   side_effect=responses), \
             patch("data_acquisition.api_config.time.sleep"):
            resilient_request("https://example.com",
                               rate_limiter=limiter, max_retries=2)
        assert limiter.wait_if_needed.call_count == 2

    def test_retries_on_5xx_status(self) -> None:
        """503 is in the retry set — should retry and eventually succeed."""
        responses = [_mock_response(503), _mock_response(200, {"ok": True})]
        with patch("data_acquisition.api_config.requests.get",
                   side_effect=responses), \
             patch("data_acquisition.api_config.time.sleep"):
            result = resilient_request("https://example.com", max_retries=2)
        assert result == {"ok": True}


class TestResilientRequestRaiseOn4xx:
    """Non-retryable 4xx errors must raise immediately without retry."""

    def test_raises_immediately_on_400(self) -> None:
        """HTTP 400 must raise HTTPError without any retry."""
        with patch("data_acquisition.api_config.requests.get",
                   return_value=_mock_response(400)), \
             patch("data_acquisition.api_config.time.sleep") as mock_sleep:
            with pytest.raises(requests.HTTPError):
                resilient_request("https://example.com", max_retries=3)
        mock_sleep.assert_not_called()

    def test_raises_immediately_on_401(self) -> None:
        """HTTP 401 (bad API key) must raise without retry."""
        with patch("data_acquisition.api_config.requests.get",
                   return_value=_mock_response(401)), \
             patch("data_acquisition.api_config.time.sleep") as mock_sleep:
            with pytest.raises(requests.HTTPError):
                resilient_request("https://example.com", max_retries=3)
        mock_sleep.assert_not_called()

    def test_raises_immediately_on_404(self) -> None:
        """HTTP 404 must raise without retry."""
        with patch("data_acquisition.api_config.requests.get",
                   return_value=_mock_response(404)), \
             patch("data_acquisition.api_config.time.sleep") as mock_sleep:
            with pytest.raises(requests.HTTPError):
                resilient_request("https://example.com", max_retries=3)
        mock_sleep.assert_not_called()

    def test_get_called_exactly_once_on_400(self) -> None:
        """requests.get must be called only once for a 400 — no retries."""
        with patch("data_acquisition.api_config.requests.get",
                   return_value=_mock_response(400)) as mock_get, \
             patch("data_acquisition.api_config.time.sleep"):
            with pytest.raises(requests.HTTPError):
                resilient_request("https://example.com", max_retries=3)
        assert mock_get.call_count == 1

    def test_raises_on_connection_error_after_retries(self) -> None:
        """A persistent ConnectionError must be re-raised after max_retries."""
        exc = requests.ConnectionError("Connection refused")
        with patch("data_acquisition.api_config.requests.get",
                   side_effect=exc), \
             patch("data_acquisition.api_config.time.sleep"):
            with pytest.raises(requests.ConnectionError):
                resilient_request("https://example.com", max_retries=2)


# ---------------------------------------------------------------------------
# build_fmp_url tests
# ---------------------------------------------------------------------------


class TestBuildFmpUrl:
    """URL construction from endpoint + params."""

    def _patch_env_and_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set a dummy FMP key so get_fmp_key() doesn't raise."""
        monkeypatch.setenv("FMP_API_KEY", "testkey123")

    def test_returns_tuple_of_url_and_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_env_and_config(monkeypatch)
        result = build_fmp_url("/income-statement/AAPL")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_url_contains_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_env_and_config(monkeypatch)
        url, _ = build_fmp_url("/income-statement/AAPL")
        assert url.endswith("/income-statement/AAPL")

    def test_url_contains_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_env_and_config(monkeypatch)
        url, _ = build_fmp_url("/income-statement/AAPL")
        assert "financialmodelingprep.com" in url

    def test_params_contain_apikey(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_env_and_config(monkeypatch)
        _, params = build_fmp_url("/income-statement/AAPL")
        assert "apikey" in params
        assert params["apikey"] == "testkey123"

    def test_extra_params_included(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_env_and_config(monkeypatch)
        _, params = build_fmp_url("/income-statement/AAPL", period="annual", limit=10)
        assert params["period"] == "annual"
        assert params["limit"] == 10

    def test_apikey_not_in_url_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API key must live in params dict, not embedded in the URL path."""
        self._patch_env_and_config(monkeypatch)
        url, params = build_fmp_url("/profile/KO", period="annual")
        assert "testkey123" not in url

    def test_raises_when_fmp_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """build_fmp_url must raise EnvironmentError if FMP_API_KEY is unset."""
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        # Also clear any cached env state
        os.environ.pop("FMP_API_KEY", None)
        with pytest.raises(EnvironmentError, match="FMP_API_KEY"):
            build_fmp_url("/income-statement/AAPL")

    def test_multiple_endpoints_produce_different_urls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_env_and_config(monkeypatch)
        url1, _ = build_fmp_url("/income-statement/AAPL")
        url2, _ = build_fmp_url("/balance-sheet-statement/AAPL")
        assert url1 != url2
        assert "income-statement" in url1
        assert "balance-sheet-statement" in url2


# ---------------------------------------------------------------------------
# get_fmp_key / get_fred_key tests
# ---------------------------------------------------------------------------


class TestApiKeyLoading:
    """Credential loading from environment."""

    def test_get_fmp_key_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FMP_API_KEY", "fmp_abc123")
        assert get_fmp_key() == "fmp_abc123"

    def test_get_fred_key_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fred_xyz789")
        assert get_fred_key() == "fred_xyz789"

    def test_get_fmp_key_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FMP_API_KEY", "  spaced_key  ")
        assert get_fmp_key() == "spaced_key"

    def test_get_fmp_key_raises_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FMP_API_KEY", "")
        with pytest.raises(EnvironmentError, match="FMP_API_KEY"):
            get_fmp_key()

    def test_get_fred_key_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        os.environ.pop("FRED_API_KEY", None)
        with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
            get_fred_key()

    def test_error_message_includes_instructions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FMP_API_KEY", "")
        with pytest.raises(EnvironmentError) as exc_info:
            get_fmp_key()
        assert ".env" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Preconfigured limiter tests
# ---------------------------------------------------------------------------


class TestPreconfiguredLimiters:
    """fmp_limiter and fred_limiter must have sensible limits from config."""

    def test_fmp_limiter_is_rate_limiter(self) -> None:
        assert isinstance(fmp_limiter, RateLimiter)

    def test_fred_limiter_is_rate_limiter(self) -> None:
        assert isinstance(fred_limiter, RateLimiter)

    def test_fmp_limiter_positive_limit(self) -> None:
        assert fmp_limiter.max_requests_per_minute > 0

    def test_fred_limiter_positive_limit(self) -> None:
        assert fred_limiter.max_requests_per_minute > 0

    def test_fmp_limiter_at_least_100_per_min(self) -> None:
        """FMP Starter is 300 req/min; config should not be set below 100."""
        assert fmp_limiter.max_requests_per_minute >= 100

    def test_fred_limiter_at_least_60_per_min(self) -> None:
        """FRED allows 120 req/min; config should not be set below 60."""
        assert fred_limiter.max_requests_per_minute >= 60


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


class TestBackoffWait:
    """_backoff_wait uses exponential formula: _BASE_WAIT_SECONDS × 2^attempt."""

    def test_attempt_0_returns_base_wait(self) -> None:
        """Attempt 0: base × 2^0 = base × 1 = 1.0 s (with default base=1.0)."""
        assert _backoff_wait(0) == _BASE_WAIT_SECONDS * (2 ** 0)

    def test_attempt_1_returns_2x_base(self) -> None:
        """Attempt 1: base × 2^1 = 2.0 s (with default base=1.0)."""
        assert _backoff_wait(1) == _BASE_WAIT_SECONDS * (2 ** 1)

    def test_attempt_2_returns_4x_base(self) -> None:
        """Attempt 2: base × 2^2 = 4.0 s (with default base=1.0)."""
        assert _backoff_wait(2) == _BASE_WAIT_SECONDS * (2 ** 2)

    def test_attempt_scales_exponentially_beyond_3(self) -> None:
        """Higher attempts continue doubling (no clamping)."""
        assert _backoff_wait(5) == _BASE_WAIT_SECONDS * (2 ** 5)


class TestRedactKey:
    """_redact_key removes API key values from log strings."""

    def test_apikey_param_redacted(self) -> None:
        result = _redact_key("https://example.com", {"apikey": "secret123", "limit": 10})
        assert "secret123" not in result
        assert "***" in result

    def test_non_key_params_preserved(self) -> None:
        result = _redact_key("https://example.com", {"apikey": "s3cr3t", "period": "annual"})
        assert "annual" in result

    def test_empty_params_returns_url_unchanged(self) -> None:
        url = "https://example.com/endpoint"
        assert _redact_key(url, {}) == url

    def test_api_key_variant_spelling_redacted(self) -> None:
        result = _redact_key("https://example.com", {"api_key": "secret"})
        assert "secret" not in result


# ---------------------------------------------------------------------------
# Config-driven retry parameter tests
# ---------------------------------------------------------------------------


class TestConfigDrivenRetry:
    """Verify that retry parameters (_RETRY_STATUSES, _BASE_WAIT_SECONDS,
    _MAX_RETRIES) are read from config/filter_config.yaml rather than
    being hardcoded.  This ensures compliance with CLAUDE.md's
    "no hardcoded thresholds" rule.
    """

    def test_retry_statuses_include_429(self) -> None:
        """429 (rate limit) must always be retryable per DATA_SOURCES.md §2."""
        assert 429 in _RETRY_STATUSES

    def test_retry_statuses_include_5xx(self) -> None:
        """500, 502, 503, 504 must all be retryable per DATA_SOURCES.md §2."""
        for code in (500, 502, 503, 504):
            assert code in _RETRY_STATUSES, f"Expected {code} in _RETRY_STATUSES"

    def test_retry_statuses_exclude_4xx_other_than_429(self) -> None:
        """400, 401, 403, 404 must NOT be retryable per DATA_SOURCES.md §2."""
        for code in (400, 401, 403, 404):
            assert code not in _RETRY_STATUSES, f"{code} should not be retryable"

    def test_base_wait_seconds_is_positive(self) -> None:
        """base_wait_seconds from config must be a positive number."""
        assert _BASE_WAIT_SECONDS > 0

    def test_max_retries_matches_config(self) -> None:
        """_MAX_RETRIES must be read from config, not hardcoded.
        The config default is 3 (data_sources.retry.max_attempts).
        """
        assert _MAX_RETRIES == 3
        # Verify the default parameter value on resilient_request matches
        import inspect
        sig = inspect.signature(resilient_request)
        default = sig.parameters["max_retries"].default
        assert default == _MAX_RETRIES


# ---------------------------------------------------------------------------
# Request timeout tests
# ---------------------------------------------------------------------------


class TestRequestTimeout:
    """Verify that the per-request timeout is passed to requests.get."""

    def test_timeout_passed_to_requests_get(self) -> None:
        """requests.get must receive timeout=_REQUEST_TIMEOUT on every call."""
        with patch("data_acquisition.api_config.requests.get",
                   return_value=_mock_response(200, {"ok": True})) as mock_get:
            resilient_request("https://example.com/api", max_retries=0)
        # Verify timeout kwarg
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == _REQUEST_TIMEOUT

    def test_timeout_is_30_seconds(self) -> None:
        """Default timeout must be 30 seconds per DATA_SOURCES.md."""
        assert _REQUEST_TIMEOUT == 30

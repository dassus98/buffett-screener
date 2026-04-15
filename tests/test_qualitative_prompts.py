"""Tests for valuation_reports.qualitative_prompts module.

Covers:
  - generate_moat_assessment with mocked API response → correct parsing
  - generate_moat_assessment with missing API key → returns None
  - generate_moat_assessment with API timeout → returns None
  - generate_moat_assessment with non-200 status → returns None
  - generate_moat_assessment with malformed JSON → returns None
  - enrich_report_with_moat integration
  - _parse_response edge cases
  - _format_financial_summary with various metric values
  - MOAT_ASSESSMENT_PROMPT template validation
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from valuation_reports.qualitative_prompts import (
    MOAT_ASSESSMENT_PROMPT,
    _format_financial_summary,
    _format_industry_context,
    _parse_response,
    enrich_report_with_moat,
    generate_moat_assessment,
)


# ---------------------------------------------------------------------------
# Fixtures and test data
# ---------------------------------------------------------------------------


def _valid_moat_json() -> dict:
    """A valid moat assessment response dict."""
    return {
        "moat_type": "Brand",
        "evidence": (
            "Apple's brand commands premium pricing with gross margins "
            "consistently above 40%. Strong ecosystem creates high "
            "switching costs for consumers."
        ),
        "threats": [
            "Regulatory unbundling of App Store",
            "AI commoditisation of hardware differentiation",
        ],
        "moat_rating": "Wide",
        "confidence": "High",
    }


def _valid_api_response() -> dict:
    """A valid Anthropic Messages API response body."""
    return {
        "id": "msg_test_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": json.dumps(_valid_moat_json())}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 500, "output_tokens": 200},
    }


def _sample_metrics() -> dict:
    return {
        "avg_roe_10yr": 0.285,
        "gross_margin_avg_10yr": 0.432,
        "eps_cagr_10yr": 0.123,
        "de_ratio_latest": 0.45,
        "debt_payoff_years": 2.1,
        "margin_of_safety": 0.233,
    }


def _mock_response(status_code: int = 200, json_body: dict | None = None) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    if json_body is not None:
        resp.json.return_value = json_body
    resp.text = json.dumps(json_body) if json_body else ""
    return resp


# ---------------------------------------------------------------------------
# Tests: _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Test the JSON response parser."""

    def test_valid_json(self):
        text = json.dumps(_valid_moat_json())
        result = _parse_response(text)
        assert result is not None
        assert result["moat_type"] == "Brand"
        assert result["moat_rating"] == "Wide"
        assert result["confidence"] == "High"
        assert len(result["threats"]) == 2

    def test_json_with_code_fences(self):
        """Parser strips markdown code fences."""
        text = "```json\n" + json.dumps(_valid_moat_json()) + "\n```"
        result = _parse_response(text)
        assert result is not None
        assert result["moat_type"] == "Brand"

    def test_json_with_bare_fences(self):
        text = "```\n" + json.dumps(_valid_moat_json()) + "\n```"
        result = _parse_response(text)
        assert result is not None

    def test_invalid_json(self):
        result = _parse_response("This is not JSON at all.")
        assert result is None

    def test_missing_required_key(self):
        incomplete = {"moat_type": "Brand", "evidence": "Strong brand."}
        result = _parse_response(json.dumps(incomplete))
        assert result is None

    def test_threats_as_string(self):
        """If threats is a single string instead of list, normalise it."""
        data = _valid_moat_json()
        data["threats"] = "Single threat"
        result = _parse_response(json.dumps(data))
        assert result is not None
        assert result["threats"] == ["Single threat"]

    def test_empty_string(self):
        result = _parse_response("")
        assert result is None

    def test_whitespace_only(self):
        result = _parse_response("   \n\n  ")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _format_financial_summary
# ---------------------------------------------------------------------------


class TestFormatFinancialSummary:
    """Test financial summary formatting for the prompt."""

    def test_all_metrics_present(self):
        text = _format_financial_summary(_sample_metrics())
        assert "28.5%" in text  # ROE
        assert "43.2%" in text  # Gross margin
        assert "12.3%" in text  # EPS CAGR
        assert "0.45" in text   # D/E
        assert "2.1 years" in text  # Debt payoff

    def test_nan_metrics_show_na(self):
        text = _format_financial_summary({"avg_roe_10yr": float("nan")})
        assert "N/A" in text

    def test_missing_metrics_show_na(self):
        text = _format_financial_summary({})
        assert text.count("N/A") >= 5  # All 6 metrics missing (one may be 0)

    def test_returns_string(self):
        result = _format_financial_summary(_sample_metrics())
        assert isinstance(result, str)


class TestFormatIndustryContext:
    """Test industry context formatting."""

    def test_both_present(self):
        text = _format_industry_context("Technology", "Consumer Electronics")
        assert "Technology" in text
        assert "Consumer Electronics" in text

    def test_empty_strings(self):
        text = _format_industry_context("", "")
        assert "not available" in text


# ---------------------------------------------------------------------------
# Tests: MOAT_ASSESSMENT_PROMPT
# ---------------------------------------------------------------------------


class TestMoatPrompt:
    """Validate the prompt template."""

    def test_prompt_is_nonempty(self):
        assert len(MOAT_ASSESSMENT_PROMPT) > 100

    def test_prompt_has_format_variables(self):
        assert "{company_name}" in MOAT_ASSESSMENT_PROMPT
        assert "{ticker}" in MOAT_ASSESSMENT_PROMPT
        assert "{financial_summary}" in MOAT_ASSESSMENT_PROMPT
        assert "{industry_context}" in MOAT_ASSESSMENT_PROMPT

    def test_prompt_formats_without_error(self):
        """Formatting with all variables should not raise."""
        result = MOAT_ASSESSMENT_PROMPT.format(
            company_name="Apple Inc.",
            ticker="AAPL",
            financial_summary="- ROE: 28%",
            industry_context="- Sector: Technology",
        )
        assert "Apple Inc." in result
        assert "AAPL" in result

    def test_prompt_mentions_moat_types(self):
        assert "Brand" in MOAT_ASSESSMENT_PROMPT
        assert "Switching Costs" in MOAT_ASSESSMENT_PROMPT
        assert "Network Effect" in MOAT_ASSESSMENT_PROMPT

    def test_prompt_requests_json(self):
        assert "JSON" in MOAT_ASSESSMENT_PROMPT


# ---------------------------------------------------------------------------
# Tests: generate_moat_assessment
# ---------------------------------------------------------------------------


class TestGenerateMoatAssessment:
    """Test the API-calling function with various mock scenarios."""

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_successful_api_call(self, mock_key, mock_post):
        """Valid API response → parsed moat dict returned."""
        mock_key.return_value = "test-key-123"
        mock_post.return_value = _mock_response(200, _valid_api_response())

        result = generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        assert result is not None
        assert result["moat_type"] == "Brand"
        assert result["moat_rating"] == "Wide"
        assert result["confidence"] == "High"
        assert isinstance(result["threats"], list)
        assert len(result["threats"]) == 2

    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_missing_api_key_returns_none(self, mock_key):
        """No API key → None without making any HTTP call."""
        mock_key.return_value = None

        result = generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        assert result is None

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_api_timeout_returns_none(self, mock_key, mock_post):
        """Timeout exception → None without crashing."""
        mock_key.return_value = "test-key-123"
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        assert result is None

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_connection_error_returns_none(self, mock_key, mock_post):
        """Network error → None without crashing."""
        mock_key.return_value = "test-key-123"
        mock_post.side_effect = requests.exceptions.ConnectionError("DNS failed")

        result = generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        assert result is None

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_http_500_returns_none(self, mock_key, mock_post):
        """Server error (500) → None."""
        mock_key.return_value = "test-key-123"
        mock_post.return_value = _mock_response(500, {"error": "Internal"})

        result = generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        assert result is None

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_http_401_returns_none(self, mock_key, mock_post):
        """Auth error (401) → None."""
        mock_key.return_value = "bad-key"
        mock_post.return_value = _mock_response(401, {"error": "Unauthorized"})

        result = generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        assert result is None

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_malformed_json_response_returns_none(self, mock_key, mock_post):
        """API returns text that isn't valid JSON → None."""
        mock_key.return_value = "test-key-123"
        bad_resp = {
            "content": [{"type": "text", "text": "Not valid JSON at all!"}],
        }
        mock_post.return_value = _mock_response(200, bad_resp)

        result = generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        assert result is None

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_empty_content_returns_none(self, mock_key, mock_post):
        """API returns empty content array → None."""
        mock_key.return_value = "test-key-123"
        mock_post.return_value = _mock_response(200, {"content": []})

        result = generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        assert result is None

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_api_called_with_correct_headers(self, mock_key, mock_post):
        """Verify the API is called with correct auth and version headers."""
        mock_key.return_value = "sk-test-key"
        mock_post.return_value = _mock_response(200, _valid_api_response())

        generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["x-api-key"] == "sk-test-key"
        assert headers["anthropic-version"] == "2023-06-01"

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_api_called_with_correct_model(self, mock_key, mock_post):
        """Verify the API body uses the configured model."""
        mock_key.return_value = "sk-test-key"
        mock_post.return_value = _mock_response(200, _valid_api_response())

        generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["max_tokens"] == 1000

    @patch("valuation_reports.qualitative_prompts.requests.post")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    def test_timeout_passed_to_requests(self, mock_key, mock_post):
        """Verify the 30-second timeout is passed to requests.post."""
        mock_key.return_value = "sk-test-key"
        mock_post.return_value = _mock_response(200, _valid_api_response())

        generate_moat_assessment(
            "AAPL", "Apple Inc.", _sample_metrics(), "Technology", "Consumer Electronics",
        )

        call_kwargs = mock_post.call_args
        timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        assert timeout == 30


# ---------------------------------------------------------------------------
# Tests: enrich_report_with_moat
# ---------------------------------------------------------------------------


class TestEnrichReportWithMoat:
    """Test the report context enrichment integration."""

    def _base_context(self) -> dict:
        return {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "roe_avg_10yr": 0.285,
            "gross_margin_avg_10yr": 0.432,
            "eps_cagr_10yr": 0.123,
            "margin_of_safety_pct": 0.233,
            "qualitative_enabled": False,
            "moat_assessment": None,
        }

    @patch("valuation_reports.qualitative_prompts.get_threshold")
    def test_disabled_by_config(self, mock_thresh):
        """When reports.enable_qualitative is False → skip."""
        mock_thresh.return_value = False
        ctx = self._base_context()
        result = enrich_report_with_moat(ctx)
        assert result["qualitative_enabled"] is False
        assert result["moat_assessment"] is None

    @patch("valuation_reports.qualitative_prompts.generate_moat_assessment")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    @patch("valuation_reports.qualitative_prompts.get_threshold")
    def test_enabled_with_api_key_success(self, mock_thresh, mock_key, mock_gen):
        """Config enabled + API key + successful call → moat populated."""
        mock_thresh.return_value = True
        mock_key.return_value = "sk-test-key"
        mock_gen.return_value = _valid_moat_json()

        ctx = self._base_context()
        result = enrich_report_with_moat(ctx)

        assert result["qualitative_enabled"] is True
        assert result["moat_assessment"] is not None
        assert result["moat_assessment"]["moat_type"] == "Brand"

    @patch("valuation_reports.qualitative_prompts._get_api_key")
    @patch("valuation_reports.qualitative_prompts.get_threshold")
    def test_enabled_without_api_key(self, mock_thresh, mock_key):
        """Config enabled but no API key → graceful skip."""
        mock_thresh.return_value = True
        mock_key.return_value = None

        ctx = self._base_context()
        result = enrich_report_with_moat(ctx)

        assert result["qualitative_enabled"] is False
        assert result["moat_assessment"] is None

    @patch("valuation_reports.qualitative_prompts.generate_moat_assessment")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    @patch("valuation_reports.qualitative_prompts.get_threshold")
    def test_enabled_api_fails(self, mock_thresh, mock_key, mock_gen):
        """Config enabled + API key but call fails → graceful None."""
        mock_thresh.return_value = True
        mock_key.return_value = "sk-test-key"
        mock_gen.return_value = None  # API failure

        ctx = self._base_context()
        result = enrich_report_with_moat(ctx)

        assert result["qualitative_enabled"] is False
        assert result["moat_assessment"] is None

    @patch("valuation_reports.qualitative_prompts.get_threshold")
    def test_config_key_missing_treated_as_disabled(self, mock_thresh):
        """If config key doesn't exist, treat as disabled."""
        mock_thresh.side_effect = KeyError("reports.enable_qualitative")

        ctx = self._base_context()
        result = enrich_report_with_moat(ctx)

        assert result["qualitative_enabled"] is False
        assert result["moat_assessment"] is None

    @patch("valuation_reports.qualitative_prompts.generate_moat_assessment")
    @patch("valuation_reports.qualitative_prompts._get_api_key")
    @patch("valuation_reports.qualitative_prompts.get_threshold")
    def test_mutates_context_in_place(self, mock_thresh, mock_key, mock_gen):
        """enrich_report_with_moat mutates the dict in place."""
        mock_thresh.return_value = True
        mock_key.return_value = "sk-test-key"
        mock_gen.return_value = _valid_moat_json()

        ctx = self._base_context()
        returned = enrich_report_with_moat(ctx)

        # Should be the same object
        assert returned is ctx
        assert ctx["moat_assessment"]["moat_type"] == "Brand"

"""Optional LLM-assisted moat assessment via the Anthropic Messages API.

Generates structured competitive-moat assessments for shortlisted securities
by calling Claude with a purpose-built prompt.  Designed to **fail gracefully**
— if the API key is absent, the request times out, or the response cannot be
parsed, the function returns ``None`` and the report renders without this
section.

Data Lineage Contract
---------------------
Upstream producers:
    - ``valuation_reports.report_generator.build_report_context``
      → provides ``ticker``, ``company_name``, ``sector``, ``industry``,
        and the metrics summary used to build the financial context.
    - Environment variable ``ANTHROPIC_API_KEY``
      → loaded from ``.env`` via ``dotenv``.

Downstream consumers:
    - ``valuation_reports.report_generator``
      → sets ``report_context["moat_assessment"]`` to the returned dict
        (or ``None``).
    - ``valuation_reports/templates/deep_dive_template.md``
      → conditionally renders the moat section when
        ``qualitative_enabled and moat_assessment`` is truthy.

Config dependencies:
    - ``reports.enable_qualitative`` (bool, default ``false``)
      → checked by ``enrich_report_with_moat`` before making any API call.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
from typing import Any

import requests
from dotenv import load_dotenv

from screener.filter_config_loader import get_threshold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

_API_URL = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 1000
_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

MOAT_ASSESSMENT_PROMPT = """\
You are a senior equity analyst specialising in Warren Buffett's investment \
philosophy.  Assess the durable competitive advantage ("economic moat") of \
the company described below.

## Company
- **Name:** {company_name}
- **Ticker:** {ticker}

## Financial Summary (10-year averages)
{financial_summary}

## Industry Context
{industry_context}

## Your Task
Analyse the company's competitive position and respond with a JSON object \
(no markdown fences, no commentary outside the JSON) containing exactly \
these keys:

{{
  "moat_type": "<one of: Brand, Switching Costs, Network Effect, Cost \
Advantage, Efficient Scale, None>",
  "evidence": "<2–3 sentences describing the quantitative and qualitative \
evidence supporting (or refuting) a durable competitive advantage>",
  "threats": ["<threat 1>", "<threat 2>", ...],
  "moat_rating": "<one of: Wide, Narrow, None>",
  "confidence": "<one of: High, Moderate, Low>"
}}

Base your assessment on the financial data provided and your general \
knowledge of the industry.  Be specific and cite the metrics when relevant.\
"""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str | None:
    """Return the Anthropic API key, or ``None`` if not configured."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key if key else None


def _format_financial_summary(metrics: dict[str, Any]) -> str:
    """Format key metrics into a readable bullet list for the prompt."""
    lines: list[str] = []

    def _fmt(val: Any, fmt: str = ".1f", suffix: str = "%") -> str:
        try:
            v = float(val)
            if math.isnan(v):
                return "N/A"
            if suffix == "%":
                return f"{v * 100:{fmt}}{suffix}"
            return f"{v:{fmt}}{suffix}"
        except (TypeError, ValueError):
            return "N/A"

    lines.append(f"- Average ROE (10yr): {_fmt(metrics.get('avg_roe_10yr'))}")
    lines.append(
        f"- Average Gross Margin (10yr): "
        f"{_fmt(metrics.get('gross_margin_avg_10yr'))}"
    )
    lines.append(f"- EPS CAGR (10yr): {_fmt(metrics.get('eps_cagr_10yr'))}")
    lines.append(
        f"- D/E Ratio (latest): "
        f"{_fmt(metrics.get('de_ratio_latest'), '.2f', '')}"
    )
    lines.append(
        f"- Debt Payoff Years: "
        f"{_fmt(metrics.get('debt_payoff_years'), '.1f', ' years')}"
    )
    lines.append(
        f"- Margin of Safety: {_fmt(metrics.get('margin_of_safety'))}"
    )

    return "\n".join(lines)


def _format_industry_context(sector: str, industry: str) -> str:
    """Format sector and industry into the prompt context block."""
    parts: list[str] = []
    if sector:
        parts.append(f"- **Sector:** {sector}")
    if industry:
        parts.append(f"- **Industry:** {industry}")
    if not parts:
        parts.append("- Sector and industry information not available.")
    return "\n".join(parts)


def _parse_response(text: str) -> dict[str, Any] | None:
    """Parse the Claude response text into a structured moat dict.

    Expects a JSON object with keys: moat_type, evidence, threats,
    moat_rating, confidence.  Returns ``None`` on any parse failure.
    """
    # Strip markdown code fences if the model wraps the JSON
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (possibly ```json)
        first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_nl + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "Failed to parse moat assessment response as JSON: %.200s",
            text,
        )
        return None

    # Validate required keys
    required = {"moat_type", "evidence", "threats", "moat_rating", "confidence"}
    if not required.issubset(data.keys()):
        missing = required - set(data.keys())
        logger.warning(
            "Moat assessment response missing keys: %s", missing,
        )
        return None

    # Normalise threats to a list
    threats = data["threats"]
    if isinstance(threats, str):
        threats = [threats]
    elif not isinstance(threats, list):
        threats = []

    return {
        "moat_type": str(data["moat_type"]),
        "evidence": str(data["evidence"]),
        "threats": [str(t) for t in threats],
        "moat_rating": str(data["moat_rating"]),
        "confidence": str(data["confidence"]),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_moat_assessment(
    ticker: str,
    company_name: str,
    metrics_summary: dict[str, Any],
    sector: str,
    industry: str,
) -> dict[str, Any] | None:
    """Call the Anthropic API to generate a qualitative moat assessment.

    Parameters
    ----------
    ticker:
        Stock ticker symbol.
    company_name:
        Full company name.
    metrics_summary:
        Dict with keys such as ``avg_roe_10yr``, ``gross_margin_avg_10yr``,
        ``eps_cagr_10yr``, ``de_ratio_latest``, ``debt_payoff_years``,
        ``margin_of_safety``.
    sector:
        GICS sector name (e.g. ``"Technology"``).
    industry:
        GICS industry name (e.g. ``"Consumer Electronics"``).

    Returns
    -------
    dict | None
        Parsed moat assessment with keys ``moat_type``, ``evidence``,
        ``threats``, ``moat_rating``, ``confidence``.
        Returns ``None`` if the API key is missing, the call fails,
        or the response cannot be parsed.
    """
    # --- Step 1: Check for API key ---
    api_key = _get_api_key()
    if api_key is None:
        logger.info(
            "ANTHROPIC_API_KEY not set; skipping moat assessment for %s.",
            ticker,
        )
        return None

    # --- Step 2: Build the prompt ---
    financial_summary = _format_financial_summary(metrics_summary)
    industry_context = _format_industry_context(sector, industry)
    prompt = MOAT_ASSESSMENT_PROMPT.format(
        company_name=company_name,
        ticker=ticker,
        financial_summary=financial_summary,
        industry_context=industry_context,
    )

    # --- Step 3: Call the Anthropic Messages API ---
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": _MODEL,
        "max_tokens": _MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        resp = requests.post(
            _API_URL,
            headers=headers,
            json=body,
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        logger.warning(
            "Anthropic API request timed out for %s (timeout=%ds).",
            ticker,
            _TIMEOUT_SECONDS,
        )
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "Anthropic API request failed for %s: %s", ticker, exc,
        )
        return None

    # --- Step 4: Check HTTP status ---
    if resp.status_code != 200:
        logger.warning(
            "Anthropic API returned HTTP %d for %s: %.200s",
            resp.status_code,
            ticker,
            resp.text,
        )
        return None

    # --- Step 5: Extract content from the Messages API response ---
    try:
        resp_json = resp.json()
        content_blocks = resp_json.get("content", [])
        if not content_blocks:
            logger.warning(
                "Anthropic API returned empty content for %s.", ticker,
            )
            return None
        text = content_blocks[0].get("text", "")
    except (ValueError, KeyError, IndexError) as exc:
        logger.warning(
            "Failed to extract text from Anthropic API response for %s: %s",
            ticker,
            exc,
        )
        return None

    # --- Step 6: Parse the response ---
    result = _parse_response(text)
    if result is None:
        logger.warning(
            "Could not parse moat assessment for %s.", ticker,
        )
        return None

    logger.info(
        "%s moat assessment: type=%s, rating=%s, confidence=%s",
        ticker,
        result["moat_type"],
        result["moat_rating"],
        result["confidence"],
    )
    return result


def enrich_report_with_moat(report_context: dict[str, Any]) -> dict[str, Any]:
    """Conditionally enrich a report context dict with a moat assessment.

    Checks ``reports.enable_qualitative`` in config.  If enabled and the
    API key is available, calls :func:`generate_moat_assessment` and sets
    ``report_context["moat_assessment"]`` and
    ``report_context["qualitative_enabled"]``.

    Parameters
    ----------
    report_context:
        The template context dict produced by
        :func:`~valuation_reports.report_generator.build_report_context`.

    Returns
    -------
    dict
        The same dict, mutated in-place with moat assessment data
        (or ``None`` if unavailable).
    """
    # --- Step 1: Check config flag ---
    try:
        enabled = bool(get_threshold("reports.enable_qualitative"))
    except (KeyError, ValueError):
        enabled = False

    if not enabled:
        report_context["qualitative_enabled"] = False
        report_context["moat_assessment"] = None
        return report_context

    # --- Step 2: Check API key before attempting the call ---
    if _get_api_key() is None:
        logger.info(
            "Qualitative analysis enabled but ANTHROPIC_API_KEY not set. "
            "Skipping moat assessment.",
        )
        report_context["qualitative_enabled"] = False
        report_context["moat_assessment"] = None
        return report_context

    # --- Step 3: Build metrics summary for the prompt ---
    metrics = {
        "avg_roe_10yr": report_context.get("roe_avg_10yr"),
        "gross_margin_avg_10yr": report_context.get("gross_margin_avg_10yr"),
        "eps_cagr_10yr": report_context.get("eps_cagr_10yr"),
        "de_ratio_latest": report_context.get("de_ratio_latest", 0.0),
        "debt_payoff_years": report_context.get("debt_payoff_years", 0.0),
        "margin_of_safety": report_context.get("margin_of_safety_pct"),
    }

    # --- Step 4: Call the API ---
    result = generate_moat_assessment(
        ticker=report_context.get("ticker", ""),
        company_name=report_context.get("company_name", ""),
        metrics_summary=metrics,
        sector=report_context.get("sector", ""),
        industry=report_context.get("industry", ""),
    )

    # --- Step 5: Enrich the context ---
    if result is not None:
        report_context["qualitative_enabled"] = True
        report_context["moat_assessment"] = result
    else:
        report_context["qualitative_enabled"] = False
        report_context["moat_assessment"] = None

    return report_context

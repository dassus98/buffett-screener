"""
valuation_reports.qualitative_prompts
========================================
Generates structured qualitative analysis prompts for human review.

Buffett's investing framework cannot be fully quantified. The qualitative
layer — moat durability, management quality, industry dynamics — requires
human judgment. This module generates structured prompts that guide an analyst
(or an LLM) through the key qualitative questions for each passing stock.

The prompts are organised into the classic Buffett checklist:
    1. Business understanding    ("Can I understand this business?")
    2. Durable competitive moat  ("What keeps competitors out?")
    3. Management quality        ("Do they allocate capital rationally?")
    4. Long-term economics        ("Will this business be stronger in 10 years?")
    5. Price rationality          ("Am I paying a fair price?")

Output is a structured dict that can be rendered in Markdown or displayed
in the Streamlit UI as an analyst checklist.
"""

from __future__ import annotations

from data_acquisition.schema import CompanyProfile, TickerDataBundle


def generate_qualitative_prompts(
    bundle: TickerDataBundle,
    metrics: dict,
) -> dict[str, list[str]]:
    """
    Generate a structured set of qualitative analysis prompts for a stock.

    Args:
        bundle:  TickerDataBundle with company profile and financial history.
        metrics: Dict of computed metrics (output of compute_all_metrics()).

    Returns:
        Dict with category keys mapping to lists of prompt strings:
            {
                "business_understanding": [...],
                "competitive_moat": [...],
                "management_quality": [...],
                "long_term_economics": [...],
                "price_rationality": [...],
            }

    Logic:
        1. Build base prompts for each category (static questions)
        2. Add data-driven prompts based on metrics:
               If gross_margin_avg_5yr > 0.40: highlight pricing power prompt
               If capex_to_revenue_avg_5yr < 0.03: highlight asset-light prompt
               If revenue_cagr_5yr < 0.05: add growth quality probe
               If debt_to_equity_latest > 1.0: add leverage scrutiny prompt
        3. Return complete prompt dict
    """
    ...


def moat_type_prompts(sector: str, gross_margin: float) -> list[str]:
    """
    Generate moat-specific prompts based on sector and margin profile.

    Args:
        sector:       Company sector string (e.g. "Consumer Staples").
        gross_margin: 5-year average gross margin (decimal).

    Returns:
        List of prompt strings tailored to the most likely moat type:
            - Cost advantage    (commodity-like sectors with low margins)
            - Switching costs   (software, enterprise services)
            - Network effects   (payments, platforms, marketplaces)
            - Intangible assets (brands, patents, licenses)
            - Efficient scale   (regulated monopolies, niche infrastructure)

    Logic:
        Map sector + gross_margin threshold to likely moat type, then return
        a curated list of 3–5 targeted questions about that moat type.
    """
    ...


def management_quality_prompts(
    bundle: TickerDataBundle,
    metrics: dict,
) -> list[str]:
    """
    Generate management quality assessment prompts informed by financial data.

    Args:
        bundle:  TickerDataBundle (for profile and financial history).
        metrics: Computed metrics dict.

    Returns:
        List of management quality prompt strings.

    Logic:
        Base prompts:
            - "Has management maintained or grown ROIC over the last 10 years?"
            - "What has been the capital allocation track record? (dividends, buybacks, M&A)"
            - "Does management own significant stock? (skin in the game)"
            - "Has management been transparent about failures?"

        Data-driven additions:
            - If shares_diluted growth > 5% over 10 years: add dilution scrutiny prompt
            - If buybacks happened at high P/B: add capital allocation prompt
            - If acquisitions show up as goodwill growth: add M&A quality prompt
    """
    ...


def business_understanding_prompts(profile: CompanyProfile) -> list[str]:
    """
    Generate prompts to test whether the business is understandable.

    Args:
        profile: CompanyProfile with name, sector, industry, description.

    Returns:
        List of 3–5 prompt strings covering:
            - Revenue model clarity
            - Customer and pricing power
            - Regulatory and disruption risk
            - Competitive landscape overview
    """
    ...

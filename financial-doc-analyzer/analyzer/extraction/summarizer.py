"""Generates an executive summary from the merged, deduplicated KPI and risk data.

Deliberately grounded in the *extracted structured data* rather than the raw
document text -- this keeps the summary consistent with what was actually
pulled out (and keeps token usage low on long documents).
"""

from __future__ import annotations

import json
from typing import Optional

from analyzer.providers.base import BaseProvider
from analyzer.schemas import ExecutiveSummary, KPI, RiskFactor
from analyzer.utils import extract_json

SYSTEM_PROMPT = (
    "You are a senior financial analyst writing an executive summary of a banking report, "
    "based only on the structured KPI and risk data you're given. You respond with valid JSON only."
)

PROMPT_TEMPLATE = """Write an executive summary of a banking/financial report based on the following extracted data.

## Extracted KPIs
{kpis_json}

## Extracted risk factors
{risks_json}

Produce:
- overview: a 3-5 sentence narrative summary of the institution's financial position and key risks
- key_takeaways: a list of 3-6 short bullet-point takeaways (as plain strings)
- overall_risk_level: "low", "medium", "high", or "critical" -- your overall assessment given the severities and number of risks found, and the trend implied by the KPIs

Respond with ONLY a JSON object, no other text, in this exact format:
{{"overview": "...", "key_takeaways": ["...", "..."], "overall_risk_level": "..."}}
"""


def _kpi_to_summary_dict(kpi: KPI) -> dict:
    return {"name": kpi.name, "value": kpi.value, "period": kpi.period, "category": kpi.category}


def _risk_to_summary_dict(risk: RiskFactor) -> dict:
    return {"risk_type": risk.risk_type, "description": risk.description, "severity": risk.severity}


def generate_summary(
    kpis: list[KPI],
    risks: list[RiskFactor],
    provider: BaseProvider,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[Optional[ExecutiveSummary], Optional[str]]:
    """Returns (summary, error_message)."""
    kpis_json = json.dumps([_kpi_to_summary_dict(k) for k in kpis], indent=2)
    risks_json = json.dumps([_risk_to_summary_dict(r) for r in risks], indent=2)

    prompt = PROMPT_TEMPLATE.format(kpis_json=kpis_json, risks_json=risks_json)
    result = provider.generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens, system=SYSTEM_PROMPT)

    if not result.ok:
        return None, result.raw_error

    try:
        data = extract_json(result.text)
    except ValueError as exc:
        return None, str(exc)

    if not isinstance(data, dict):
        return None, "Expected a JSON object for the executive summary."

    try:
        return ExecutiveSummary.from_dict(data), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not parse executive summary: {exc}"

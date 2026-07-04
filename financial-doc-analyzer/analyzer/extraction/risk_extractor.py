"""Extracts structured risk factors from a chunk of document text via an LLM call."""

from __future__ import annotations

from typing import Optional

from analyzer.providers.base import BaseProvider
from analyzer.schemas import RiskFactor
from analyzer.utils import extract_json

SYSTEM_PROMPT = (
    "You are a meticulous risk analyst reviewing banking and financial reports for "
    "disclosed or implied risk factors. You ground every risk you report in the provided "
    "text and quote supporting evidence. You respond with valid JSON only."
)

PROMPT_TEMPLATE = """Identify risk factors discussed in the following excerpt of a banking or financial report. \
This includes explicitly labeled "risk factors" sections as well as risks implied by disclosed trends \
(e.g. rising delinquencies, concentration in a sector, declining capital ratios, litigation, regulatory action).

Guidance categories (use the closest fit, or "other" if none apply): {categories}

For each risk found, report:
- risk_type: one of the guidance categories above
- description: a concise (1-2 sentence) description of the risk, in your own words
- severity: "low", "medium", "high", or "critical", based on the language and figures used in the text
- evidence_quote: a short direct quote from the text supporting this risk (or null if paraphrased from data, e.g. a ratio)
- section: the report section this appeared in, if identifiable (or null)

Only include risks that are actually supported by content in the text below. If no risks are discussed, return an empty array.

Respond with ONLY a JSON array, no other text, in this exact format:
[{{"risk_type": "...", "description": "...", "severity": "...", "evidence_quote": "...", "section": "..."}}]

## Document excerpt
{text}
"""


def extract_risks(
    chunk_text: str,
    page_range: tuple[int, int],
    provider: BaseProvider,
    model: str,
    risk_categories: list[str],
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> tuple[list[RiskFactor], Optional[str]]:
    """Run risk extraction on a single chunk. Returns (risks, error_message)."""
    prompt = PROMPT_TEMPLATE.format(categories=", ".join(risk_categories), text=chunk_text)
    result = provider.generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens, system=SYSTEM_PROMPT)

    if not result.ok:
        return [], result.raw_error

    try:
        data = extract_json(result.text)
    except ValueError as exc:
        return [], str(exc)

    if isinstance(data, dict):
        data = data.get("risks", [])
    if not isinstance(data, list):
        return [], "Expected a JSON array of risk objects from the model."

    risks: list[RiskFactor] = []
    parse_errors: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            risk = RiskFactor.from_dict(item)
        except ValueError as exc:
            parse_errors.append(str(exc))
            continue
        if not risk.source_pages:
            risk.source_pages = list(range(page_range[0], page_range[1] + 1))
        risks.append(risk)

    error = "; ".join(parse_errors) if parse_errors else None
    return risks, error

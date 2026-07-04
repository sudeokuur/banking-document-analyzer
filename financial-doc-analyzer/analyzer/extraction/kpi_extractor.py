"""Extracts structured KPIs from a chunk of document text via an LLM call."""

from __future__ import annotations

from typing import Optional

from analyzer.providers.base import BaseProvider
from analyzer.schemas import KPI
from analyzer.utils import extract_json

SYSTEM_PROMPT = (
    "You are a meticulous financial analyst extracting structured data from banking and "
    "financial reports. You only report figures that are explicitly stated in the provided "
    "text. You never invent, estimate, or infer numbers that aren't present. You respond "
    "with valid JSON only."
)

PROMPT_TEMPLATE = """Extract every financial KPI/metric explicitly stated in the following excerpt of a banking or financial report.

Guidance categories (use the closest fit, or "other" if none apply): {categories}

For each KPI found, report:
- name: the metric's name as stated (e.g. "Return on Equity", "Net Interest Margin", "CET1 Ratio")
- value: the value as stated, including any symbol (e.g. "14.2%", "$45.3M")
- unit: the unit if separable from value (e.g. "%", "$M", "$B", "bps"), or null
- period: the reporting period if stated (e.g. "Q3 2025", "FY2024"), or null
- category: one of the guidance categories above
- confidence: your confidence this was extracted correctly, from 0.0 to 1.0

Only include KPIs that are explicitly present in the text below -- do not infer or calculate values that aren't directly stated. If no KPIs are present, return an empty array.

Respond with ONLY a JSON array, no other text, in this exact format:
[{{"name": "...", "value": "...", "unit": "...", "period": "...", "category": "...", "confidence": 0.9}}]

## Document excerpt
{text}
"""


def extract_kpis(
    chunk_text: str,
    page_range: tuple[int, int],
    provider: BaseProvider,
    model: str,
    kpi_categories: list[str],
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> tuple[list[KPI], Optional[str]]:
    """Run KPI extraction on a single chunk. Returns (kpis, error_message)."""
    prompt = PROMPT_TEMPLATE.format(categories=", ".join(kpi_categories), text=chunk_text)
    result = provider.generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens, system=SYSTEM_PROMPT)

    if not result.ok:
        return [], result.raw_error

    try:
        data = extract_json(result.text)
    except ValueError as exc:
        return [], str(exc)

    if isinstance(data, dict):
        data = data.get("kpis", [])
    if not isinstance(data, list):
        return [], "Expected a JSON array of KPI objects from the model."

    kpis: list[KPI] = []
    parse_errors: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            kpi = KPI.from_dict(item)
        except ValueError as exc:
            parse_errors.append(str(exc))
            continue
        if not kpi.source_pages:
            kpi.source_pages = list(range(page_range[0], page_range[1] + 1))
        kpis.append(kpi)

    error = "; ".join(parse_errors) if parse_errors else None
    return kpis, error

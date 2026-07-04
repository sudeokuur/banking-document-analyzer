"""Data model for extracted financial analysis: KPIs, risk factors, executive
summary, and the top-level AnalysisResult. Plain dataclasses (no pydantic
dependency) with explicit `from_dict`/`to_dict` so LLM JSON output is
defensively parsed rather than trusted as-is.
"""

from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from analyzer.utils import parse_numeric

VALID_SEVERITIES = ("low", "medium", "high", "critical")
_SEVERITY_RANK = {s: i for i, s in enumerate(VALID_SEVERITIES)}


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _clean_severity(value: Any) -> str:
    s = _clean_str(value, "medium").lower()
    return s if s in VALID_SEVERITIES else "medium"


@dataclass
class KPI:
    name: str
    value: Optional[str] = None
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    period: Optional[str] = None
    category: str = "other"
    confidence: float = 0.7
    source_pages: list[int] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "KPI":
        name = _clean_str(d.get("name"))
        if not name:
            raise ValueError(f"KPI missing required 'name' field: {d!r}")
        value = d.get("value")
        numeric_value = d.get("numeric_value")
        if numeric_value is None:
            numeric_value = parse_numeric(value)
        try:
            confidence = float(d.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        source_pages = d.get("source_pages") or []
        if isinstance(source_pages, int):
            source_pages = [source_pages]

        return KPI(
            name=name,
            value=_clean_str(value) or None,
            numeric_value=numeric_value,
            unit=_clean_str(d.get("unit")) or None,
            period=_clean_str(d.get("period")) or None,
            category=_clean_str(d.get("category"), "other").lower() or "other",
            confidence=confidence,
            source_pages=sorted(set(source_pages)),
        )


@dataclass
class RiskFactor:
    risk_type: str
    description: str
    severity: str = "medium"
    evidence_quote: Optional[str] = None
    section: Optional[str] = None
    source_pages: list[int] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "RiskFactor":
        risk_type = _clean_str(d.get("risk_type"), "other").lower() or "other"
        description = _clean_str(d.get("description"))
        if not description:
            raise ValueError(f"RiskFactor missing required 'description' field: {d!r}")

        source_pages = d.get("source_pages") or []
        if isinstance(source_pages, int):
            source_pages = [source_pages]

        return RiskFactor(
            risk_type=risk_type,
            description=description,
            severity=_clean_severity(d.get("severity")),
            evidence_quote=_clean_str(d.get("evidence_quote")) or None,
            section=_clean_str(d.get("section")) or None,
            source_pages=sorted(set(source_pages)),
        )


@dataclass
class ExecutiveSummary:
    overview: str
    key_takeaways: list[str] = field(default_factory=list)
    overall_risk_level: str = "medium"

    @staticmethod
    def from_dict(d: dict) -> "ExecutiveSummary":
        takeaways = d.get("key_takeaways") or []
        if isinstance(takeaways, str):
            takeaways = [takeaways]
        return ExecutiveSummary(
            overview=_clean_str(d.get("overview")),
            key_takeaways=[_clean_str(t) for t in takeaways if _clean_str(t)],
            overall_risk_level=_clean_severity(d.get("overall_risk_level")),
        )


@dataclass
class AnalysisResult:
    document: str
    generated_at: str
    provider: str
    model: str
    kpis: list[KPI]
    risks: list[RiskFactor]
    summary: ExecutiveSummary
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Merging / deduplication across chunks
# --------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _token_set(text: str) -> set[str]:
    return {tok for tok in _normalize(text).split() if len(tok) > 2}


def merge_kpis(kpis: list[KPI]) -> list[KPI]:
    """Deduplicate KPIs extracted from multiple chunks.

    Groups by normalized (name, period); within a group, keeps the
    highest-confidence extraction and unions the source pages.
    """
    groups: dict[tuple[str, str], list[KPI]] = {}
    for kpi in kpis:
        key = (_normalize(kpi.name), _normalize(kpi.period or ""))
        groups.setdefault(key, []).append(kpi)

    merged: list[KPI] = []
    for group in groups.values():
        best = max(group, key=lambda k: k.confidence)
        all_pages = sorted({p for k in group for p in k.source_pages})
        merged.append(
            KPI(
                name=best.name,
                value=best.value,
                numeric_value=best.numeric_value,
                unit=best.unit,
                period=best.period,
                category=best.category,
                confidence=best.confidence,
                source_pages=all_pages,
            )
        )

    merged.sort(key=lambda k: (k.category, k.name.lower()))
    return merged


def _risks_similar(a: RiskFactor, b: RiskFactor, threshold: float = 0.5) -> bool:
    if _normalize(a.risk_type) != _normalize(b.risk_type):
        return False
    tokens_a, tokens_b = _token_set(a.description), _token_set(b.description)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))
    return overlap >= threshold


def merge_risks(risks: list[RiskFactor]) -> list[RiskFactor]:
    """Deduplicate risk factors extracted from multiple chunks.

    Groups risks with the same normalized risk_type and sufficiently
    overlapping descriptions; keeps the higher severity and the longer
    (more detailed) description, and unions source pages.
    """
    merged: list[RiskFactor] = []
    for risk in risks:
        match = next((m for m in merged if _risks_similar(risk, m)), None)
        if match is None:
            merged.append(
                RiskFactor(
                    risk_type=risk.risk_type,
                    description=risk.description,
                    severity=risk.severity,
                    evidence_quote=risk.evidence_quote,
                    section=risk.section,
                    source_pages=list(risk.source_pages),
                )
            )
            continue

        if _SEVERITY_RANK[risk.severity] > _SEVERITY_RANK[match.severity]:
            match.severity = risk.severity
        if len(risk.description) > len(match.description):
            match.description = risk.description
        if not match.evidence_quote and risk.evidence_quote:
            match.evidence_quote = risk.evidence_quote
        match.source_pages = sorted(set(match.source_pages) | set(risk.source_pages))

    merged.sort(key=lambda r: (-_SEVERITY_RANK[r.severity], r.risk_type))
    return merged

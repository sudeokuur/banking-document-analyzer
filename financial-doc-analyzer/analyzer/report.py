"""Writes an AnalysisResult out to analysis.json and a formatted analysis.md."""

from __future__ import annotations

import json
from pathlib import Path

from tabulate import tabulate

from analyzer.schemas import AnalysisResult

_SEVERITY_ORDER = ("critical", "high", "medium", "low")
_SEVERITY_LABEL = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}


def write_json(result: AnalysisResult, output_dir: Path) -> Path:
    path = output_dir / "analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def _kpi_table(result: AnalysisResult) -> str:
    if not result.kpis:
        return "_No KPIs were extracted._\n"
    rows = [
        [k.category, k.name, k.value or "-", k.period or "-", f"{k.confidence:.2f}", ", ".join(map(str, k.source_pages)) or "-"]
        for k in result.kpis
    ]
    return tabulate(rows, headers=["Category", "Metric", "Value", "Period", "Confidence", "Page(s)"], tablefmt="github")


def _risks_section(result: AnalysisResult) -> str:
    if not result.risks:
        return "_No risk factors were identified._\n"

    by_severity: dict[str, list] = {s: [] for s in _SEVERITY_ORDER}
    for risk in result.risks:
        by_severity.setdefault(risk.severity, []).append(risk)

    parts = []
    for severity in _SEVERITY_ORDER:
        risks = by_severity.get(severity, [])
        if not risks:
            continue
        parts.append(f"\n### {_SEVERITY_LABEL[severity]} severity\n")
        for r in risks:
            quote = f' > "{r.evidence_quote}"' if r.evidence_quote else ""
            section = f" (section: {r.section})" if r.section else ""
            pages = ", ".join(map(str, r.source_pages)) or "-"
            parts.append(f"- **{r.risk_type}**{section} — {r.description} [pages {pages}]{quote}")
    return "\n".join(parts) + "\n"


def write_markdown(result: AnalysisResult, output_dir: Path) -> Path:
    path = output_dir / "analysis.md"
    meta = result.metadata

    lines = [
        f"# Financial Document Analysis: {Path(result.document).name}",
        "",
        f"*Generated {result.generated_at} using {result.provider}/{result.model}*",
        "",
        "## Executive Summary",
        "",
        result.summary.overview,
        "",
        f"**Overall risk level: {_SEVERITY_LABEL.get(result.summary.overall_risk_level, result.summary.overall_risk_level.title())}**",
        "",
    ]

    if result.summary.key_takeaways:
        lines.append("### Key takeaways\n")
        lines.extend(f"- {t}" for t in result.summary.key_takeaways)
        lines.append("")

    lines.append("## Extracted KPIs\n")
    lines.append(_kpi_table(result))
    lines.append("")

    lines.append("## Risk Factors")
    lines.append(_risks_section(result))

    lines.append("## Document metadata\n")
    lines.append(f"- Pages: {meta.get('total_pages')}")
    if meta.get("ocr_pages"):
        lines.append(f"- Pages requiring OCR: {meta['ocr_pages']}")
    if meta.get("empty_pages"):
        lines.append(f"- Pages with no extractable text: {meta['empty_pages']}")
    lines.append(f"- Chunks processed: {meta.get('total_chunks')}")
    lines.append(f"- Raw KPI extractions before dedup: {meta.get('raw_kpi_count')} -> {len(result.kpis)} after merge")
    lines.append(f"- Raw risk extractions before dedup: {meta.get('raw_risk_count')} -> {len(result.risks)} after merge")

    if meta.get("errors"):
        lines.append("\n### Warnings\n")
        lines.extend(f"- {e}" for e in meta["errors"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def generate_reports(result: AnalysisResult, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": write_json(result, output_dir),
        "markdown": write_markdown(result, output_dir),
    }

"""Orchestrates the full pipeline: ingest PDF -> chunk -> extract KPIs/risks
per chunk (in parallel) -> merge/deduplicate -> summarize -> AnalysisResult.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from analyzer.extraction import extract_kpis, extract_risks, generate_summary
from analyzer.ingestion import Chunk, PageResult, chunk_pages, extract_pdf
from analyzer.providers import get_provider
from analyzer.schemas import AnalysisResult, ExecutiveSummary, KPI, RiskFactor, merge_kpis, merge_risks

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _derive_fallback_risk_level(risks: list[RiskFactor]) -> str:
    if not risks:
        return "low"
    worst = max(risks, key=lambda r: _SEVERITY_RANK.get(r.severity, 1))
    return worst.severity


def _process_chunk(
    chunk: Chunk,
    provider,
    model: str,
    temperature: float,
    max_tokens: int,
    kpi_categories: list[str],
    risk_categories: list[str],
) -> dict:
    page_range = (chunk.start_page, chunk.end_page)
    kpis, kpi_error = extract_kpis(chunk.text, page_range, provider, model, kpi_categories, temperature, max_tokens)
    risks, risk_error = extract_risks(chunk.text, page_range, provider, model, risk_categories, temperature, max_tokens)
    errors = [e for e in (kpi_error, risk_error) if e]
    return {"chunk_id": chunk.chunk_id, "kpis": kpis, "risks": risks, "errors": errors}


def run_pipeline(
    pdf_path: str | Path,
    config: dict[str, Any],
    force_ocr: bool = False,
    workers: int = 4,
    progress_callback: Optional[Any] = None,
) -> AnalysisResult:
    pdf_path = Path(pdf_path)

    provider_cfg = config["provider"]
    provider = get_provider(provider_cfg["name"])
    model = provider_cfg["model"]
    temperature = provider_cfg.get("temperature", 0.1)
    max_tokens = provider_cfg.get("max_tokens", 4096)

    ocr_cfg = config.get("ocr", {})
    chunking_cfg = config.get("chunking", {})
    kpi_categories = config.get("kpi_categories", ["other"])
    risk_categories = config.get("risk_categories", ["other"])

    pages: list[PageResult] = extract_pdf(
        pdf_path,
        min_chars_for_text_page=ocr_cfg.get("min_chars_for_text_page", 40),
        dpi=ocr_cfg.get("dpi", 300),
        force_ocr=force_ocr,
    )
    ocr_pages = [p.page_number for p in pages if p.method == "ocr"]
    empty_pages = [p.page_number for p in pages if p.method == "empty"]

    chunks: list[Chunk] = chunk_pages(
        pages,
        max_chars_per_chunk=chunking_cfg.get("max_chars_per_chunk", 12000),
        overlap_chars=chunking_cfg.get("overlap_chars", 500),
    )

    if not chunks:
        raise ValueError(
            f"No extractable text found in {pdf_path} (all {len(pages)} page(s) came back empty, "
            "even after OCR). Check the file or try --force-ocr."
        )

    all_kpis: list[KPI] = []
    all_risks: list[RiskFactor] = []
    chunk_errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_chunk, chunk, provider, model, temperature, max_tokens, kpi_categories, risk_categories): chunk
            for chunk in chunks
        }
        completed = 0
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                result = future.result()
                all_kpis.extend(result["kpis"])
                all_risks.extend(result["risks"])
                chunk_errors.extend(result["errors"])
            except Exception as exc:  # noqa: BLE001
                chunk_errors.append(f"Chunk {chunk.chunk_id} (pages {chunk.start_page}-{chunk.end_page}): {exc}")
            completed += 1
            if progress_callback:
                progress_callback(completed, len(chunks))
            else:
                print(f"[{completed}/{len(chunks)}] chunks processed", file=sys.stderr)

    merged_kpis = merge_kpis(all_kpis)
    merged_risks = merge_risks(all_risks)

    summary, summary_error = generate_summary(merged_kpis, merged_risks, provider, model, temperature=0.2, max_tokens=2048)
    if summary is None:
        chunk_errors.append(f"Executive summary generation failed: {summary_error}")
        summary = ExecutiveSummary(
            overview="Executive summary could not be generated automatically; see extracted KPIs and risks below.",
            key_takeaways=[],
            overall_risk_level=_derive_fallback_risk_level(merged_risks),
        )

    return AnalysisResult(
        document=str(pdf_path),
        generated_at=datetime.now(timezone.utc).isoformat(),
        provider=provider_cfg["name"],
        model=model,
        kpis=merged_kpis,
        risks=merged_risks,
        summary=summary,
        metadata={
            "total_pages": len(pages),
            "ocr_pages": ocr_pages,
            "empty_pages": empty_pages,
            "total_chunks": len(chunks),
            "raw_kpi_count": len(all_kpis),
            "raw_risk_count": len(all_risks),
            "errors": chunk_errors,
        },
    )

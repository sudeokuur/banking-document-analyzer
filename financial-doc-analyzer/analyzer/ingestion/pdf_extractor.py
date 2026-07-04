"""PDF text extraction with automatic OCR fallback for scanned pages.

Primary path: `pdfplumber` (good layout-aware text extraction). If a page
comes back with little/no text (a strong signal it's a scanned image rather
than a text layer), or `--force-ocr` is set, the page is rendered to an image
via `pdf2image` and run through `pytesseract`. If `pdfplumber` itself fails to
open the document at all, we fall back to `pypdf` for basic extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PageResult:
    page_number: int  # 1-indexed
    text: str
    method: str  # "text" | "ocr" | "empty"
    char_count: int


def _ocr_page(pdf_path: Path, page_number: int, dpi: int) -> str:
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_number, last_page=page_number)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def _extract_with_pdfplumber(pdf_path: Path, min_chars_for_text_page: int, dpi: int, force_ocr: bool) -> list[PageResult]:
    import pdfplumber

    results: list[PageResult] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "") if not force_ocr else ""
            if force_ocr or len(text.strip()) < min_chars_for_text_page:
                ocr_text = _ocr_page(pdf_path, i, dpi)
                if len(ocr_text.strip()) > len(text.strip()):
                    results.append(PageResult(page_number=i, text=ocr_text, method="ocr", char_count=len(ocr_text)))
                    continue
            method = "text" if text.strip() else "empty"
            results.append(PageResult(page_number=i, text=text, method=method, char_count=len(text)))
    return results


def _extract_with_pypdf(pdf_path: Path) -> list[PageResult]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    results = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        method = "text" if text.strip() else "empty"
        results.append(PageResult(page_number=i, text=text, method=method, char_count=len(text)))
    return results


def extract_pdf(
    pdf_path: str | Path,
    min_chars_for_text_page: int = 40,
    dpi: int = 300,
    force_ocr: bool = False,
) -> list[PageResult]:
    """Extract text from every page of a PDF, falling back to OCR per-page as needed.

    Returns a list of `PageResult`, one per page, in document order.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        return _extract_with_pdfplumber(pdf_path, min_chars_for_text_page, dpi, force_ocr)
    except Exception:  # noqa: BLE001 - pdfplumber failed to even open the file; fall back
        return _extract_with_pypdf(pdf_path)


def combine_pages(pages: list[PageResult]) -> str:
    """Join per-page text into a single document string with page markers,
    used for chunking downstream."""
    parts = []
    for p in pages:
        parts.append(f"\n\n[--- Page {p.page_number} ---]\n{p.text}")
    return "".join(parts)

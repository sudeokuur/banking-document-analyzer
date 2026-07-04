"""Tests for PDF text extraction against the real generated sample PDF.

These exercise the actual `pdfplumber` extraction path (and, if Tesseract /
Poppler are installed, the OCR fallback path too) rather than mocking them --
PDF parsing bugs are exactly the kind of thing that only show up against a
real file.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ingestion.pdf_extractor import combine_pages, extract_pdf

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = REPO_ROOT / "sample_data" / "sample_bank_report.pdf"

HAS_OCR_DEPS = shutil.which("tesseract") is not None and shutil.which("pdftoppm") is not None


@pytest.fixture(scope="module", autouse=True)
def ensure_sample_pdf_exists():
    if not SAMPLE_PDF.exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_sample_pdf.py")], check=True)


def test_extract_pdf_returns_text_for_every_page():
    pages = extract_pdf(SAMPLE_PDF)
    assert len(pages) == 2
    for page in pages:
        assert page.method == "text"
        assert page.char_count > 0


def test_extracted_text_contains_known_content():
    pages = extract_pdf(SAMPLE_PDF)
    full_text = combine_pages(pages)
    assert "NORTHBRIDGE REGIONAL BANK" in full_text
    assert "11.4%" in full_text
    assert "Commercial real estate concentration" in full_text


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_pdf(REPO_ROOT / "sample_data" / "does_not_exist.pdf")


@pytest.mark.skipif(not HAS_OCR_DEPS, reason="tesseract/poppler not installed")
def test_force_ocr_still_recovers_recognizable_text():
    pages = extract_pdf(SAMPLE_PDF, force_ocr=True)
    assert all(p.method == "ocr" for p in pages)
    full_text = combine_pages(pages).upper()
    # OCR of a clean, computer-rendered PDF page should reliably recover this.
    assert "NORTHBRIDGE" in full_text
    assert "BANK" in full_text

#!/usr/bin/env python
"""Generates sample_data/sample_bank_report.pdf from sample_bank_report.txt.

Deliberately dependency-free (no reportlab/fpdf) -- it writes raw PDF syntax
directly using one of the 14 standard PDF fonts (Helvetica), so this script
runs with nothing beyond the Python standard library. Re-run it any time you
edit sample_bank_report.txt.

Usage:
    python scripts/generate_sample_pdf.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

PAGE_WIDTH = 612  # Letter, points
PAGE_HEIGHT = 792
MARGIN = 50
FONT_SIZE = 10
LEADING = 14
CHARS_PER_LINE = 95
LINES_PER_PAGE = (PAGE_HEIGHT - 2 * MARGIN) // LEADING


def _escape_pdf_string(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _wrap_source_lines(raw_text: str) -> list[str]:
    """Word-wrap the source text into fixed-width lines, preserving blank
    lines between paragraphs and treating ALL-CAPS section headers as their
    own line."""
    lines: list[str] = []
    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            lines.append("")
            continue
        wrapped = textwrap.wrap(stripped, width=CHARS_PER_LINE) or [""]
        lines.extend(wrapped)
    return lines


def _paginate(lines: list[str], lines_per_page: int) -> list[list[str]]:
    return [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)] or [[]]


def _build_content_stream(page_lines: list[str]) -> bytes:
    y = PAGE_HEIGHT - MARGIN
    parts = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEADING} TL", f"{MARGIN} {y} Td"]
    first = True
    for line in page_lines:
        if not first:
            parts.append("T*")
        first = False
        parts.append(f"({_escape_pdf_string(line)}) Tj")
    parts.append("ET")
    stream = "\n".join(parts).encode("latin-1", errors="replace")
    return stream


def write_pdf(pages_lines: list[list[str]], output_path: Path) -> None:
    objects: list[bytes] = []

    def add_object(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # 1-indexed object number

    font_obj_num = add_object(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )

    page_obj_nums = []
    content_obj_nums = []
    for page_lines in pages_lines:
        stream = _build_content_stream(page_lines)
        content_body = f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream"
        content_obj_nums.append(add_object(content_body))

    # Object numbers are sequential and known in advance: font (already added),
    # one content object per page (already added), one page object per page
    # (about to be added), then the Pages object itself right after.
    pages_obj_placeholder = len(objects) + len(pages_lines) + 1

    # Page objects reference the (not-yet-written) Pages parent by number.
    for i, page_lines in enumerate(pages_lines):
        content_num = content_obj_nums[i]
        body = (
            f"<< /Type /Page /Parent {pages_obj_placeholder} 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
            f"/Contents {content_num} 0 R >>"
        ).encode("latin-1")
        page_obj_nums.append(add_object(body))

    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    pages_body = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_nums)} >>".encode("latin-1")
    pages_obj_num = add_object(pages_body)
    assert pages_obj_num == pages_obj_placeholder, "Object numbering drifted; check add_object ordering."

    catalog_body = f"<< /Type /Catalog /Pages {pages_obj_num} 0 R >>".encode("latin-1")
    catalog_obj_num = add_object(catalog_body)

    # --- Serialize with a correct xref table ---
    buf = bytearray()
    buf += b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0] * (len(objects) + 1)  # index 0 unused (free list head)

    for i, body in enumerate(objects, start=1):
        offsets[i] = len(buf)
        buf += f"{i} 0 obj\n".encode("latin-1")
        buf += body
        buf += b"\nendobj\n"

    xref_start = len(buf)
    buf += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    buf += b"0000000000 65535 f \n"
    for i in range(1, len(objects) + 1):
        buf += f"{offsets[i]:010d} 00000 n \n".encode("latin-1")

    buf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj_num} 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF"
    ).encode("latin-1")

    output_path.write_bytes(bytes(buf))


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    txt_path = here / "sample_data" / "sample_bank_report.txt"
    pdf_path = here / "sample_data" / "sample_bank_report.pdf"

    raw_text = txt_path.read_text(encoding="utf-8")
    lines = _wrap_source_lines(raw_text)
    pages_lines = _paginate(lines, LINES_PER_PAGE)
    write_pdf(pages_lines, pdf_path)
    print(f"Wrote {pdf_path} ({len(pages_lines)} page(s)).")


if __name__ == "__main__":
    main()

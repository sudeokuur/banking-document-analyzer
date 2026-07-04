"""Splits extracted document pages into overlapping character-based chunks
for LLM extraction, keeping track of which page range each chunk covers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pdf_extractor import PageResult


@dataclass
class Chunk:
    chunk_id: int
    text: str
    start_page: int
    end_page: int
    char_count: int


def chunk_pages(pages: list[PageResult], max_chars_per_chunk: int = 12000, overlap_chars: int = 500) -> list[Chunk]:
    """Concatenate page texts (skipping empty pages) and split into overlapping
    chunks of at most `max_chars_per_chunk` characters, tracking the page range
    each chunk was drawn from.

    A single very long page is itself split across multiple chunks if needed.
    """
    if max_chars_per_chunk <= overlap_chars:
        raise ValueError("max_chars_per_chunk must be greater than overlap_chars.")

    # Build a flat list of (page_number, char) so we can map any offset in the
    # concatenated text back to the page it came from.
    combined = []
    page_of_char: list[int] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        if combined:
            combined.append("\n\n")
            page_of_char.extend([page.page_number] * 2)
        combined.append(text)
        page_of_char.extend([page.page_number] * len(text))

    full_text = "".join(combined)
    if not full_text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    n = len(full_text)

    while start < n:
        end = min(start + max_chars_per_chunk, n)
        chunk_text = full_text[start:end]
        start_page = page_of_char[start]
        end_page = page_of_char[end - 1]
        chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, start_page=start_page, end_page=end_page, char_count=len(chunk_text)))
        chunk_id += 1
        if end == n:
            break
        start = end - overlap_chars

    return chunks

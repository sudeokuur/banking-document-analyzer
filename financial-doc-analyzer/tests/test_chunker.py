"""Unit tests for the character-based chunker (no API calls required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.ingestion.chunker import chunk_pages
from analyzer.ingestion.pdf_extractor import PageResult


def test_single_small_page_produces_one_chunk():
    pages = [PageResult(page_number=1, text="Hello world. " * 5, method="text", char_count=65)]
    chunks = chunk_pages(pages, max_chars_per_chunk=1000, overlap_chars=100)
    assert len(chunks) == 1
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 1


def test_empty_pages_are_skipped():
    pages = [
        PageResult(page_number=1, text="", method="empty", char_count=0),
        PageResult(page_number=2, text="Some real content here.", method="text", char_count=24),
    ]
    chunks = chunk_pages(pages, max_chars_per_chunk=1000, overlap_chars=100)
    assert len(chunks) == 1
    assert "Some real content" in chunks[0].text


def test_long_text_splits_into_multiple_chunks_with_overlap():
    text = "word " * 2000  # 10000 chars
    pages = [PageResult(page_number=1, text=text, method="text", char_count=len(text))]
    chunks = chunk_pages(pages, max_chars_per_chunk=3000, overlap_chars=200)
    assert len(chunks) > 1
    # consecutive chunks should overlap
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text[-200:]
        assert tail[:50] in b.text


def test_chunk_tracks_page_range_across_pages():
    pages = [
        PageResult(page_number=1, text="A" * 1000, method="text", char_count=1000),
        PageResult(page_number=2, text="B" * 1000, method="text", char_count=1000),
        PageResult(page_number=3, text="C" * 1000, method="text", char_count=1000),
    ]
    chunks = chunk_pages(pages, max_chars_per_chunk=5000, overlap_chars=100)
    assert len(chunks) == 1
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 3


def test_rejects_invalid_overlap_config():
    pages = [PageResult(page_number=1, text="x" * 100, method="text", char_count=100)]
    try:
        chunk_pages(pages, max_chars_per_chunk=100, overlap_chars=200)
        assert False, "expected ValueError"
    except ValueError:
        pass

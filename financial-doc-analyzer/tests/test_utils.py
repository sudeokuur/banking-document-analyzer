"""Unit tests for the retry decorator and JSON/numeric parsing helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.utils import extract_json, parse_numeric, retry


def test_extract_json_from_fenced_block():
    text = 'Here is the data:\n```json\n{"a": 1, "b": [1, 2]}\n```\nHope that helps.'
    data = extract_json(text)
    assert data == {"a": 1, "b": [1, 2]}


def test_extract_json_bare():
    text = '{"a": 1}'
    assert extract_json(text) == {"a": 1}


def test_extract_json_array_with_surrounding_prose():
    text = 'Sure, here is the array: [{"name": "x"}, {"name": "y"}] -- let me know if you need more.'
    data = extract_json(text)
    assert len(data) == 2


def test_extract_json_raises_on_garbage():
    try:
        extract_json("this is not json at all, sorry")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_parse_numeric_percentage():
    assert parse_numeric("14.2%") == 14.2


def test_parse_numeric_currency_with_commas():
    assert parse_numeric("$1,234.56") == 1234.56


def test_parse_numeric_none_for_no_digits():
    assert parse_numeric("N/A") is None


def test_retry_succeeds_after_transient_failures():
    calls = {"count": 0}

    @retry(attempts=3, base_delay=0.01, max_delay=0.02)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3


def test_retry_raises_after_exhausting_attempts():
    @retry(attempts=2, base_delay=0.01, max_delay=0.02)
    def always_fails():
        raise RuntimeError("permanent")

    try:
        always_fails()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

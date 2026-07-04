"""Small stdlib-only helpers shared across the pipeline: a retry decorator for
flaky API calls, and a tolerant JSON extractor for parsing LLM output that
may be wrapped in markdown fences or preceded/followed by prose.
"""

from __future__ import annotations

import functools
import json
import re
import time
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def retry(attempts: int = 3, base_delay: float = 1.0, max_delay: float = 10.0, exceptions: tuple = (Exception,)):
    """Simple exponential-backoff retry decorator (no external dependency).

    Retries `attempts` times total, sleeping `base_delay * 2**n` seconds
    (capped at `max_delay`) between attempts.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt == attempts - 1:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    time.sleep(delay)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def extract_json(text: str) -> Any:
    """Extract and parse a JSON object/array from raw LLM output.

    Tries, in order: a fenced ```json ... ``` block, the whole string as-is,
    then the largest {...} or [...] span in the text. Raises `ValueError` if
    nothing parseable is found.
    """
    if not text or not text.strip():
        raise ValueError("Empty text; nothing to parse as JSON.")

    candidates = []

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    candidates.append(text.strip())

    # Largest {...} span
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1])

    # Largest [...] span
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        candidates.append(text[bracket_start : bracket_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON from LLM output (first 200 chars): {text[:200]!r}")


def parse_numeric(value: Any) -> Optional[float]:
    """Best-effort parse of a financial value string (e.g. '$12.4M', '14.2%', '1,234') to a float.

    Returns None if no numeric value can be extracted. Does not attempt unit
    conversion (e.g. millions vs billions) -- that's preserved separately in
    the `unit` field of a KPI.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    match = re.search(r"-?\d[\d,]*\.?\d*", value)
    if not match:
        return None
    cleaned = match.group(0).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None

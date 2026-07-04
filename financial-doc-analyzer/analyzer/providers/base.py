"""Base provider interface that all LLM provider integrations implement."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GenerationResult:
    """Normalized result returned by every provider's `generate` call."""

    text: str
    model: str
    provider: str
    latency_seconds: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw_error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.raw_error is None


class BaseProvider(ABC):
    """Subclass this to add a new LLM provider.

    Implementations should:
      - Read their API key from the environment lazily (not at import time).
      - Raise a clear error if the SDK isn't installed or the key is missing.
      - Wrap the underlying SDK call with timing and normalize the response
        into a `GenerationResult`.
    """

    name: str = "base"

    @abstractmethod
    def _call(
        self, prompt: str, model: str, temperature: float, max_tokens: int, system: Optional[str], **kwargs: Any
    ) -> tuple[str, Optional[int], Optional[int]]:
        """Perform the actual API call. Returns (text, input_tokens, output_tokens)."""
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        start = time.monotonic()
        try:
            text, in_tok, out_tok = self._call(
                prompt, model=model, temperature=temperature, max_tokens=max_tokens, system=system, **kwargs
            )
            latency = time.monotonic() - start
            return GenerationResult(
                text=text,
                model=model,
                provider=self.name,
                latency_seconds=latency,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        except Exception as exc:  # noqa: BLE001 - capture any provider failure rather than crash the pipeline
            latency = time.monotonic() - start
            return GenerationResult(
                text="",
                model=model,
                provider=self.name,
                latency_seconds=latency,
                raw_error=f"{type(exc).__name__}: {exc}",
            )

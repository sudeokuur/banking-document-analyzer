"""Anthropic provider integration."""

from __future__ import annotations

import os
from typing import Any, Optional

from analyzer.utils import retry

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "The `anthropic` package is required for AnthropicProvider. Install with `pip install anthropic`."
                ) from exc
            self._client = Anthropic(api_key=api_key)
        return self._client

    @retry(attempts=3, base_delay=1.0, max_delay=10.0)
    def _call(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system: Optional[str],
        **kwargs: Any,
    ) -> tuple[str, Optional[int], Optional[int]]:
        client = self._get_client()
        kwargs_ = {}
        if system:
            kwargs_["system"] = system

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs_,
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", None) if usage else None
        out_tok = getattr(usage, "output_tokens", None) if usage else None
        return text, in_tok, out_tok

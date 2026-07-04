"""Google (Gemini) provider integration."""

from __future__ import annotations

import os
from typing import Any, Optional

from analyzer.utils import retry

from .base import BaseProvider


class GoogleProvider(BaseProvider):
    name = "google"

    def __init__(self) -> None:
        self._configured = False

    def _configure(self):
        if not self._configured:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY is not set in the environment.")
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise RuntimeError(
                    "The `google-generativeai` package is required for GoogleProvider. "
                    "Install with `pip install google-generativeai`."
                ) from exc
            genai.configure(api_key=api_key)
            self._configured = True

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
        self._configure()
        import google.generativeai as genai

        gen_model = genai.GenerativeModel(model, system_instruction=system)
        response = gen_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", None) if usage else None
        out_tok = getattr(usage, "candidates_token_count", None) if usage else None
        return text, in_tok, out_tok

"""Provider registry. Add new providers here so config-driven lookup can find them by name."""

from .anthropic_provider import AnthropicProvider
from .base import BaseProvider, GenerationResult
from .google_provider import GoogleProvider
from .openai_provider import OpenAIProvider

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}

_instances: dict[str, BaseProvider] = {}


def get_provider(name: str) -> BaseProvider:
    """Return a cached provider instance by registry name (e.g. 'anthropic')."""
    key = name.lower()
    if key not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {sorted(PROVIDER_REGISTRY)}"
        )
    if key not in _instances:
        _instances[key] = PROVIDER_REGISTRY[key]()
    return _instances[key]


__all__ = ["BaseProvider", "GenerationResult", "PROVIDER_REGISTRY", "get_provider"]

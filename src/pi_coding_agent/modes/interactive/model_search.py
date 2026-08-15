"""Search text used to rank models in selectors.

Ported from ``packages/coding-agent/src/modes/interactive/model-search.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSearchItem:
    id: str
    provider: str
    name: str | None = None


def get_model_search_text(item: ModelSearchItem) -> str:
    name = f" {item.name}" if item.name else ""
    return f"{item.id} {item.provider} {item.provider}/{item.id} {item.provider} {item.id}{name}"


def get_model_selector_search_text(item: ModelSearchItem) -> str:
    """Keep the bare model ID out of the leading position.

    The `/model` selector must rank an exact provider-prefixed query above
    proxy-provider IDs such as ``openrouter/openai/gpt-5``.
    """
    name = f" {item.name}" if item.name else ""
    return f"{item.provider} {item.provider}/{item.id} {item.provider} {item.id}{name}"


__all__ = ["ModelSearchItem", "get_model_search_text", "get_model_selector_search_text"]

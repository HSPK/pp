"""``pi --list-models``.

Ported from ``packages/coding-agent/src/cli/list-models.ts``: prints an aligned
table of every available model, optionally narrowed by a fuzzy search pattern.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from pi_ai import Model
from pi_tui.fuzzy import fuzzy_filter

from pi_coding_agent.core.auth_guidance import format_no_models_available_message
from pi_coding_agent.utils.js_number import to_fixed

_HEADERS = ("provider", "model", "context", "max-out", "thinking", "images")


def format_token_count(count: int) -> str:
    """``200000`` -> ``200K``, ``1500000`` -> ``1.5M`` (TS ``formatTokenCount``)."""
    if count >= 1_000_000:
        millions = count / 1_000_000
        return f"{millions:.0f}M" if millions % 1 == 0 else f"{to_fixed(millions, 1)}M"
    if count >= 1_000:
        thousands = count / 1_000
        return f"{thousands:.0f}K" if thousands % 1 == 0 else f"{to_fixed(thousands, 1)}K"
    return str(count)


def _row(model: Model) -> tuple[str, str, str, str, str, str]:
    return (
        model.provider,
        model.id,
        format_token_count(model.context_window),
        format_token_count(model.max_tokens),
        "yes" if model.reasoning else "no",
        "yes" if "image" in model.input else "no",
    )


async def list_models(
    model_runtime: Any,
    search_pattern: str | None = None,
    write: Callable[[str], None] | None = None,
) -> None:
    emit = write or print

    get_error = getattr(model_runtime, "get_error", None)
    load_error = get_error() if get_error is not None else None
    if load_error:
        print(f"Warning: errors loading models.json:\n{load_error}", file=sys.stderr)

    models = list(await model_runtime.get_available())
    if not models:
        emit(format_no_models_available_message())
        return

    if search_pattern:
        models = fuzzy_filter(models, search_pattern, lambda m: f"{m.provider} {m.id}")
    if not models:
        emit(f'No models matching "{search_pattern}"')
        return

    models.sort(key=lambda m: (m.provider, m.id))
    rows = [_row(model) for model in models]
    widths = [max(len(_HEADERS[i]), *(len(row[i]) for row in rows)) for i in range(len(_HEADERS))]

    def render(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    emit(render(_HEADERS))
    for row in rows:
        emit(render(row))


async def handle_list_models(
    search_pattern: str | None = None,
    *,
    agent_dir: str | None = None,
    model_runtime: Any = None,
    write: Callable[[str], None] | None = None,
) -> int:
    """Build a `ModelRuntime` if needed, then print the table."""
    if model_runtime is None:
        from pi_coding_agent.core.model_runtime import ModelRuntime

        model_runtime = await ModelRuntime.create(agent_dir=agent_dir)
    await list_models(model_runtime, search_pattern, write)
    return 0


__all__ = ["format_token_count", "handle_list_models", "list_models"]

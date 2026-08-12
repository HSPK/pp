"""Auto-compaction on context-usage threshold.

Python port of `packages/coding-agent/examples/extensions/trigger-compact.ts`.

Compacts the session automatically the first time context usage crosses
`COMPACT_THRESHOLD_TOKENS`, and registers a `/trigger-compact` command that
compacts on demand.

Start pi with this extension::

    pi -e ./examples/extensions/trigger_compact.py
"""

from __future__ import annotations

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import (
    CompactOptions,
    ExtensionCommandContext,
    ExtensionContext,
    TurnEndEvent,
)

COMPACT_THRESHOLD_TOKENS = 100_000

_UNSET = object()


def pi_extension(pi: ExtensionAPI) -> None:
    # `previous_tokens` distinguishes three states, exactly as the TypeScript
    # `number | null | undefined` does: `_UNSET` = no turn seen yet,
    # `None` = a turn reported no usage, otherwise the token count.
    state: dict[str, object] = {"previous_tokens": _UNSET}

    def trigger_compaction(ctx: ExtensionContext, custom_instructions: str | None = None) -> None:
        if ctx.has_ui:
            ctx.ui.notify("Compaction started", "info")

        def on_complete(_result: object) -> None:
            if ctx.has_ui:
                ctx.ui.notify("Compaction completed", "info")

        def on_error(error: Exception) -> None:
            if ctx.has_ui:
                ctx.ui.notify(f"Compaction failed: {error}", "error")

        ctx.compact(
            CompactOptions(
                custom_instructions=custom_instructions,
                on_complete=on_complete,
                on_error=on_error,
            )
        )

    def on_turn_end(_event: TurnEndEvent, ctx: ExtensionContext) -> None:
        usage = ctx.get_context_usage()
        current_tokens = usage.tokens if usage is not None else None
        if current_tokens is None:
            return

        previous_tokens = state["previous_tokens"]
        crossed_threshold = (
            previous_tokens is not _UNSET
            and previous_tokens is not None
            and previous_tokens <= COMPACT_THRESHOLD_TOKENS  # type: ignore[operator]
        )
        state["previous_tokens"] = current_tokens
        if not crossed_threshold or current_tokens <= COMPACT_THRESHOLD_TOKENS:
            return
        trigger_compaction(ctx)

    pi.on("turn_end", on_turn_end)

    async def handler(args: str, ctx: ExtensionCommandContext) -> None:
        instructions = args.strip() or None
        trigger_compaction(ctx, instructions)

    pi.register_command(
        "trigger-compact",
        handler=handler,
        description="Trigger compaction immediately",
    )

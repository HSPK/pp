"""Python port of `packages/coding-agent/test/trigger-compact-extension.test.ts`.

Exercises `examples/extensions/trigger_compact.py`, the Python port of
`examples/extensions/trigger-compact.ts`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pi_coding_agent.core.extensions.loader import ExtensionAPI, ExtensionRuntimeActions
from pi_coding_agent.core.extensions.types import (
    CompactOptions,
    ContextUsage,
    Extension,
    ExtensionContext,
    NullExtensionUIContext,
    TurnEndEvent,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "extensions"))

from trigger_compact import pi_extension


def create_context(tokens: int | None, compact_calls: list[CompactOptions]) -> ExtensionContext:
    """Port of the TypeScript `createContext` helper.

    Fields the TypeScript literal supplies that this port's
    `ExtensionContext` does not carry (`modelRegistry`) are simply absent;
    everything the handler under test reads is present.
    """
    return ExtensionContext(
        ui=NullExtensionUIContext(),
        mode="print",
        has_ui=False,
        cwd=str(Path.cwd()),
        session_manager=None,
        model=None,
        scoped_models=(),
        is_idle=lambda: True,
        is_project_trusted=lambda: True,
        signal=None,
        abort=lambda: None,
        has_pending_messages=lambda: False,
        shutdown=lambda: None,
        get_context_usage=lambda: ContextUsage(
            tokens=tokens,
            context_window=200_000,
            percent=None if tokens is None else tokens / 2000,
        ),
        compact=compact_calls.append,
        get_system_prompt=lambda: "",
    )


def test_only_auto_compacts_when_context_usage_crosses_the_threshold() -> None:
    extension = Extension(path="<inline:1>", resolved_path="<inline:1>")
    registered_commands: dict[str, Any] = extension.commands
    api = ExtensionAPI(extension, ExtensionRuntimeActions())

    pi_extension(api)

    turn_end_handlers = extension.handlers.get("turn_end") or []
    assert len(turn_end_handlers) == 1
    turn_end_handler = turn_end_handlers[0]

    compact_calls: list[CompactOptions] = []
    event = TurnEndEvent(turn_index=0, message=None)

    turn_end_handler(event, create_context(110_000, compact_calls))
    assert compact_calls == []

    turn_end_handler(event, create_context(120_000, compact_calls))
    assert compact_calls == []

    turn_end_handler(event, create_context(95_000, compact_calls))
    assert compact_calls == []

    turn_end_handler(event, create_context(105_000, compact_calls))
    assert len(compact_calls) == 1

    # TypeScript's `registerCommand` is a `vi.fn()` spy that is never asserted
    # beyond existing on the API object; assert the real registration instead.
    assert "trigger-compact" in registered_commands
    assert registered_commands["trigger-compact"].description == "Trigger compaction immediately"

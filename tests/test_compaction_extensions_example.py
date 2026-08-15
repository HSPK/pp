"""Python port of `packages/coding-agent/test/compaction-extensions-example.test.ts`.

The TypeScript test verifies that the documentation example from `extensions.md`
(shipped as `examples/extensions/custom-compaction.ts`) type-checks and works.

Two of the three TypeScript cases are compile-time smoke tests: they build a
handler closure whose body asserts on the event fields, never invoke it, and
then assert `typeof exampleExtension === "function"`. Python has no compile
step, so those are ported as runtime assertions on the real event dataclasses
-- the same fields the documented example destructures.

The third case (`dispatches through modelRegistry.complete`) is not portable;
see the comment on `test_custom_compaction_example_dispatch` below.
"""

from __future__ import annotations

from typing import Any

import pytest
from pi_agent.harness.compaction.compaction import CompactionSettings
from pi_agent.harness.compaction.utils import FileOperations
from pi_ai.utils.abort import AbortController

from pi_coding_agent.core.compaction import CompactionPreparation, CompactionResult
from pi_coding_agent.core.extensions.types import (
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
    SessionCompactEvent,
)
from pi_coding_agent.core.session_manager import CompactionEntry


def _preparation() -> CompactionPreparation:
    return CompactionPreparation(
        first_kept_entry_id="entry-1",
        messages_to_summarize=[
            {"role": "user", "content": [{"type": "text", "text": "please remember this"}], "timestamp": 0}  # type: ignore[list-item]
        ],
        turn_prefix_messages=[],
        is_split_turn=False,
        tokens_before=42,
        previous_summary=None,
        file_ops=FileOperations(),
        settings=CompactionSettings(enabled=True, reserve_tokens=0, keep_recent_tokens=20000),
    )


class _RecordingApi:
    """Stand-in for `ExtensionAPI` that only records `on()` registrations."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler


class TestDocumentationExample:
    async def test_custom_compaction_example_fields_are_accessible(self) -> None:
        """Port of "custom compaction example should type-check correctly"."""
        api = _RecordingApi()
        seen: dict[str, Any] = {}

        def example_extension(pi: _RecordingApi) -> None:
            async def handler(event: SessionBeforeCompactEvent, ctx: Any) -> SessionBeforeCompactResult:
                preparation = event.preparation
                branch_entries = event.branch_entries
                session_manager = ctx.session_manager

                seen["messages_to_summarize"] = preparation.messages_to_summarize
                seen["turn_prefix_messages"] = preparation.turn_prefix_messages
                seen["is_split_turn"] = preparation.is_split_turn
                seen["tokens_before"] = preparation.tokens_before
                seen["first_kept_entry_id"] = preparation.first_kept_entry_id
                seen["branch_entries"] = branch_entries
                seen["get_entries"] = session_manager.get_entries

                summary = "\n".join(
                    f"- {m['content'][:100] if isinstance(m['content'], str) else '[complex]'}"
                    for m in preparation.messages_to_summarize
                    if m["role"] == "user"
                )
                return SessionBeforeCompactResult(
                    compaction=CompactionResult(
                        summary=f"User requests:\n{summary}",
                        first_kept_entry_id=preparation.first_kept_entry_id,
                        tokens_before=preparation.tokens_before,
                    )
                )

            pi.on("session_before_compact", handler)

        example_extension(api)
        assert callable(api.handlers["session_before_compact"])

        class _SessionManager:
            def get_entries(self) -> list[Any]:
                return []

        class _Ctx:
            session_manager = _SessionManager()

        result = await api.handlers["session_before_compact"](
            SessionBeforeCompactEvent(
                preparation=_preparation(),
                branch_entries=[],
                reason="manual",
                will_retry=False,
                signal=AbortController().signal,
            ),
            _Ctx(),
        )

        # The TypeScript body asserts on the shape of everything it destructures.
        assert isinstance(seen["messages_to_summarize"], list)
        assert isinstance(seen["turn_prefix_messages"], list)
        assert isinstance(seen["is_split_turn"], bool)
        assert isinstance(seen["tokens_before"], int)
        assert callable(seen["get_entries"])
        assert isinstance(seen["first_kept_entry_id"], str)
        assert isinstance(seen["branch_entries"], list)
        # `modelRegistry.getApiKeyAndHeaders` has no counterpart: this port has no
        # `ModelRegistry` on the extension context (documented in
        # `core/extensions/types.py`), so that single assertion is dropped.

        assert result.compaction is not None
        assert result.compaction.summary == "User requests:\n- [complex]"
        assert result.compaction.first_kept_entry_id == "entry-1"
        assert result.compaction.tokens_before == 42

    @pytest.mark.skip(
        reason="`examples/extensions/custom-compaction.ts` calls `ctx.modelRegistry.find()` and "
        "`ctx.modelRegistry.complete()`. This port deliberately has no `ModelRegistry` on the "
        "extension context (see `core/extensions/types.py`: 'no `ModelRegistry` (see "
        "`model_runtime.py`'s own documented boundary)'), so the example extension itself cannot "
        "be ported and neither can this dispatch test."
    )
    def test_custom_compaction_example_dispatch(self) -> None:
        """Port of "custom compaction example dispatches through modelRegistry.complete"."""

    async def test_compact_event_should_have_correct_fields(self) -> None:
        """Port of "compact event should have correct fields"."""
        api = _RecordingApi()

        def check_compact_event(pi: _RecordingApi) -> None:
            async def handler(event: SessionCompactEvent) -> None:
                entry = event.compaction_entry
                from_extension = event.from_extension

                assert entry.type == "compaction"
                assert isinstance(entry.summary, str)
                assert isinstance(entry.tokens_before, int)
                assert isinstance(from_extension, bool)

            pi.on("session_compact", handler)

        check_compact_event(api)
        assert callable(api.handlers["session_compact"])

        await api.handlers["session_compact"](
            SessionCompactEvent(
                compaction_entry=CompactionEntry(
                    id="c1",
                    parent_id=None,
                    timestamp="2024-01-01T00:00:00.000Z",
                    summary="summary",
                    first_kept_entry_id="entry-1",
                    tokens_before=42,
                ),
                from_extension=True,
                reason="manual",
                will_retry=False,
            )
        )

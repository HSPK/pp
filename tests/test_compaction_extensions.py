"""Python port of `packages/coding-agent/test/compaction-extensions.test.ts`.

Tests for compaction extension events (`session_before_compact` /
`session_compact`).

The TypeScript suite is gated on a real `ANTHROPIC_API_KEY` and drives a live
Claude model. This port uses the scripted stream function from
`test_agent_session` instead -- no network call is made -- and seeds a
compactable history directly rather than issuing real prompts. Every assertion
about the events themselves is preserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_agent_session import (
    COMPACTION_SETTINGS,
    _seed_compactable_history,
    build_session,
    text_response,
)

from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.compaction import CompactionResult
from pi_coding_agent.core.extensions.types import (
    Extension,
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
    SessionCompactEvent,
)
from pi_coding_agent.core.session_manager import CompactionEntry, SessionManager

DEFAULT_RESPONSES = [text_response("SUMMARY OF HISTORY"), text_response("TURN PREFIX")]


def create_extension(
    captured_events: list[object],
    on_before_compact=None,
    on_compact=None,
    *,
    path: str = "test-extension",
) -> Extension:
    async def before(event, ctx):
        captured_events.append(event)
        if on_before_compact is not None:
            return on_before_compact(event)
        return None

    async def after(event, ctx):
        captured_events.append(event)
        if on_compact is not None:
            on_compact(event)
        return None

    return Extension(
        path=path,
        resolved_path=f"/test/{path}.py",
        handlers={"session_before_compact": [before], "session_compact": [after]},
    )


async def create_session(
    tmp_path: Path, extensions: list[Extension], responses=None
) -> tuple[AgentSession, SessionManager]:
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=list(responses if responses is not None else DEFAULT_RESPONSES),
        settings=COMPACTION_SETTINGS,
        extensions=extensions,
    )
    _seed_compactable_history(session_manager, session)
    return session, session_manager


class TestCompactionExtensions:
    async def test_should_emit_before_compact_and_compact_events(self, tmp_path: Path) -> None:
        captured: list[object] = []
        session, _sm = await create_session(tmp_path, [create_extension(captured)])
        try:
            await session.compact()

            before_events = [e for e in captured if isinstance(e, SessionBeforeCompactEvent)]
            compact_events = [e for e in captured if isinstance(e, SessionCompactEvent)]

            assert len(before_events) == 1
            assert len(compact_events) == 1

            before = before_events[0]
            assert before.preparation is not None
            assert before.preparation.messages_to_summarize is not None
            assert before.preparation.turn_prefix_messages is not None
            assert before.preparation.tokens_before >= 0
            assert isinstance(before.preparation.is_split_turn, bool)
            assert before.branch_entries is not None
            # session_manager/model live on ctx, not on the event.

            after = compact_events[0]
            assert after.compaction_entry is not None
            assert len(after.compaction_entry.summary) > 0
            assert after.compaction_entry.tokens_before >= 0
            assert after.from_extension is False
        finally:
            session.dispose()

    async def test_should_allow_extensions_to_cancel_compaction(self, tmp_path: Path) -> None:
        captured: list[object] = []
        extension = create_extension(captured, lambda event: SessionBeforeCompactResult(cancel=True))
        session, _sm = await create_session(tmp_path, [extension])
        try:
            with pytest.raises(RuntimeError, match="Compaction cancelled"):
                await session.compact()

            assert [e for e in captured if isinstance(e, SessionCompactEvent)] == []
        finally:
            session.dispose()

    async def test_should_allow_extensions_to_provide_custom_compaction(self, tmp_path: Path) -> None:
        custom_summary = "Custom summary from extension"
        captured: list[object] = []

        def on_before(event):
            if event.type == "session_before_compact":
                return SessionBeforeCompactResult(
                    compaction=CompactionResult(
                        summary=custom_summary,
                        first_kept_entry_id=event.preparation.first_kept_entry_id,
                        tokens_before=event.preparation.tokens_before,
                    )
                )
            return None

        session, _sm = await create_session(tmp_path, [create_extension(captured, on_before)])
        try:
            result = await session.compact()
            assert result.summary == custom_summary

            compact_events = [e for e in captured if isinstance(e, SessionCompactEvent)]
            assert len(compact_events) == 1
            assert compact_events[0].compaction_entry.summary == custom_summary
            assert compact_events[0].from_extension is True
        finally:
            session.dispose()

    async def test_should_include_entries_in_compact_event_after_compaction_is_saved(self, tmp_path: Path) -> None:
        captured: list[object] = []
        session, session_manager = await create_session(tmp_path, [create_extension(captured)])
        try:
            await session.compact()

            compact_events = [e for e in captured if isinstance(e, SessionCompactEvent)]
            assert len(compact_events) == 1
            entries = session_manager.get_entries()
            assert any(isinstance(e, CompactionEntry) for e in entries)
        finally:
            session.dispose()

    async def test_should_continue_with_default_compaction_if_extension_throws_error(self, tmp_path: Path) -> None:
        captured: list[object] = []

        async def throwing(event, ctx):
            captured.append(event)
            raise RuntimeError("Extension intentionally throws")

        async def after(event, ctx):
            captured.append(event)
            return None

        throwing_extension = Extension(
            path="throwing-extension",
            resolved_path="/test/throwing-extension.py",
            handlers={"session_before_compact": [throwing], "session_compact": [after]},
        )

        session, _sm = await create_session(tmp_path, [throwing_extension])
        try:
            result = await session.compact()

            assert result.summary is not None
            assert len(result.summary) > 0

            compact_events = [e for e in captured if isinstance(e, SessionCompactEvent)]
            assert len(compact_events) == 1
            assert compact_events[0].from_extension is False
        finally:
            session.dispose()

    async def test_should_call_multiple_extensions_in_order(self, tmp_path: Path) -> None:
        call_order: list[str] = []

        def make(name: str) -> Extension:
            async def before(event, ctx):
                call_order.append(f"{name}-before")
                return None

            async def after(event, ctx):
                call_order.append(f"{name}-after")
                return None

            return Extension(
                path=name,
                resolved_path=f"/test/{name}.py",
                handlers={"session_before_compact": [before], "session_compact": [after]},
            )

        session, _sm = await create_session(tmp_path, [make("extension1"), make("extension2")])
        try:
            await session.compact()

            assert call_order == [
                "extension1-before",
                "extension2-before",
                "extension1-after",
                "extension2-after",
            ]
        finally:
            session.dispose()

    async def test_should_pass_correct_data_in_before_compact_event(self, tmp_path: Path) -> None:
        captured_before: list[SessionBeforeCompactEvent] = []
        captured: list[object] = []

        def on_before(event):
            captured_before.append(event)
            return None

        session, session_manager = await create_session(tmp_path, [create_extension(captured, on_before)])
        try:
            await session.compact()

            assert len(captured_before) == 1
            event = captured_before[0]
            assert isinstance(event.preparation.is_split_turn, bool)
            assert event.preparation.first_kept_entry_id is not None

            assert isinstance(event.preparation.messages_to_summarize, list)
            assert isinstance(event.preparation.turn_prefix_messages, list)

            assert isinstance(event.preparation.tokens_before, int)

            assert isinstance(event.branch_entries, list)

            # session_manager and model runtime remain available on the session.
            assert callable(session.session_manager.get_entries)
            assert callable(session.model_runtime.get_auth)

            entries = session_manager.get_entries()
            assert isinstance(entries, list)
            assert len(entries) > 0
        finally:
            session.dispose()

    async def test_should_use_extension_compaction_even_with_different_values(self, tmp_path: Path) -> None:
        custom_summary = "Custom summary with modified values"
        captured: list[object] = []

        def on_before(event):
            if event.type == "session_before_compact":
                return SessionBeforeCompactResult(
                    compaction=CompactionResult(
                        summary=custom_summary,
                        first_kept_entry_id=event.preparation.first_kept_entry_id,
                        tokens_before=999,
                    )
                )
            return None

        session, _sm = await create_session(tmp_path, [create_extension(captured, on_before)])
        try:
            result = await session.compact()

            assert result.summary == custom_summary
            assert result.tokens_before == 999
        finally:
            session.dispose()

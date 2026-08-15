"""Python port of `packages/coding-agent/test/agent-session-compaction.test.ts`.

End-to-end tests for `AgentSession` compaction behavior:
- Manual compaction works correctly
- Session persistence during compaction
- The compaction entry is saved to the session file

The TypeScript suite is gated on a real `ANTHROPIC_API_KEY` and issues live LLM
calls. This port drives the same code path with the scripted stream function
from `test_agent_session` -- no network call is made -- so the assistant
responses (and their token usage) are deterministic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pi_ai.types import AssistantMessage
from test_agent_session import (
    COMPACTION_SETTINGS,
    _assistant,
    build_session,
    text_response,
)

from pi_coding_agent.core.agent_session import AgentSession, CompactionStartEvent
from pi_coding_agent.core.session_manager import CompactionEntry, SessionManager


def _answer(text: str) -> AssistantMessage:
    """A scripted assistant reply carrying enough usage for compaction to have
    something to measure (`text_response`'s default `Usage()` is all zeros, so
    `tokensBefore` would be 0)."""
    return _assistant(text, 200, 1)


async def create_session(
    tmp_path: Path, *, in_memory: bool = False, responses: list[AssistantMessage] | None = None
) -> tuple[AgentSession, SessionManager, list]:
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=list(responses or []),
        settings=COMPACTION_SETTINGS,
        persist_to_disk=not in_memory,
    )
    events: list = []
    session.subscribe(events.append)
    return session, session_manager, events


class TestAgentSessionCompactionE2E:
    async def test_should_trigger_manual_compaction_via_compact(self, tmp_path: Path) -> None:
        session, _sm, _events = await create_session(
            tmp_path,
            responses=[
                _answer("4"),
                _answer("6"),
                text_response("SUMMARY OF HISTORY"),
                text_response("TURN PREFIX"),
            ],
        )
        try:
            await asyncio.wait_for(session.prompt("What is 2+2? Reply with just the number."), timeout=10)
            await asyncio.wait_for(session.prompt("What is 3+3? Reply with just the number."), timeout=10)

            result = await asyncio.wait_for(session.compact(), timeout=10)

            assert result.summary is not None
            assert len(result.summary) > 0
            assert result.tokens_before > 0

            messages = session.messages
            assert len(messages) > 0

            # First message should be the summary.
            assert messages[0].role == "compactionSummary"
        finally:
            session.dispose()

    async def test_should_maintain_valid_session_state_after_compaction(self, tmp_path: Path) -> None:
        session, _sm, _events = await create_session(
            tmp_path,
            responses=[
                _answer("Paris"),
                _answer("Berlin"),
                text_response("SUMMARY OF HISTORY"),
                text_response("TURN PREFIX"),
                _answer("Rome"),
            ],
        )
        try:
            await asyncio.wait_for(session.prompt("What is the capital of France? One word answer."), timeout=10)
            await asyncio.wait_for(session.prompt("What is the capital of Germany? One word answer."), timeout=10)

            await asyncio.wait_for(session.compact(), timeout=10)

            # Session should still be usable.
            await asyncio.wait_for(session.prompt("What is the capital of Italy? One word answer."), timeout=10)

            assert len(session.messages) > 0

            assistant_messages = [m for m in session.messages if getattr(m, "role", None) == "assistant"]
            assert len(assistant_messages) > 0
        finally:
            session.dispose()

    async def test_should_persist_compaction_to_session_file(self, tmp_path: Path) -> None:
        session, session_manager, _events = await create_session(
            tmp_path,
            responses=[
                _answer("hello"),
                _answer("goodbye"),
                text_response("SUMMARY OF HISTORY"),
                text_response("TURN PREFIX"),
            ],
        )
        try:
            await asyncio.wait_for(session.prompt("Say hello"), timeout=10)
            await asyncio.wait_for(session.prompt("Say goodbye"), timeout=10)

            await asyncio.wait_for(session.compact(), timeout=10)

            entries = session_manager.get_entries()
            compaction_entries = [e for e in entries if isinstance(e, CompactionEntry)]
            assert len(compaction_entries) == 1

            compaction = compaction_entries[0]
            assert compaction.type == "compaction"
            assert len(compaction.summary) > 0
            assert isinstance(compaction.first_kept_entry_id, str)
            assert compaction.tokens_before > 0
        finally:
            session.dispose()

    async def test_should_work_with_no_session_mode_in_memory_only(self, tmp_path: Path) -> None:
        session, session_manager, _events = await create_session(
            tmp_path,
            in_memory=True,
            responses=[
                _answer("4"),
                _answer("6"),
                text_response("SUMMARY OF HISTORY"),
                text_response("TURN PREFIX"),
            ],
        )
        try:
            await asyncio.wait_for(session.prompt("What is 2+2? Reply with just the number."), timeout=10)
            await asyncio.wait_for(session.prompt("What is 3+3? Reply with just the number."), timeout=10)

            result = await asyncio.wait_for(session.compact(), timeout=10)

            assert result.summary is not None
            assert len(result.summary) > 0

            entries = session_manager.get_entries()
            assert len([e for e in entries if isinstance(e, CompactionEntry)]) == 1
        finally:
            session.dispose()

    async def test_should_emit_compaction_events_during_manual_compaction(self, tmp_path: Path) -> None:
        session, _sm, events = await create_session(
            tmp_path,
            responses=[
                _answer("hello"),
                _answer("goodbye"),
                text_response("SUMMARY OF HISTORY"),
                text_response("TURN PREFIX"),
            ],
        )
        try:
            await asyncio.wait_for(session.prompt("Say hello"), timeout=10)
            await asyncio.wait_for(session.prompt("Say goodbye"), timeout=10)

            await asyncio.wait_for(session.compact(), timeout=10)

            compaction_events = [e for e in events if e.type in ("compaction_start", "compaction_end")]
            assert len(compaction_events) == 2
            # TS uses `toEqual({ type, reason })`, i.e. the event carries no other field.
            assert compaction_events[0] == CompactionStartEvent(reason="manual")
            assert compaction_events[1].type == "compaction_end"
            assert compaction_events[1].reason == "manual"
            assert compaction_events[1].aborted is False
            assert compaction_events[1].will_retry is False

            message_end_events = [e for e in events if e.type == "message_end"]
            assert len(message_end_events) > 0
        finally:
            session.dispose()

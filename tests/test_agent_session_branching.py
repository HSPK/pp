"""Python port of `packages/coding-agent/test/agent-session-branching.test.ts`.

Tests for `AgentSession` forking behavior:
- Forking from a single message works
- Forking in `--no-session` mode (in-memory only)
- `get_user_messages_for_forking()` returns correct entries

The TypeScript suite is gated on a real `ANTHROPIC_API_KEY` and prompts a live
model. Here the session is built by `test_agent_session_runtime`'s
`_make_runtime` helper, which wires a real `AgentSession`/`SessionManager`/
`AgentSessionRuntime` to a scripted stream function -- no network call is made.
"""

from __future__ import annotations

from pathlib import Path

from pi_ai.types import TextContent
from test_agent_session_runtime import _make_assistant_message, _make_runtime, _wait


class TestAgentSessionForking:
    async def test_should_allow_forking_from_single_message(self, tmp_path: Path) -> None:
        runtime, _factory = await _make_runtime(
            tmp_path, persisted=True, responses=[_make_assistant_message([TextContent(text="hello")])]
        )
        try:
            session = runtime.session
            await _wait(session.prompt("Say hello"))

            user_messages = session.get_user_messages_for_forking()
            assert len(user_messages) == 1
            assert user_messages[0]["text"] == "Say hello"

            result = await _wait(runtime.fork(user_messages[0]["entryId"]))
            assert result["cancelled"] is False
            session = runtime.session
            assert result["selected_text"] == "Say hello"

            assert len(session.messages) == 0
            assert session.session_file is not None
            # Forking before the only user message starts a fresh child session,
            # which has nothing to flush yet, so no file exists on disk.
            assert Path(session.session_file).exists() is False
        finally:
            await _wait(runtime.dispose())

    async def test_should_support_in_memory_forking_in_no_session_mode(self, tmp_path: Path) -> None:
        runtime, _factory = await _make_runtime(tmp_path, responses=[_make_assistant_message([TextContent(text="hi")])])
        try:
            session = runtime.session
            # TS asserts `undefined`; this port models "no session file" as `None`.
            assert session.session_file is None

            await _wait(session.prompt("Say hi"))

            user_messages = session.get_user_messages_for_forking()
            assert len(user_messages) == 1
            assert len(session.messages) > 0

            result = await _wait(runtime.fork(user_messages[0]["entryId"]))
            assert result["cancelled"] is False
            session = runtime.session
            assert result["selected_text"] == "Say hi"

            assert len(session.messages) == 0
            assert session.session_file is None
        finally:
            await _wait(runtime.dispose())

    async def test_should_fork_from_middle_of_conversation(self, tmp_path: Path) -> None:
        runtime, _factory = await _make_runtime(
            tmp_path,
            persisted=True,
            responses=[
                _make_assistant_message([TextContent(text="one")]),
                _make_assistant_message([TextContent(text="two")]),
                _make_assistant_message([TextContent(text="three")]),
            ],
        )
        try:
            session = runtime.session
            await _wait(session.prompt("Say one"))
            await _wait(session.prompt("Say two"))
            await _wait(session.prompt("Say three"))

            user_messages = session.get_user_messages_for_forking()
            assert len(user_messages) == 3

            second_message = user_messages[1]
            result = await _wait(runtime.fork(second_message["entryId"]))
            assert result["cancelled"] is False
            session = runtime.session
            assert result["selected_text"] == "Say two"

            assert len(session.messages) == 2
            assert session.messages[0].role == "user"
            assert session.messages[1].role == "assistant"
        finally:
            await _wait(runtime.dispose())

"""Python port of `packages/coding-agent/test/suite/regressions/7150-rpc-prompt-during-compaction.test.ts`.

`source="rpc"` is passed through to `AgentSession.prompt` exactly as in
TypeScript; the RPC *driver* is not ported (see the README), but the session
behavior this test pins -- rejecting a prompt while a manual compaction is
still running, without persisting the prompt or starting a turn -- is.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness import create_harness, get_message_text, get_user_texts
from pi_ai.providers.faux import faux_assistant_message
from pi_ai.types import TextContent, UserMessage, now_ms

from pi_coding_agent.core.compaction import CompactionResult
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import SessionBeforeCompactResult
from pi_coding_agent.core.session_manager import SessionMessageEntry


async def test_rejects_rpc_prompt_while_manual_compaction_is_in_progress(tmp_path: Path) -> None:
    loop = asyncio.get_running_loop()
    compaction_started: asyncio.Future[None] = loop.create_future()
    compaction_released = asyncio.Event()

    def factory(pi: ExtensionAPI) -> None:
        async def on_before_compact(event, ctx) -> SessionBeforeCompactResult:
            if not compaction_started.done():
                compaction_started.set_result(None)
            await compaction_released.wait()
            return SessionBeforeCompactResult(
                compaction=CompactionResult(
                    summary="manual compacted",
                    first_kept_entry_id=event.preparation.first_kept_entry_id,
                    tokens_before=event.preparation.tokens_before,
                    details={},
                )
            )

        pi.on("session_before_compact", on_before_compact)

    harness = await create_harness(
        tmp_path,
        settings={"compaction": {"keepRecentTokens": 1}},
        extension_factories=[factory],
    )
    try:
        timestamp = now_ms()
        harness.session_manager.append_message(
            UserMessage(content=[TextContent(text="old user message")], timestamp=timestamp - 1000)
        )
        harness.session_manager.append_message(
            faux_assistant_message("old assistant response", timestamp=timestamp - 500)
        )
        harness.session.agent.state.messages = harness.session_manager.build_session_context().messages
        harness.set_responses([faux_assistant_message("probe response")])

        compact_task = asyncio.ensure_future(harness.session.compact())
        await compaction_started

        preflight_results: list[bool] = []
        prompt_error: Exception | None = None
        try:
            await harness.session.prompt(
                "PROBE-7150",
                source="rpc",
                preflight_result=preflight_results.append,
            )
        except Exception as error:
            prompt_error = error
        finally:
            compaction_released.set()
            await compact_task

        persisted_user_texts = [
            get_message_text(entry.message)
            for entry in harness.session_manager.get_entries()
            if isinstance(entry, SessionMessageEntry) and getattr(entry.message, "role", None) == "user"
        ]

        assert preflight_results == [False]
        assert prompt_error is not None
        assert "compaction is in progress" in str(prompt_error)
        assert "PROBE-7150" not in get_user_texts(harness)
        assert "PROBE-7150" not in persisted_user_texts
        assert harness.events_of_type("agent_start") == []
        assert harness.events_of_type("agent_settled") == []
    finally:
        harness.cleanup()

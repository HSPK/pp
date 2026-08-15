"""Python port of `packages/coding-agent/test/suite/regressions/pre-prompt-compaction-no-continue.test.ts`.

A session whose last assistant message stopped with `length` used to be
resumed with `agent.continue()` after the pre-prompt overflow compaction, which
dropped the new prompt. This pins that the new prompt runs as a normal turn.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from harness import create_harness, get_user_texts
from pi_ai.providers.faux import FauxModelDefinition, faux_assistant_message
from pi_ai.types import Cost, Usage, UserMessage

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import (
    CompactionResult,
    SessionBeforeCompactResult,
)


def _create_usage(total_tokens: int) -> Usage:
    return Usage(
        input=total_tokens,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=total_tokens,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


def _compacting_extension(pi: ExtensionAPI) -> None:
    async def on_before_compact(event, _ctx) -> SessionBeforeCompactResult:
        return SessionBeforeCompactResult(
            compaction=CompactionResult(
                summary="pre-prompt summary",
                first_kept_entry_id=event.preparation.first_kept_entry_id,
                tokens_before=event.preparation.tokens_before,
                details={},
            )
        )

    pi.on("session_before_compact", on_before_compact)


async def test_compacts_length_stop_overflow_before_a_new_prompt_without_continuing(
    tmp_path: Path,
) -> None:
    harness = await create_harness(
        tmp_path,
        models=[FauxModelDefinition(id="faux-1", context_window=100, max_tokens=100)],
        settings={"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[_compacting_extension],
    )
    try:
        import time

        now = int(time.time() * 1000)
        model = harness.get_model()
        assert model is not None
        harness.session_manager.append_message(
            UserMessage(
                role="user",
                content="previous prompt",
                timestamp=now - 1000,
            )
        )
        length_stop_assistant = dataclasses.replace(
            faux_assistant_message("length-stop assistant response", stop_reason="length", timestamp=now - 500),
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=_create_usage(100),
        )
        harness.session_manager.append_message(length_stop_assistant)
        harness.session.agent.state.messages = harness.session_manager.build_session_context().messages
        harness.set_responses([faux_assistant_message("answered next prompt")])

        continue_calls: list[object] = []
        original_continue = harness.session.agent.continue_

        async def spy_continue(*args, **kwargs):
            continue_calls.append((args, kwargs))
            return await original_continue(*args, **kwargs)

        harness.session.agent.continue_ = spy_continue  # type: ignore[method-assign]

        assert await harness.session.prompt("next prompt") is None

        assert continue_calls == []
        compaction_end = harness.events_of_type("compaction_end")[-1]
        assert compaction_end.reason == "overflow"
        assert compaction_end.aborted is False
        assert compaction_end.will_retry is True
        assert "next prompt" in get_user_texts(harness)
        assert harness.faux.state.call_count == 1
    finally:
        harness.cleanup()

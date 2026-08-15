"""Python port of `packages/coding-agent/test/suite/regressions/7253-manual-compact-during-response.test.ts`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from harness import Harness, create_harness
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import FauxModelDefinition, faux_assistant_message, faux_tool_call
from pi_ai.types import TextContent

from pi_coding_agent.core.compaction import CompactionResult
from pi_coding_agent.core.extensions.types import SessionBeforeCompactResult


def create_noop_tool() -> AgentTool:
    async def execute(_tool_call_id: str, _params: Any, _signal: Any = None, _on_update: Any = None):
        return AgentToolResult(content=[TextContent(text="done")], details={})

    return AgentTool(
        name="noop",
        label="No-op",
        description="Return immediately",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


@pytest.fixture
def harnesses() -> list[Harness]:
    created: list[Harness] = []
    yield created
    while created:
        created.pop().cleanup()


async def test_runs_only_the_requested_manual_compaction(tmp_path: Path, harnesses: list[Harness]) -> None:
    loop = asyncio.get_running_loop()
    second_response_started: asyncio.Future[None] = loop.create_future()
    second_response_released = asyncio.Event()

    def register(pi: Any) -> None:
        async def on_before_compact(event: Any, _ctx: Any) -> SessionBeforeCompactResult:
            return SessionBeforeCompactResult(
                compaction=CompactionResult(
                    summary=f"{event.reason} summary",
                    first_kept_entry_id=event.preparation.first_kept_entry_id,
                    tokens_before=event.preparation.tokens_before,
                    details={},
                )
            )

        pi.on("session_before_compact", on_before_compact)

    harness = await create_harness(
        tmp_path,
        models=[FauxModelDefinition(id="faux-1", context_window=1000, max_tokens=100)],
        settings={"compaction": {"enabled": True, "reserveTokens": 999, "keepRecentTokens": 2}},
        tools=[create_noop_tool()],
        extension_factories=[register],
    )
    harnesses.append(harness)

    async def second_response(*_args: Any, **_kwargs: Any):
        if not second_response_started.done():
            second_response_started.set_result(None)
        await second_response_released.wait()
        return faux_assistant_message("second response")

    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("noop", {}), stop_reason="toolUse"),
            second_response,
        ]
    )

    prompt_task = asyncio.ensure_future(harness.session.prompt("Run the tool, then continue responding."))
    await second_response_started

    compact_task = asyncio.ensure_future(harness.session.compact())
    second_response_released.set()
    result, _ = await asyncio.gather(compact_task, prompt_task)

    assert result.summary == "manual summary"
    assert [event.reason for event in harness.events_of_type("compaction_start")] == ["manual"]
    assert [event.reason for event in harness.events_of_type("compaction_end")] == ["manual"]
    assert len([entry for entry in harness.session_manager.get_entries() if entry.type == "compaction"]) == 1

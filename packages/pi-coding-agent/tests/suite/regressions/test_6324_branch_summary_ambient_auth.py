"""Python port of `packages/coding-agent/test/suite/regressions/6324-branch-summary-ambient-auth.test.ts`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from harness import Harness, assistant_msg, create_harness, user_msg
from pi_ai.types import AssistantMessage, Cost, DoneEvent, Model, TextContent, Usage, now_ms
from pi_ai.utils.event_stream import create_assistant_message_event_stream


@pytest.fixture
def harnesses() -> list[Harness]:
    created: list[Harness] = []
    yield created
    while created:
        created.pop().cleanup()


async def test_summarizes_tree_branches_when_request_auth_has_no_api_key(
    tmp_path: Path, harnesses: list[Harness]
) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    harnesses.append(harness)

    stream_call_count = 0

    def stream_function(model: Model, _context: Any, options: Any = None, **_kwargs: Any) -> Any:
        nonlocal stream_call_count
        stream_call_count += 1
        assert getattr(options, "api_key", None) is None

        stream = create_assistant_message_event_stream()
        message = AssistantMessage(
            content=[TextContent(text="branch summary text")],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(
                input=1,
                output=1,
                cache_read=0,
                cache_write=0,
                total_tokens=2,
                cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0.25),
            ),
            stop_reason="stop",
            timestamp=now_ms(),
        )
        stream.push(DoneEvent(reason="stop", message=message))
        stream.end(message)
        return stream

    harness.session.agent.stream_function = stream_function

    target_id = harness.session_manager.append_message(user_msg("first branch"))
    harness.session_manager.append_message(assistant_msg("first reply"))
    harness.session_manager.append_message(user_msg("abandoned branch work"))
    harness.session_manager.append_message(assistant_msg("abandoned reply"))

    result = await harness.session.navigate_tree(target_id, summarize=True)

    assert result.cancelled is False
    assert stream_call_count == 1
    assert result.summary_entry is not None
    assert result.summary_entry.type == "branch_summary"
    assert "branch summary text" in result.summary_entry.summary
    assert result.summary_entry.usage is not None
    assert result.summary_entry.usage.cost.total == 0.25

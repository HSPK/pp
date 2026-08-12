"""Python port of `packages/coding-agent/test/compaction-summary-reasoning.test.ts`.

TypeScript mocks `completeSimple` from `@earendil-works/pi-ai/compat`. This
port has no `Models.completeSimple`; `generate_summary*`/`compact` take a
`StreamFn` instead (see the docstring of
`pi_agent.harness.compaction.compaction`), so the "mock" here is a scripted
stream function that records the `SimpleStreamOptions` it was handed. The
`apiKey` assertions have no counterpart because the credential never reaches
these functions in this port; everything else is asserted unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pi_agent.harness.compaction.compaction import CompactionSettings
from pi_agent.harness.compaction.utils import create_file_ops
from pi_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    Usage,
    UserMessage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_coding_agent.core.compaction import (
    CompactionPreparation,
    compact,
    generate_summary,
    generate_summary_with_usage,
)


def create_model(reasoning: bool, max_tokens: int = 8192) -> Model:
    return Model(
        id="reasoning-model" if reasoning else "non-reasoning-model",
        name="Reasoning Model" if reasoning else "Non-reasoning Model",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=200000,
        max_tokens=max_tokens,
    )


def mock_summary_response() -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="## Goal\nTest summary")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(input=10, output=10, cache_read=0, cache_write=0, total_tokens=20, cost=Cost()),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def _replay_stream(message: AssistantMessage) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.push(StartEvent(partial=message))
    for index, block in enumerate(message.content):
        if block.type == "text":
            stream.push(TextStartEvent(content_index=index, partial=message))
            stream.push(TextDeltaEvent(content_index=index, delta=block.text, partial=message))
            stream.push(TextEndEvent(content_index=index, content=block.text, partial=message))
    stream.push(DoneEvent(reason=message.stop_reason, message=message))
    stream.end()
    return stream


@dataclass
class RecordingStreamFn:
    """Stand-in for the `completeSimple` mock: records every request's options."""

    options: list[SimpleStreamOptions] = field(default_factory=list)

    def __call__(self, model: Model, context: Any, options: SimpleStreamOptions | None = None) -> Any:
        assert options is not None
        self.options.append(options)
        return _replay_stream(mock_summary_response())


MESSAGES = [UserMessage(content=[TextContent(text="Summarize this.")], timestamp=int(time.time() * 1000))]


async def test_uses_the_provided_thinking_level_for_reasoning_capable_models() -> None:
    stream_fn = RecordingStreamFn()

    text, usage = await generate_summary_with_usage(
        MESSAGES,
        stream_fn,
        create_model(True),
        2000,
        None,
        None,
        None,
        "medium",
    )

    assert text == "## Goal\nTest summary"
    assert usage == mock_summary_response().usage

    assert len(stream_fn.options) == 1
    assert stream_fn.options[0].reasoning == "medium"
    # TS also asserts `apiKey: "test-key"` on the same options object. This port
    # threads credentials through the caller-supplied `stream_fn`, so no API key
    # ever reaches `generate_summary_with_usage` and there is nothing to assert.


async def test_preserves_the_string_result_from_generate_summary() -> None:
    stream_fn = RecordingStreamFn()

    assert await generate_summary(MESSAGES, stream_fn, create_model(False), 2000) == "## Goal\nTest summary"


async def test_uses_fresh_routing_sessions_without_prompt_caching() -> None:
    stream_fn = RecordingStreamFn()

    await generate_summary(MESSAGES, stream_fn, create_model(False), 2000)
    await generate_summary(MESSAGES, stream_fn, create_model(False), 2000)

    assert len(stream_fn.options) == 2
    assert all(options.cache_retention == "none" for options in stream_fn.options)

    session_ids = [options.session_id for options in stream_fn.options]
    assert session_ids[0] != session_ids[1]


async def test_does_not_set_reasoning_when_thinking_is_off() -> None:
    stream_fn = RecordingStreamFn()

    await generate_summary(MESSAGES, stream_fn, create_model(True), 2000, None, None, None, "off")

    assert len(stream_fn.options) == 1
    # TS asserts `not.toHaveProperty("reasoning")`; `SimpleStreamOptions` is a
    # dataclass whose `reasoning` field always exists, so `is None` is the
    # equivalent "was never set" claim. (The `apiKey` assertion has no counterpart
    # here -- see the note in the reasoning-level test above.)
    assert stream_fn.options[0].reasoning is None


async def test_does_not_set_reasoning_for_non_reasoning_models() -> None:
    stream_fn = RecordingStreamFn()

    await generate_summary(MESSAGES, stream_fn, create_model(False), 2000, None, None, None, "medium")

    assert len(stream_fn.options) == 1
    # TS's "does not set reasoning for non-reasoning models" also asserts
    # `toMatchObject({ apiKey: "test-key" })`, which has no counterpart here --
    # see the note in the reasoning-level test above. The `reasoning` claim
    # (`is None` standing in for `not.toHaveProperty("reasoning")`) is asserted
    # below.
    assert stream_fn.options[0].reasoning is None


async def test_clamps_compaction_summary_max_tokens_to_the_model_output_cap() -> None:
    stream_fn = RecordingStreamFn()
    preparation = CompactionPreparation(
        first_kept_entry_id="entry-keep",
        messages_to_summarize=MESSAGES,
        turn_prefix_messages=MESSAGES,
        is_split_turn=True,
        tokens_before=600000,
        previous_summary=None,
        file_ops=create_file_ops(),
        settings=CompactionSettings(enabled=True, reserve_tokens=500000, keep_recent_tokens=20000),
    )

    result = await compact(preparation, stream_fn, create_model(False, 128000))

    assert result.usage == Usage(input=20, output=20, cache_read=0, cache_write=0, total_tokens=40, cost=Cost())
    assert [options.max_tokens for options in stream_fn.options] == [128000, 128000]

"""Tests for `pi_coding_agent.client.transcript`, ported from
`packages/coding-agent/test/client/transcript.test.ts` and extended to cover
the reducer branches that test does not reach (thinking deltas, buffer
eviction on `item_finished`, and progress-only items in `select_transcript`).

The reducer is pure: wire values are plain dicts, so no filesystem, network or
environment access is involved.
"""

from __future__ import annotations

from typing import Any

from pi_coding_agent.client.transcript import (
    apply_transcript_progress,
    apply_transcript_snapshot,
    create_transcript_state,
    select_transcript,
)

MODEL = {"provider": "faux", "id": "faux-1"}


def assistant_item(text: str = "saved", item_id: str = "assistant-1") -> dict[str, Any]:
    return {
        "id": item_id,
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "status": "streaming",
        "model": MODEL,
        "timestamp": 1,
    }


def snapshot(revision: int, text: str = "saved") -> dict[str, Any]:
    return {
        "id": "session-1",
        "cwd": "/workspace",
        "createdAt": 1,
        "updatedAt": revision + 1,
        "phase": "turn",
        "model": MODEL,
        "thinkingLevel": "off",
        "attached": True,
        "locked": True,
        "revision": revision,
        "transcript": [assistant_item(text)],
        "queuedSteer": [],
        "queuedSteerCount": 0,
    }


def tool_call_snapshot(input_value: Any) -> dict[str, Any]:
    return {
        **snapshot(1),
        "transcript": [
            {
                "id": "assistant-1",
                "role": "assistant",
                "content": [{"type": "toolCall", "toolCallId": "call-1", "toolName": "bash", "input": input_value}],
                "status": "streaming",
                "model": MODEL,
                "timestamp": 1,
            }
        ],
    }


def delta(kind: str, value: str, content_index: int = 0, message_id: str = "assistant-1") -> dict[str, Any]:
    return {
        "type": "assistant_delta",
        "messageId": message_id,
        "contentIndex": content_index,
        "kind": kind,
        "delta": value,
    }


def test_projects_progress_without_mutating_the_authoritative_snapshot():
    state = create_transcript_state(snapshot(1))
    state = apply_transcript_progress(state, delta("text", " response"))

    assert state.snapshot["transcript"][0]["content"] == [{"type": "text", "text": "saved"}]
    assert select_transcript(state)[0]["content"] == [{"type": "text", "text": "saved response"}]


def test_create_transcript_state_deep_copies_the_snapshot():
    original = snapshot(1)
    state = create_transcript_state(original)

    original["transcript"][0]["content"][0]["text"] = "mutated"

    assert state.snapshot["transcript"][0]["content"][0]["text"] == "saved"


def test_applies_streamed_tool_call_argument_deltas():
    state = create_transcript_state(tool_call_snapshot(None))
    state = apply_transcript_progress(state, delta("toolCall", '{"command":'))

    assert select_transcript(state)[0]["content"][0]["input"] == '{"command":'

    state = apply_transcript_progress(
        state,
        {
            "type": "item_updated",
            "item": {
                "id": "assistant-1",
                "role": "assistant",
                "content": [{"type": "toolCall", "toolCallId": "call-1", "toolName": "bash", "input": None}],
                "status": "streaming",
                "model": MODEL,
                "timestamp": 1,
            },
        },
    )
    state = apply_transcript_progress(state, delta("toolCall", '"pwd"}'))

    assert select_transcript(state)[0]["content"][0]["input"] == {"command": "pwd"}


def test_appends_tool_call_deltas_to_a_partial_input_restored_from_a_snapshot():
    state = create_transcript_state(tool_call_snapshot('{"command":'))
    state = apply_transcript_progress(state, delta("toolCall", '"pwd"}'))

    assert select_transcript(state)[0]["content"][0]["input"] == {"command": "pwd"}


def test_tool_call_buffer_is_dropped_when_the_item_finishes():
    state = create_transcript_state(tool_call_snapshot(None))
    state = apply_transcript_progress(state, delta("toolCall", '{"a":1}'))
    assert state.tool_call_buffers == {"assistant-1:0": '{"a":1}'}

    finished = {
        "id": "assistant-1",
        "role": "assistant",
        "content": [{"type": "toolCall", "toolCallId": "call-1", "toolName": "bash", "input": {"a": 1}}],
        "status": "complete",
        "model": MODEL,
        "timestamp": 1,
    }
    state = apply_transcript_progress(state, {"type": "item_finished", "item": finished})

    assert state.tool_call_buffers == {}
    assert select_transcript(state)[0]["status"] == "complete"

    # A later delta for the same slot restarts from the item's own string input.
    state = apply_transcript_progress(state, delta("toolCall", "x"))
    assert select_transcript(state)[0]["content"][0]["input"] == "x"


def test_item_finished_keeps_buffers_belonging_to_other_items():
    state = create_transcript_state(tool_call_snapshot(None))
    state = apply_transcript_progress(state, delta("toolCall", '{"a":'))
    state = apply_transcript_progress(
        state,
        {
            "type": "item_started",
            "item": {
                "id": "assistant-2",
                "role": "assistant",
                "content": [{"type": "toolCall", "toolCallId": "call-2", "toolName": "bash", "input": None}],
                "status": "streaming",
                "model": MODEL,
                "timestamp": 2,
            },
        },
    )
    state = apply_transcript_progress(state, delta("toolCall", '{"b":', message_id="assistant-2"))
    assert set(state.tool_call_buffers) == {"assistant-1:0", "assistant-2:0"}

    state = apply_transcript_progress(
        state,
        {
            "type": "item_finished",
            "item": {
                "id": "assistant-2",
                "role": "assistant",
                "content": [{"type": "toolCall", "toolCallId": "call-2", "toolName": "bash", "input": None}],
                "status": "complete",
                "model": MODEL,
                "timestamp": 2,
            },
        },
    )

    assert set(state.tool_call_buffers) == {"assistant-1:0"}


def test_non_finite_json_literals_stay_raw_text():
    # `json.loads` accepts `NaN`/`Infinity`, which `JSON.parse` rejects, so the
    # buffer must stay a string prefix until it forms a real JSON value.
    state = create_transcript_state(tool_call_snapshot(None))
    state = apply_transcript_progress(state, delta("toolCall", "NaN"))
    assert select_transcript(state)[0]["content"][0]["input"] == "NaN"

    state = create_transcript_state(tool_call_snapshot(None))
    state = apply_transcript_progress(state, delta("toolCall", '{"a": Infinity}'))
    assert select_transcript(state)[0]["content"][0]["input"] == '{"a": Infinity}'


def test_parsed_tool_input_accepts_every_json_shape():
    for raw, expected in (
        ("null", None),
        ("true", True),
        ("12", 12),
        ("1.5", 1.5),
        ('"str"', "str"),
        ('[1, [2], {"k": null}]', [1, [2], {"k": None}]),
        ('{"nested": {"list": [true, "x"]}}', {"nested": {"list": [True, "x"]}}),
    ):
        state = create_transcript_state(tool_call_snapshot(None))
        state = apply_transcript_progress(state, delta("toolCall", raw))
        assert select_transcript(state)[0]["content"][0]["input"] == expected


def test_applies_thinking_deltas():
    thinking_snapshot = {
        **snapshot(1),
        "transcript": [
            {
                "id": "assistant-1",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "step"},
                    {"type": "text", "text": "answer"},
                ],
                "status": "streaming",
                "model": MODEL,
                "timestamp": 1,
            }
        ],
    }
    state = create_transcript_state(thinking_snapshot)
    state = apply_transcript_progress(state, delta("thinking", " one"))

    content = select_transcript(state)[0]["content"]
    assert content[0] == {"type": "thinking", "thinking": "step one"}
    assert content[1] == {"type": "text", "text": "answer"}


def test_delta_kind_mismatching_the_part_type_leaves_it_untouched():
    state = create_transcript_state(snapshot(1))
    state = apply_transcript_progress(state, delta("thinking", "ignored"))

    assert select_transcript(state)[0]["content"] == [{"type": "text", "text": "saved"}]


def test_delta_for_an_unknown_message_is_ignored():
    state = create_transcript_state(snapshot(1))
    updated = apply_transcript_progress(state, delta("text", "x", message_id="missing"))

    assert updated is state
    assert select_transcript(updated)[0]["content"] == [{"type": "text", "text": "saved"}]


def test_delta_for_a_non_assistant_item_is_ignored():
    state = create_transcript_state(snapshot(1))
    state = apply_transcript_progress(
        state,
        {
            "type": "item_started",
            "item": {
                "id": "tool-call-1",
                "role": "tool",
                "toolCallId": "call-1",
                "toolName": "bash",
                "input": {"command": "printf hi"},
                "content": [{"type": "text", "text": "hi"}],
                "status": "running",
                "isError": False,
                "timestamp": 2,
            },
        },
    )
    before = state
    state = apply_transcript_progress(state, delta("text", "x", message_id="tool-call-1"))

    assert state is before


def test_appends_transient_tool_progress_and_replaces_it_by_id():
    state = create_transcript_state(snapshot(1))
    state = apply_transcript_progress(
        state,
        {
            "type": "item_started",
            "item": {
                "id": "tool-call-1",
                "role": "tool",
                "toolCallId": "call-1",
                "toolName": "bash",
                "input": {"command": "printf hi"},
                "content": [],
                "status": "running",
                "isError": False,
                "timestamp": 2,
            },
        },
    )
    last = select_transcript(state)[-1]
    assert last["id"] == "tool-call-1"
    assert last["role"] == "tool"
    assert last["status"] == "running"
    assert last["content"] == []

    state = apply_transcript_progress(
        state,
        {
            "type": "item_updated",
            "item": {
                "id": "tool-call-1",
                "role": "tool",
                "toolCallId": "call-1",
                "toolName": "bash",
                "input": {"command": "printf hi"},
                "content": [{"type": "text", "text": "hi"}],
                "status": "running",
                "isError": False,
                "timestamp": 2,
            },
        },
    )

    transcript = select_transcript(state)
    assert len(transcript) == 2
    assert transcript[1]["role"] == "tool"
    assert transcript[1]["status"] == "running"
    assert transcript[1]["content"] == [{"type": "text", "text": "hi"}]
    # Updating an existing item must not append a second entry to the order.
    assert state.progress_order == ["tool-call-1"]


def test_progress_only_items_keep_their_arrival_order():
    state = create_transcript_state(snapshot(1))
    for item_id in ("tool-a", "tool-b", "tool-c"):
        state = apply_transcript_progress(
            state,
            {
                "type": "item_started",
                "item": {
                    "id": item_id,
                    "role": "tool",
                    "toolCallId": item_id,
                    "toolName": "bash",
                    "input": {},
                    "content": [],
                    "status": "running",
                    "isError": False,
                    "timestamp": 2,
                },
            },
        )

    assert [item["id"] for item in select_transcript(state)] == [
        "assistant-1",
        "tool-a",
        "tool-b",
        "tool-c",
    ]


def test_resets_revision_history_when_the_same_session_runtime_is_reacquired():
    state = create_transcript_state(snapshot(50, "old runtime"))
    state = create_transcript_state(snapshot(0, "new runtime"))

    assert state.snapshot["revision"] == 0
    assert select_transcript(state)[0]["content"] == [{"type": "text", "text": "new runtime"}]


def test_accepts_a_lower_revision_when_switching_to_a_different_session():
    state = create_transcript_state(snapshot(50, "old session"))
    state = apply_transcript_snapshot(state, {**snapshot(0, "new session"), "id": "session-2"})

    assert state.snapshot["id"] == "session-2"
    assert select_transcript(state)[0]["content"] == [{"type": "text", "text": "new session"}]


def test_equal_revision_for_the_same_session_replaces_the_state():
    state = create_transcript_state(snapshot(3, "first"))
    state = apply_transcript_progress(state, delta("text", " streamed"))
    state = apply_transcript_snapshot(state, snapshot(3, "second"))

    assert state.progress_items == {}
    assert select_transcript(state)[0]["content"] == [{"type": "text", "text": "second"}]


def test_renders_accepted_steering_messages_from_authoritative_queued_state():
    state = create_transcript_state(
        {
            **snapshot(2),
            "queuedSteerCount": 1,
            "queuedSteer": [
                {
                    "id": "user-steer",
                    "role": "user",
                    "content": [{"type": "text", "text": "adjust the approach"}],
                    "timestamp": 2,
                }
            ],
        }
    )

    last = select_transcript(state)[-1]
    assert last["role"] == "user"
    assert last["content"] == [{"type": "text", "text": "adjust the approach"}]


def test_queued_steer_already_present_in_the_transcript_is_not_duplicated():
    steer = {
        "id": "user-steer",
        "role": "user",
        "content": [{"type": "text", "text": "adjust"}],
        "timestamp": 2,
    }
    base = snapshot(2)
    state = create_transcript_state(
        {**base, "transcript": [*base["transcript"], steer], "queuedSteer": [steer], "queuedSteerCount": 1}
    )

    assert [item["id"] for item in select_transcript(state)] == ["assistant-1", "user-steer"]


def test_a_newer_snapshot_is_authoritative_and_stale_snapshots_are_ignored():
    state = create_transcript_state(snapshot(3, "new"))
    state = apply_transcript_progress(state, delta("text", " transient"))
    state = apply_transcript_snapshot(state, snapshot(4, "authoritative"))
    stale = apply_transcript_snapshot(state, snapshot(2, "stale"))

    assert stale is state
    assert stale.snapshot["revision"] == 4
    assert select_transcript(stale)[0]["content"] == [{"type": "text", "text": "authoritative"}]

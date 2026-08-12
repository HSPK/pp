"""Python port of `packages/coding-agent/test/session-manager/labels.test.ts`."""

from __future__ import annotations

import pytest
from pi_ai.types import AssistantMessage, Cost, TextContent, Usage, UserMessage
from pi_coding_agent.core.session_manager import LabelEntry, SessionManager


def _usage() -> Usage:
    return Usage(
        input=1,
        output=1,
        cache_read=0,
        cache_write=0,
        total_tokens=2,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )


def _assistant(text: str, timestamp: int) -> AssistantMessage:
    return AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="test",
        content=[TextContent(text=text)],
        usage=_usage(),
        stop_reason="stop",
        timestamp=timestamp,
    )


def test_sets_and_gets_labels():
    session = SessionManager.in_memory()
    msg_id = session.append_message(UserMessage(content="hello", timestamp=1))

    assert session.get_label(msg_id) is None

    label_id = session.append_label_change(msg_id, "checkpoint")
    assert session.get_label(msg_id) == "checkpoint"

    label_entry = next(e for e in session.get_entries() if e.type == "label")
    assert isinstance(label_entry, LabelEntry)
    assert label_entry.id == label_id
    assert label_entry.target_id == msg_id
    assert label_entry.label == "checkpoint"


def test_clears_labels_with_none():
    session = SessionManager.in_memory()
    msg_id = session.append_message(UserMessage(content="hello", timestamp=1))

    session.append_label_change(msg_id, "checkpoint")
    assert session.get_label(msg_id) == "checkpoint"

    session.append_label_change(msg_id, None)
    assert session.get_label(msg_id) is None


def test_last_label_wins():
    session = SessionManager.in_memory()
    msg_id = session.append_message(UserMessage(content="hello", timestamp=1))

    session.append_label_change(msg_id, "first")
    session.append_label_change(msg_id, "second")
    last_label_id = session.append_label_change(msg_id, "third")

    assert session.get_label(msg_id) == "third"

    last_label_entry = next(e for e in session.get_entries() if e.id == last_label_id)
    msg_node = next(n for n in session.get_tree() if n.entry.id == msg_id)
    assert msg_node.label_timestamp == last_label_entry.timestamp


def test_labels_are_included_in_tree_nodes():
    session = SessionManager.in_memory()
    msg1_id = session.append_message(UserMessage(content="hello", timestamp=1))
    msg2_id = session.append_message(_assistant("hi", 2))

    msg1_label_id = session.append_label_change(msg1_id, "start")
    msg2_label_id = session.append_label_change(msg2_id, "response")

    entries = session.get_entries()
    msg1_label_entry = next(e for e in entries if e.id == msg1_label_id)
    msg2_label_entry = next(e for e in entries if e.id == msg2_label_id)

    tree = session.get_tree()
    msg1_node = next(n for n in tree if n.entry.id == msg1_id)
    assert msg1_node.label == "start"
    assert msg1_node.label_timestamp == msg1_label_entry.timestamp

    msg2_node = next(n for n in msg1_node.children if n.entry.id == msg2_id)
    assert msg2_node.label == "response"
    assert msg2_node.label_timestamp == msg2_label_entry.timestamp


def test_labels_are_preserved_in_create_branched_session():
    session = SessionManager.in_memory()
    msg1_id = session.append_message(UserMessage(content="hello", timestamp=1))
    msg2_id = session.append_message(_assistant("hi", 2))

    msg1_label_id = session.append_label_change(msg1_id, "important")
    msg2_label_id = session.append_label_change(msg2_id, "also-important")
    original_entries = session.get_entries()
    msg1_label_entry = next(e for e in original_entries if e.id == msg1_label_id)
    msg2_label_entry = next(e for e in original_entries if e.id == msg2_label_id)

    session.create_branched_session(msg2_id)

    assert session.get_label(msg1_id) == "important"
    assert session.get_label(msg2_id) == "also-important"

    label_entries = [e for e in session.get_entries() if e.type == "label"]
    assert len(label_entries) == 2

    tree = session.get_tree()
    msg1_node = next(n for n in tree if n.entry.id == msg1_id)
    msg2_node = next(n for n in msg1_node.children if n.entry.id == msg2_id)
    assert msg1_node.label_timestamp == msg1_label_entry.timestamp
    assert msg2_node.label_timestamp == msg2_label_entry.timestamp


def test_rewires_children_of_removed_labels_when_forking():
    session = SessionManager.in_memory()
    msg1_id = session.append_message(UserMessage(content="hello", timestamp=1))
    session.append_label_change(msg1_id, "checkpoint")
    model_change_id = session.append_model_change("anthropic", "claude-test")
    msg2_id = session.append_message(UserMessage(content="followup", timestamp=2))

    session.create_branched_session(msg2_id)

    entry = session.get_entry(model_change_id)
    assert entry is not None
    assert entry.parent_id == msg1_id


def test_labels_not_on_path_are_not_preserved_in_create_branched_session():
    session = SessionManager.in_memory()
    msg1_id = session.append_message(UserMessage(content="hello", timestamp=1))
    msg2_id = session.append_message(_assistant("hi", 2))
    msg3_id = session.append_message(UserMessage(content="followup", timestamp=3))

    session.append_label_change(msg1_id, "first")
    session.append_label_change(msg2_id, "second")
    session.append_label_change(msg3_id, "third")

    session.create_branched_session(msg2_id)

    assert session.get_label(msg1_id) == "first"
    assert session.get_label(msg2_id) == "second"
    assert session.get_label(msg3_id) is None


def test_labels_are_not_included_in_build_session_context():
    session = SessionManager.in_memory()
    msg_id = session.append_message(UserMessage(content="hello", timestamp=1))
    session.append_label_change(msg_id, "checkpoint")

    ctx = session.build_session_context()
    assert len(ctx.messages) == 1
    assert ctx.messages[0].role == "user"


def test_throws_when_labeling_non_existent_entry():
    session = SessionManager.in_memory()

    with pytest.raises(ValueError, match="Entry non-existent not found"):
        session.append_label_change("non-existent", "label")

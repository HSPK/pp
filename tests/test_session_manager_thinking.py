"""Thinking blocks survive a session-file round trip.

Persisting an assistant message that contained a `ThinkingContent` raised
`AttributeError: 'ThinkingContent' object has no attribute 'signature'`, and
nothing caught it: `AgentSession._handle_agent_event` calls
`append_message()` directly on `message_end`. So every turn from a reasoning
model failed to save.

Both halves of the round trip were wrong, in opposite directions:

  * the writer read `block.signature`, but the field is `thinking_signature`
  * the reader passed `signature=` to `ThinkingContent(...)`, which has no
    such parameter, so it would have raised `TypeError` had anything reached it

The whole suite stayed green because no test ever put a `ThinkingContent`
through the session manager -- `ThinkingContent` appeared only in rendering
tests. These go through the real file, because that is the path that broke.
"""

from __future__ import annotations

import json
from pathlib import Path

from pi_ai.types import AssistantMessage, Cost, TextContent, ThinkingContent, Usage

from pi_coding_agent.core.session_manager import SessionManager, load_entries_from_file


def _usage() -> Usage:
    return Usage(input=1, output=1, cache_read=0, cache_write=0, total_tokens=2, cost=Cost())


def _assistant(content: list) -> AssistantMessage:
    return AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="claude-test",
        content=content,
        usage=_usage(),
        stop_reason="stop",
        timestamp=1,
    )


def test_appending_a_thinking_block_does_not_raise(tmp_path: Path) -> None:
    """The original crash: this call raised and the assistant turn was lost."""
    session = SessionManager.create(str(tmp_path), session_dir=str(tmp_path))
    session.append_message(_assistant([ThinkingContent(thinking="let me think"), TextContent(text="done")]))

    entries = load_entries_from_file(Path(session.get_session_file()))
    assert [entry.type for entry in entries if entry.type == "message"] == ["message"]


def test_thinking_signature_and_redacted_survive_the_round_trip(tmp_path: Path) -> None:
    """`thinking_signature` is not decoration.

    Anthropic rejects a conversation that replays thinking without the
    signature it issued, and for redacted thinking the signature carries the
    only copy of the reasoning. Dropping either breaks the *next* turn, not
    this one, so it would surface far from its cause.
    """
    session = SessionManager.create(str(tmp_path), session_dir=str(tmp_path))
    session.append_message(
        _assistant(
            [
                ThinkingContent(thinking="reasoning", thinking_signature="sig-abc", redacted=True),
                TextContent(text="answer"),
            ]
        )
    )

    entries = load_entries_from_file(Path(session.get_session_file()))
    message = next(entry.message for entry in entries if entry.type == "message")
    thinking = message.content[0]

    assert isinstance(thinking, ThinkingContent)
    assert thinking.thinking == "reasoning"
    assert thinking.thinking_signature == "sig-abc"
    assert thinking.redacted is True


def test_the_wire_key_is_camel_case(tmp_path: Path) -> None:
    """Pinned on the file itself, not just the round trip.

    A reader and writer that agree on the *wrong* key round-trip perfectly and
    still produce a session file no other pi implementation can read. The
    TypeScript `ThinkingContent` field is `thinkingSignature`, and it reaches
    the file through a plain `JSON.stringify`.
    """
    session = SessionManager.create(str(tmp_path), session_dir=str(tmp_path))
    session.append_message(_assistant([ThinkingContent(thinking="r", thinking_signature="sig-abc")]))

    lines = Path(session.get_session_file()).read_text(encoding="utf-8").splitlines()
    block = next(
        block
        for line in lines
        for block in json.loads(line).get("message", {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "thinking"
    )

    assert block["thinkingSignature"] == "sig-abc"
    assert "signature" not in block


def test_a_thinking_block_without_a_signature_round_trips(tmp_path: Path) -> None:
    """Most providers emit no signature at all; `None` must stay `None`."""
    session = SessionManager.create(str(tmp_path), session_dir=str(tmp_path))
    session.append_message(_assistant([ThinkingContent(thinking="plain")]))

    entries = load_entries_from_file(Path(session.get_session_file()))
    thinking = next(entry.message for entry in entries if entry.type == "message").content[0]

    assert thinking.thinking_signature is None
    assert thinking.redacted is None

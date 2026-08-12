"""State reduction and event subscription tests.

Python port of `packages/client/test/state.test.ts`.
"""

from __future__ import annotations

import asyncio

import pytest
from support import (
    MemoryByteServer,
    attach_session,
    base_server_snapshot,
    collect_requests,
    connect_client,
    session_snapshot,
)


async def _next_request(requests, predicate, attempts=1000):
    for _ in range(attempts):
        found = next((candidate for candidate in requests if predicate(candidate)), None)
        if found is not None:
            return found
        await asyncio.sleep(0)
    raise RuntimeError("Expected request was never sent")


@pytest.mark.asyncio
async def test_reduces_only_authoritative_snapshots_and_supports_unsubscribe():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = collect_requests(server)
    initial = session_snapshot("session-1", revision=1, phase="idle")
    handle = await attach_session(client, server, initial)
    observed = []
    progress_types = []
    unsubscribe = handle.subscribe(lambda snapshot: observed.append(snapshot["revision"]))
    unsubscribe_events = handle.on_event(lambda event: progress_types.append(event["type"]))

    server.send(
        {
            "type": "event",
            "event": {
                "type": "session_progress",
                "sessionId": "session-1",
                "progress": {
                    "type": "assistant_delta",
                    "messageId": "assistant-1",
                    "contentIndex": 0,
                    "kind": "text",
                    "delta": "hi",
                },
            },
        }
    )
    assert progress_types == ["session_progress"]
    assert handle.snapshot == initial

    prompting = asyncio.ensure_future(handle.prompt("hello"))
    await asyncio.sleep(0.01)
    assert handle.snapshot == initial
    prompt_request = await _next_request(requests, lambda r: r["request"]["command"] == "prompt")
    updated = session_snapshot("session-1", revision=2, phase="turn")
    server.send(
        {
            "type": "response",
            "id": prompt_request["id"],
            "ok": True,
            "result": {"command": "prompt", "session": updated},
        }
    )
    assert await asyncio.wait_for(prompting, timeout=5) == updated
    assert handle.snapshot == updated
    assert observed == [2]

    unsubscribe()
    unsubscribe_events()
    server.send(
        {"type": "event", "event": {"type": "session_snapshot", "snapshot": session_snapshot("session-1", revision=3)}}
    )
    assert observed == [2]


@pytest.mark.asyncio
async def test_keeps_session_leases_attached_across_server_metadata_snapshots():
    server = MemoryByteServer()
    client = await connect_client(server)
    handle = await attach_session(client, server, session_snapshot("session-1"))

    server.send(
        {
            "type": "event",
            "event": {
                "type": "server_snapshot",
                "snapshot": {
                    **base_server_snapshot,
                    "revision": 2,
                    "sessions": [{"id": "session-1", "createdAt": 1, "sessionName": "Named session"}],
                },
            },
        }
    )

    assert handle.attached is True


@pytest.mark.asyncio
async def test_does_not_let_delayed_command_response_replace_newer_event_snapshot():
    server = MemoryByteServer()
    client = await connect_client(server)
    initial = session_snapshot("session-1", revision=1, thinkingLevel="off")
    handle = await attach_session(client, server, initial)
    requests = collect_requests(server)
    changing = asyncio.ensure_future(handle.set_thinking("high"))
    request = await _next_request(requests, lambda r: r["request"]["command"] == "set_thinking")
    server.send(
        {
            "type": "event",
            "event": {
                "type": "session_snapshot",
                "snapshot": session_snapshot("session-1", revision=3, thinkingLevel="high"),
            },
        }
    )
    server.send(
        {
            "type": "response",
            "id": request["id"],
            "ok": True,
            "result": {
                "command": "set_thinking",
                "session": session_snapshot("session-1", revision=2, thinkingLevel="medium"),
            },
        }
    )

    await asyncio.wait_for(changing, timeout=5)
    assert handle.snapshot["revision"] == 3
    assert handle.snapshot["thinkingLevel"] == "high"


@pytest.mark.asyncio
async def test_does_not_let_attach_response_replace_newer_snapshot_from_reacquired_runtime():
    server = MemoryByteServer()
    client = await connect_client(server)
    server.send(
        {
            "type": "event",
            "event": {
                "type": "session_snapshot",
                "snapshot": session_snapshot("session-1", revision=10, attached=False),
            },
        }
    )

    def on_message(message):
        if message["type"] != "request" or message["request"]["command"] != "attach":
            return
        server.send(
            {
                "type": "event",
                "event": {
                    "type": "session_snapshot",
                    "snapshot": session_snapshot("session-1", revision=3, thinkingLevel="high"),
                },
            }
        )
        server.send(
            {
                "type": "response",
                "id": message["id"],
                "ok": True,
                "result": {
                    "command": "attach",
                    "session": session_snapshot("session-1", revision=2, thinkingLevel="medium"),
                },
            }
        )

    server.on_message(on_message)

    handle = await asyncio.wait_for(client.attach_session("session-1"), timeout=5)
    assert handle.snapshot["revision"] == 3
    assert handle.snapshot["thinkingLevel"] == "high"

"""Tests for `pi_coding_agent.client.remote_session`.

Ports the cases from `packages/coding-agent/test/client/remote-session.test.ts`,
`remote-session-lifecycle.test.ts`, and `remote-session-ownership.test.ts` in
the TypeScript "pi" monorepo onto `RemoteSession`, driven through the REAL
`pi_client.PiClient`/`SessionHandle` wire-protocol code against an in-memory
fake server (`_remote_session_support.MemoryByteServer`, itself a port of
`packages/client/test/support.ts`), so encoding/decoding and lease bookkeeping
are exercised for real -- only the transport (a socket) is faked. No test
performs real network or filesystem I/O, and every await is bounded.
"""

from __future__ import annotations

import asyncio

import pytest
from _remote_session_support import (
    MemoryByteServer,
    collect_requests,
    connect_client,
    next_request,
    open_remote_session,
    session_snapshot,
)
from pi_client import PiSessionOwnershipError
from pi_coding_agent.client.remote_session import CreateRemoteSessionOptions, RemoteSession, RemoteSessionOptions


async def _await_soon(awaitable: object) -> object:
    return await asyncio.wait_for(awaitable, timeout=2)  # type: ignore[arg-type]


async def _find_request_for(requests: list[dict], command: str, session_id: str) -> dict:
    """Waits for an already-collected request matching `command` for `session_id`."""
    for _ in range(2000):
        for request in requests:
            if request["request"]["command"] == command and request["request"].get("sessionId") == session_id:
                return request
        await asyncio.sleep(0)
    raise AssertionError(f"Missing {command} request for {session_id}")


class TestRemoteSessionOperations:
    async def test_progress_projects_onto_subscribers_without_mutating_snapshot(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(
            client,
            server,
            session_snapshot(
                "session-1",
                phase="turn",
                transcript=[
                    {
                        "id": "assistant-1",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hello"}],
                        "status": "streaming",
                        "model": {"provider": "faux", "id": "model"},
                        "timestamp": 1,
                    }
                ],
            ),
        )
        views: list[str] = []

        def on_state(state: object) -> None:
            item = state.transcript[0] if state.transcript else None  # type: ignore[attr-defined]
            if item is not None and item["role"] == "assistant" and item["content"][0]["type"] == "text":
                views.append(item["content"][0]["text"])

        remote_session.subscribe(on_state)

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
                        "delta": " world",
                    },
                },
            }
        )

        assert views == ["hello", "hello world"]
        assert remote_session.snapshot["transcript"][0]["content"][0]["text"] == "hello"

    async def test_becomes_unbound_and_can_reopen_after_session_removed(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1"))

        server.send({"type": "event", "event": {"type": "session_removed", "sessionId": "session-1"}})

        assert remote_session.id is None
        assert remote_session.snapshot is None
        assert remote_session.state.transcript == []
        assert remote_session.state.lifecycle.status == "unbound"

        requests = collect_requests(server)
        reopening = asyncio.ensure_future(remote_session.open_session("session-1"))
        request = await next_request(server, "attach")
        assert request["request"] == {"command": "attach", "sessionId": "session-1"}
        server.send(
            {
                "type": "response",
                "id": request["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-1", revision=2)},
            }
        )
        await _await_soon(reopening)
        assert remote_session.state.lifecycle.status == "ready"
        assert requests

    async def test_exposes_active_operation_while_prompting(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1"))
        lifecycles: list[str] = []

        def on_state(state: object) -> None:
            lifecycle = state.lifecycle  # type: ignore[attr-defined]
            lifecycles.append(f"busy:{lifecycle.operation}" if lifecycle.status == "busy" else lifecycle.status)

        remote_session.subscribe(on_state)

        prompting = asyncio.ensure_future(remote_session.submit("  first prompt  "))
        request = await next_request(server, "prompt")
        assert request["request"] == {"command": "prompt", "sessionId": "session-1", "text": "first prompt"}
        assert remote_session.operation == "submit"
        server.send(
            {
                "type": "response",
                "id": request["id"],
                "ok": True,
                "result": {"command": "prompt", "session": session_snapshot("session-1", revision=2, phase="turn")},
            }
        )
        await _await_soon(prompting)

        assert lifecycles == ["ready", "busy:submit", "busy:submit", "ready"]
        assert remote_session.state.lifecycle.status == "ready"

    async def test_steers_when_server_session_is_in_a_turn(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1", phase="turn"))

        steering = asyncio.ensure_future(remote_session.submit("adjust"))
        request = await next_request(server, "steer")
        assert request["request"] == {"command": "steer", "sessionId": "session-1", "text": "adjust"}
        server.send(
            {
                "type": "response",
                "id": request["id"],
                "ok": True,
                "result": {"command": "steer", "session": session_snapshot("session-1", revision=2, phase="turn")},
            }
        )
        await _await_soon(steering)

    async def test_aborts_while_a_prompt_response_is_pending(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1"))

        prompting = asyncio.ensure_future(remote_session.submit("hello"))
        prompt_request = await next_request(server, "prompt")
        server.send(
            {
                "type": "event",
                "event": {
                    "type": "session_snapshot",
                    "snapshot": session_snapshot("session-1", revision=2, phase="turn"),
                },
            }
        )

        aborting = asyncio.ensure_future(remote_session.abort())
        abort_request = await next_request(server, "abort")
        assert abort_request["request"] == {"command": "abort", "sessionId": "session-1"}
        assert remote_session.operation == "abort"
        server.send(
            {
                "type": "response",
                "id": prompt_request["id"],
                "ok": True,
                "result": {"command": "prompt", "session": session_snapshot("session-1", revision=3, phase="turn")},
            }
        )
        await _await_soon(prompting)
        assert remote_session.operation == "abort"
        server.send(
            {
                "type": "response",
                "id": abort_request["id"],
                "ok": True,
                "result": {"command": "abort", "session": session_snapshot("session-1", revision=4)},
            }
        )
        await _await_soon(aborting)
        assert remote_session.state.lifecycle.status == "ready"

    async def test_rejects_conflicting_operations_while_locally_busy(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1"))

        prompting = asyncio.ensure_future(remote_session.submit("hello"))
        request = await next_request(server, "prompt")
        with pytest.raises(RuntimeError, match="Remote session is busy with submit"):
            await remote_session.set_thinking("high")
        with pytest.raises(RuntimeError, match="Remote session is busy with submit"):
            await remote_session.open_session("session-2")
        server.send(
            {
                "type": "response",
                "id": request["id"],
                "ok": True,
                "result": {"command": "prompt", "session": session_snapshot("session-1", revision=2, phase="turn")},
            }
        )
        await _await_soon(prompting)

    async def test_reports_subscriber_failures_without_interrupting_others(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        listener_errors: list[Exception] = []
        remote_session = await open_remote_session(
            client,
            server,
            session_snapshot("session-1"),
            RemoteSessionOptions(on_listener_error=listener_errors.append),
        )

        def failing_listener(_state: object) -> None:
            raise RuntimeError("render failed")

        remote_session.subscribe(failing_listener)
        notified = False

        def ok_listener(_state: object) -> None:
            nonlocal notified
            notified = True

        remote_session.subscribe(ok_listener)

        assert [str(error) for error in listener_errors] == ["render failed"]
        assert notified is True


class TestRemoteSessionLifecycle:
    async def test_opens_replacement_before_detaching_current_session(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1"))

        opening = asyncio.ensure_future(remote_session.open_session("session-2"))
        attach_request = await next_request(server, "attach")
        detach_request_task = asyncio.ensure_future(next_request(server, "detach"))
        server.send(
            {
                "type": "response",
                "id": attach_request["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-2")},
            }
        )
        detach_request = await _await_soon(detach_request_task)
        assert detach_request["request"] == {"command": "detach", "sessionId": "session-1"}
        server.send(
            {
                "type": "response",
                "id": detach_request["id"],
                "ok": True,
                "result": {"command": "detach", "sessionId": "session-1"},
            }
        )
        await _await_soon(opening)

        assert remote_session.id == "session-2"

    async def test_rejects_mutation_while_replacement_attachment_pending(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1"))
        requests = collect_requests(server)

        opening = asyncio.ensure_future(remote_session.open_session("session-2"))
        attach_request = await next_request(server, "attach")
        with pytest.raises(RuntimeError, match="Remote session is busy with open"):
            await remote_session.submit("race")
        with pytest.raises(RuntimeError, match="Remote session is busy with open"):
            await remote_session.create_session(CreateRemoteSessionOptions(cwd="/other"))
        assert [request["request"] for request in requests] == [{"command": "attach", "sessionId": "session-2"}]

        detach_request_task = asyncio.ensure_future(next_request(server, "detach"))
        server.send(
            {
                "type": "response",
                "id": attach_request["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-2")},
            }
        )
        detach_request = await _await_soon(detach_request_task)
        server.send(
            {
                "type": "response",
                "id": detach_request["id"],
                "ok": True,
                "result": {"command": "detach", "sessionId": "session-1"},
            }
        )
        await _await_soon(opening)

    async def test_rolls_back_replacement_when_current_session_becomes_active(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1"))

        opening = asyncio.ensure_future(remote_session.open_session("session-2"))
        attach_request = await next_request(server, "attach")
        detach_request_task = asyncio.ensure_future(next_request(server, "detach"))
        server.send(
            {
                "type": "event",
                "event": {
                    "type": "session_snapshot",
                    "snapshot": session_snapshot("session-1", phase="turn", revision=2),
                },
            }
        )
        server.send(
            {
                "type": "response",
                "id": attach_request["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-2")},
            }
        )
        detach_request = await _await_soon(detach_request_task)
        assert detach_request["request"] == {"command": "detach", "sessionId": "session-2"}
        server.send(
            {
                "type": "response",
                "id": detach_request["id"],
                "ok": True,
                "result": {"command": "detach", "sessionId": "session-2"},
            }
        )

        with pytest.raises(RuntimeError, match="Cannot open a session while session is turn"):
            await _await_soon(opening)
        assert remote_session.id == "session-1"
        assert remote_session.state.lifecycle.status == "ready"

    async def test_dispose_awaits_attachment_cleanup_started_by_reconnect(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1"))
        client.disconnect("test reconnect")

        attach_request_task = asyncio.ensure_future(next_request(server, "attach"))
        reconnecting = asyncio.ensure_future(remote_session.reconnect())
        attach_request = await _await_soon(attach_request_task)
        disposing = remote_session.dispose()
        disposal_settled = False

        def _mark_settled(_task: asyncio.Task[None]) -> None:
            nonlocal disposal_settled
            disposal_settled = True

        disposing.add_done_callback(_mark_settled)

        with pytest.raises(RuntimeError, match="Remote session is disposed"):
            await _await_soon(reconnecting)

        detach_request_task = asyncio.ensure_future(next_request(server, "detach"))
        server.send(
            {
                "type": "response",
                "id": attach_request["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-1", revision=2)},
            }
        )
        detach_request = await _await_soon(detach_request_task)
        assert disposal_settled is False
        server.send(
            {
                "type": "response",
                "id": detach_request["id"],
                "ok": True,
                "result": {"command": "detach", "sessionId": "session-1"},
            }
        )

        await _await_soon(disposing)
        assert disposal_settled is True
        assert client.connected is True

    async def test_dispose_immediately_preempts_pending_work_and_awaits_attachment_cleanup(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)
        remote_session = await open_remote_session(client, server, session_snapshot("session-1"))
        states: list[str] = []

        def on_state(state: object) -> None:
            states.append(state.lifecycle.status)  # type: ignore[attr-defined]

        remote_session.subscribe(on_state)
        requests = collect_requests(server)

        opening = asyncio.ensure_future(remote_session.open_session("session-2"))
        attach_request = await next_request(server, "attach")
        disposing = remote_session.dispose()
        # TS reads the detach request synchronously right after `dispose()` because a JS async
        # function body runs up to its first `await`. A Python coroutine does nothing until the
        # event loop schedules it, so the request lands one tick later; everything else about the
        # ordering (detach for the current session is issued before the replacement attach
        # resolves) is asserted exactly as TS does.
        assert client.connected is True
        assert remote_session.state.lifecycle.status == "disposed"
        current_detach_request = await _find_request_for(requests, "detach", "session-1")

        with pytest.raises(RuntimeError, match="Remote session is disposed"):
            await _await_soon(opening)

        replacement_detach_task = asyncio.ensure_future(next_request(server, "detach"))
        server.send(
            {
                "type": "response",
                "id": attach_request["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-2")},
            }
        )
        server.send(
            {
                "type": "response",
                "id": current_detach_request["id"],
                "ok": True,
                "result": {"command": "detach", "sessionId": "session-1"},
            }
        )
        replacement_detach_request = await _await_soon(replacement_detach_task)
        assert replacement_detach_request["request"] == {"command": "detach", "sessionId": "session-2"}
        server.send(
            {
                "type": "response",
                "id": replacement_detach_request["id"],
                "ok": True,
                "result": {"command": "detach", "sessionId": "session-2"},
            }
        )
        await _await_soon(disposing)
        assert "disposed" in states
        with pytest.raises(RuntimeError, match="Remote session is disposed"):
            remote_session.subscribe(lambda _state: None)


class TestRemoteSessionOwnership:
    async def test_factory_open_disposal_awaits_detach_without_disconnecting_client(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)

        def respond_to_attach(message: dict) -> None:
            if message["type"] == "request" and message["request"]["command"] == "attach":
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": True,
                        "result": {"command": "attach", "session": session_snapshot("session-1")},
                    }
                )

        server.on_message(respond_to_attach)
        remote_session = await RemoteSession.open(client, "session-1")

        first_disposal = remote_session.dispose()
        second_disposal = remote_session.dispose()
        disposal_settled = False

        def _mark_settled(_task: asyncio.Task[None]) -> None:
            nonlocal disposal_settled
            disposal_settled = True

        first_disposal.add_done_callback(_mark_settled)

        assert second_disposal is first_disposal
        detach_request = await next_request(server, "detach")
        assert disposal_settled is False
        assert detach_request["request"] == {"command": "detach", "sessionId": "session-1"}
        server.send(
            {
                "type": "response",
                "id": detach_request["id"],
                "ok": True,
                "result": {"command": "detach", "sessionId": "session-1"},
            }
        )
        await _await_soon(first_disposal)
        assert remote_session.disposed is True
        assert client.connected is True

    async def test_rejects_exclusive_coordinator_while_direct_shared_lease_active(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)

        def respond(message: dict) -> None:
            if message["type"] != "request":
                return
            if message["request"]["command"] == "attach":
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": True,
                        "result": {"command": "attach", "session": session_snapshot("session-1")},
                    }
                )
            elif message["request"]["command"] == "detach":
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": True,
                        "result": {"command": "detach", "sessionId": "session-1"},
                    }
                )

        server.on_message(respond)
        direct_handle = await client.attach_session("session-1")
        requests = collect_requests(server)

        with pytest.raises(PiSessionOwnershipError):
            await RemoteSession.open(client, "session-1")

        assert requests == []
        assert direct_handle.active is True
        await direct_handle.detach()
        assert [request["request"]["command"] for request in requests] == ["detach"]

    async def test_factory_creates_a_session(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)

        def respond(message: dict) -> None:
            if message["type"] == "request" and message["request"]["command"] == "create":
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": True,
                        "result": {"command": "create", "session": session_snapshot("session-1")},
                    }
                )

        server.on_message(respond)
        remote_session = await RemoteSession.create(client, CreateRemoteSessionOptions(cwd="/workspace"))

        assert remote_session.id == "session-1"
        assert remote_session.state.lifecycle.status == "ready"

    async def test_disposal_reports_cleanup_failure_without_retaining_exclusive_ownership(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)

        def respond_to_attach(message: dict) -> None:
            if message["type"] == "request" and message["request"]["command"] == "attach":
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": True,
                        "result": {"command": "attach", "session": session_snapshot("session-1")},
                    }
                )

        server.on_message(respond_to_attach)
        remote_session = await RemoteSession.open(client, "session-1")
        detach_count = 0

        def respond_to_detach(message: dict) -> None:
            nonlocal detach_count
            if message["type"] != "request" or message["request"]["command"] != "detach":
                return
            detach_count += 1
            if detach_count == 1:
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": False,
                        "error": {"code": "invalid_request", "message": "no"},
                    }
                )
            else:
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": True,
                        "result": {"command": "detach", "sessionId": "session-1"},
                    }
                )

        server.on_message(respond_to_detach)

        with pytest.raises(Exception, match="no"):
            await _await_soon(remote_session.dispose())
        assert remote_session.disposed is True
        assert client.connected is True

        replacement = await RemoteSession.open(client, "session-1")
        assert replacement.state.lifecycle.status == "ready"
        assert detach_count == 2
        await replacement.dispose()

    async def test_multiple_sessions_borrow_one_client_independently(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)

        def respond(message: dict) -> None:
            if message["type"] != "request":
                return
            command = message["request"]["command"]
            if command == "attach":
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": True,
                        "result": {"command": "attach", "session": session_snapshot(message["request"]["sessionId"])},
                    }
                )
            elif command == "detach":
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": True,
                        "result": {"command": "detach", "sessionId": message["request"]["sessionId"]},
                    }
                )

        server.on_message(respond)
        first = await RemoteSession.open(client, "session-1")
        second = await RemoteSession.open(client, "session-2")
        await first.dispose()

        assert first.disposed is True
        assert second.disposed is False
        assert client.connected is True
        await second.dispose()

    async def test_client_first_disposal_treated_as_released(self) -> None:
        server = MemoryByteServer()
        client = await connect_client(server)

        def respond_to_attach(message: dict) -> None:
            if message["type"] == "request" and message["request"]["command"] == "attach":
                server.send(
                    {
                        "type": "response",
                        "id": message["id"],
                        "ok": True,
                        "result": {"command": "attach", "session": session_snapshot("session-1")},
                    }
                )

        server.on_message(respond_to_attach)
        remote_session = await RemoteSession.open(client, "session-1")
        requests = collect_requests(server)

        await client.dispose()
        await remote_session.dispose()

        assert remote_session.disposed is True
        assert requests == []

"""Port of `packages/server/test/conformance.test.ts`."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from conftest import wait
from pi_protocol import PROTOCOL_VERSION, encode_client_message, encode_frame
from pi_server.errors import InternalServerError, NotImplementedProtocolError, PiServerError
from pi_server.testing.service import Deferred, TestServerService


async def test_accepts_a_transport_fragmented_framed_cbor_hello(harness: Any) -> None:
    started = await harness.start_server()
    client = await harness.connect(started.server)
    response = client.next(lambda message: message["type"] == "hello")
    await wait(client.send_fragmented_message({"type": "hello", "version": PROTOCOL_VERSION}, 2))
    resolved = await wait(response)
    assert resolved["type"] == "hello"
    assert resolved["version"] == PROTOCOL_VERSION


async def test_enforces_version_and_exactly_one_first_message_hello(harness: Any) -> None:
    started = await harness.start_server()

    bad_version = await harness.connect(started.server)
    hello_response = await wait(bad_version.hello(PROTOCOL_VERSION + 1))
    assert hello_response["type"] == "hello_error"
    assert hello_response["error"]["code"] == "version"
    await wait(bad_version.wait_for_close())

    request_first = await harness.connect(started.server)
    first_error = request_first.next(lambda message: message["type"] == "hello_error")
    await wait(request_first.send_message({"type": "request", "id": "too-early", "request": {"command": "list"}}))
    resolved = await wait(first_error)
    assert resolved["type"] == "hello_error"
    assert resolved["error"]["code"] == "invalid_request"
    await wait(request_first.wait_for_close())

    duplicate = await harness.connect(started.server)
    hello = await wait(duplicate.hello())
    assert hello["type"] == "hello"
    duplicate_error = duplicate.next(lambda message: message["type"] == "hello_error")
    await wait(duplicate.send_message({"type": "hello", "version": PROTOCOL_VERSION}))
    resolved = await wait(duplicate_error)
    assert resolved["type"] == "hello_error"
    assert resolved["error"]["code"] == "invalid_request"
    await wait(duplicate.wait_for_close())


async def test_closes_connections_that_do_not_complete_hello_before_timeout(harness: Any) -> None:
    started = await harness.start_server(TestServerService(), handshake_timeout_ms=20)
    client = await harness.connect(started.server)
    await wait(client.wait_for_close())
    assert any(
        message["type"] == "hello_error" and message["error"]["code"] == "invalid_request"
        for message in client.messages
    )


async def test_keeps_the_handshake_timeout_active_until_server_hello_is_sent(harness: Any) -> None:
    service = TestServerService()
    delay = service.delay_next_list()
    started = await harness.start_server(service, handshake_timeout_ms=20)
    client = await harness.connect(started.server)
    await wait(client.send_message({"type": "hello", "version": PROTOCOL_VERSION}))
    await wait(delay.entered.future)
    await wait(client.wait_for_close())
    delay.release.resolve(None)
    assert any(
        message["type"] == "hello_error" and message["error"]["code"] == "invalid_request"
        for message in client.messages
    )


async def test_bounds_and_closes_malformed_or_oversized_frames(harness: Any) -> None:
    malformed_started = await harness.start_server()
    malformed = await harness.connect(malformed_started.server)
    malformed_error = malformed.next(lambda message: message["type"] == "hello_error")
    await wait(malformed.send_bytes(encode_frame(bytes([0xFF]))))
    resolved = await wait(malformed_error)
    assert resolved["type"] == "hello_error"
    assert resolved["error"]["code"] == "invalid_request"
    await wait(malformed.wait_for_close())

    bounded_started = await harness.start_server(TestServerService(), max_frame_length=128)
    oversized = await harness.connect(bounded_started.server)
    frame = bytearray(4 + 129)
    frame[3] = 129
    await wait(oversized.send_bytes(bytes(frame)))
    await wait(oversized.wait_for_close())
    assert not any(message["type"] == "hello" for message in oversized.messages)

    outbound_started = await harness.start_server(TestServerService(), max_frame_length=128)
    outbound = await harness.connect(outbound_started.server)
    await wait(outbound.send_message({"type": "hello", "version": PROTOCOL_VERSION}))
    await wait(outbound.wait_for_close())
    assert outbound.messages == []


async def test_catches_up_a_handshaking_client_after_a_concurrent_server_change(harness: Any) -> None:
    class RacingService(TestServerService):
        def __init__(self) -> None:
            super().__init__()
            self.entered = Deferred()
            self.release = Deferred()
            self.race = False

        async def list_sessions(self) -> list[dict[str, Any]]:
            sessions = await super().list_sessions()
            if not self.race:
                return sessions
            self.entered.resolve(None)
            await self.release.future
            return sessions

    service = RacingService()
    service.seed("shared")
    started = await harness.start_server(service)
    controller = await harness.connect(started.server)
    await wait(controller.hello())
    service.race = True
    joining = await harness.connect(started.server)
    hello_future = asyncio.ensure_future(joining.hello())
    await wait(service.entered.future)
    await wait(controller.request({"command": "attach", "sessionId": "shared"}))
    service.release.resolve(None)
    handshake = await wait(hello_future)
    assert handshake["type"] == "hello"
    catchup = await wait(
        joining.next(
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "server_snapshot"
                and message["event"]["snapshot"]["revision"] > handshake["snapshot"]["revision"]
            )
        )
    )
    assert catchup["type"] == "event"
    assert catchup["event"]["type"] == "server_snapshot"
    # `toMatchObject` on an array also pins its length.
    assert len(catchup["event"]["snapshot"]["sessions"]) == 1
    assert catchup["event"]["snapshot"]["sessions"][0]["id"] == "shared"
    assert catchup["event"]["snapshot"]["sessions"][0]["sessionName"] == "Session shared"


async def test_shares_request_event_attachment_and_disconnect_behavior(harness: Any) -> None:
    service = TestServerService()
    service.seed("first")
    service.seed("second")
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    hello = await wait(client.hello())
    assert [s["id"] for s in hello["snapshot"]["sessions"]] == ["first", "second"]

    listed = await wait(client.request({"command": "list"}))
    assert listed["ok"] is True
    assert [s["id"] for s in listed["result"]["sessions"]] == ["first", "second"]
    first_attach = await wait(client.request({"command": "attach", "sessionId": "first"}))
    assert first_attach["ok"] is True
    assert first_attach["result"]["session"]["id"] == "first"
    assert first_attach["result"]["session"]["attached"] is True
    second_attach = await wait(client.request({"command": "attach", "sessionId": "second"}))
    assert second_attach["ok"] is True
    assert second_attach["result"]["session"]["id"] == "second"
    assert second_attach["result"]["session"]["attached"] is True

    progress = {
        "type": "assistant_delta",
        "messageId": "assistant-1",
        "contentIndex": 0,
        "kind": "text",
        "delta": "hello",
    }
    progress_event = client.next(
        lambda message: message["type"] == "event" and message["event"]["type"] == "session_progress"
    )
    service.latest_runtime("first").emit_progress(progress)
    resolved = await wait(progress_event)
    assert resolved == {
        "type": "event",
        "event": {"type": "session_progress", "sessionId": "first", "progress": progress},
    }

    detached = await wait(client.request({"command": "detach", "sessionId": "first"}))
    assert detached["ok"] is True
    assert detached["result"] == {"command": "detach", "sessionId": "first"}
    assert service.latest_runtime("first").dispose_count == 1
    thinking_response = await wait(
        client.request({"command": "set_thinking", "sessionId": "second", "thinkingLevel": "high"})
    )
    assert thinking_response["ok"] is True
    assert thinking_response["result"]["session"]["id"] == "second"
    assert thinking_response["result"]["session"]["thinkingLevel"] == "high"

    second_runtime = service.latest_runtime("second")
    await wait(client.close())
    await wait(second_runtime.disposed.future)
    assert second_runtime.dispose_count == 1


async def test_disconnects_attached_clients_when_a_runtime_reports_a_terminal_error(harness: Any) -> None:
    service = TestServerService()
    service.seed("terminal")
    errors: list[Exception] = []
    started = await harness.start_server(service, on_error=lambda error: errors.append(error))
    client = await harness.connect(started.server)
    await wait(client.hello())
    await wait(client.request({"command": "attach", "sessionId": "terminal"}))
    runtime = service.latest_runtime("terminal")

    runtime.set_phase("turn")
    runtime.emit_error(PiServerError("session_locked", "lock ownership lost"))
    await wait(client.wait_for_close())
    await wait(runtime.disposed.future)
    assert runtime.dispose_count == 1
    assert "terminal" not in service.locked
    assert any(getattr(error, "code", None) == "session_locked" for error in errors)

    next_client = await harness.connect(started.server)
    await wait(next_client.hello())
    reattached = await wait(next_client.request({"command": "attach", "sessionId": "terminal"}))
    assert reattached["ok"] is True
    assert reattached["result"]["session"]["id"] == "terminal"
    assert service.latest_runtime("terminal") is not runtime


async def test_does_not_expose_unexpected_service_errors_to_clients(harness: Any) -> None:
    class FailingService(TestServerService):
        def __init__(self) -> None:
            super().__init__()
            self.list_count = 0

        async def list_sessions(self) -> list[dict[str, Any]]:
            self.list_count += 1
            if self.list_count > 1:
                raise RuntimeError("private service detail")
            return await super().list_sessions()

    errors: list[Exception] = []

    def on_error(error: Exception) -> None:
        errors.append(error)
        raise RuntimeError("observer failure")

    service = FailingService()
    started = await harness.start_server(service, on_error=on_error)
    client = await harness.connect(started.server)
    await wait(client.hello())
    response = await wait(client.request({"command": "list"}))
    assert response["ok"] is False
    assert response["error"]["code"] == "internal_error"
    assert response["error"]["message"] == "Internal server error"
    assert any("private service detail" in str(error) for error in errors)


async def test_keeps_not_implemented_stable(harness: Any) -> None:
    class IncompleteService(TestServerService):
        def __init__(self) -> None:
            super().__init__()
            self.list_count = 0

        async def list_sessions(self) -> list[dict[str, Any]]:
            self.list_count += 1
            if self.list_count > 1:
                raise NotImplementedProtocolError()
            return await super().list_sessions()

    started = await harness.start_server(IncompleteService())
    client = await harness.connect(started.server)
    await wait(client.hello())
    response = await wait(client.request({"command": "list"}))
    assert response["ok"] is False
    assert response["error"]["code"] == "not_implemented"
    assert response["error"]["message"] == "Operation is not implemented"


async def test_reports_wrapped_internal_causes_without_exposing_them(harness: Any) -> None:
    cause = RuntimeError("private storage detail")
    cause.__cause__ = RuntimeError("private root cause")

    class WrappedFailureService(TestServerService):
        def __init__(self) -> None:
            super().__init__()
            self.list_count = 0

        async def list_sessions(self) -> list[dict[str, Any]]:
            self.list_count += 1
            if self.list_count > 1:
                raise InternalServerError(cause)
            return await super().list_sessions()

    errors: list[Exception] = []
    started = await harness.start_server(WrappedFailureService(), on_error=lambda error: errors.append(error))
    client = await harness.connect(started.server)
    await wait(client.hello())
    response = await wait(client.request({"command": "list"}))
    assert response["ok"] is False
    assert response["error"]["code"] == "internal_error"
    assert response["error"]["message"] == "Internal server error"
    assert "private" not in repr(response)
    assert cause in errors
    assert not any(isinstance(error, InternalServerError) for error in errors)


async def test_can_respond_out_of_request_order_after_the_handshake(harness: Any) -> None:
    service = TestServerService()
    service.seed("first")
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())

    delay = service.delay_next_list()
    slow = asyncio.ensure_future(client.request({"command": "list"}, "slow"))
    await wait(delay.entered.future)
    fast = asyncio.ensure_future(client.request({"command": "attach", "sessionId": "first"}, "fast"))
    fast_result = await wait(fast)
    assert fast_result["ok"] is True
    assert fast_result["id"] == "fast"
    assert fast_result["result"]["command"] == "attach"
    assert not any(message["type"] == "response" and message["id"] == "slow" for message in client.messages)

    delay.release.resolve(None)
    slow_result = await wait(slow)
    assert slow_result["ok"] is True
    assert slow_result["id"] == "slow"
    assert slow_result["result"]["command"] == "list"
    response_ids = [
        message["id"]
        for message in client.messages
        if message["type"] == "response" and message["id"] in ("slow", "fast")
    ]
    assert response_ids == ["fast", "slow"]


async def test_gracefully_closes_connections_sessions_and_listener_resources(harness: Any) -> None:
    service = TestServerService()
    service.seed("first")
    started = await harness.start_server(service)
    socket_path = started.server.addresses[0]
    client = await harness.connect(started.server)
    await wait(client.hello())
    await wait(client.request({"command": "attach", "sessionId": "first"}))
    runtime = service.latest_runtime("first")
    client_closed = asyncio.ensure_future(client.wait_for_close())

    await wait(started.server.close())
    await wait(client_closed)
    assert runtime.dispose_count == 1
    assert started.server.addresses == []
    if socket_path:
        # `lexists` rather than `exists`: TypeScript asserts `lstat` rejects
        # with ENOENT, which a dangling symlink would not satisfy.
        assert not os.path.lexists(socket_path)
    await wait(started.server.close())


async def test_decodes_multiple_framed_requests_from_one_raw_chunk(harness: Any) -> None:
    started = await harness.start_server()
    client = await harness.connect(started.server)
    await wait(client.hello())
    first = encode_client_message({"type": "request", "id": "first", "request": {"command": "list"}})
    second = encode_client_message({"type": "request", "id": "second", "request": {"command": "list"}})
    combined = first + second
    first_response = client.next(lambda message: message["type"] == "response" and message["id"] == "first")
    second_response = client.next(lambda message: message["type"] == "response" and message["id"] == "second")
    await wait(client.send_bytes(combined))
    first_resolved = await wait(first_response)
    second_resolved = await wait(second_response)
    assert first_resolved["type"] == "response"
    assert first_resolved["id"] == "first"
    assert first_resolved["ok"] is True
    assert second_resolved["type"] == "response"
    assert second_resolved["id"] == "second"
    assert second_resolved["ok"] is True

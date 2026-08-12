"""Port of `packages/server/test/sessions.test.ts`."""

from __future__ import annotations

import asyncio
from typing import Any

from conftest import attach, wait
from pi_server.testing.service import TEST_MODEL, Deferred, TestServerService


class OrderedSnapshotService(TestServerService):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = Deferred()
        self.second_started = Deferred()
        self.first_release = Deferred()
        self.second_release = Deferred()
        self.controlled = False
        self.started_count = 0

    async def list_models(self) -> list[dict[str, Any]]:
        if not self.controlled:
            return await super().list_models()
        self.started_count += 1
        if self.started_count == 1:
            self.first_started.resolve(None)
            await self.first_release.future
        elif self.started_count == 2:
            self.second_started.resolve(None)
            await self.second_release.future
        return [TEST_MODEL]


async def test_serializes_server_snapshot_revisions(harness: Any) -> None:
    service = OrderedSnapshotService()
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())
    service.controlled = True
    message_index = len(client.messages)

    first_create = asyncio.ensure_future(client.request({"command": "create", "name": "first"}))
    await wait(service.first_started.future)
    second_create = asyncio.ensure_future(client.request({"command": "create", "name": "second"}))
    # Upstream does a single `await Promise.resolve()` here. One `sleep(0)` is
    # the literal translation but not the equivalent: a microtask tick drains
    # the whole queue, while a loop iteration only runs what is already ready,
    # and the second request needs several hops (write, read, dispatch) before
    # it could reach `list_models` at all. So yield repeatedly and assert the
    # invariant after every hop. Extra yields can only give a serialization bug
    # more chances to show itself; they cannot cause a false failure, because
    # the first create still holds the lock throughout.
    for _ in range(50):
        await asyncio.sleep(0)
        assert service.started_count == 1
        assert not service.second_started.future.done()

    service.first_release.resolve(None)
    await wait(service.second_started.future)
    service.second_release.resolve(None)
    await wait(asyncio.gather(first_create, second_create))
    await wait(
        client.next_from(
            message_index,
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "server_snapshot"
                and message["event"]["snapshot"]["revision"] == 2
            ),
        )
    )

    revisions = [
        message["event"]["snapshot"]["revision"]
        for message in client.messages[message_index:]
        if message["type"] == "event" and message["event"]["type"] == "server_snapshot"
    ]
    assert revisions == [1, 2]


async def test_creates_server_assigned_durable_ids_and_supports_list_attach_detach(harness: Any) -> None:
    started = await harness.start_server()
    client = await harness.connect(started.server)
    await wait(client.hello())
    created = await wait(client.request({"command": "create", "cwd": "/work", "name": "Created"}))
    assert created["ok"] is True
    created_id = created["result"]["session"]["id"]
    assert created["result"]["session"]["id"] == started.service.last_created_id
    assert created["result"]["session"]["cwd"] == "/work"
    assert created["result"]["session"]["name"] == "Created"
    assert created["result"]["session"]["attached"] is True
    assert created["result"]["session"]["locked"] is True

    listed = await wait(client.request({"command": "list"}))
    assert listed["result"]["sessions"] == [
        {
            "id": started.service.last_created_id,
            "createdAt": 1,
            "updatedAt": 1,
            "sessionName": "Created",
            "cwd": "/work",
        }
    ]
    detached = await wait(client.request({"command": "detach", "sessionId": created_id}))
    assert detached["ok"] is True
    assert detached["result"] == {"command": "detach", "sessionId": created_id}
    assert started.service.latest_runtime(created_id).dispose_count == 1
    detached_again = await wait(client.request({"command": "detach", "sessionId": created_id}))
    assert detached_again["ok"] is True
    assert detached_again["result"] == {"command": "detach", "sessionId": created_id}

    attached = await attach(client, created_id)
    assert attached["id"] == started.service.last_created_id
    assert len(started.service.runtimes[created_id]) == 2


async def test_preserves_backend_metadata_while_refreshing_live_session_metadata(harness: Any) -> None:
    class ExtendedMetadataService(TestServerService):
        async def list_sessions(self) -> list[dict[str, Any]]:
            listed = await super().list_sessions()
            return [{**metadata, "parentSessionId": "parent-1", "sessionName": "stale name"} for metadata in listed]

    service = ExtendedMetadataService()
    service.seed("session-1", "Live name")
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())
    await attach(client, "session-1")

    listed = await wait(client.request({"command": "list"}))
    assert listed["result"]["sessions"] == [
        {
            "id": "session-1",
            "createdAt": 1,
            "updatedAt": 1,
            "parentSessionId": "parent-1",
            "sessionName": "Live name",
            "cwd": "/tmp/pi-server-conformance",
        }
    ]


async def test_keeps_multiple_attachments_on_one_connection_independent(harness: Any) -> None:
    service = TestServerService()
    service.seed("first")
    service.seed("second")
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())
    await attach(client, "first")
    await attach(client, "second")

    await wait(client.request({"command": "detach", "sessionId": "first"}))
    assert service.latest_runtime("first").dispose_count == 1
    assert service.latest_runtime("second").dispose_count == 0
    response = await wait(client.request({"command": "set_thinking", "sessionId": "second", "thinkingLevel": "medium"}))
    assert response["ok"] is True
    assert response["result"]["session"]["id"] == "second"
    assert response["result"]["session"]["thinkingLevel"] == "medium"


async def test_broadcasts_full_snapshots_and_progress_only_to_attached_clients(harness: Any) -> None:
    service = TestServerService()
    service.seed()
    started = await harness.start_server(service)
    attached_client = await harness.connect(started.server)
    unattached_client = await harness.connect(started.server)
    await wait(attached_client.hello())
    await wait(unattached_client.hello())
    await attach(attached_client, "session-1")
    runtime = service.latest_runtime("session-1")
    progress = {
        "type": "assistant_delta",
        "messageId": "assistant-1",
        "contentIndex": 0,
        "kind": "text",
        "delta": "hello",
    }
    runtime.emit_progress(progress)
    progress_message = await wait(
        attached_client.next(
            lambda message: message["type"] == "event" and message["event"]["type"] == "session_progress"
        )
    )
    assert progress_message == {
        "type": "event",
        "event": {"type": "session_progress", "sessionId": "session-1", "progress": progress},
    }
    assert not any(
        message["type"] == "event" and message["event"]["type"] == "session_progress"
        for message in unattached_client.messages
    )

    message_count = len(attached_client.messages)
    runtime.emit_snapshot()
    expected_revision = runtime.snapshot()["revision"]
    snapshot_message = await wait(
        attached_client.next_from(
            message_count,
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["revision"] == expected_revision
            ),
        )
    )
    assert snapshot_message["event"]["snapshot"]["id"] == "session-1"
    assert snapshot_message["event"]["snapshot"]["attached"] is True
    assert snapshot_message["event"]["snapshot"]["locked"] is True
    assert not any(
        message["type"] == "event" and message["event"]["type"] == "session_snapshot"
        for message in unattached_client.messages
    )


async def test_allows_every_attached_client_to_control_a_singleton_live_runtime(harness: Any) -> None:
    service = TestServerService()
    service.seed()
    started = await harness.start_server(service)
    first = await harness.connect(started.server)
    second = await harness.connect(started.server)
    await wait(first.hello())
    await wait(second.hello())
    await attach(first, "session-1")
    second_list = await wait(second.request({"command": "list"}))
    assert second_list["result"]["sessions"] == [
        {
            "id": "session-1",
            "createdAt": 1,
            "updatedAt": 1,
            "sessionName": "Session session-1",
            "cwd": "/tmp/pi-server-conformance",
        }
    ]
    await attach(second, "session-1")
    assert len(service.runtimes["session-1"]) == 1

    model_response = await wait(
        second.request({"command": "set_model", "sessionId": "session-1", "model": {"provider": "test", "id": "large"}})
    )
    assert model_response["ok"] is True
    assert model_response["result"]["session"]["model"]["id"] == "large"
    await wait(
        first.next(
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["model"]["id"] == "large"
            )
        )
    )
    thinking_response = await wait(
        first.request({"command": "set_thinking", "sessionId": "session-1", "thinkingLevel": "high"})
    )
    assert thinking_response["ok"] is True
    assert thinking_response["result"]["session"]["thinkingLevel"] == "high"


async def test_does_not_queue_prompts_and_processes_steer_and_abort_while_pending(harness: Any) -> None:
    service = TestServerService()
    service.seed()
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())
    await attach(client, "session-1")

    prompt = asyncio.ensure_future(client.request({"command": "prompt", "sessionId": "session-1", "text": "first"}))
    await wait(
        client.next(
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["phase"] == "turn"
            )
        )
    )
    busy = await wait(client.request({"command": "prompt", "sessionId": "session-1", "text": "second"}))
    assert busy["ok"] is False
    assert busy["error"]["code"] == "busy"

    steer = await wait(client.request({"command": "steer", "sessionId": "session-1", "text": "adjust"}))
    assert steer["ok"] is True
    assert steer["result"]["command"] == "steer"
    assert [s.text for s in service.latest_runtime("session-1").steers] == ["adjust"]
    abort = await wait(client.request({"command": "abort", "sessionId": "session-1"}))
    assert abort["ok"] is True
    assert abort["result"]["command"] == "abort"
    prompt_result = await wait(prompt)
    assert prompt_result["ok"] is True
    assert prompt_result["result"]["command"] == "prompt"
    assert prompt_result["result"]["session"]["phase"] == "idle"


async def test_returns_operation_attachment_state_relative_to_the_requesting_connection(harness: Any) -> None:
    service = TestServerService()
    service.seed()
    started = await harness.start_server(service)
    first = await harness.connect(started.server)
    second = await harness.connect(started.server)
    await wait(first.hello())
    await wait(second.hello())
    await attach(first, "session-1")
    await attach(second, "session-1")

    prompt = asyncio.ensure_future(first.request({"command": "prompt", "sessionId": "session-1", "text": "hello"}))
    await wait(
        first.next(
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["phase"] == "turn"
            )
        )
    )
    await wait(first.request({"command": "detach", "sessionId": "session-1"}))
    service.latest_runtime("session-1").finish_prompt()

    result = await wait(prompt)
    assert result["ok"] is True
    assert result["result"]["command"] == "prompt"
    assert result["result"]["session"]["id"] == "session-1"
    assert result["result"]["session"]["attached"] is False


async def test_keeps_busy_work_alive_after_disconnect_and_disposes_when_idle(harness: Any) -> None:
    service = TestServerService()
    service.seed()
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())
    await attach(client, "session-1")
    prompt = asyncio.ensure_future(client.request({"command": "prompt", "sessionId": "session-1", "text": "survive"}))
    await wait(
        client.next(
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["phase"] == "turn"
            )
        )
    )
    runtime = service.latest_runtime("session-1")
    await wait(client.close())
    try:
        await wait(prompt)
    except TimeoutError:
        raise AssertionError("prompt request never settled after the client disconnected") from None
    except Exception:
        pass
    else:
        raise AssertionError("expected prompt request to be rejected on disconnect")
    assert runtime.dispose_count == 0
    runtime.finish_prompt()
    await wait(runtime.disposed.future)
    assert runtime.dispose_count == 1

    reconnect = await harness.connect(started.server)
    await wait(reconnect.hello())
    snapshot = await attach(reconnect, "session-1")
    assert len(snapshot["transcript"]) == 2
    assert snapshot["transcript"][1]["role"] == "assistant"
    assert snapshot["transcript"][1]["content"][0]["text"] == "reply:survive"


async def test_restores_persisted_sessions_lazily_after_a_server_restart(harness: Any) -> None:
    service = TestServerService()
    service.seed()
    first_started = await harness.start_server(service)
    first_client = await harness.connect(first_started.server)
    await wait(first_client.hello())
    await attach(first_client, "session-1")
    await wait(first_client.request({"command": "set_thinking", "sessionId": "session-1", "thinkingLevel": "high"}))
    await wait(first_client.close())
    await wait(first_started.server.close())

    second_started = await harness.start_server(service)
    assert len(service.runtimes["session-1"]) == 1
    second_client = await harness.connect(second_started.server)
    await wait(second_client.hello())
    restored = await attach(second_client, "session-1")
    assert restored["thinkingLevel"] == "high"
    assert len(service.runtimes["session-1"]) == 2


async def test_rejects_and_disposes_a_service_runtime_with_the_wrong_server_assigned_id(harness: Any) -> None:
    class WrongIdService(TestServerService):
        async def create_session(self, options: Any) -> Any:
            options.id = "wrong-id"
            return await super().create_session(options)

    service = WrongIdService()
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())
    response = await wait(client.request({"command": "create"}))
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_request"
    assert service.latest_runtime("wrong-id").dispose_count == 1


async def test_maps_service_lock_errors_and_rejects_control_from_unattached_clients(harness: Any) -> None:
    service = TestServerService()
    service.seed("locked")
    service.locked.add("locked")
    started = await harness.start_server(service)
    client = await harness.connect(started.server)
    await wait(client.hello())
    locked = await wait(client.request({"command": "attach", "sessionId": "locked"}))
    assert locked["ok"] is False
    assert locked["error"]["code"] == "session_locked"
    unattached = await wait(client.request({"command": "abort", "sessionId": "locked"}))
    assert unattached["ok"] is False
    assert unattached["error"]["code"] == "invalid_request"

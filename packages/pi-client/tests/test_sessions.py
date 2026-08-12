"""Session lease semantics tests.

Python port of `packages/client/test/sessions.test.ts`.
"""

from __future__ import annotations

import asyncio

import pytest
from pi_client import PiSessionDetachedError, PiSessionOwnershipError
from support import MemoryByteServer, connect_client, session_snapshot


@pytest.mark.asyncio
async def test_keeps_multiple_session_handles_independent_and_enforces_detach():
    server = MemoryByteServer()
    client = await connect_client(server)

    def on_message(message):
        if message["type"] != "request":
            return
        request = message["request"]
        if request["command"] == "attach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "attach", "session": session_snapshot(request["sessionId"])},
                }
            )
        if request["command"] == "detach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "detach", "sessionId": request["sessionId"]},
                }
            )

    server.on_message(on_message)

    first = await asyncio.wait_for(client.attach_session("session-1"), timeout=5)
    second = await asyncio.wait_for(client.attach_session("session-2"), timeout=5)
    assert first.attached is True
    assert second.attached is True
    await asyncio.wait_for(first.detach(), timeout=5)
    assert first.attached is False
    assert second.attached is True
    with pytest.raises(PiSessionDetachedError):
        await asyncio.wait_for(first.abort(), timeout=5)


@pytest.mark.asyncio
async def test_detaches_shared_session_only_after_final_lease_released():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = []

    def on_message(message):
        if message["type"] != "request":
            return
        requests.append(message["request"]["command"])
        if message["request"]["command"] == "attach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "attach", "session": session_snapshot("session-1")},
                }
            )
        if message["request"]["command"] == "detach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "detach", "sessionId": "session-1"},
                }
            )

    server.on_message(on_message)

    first = await asyncio.wait_for(client.attach_session("session-1"), timeout=5)
    second = await asyncio.wait_for(client.attach_session("session-1"), timeout=5)
    assert second is not first
    assert requests == ["attach"]

    await asyncio.wait_for(first.detach(), timeout=5)
    assert first.attached is False
    assert second.attached is True
    assert requests == ["attach"]

    await asyncio.wait_for(second.detach(), timeout=5)
    assert second.attached is False
    assert requests == ["attach", "detach"]


@pytest.mark.asyncio
async def test_enforces_exclusive_and_shared_lease_modes():
    server = MemoryByteServer()
    client = await connect_client(server)

    def on_message(message):
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
        if message["request"]["command"] == "detach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "detach", "sessionId": "session-1"},
                }
            )

    server.on_message(on_message)

    shared = await asyncio.wait_for(client.acquire_session("session-1", "shared"), timeout=5)
    with pytest.raises(PiSessionOwnershipError):
        await asyncio.wait_for(client.acquire_session("session-1", "exclusive"), timeout=5)
    await asyncio.wait_for(shared.dispose(), timeout=5)

    exclusive = await asyncio.wait_for(client.acquire_session("session-1", "exclusive"), timeout=5)
    with pytest.raises(PiSessionOwnershipError):
        await asyncio.wait_for(client.acquire_session("session-1", "shared"), timeout=5)
    async with exclusive:
        pass


@pytest.mark.asyncio
async def test_invalidated_leases_dispose_without_protocol_cleanup():
    server = MemoryByteServer()
    client = await connect_client(server)

    def on_message(message):
        if message["type"] != "request" or message["request"]["command"] != "attach":
            return
        server.send(
            {
                "type": "response",
                "id": message["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-1")},
            }
        )

    server.on_message(on_message)
    lease = await asyncio.wait_for(client.acquire_session("session-1", "exclusive"), timeout=5)

    client.disconnect()

    await asyncio.wait_for(lease.dispose(), timeout=5)
    assert lease.active is False


@pytest.mark.asyncio
async def test_rejects_commands_while_releasing_and_restores_explicit_detach_after_failure():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = []

    def on_message(message):
        if message["type"] != "request":
            return
        requests.append({"id": message["id"], "command": message["request"]["command"]})

    server.on_message(on_message)
    acquiring = asyncio.ensure_future(client.acquire_session("session-1", "exclusive"))
    await asyncio.sleep(0.02)
    attach_request = requests[-1]
    server.send(
        {
            "type": "response",
            "id": attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1")},
        }
    )
    lease = await asyncio.wait_for(acquiring, timeout=5)

    first_detach = asyncio.ensure_future(lease.detach())
    await asyncio.sleep(0.02)
    failed_detach_request = requests[-1]
    with pytest.raises(PiSessionDetachedError):
        await asyncio.wait_for(lease.abort(), timeout=5)
    server.send(
        {
            "type": "response",
            "id": failed_detach_request["id"],
            "ok": False,
            "error": {"code": "invalid_request", "message": "retry"},
        }
    )
    with pytest.raises(Exception, match="retry"):
        await asyncio.wait_for(first_detach, timeout=5)
    assert lease.active is True

    second_detach = asyncio.ensure_future(lease.detach())
    await asyncio.sleep(0.02)
    successful_detach_request = requests[-1]
    server.send(
        {
            "type": "response",
            "id": successful_detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-1"},
        }
    )
    await asyncio.wait_for(second_detach, timeout=5)
    assert lease.active is False


@pytest.mark.asyncio
async def test_serializes_reacquisition_behind_final_lease_detachment():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = []

    def on_message(message):
        if message["type"] != "request":
            return
        requests.append({"id": message["id"], "command": message["request"]["command"]})

    server.on_message(on_message)

    first_attachment = asyncio.ensure_future(client.attach_session("session-1"))
    await asyncio.sleep(0.02)
    first_attach_request = requests[-1]
    server.send(
        {
            "type": "response",
            "id": first_attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1")},
        }
    )
    first = await asyncio.wait_for(first_attachment, timeout=5)
    detaching = asyncio.ensure_future(first.detach())
    await asyncio.sleep(0.02)
    detach_request = requests[-1]
    reacquiring = asyncio.ensure_future(client.attach_session("session-1"))
    await asyncio.sleep(0.02)
    assert [r["command"] for r in requests] == ["attach", "detach"]

    server.send(
        {
            "type": "response",
            "id": detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-1"},
        }
    )
    await asyncio.wait_for(detaching, timeout=5)
    await asyncio.sleep(0.02)
    second_attach_request = requests[-1]
    assert second_attach_request["command"] == "attach"
    server.send(
        {
            "type": "response",
            "id": second_attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1", revision=2)},
        }
    )

    reacquired = await asyncio.wait_for(reacquiring, timeout=5)
    assert reacquired.attached is True


@pytest.mark.asyncio
async def test_accepts_lower_revision_after_detaching_and_reacquiring_same_session():
    server = MemoryByteServer()
    client = await connect_client(server)
    attach_count = {"value": 0}

    def on_message(message):
        if message["type"] != "request":
            return
        if message["request"]["command"] == "attach":
            revision = 10 if attach_count["value"] == 0 else 0
            attach_count["value"] += 1
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "attach", "session": session_snapshot("session-1", revision=revision)},
                }
            )
        if message["request"]["command"] == "detach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "detach", "sessionId": "session-1"},
                }
            )

    server.on_message(on_message)

    first = await asyncio.wait_for(client.attach_session("session-1"), timeout=5)
    assert first.snapshot["revision"] == 10
    await asyncio.wait_for(first.detach(), timeout=5)
    reopened = await asyncio.wait_for(client.attach_session("session-1"), timeout=5)
    assert reopened is not first
    assert reopened.snapshot["revision"] == 0

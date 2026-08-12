"""Client/server integration test: real `PiClient` against a real `PiServer`.

Unlike every other test in this package (which drive the server with a raw
wire-level test client, `pi_server.testing.client.ProtocolTestClient`), this
test proves the two ported packages actually interoperate: it starts a real
`pi_server.transports.unix` server, connects a real `pi_client.PiClient` to
it over a real Unix-domain socket in `tmp_path`, and exercises hello/version
negotiation, list, create, attach, prompt, a streamed progress event, abort,
detach, and a protocol error end to end.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pi_client import PiClient, PiClientOptions, PiServerError, create_unix_transport_factory
from pi_server.testing.service import TestServerService
from pi_server.transports.unix import UnixServerOptions, create_unix_server

TIMEOUT = 5.0


async def _wait(awaitable: Any, timeout: float = TIMEOUT) -> Any:
    return await asyncio.wait_for(awaitable, timeout=timeout)


@pytest.fixture
async def rig():
    # A Unix socket path is capped at 107 bytes, and pytest's `tmp_path`
    # (plus a worker id under `pytest-xdist`) overflows it, so the socket
    # lives in a short private temp directory instead.
    socket_root = tempfile.mkdtemp(prefix="pi-i-")
    path = str(Path(socket_root) / "i.sock")
    service = TestServerService()
    server = create_unix_server(service, UnixServerOptions(path=path))
    await _wait(server.start())
    client = PiClient(PiClientOptions(transport_factory=create_unix_transport_factory(path)))
    try:
        await _wait(client.connect())
        yield server, service, client
    finally:
        await _wait(client.dispose())
        await _wait(server.close())
        shutil.rmtree(socket_root, ignore_errors=True)


async def test_hello_version_negotiation_populates_the_initial_snapshot(rig: Any) -> None:
    _server, _service, client = rig
    assert client.connected is True
    snapshot = client.snapshot
    assert snapshot is not None
    assert snapshot["sessions"] == []
    assert snapshot["protocolVersion"] >= 1


async def test_list_returns_seeded_sessions(rig: Any) -> None:
    server, service, client = rig
    service.seed("seed-1")
    # `seed` mutates the fake's in-memory store directly, so re-list to observe it.
    sessions = await _wait(client.list_sessions())
    assert any(s["id"] == "seed-1" for s in sessions)
    assert server is not None


async def test_create_attach_prompt_progress_abort_detach(rig: Any) -> None:
    _server, service, client = rig

    handle = await _wait(client.create_session(name="integration"))
    assert handle.id
    assert handle.attached is True
    assert handle.snapshot is not None
    assert handle.snapshot["name"] == "integration"

    progress_events: list[dict[str, Any]] = []
    unsubscribe = handle.on_event(lambda event: progress_events.append(event))

    prompt_task = asyncio.ensure_future(handle.prompt("hello there"))

    async def _runtime_ready() -> Any:
        while True:
            try:
                return service.latest_runtime(handle.id)
            except RuntimeError:
                await asyncio.sleep(0.01)

    runtime = await _wait(_runtime_ready())

    async def _phase_is_turn() -> None:
        while runtime.get_phase() != "turn":
            await asyncio.sleep(0.01)

    await _wait(_phase_is_turn())

    delta = {
        "type": "assistant_delta",
        "messageId": "assistant-progress",
        "contentIndex": 0,
        "kind": "text",
        "delta": "partial reply",
    }
    runtime.emit_progress(delta)

    async def _progress_seen() -> None:
        while not any(event["type"] == "session_progress" and event["progress"] == delta for event in progress_events):
            await asyncio.sleep(0.01)

    await _wait(_progress_seen())

    aborted_session = await _wait(handle.abort())
    assert aborted_session is not None
    prompt_result = await _wait(prompt_task)
    assert prompt_result["phase"] == "idle"
    assert any(
        entry.get("status") == "aborted" for entry in prompt_result["transcript"] if entry["role"] == "assistant"
    )

    unsubscribe()
    await _wait(handle.detach())
    assert handle.attached is False


async def test_attaching_an_unknown_session_surfaces_a_protocol_error(rig: Any) -> None:
    _server, _service, client = rig
    with pytest.raises(PiServerError) as excinfo:
        await _wait(client.attach_session("does-not-exist"))
    assert excinfo.value.code == "not_found"

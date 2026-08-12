"""Request/response correlation tests.

Python port of `packages/client/test/requests.test.ts`.
"""

from __future__ import annotations

import asyncio

import pytest
from support import MemoryByteServer, collect_requests, connect_client, session_snapshot


@pytest.mark.asyncio
async def test_correlates_coalesced_out_of_order_responses():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = collect_requests(server)
    listed = asyncio.ensure_future(client.list_sessions())
    attached = asyncio.ensure_future(client.attach_session("session-1"))
    await asyncio.sleep(0.02)
    assert len(requests) == 2

    attach_request = next(r for r in requests if r["request"]["command"] == "attach")
    list_request = next(r for r in requests if r["request"]["command"] == "list")
    server.send_together(
        [
            {
                "type": "response",
                "id": attach_request["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-1")},
            },
            {"type": "response", "id": list_request["id"], "ok": True, "result": {"command": "list", "sessions": []}},
        ]
    )

    assert await asyncio.wait_for(listed, timeout=5) == []
    attached_handle = await asyncio.wait_for(attached, timeout=5)
    assert attached_handle.id == "session-1"
    assert attached_handle.attached is True


@pytest.mark.asyncio
async def test_rejects_mismatched_response_instead_of_leaving_request_pending():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = collect_requests(server)
    listed = asyncio.ensure_future(client.list_sessions())
    await asyncio.sleep(0.02)
    assert [r["request"]["command"] for r in requests] == ["list"]
    server.send(
        {
            "type": "response",
            "id": requests[0]["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1")},
        }
    )

    with pytest.raises(Exception, match="Response command attach does not match list"):
        await asyncio.wait_for(listed, timeout=5)
    assert client.connection_state == "disconnected"


@pytest.mark.asyncio
async def test_surfaces_typed_request_errors():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = collect_requests(server)
    attaching = asyncio.ensure_future(client.attach_session("locked"))
    await asyncio.sleep(0.02)
    request_id = requests[0]["id"] if requests else "missing"
    server.send(
        {
            "type": "response",
            "id": request_id,
            "ok": False,
            "error": {"code": "session_locked", "message": "Already attached"},
        }
    )

    with pytest.raises(Exception) as exc_info:
        await asyncio.wait_for(attaching, timeout=5)
    assert exc_info.value.code == "session_locked"

"""Python port of `packages/coding-agent/test/rpc-client-process-exit.test.ts`.

The TypeScript test spawns a real child `pi` process through
`RpcClient({cliPath})`, has the child `process.exit(43)` on its first stdin
chunk, and asserts the in-flight `getCommands()` promise rejects with
``Agent process exited (code=43 signal=null)``.

`packages/coding-agent/src/modes/rpc/rpc-client.ts` is not ported -- see the
module docstring of `pi_coding_agent.modes.rpc`. Only `jsonl.py` (the LF-only
framing) was taken from the legacy stdio RPC mode; the mode driver and its
client are superseded by the CBOR-over-Unix-socket
`pi_server`/`pi_protocol`/`pi_client` stack.

The *behaviour* the TypeScript test pins -- an in-flight request must reject
rather than hang forever when the peer goes away -- does have a counterpart in
the ported stack, so that is exercised below against a real Unix socket
instead of being skipped.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Iterator

import pytest
from pi_client import PiClient, PiClientOptions, create_unix_transport_factory
from pi_client.errors import PiDisconnectedError

RPC_CLIENT_NOT_PORTED = (
    "modes/rpc/rpc-client.ts is deliberately not ported: only jsonl.py framing "
    "was taken from the legacy stdio RPC mode (see "
    "pi_coding_agent.modes.rpc.__doc__), so there is no child-process spawning "
    "RpcClient whose stdin/exit lifecycle could be driven here."
)


@pytest.fixture
def socket_dir() -> Iterator[str]:
    # AF_UNIX paths cap at 107 bytes, so keep the prefix short rather than
    # using `tmp_path` (which already carries an xdist worker id).
    directory = tempfile.mkdtemp(prefix="pi-t-")
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.skip(reason=RPC_CLIENT_NOT_PORTED)
def test_rejects_an_in_flight_request_when_the_child_process_exits() -> None:
    r"""`await expect(client.getCommands()).rejects.toThrow(
    /Agent process exited \(code=43 signal=null\)/)`."""
    raise AssertionError("unreachable")


def test_in_flight_request_rejects_when_the_peer_goes_away(socket_dir: str) -> None:
    """Same guarantee, in the stack that actually got ported.

    A server that accepts the connection and then drops it without ever
    answering must surface as a rejection, not a hang -- exactly what the
    TypeScript test asserts for a child process that exits mid-request.
    """
    socket_path = f"{socket_dir}/s.sock"

    async def scenario() -> None:
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await reader.read(1)
            writer.close()

        server = await asyncio.start_unix_server(handle, path=socket_path)
        client = PiClient(PiClientOptions(transport_factory=create_unix_transport_factory(socket_path)))
        try:
            with pytest.raises(Exception) as excinfo:
                await asyncio.wait_for(client.connect(), timeout=5)
            assert not isinstance(excinfo.value, asyncio.TimeoutError)
            assert client.connected is False
        finally:
            await client.dispose()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_connect_fails_fast_when_no_peer_is_listening(socket_dir: str) -> None:
    """The degenerate case: nothing is listening at all."""
    socket_path = f"{socket_dir}/missing.sock"

    async def scenario() -> None:
        client = PiClient(PiClientOptions(transport_factory=create_unix_transport_factory(socket_path)))
        try:
            with pytest.raises(PiDisconnectedError) as excinfo:
                await asyncio.wait_for(client.connect(), timeout=5)
            assert "No such file or directory" in str(excinfo.value)
        finally:
            await client.dispose()

    asyncio.run(scenario())

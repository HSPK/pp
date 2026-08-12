"""Python port of `packages/coding-agent/test/rpc-client-clone.test.ts`.

The TypeScript test drives `RpcClient` from
`packages/coding-agent/src/modes/rpc/rpc-client.ts`: it stubs the private
`send`/`getData` members and asserts that `client.clone()` writes
`{type: "clone"}` down the stdio pipe and unwraps `response.data`.

That class is not ported. See the module docstring of
`pi_coding_agent.modes.rpc` (quoted in the skip reason below): only the
LF-only JSON-line framing (`jsonl.py`) was ported from the legacy stdio RPC
mode. The mode driver and its client are superseded by the CBOR-over-Unix-socket
`pi_server`/`pi_protocol`/`pi_client` stack, which has no `clone` command --
`pi_coding_agent.client.remote_session.RemoteSession` exposes no `clone()`, so
there is no equivalent behaviour to pin here.
"""

from __future__ import annotations

import json

import pytest
from pi_coding_agent.modes.rpc import iter_json_lines, serialize_json_line

RPC_CLIENT_NOT_PORTED = (
    "modes/rpc/rpc-client.ts is deliberately not ported: only jsonl.py "
    "framing was taken from the legacy stdio RPC mode (see "
    "pi_coding_agent.modes.rpc.__doc__). The socket stack that replaces it "
    "has no clone command, so RpcClient.clone() has no counterpart."
)


@pytest.mark.skip(reason=RPC_CLIENT_NOT_PORTED)
def test_sends_the_clone_rpc_command() -> None:
    """`expect(send).toHaveBeenCalledWith({type: "clone"})` and
    `expect(result).toEqual({cancelled: false})`."""
    raise AssertionError("unreachable")


def test_clone_command_framing_round_trips() -> None:
    """The one piece of the TypeScript test that does have a counterpart.

    `RpcClient.send` writes `serializeJsonLine(command)` to the child's stdin
    and the child reads it back with the same LF-only framing. That framing is
    ported, so pin it with the exact payload the TypeScript test asserts on.
    """
    line = serialize_json_line({"type": "clone"})
    assert line == '{"type":"clone"}\n'
    assert [json.loads(text) for text in iter_json_lines([line.encode()])] == [{"type": "clone"}]

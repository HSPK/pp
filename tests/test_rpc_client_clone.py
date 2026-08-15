"""Python port of `packages/coding-agent/test/rpc-client-clone.test.ts`.

The TypeScript test drives `RpcClient` from
`packages/coding-agent/src/modes/rpc/rpc-client.ts`: it stubs the private
`send`/`getData` members and asserts that `client.clone()` writes
`{type: "clone"}` down the stdio pipe and unwraps `response.data`.

The stdio RPC *mode* is ported, and so is the `clone` command -- see
`tests/suite/test_rpc_mode.py::test_clone_forks_at_the_leaf`, which pins the
agent side of exactly this exchange. What is not ported is `RpcClient`, the
host-*side* helper for driving a spawned agent: the client half belongs to
whatever application embeds pi, and the wire protocol is all it needs.

So the assertion below has no counterpart (there is no `client.clone()` to
call), while the framing it depends on does, and is pinned underneath.
"""

from __future__ import annotations

import json

import pytest

from pi_coding_agent.modes.rpc import iter_json_lines, serialize_json_line

RPC_CLIENT_NOT_PORTED = (
    "modes/rpc/rpc-client.ts is deliberately not ported: it is the host-side "
    "half of the protocol, which belongs to the embedding application. The "
    "agent side of the clone command is covered by "
    "tests/suite/test_rpc_mode.py::test_clone_forks_at_the_leaf."
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

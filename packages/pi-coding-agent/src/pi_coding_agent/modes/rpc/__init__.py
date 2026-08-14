"""Stdio JSON-line RPC mode.

Port of `packages/coding-agent/src/modes/rpc/`. A host process spawns
`pi --mode rpc`, writes one JSON command per line to stdin, and reads
responses and session events back as JSON lines on stdout.

Layout:

- `jsonl.py` -- strict LF-only framing, verified byte for byte against the
  TypeScript implementation, including the cases that make the framing subtle:
  U+2028/U+2029 inside payload strings must not create a record boundary, and a
  multi-byte character split across two reads must reassemble.
- `types.py` -- the command set, the two structured payloads, and the response
  constructors.
- `dispatcher.py` -- `RpcDispatcher`, all 34 commands. Takes an `output`
  callable rather than owning stdout, so it is drivable from a test.
- `ui_context.py` -- `ExtensionUIContext` over `extension_ui_request` /
  `extension_ui_response` line pairs.
- `rpc_mode.py` -- `run_rpc_mode`, the process-level driver.

`rpc-client.ts` (the host-side helper for *speaking* this protocol to a spawned
`pi`) is deliberately not ported: this package is the agent, and the client half
belongs to whatever host embeds it. The `pi_server`/`pi_protocol`/`pi_client`
stack remains the richer alternative for hosts that can speak a socket protocol
-- CBOR over a Unix socket with multi-client attach/detach and session leases --
but it is not a substitute for this one, because it requires the host to connect
to a server rather than simply spawn a subprocess and use its pipes.
"""

from __future__ import annotations

from .dispatcher import RpcDispatcher
from .jsonl import JsonlLineReader, iter_json_lines, serialize_json_line
from .rpc_mode import run_rpc_mode
from .types import RPC_COMMAND_TYPES, RpcSessionState, RpcSlashCommand, make_error, make_success
from .ui_context import RpcExtensionUIContext

__all__ = [
    "RPC_COMMAND_TYPES",
    "JsonlLineReader",
    "RpcDispatcher",
    "RpcExtensionUIContext",
    "RpcSessionState",
    "RpcSlashCommand",
    "iter_json_lines",
    "make_error",
    "make_success",
    "run_rpc_mode",
    "serialize_json_line",
]

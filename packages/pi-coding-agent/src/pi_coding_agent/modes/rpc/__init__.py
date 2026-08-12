"""Legacy stdio JSON-line RPC mode.

`packages/coding-agent/src/modes/rpc/rpc-mode.ts` (817 lines) and
`rpc-client.ts` (601 lines) implement a legacy stdio embedding protocol: the
coding agent takes over its own stdout, frames one JSON object per line, and
drives a single session for a host process that spawned it.

**What is ported here:** `jsonl.py`, the strict LF-only framing. It is verified
byte for byte against the TypeScript implementation, including the cases that
make the framing subtle: U+2028/U+2029 inside payload strings must not create a
record boundary, and a multi-byte character split across two reads must
reassemble.

**What is not ported:** the mode driver itself. It is superseded in this
monorepo by the `pi_server`/`pi_protocol`/`pi_client` stack — a CBOR over Unix
socket protocol with multi-client attach/detach, session leases and structured
transcript snapshots — which IS fully ported and is this package's real RPC
surface:

- `pi_coding_agent.core.agent_session_runtime` — the session runtime driving `pi_server`
- `pi_coding_agent.client.remote_session` — the client-side session handle
- `tests/test_agent_session_runtime.py` — the end-to-end test over a real socket

The driver additionally depends on the interactive mode's `ExtensionUIContext`
(dialogs, widgets, theme) and on `output-guard.ts`'s raw-stdout takeover,
neither of which this port implements. Port it only if an embedding host that
cannot speak the socket protocol actually needs it.
"""

from __future__ import annotations

from .jsonl import JsonlLineReader, iter_json_lines, serialize_json_line

__all__ = ["JsonlLineReader", "iter_json_lines", "serialize_json_line"]

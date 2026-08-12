# pi-server

Experimental. This package is under active development and may change or be removed without notice. Its APIs and behavior are not yet stable.

Server package for pi.

## Session server core

The package exports the `PiServer` session server.

```python
from typing import Any

from pi_server import CreateSessionOptions, PiServerService
from pi_server.transports.unix import UnixServerOptions, create_unix_server


class Service(PiServerService):
    async def list_sessions(self) -> list[dict[str, Any]]:
        return []

    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def create_session(self, options: CreateSessionOptions):
        raise NotImplementedError

    async def open_session(self, session_id: str):
        raise NotImplementedError


async def main() -> None:
    service = Service()
    server = create_unix_server(service, UnixServerOptions(path=".scratch/pi-server.sock"))
    await server.start()
```

`PiServer` composes transport listeners through the `PiServerListener` protocol. Each listener must complete any transport-specific authentication and authorization before passing a connection to `PiServer`. For example, a WebSocket listener can validate credentials during the HTTP upgrade, while the Unix listener relies on socket filesystem permissions. The Unix submodule exports the `create_unix_listener()` building block and `create_unix_server()` preset, keeping the common case concise without coupling the primary server to Unix sockets. The listener uses length-prefixed CBOR messages from `pi_protocol`.

This package does not provide a standalone CLI or coding-agent service. Applications supply the `PiServerService` implementation.

`PiServerService.list_sessions()` returns protocol `SessionMetadata`, not acquired runtime state. Services should map the durable fields their storage supports and may omit `updatedAt`, `parentSessionId`, `sessionName`, and `cwd`. `PiServer` refreshes available metadata from live snapshots without requiring stored sessions to fabricate phase, model, thinking-level, attachment, or lock values.

The real coding-agent service boundary lives in `pi_coding_agent.core.agent_session_runtime`. `pi_server` itself does not depend on `pi_coding_agent`.

## Transport testing

Custom transports can use `pi_server.testing` for deterministic protocol conformance tests. It exports `create_test_server()`, `TestServerService`, `TestSessionRuntime`, `ProtocolTestClient`, and the transport-neutral `WireChannel` contract. `connect_unix_test_client()` is provided for Unix transport tests.

```python
from pi_server.testing import TestServerService, create_test_server


async def main() -> None:
    service = TestServerService()
    server = create_test_server(service)
    await server.start()
    await server.close()
```

## `pi-ai` protocol bridge

`pi_ai` domain objects and `pi_protocol` wire DTOs remain independent. This package owns their boundary and exports `to_protocol_model_metadata()`, `to_protocol_assistant_message()`, `to_protocol_user_message()`, and `to_protocol_tool_result_message()`.

The adapters reject invalid tool inputs, identifiers, timestamps, and mismatched tool results; `to_protocol_tool_result_message()` requires the original `ToolCall` so it can verify the association and convert its arguments itself. Diagnostic details are explicitly sanitized. Closed `pi_ai` unions are mapped exhaustively. The protocol mirrors `pi_ai` vocabulary such as `toolCall` and `toolUse` where the semantics are identical. Protocol schemas enforce consistent lifecycle states, and tests encode adapter output through the runtime schemas so incompatible changes fail in the bridging package.

## Development

From the repository root:

```bash
uv sync --all-packages
uv run pytest packages/pi-server
uv run ruff check packages/pi-server
```

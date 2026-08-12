# RPC Mode

The TypeScript stdio JSONL RPC mode (`pi --mode rpc`) is **not available in this Python port**. The CLI accepts `--mode rpc` only to report the incompatibility and exit.

For programmatic control, use the ported socket stack instead:

- `pi_protocol` validates and frames CBOR messages.
- `pi_server` serves durable sessions over a Unix-domain socket.
- `pi_client` connects to the socket and manages session leases.
- `pi_coding_agent.core.agent_session_runtime.PiAgentSessionRuntimeService` adapts the real coding-agent session runtime to `pi_server`.
- `pi_coding_agent.client.remote_session.RemoteSession` is a higher-level single-session client wrapper.

If you are embedding pi in the same Python process, prefer the SDK in [sdk.md](sdk.md). If you need process isolation or multiple clients, use the socket stack documented here.

## Starting RPC Mode

Legacy stdio RPC is not ported:

```bash
uv run pp --mode rpc
# stderr: RPC mode is not ported; use the pi_server/pi_client socket stack instead.
```

The socket server is currently a library API, not an installed console script. Start it by composing `PiAgentSessionRuntimeService` with a Unix listener:

```python
import asyncio
from pathlib import Path

from pi_coding_agent.core.agent_session_runtime import PiAgentSessionRuntimeService
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_server.transports.unix.preset import create_unix_server
from pi_server.transports.unix.types import UnixServerOptions


async def main() -> None:
    cwd = Path.cwd()
    socket_path = cwd / ".pi" / "pi.sock"
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    agent_dir = get_agent_dir()
    model_runtime = await ModelRuntime.create(agent_dir=agent_dir)
    service = PiAgentSessionRuntimeService(
        agent_dir=agent_dir,
        default_cwd=str(cwd),
        model_runtime=model_runtime,
    )
    server = create_unix_server(service, UnixServerOptions(path=str(socket_path)))
    await server.start()
    try:
        await asyncio.Event().wait()
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Connect with `pi_client`:

```python
import asyncio
from pathlib import Path

from pi_client import PiClient, PiClientOptions, create_unix_transport_factory


async def main() -> None:
    socket_path = Path.cwd() / ".pi" / "pi.sock"
    client = await PiClient.open(PiClientOptions(transport_factory=create_unix_transport_factory(str(socket_path))))
    try:
        session = await client.create_session(cwd=str(Path.cwd()), name="demo")
        session.on_event(lambda event: print(event))
        await session.prompt("Say hello in one sentence.")
        await session.detach()
    finally:
        await client.dispose()


if __name__ == "__main__":
    asyncio.run(main())
```

Common CLI commands in this Python port:

- `uv run pp "prompt"` or `uv run pp -p "prompt"`: single-shot print mode.
- `uv run pp --mode json "prompt"`: newline-delimited JSON event output for one local process.
- `uv run pp` on a TTY: interactive mode.
- `uv run pp --mode rpc`: not ported; use the socket stack above.

## Protocol Overview

The socket protocol is not JSONL over stdin/stdout. It uses validated message dictionaries encoded as CBOR and framed over a byte transport.

- **Client handshake**: first frame is `{"type": "hello", "version": 1}`.
- **Server handshake**: server responds with `{"type": "hello", "version": 1, "connectionId": "...", "snapshot": ...}` or `hello_error`.
- **Requests**: `{"type": "request", "id": "request-1", "request": {...}}`.
- **Responses**: `{"type": "response", "id": "request-1", "ok": true, "result": {...}}` or `ok: false` with a protocol error.
- **Events**: `{"type": "event", "event": {...}}`.

`PiClient` hides the envelope details and exposes `list_sessions()`, `create_session()`, `attach_session()`, `acquire_session()`, and `SessionHandle` methods.

### Framing

Each protocol message is:

1. a CBOR payload produced by `pi_protocol.encode_client_message()` or `encode_server_message()`;
2. prefixed by a 4-byte unsigned big-endian payload length.

The default maximum payload length is `pi_protocol.DEFAULT_MAX_FRAME_LENGTH` (16 MiB). `FrameDecoder`, `ClientMessageDecoder`, and `ServerMessageDecoder` incrementally decode arbitrary byte chunks.

`pi_coding_agent.modes.rpc.JsonlLineReader` is ported only as the strict LF-only helper used by tests for the unported legacy stdio mode. It is not a working RPC driver.

## Commands

The socket command names are defined by `pi_protocol.COMMAND_NAMES`:

- `list`
- `create`
- `attach`
- `detach`
- `prompt`
- `steer`
- `abort`
- `set_model`
- `set_thinking`

All command fields use the on-wire camelCase names in `pi_protocol.schemas`.

### Prompting

#### prompt

Send text to an attached session. The server rejects the command with `busy` if the session is not idle.

```json
{"command": "prompt", "sessionId": "session-id", "text": "Hello"}
```

`SessionHandle.prompt()` sends this command and returns the updated session snapshot:

```python
async def send_prompt(session: object) -> None:
    snapshot = await session.prompt("Hello")
    print(snapshot["phase"])
```

Images are not accepted by the socket protocol v1 `prompt` command. Use the in-process SDK if you need image input.

#### steer

Queue steering text while a prompt is running. The server rejects the command with `busy` if the session is idle.

```json
{"command": "steer", "sessionId": "session-id", "text": "Focus on tests first"}
```

```python
async def send_steer(session: object) -> None:
    await session.steer("Focus on tests first")
```

#### follow_up

Unavailable in the Python socket protocol. The only queued-input command is `steer`.

#### abort

Abort the active operation for an attached session.

```json
{"command": "abort", "sessionId": "session-id"}
```

```python
async def abort_session(session: object) -> None:
    await session.abort()
```

#### new_session

Legacy stdio `new_session` is unavailable. Use the socket `create` command or `PiClient.create_session()` to create a new durable session.

```json
{"command": "create", "cwd": "/path/to/project", "name": "my-session"}
```

```python
async def create_named_session(client: object) -> object:
    return await client.create_session(cwd="/path/to/project", name="my-session")
```

### State

#### get_state

Unavailable as a command. State is delivered as authoritative snapshots:

- the server handshake includes a `server_snapshot`;
- `attach`, `create`, `prompt`, `steer`, `abort`, `set_model`, and `set_thinking` return a `session` snapshot;
- `session_snapshot` events broadcast later changes.

A session snapshot contains:

```json
{
  "id": "session-id",
  "cwd": "/path/to/project",
  "phase": "idle",
  "model": {"provider": "anthropic", "id": "claude-sonnet-4-5"},
  "thinkingLevel": "medium",
  "attached": true,
  "locked": true,
  "revision": 3,
  "transcript": [],
  "queuedSteer": [],
  "queuedSteerCount": 0
}
```

`phase` is one of `idle`, `turn`, `compaction`, `branch_summary`, or `retry`.

#### get_messages

Unavailable as a command. Read `snapshot["transcript"]` from the session snapshot, or use `RemoteSession.state.transcript`.

### Model

#### set_model

Switch an idle attached session to a model reference.

```json
{"command": "set_model", "sessionId": "session-id", "model": {"provider": "openai", "id": "gpt-5.6-sol"}}
```

```python
async def select_model(session: object) -> None:
    await session.set_model({"provider": "openai", "id": "gpt-5.6-sol"})
```

#### cycle_model

Unavailable in the socket protocol. Choose a model client-side from `client.snapshot["models"]` and call `set_model`.

#### get_available_models

Unavailable as a command. The server snapshot contains `models`, each with authentication metadata.

```python
for model in client.snapshot["models"] if client.snapshot else []:
    print(model["provider"], model["id"], model["authenticated"])
```

### Thinking

#### set_thinking_level

The socket command is named `set_thinking` and uses `thinkingLevel`:

```json
{"command": "set_thinking", "sessionId": "session-id", "thinkingLevel": "high"}
```

```python
async def select_thinking(session: object) -> None:
    await session.set_thinking("high")
```

Levels are `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`.

#### cycle_thinking_level

Unavailable in the socket protocol. Choose a level client-side and call `set_thinking`.

#### get_available_thinking_levels

Unavailable as a command. Each model metadata entry includes `supportedThinkingLevels`.

### Queue Modes

#### set_steering_mode

Unavailable in the socket protocol. Queue-mode settings belong to the local `AgentSession`/settings layer.

#### set_follow_up_mode

Unavailable in the socket protocol. Follow-up queueing is not exposed by protocol v1.

### Compaction

#### compact

Unavailable in the socket protocol. Automatic compaction still runs inside the agent session when enabled.

#### set_auto_compaction

Unavailable in the socket protocol. Configure it through settings or the in-process SDK.

### Retry

#### set_auto_retry

Unavailable in the socket protocol. Configure it through settings or the in-process SDK.

#### abort_retry

Unavailable in the socket protocol. Use `abort` to stop the active session operation.

### Bash

#### bash

Unavailable as a direct socket command. The agent can still call the `bash` tool during a prompt when that tool is active.

#### abort_bash

Unavailable as a direct socket command. Use `abort` to cancel the active operation.

### Session

The socket stack supports durable sessions and multi-client attachment.

#### list

List persisted sessions plus currently live sessions:

```json
{"command": "list"}
```

```python
async def print_sessions(client: object) -> None:
    sessions = await client.list_sessions()
    for item in sessions:
        print(item["id"], item.get("sessionName"))
```

#### create

Create and exclusively attach a new session:

```python
async def create_feature_session(client: object) -> object:
    return await client.create_session(
        cwd="/path/to/project",
        name="feature-work",
        model={"provider": "openai", "id": "gpt-5.6-sol"},
        thinking_level="medium",
    )
```

#### attach / detach

Attach to an existing session with a shared lease, or explicitly acquire shared/exclusive leases:

```python
async def attach_and_detach(client: object) -> None:
    shared = await client.attach_session("session-id")
    exclusive = await client.acquire_session("session-id", "exclusive")
    await shared.detach()
    await exclusive.dispose()
```

#### get_session_stats

Unavailable in the socket protocol. Use `AgentSession.get_session_stats()` in the in-process SDK.

#### export_html

Unavailable in this Python port. `AgentSession.export_to_html()` raises `NotImplementedError`.

#### switch_session

Unavailable as a socket command. Attach to a different session id instead.

#### fork

Unavailable in the socket protocol. Use `AgentSessionRuntime.fork()` in-process.

#### clone

Unavailable in the socket protocol. Use `AgentSessionRuntime.fork(entry_id, position="at")` in-process.

#### get_fork_messages

Unavailable in the socket protocol. Use `AgentSession.get_user_messages_for_forking()` in-process.

#### get_entries

Unavailable in the socket protocol. Use `SessionManager.get_entries()` in-process.

#### get_tree

Unavailable in the socket protocol. Use `SessionManager.get_tree()` in-process.

#### get_last_assistant_text

Unavailable in the socket protocol. Use `AgentSession.get_last_assistant_text()` in-process.

#### set_session_name

Unavailable as a socket command after creation. Pass `name` to `create_session()` or set it in-process with `AgentSession.set_session_name()`.

### Commands

#### get_commands

Unavailable in the socket protocol. Slash command discovery is part of the in-process `AgentSession`/`ResourceLoader` path.

## Events

Events are server messages with `type: "event"`. They are delivered to every connection attached to the affected session, or to every ready connection for server-wide snapshots.

### Event Types

| Event | Description |
|-------|-------------|
| `server_snapshot` | Full server state: sessions, models, revision |
| `session_snapshot` | Full authoritative snapshot for one session |
| `session_progress` | Incremental transcript progress for one session |
| `session_removed` | A live session was removed or invalidated |

### server_snapshot

```json
{
  "type": "server_snapshot",
  "snapshot": {
    "serverId": "server-id",
    "protocolVersion": 1,
    "revision": 2,
    "sessions": [],
    "models": []
  }
}
```

### session_snapshot

```json
{
  "type": "session_snapshot",
  "snapshot": {
    "id": "session-id",
    "phase": "idle",
    "transcript": [],
    "queuedSteer": [],
    "queuedSteerCount": 0
  }
}
```

### session_progress

Progress updates normalize streaming agent activity. Snapshots remain authoritative.

```json
{
  "type": "session_progress",
  "sessionId": "session-id",
  "progress": {
    "type": "assistant_delta",
    "messageId": "item-2",
    "contentIndex": 0,
    "kind": "text",
    "delta": "Hello"
  }
}
```

Progress variants:

| Progress | Description |
|----------|-------------|
| `item_started` | A user, assistant, or tool transcript item started |
| `assistant_delta` | Streaming text, thinking, or tool-call argument fragment |
| `item_updated` | A running assistant or tool item changed |
| `item_finished` | An assistant or tool item completed, errored, or was aborted |

### session_removed

```json
{"type": "session_removed", "sessionId": "session-id"}
```

## Extension UI Protocol

The legacy stdio extension UI sub-protocol is not available in this Python port. Python extensions keep a headless-safe `ExtensionUIContext` subset for in-process use, but no socket-level UI dialog protocol is exposed.

### Extension UI Requests (stdout)

Unavailable; there is no stdout RPC driver.

#### select

Unavailable in socket protocol v1.

#### confirm

Unavailable in socket protocol v1.

#### input

Unavailable in socket protocol v1.

#### editor

Unavailable in this Python extension UI subset.

#### notify

Available only through the in-process extension UI context; not sent over the socket protocol.

#### setStatus

Available only as `set_status()` on the in-process extension UI context; not sent over the socket protocol.

#### setWidget

Unavailable in this Python extension UI subset.

#### setTitle

Available only as `set_title()` on the in-process extension UI context; not sent over the socket protocol.

#### set_editor_text

Unavailable in this Python extension UI subset.

### Extension UI Responses (stdin)

Unavailable; there is no stdin RPC driver.

#### Value response (select, input, editor)

Unavailable.

#### Confirmation response (confirm)

Unavailable.

#### Cancellation response (any dialog)

Unavailable.

## Error Handling

Failed requests return `ok: false` with a protocol error:

```json
{
  "type": "response",
  "id": "request-1",
  "ok": false,
  "error": {"code": "not_found", "message": "Unknown session: session-id"}
}
```

Error codes are:

- `version`
- `busy`
- `session_locked`
- `not_found`
- `invalid_request`
- `not_implemented`
- `internal_error`

`PiClient` raises `PiServerError` for server errors and `PiDisconnectedError` when the transport is unavailable.

## Types

Source files:

- [`packages/pi-protocol/src/pi_protocol/schemas.py`](../../pi-protocol/src/pi_protocol/schemas.py) - wire schemas and command/event names
- [`packages/pi-protocol/src/pi_protocol/codec.py`](../../pi-protocol/src/pi_protocol/codec.py) - validated CBOR message codec
- [`packages/pi-protocol/src/pi_protocol/framing.py`](../../pi-protocol/src/pi_protocol/framing.py) - length-prefixed frame codec
- [`packages/pi-client/src/pi_client/client.py`](../../pi-client/src/pi_client/client.py) - `PiClient`
- [`packages/pi-client/src/pi_client/session_handle.py`](../../pi-client/src/pi_client/session_handle.py) - `SessionHandle`
- [`packages/pi-server/src/pi_server/server.py`](../../pi-server/src/pi_server/server.py) - `PiServer`
- [`packages/pi-server/src/pi_server/types.py`](../../pi-server/src/pi_server/types.py) - `PiServerService` and `PiSessionRuntime`
- [`src/pi_coding_agent/core/agent_session_runtime.py`](../src/pi_coding_agent/core/agent_session_runtime.py) - real coding-agent socket adapter

### Model

Socket model metadata includes authentication state and supported thinking levels:

```json
{
  "provider": "openai",
  "id": "gpt-5.6-sol",
  "name": "GPT-5.6 Sol",
  "api": "openai-responses",
  "reasoning": true,
  "input": ["text", "image"],
  "contextWindow": 400000,
  "maxTokens": 128000,
  "cost": {"input": 1.25, "output": 10, "cacheRead": 0.125, "cacheWrite": 1.25},
  "supportedThinkingLevels": ["off", "minimal", "low", "medium", "high"],
  "authenticated": true
}
```

### UserMessage

Socket transcripts normalize user messages as transcript items:

```json
{
  "id": "item-1",
  "role": "user",
  "content": [{"type": "text", "text": "Hello"}],
  "timestamp": 1733234567890
}
```

### AssistantMessage

```json
{
  "id": "item-2",
  "role": "assistant",
  "status": "complete",
  "content": [{"type": "text", "text": "Hello!"}],
  "model": {"provider": "openai", "id": "gpt-5.6-sol"},
  "stopReason": "stop",
  "timestamp": 1733234567890
}
```

Assistant statuses are `streaming`, `complete`, `error`, and `aborted`.

### ToolResultMessage

Tool results are transcript items with role `tool`:

```json
{
  "id": "item-3",
  "role": "tool",
  "toolCallId": "call-1",
  "toolName": "bash",
  "input": {"command": "ls"},
  "content": [{"type": "text", "text": "README.md\n"}],
  "status": "complete",
  "isError": false,
  "timestamp": 1733234567890
}
```

### BashExecutionMessage

The socket protocol has no direct `bashExecution` transcript item. Direct bash execution exists only on the in-process `AgentSession.execute_bash()` API and in persisted local session history.

### Attachment

The socket protocol v1 has no attachment type. Use the in-process SDK for image content.

## Example: Basic Client (Python)

```python
import asyncio
from pathlib import Path

from pi_client import PiClient, PiClientOptions, create_unix_transport_factory


async def main() -> None:
    client = await PiClient.open(
        PiClientOptions(transport_factory=create_unix_transport_factory(str(Path.cwd() / ".pi" / "pi.sock")))
    )
    try:
        session = await client.create_session(cwd=str(Path.cwd()))

        def on_event(event: dict) -> None:
            if event["type"] == "session_progress":
                progress = event["progress"]
                if progress["type"] == "assistant_delta" and progress["kind"] == "text":
                    print(progress["delta"], end="", flush=True)

        session.on_event(on_event)
        await session.prompt("Hello")
        await session.detach()
    finally:
        await client.dispose()


if __name__ == "__main__":
    asyncio.run(main())
```

## Example: Interactive Client (Node.js)

The TypeScript Node.js stdio example does not apply to this Python port. Use `pi_client` from Python, or implement the `pi_protocol` CBOR framing and Unix-socket transport in your host language.

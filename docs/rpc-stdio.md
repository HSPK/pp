# Stdio RPC Mode

`pp --mode rpc` runs the agent headless behind a JSON protocol on stdin/stdout.
A host process spawns it, writes one JSON command per line, and reads responses
and events back as JSON lines. Use it to embed the agent in an editor, an IDE
plugin, or a custom UI.

This is the port of the TypeScript `--mode rpc`. For the alternative
Unix-socket protocol (multiple clients, durable session leases), see
[rpc.md](rpc.md). Choose stdio when your host can spawn a subprocess and wants
one session; choose the socket stack when several clients share sessions or the
agent must outlive its caller.

If you are embedding pi in the same Python process, skip both and use the SDK
in [sdk.md](sdk.md).

## Starting

```bash
pp --mode rpc [options]
```

Common options:

- `--provider <name>`: LLM provider (anthropic, openai, google, ...)
- `--model <pattern>`: model pattern or ID (`provider/id`, optional `:<thinking>`)
- `--name <name>` / `-n <name>`: session display name at startup
- `--no-session`: disable session persistence
- `--session-dir <path>`: custom session storage directory

`@file` arguments are rejected: they exist to prepend file contents to a
prompt, and in RPC mode prompts arrive as commands. Stdin is the command
channel, so it is never read as piped prompt text.

## Protocol Overview

- **Commands**: JSON objects on stdin, one per line
- **Responses**: JSON objects with `"type": "response"`, carrying `success` and
  either `data` or `error`
- **Events**: session events streamed to stdout as JSON lines

Every command takes an optional `id`; the matching response echoes it.
`bash_execution_update` events carry the `id` of the `bash` command that
produced them.

The process serves commands until stdin closes or an extension calls
`ctx.shutdown()`.

### Framing

Framing is strict JSONL with LF (`\n`) as the only record delimiter.

For clients:

- split records on `\n` only
- accept optional `\r\n` by stripping a trailing `\r`
- do not use a generic line reader that also splits on Unicode separators

`U+2028` and `U+2029` are valid inside JSON strings, so a reader that treats
them as line breaks will corrupt records. Node's `readline` has this problem;
so does Python's `str.splitlines()`. `pi_coding_agent.modes.rpc.JsonlLineReader`
is the compliant reader, and it also reassembles multi-byte characters split
across two reads.

### Null fields

Response fields that would be null are omitted rather than sent as `null`.
`get_tree` on a fresh session answers `{"tree": []}` where TypeScript answers
`{"tree": [], "leafId": null}`. This is the convention the whole port's JSON
output uses -- `--mode json` and print mode emit through the same encoder -- so
a client reading pi's JSON handles one shape everywhere. Treat a missing key as
null.

## Commands

### Prompting

#### prompt

```json
{"id": "1", "type": "prompt", "message": "List the files here", "images": [], "streamingBehavior": "steer"}
```

The success response is emitted as soon as the prompt passes preflight, not
when the turn finishes -- a host needs to know its input was accepted while the
model is still answering. The turn's output then arrives as events. If preflight
fails (no model, no credentials), the response is an error instead.

`streamingBehavior` (`"steer"` or `"followUp"`) decides how the prompt is queued
if one is already running.

#### steer / follow_up

```json
{"id": "2", "type": "steer", "message": "Focus on tests first"}
{"id": "3", "type": "follow_up", "message": "Then update the changelog"}
```

#### abort

```json
{"id": "4", "type": "abort"}
```

#### new_session

```json
{"id": "5", "type": "new_session", "parentSession": "/path/to/parent.jsonl"}
```

Answers `{"cancelled": false}`, or `{"cancelled": true}` if an extension vetoed
the replacement via `session_before_switch`.

### State

#### get_state

```json
{"id": "6", "type": "get_state"}
```

```json
{
  "id": "6", "type": "response", "command": "get_state", "success": true,
  "data": {
    "model": {"id": "...", "provider": "..."},
    "thinkingLevel": "off",
    "isStreaming": false,
    "isCompacting": false,
    "steeringMode": "all",
    "followUpMode": "all",
    "sessionFile": "/path/to/session.jsonl",
    "sessionId": "...",
    "sessionName": "triage",
    "autoCompactionEnabled": true,
    "messageCount": 4,
    "pendingMessageCount": 0
  }
}
```

### Model

```json
{"id": "7", "type": "set_model", "provider": "anthropic", "modelId": "claude-sonnet-4"}
{"id": "8", "type": "cycle_model"}
{"id": "9", "type": "get_available_models"}
```

`set_model` errors with `Model not found: <provider>/<modelId>` if the pair is
not in the available snapshot. `cycle_model` answers `data: null` when there is
nothing to cycle to.

### Thinking

```json
{"id": "10", "type": "set_thinking_level", "level": "medium"}
{"id": "11", "type": "cycle_thinking_level"}
{"id": "12", "type": "get_available_thinking_levels"}
```

### Queue modes

```json
{"id": "13", "type": "set_steering_mode", "mode": "one-at-a-time"}
{"id": "14", "type": "set_follow_up_mode", "mode": "all"}
```

### Compaction

```json
{"id": "15", "type": "compact", "customInstructions": "Keep the API decisions"}
{"id": "16", "type": "set_auto_compaction", "enabled": false}
```

### Retry

```json
{"id": "17", "type": "set_auto_retry", "enabled": false}
{"id": "18", "type": "abort_retry"}
```

### Bash

```json
{"id": "19", "type": "bash", "command": "git status", "excludeFromContext": false}
{"id": "20", "type": "abort_bash"}
```

Output streams as `bash_execution_update` events carrying the command's `id`,
and the response carries the final `BashResult`. An extension's `user_bash`
handler may return a result, in which case the command is recorded but not run
again here.

### Session

```json
{"id": "21", "type": "get_session_stats"}
{"id": "22", "type": "export_html", "outputPath": "/tmp/session.html"}
{"id": "23", "type": "switch_session", "sessionPath": "/path/to/other.jsonl"}
{"id": "24", "type": "fork", "entryId": "entry-id"}
{"id": "25", "type": "clone"}
{"id": "26", "type": "get_fork_messages"}
{"id": "27", "type": "get_entries", "since": "entry-id"}
{"id": "28", "type": "get_tree"}
{"id": "29", "type": "get_last_assistant_text"}
{"id": "30", "type": "set_session_name", "name": "triage"}
```

`get_entries` without `since` returns every entry; with `since` it returns only
what follows that entry, and errors with `Entry not found: <id>` if the marker
is unknown. `clone` forks at the current leaf and errors if there is none.
`set_session_name` trims the name and rejects a blank one.

### Messages

```json
{"id": "31", "type": "get_messages"}
```

### Commands

```json
{"id": "32", "type": "get_commands"}
```

Lists what the host can invoke by sending it as a prompt: extension commands,
prompt templates, and skills (as `skill:<name>`). Each entry carries `name`,
`description`, `source` (`"extension"`, `"prompt"` or `"skill"`) and
`sourceInfo`.

## Events

Session events are streamed as they occur, in the same shape `--mode json`
uses. See [json.md](json.md) for the event catalogue.

## Extension UI Protocol

An extension asking the user something has no terminal to draw on here, so the
request goes to the host.

### Requests (stdout)

```json
{"type": "extension_ui_request", "id": "uuid", "method": "select", "title": "Pick one", "options": ["a", "b"]}
{"type": "extension_ui_request", "id": "uuid", "method": "confirm", "title": "Delete?", "message": "Cannot be undone"}
{"type": "extension_ui_request", "id": "uuid", "method": "input", "title": "Name?", "placeholder": "my-session"}
{"type": "extension_ui_request", "id": "uuid", "method": "notify", "message": "Build finished", "notifyType": "info"}
{"type": "extension_ui_request", "id": "uuid", "method": "setStatus", "statusKey": "build", "statusText": "running"}
{"type": "extension_ui_request", "id": "uuid", "method": "setWidget", "widgetKey": "stats", "widgetLines": ["a"], "widgetPlacement": "aboveEditor"}
{"type": "extension_ui_request", "id": "uuid", "method": "setTitle", "title": "pi"}
```

`select`, `confirm` and `input` wait for an answer. `notify`, `setStatus`,
`setWidget` and `setTitle` are fire-and-forget.

A widget supplied as a component factory rather than plain lines is dropped:
building it needs a TUI that RPC mode does not have.

### Responses (stdin)

```json
{"type": "extension_ui_response", "id": "uuid", "value": "b"}
{"type": "extension_ui_response", "id": "uuid", "confirmed": true}
{"type": "extension_ui_response", "id": "uuid", "cancelled": true}
```

A response whose `id` matches nothing is dropped, so answering twice is safe. A
cancelled `select` or `input` resolves to `None`; a cancelled `confirm` resolves
to `False`. If stdin closes while a dialog is pending, it resolves the same way
rather than blocking shutdown.

### Extension errors

```json
{"type": "extension_error", "extensionPath": "/path/ext.py", "event": "session_start", "error": "..."}
```

## Error Handling

A failed command answers on the same `id`:

```json
{"id": "7", "type": "response", "command": "set_model", "success": false, "error": "Model not found: x/y"}
```

A line that is not valid JSON, or is JSON but not an object, answers with
`"command": "parse"` and no `id`. An unknown command name answers
`Unknown command: <name>`. None of these end the session -- one bad line does
not cost the host its agent.

## Example client

```python
import asyncio
import json


async def main() -> None:
    process = await asyncio.create_subprocess_exec(
        "pp",
        "--mode",
        "rpc",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )

    def send(command: dict) -> None:
        process.stdin.write((json.dumps(command) + "\n").encode())

    send({"id": "1", "type": "prompt", "message": "What files are here?"})

    async for raw in process.stdout:
        message = json.loads(raw)
        if message.get("type") == "response" and message.get("id") == "1":
            if not message["success"]:
                raise RuntimeError(message["error"])
        elif message.get("type") == "message_end":
            break

    process.stdin.close()
    await process.wait()


asyncio.run(main())
```

## Differences from the TypeScript mode

- Null-valued response fields are omitted rather than sent as `null` (see
  [Null fields](#null-fields)).
- The extension UI protocol carries the methods this port's
  `ExtensionUIContext` defines. TypeScript additionally emits `editor` and
  `set_editor_text`; those have no counterpart here.
- `rpc-client.ts`, the TypeScript host-side helper for driving a spawned agent,
  is not ported. The client half belongs to whatever host embeds the agent, and
  the protocol above is all it needs.

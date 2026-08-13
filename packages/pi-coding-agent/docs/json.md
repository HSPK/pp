# JSON Event Stream Mode

```bash
uv run --project /path/to/pp pp --mode json "Your prompt"
```

Outputs all session events as JSON lines to stdout. Useful for integrating pi into other tools or custom UIs.

The legacy stdio `--mode rpc` is not ported in Python; use the `pi_server` / `pi_client` Unix-socket stack instead.

## Event Types

Wire events are produced by `pi_coding_agent.modes.json_event.to_json_event()`. They match the session event objects after conversion through `utils.wire.to_wire()`, except that streaming message updates omit cumulative snapshots:

```typescript
type JsonAgentSessionEvent =
  | Exclude<AgentSessionEvent, { type: "message_update" }>
  | {
      type: "message_update";
      usage: Usage;
      assistantMessageEvent: AssistantMessageEventWithoutPartial;
    };
```

`message_update` emits only the delta event. If the internal assistant event has a `partial` field, JSON mode removes it so stream size stays linear.

Base events include agent lifecycle, turn lifecycle, message lifecycle, tool execution, bash execution updates, queue updates, and compaction events. The exact fields are the Python wire form of the corresponding dataclasses.

## Message Types

Messages are the wire form of the Python message dataclasses used by `pi_ai`, `pi_agent`, and `pi_coding_agent`, including:

- user messages
- assistant messages
- tool-result messages
- bash execution messages
- custom messages
- branch summary messages
- compaction summary messages

## Output Format

Each line is a JSON object. The first line is the session header when the session has one:

```json
{"type":"session","version":3,"id":"uuid","timestamp":"...","cwd":"/path"}
```

Followed by events as they occur:

```json
{"type":"agent_start"}
{"type":"turn_start"}
{"type":"message_start","message":{"role":"assistant","content":[]}}
{"type":"message_update","usage":{...},"assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":"Hello"}}
{"type":"message_end","message":{"role":"assistant","content":[]}}
{"type":"turn_end","message":{"role":"assistant","content":[]},"toolResults":[]}
{"type":"agent_end","messages":[]}
```

`message_update` records are delta-only. They omit both the cumulative `message` field and `assistantMessageEvent.partial` to keep stream size linear. The top-level `usage` field contains the latest cumulative provider-reported usage and may remain zero when a provider only reports usage at completion. Use `contentIndex` and `delta` to assemble live text, thinking, or tool-call arguments if needed. `message_end` contains the final authoritative message.

## Example

```bash
uv run --project /path/to/pp pp --mode json "List files" 2>/dev/null | jq -c 'select(.type == "message_end")'
```

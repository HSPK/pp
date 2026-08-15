# Session File Format

Sessions are stored as append-only JSONL (JSON Lines) files. Each line is a JSON object with a `type` field. Session entries form a tree via `id`/`parentId`, enabling in-place branching without rewriting earlier lines. `SessionManager.branch()` and `SessionManager.reset_leaf()` move the in-memory leaf pointer only; later appends create new children.

## File Location

```
~/.pi/agent/sessions/--<slugified-cwd>--/<timestamp>_<uuidv7>.jsonl
```

`<slugified-cwd>` is the resolved working directory with leading `/` removed and `/`, `\`, and `:` replaced by `-`. For `/home/me/project`, the default directory is:

```
~/.pi/agent/sessions/--home-me-project--/
```

The filename timestamp is the session header timestamp with `:` and `.` replaced by `-`, followed by the session uuidv7:

```
2026-08-12T10-28-43-797Z_019ff584-6a95-7611-be13-a1010206e6be.jsonl
```

## Deleting Sessions

Sessions can be removed by deleting their `.jsonl` files under `~/.pi/agent/sessions/`.

Pi also supports deleting sessions interactively from `/resume` (select a session and press `Ctrl+D`, then confirm). When available, pi uses the `trash` CLI to avoid permanent deletion.

## Session Version

Sessions have a version field in the header:

- **Version 1**: Linear entry sequence (legacy, auto-migrated on load)
- **Version 2**: Tree structure with `id`/`parentId` linking
- **Version 3**: Renamed message role `hookMessage` to `custom`

Existing sessions are automatically migrated to the current version (v3) when loaded through `SessionManager.open()` or `SessionManager.set_session_file()`. Migration rewrites the session file.

## Source Files

Python source:

- `packages/pi-coding-agent/src/pi_coding_agent/core/session_manager.py` - session entry types, migrations, JSONL serialization, and `SessionManager`
- `packages/pi-coding-agent/src/pi_coding_agent/core/compaction.py` - compaction and branch summarization entries
- `packages/pi-agent/src/pi_agent/harness/messages.py` - extended message types (`BashExecutionMessage`, `CustomMessage`, summary messages)
- `packages/pi-ai/src/pi_ai/types.py` - base message and content types

## Message Types

Session entries contain `HarnessMessage` objects. Understanding these types is essential for parsing sessions and writing extensions.

### Content Blocks

Messages contain arrays of typed content blocks. Python dataclass attributes use snake_case, but the persisted JSON uses the camelCase-compatible wire fields shown here.

```json
{ "type": "text", "text": "plain text" }
{ "type": "image", "data": "base64...", "mediaType": "image/png" }
{ "type": "thinking", "thinking": "reasoning text", "signature": "optional" }
{ "type": "toolCall", "id": "call_123", "name": "bash", "arguments": { "command": "pwd" } }
```

`ImageContent` is written with `mediaType`; the Python loader also accepts `mime_type` for compatibility.

### Base Message Types (from pi-ai)

```json
{
  "role": "user",
  "content": "Hello",
  "timestamp": 1786530523798
}
```

```json
{
  "role": "assistant",
  "api": "openai-completions",
  "provider": "faux",
  "model": "faux",
  "content": [{ "type": "text", "text": "Hi!" }],
  "usage": {
    "input": 1,
    "output": 2,
    "cacheRead": 0,
    "cacheWrite": 0,
    "cacheWrite1h": null,
    "reasoning": null,
    "totalTokens": 3,
    "cost": {
      "input": 0,
      "output": 0,
      "cacheRead": 0,
      "cacheWrite": 0,
      "total": 0
    }
  },
  "stopReason": "stop",
  "responseModel": null,
  "responseId": null,
  "diagnostics": null,
  "errorMessage": null,
  "rawStopReason": null,
  "endTurn": null,
  "timestamp": 1786530523798
}
```

```json
{
  "role": "toolResult",
  "toolCallId": "call_123",
  "toolName": "bash",
  "content": [{ "type": "text", "text": "output" }],
  "details": null,
  "usage": null,
  "addedToolNames": null,
  "isError": false,
  "timestamp": 1786530523798
}
```

`StopReason` includes `"pending"`, but terminal assistant messages should be persisted with a completion reason such as `"stop"`, `"length"`, `"toolUse"`, `"error"`, `"aborted"`, or `"deferred"`.

### Extended Message Types

```json
{
  "role": "bashExecution",
  "command": "ls",
  "output": "file.py\n",
  "exitCode": 0,
  "cancelled": false,
  "truncated": false,
  "fullOutputPath": null,
  "excludeFromContext": false,
  "timestamp": 1786530523798
}
```

```json
{
  "role": "custom",
  "customType": "my-extension",
  "content": "Injected context",
  "display": true,
  "details": { "source": "example" },
  "timestamp": 1786530523798
}
```

```json
{
  "role": "branchSummary",
  "summary": "Branch explored approach A...",
  "fromId": "a1b2c3d4",
  "timestamp": 1786530523798
}
```

```json
{
  "role": "compactionSummary",
  "summary": "Earlier work summary...",
  "tokensBefore": 50000,
  "timestamp": 1786530523798
}
```

### AgentMessage Union

Python uses `pi_agent.harness.messages.HarnessMessage` for the persisted message union:

- `pi_ai.types.UserMessage`
- `pi_ai.types.AssistantMessage`
- `pi_ai.types.ToolResultMessage`
- `pi_agent.harness.messages.BashExecutionMessage`
- `pi_agent.harness.messages.CustomMessage`
- `pi_agent.harness.session.context.BranchSummaryMessage`
- `pi_agent.harness.session.context.CompactionSummaryMessage`

## Entry Base

All entries except `SessionHeader` share these persisted fields:

```json
{
  "type": "message",
  "id": "a1b2c3d4",
  "parentId": null,
  "timestamp": "2026-08-12T10:28:43.798Z"
}
```

| Field | Description |
|-------|-------------|
| `type` | Entry discriminator |
| `id` | 8-character hex id; full UUID fallback after repeated collisions |
| `parentId` | Parent entry id, or `null` for a root entry |
| `timestamp` | ISO timestamp with millisecond precision and `Z` suffix |

The Python dataclasses expose `parent_id`, `model_id`, `first_kept_entry_id`, etc. The JSONL file uses `parentId`, `modelId`, `firstKeptEntryId`, and the other camelCase wire names documented here.

## Entry Types

### SessionHeader

First line of the file. Metadata only, not part of the tree and has no `id`/`parentId` tree relationship beyond its session id.

```json
{"type":"session","version":3,"id":"019ff584-6a95-7611-be13-a1010206e6be","timestamp":"2026-08-12T10:28:43.797Z","cwd":"/path/to/project"}
```

For sessions with a parent (created via `/fork`, `/clone`, or `new_session(NewSessionOptions(parent_session=...))`):

```json
{"type":"session","version":3,"id":"019ff584-6a95-7611-be13-a1010206e6be","timestamp":"2026-08-12T10:28:43.797Z","cwd":"/path/to/project","parentSession":"/path/to/original/session.jsonl"}
```

### SessionMessageEntry

A message in the conversation. The `message` field contains a `HarnessMessage`.

```json
{"type":"message","id":"a1b2c3d4","parentId":null,"timestamp":"2026-08-12T10:28:43.798Z","message":{"role":"user","content":"Hello","timestamp":1786530523798}}
```

```json
{"type":"message","id":"b2c3d4e5","parentId":"a1b2c3d4","timestamp":"2026-08-12T10:28:43.798Z","message":{"role":"assistant","api":"openai-completions","provider":"faux","model":"faux","content":[{"type":"text","text":"Hi!"}],"usage":{"input":1,"output":2,"cacheRead":0,"cacheWrite":0,"cacheWrite1h":null,"reasoning":null,"totalTokens":3,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop","timestamp":1786530523798}}
```

### ModelChangeEntry

Emitted when the user switches models mid-session.

```json
{"type":"model_change","id":"d4e5f6a7","parentId":"c3d4e5f6","timestamp":"2026-08-12T10:30:00.000Z","provider":"openai","modelId":"gpt-5.6-sol"}
```

### ThinkingLevelChangeEntry

Emitted when the user changes the thinking/reasoning level.

```json
{"type":"thinking_level_change","id":"e5f6a7b8","parentId":"d4e5f6a7","timestamp":"2026-08-12T10:31:00.000Z","thinkingLevel":"high"}
```

### CompactionEntry

Created when context is compacted. Stores a summary of earlier messages and the first retained entry id.

```json
{"type":"compaction","id":"f6a7b8c9","parentId":"e5f6a7b8","timestamp":"2026-08-12T10:35:00.000Z","summary":"User discussed X, Y, Z...","firstKeptEntryId":"c3d4e5f6","tokensBefore":50000,"details":{"readFiles":["src/app.py"],"modifiedFiles":["src/app.py"]}}
```

Optional fields:

- `usage`: LLM usage from generating the summary; included in session token and cost totals
- `details`: implementation-specific data. Default compaction writes `{ "readFiles": string[], "modifiedFiles": string[] }`
- `fromHook`: `true` if generated by an extension, `false`/absent if pi-generated

`retainedTail` is not part of `pi_coding_agent.core.session_manager`'s JSONL format. It exists in the lower-level `pi_agent` harness session codec, not in coding-agent session files.

### BranchSummaryEntry

Created when switching branches via `/tree` with a generated summary. It captures context from the branch being left and attaches it at the new leaf.

```json
{"type":"branch_summary","id":"a7b8c9d0","parentId":"a1b2c3d4","timestamp":"2026-08-12T10:40:00.000Z","fromId":"a1b2c3d4","summary":"Branch explored approach A...","details":{"readFiles":[],"modifiedFiles":[]}}
```

Optional fields:

- `usage`: LLM usage from generating the summary; included in session token and cost totals
- `details`: file tracking data (`{ "readFiles": string[], "modifiedFiles": string[] }`) for default summaries, or extension-specific data
- `fromHook`: `true` if generated by an extension, `false`/absent if pi-generated

In the Python `branch_with_summary()` implementation, `fromId` is written as the id where the summary is attached, or `"root"` when attached before any entry.

### CustomEntry

Extension state persistence. Does not participate in LLM context.

```json
{"type":"custom","id":"b8c9d0e1","parentId":"a7b8c9d0","timestamp":"2026-08-12T10:45:00.000Z","customType":"my-extension","data":{"count":42}}
```

Use `customType` to identify your extension's entries on reload. Entry renderer registration is not ported in Python; custom entries remain state only.

### CustomMessageEntry

Extension-injected messages that do participate in LLM context.

```json
{"type":"custom_message","id":"c9d0e1f2","parentId":"b8c9d0e1","timestamp":"2026-08-12T10:50:00.000Z","customType":"my-extension","content":"Injected context...","display":true,"details":{"source":"example"}}
```

Fields:

- `content`: string or `(TextContent | ImageContent)[]` shape
- `display`: `true` = show in TUI with distinct styling, `false` = hidden
- `details`: optional extension-specific metadata, not sent to LLM

### LabelEntry

User-defined bookmark/marker on an entry.

```json
{"type":"label","id":"d0e1f2a3","parentId":"c9d0e1f2","timestamp":"2026-08-12T10:55:00.000Z","targetId":"a1b2c3d4","label":"checkpoint-1"}
```

Set `label` to `null` to clear a label.

### SessionInfoEntry

Session metadata, currently the user-defined display name. Set via `/name`, `--name` / `-n`, or extension context `set_session_name()`.

```json
{"type":"session_info","id":"e1f2a3b4","parentId":"d0e1f2a3","timestamp":"2026-08-12T11:00:00.000Z","name":"Refactor auth module"}
```

The session name is displayed in the session selector (`/resume`) instead of the first message when set.

## Tree Structure

Entries form a tree:

- First entry has `parentId: null`.
- Each subsequent entry points to its parent via `parentId`.
- Branching creates new children from an earlier entry.
- The leaf is the current in-memory position in the tree.
- The leaf pointer is not a separate JSONL record; it is reconstructed as the last entry on load, then changed in memory by `branch()`/`reset_leaf()`.

```
[user msg] ─── [assistant] ─── [user msg] ─── [assistant] ─┬─ [user msg] ← current leaf
                                                            │
                                                            └─ [branch_summary] ─── [user msg] ← alternate branch
```

## Context Building

`build_context_entries()` walks from the current leaf to the root, then resolves the latest compaction on that path:

1. Build the active root-to-leaf path.
2. Find the latest `CompactionEntry` on that path.
3. If none exists, return the full path.
4. If one exists, return the compaction entry, then entries from `firstKeptEntryId` up to that compaction, then entries after the compaction.

`build_session_context()` builds on that entry list to produce the message list for the LLM:

1. Extract current model and thinking level from the full path.
2. Convert selected entries to messages:
   - `message` -> stored `HarnessMessage`
   - `compaction` -> `compactionSummary`
   - `branch_summary` -> `branchSummary`
   - `custom_message` -> `CustomMessage`
   - `custom`, `label`, `session_info`, `model_change`, `thinking_level_change` -> no direct context message

Because only the latest compaction on the active path is used, repeated compactions replace older compaction boundaries with the newest summary.

## Parsing Example

```python
import json
from pathlib import Path

lines = Path("session.jsonl").read_text(encoding="utf-8").strip().splitlines()

for line in lines:
    entry = json.loads(line)
    entry_type = entry.get("type")

    if entry_type == "session":
        print(f"Session v{entry.get('version', 1)}: {entry['id']}")
    elif entry_type == "message":
        message = entry["message"]
        print(f"[{entry['id']}] {message.get('role')}: {message.get('content')!r}")
    elif entry_type == "compaction":
        print(f"[{entry['id']}] Compaction: {entry['tokensBefore']} tokens summarized")
    elif entry_type == "branch_summary":
        print(f"[{entry['id']}] Branch summary attached at {entry['fromId']}")
    elif entry_type == "custom":
        print(f"[{entry['id']}] Custom ({entry['customType']}): {entry.get('data')!r}")
    elif entry_type == "custom_message":
        print(f"[{entry['id']}] Extension message ({entry['customType']}): {entry['content']!r}")
    elif entry_type == "label":
        print(f"[{entry['id']}] Label {entry.get('label')!r} on {entry['targetId']}")
    elif entry_type == "model_change":
        print(f"[{entry['id']}] Model: {entry['provider']}/{entry['modelId']}")
    elif entry_type == "thinking_level_change":
        print(f"[{entry['id']}] Thinking: {entry['thinkingLevel']}")
```

## SessionManager API

Key methods for working with sessions programmatically.

### Static Creation Methods

- `SessionManager.create(cwd, session_dir=None, options=None)` - new persisted session
- `SessionManager.open(path, session_dir=None, cwd_override=None)` - open existing session file
- `SessionManager.continue_recent(cwd, session_dir=None)` - continue most recent or create new
- `SessionManager.in_memory(cwd=None, options=None)` - no file persistence
- `SessionManager.fork_from(source_path, target_cwd, session_dir=None, options=None)` - fork session from another project

### Static Listing Methods

- `await SessionManager.list(cwd, session_dir=None, on_progress=None)` - list sessions for a directory
- `await SessionManager.list_all(session_dir=None, on_progress=None)` - list all sessions across projects

### Instance Methods - Session Management

- `new_session(options=None)` - start a new session
- `set_session_file(path)` - switch to a different session file
- `create_branched_session(leaf_id)` - extract branch to new session file

### Instance Methods - Appending

All append methods return the new entry id.

- `append_message(message)` - add a message
- `append_thinking_level_change(thinking_level)` - record thinking change
- `append_model_change(provider, model_id)` - record model change
- `append_compaction(summary, first_kept_entry_id, tokens_before, details=None, from_hook=None, usage=None)` - add compaction
- `append_custom_entry(custom_type, data=None)` - extension state, not in context
- `append_session_info(name)` - set session display name
- `append_custom_message_entry(custom_type, content, display, details=None)` - extension message, in context
- `append_label_change(target_id, label)` - set or clear label

### Instance Methods - Tree Navigation

- `get_leaf_id()` - current position
- `get_leaf_entry()` - current leaf entry
- `get_entry(entry_id)` - get entry by id
- `get_branch(from_id=None)` - walk from entry to root
- `get_tree()` - get full tree structure
- `get_children(parent_id)` - get direct children
- `get_label(entry_id)` - get label for entry
- `branch(branch_from_id)` - move leaf to earlier entry
- `reset_leaf()` - reset leaf to before any entries
- `branch_with_summary(branch_from_id, summary, details=None, from_hook=None, usage=None)` - attach branch summary and advance leaf to it

### Instance Methods - Context & Info

- `build_context_entries()` - get active branch entries with compaction applied
- `build_session_context()` - get messages, thinking level, and model for LLM
- `get_entries()` - all entries excluding header
- `get_header()` - session header metadata
- `get_session_name()` - latest session display name
- `get_cwd()` - working directory
- `get_session_dir()` - session storage directory
- `get_session_id()` - session uuid
- `get_session_file()` - session file path, or `None` for in-memory
- `is_persisted()` - whether session is saved to disk

### Programmatic Example

```python
from pi_ai.types import AssistantMessage, TextContent, Usage, UserMessage
from pi_coding_agent.core.session_manager import SessionManager

manager = SessionManager.create("/path/to/project")
user_id = manager.append_message(UserMessage(content="hello"))
manager.append_message(
    AssistantMessage(
        api="openai-completions",
        provider="faux",
        model="faux",
        content=[TextContent(text="hi")],
        usage=Usage(input=1, output=1, total_tokens=2),
        stop_reason="stop",
    )
)

manager.branch(user_id)
manager.append_message(UserMessage(content="try a different path"))
context = manager.build_session_context()
print([message.role for message in context.messages])
```

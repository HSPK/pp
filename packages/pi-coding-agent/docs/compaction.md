# Compaction & Branch Summarization

LLMs have limited context windows. When conversations grow too long, Pi uses compaction to summarize older content while preserving recent work. This page covers both auto-compaction and branch summarization.

**Source files:**

- `packages/pi-coding-agent/src/pi_coding_agent/core/compaction.py` - auto-compaction, branch summarization, file tracking, and serialization helpers
- `packages/pi-coding-agent/src/pi_coding_agent/core/session_manager.py` - `CompactionEntry` and `BranchSummaryEntry`
- `packages/pi-coding-agent/src/pi_coding_agent/core/extensions/types.py` - extension event types
- `packages/pi-agent/src/pi_agent/harness/compaction/compaction.py` and `utils.py` - shared token estimates, summary generation, and `serialize_conversation()`

## Overview

Pi has two summarization mechanisms:

| Mechanism | Trigger | Purpose |
|-----------|---------|---------|
| Compaction | Context exceeds threshold, or `/compact` | Summarize old messages to free up context |
| Branch summarization | `/tree` navigation | Preserve context when switching branches |

Both use a structured summary format and track file operations cumulatively. Compaction and branch-summary calls go through the caller-supplied `stream_fn`; API keys and headers are resolved by the runtime before that function is called.

## Compaction

### When It Triggers

Auto-compaction triggers when:

```
context_tokens > context_window - reserve_tokens
```

By default, `reserveTokens` is 16384 tokens (configurable in `~/.pi/agent/settings.json` or `<project-dir>/.pi/settings.json`). This leaves room for the LLM's response.

You can also trigger manually with `/compact [instructions]`, where optional instructions focus the summary.

### How It Works

1. **Find cut point**: Walk backwards from newest entry, accumulating token estimates until `keepRecentTokens` (default 20k) is reached.
2. **Extract messages**: Collect messages from the previous kept boundary (or session start) up to the cut point.
3. **Generate summary**: Call the LLM to summarize, passing the previous summary as iterative context when present.
4. **Append entry**: Save `CompactionEntry` with `summary`, `firstKeptEntryId`, `tokensBefore`, optional `usage`, and `details`.
5. **Rebuild context**: `SessionManager.build_session_context()` uses the summary plus messages from `firstKeptEntryId` onwards.

```
Before compaction:

  entry:  0     1     2     3      4     5     6      7      8     9
        ┌─────┬─────┬─────┬─────┬──────┬─────┬──── ─┬──────┬─────┬─────┐
        │ hdr │ usr │ ass │ tool │ usr │ ass │ tool │ tool │ ass │ tool│
        └─────┴─────┴─────┴──────┴─────┴─────┴──────┴──────┴─────┴─────┘
                └────────┬───────┘ └──────────────┬──────────────┘
               messagesToSummarize            kept messages
                                   ↑
                          firstKeptEntryId (entry 4)

After compaction (new entry appended):

  entry:  0     1     2     3      4     5     6      7      8     9     10
        ┌─────┬─────┬─────┬─────┬──────┬─────┬──── ─┬──────┬─────┬─────┬─────┐
        │ hdr │ usr │ ass │ tool │ usr │ ass │ tool │ tool │ ass │ tool│ cmp │
        └─────┴─────┴─────┴──────┴─────┴─────┴──────┴──────┴─────┴─────┴─────┘
               └──────────┬──────┘ └──────────────────────┬───────────────────┘
                 not sent to LLM                    sent to LLM
                                                         ↑
                                              starts from firstKeptEntryId
```

On repeated compactions, the summarized span starts at the previous compaction's kept boundary (`firstKeptEntryId`), not at the compaction entry itself. If that kept entry cannot be found, the boundary falls back to the entry after the previous compaction. Pi recalculates `tokensBefore` from the rebuilt session context before writing the new `CompactionEntry`.

`retainedTail` is not used by `pi_coding_agent.core.session_manager`; coding-agent compactions use `firstKeptEntryId`.

### Split Turns

A turn starts with a user message and includes all assistant responses and tool results until the next user message. Normally, compaction cuts at turn boundaries.

When a single turn exceeds `keepRecentTokens`, the cut point can land mid-turn at an assistant message. This is a split turn:

```
Split turn:

  entry:  0     1     2      3     4      5      6     7      8
        ┌─────┬─────┬─────┬──────┬─────┬──────┬──────┬─────┬──────┐
        │ hdr │ usr │ ass │ tool │ ass │ tool │ tool │ ass │ tool │
        └─────┴─────┴─────┴──────┴─────┴──────┴──────┴─────┴──────┘
                ↑                                     ↑
         turnStartIndex = 1                  firstKeptEntryId = 7
                │                                     │
                └──── turnPrefixMessages (1-6) ───────┘
                                                      └── kept (7-8)
```

For split turns, Pi generates two summaries and merges them:

1. **History summary**: previous context, if any.
2. **Turn prefix summary**: the early part of the split turn.

### Cut Point Rules

Valid cut points are:

- User messages
- Assistant messages
- Bash execution messages
- Custom messages
- Branch summary messages
- Compaction summary messages

Never cut at tool results; they must stay with their tool call.

### CompactionEntry Structure

Defined in `session_manager.py`:

```python
from dataclasses import dataclass
from typing import Any, Literal

from pi_ai.types import Usage


@dataclass
class CompactionEntry:
    id: str
    parent_id: str | None
    timestamp: str
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: Any = None
    usage: Usage | None = None
    from_hook: bool | None = None
    type: Literal["compaction"] = "compaction"
```

Persisted JSON uses camelCase:

```json
{
  "type": "compaction",
  "id": "f6a7b8c9",
  "parentId": "e5f6a7b8",
  "timestamp": "2026-08-12T10:35:00.000Z",
  "summary": "User discussed X, Y, Z...",
  "firstKeptEntryId": "c3d4e5f6",
  "tokensBefore": 50000,
  "details": {
    "readFiles": ["src/app.py"],
    "modifiedFiles": ["src/app.py"]
  }
}
```

Extensions can store any JSON-serializable data in `details`. Default compaction tracks file operations in `{ "readFiles": string[], "modifiedFiles": string[] }`. Generated and extension-provided summaries store LLM `usage` when available so session totals include summarization work.

For direct programmatic summarization, `generate_summary()` returns the summary text and `generate_summary_with_usage()` returns `(text, usage)`.

## Branch Summarization

### When It Triggers

When you use `/tree` to navigate to a different branch, Pi can summarize the work you are leaving. This injects context from the left branch into the new branch.

### How It Works

1. **Find common ancestor**: deepest node shared by old and new positions.
2. **Collect entries**: walk from old leaf back to the common ancestor.
3. **Prepare with budget**: include messages up to token budget, newest first.
4. **Generate summary**: call LLM with structured format.
5. **Append entry**: save `BranchSummaryEntry` at the navigation point.

```
Tree before navigation:

         ┌─ B ─ C ─ D (old leaf, being abandoned)
    A ───┤
         └─ E ─ F (target)

Common ancestor: A
Entries to summarize: B, C, D

After navigation with summary:

         ┌─ B ─ C ─ D
    A ───┤
         └─ E ─ F ─ [summary of B,C,D] (new leaf)
```

### Cumulative File Tracking

Both compaction and branch summarization track files cumulatively. When generating a summary, Pi extracts file operations from:

- Tool calls in the messages being summarized
- Previous compaction or branch summary `details`, when present and not extension-provided

This means file tracking accumulates across multiple compactions or nested branch summaries, preserving the full history of read and modified files.

### BranchSummaryEntry Structure

Defined in `session_manager.py`:

```python
from dataclasses import dataclass
from typing import Any, Literal

from pi_ai.types import Usage


@dataclass
class BranchSummaryEntry:
    id: str
    parent_id: str | None
    timestamp: str
    from_id: str
    summary: str
    details: Any = None
    usage: Usage | None = None
    from_hook: bool | None = None
    type: Literal["branch_summary"] = "branch_summary"
```

Persisted JSON uses camelCase:

```json
{
  "type": "branch_summary",
  "id": "a7b8c9d0",
  "parentId": "a1b2c3d4",
  "timestamp": "2026-08-12T10:40:00.000Z",
  "fromId": "a1b2c3d4",
  "summary": "Branch explored approach A...",
  "details": {
    "readFiles": [],
    "modifiedFiles": []
  }
}
```

`branch_with_summary()` writes `fromId` as the attachment id (or `"root"` when the summary is attached before any entry). Same as compaction, extensions can store custom data in `details`.

See `collect_entries_for_branch_summary()`, `prepare_branch_entries()`, and `generate_branch_summary()` in `compaction.py` for the implementation.

## Summary Format

Both compaction and branch summarization use the same structured format:

```markdown
## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
- [Requirements mentioned by user]

## Progress
### Done
- [x] [Completed tasks]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues, if any]

## Key Decisions
- **[Decision]**: [Rationale]

## Next Steps
1. [What should happen next]

## Critical Context
- [Data needed to continue]

<read-files>
path/to/file1.py
path/to/file2.py
</read-files>

<modified-files>
path/to/changed.py
</modified-files>
```

### Message Serialization

Before summarization, messages are serialized to text via `serialize_conversation()` from `pi_agent.harness.compaction.utils`:

```
[User]: What they said
[Assistant thinking]: Internal reasoning
[Assistant]: Response text
[Assistant tool calls]: read(path="foo.py"); edit(path="bar.py", ...)
[Tool result]: Output from tool
```

This prevents the model from treating the transcript as a conversation to continue.

Tool results are truncated during serialization to keep summarization requests within budget. Content beyond the limit is replaced with a truncation marker.

## Custom Summarization via Extensions

Extensions can intercept and customize both compaction and branch summarization. Python extensions expose a `pi_extension(pi)` entry point and register handlers with `pi.on()`.

### session_before_compact

Fired before auto-compaction or `/compact`. Can cancel or provide a custom summary. See `SessionBeforeCompactEvent`, `SessionBeforeCompactResult`, and `CompactionPreparation`.

```python
from pi_coding_agent.core.compaction import CompactionResult
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import (
    ExtensionContext,
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
)


def pi_extension(pi: ExtensionAPI) -> None:
    async def before_compact(
        event: SessionBeforeCompactEvent,
        ctx: ExtensionContext,
    ) -> SessionBeforeCompactResult | None:
        preparation = event.preparation

        if event.reason == "manual" and preparation.tokens_before < 1000:
            return SessionBeforeCompactResult(cancel=True)

        if event.custom_instructions == "short":
            return SessionBeforeCompactResult(
                compaction=CompactionResult(
                    summary="Short custom summary.",
                    first_kept_entry_id=preparation.first_kept_entry_id,
                    tokens_before=preparation.tokens_before,
                    details={"readFiles": [], "modifiedFiles": []},
                )
            )

        return None

    pi.on("session_before_compact", before_compact)
```

Important event fields:

- `preparation.messages_to_summarize` - messages to summarize
- `preparation.turn_prefix_messages` - split-turn prefix, if `is_split_turn`
- `preparation.previous_summary` - previous compaction summary
- `preparation.file_ops` - extracted file operations
- `preparation.tokens_before` - context tokens before compaction
- `preparation.first_kept_entry_id` - where kept messages start
- `preparation.settings` - compaction settings
- `event.branch_entries` - all entries on current branch
- `event.reason` - `"manual"`, `"threshold"`, or `"overflow"`
- `event.will_retry` - whether the aborted turn is retried after compaction
- `event.signal` - abort signal for LLM calls

#### Converting Messages to Text

To generate a summary with your own model, convert messages to LLM messages and serialize them:

```python
from pi_agent.harness.compaction.utils import serialize_conversation
from pi_agent.harness.messages import convert_to_llm
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import (
    ExtensionContext,
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
)


def pi_extension(pi: ExtensionAPI) -> None:
    async def before_compact(
        event: SessionBeforeCompactEvent,
        ctx: ExtensionContext,
    ) -> SessionBeforeCompactResult | None:
        conversation_text = serialize_conversation(convert_to_llm(event.preparation.messages_to_summarize))
        ctx.ui.notify(conversation_text[:120], "info")
        return None

    pi.on("session_before_compact", before_compact)
```

See `examples/extensions/trigger_compact.py` for a complete Python extension that triggers compaction.

### session_before_tree

Fired before `/tree` navigation. Always fires regardless of whether the user chose to summarize. Can cancel navigation or provide a custom summary.

```python
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import (
    ExtensionContext,
    SessionBeforeTreeEvent,
    SessionBeforeTreeResult,
)


def pi_extension(pi: ExtensionAPI) -> None:
    async def before_tree(
        event: SessionBeforeTreeEvent,
        ctx: ExtensionContext,
    ) -> SessionBeforeTreeResult | None:
        preparation = event.preparation

        if preparation.user_wants_summary:
            return SessionBeforeTreeResult(
                summary={
                    "summary": "Custom branch summary.",
                    "details": {"readFiles": [], "modifiedFiles": []},
                }
            )

        return None

    pi.on("session_before_tree", before_tree)
```

Important preparation fields:

- `target_id` - where navigation is going
- `old_leaf_id` - current position being abandoned
- `common_ancestor_id` - shared ancestor
- `entries_to_summarize` - entries that would be summarized
- `user_wants_summary` - whether the user chose to summarize
- `custom_instructions` - optional summary focus instructions
- `replace_instructions` - whether custom instructions replace defaults
- `label` - optional label for the resulting entry

## Settings

Configure compaction in `~/.pi/agent/settings.json` or `<project-dir>/.pi/settings.json`:

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable auto-compaction |
| `reserveTokens` | `16384` | Tokens to reserve for LLM response |
| `keepRecentTokens` | `20000` | Recent tokens to keep, not summarized |

Disable auto-compaction with `"enabled": false`. You can still compact manually with `/compact`.

Branch summarization has a separate settings object:

```json
{
  "branchSummary": {
    "reserveTokens": 16384,
    "skipPrompt": false
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `reserveTokens` | `16384` | Tokens reserved for prompt and summary response |
| `skipPrompt` | `false` | Skip the interactive summary prompt when supported by the caller |

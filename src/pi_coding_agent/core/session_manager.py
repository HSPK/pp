"""Append-only JSONL session tree storage.

Python port of `packages/coding-agent/src/core/session-manager.ts`. Each
session entry has an `id`/`parent_id` forming a tree; the "leaf" pointer
tracks the current position, and `append_*` methods create a new child of
the leaf. `branch()`/`reset_leaf()` move the leaf to enable exploring
alternate paths without modifying history. `build_session_context()` follows
the leaf's root path, resolving the latest compaction (if any) into the
active message list sent to the LLM.

Documented simplifications versus the TypeScript original (see the task's
scope-narrowing instructions):

- **Full-file reads, no bounded byte-buffer header scan.** TypeScript's
  `readSessionHeader` reads at most `MAX_SESSION_HEADER_SCAN_BYTES` to avoid
  loading an entire large session just to discover its header/cwd for
  session *discovery* (`findMostRecentSession`, `list`). This port always
  parses the whole file (`load_entries_from_file`) even for header-only
  reads; acceptable because tests and expected CLI usage do not deal with
  session files anywhere near that scan limit.
- **Sequential session-info loading, no bounded-concurrency scheduler.**
  `buildSessionInfosWithConcurrency`'s `MAX_CONCURRENT_SESSION_INFO_LOADS`
  worker pool has no Python port; `list()`/`list_all()` build `SessionInfo`
  sequentially. Both remain `async def` for API parity with the TypeScript
  original (which is async only because of `fs/promises`), even though
  nothing in this port genuinely overlaps I/O.
- **Messages are `pi_agent.harness.messages.HarnessMessage`-shaped values**
  (`pi_ai.types.Message | BashExecutionMessage | CustomMessage |
  BranchSummaryMessage | CompactionSummaryMessage`), reusing the already
  ported harness message types instead of redefining
  `packages/coding-agent/src/core/messages.ts`. Those harness message
  dataclasses use `timestamp: int` (epoch ms), while this module's own
  `SessionEntryBase.timestamp` (like TypeScript's) is an ISO-8601 string;
  `_iso_to_ms` bridges the two at the entry -> context-message boundary.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pi_agent.harness.messages import (
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    HarnessMessage,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from pi_ai.types import ImageContent, TextContent, Usage
from pi_ai.utils.uuid import uuidv7

from pi_coding_agent.utils.paths import normalize_path, resolve_path

from .config import get_agent_dir as _get_default_agent_dir
from .config import get_sessions_dir as _get_default_sessions_dir

CURRENT_SESSION_VERSION = 3

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def assert_valid_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', and start "
            "and end with an alphanumeric character"
        )


def _now_iso() -> str:
    """Port of `new Date().toISOString()` — always millisecond precision, `Z` suffix."""
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _iso_to_ms(timestamp: str) -> int:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 0
    return int(dt.timestamp() * 1000)


def _create_session_id() -> str:
    return uuidv7()


def _generate_id(existing_ids) -> str:
    """8-hex-char id, collision-checked; falls back to a full UUID."""
    for _ in range(100):
        candidate = uuid.uuid4().hex[:8]
        if candidate not in existing_ids:
            return candidate
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Entry types
# ---------------------------------------------------------------------------


@dataclass
class SessionHeader:
    id: str
    timestamp: str
    cwd: str
    version: int | None = CURRENT_SESSION_VERSION
    parent_session: str | None = None
    type: Literal["session"] = "session"


@dataclass
class SessionMessageEntry:
    id: str
    parent_id: str | None
    timestamp: str
    message: HarnessMessage
    type: Literal["message"] = "message"


@dataclass
class ThinkingLevelChangeEntry:
    id: str
    parent_id: str | None
    timestamp: str
    thinking_level: str
    type: Literal["thinking_level_change"] = "thinking_level_change"


@dataclass
class ModelChangeEntry:
    id: str
    parent_id: str | None
    timestamp: str
    provider: str
    model_id: str
    type: Literal["model_change"] = "model_change"


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


@dataclass
class CustomEntry:
    id: str
    parent_id: str | None
    timestamp: str
    custom_type: str
    data: Any = None
    type: Literal["custom"] = "custom"


@dataclass
class LabelEntry:
    id: str
    parent_id: str | None
    timestamp: str
    target_id: str
    label: str | None
    type: Literal["label"] = "label"


@dataclass
class SessionInfoEntry:
    id: str
    parent_id: str | None
    timestamp: str
    name: str | None = None
    type: Literal["session_info"] = "session_info"


@dataclass
class CustomMessageEntry:
    id: str
    parent_id: str | None
    timestamp: str
    custom_type: str
    content: str | list[TextContent | ImageContent]
    display: bool
    details: Any = None
    type: Literal["custom_message"] = "custom_message"


SessionEntry = (
    SessionMessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry
    | SessionInfoEntry
)

FileEntry = SessionHeader | SessionEntry

_ENTRY_TYPES: dict[str, type] = {
    "message": SessionMessageEntry,
    "thinking_level_change": ThinkingLevelChangeEntry,
    "model_change": ModelChangeEntry,
    "compaction": CompactionEntry,
    "branch_summary": BranchSummaryEntry,
    "custom": CustomEntry,
    "custom_message": CustomMessageEntry,
    "label": LabelEntry,
    "session_info": SessionInfoEntry,
}


@dataclass
class SessionTreeNode:
    entry: SessionEntry
    children: list[SessionTreeNode] = field(default_factory=list)
    label: str | None = None
    label_timestamp: str | None = None


@dataclass
class SessionModelRef:
    provider: str
    model_id: str


@dataclass
class SessionContext:
    messages: list[HarnessMessage]
    thinking_level: str
    model: SessionModelRef | None


@dataclass
class SessionInfo:
    path: str
    id: str
    cwd: str
    created: datetime
    modified: datetime
    message_count: int
    first_message: str
    all_messages_text: str
    name: str | None = None
    parent_session_path: str | None = None


class ReadonlySessionManager(Protocol):
    """The read-only subset of `SessionManager` used by compaction/branch-summarization."""

    def get_cwd(self) -> str: ...
    def get_session_dir(self) -> str: ...
    def get_session_id(self) -> str: ...
    def get_session_file(self) -> str | None: ...
    def get_leaf_id(self) -> str | None: ...
    def get_leaf_entry(self) -> SessionEntry | None: ...
    def get_entry(self, entry_id: str) -> SessionEntry | None: ...
    def get_label(self, entry_id: str) -> str | None: ...
    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]: ...
    def build_context_entries(self) -> list[SessionEntry]: ...
    def get_header(self) -> SessionHeader | None: ...
    def get_entries(self) -> list[SessionEntry]: ...
    def get_tree(self) -> list[SessionTreeNode]: ...
    def get_session_name(self) -> str | None: ...


# ---------------------------------------------------------------------------
# Free functions (ported 1:1 so compaction.py etc. can reuse them, matching
# the TypeScript module exporting them alongside the class)
# ---------------------------------------------------------------------------


def _entry_message_role(entry: SessionMessageEntry) -> str | None:
    message = entry.message
    return getattr(message, "role", None)


def parse_session_entries(content: str) -> list[dict[str, Any]]:
    """Parse raw JSONL into plain dicts (used before `_entries_from_raw`)."""
    entries: list[dict[str, Any]] = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _migrate_v1_to_v2(entries: list[dict[str, Any]]) -> None:
    """Port of `migrateV1ToV2` — add id/parentId tree structure. Mutates `entries` in place."""
    ids: set[str] = set()
    prev_id: str | None = None

    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 2
            continue

        entry_id = _generate_id(ids)
        ids.add(entry_id)
        entry["id"] = entry_id
        entry["parentId"] = prev_id
        prev_id = entry_id

        if entry.get("type") == "compaction":
            first_kept_index = entry.get("firstKeptEntryIndex")
            if isinstance(first_kept_index, int):
                if 0 <= first_kept_index < len(entries):
                    target_entry = entries[first_kept_index]
                    if target_entry.get("type") != "session":
                        entry["firstKeptEntryId"] = target_entry.get("id")
                del entry["firstKeptEntryIndex"]


def _migrate_v2_to_v3(entries: list[dict[str, Any]]) -> None:
    """Port of `migrateV2ToV3` — rename `hookMessage` role to `custom`. Mutates in place."""
    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 3
            continue
        if entry.get("type") == "message":
            message = entry.get("message")
            if isinstance(message, dict) and message.get("role") == "hookMessage":
                message["role"] = "custom"


def _migrate_to_current_version(entries: list[dict[str, Any]]) -> bool:
    """Port of `migrateToCurrentVersion`. Mutates `entries` in place; returns True if applied."""
    header = next((e for e in entries if e.get("type") == "session"), None)
    version = header.get("version", 1) if header else 1

    if version >= CURRENT_SESSION_VERSION:
        return False

    if version < 2:
        _migrate_v1_to_v2(entries)
    if version < 3:
        _migrate_v2_to_v3(entries)

    return True


def migrate_session_entries(entries: list[dict[str, Any]]) -> None:
    """Port of `migrateSessionEntries` — exported for testing, mutates raw dict entries in place."""
    _migrate_to_current_version(entries)


def _usage_to_raw(usage: Usage) -> dict[str, Any]:
    """Serialise usage with the camelCase keys the TypeScript writes on disk.

    Every other persisted field is camelCased for on-disk parity, so usage must
    be too; otherwise a session written by the TypeScript `pi` reads back with
    zeroed cache and total token counts, which silently under-reports context
    size and suppresses compaction.
    """
    return {
        "input": usage.input,
        "output": usage.output,
        "cacheRead": usage.cache_read,
        "cacheWrite": usage.cache_write,
        "cacheWrite1h": usage.cache_write_1h,
        "reasoning": usage.reasoning,
        "totalTokens": usage.total_tokens,
        "cost": {
            "input": usage.cost.input,
            "output": usage.cost.output,
            "cacheRead": usage.cost.cache_read,
            "cacheWrite": usage.cost.cache_write,
            "total": usage.cost.total,
        },
    }


_USAGE_KEY_ALIASES = {
    "cacheRead": "cache_read",
    "cacheWrite": "cache_write",
    "cacheWrite1h": "cache_write_1h",
    "totalTokens": "total_tokens",
}
_COST_KEY_ALIASES = {"cacheRead": "cache_read", "cacheWrite": "cache_write"}


def _usage_from_raw(raw: dict[str, Any]) -> Usage:
    """Read usage, accepting both the camelCase wire keys and snake_case."""
    from pi_ai.types import Cost

    cost_raw = raw.get("cost")
    cost = Cost()
    if cost_raw:
        cost_fields = {}
        for key, value in cost_raw.items():
            name = _COST_KEY_ALIASES.get(key, key)
            if name in Cost.__dataclass_fields__:
                cost_fields[name] = value
        cost = Cost(**cost_fields)

    fields = {}
    for key, value in raw.items():
        name = _USAGE_KEY_ALIASES.get(key, key)
        if name in Usage.__dataclass_fields__ and name != "cost":
            fields[name] = value
    return Usage(cost=cost, **fields)


def _raw_content_blocks(content: object) -> list[dict[str, Any]]:
    """Content blocks from a raw payload, ignoring malformed shapes.

    TypeScript reads session JSONL without validation, so a message whose
    ``content`` is a plain string (or otherwise malformed) must not crash the
    loader -- it is simply not usable as structured content.
    """
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _message_from_raw(raw: dict[str, Any]) -> HarnessMessage:
    role = raw.get("role")
    if role == "bashExecution":
        return BashExecutionMessage(
            command=raw.get("command", ""),
            output=raw.get("output", ""),
            exit_code=raw.get("exit_code", raw.get("exitCode")),
            cancelled=raw.get("cancelled", False),
            truncated=raw.get("truncated", False),
            timestamp=raw.get("timestamp", 0),
            full_output_path=raw.get("full_output_path", raw.get("fullOutputPath")),
            exclude_from_context=raw.get("exclude_from_context", raw.get("excludeFromContext", False)),
        )
    if role == "custom":
        return CustomMessage(
            custom_type=raw.get("custom_type", raw.get("customType", "")),
            content=raw.get("content", ""),
            display=raw.get("display", False),
            timestamp=raw.get("timestamp", 0),
            details=raw.get("details"),
        )
    if role == "branchSummary":
        return BranchSummaryMessage(
            summary=raw.get("summary", ""),
            from_id=raw.get("from_id", raw.get("fromId", "")),
            timestamp=raw.get("timestamp", 0),
        )
    if role == "compactionSummary":
        return CompactionSummaryMessage(
            summary=raw.get("summary", ""),
            tokens_before=raw.get("tokens_before", raw.get("tokensBefore", 0)),
            timestamp=raw.get("timestamp", 0),
        )
    from pi_ai.types import AssistantMessage, ToolResultMessage, UserMessage

    if role == "assistant":
        from pi_ai.types import AssistantContent, AssistantMessageDiagnostic, ThinkingContent, ToolCall

        content: list[AssistantContent] = []
        for block in _raw_content_blocks(raw.get("content")):
            block_type = block.get("type")
            if block_type == "text":
                content.append(TextContent(text=block.get("text", "")))
            elif block_type == "thinking":
                content.append(
                    ThinkingContent(
                        thinking=block.get("thinking", ""),
                        thinking_signature=block.get("thinkingSignature"),
                        redacted=block.get("redacted"),
                    )
                )
            elif block_type == "toolCall":
                content.append(
                    ToolCall(id=block.get("id", ""), name=block.get("name", ""), arguments=block.get("arguments", {}))
                )
        usage_raw = raw.get("usage") or {}
        diagnostics_raw = raw.get("diagnostics") or []
        diagnostics = [
            AssistantMessageDiagnostic(
                kind=d.get("kind", ""),
                message=d.get("message", ""),
                detail=d.get("detail"),
                timestamp=d.get("timestamp", 0),
            )
            for d in diagnostics_raw
        ]
        return AssistantMessage(
            api=raw.get("api", ""),
            provider=raw.get("provider", ""),
            model=raw.get("model", ""),
            content=content,
            usage=_usage_from_raw(usage_raw),
            stop_reason=raw.get("stop_reason", raw.get("stopReason", "pending")),
            response_model=raw.get("response_model", raw.get("responseModel")),
            response_id=raw.get("response_id", raw.get("responseId")),
            diagnostics=diagnostics,
            error_message=raw.get("error_message", raw.get("errorMessage")),
            raw_stop_reason=raw.get("raw_stop_reason", raw.get("rawStopReason")),
            end_turn=raw.get("end_turn", raw.get("endTurn")),
            timestamp=raw.get("timestamp", 0),
        )
    if role == "toolResult":
        content = []
        for block in _raw_content_blocks(raw.get("content")):
            if block.get("type") == "text":
                content.append(TextContent(text=block.get("text", "")))
            elif block.get("type") == "image":
                content.append(
                    ImageContent(
                        data=block.get("data", ""), mime_type=block.get("mime_type", block.get("mediaType", ""))
                    )
                )
        usage_raw = raw.get("usage")
        return ToolResultMessage(
            tool_call_id=raw.get("tool_call_id", raw.get("toolCallId", "")),
            tool_name=raw.get("tool_name", raw.get("toolName", "")),
            content=content,
            details=raw.get("details"),
            usage=_usage_from_raw(usage_raw) if usage_raw else None,
            added_tool_names=raw.get("added_tool_names", raw.get("addedToolNames")),
            is_error=raw.get("is_error", raw.get("isError", False)),
            timestamp=raw.get("timestamp", 0),
        )
    # Default: user message (role omitted/"user", or unknown -- kept as-is via raw content).
    content = raw.get("content")
    if content is None:
        content = []
    elif isinstance(content, list):
        parsed: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                parsed.append(
                    ImageContent(
                        data=block.get("data", ""), mime_type=block.get("mime_type", block.get("mediaType", ""))
                    )
                )
            else:
                parsed.append(TextContent(text=block.get("text", "")))
        content = parsed
    return UserMessage(content=content, timestamp=raw.get("timestamp", 0))


def _entry_from_raw(raw: dict[str, Any]) -> FileEntry | None:
    entry_type = raw.get("type")
    if entry_type == "session":
        return SessionHeader(
            id=raw.get("id", ""),
            timestamp=raw.get("timestamp", ""),
            cwd=raw.get("cwd", ""),
            version=raw.get("version"),
            parent_session=raw.get("parentSession", raw.get("parent_session")),
        )
    if entry_type == "message":
        return SessionMessageEntry(
            id=raw.get("id", ""),
            parent_id=raw.get("parentId", raw.get("parent_id")),
            timestamp=raw.get("timestamp", ""),
            message=_message_from_raw(raw.get("message") or {}),
        )
    if entry_type == "thinking_level_change":
        return ThinkingLevelChangeEntry(
            id=raw.get("id", ""),
            parent_id=raw.get("parentId", raw.get("parent_id")),
            timestamp=raw.get("timestamp", ""),
            thinking_level=raw.get("thinkingLevel", raw.get("thinking_level", "off")),
        )
    if entry_type == "model_change":
        return ModelChangeEntry(
            id=raw.get("id", ""),
            parent_id=raw.get("parentId", raw.get("parent_id")),
            timestamp=raw.get("timestamp", ""),
            provider=raw.get("provider", ""),
            model_id=raw.get("modelId", raw.get("model_id", "")),
        )
    if entry_type == "compaction":
        usage_raw = raw.get("usage")
        return CompactionEntry(
            id=raw.get("id", ""),
            parent_id=raw.get("parentId", raw.get("parent_id")),
            timestamp=raw.get("timestamp", ""),
            summary=raw.get("summary", ""),
            first_kept_entry_id=raw.get("firstKeptEntryId", raw.get("first_kept_entry_id", "")),
            tokens_before=raw.get("tokensBefore", raw.get("tokens_before", 0)),
            details=raw.get("details"),
            usage=_usage_from_raw(usage_raw) if usage_raw else None,
            from_hook=raw.get("fromHook", raw.get("from_hook")),
        )
    if entry_type == "branch_summary":
        usage_raw = raw.get("usage")
        return BranchSummaryEntry(
            id=raw.get("id", ""),
            parent_id=raw.get("parentId", raw.get("parent_id")),
            timestamp=raw.get("timestamp", ""),
            from_id=raw.get("fromId", raw.get("from_id", "")),
            summary=raw.get("summary", ""),
            details=raw.get("details"),
            usage=_usage_from_raw(usage_raw) if usage_raw else None,
            from_hook=raw.get("fromHook", raw.get("from_hook")),
        )
    if entry_type == "custom":
        return CustomEntry(
            id=raw.get("id", ""),
            parent_id=raw.get("parentId", raw.get("parent_id")),
            timestamp=raw.get("timestamp", ""),
            custom_type=raw.get("customType", raw.get("custom_type", "")),
            data=raw.get("data"),
        )
    if entry_type == "custom_message":
        content = raw.get("content", "")
        if isinstance(content, list):
            content = [
                ImageContent(data=b.get("data", ""), mime_type=b.get("mime_type", b.get("mediaType", "")))
                if b.get("type") == "image"
                else TextContent(text=b.get("text", ""))
                for b in content
            ]
        return CustomMessageEntry(
            id=raw.get("id", ""),
            parent_id=raw.get("parentId", raw.get("parent_id")),
            timestamp=raw.get("timestamp", ""),
            custom_type=raw.get("customType", raw.get("custom_type", "")),
            content=content,
            display=raw.get("display", False),
            details=raw.get("details"),
        )
    if entry_type == "label":
        return LabelEntry(
            id=raw.get("id", ""),
            parent_id=raw.get("parentId", raw.get("parent_id")),
            timestamp=raw.get("timestamp", ""),
            target_id=raw.get("targetId", raw.get("target_id", "")),
            label=raw.get("label"),
        )
    if entry_type == "session_info":
        return SessionInfoEntry(
            id=raw.get("id", ""),
            parent_id=raw.get("parentId", raw.get("parent_id")),
            timestamp=raw.get("timestamp", ""),
            name=raw.get("name"),
        )
    return None


def _message_to_raw(message: HarnessMessage) -> dict[str, Any]:
    role = getattr(message, "role", None)

    def content_to_raw(content: Any) -> Any:
        if isinstance(content, str):
            return content
        result = []
        for block in content:
            if isinstance(block, TextContent):
                result.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                result.append({"type": "image", "data": block.data, "mediaType": block.mime_type})
            else:
                result.append(block)
        return result

    if role == "bashExecution":
        return {
            "role": role,
            "command": message.command,
            "output": message.output,
            "exitCode": message.exit_code,
            "cancelled": message.cancelled,
            "truncated": message.truncated,
            "timestamp": message.timestamp,
            "fullOutputPath": message.full_output_path,
            "excludeFromContext": message.exclude_from_context,
        }
    if role == "custom":
        return {
            "role": role,
            "customType": message.custom_type,
            "content": content_to_raw(message.content),
            "display": message.display,
            "timestamp": message.timestamp,
            "details": message.details,
        }
    if role == "branchSummary":
        return {"role": role, "summary": message.summary, "fromId": message.from_id, "timestamp": message.timestamp}
    if role == "compactionSummary":
        return {
            "role": role,
            "summary": message.summary,
            "tokensBefore": message.tokens_before,
            "timestamp": message.timestamp,
        }
    if role == "user":
        return {"role": role, "content": content_to_raw(message.content), "timestamp": message.timestamp}
    if role == "assistant":
        content = []
        for block in message.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content.append({"type": "text", "text": block.text})
            elif block_type == "thinking":
                content.append(
                    {
                        "type": "thinking",
                        "thinking": block.thinking,
                        "thinkingSignature": block.thinking_signature,
                        "redacted": block.redacted,
                    }
                )
            elif block_type == "toolCall":
                content.append({"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.arguments})
        return {
            "role": role,
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "content": content,
            "usage": _usage_to_raw(message.usage),
            "stopReason": message.stop_reason,
            "responseModel": message.response_model,
            "responseId": message.response_id,
            "diagnostics": [asdict(d) for d in message.diagnostics] if message.diagnostics else None,
            "errorMessage": message.error_message,
            "rawStopReason": message.raw_stop_reason,
            "endTurn": message.end_turn,
            "timestamp": message.timestamp,
        }
    if role == "toolResult":
        return {
            "role": role,
            "toolCallId": message.tool_call_id,
            "toolName": message.tool_name,
            "content": content_to_raw(message.content),
            "details": message.details,
            "usage": _usage_to_raw(message.usage) if message.usage is not None else None,
            "addedToolNames": message.added_tool_names,
            "isError": message.is_error,
            "timestamp": message.timestamp,
        }
    return {"role": role}


def _entry_to_raw(entry: FileEntry) -> dict[str, Any]:
    if isinstance(entry, SessionHeader):
        raw: dict[str, Any] = {
            "type": "session",
            "version": entry.version,
            "id": entry.id,
            "timestamp": entry.timestamp,
            "cwd": entry.cwd,
        }
        if entry.parent_session is not None:
            raw["parentSession"] = entry.parent_session
        return raw
    base: dict[str, Any] = {
        "type": entry.type,
        "id": entry.id,
        "parentId": entry.parent_id,
        "timestamp": entry.timestamp,
    }
    if isinstance(entry, SessionMessageEntry):
        base["message"] = _message_to_raw(entry.message)
    elif isinstance(entry, ThinkingLevelChangeEntry):
        base["thinkingLevel"] = entry.thinking_level
    elif isinstance(entry, ModelChangeEntry):
        base["provider"] = entry.provider
        base["modelId"] = entry.model_id
    elif isinstance(entry, CompactionEntry):
        base["summary"] = entry.summary
        base["firstKeptEntryId"] = entry.first_kept_entry_id
        base["tokensBefore"] = entry.tokens_before
        if entry.details is not None:
            base["details"] = entry.details
        if entry.usage is not None:
            base["usage"] = _usage_to_raw(entry.usage)
        if entry.from_hook is not None:
            base["fromHook"] = entry.from_hook
    elif isinstance(entry, BranchSummaryEntry):
        base["fromId"] = entry.from_id
        base["summary"] = entry.summary
        if entry.details is not None:
            base["details"] = entry.details
        if entry.usage is not None:
            base["usage"] = _usage_to_raw(entry.usage)
        if entry.from_hook is not None:
            base["fromHook"] = entry.from_hook
    elif isinstance(entry, CustomEntry):
        base["customType"] = entry.custom_type
        if entry.data is not None:
            base["data"] = entry.data
    elif isinstance(entry, CustomMessageEntry):
        base["customType"] = entry.custom_type
        content = entry.content
        if isinstance(content, list):
            content = [
                {"type": "image", "data": b.data, "mediaType": b.mime_type}
                if isinstance(b, ImageContent)
                else {"type": "text", "text": b.text}
                for b in content
            ]
        base["content"] = content
        base["display"] = entry.display
        if entry.details is not None:
            base["details"] = entry.details
    elif isinstance(entry, LabelEntry):
        base["targetId"] = entry.target_id
        base["label"] = entry.label
    elif isinstance(entry, SessionInfoEntry) and entry.name is not None:
        base["name"] = entry.name
    return base


def get_latest_compaction_entry(entries: list[SessionEntry]) -> CompactionEntry | None:
    for entry in reversed(entries):
        if isinstance(entry, CompactionEntry):
            return entry
    return None


def _build_entry_index(entries: list[SessionEntry]) -> dict[str, SessionEntry]:
    return {entry.id: entry for entry in entries}


_UNSET = object()


def build_session_path(
    entries: list[SessionEntry], leaf_id: str | None = _UNSET, by_id: dict[str, SessionEntry] | None = None
) -> list[SessionEntry]:
    """Walk from `leaf_id` (default: the last entry) to root. `leaf_id=None` means "empty path"."""
    index = by_id if by_id is not None else _build_entry_index(entries)
    if leaf_id is None:
        return []
    leaf: SessionEntry | None = index.get(leaf_id) if leaf_id is not _UNSET else None
    if leaf is None:
        leaf = entries[-1] if entries else None
    if leaf is None:
        return []
    path: list[SessionEntry] = []
    current: SessionEntry | None = leaf
    while current is not None:
        path.append(current)
        current = index.get(current.parent_id) if current.parent_id else None
    path.reverse()
    return path


def _get_session_context_settings(path: list[SessionEntry]) -> tuple[str, SessionModelRef | None]:
    thinking_level = "off"
    model: SessionModelRef | None = None
    for entry in path:
        if isinstance(entry, ThinkingLevelChangeEntry):
            thinking_level = entry.thinking_level
        elif isinstance(entry, ModelChangeEntry):
            model = SessionModelRef(provider=entry.provider, model_id=entry.model_id)
        elif isinstance(entry, SessionMessageEntry) and _entry_message_role(entry) == "assistant":
            message = entry.message
            model = SessionModelRef(provider=message.provider, model_id=message.model)
    return thinking_level, model


def session_entry_to_context_messages(entry: SessionEntry) -> list[HarnessMessage]:
    """Project one selected session entry into LLM/runtime messages."""
    if isinstance(entry, SessionMessageEntry):
        message = entry.message
        role = getattr(message, "role", None)
        if role in ("user", "assistant", "toolResult") and getattr(message, "content", "sentinel") is None:
            return [replace(message, content=[])]
        return [message]
    if isinstance(entry, CustomMessageEntry):
        return [
            create_custom_message(
                entry.custom_type, entry.content or [], entry.display, entry.details, _iso_to_ms(entry.timestamp)
            )
        ]
    if isinstance(entry, BranchSummaryEntry) and entry.summary:
        return [create_branch_summary_message(entry.summary, entry.from_id, _iso_to_ms(entry.timestamp))]
    if isinstance(entry, CompactionEntry):
        return [create_compaction_summary_message(entry.summary, entry.tokens_before, _iso_to_ms(entry.timestamp))]
    return []


def build_context_entries(
    entries: list[SessionEntry], leaf_id: str | None = _UNSET, by_id: dict[str, SessionEntry] | None = None
) -> list[SessionEntry]:
    """Build the active, compaction-aware session entry list along the leaf's root path."""
    path = build_session_path(entries, leaf_id, by_id)
    compaction: CompactionEntry | None = None
    for entry in path:
        if isinstance(entry, CompactionEntry):
            compaction = entry
    if compaction is None:
        return path

    compaction_idx = next((i for i, e in enumerate(path) if e.id == compaction.id), -1)
    if compaction_idx < 0:
        return path

    context_entries: list[SessionEntry] = [compaction]
    found_first_kept = False
    for entry in path[:compaction_idx]:
        if entry.id == compaction.first_kept_entry_id:
            found_first_kept = True
        if found_first_kept:
            context_entries.append(entry)
    context_entries.extend(path[compaction_idx + 1 :])
    return context_entries


def build_session_context(
    entries: list[SessionEntry], leaf_id: str | None = _UNSET, by_id: dict[str, SessionEntry] | None = None
) -> SessionContext:
    path = build_session_path(entries, leaf_id, by_id)
    thinking_level, model = _get_session_context_settings(path)
    messages: list[HarnessMessage] = []
    for entry in build_context_entries(entries, leaf_id, by_id):
        messages.extend(session_entry_to_context_messages(entry))
    return SessionContext(messages=messages, thinking_level=thinking_level, model=model)


def _get_default_session_dir_path(cwd: str, agent_dir: str | None = None) -> str:
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir or _get_default_agent_dir())
    stripped = re.sub(r"^[/\\]", "", resolved_cwd)
    safe = re.sub(r"[/\\:]", "-", stripped)
    safe_path = f"--{safe}--"
    return str(Path(resolved_agent_dir) / "sessions" / safe_path)


def get_default_session_dir(cwd: str, agent_dir: str | None = None) -> str:
    session_dir = _get_default_session_dir_path(cwd, agent_dir)
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    return session_dir


def load_entries_from_file(file_path: str | Path) -> list[FileEntry]:
    """Port of `loadEntriesFromFile` — parses raw JSONL into typed entries. Deliberately does NOT
    run migration (matching TS: migration is applied separately by `SessionManager._setSessionFile`)."""
    entries, _migrated = _load_entries_from_file_with_migration(file_path, migrate=False)
    return entries


def _load_entries_from_file_with_migration(file_path: str | Path, migrate: bool) -> tuple[list[FileEntry], bool]:
    """Parse a session file, optionally migrating the RAW dicts first.

    Migration must run before typing: an old ``hookMessage`` role has no typed
    representation, so typing it first silently downgrades it to a plain user
    message and the rename can never match.
    """
    resolved = normalize_path(str(file_path))
    path = Path(resolved)
    if not path.exists():
        return [], False
    content = path.read_text(encoding="utf-8")
    raw_entries = parse_session_entries(content)
    migrated = _migrate_to_current_version(raw_entries) if migrate else False
    entries: list[FileEntry] = []
    for raw in raw_entries:
        entry = _entry_from_raw(raw)
        if entry is not None:
            entries.append(entry)
    if not entries:
        return entries, migrated
    header = entries[0]
    if not isinstance(header, SessionHeader) or not isinstance(header.id, str):
        return [], migrated
    return entries, migrated


def _migrate_typed_entries(entries: list[FileEntry]) -> tuple[list[FileEntry], bool]:
    """Apply `migrate_session_entries` to a typed `FileEntry` list by round-tripping through the
    raw-dict representation (migration mutates the loosely-typed JSON shape, matching TS where
    `FileEntry` values are just parsed JSON with no separate raw/typed distinction)."""
    raw_entries = [_entry_to_raw(e) for e in entries]
    changed = _migrate_to_current_version(raw_entries)
    if not changed:
        return entries, False
    migrated = [e for e in (_entry_from_raw(r) for r in raw_entries) if e is not None]
    return migrated, True


def _read_session_header(file_path: str | Path) -> SessionHeader | None:
    entries = load_entries_from_file(file_path)
    if entries and isinstance(entries[0], SessionHeader):
        return entries[0]
    return None


def _get_session_header_cwd(header: SessionHeader) -> str | None:
    return header.cwd if isinstance(header.cwd, str) else None


def _session_cwd_matches(cwd: str | None, resolved_cwd: str) -> bool:
    return bool(cwd) and resolve_path(cwd) == resolved_cwd


def _read_session_header_for_discovery(file_path: str | Path) -> SessionHeader | None:
    """Port of `readSessionHeaderForDiscovery` — discovery is best-effort: an unreadable or
    corrupt file (permission denied, bad encoding, ...) must not abort discovery of the other
    files in the directory, so any error here is swallowed and treated as "not a session"."""
    try:
        return _read_session_header(file_path)
    except Exception:
        return None


def find_most_recent_session(session_dir: str, cwd: str | None = None) -> str | None:
    resolved_dir = normalize_path(session_dir)
    resolved_cwd = resolve_path(cwd) if cwd else None
    try:
        entries = list(Path(resolved_dir).iterdir())
    except OSError:
        return None
    candidates: list[tuple[str, float]] = []
    for entry in entries:
        if not entry.name.endswith(".jsonl"):
            continue
        header = _read_session_header_for_discovery(entry)
        if header is None:
            continue
        if resolved_cwd and not _session_cwd_matches(_get_session_header_cwd(header), resolved_cwd):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        candidates.append((str(entry), mtime))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def _extract_text_content(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(block.text for block in content if isinstance(block, TextContent))


def _get_message_activity_time(entry: SessionMessageEntry) -> int | None:
    message = entry.message
    role = getattr(message, "role", None)
    if role not in ("user", "assistant"):
        return None
    timestamp = getattr(message, "timestamp", None)
    if isinstance(timestamp, int):
        return timestamp
    return _iso_to_ms(entry.timestamp) or None


def _build_session_info(file_path: str) -> SessionInfo | None:
    path = Path(file_path)
    try:
        stats = path.stat()
    except OSError:
        return None

    header: SessionHeader | None = None
    message_count = 0
    first_message = ""
    all_messages: list[str] = []
    name: str | None = None
    last_activity_time: int | None = None

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None

    for raw in parse_session_entries(content):
        entry = _entry_from_raw(raw)
        if entry is None:
            continue
        if header is None:
            if not isinstance(entry, SessionHeader):
                return None
            header = entry
            continue
        if isinstance(entry, SessionInfoEntry):
            name = (entry.name or "").strip() or None
        if not isinstance(entry, SessionMessageEntry):
            continue
        message_count += 1
        activity_time = _get_message_activity_time(entry)
        if activity_time is not None:
            last_activity_time = max(last_activity_time or 0, activity_time)
        message = entry.message
        role = getattr(message, "role", None)
        if role not in ("user", "assistant"):
            continue
        text_content = _extract_text_content(message)
        if not text_content:
            continue
        all_messages.append(text_content)
        if not first_message and role == "user":
            first_message = text_content

    if header is None:
        return None

    cwd = header.cwd if isinstance(header.cwd, str) else ""
    parent_session_path = header.parent_session
    try:
        header_time = datetime.fromisoformat(header.timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        header_time = None

    if last_activity_time:
        modified = datetime.fromtimestamp(last_activity_time / 1000, tz=UTC)
    elif header_time is not None:
        modified = header_time
    else:
        modified = datetime.fromtimestamp(stats.st_mtime, tz=UTC)

    return SessionInfo(
        path=file_path,
        id=header.id,
        cwd=cwd,
        name=name,
        parent_session_path=parent_session_path,
        created=header_time or datetime.fromtimestamp(stats.st_mtime, tz=UTC),
        modified=modified,
        message_count=message_count,
        first_message=first_message or "(no messages)",
        all_messages_text=" ".join(all_messages),
    )


async def _list_sessions_from_dir(directory: str, on_progress: Any = None) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    dir_path = Path(directory)
    if not dir_path.exists():
        return sessions
    try:
        files = sorted(f for f in dir_path.iterdir() if f.name.endswith(".jsonl"))
    except OSError:
        return sessions
    total = len(files)
    for loaded, file in enumerate(files, start=1):
        info = _build_session_info(str(file))
        if on_progress:
            on_progress(loaded, total)
        if info is not None:
            sessions.append(info)
    return sessions


@dataclass
class NewSessionOptions:
    id: str | None = None
    parent_session: str | None = None


class SessionManager:
    """Manages conversation sessions as append-only trees stored in JSONL files.

    Use `build_session_context()` to get the resolved message list for the
    LLM (handles compaction summaries and follows the leaf's root path).
    """

    def __init__(
        self,
        cwd: str,
        session_dir: str,
        session_file: str | None = None,
        persist: bool = True,
        new_session_options: NewSessionOptions | None = None,
        preloaded_file_entries: list[FileEntry] | None = None,
    ) -> None:
        self._cwd = resolve_path(cwd)
        self._session_dir = normalize_path(session_dir) if session_dir else ""
        self._persist = persist
        self._flushed = False
        self._file_entries: list[FileEntry] = []
        self._by_id: dict[str, SessionEntry] = {}
        self._labels_by_id: dict[str, str] = {}
        self._label_timestamps_by_id: dict[str, str] = {}
        self._leaf_id: str | None = None
        self._session_id = ""
        self._session_file: str | None = None

        if persist and self._session_dir and not Path(self._session_dir).exists():
            Path(self._session_dir).mkdir(parents=True, exist_ok=True)

        if session_file:
            self._set_session_file(session_file, preloaded_file_entries)
        else:
            self.new_session(new_session_options)

    # -- Session-file switching ---------------------------------------------

    def set_session_file(self, session_file: str) -> None:
        self._set_session_file(session_file)

    def _set_session_file(self, session_file: str, preloaded_file_entries: list[FileEntry] | None = None) -> None:
        self._session_file = resolve_path(session_file)
        path = Path(self._session_file)
        if path.exists():
            raw_migrated = False
            if preloaded_file_entries is not None:
                self._file_entries = preloaded_file_entries
            else:
                # Migrate the raw JSON before typing so roles with no typed
                # representation (v2 "hookMessage") survive the rename.
                self._file_entries, raw_migrated = _load_entries_from_file_with_migration(
                    self._session_file, migrate=True
                )
            if not self._file_entries:
                explicit_path = self._session_file
                if path.stat().st_size > 0:
                    raise ValueError(f"Session file is not a valid pi session: {explicit_path}")
                self.new_session()
                self._session_file = explicit_path
                self._rewrite_file()
                self._flushed = True
                return

            header = next((e for e in self._file_entries if isinstance(e, SessionHeader)), None)
            self._session_id = header.id if header else _create_session_id()

            self._file_entries, typed_migrated = _migrate_typed_entries(self._file_entries)
            if raw_migrated or typed_migrated:
                self._rewrite_file()

            self._build_index()
            self._flushed = True
        else:
            explicit_path = self._session_file
            self.new_session()
            self._session_file = explicit_path

    def new_session(self, options: NewSessionOptions | None = None) -> str | None:
        if options and options.id is not None:
            assert_valid_session_id(options.id)
        self._session_id = options.id if options and options.id is not None else _create_session_id()
        timestamp = _now_iso()
        header = SessionHeader(
            id=self._session_id,
            timestamp=timestamp,
            cwd=self._cwd,
            version=CURRENT_SESSION_VERSION,
            parent_session=options.parent_session if options else None,
        )
        self._file_entries = [header]
        self._by_id.clear()
        self._labels_by_id.clear()
        self._label_timestamps_by_id.clear()
        self._leaf_id = None
        self._flushed = False

        if self._persist:
            file_timestamp = timestamp.replace(":", "-").replace(".", "-")
            self._session_file = str(Path(self.get_session_dir()) / f"{file_timestamp}_{self._session_id}.jsonl")
        return self._session_file

    def _build_index(self) -> None:
        self._by_id.clear()
        self._labels_by_id.clear()
        self._label_timestamps_by_id.clear()
        self._leaf_id = None
        for entry in self._file_entries:
            if isinstance(entry, SessionHeader):
                continue
            self._by_id[entry.id] = entry
            self._leaf_id = entry.id
            if isinstance(entry, LabelEntry):
                if entry.label:
                    self._labels_by_id[entry.target_id] = entry.label
                    self._label_timestamps_by_id[entry.target_id] = entry.timestamp
                else:
                    self._labels_by_id.pop(entry.target_id, None)
                    self._label_timestamps_by_id.pop(entry.target_id, None)

    def _rewrite_file(self) -> None:
        if not self._persist or not self._session_file:
            return
        lines = [json.dumps(_entry_to_raw(entry)) + "\n" for entry in self._file_entries]
        Path(self._session_file).write_text("".join(lines), encoding="utf-8")

    def is_persisted(self) -> bool:
        return self._persist

    def get_cwd(self) -> str:
        return self._cwd

    def get_session_dir(self) -> str:
        return self._session_dir

    def uses_default_session_dir(self) -> bool:
        return self._session_dir == _get_default_session_dir_path(self._cwd)

    def get_session_id(self) -> str:
        return self._session_id

    def get_session_file(self) -> str | None:
        return self._session_file

    def _persist_entry(self, entry: SessionEntry) -> None:
        if not self._persist or not self._session_file:
            return
        has_assistant = any(
            isinstance(e, SessionMessageEntry) and _entry_message_role(e) == "assistant" for e in self._file_entries
        )
        path = Path(self._session_file)
        if not has_assistant:
            if self._flushed:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(_entry_to_raw(entry)) + "\n")
            else:
                self._flushed = False
            return

        if not self._flushed:
            lines = [json.dumps(_entry_to_raw(e)) + "\n" for e in self._file_entries]
            path.write_text("".join(lines), encoding="utf-8")
            self._flushed = True
        else:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_entry_to_raw(entry)) + "\n")

    def _append_entry(self, entry: SessionEntry) -> None:
        # Persist first: if the write fails, the in-memory transcript must not
        # advance past it, otherwise every later entry is written with a
        # parentId that does not exist on disk and reloading loses the branch.
        previous_leaf_id = self._leaf_id
        self._file_entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = entry.id
        try:
            self._persist_entry(entry)
        except Exception:
            self._file_entries.pop()
            self._by_id.pop(entry.id, None)
            self._leaf_id = previous_leaf_id
            raise

    # -- Append methods -------------------------------------------------------

    def append_message(self, message: HarnessMessage) -> str:
        entry = SessionMessageEntry(
            id=_generate_id(self._by_id), parent_id=self._leaf_id, timestamp=_now_iso(), message=message
        )
        self._append_entry(entry)
        return entry.id

    def append_thinking_level_change(self, thinking_level: str) -> str:
        entry = ThinkingLevelChangeEntry(
            id=_generate_id(self._by_id), parent_id=self._leaf_id, timestamp=_now_iso(), thinking_level=thinking_level
        )
        self._append_entry(entry)
        return entry.id

    def append_model_change(self, provider: str, model_id: str) -> str:
        entry = ModelChangeEntry(
            id=_generate_id(self._by_id),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            provider=provider,
            model_id=model_id,
        )
        self._append_entry(entry)
        return entry.id

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Any = None,
        from_hook: bool | None = None,
        usage: Usage | None = None,
    ) -> str:
        entry = CompactionEntry(
            id=_generate_id(self._by_id),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
            usage=usage,
            from_hook=from_hook,
        )
        self._append_entry(entry)
        return entry.id

    def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        entry = CustomEntry(
            id=_generate_id(self._by_id),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            custom_type=custom_type,
            data=data,
        )
        self._append_entry(entry)
        return entry.id

    def append_session_info(self, name: str) -> str:
        sanitized = re.sub(r"[\r\n]+", " ", name).strip()
        entry = SessionInfoEntry(
            id=_generate_id(self._by_id), parent_id=self._leaf_id, timestamp=_now_iso(), name=sanitized
        )
        self._append_entry(entry)
        return entry.id

    def get_session_name(self) -> str | None:
        for entry in reversed(self.get_entries()):
            if isinstance(entry, SessionInfoEntry):
                return (entry.name or "").strip() or None
        return None

    def append_custom_message_entry(
        self, custom_type: str, content: str | list[TextContent | ImageContent], display: bool, details: Any = None
    ) -> str:
        entry = CustomMessageEntry(
            id=_generate_id(self._by_id),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            custom_type=custom_type,
            content=content,
            display=display,
            details=details,
        )
        self._append_entry(entry)
        return entry.id

    # -- Tree traversal -------------------------------------------------------

    def get_leaf_id(self) -> str | None:
        return self._leaf_id

    def get_leaf_entry(self) -> SessionEntry | None:
        return self._by_id.get(self._leaf_id) if self._leaf_id else None

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return self._by_id.get(entry_id)

    def get_children(self, parent_id: str) -> list[SessionEntry]:
        return [entry for entry in self._by_id.values() if entry.parent_id == parent_id]

    def get_label(self, entry_id: str) -> str | None:
        return self._labels_by_id.get(entry_id)

    def append_label_change(self, target_id: str, label: str | None) -> str:
        if target_id not in self._by_id:
            raise ValueError(f"Entry {target_id} not found")
        entry = LabelEntry(
            id=_generate_id(self._by_id),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            target_id=target_id,
            label=label,
        )
        self._append_entry(entry)
        if label:
            self._labels_by_id[target_id] = label
            self._label_timestamps_by_id[target_id] = entry.timestamp
        else:
            self._labels_by_id.pop(target_id, None)
            self._label_timestamps_by_id.pop(target_id, None)
        return entry.id

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        path: list[SessionEntry] = []
        start_id = from_id if from_id is not None else self._leaf_id
        current = self._by_id.get(start_id) if start_id else None
        while current is not None:
            path.append(current)
            current = self._by_id.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def build_context_entries(self) -> list[SessionEntry]:
        return build_context_entries(self.get_entries(), self._leaf_id, self._by_id)

    def build_session_context(self) -> SessionContext:
        return build_session_context(self.get_entries(), self._leaf_id, self._by_id)

    def get_header(self) -> SessionHeader | None:
        return next((e for e in self._file_entries if isinstance(e, SessionHeader)), None)

    def get_entries(self) -> list[SessionEntry]:
        return [e for e in self._file_entries if not isinstance(e, SessionHeader)]

    def get_tree(self) -> list[SessionTreeNode]:
        entries = self.get_entries()
        node_map: dict[str, SessionTreeNode] = {}
        roots: list[SessionTreeNode] = []

        for entry in entries:
            node_map[entry.id] = SessionTreeNode(
                entry=entry,
                label=self._labels_by_id.get(entry.id),
                label_timestamp=self._label_timestamps_by_id.get(entry.id),
            )

        for entry in entries:
            node = node_map[entry.id]
            if entry.parent_id is None or entry.parent_id == entry.id:
                roots.append(node)
            else:
                parent = node_map.get(entry.parent_id)
                if parent is not None:
                    parent.children.append(node)
                else:
                    roots.append(node)

        stack = list(roots)
        while stack:
            node = stack.pop()
            node.children.sort(key=lambda n: n.entry.timestamp)
            stack.extend(node.children)

        return roots

    # -- Branching -------------------------------------------------------------

    def branch(self, branch_from_id: str) -> None:
        if branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        self._leaf_id = None

    def branch_with_summary(
        self,
        branch_from_id: str | None,
        summary: str,
        details: Any = None,
        from_hook: bool | None = None,
        usage: Usage | None = None,
    ) -> str:
        if branch_from_id is not None and branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id
        entry = BranchSummaryEntry(
            id=_generate_id(self._by_id),
            parent_id=branch_from_id,
            timestamp=_now_iso(),
            from_id=branch_from_id or "root",
            summary=summary,
            details=details,
            usage=usage,
            from_hook=from_hook,
        )
        self._append_entry(entry)
        return entry.id

    def create_branched_session(self, leaf_id: str) -> str | None:
        previous_session_file = self._session_file
        path = self.get_branch(leaf_id)
        if not path:
            raise ValueError(f"Entry {leaf_id} not found")

        path_without_labels: list[SessionEntry] = []
        path_parent_id: str | None = None
        for entry in path:
            if isinstance(entry, LabelEntry):
                continue
            path_without_labels.append(replace(entry, parent_id=path_parent_id))
            path_parent_id = entry.id

        new_session_id = _create_session_id()
        timestamp = _now_iso()
        file_timestamp = timestamp.replace(":", "-").replace(".", "-")
        new_session_file = str(Path(self.get_session_dir()) / f"{file_timestamp}_{new_session_id}.jsonl")

        header = SessionHeader(
            id=new_session_id,
            timestamp=timestamp,
            cwd=self._cwd,
            version=CURRENT_SESSION_VERSION,
            parent_session=previous_session_file if self._persist else None,
        )

        path_entry_ids = {e.id for e in path_without_labels}
        labels_to_write = [
            (target_id, label, self._label_timestamps_by_id[target_id])
            for target_id, label in self._labels_by_id.items()
            if target_id in path_entry_ids
        ]

        label_entries: list[LabelEntry] = []
        parent_id = path_without_labels[-1].id if path_without_labels else None
        seen_ids = set(path_entry_ids)
        for target_id, label, label_timestamp in labels_to_write:
            label_entry = LabelEntry(
                id=_generate_id(seen_ids),
                parent_id=parent_id,
                timestamp=label_timestamp,
                target_id=target_id,
                label=label,
            )
            seen_ids.add(label_entry.id)
            label_entries.append(label_entry)
            parent_id = label_entry.id

        self._file_entries = [header, *path_without_labels, *label_entries]
        self._session_id = new_session_id
        self._build_index()

        if self._persist:
            self._session_file = new_session_file
            has_assistant = any(
                isinstance(e, SessionMessageEntry) and _entry_message_role(e) == "assistant" for e in self._file_entries
            )
            if has_assistant:
                self._rewrite_file()
                self._flushed = True
            else:
                self._flushed = False
            return new_session_file

        return None

    # -- Static factories -------------------------------------------------------

    @classmethod
    def create(
        cls, cwd: str, session_dir: str | None = None, options: NewSessionOptions | None = None
    ) -> SessionManager:
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(cwd)
        return cls(cwd, directory, None, True, options)

    @classmethod
    def open(cls, path: str, session_dir: str | None = None, cwd_override: str | None = None) -> SessionManager:
        resolved_path = resolve_path(path)
        header: SessionHeader | None = None
        if cwd_override is None and Path(resolved_path).exists():
            header = _read_session_header(resolved_path)
        cwd = cwd_override or (_get_session_header_cwd(header) if header else None) or str(Path.cwd())
        directory = normalize_path(session_dir) if session_dir else str(Path(resolved_path).parent)
        return cls(cwd, directory, resolved_path, True)

    @classmethod
    def continue_recent(cls, cwd: str, session_dir: str | None = None) -> SessionManager:
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(cwd)
        filter_cwd = session_dir is not None and directory != _get_default_session_dir_path(cwd)
        most_recent = find_most_recent_session(directory, cwd if filter_cwd else None)
        if most_recent:
            return cls(cwd, directory, most_recent, True)
        return cls(cwd, directory, None, True)

    @classmethod
    def in_memory(cls, cwd: str | None = None, options: NewSessionOptions | None = None) -> SessionManager:
        return cls(cwd or str(Path.cwd()), "", None, False, options)

    @classmethod
    def fork_from(
        cls,
        source_path: str,
        target_cwd: str,
        session_dir: str | None = None,
        options: NewSessionOptions | None = None,
    ) -> SessionManager:
        resolved_source_path = resolve_path(source_path)
        resolved_target_cwd = resolve_path(target_cwd)
        source_entries = load_entries_from_file(resolved_source_path)
        if not source_entries:
            raise ValueError(f"Cannot fork: source session file is empty or invalid: {resolved_source_path}")
        source_header = next((e for e in source_entries if isinstance(e, SessionHeader)), None)
        if source_header is None:
            raise ValueError(f"Cannot fork: source session has no header: {resolved_source_path}")

        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(resolved_target_cwd)
        Path(directory).mkdir(parents=True, exist_ok=True)

        if options and options.id is not None:
            assert_valid_session_id(options.id)
        new_session_id = options.id if options and options.id is not None else _create_session_id()
        timestamp = _now_iso()
        file_timestamp = timestamp.replace(":", "-").replace(".", "-")
        new_session_file = str(Path(directory) / f"{file_timestamp}_{new_session_id}.jsonl")

        new_header = SessionHeader(
            id=new_session_id,
            timestamp=timestamp,
            cwd=resolved_target_cwd,
            version=CURRENT_SESSION_VERSION,
            parent_session=resolved_source_path,
        )
        lines = [json.dumps(_entry_to_raw(new_header)) + "\n"]
        for entry in source_entries:
            if not isinstance(entry, SessionHeader):
                lines.append(json.dumps(_entry_to_raw(entry)) + "\n")
        Path(new_session_file).write_text("".join(lines), encoding="utf-8")

        return cls(resolved_target_cwd, directory, new_session_file, True)

    @classmethod
    async def list(cls, cwd: str, session_dir: str | None = None, on_progress: Any = None) -> list[SessionInfo]:
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(cwd)
        filter_cwd = session_dir is not None and directory != _get_default_session_dir_path(cwd)
        resolved_cwd = resolve_path(cwd)
        sessions = await _list_sessions_from_dir(directory, on_progress)
        sessions = [s for s in sessions if not filter_cwd or _session_cwd_matches(s.cwd, resolved_cwd)]
        sessions.sort(key=lambda s: s.modified, reverse=True)
        return sessions

    @classmethod
    async def list_all(cls, session_dir: str | None = None, on_progress: Any = None) -> list[SessionInfo]:
        if session_dir is not None:
            sessions = await _list_sessions_from_dir(normalize_path(session_dir), on_progress)
            sessions.sort(key=lambda s: s.modified, reverse=True)
            return sessions

        sessions_dir = _get_default_sessions_dir()
        base = Path(sessions_dir)
        if not base.exists():
            return []
        try:
            dirs = [d for d in base.iterdir() if d.is_dir()]
        except OSError:
            return []

        all_files: list[str] = []
        for d in dirs:
            try:
                files = sorted(str(f) for f in d.iterdir() if f.name.endswith(".jsonl"))
            except OSError:
                files = []
            all_files.extend(files)

        total = len(all_files)
        sessions: list[SessionInfo] = []
        for loaded, file in enumerate(all_files, start=1):
            info = _build_session_info(file)
            if on_progress:
                on_progress(loaded, total)
            if info is not None:
                sessions.append(info)

        sessions.sort(key=lambda s: s.modified, reverse=True)
        return sessions


__all__ = [
    "CURRENT_SESSION_VERSION",
    "BranchSummaryEntry",
    "CompactionEntry",
    "CustomEntry",
    "CustomMessageEntry",
    "FileEntry",
    "LabelEntry",
    "ModelChangeEntry",
    "NewSessionOptions",
    "ReadonlySessionManager",
    "SessionContext",
    "SessionEntry",
    "SessionHeader",
    "SessionInfo",
    "SessionInfoEntry",
    "SessionManager",
    "SessionMessageEntry",
    "SessionModelRef",
    "SessionTreeNode",
    "ThinkingLevelChangeEntry",
    "assert_valid_session_id",
    "build_context_entries",
    "build_session_context",
    "build_session_path",
    "find_most_recent_session",
    "get_default_session_dir",
    "get_latest_compaction_entry",
    "load_entries_from_file",
    "migrate_session_entries",
    "parse_session_entries",
    "session_entry_to_context_messages",
]

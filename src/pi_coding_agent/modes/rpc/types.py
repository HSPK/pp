"""Wire types for the stdio RPC protocol.

Python port of `packages/coding-agent/src/modes/rpc/rpc-types.ts`.

TypeScript models commands and responses as discriminated unions, which exist
only at compile time -- on the wire they are plain JSON objects. So the command
and response types here are `dict[str, Any]`, and what this module contributes
is the part that survives into runtime: the set of command names the dispatcher
recognises, the two payload shapes that have their own structure
(`RpcSessionState`, `RpcSlashCommand`), and the response constructors.

Keys are camelCase because that is what the protocol puts on the wire; the
dataclasses go through `to_wire`, which converts field names for us.

One deliberate difference from TypeScript: `to_wire` omits keys whose value is
`None` rather than emitting `null`, so `get_tree` on a fresh session answers
`{"tree": []}` where TypeScript answers `{"tree": [], "leafId": null}`. This is
the convention the whole port's JSON output already follows (print mode and
`--mode json` emit through the same function), and a host reading one protocol
should not have to handle two conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from ...utils.wire import to_wire

QueueMode = Literal["all", "one-at-a-time"]
SlashCommandSource = Literal["extension", "prompt", "skill"]


RPC_COMMAND_TYPES: Final[frozenset[str]] = frozenset(
    {
        # Prompting
        "prompt",
        "steer",
        "follow_up",
        "abort",
        "new_session",
        # State
        "get_state",
        # Model
        "set_model",
        "cycle_model",
        "get_available_models",
        # Thinking
        "set_thinking_level",
        "cycle_thinking_level",
        "get_available_thinking_levels",
        # Queue modes
        "set_steering_mode",
        "set_follow_up_mode",
        # Compaction
        "compact",
        "set_auto_compaction",
        # Retry
        "set_auto_retry",
        "abort_retry",
        # Bash
        "bash",
        "abort_bash",
        # Session
        "get_session_stats",
        "export_html",
        "switch_session",
        "fork",
        "clone",
        "get_fork_messages",
        "get_entries",
        "get_tree",
        "get_last_assistant_text",
        "set_session_name",
        # Messages
        "get_messages",
        # Commands
        "get_commands",
    }
)
"""Every command name `RpcDispatcher.handle_command` accepts.

Kept as data rather than left implicit in the dispatch chain so a test can
assert the port covers the same set TypeScript does.
"""


@dataclass
class RpcSlashCommand:
    """A command a host can invoke by sending it as a prompt."""

    name: str
    source: SlashCommandSource
    source_info: Any = None
    description: str | None = None


@dataclass
class RpcSessionState:
    """Everything `get_state` reports about the live session."""

    thinking_level: Any
    is_streaming: bool
    is_compacting: bool
    steering_mode: QueueMode
    follow_up_mode: QueueMode
    session_id: str
    auto_compaction_enabled: bool
    message_count: int
    pending_message_count: int
    model: Any = None
    session_file: str | None = None
    session_name: str | None = None


class _Unset:
    """Distinguishes "no data field" from `data: null`.

    `cycle_model` and `cycle_thinking_level` answer `data: null` when there was
    nothing to cycle to, which is a different response from `abort`'s, which
    carries no `data` key at all. A plain `None` default would collapse the two.
    """


UNSET: Final = _Unset()


def make_success(command_id: str | None, command: str, data: Any = UNSET) -> dict[str, Any]:
    response: dict[str, Any] = {"id": command_id, "type": "response", "command": command, "success": True}
    if not isinstance(data, _Unset):
        response["data"] = to_wire(data)
    return response


def make_error(command_id: str | None, command: str, message: str) -> dict[str, Any]:
    return {"id": command_id, "type": "response", "command": command, "success": False, "error": message}


__all__ = [
    "RPC_COMMAND_TYPES",
    "UNSET",
    "QueueMode",
    "RpcSessionState",
    "RpcSlashCommand",
    "SlashCommandSource",
    "make_error",
    "make_success",
]

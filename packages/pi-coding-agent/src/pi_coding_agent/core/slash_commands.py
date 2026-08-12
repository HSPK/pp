"""Built-in slash commands.

Python port of `packages/coding-agent/src/core/slash-commands.ts`. The command
names and descriptions are user-visible and are copied verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import APP_NAME
from .source_info import SourceInfo

SlashCommandSource = Literal["extension", "prompt", "skill"]


@dataclass
class SlashCommandInfo:
    """A slash command contributed by an extension, prompt template or skill."""

    name: str
    source: SlashCommandSource
    source_info: SourceInfo
    description: str | None = None


@dataclass(frozen=True)
class BuiltinSlashCommand:
    name: str
    description: str
    argument_hint: str | None = None


BUILTIN_SLASH_COMMANDS: tuple[BuiltinSlashCommand, ...] = (
    BuiltinSlashCommand("settings", "Open settings menu"),
    BuiltinSlashCommand("model", "Select model (opens selector UI)", "<provider/model>"),
    BuiltinSlashCommand("scoped-models", "Enable/disable models for Ctrl+P cycling"),
    BuiltinSlashCommand("export", "Export session (HTML default, or specify path: .html/.jsonl)"),
    BuiltinSlashCommand("import", "Import and resume a session from a JSONL file"),
    BuiltinSlashCommand("share", "Share session as a secret GitHub gist"),
    BuiltinSlashCommand("copy", "Copy last agent message to clipboard"),
    BuiltinSlashCommand("name", "Set session display name"),
    BuiltinSlashCommand("session", "Show session info and stats"),
    BuiltinSlashCommand("changelog", "Show changelog entries"),
    BuiltinSlashCommand("hotkeys", "Show all keyboard shortcuts"),
    BuiltinSlashCommand("fork", "Create a new fork from a previous user message"),
    BuiltinSlashCommand("clone", "Duplicate the current session at the current position"),
    BuiltinSlashCommand("tree", "Navigate session tree (switch branches)"),
    BuiltinSlashCommand("trust", "Save project trust decision for future sessions"),
    BuiltinSlashCommand("login", "Configure provider authentication", "<provider>"),
    BuiltinSlashCommand("logout", "Remove provider authentication"),
    BuiltinSlashCommand("new", "Start a new session"),
    BuiltinSlashCommand("compact", "Manually compact the session context"),
    BuiltinSlashCommand("resume", "Resume a different session"),
    BuiltinSlashCommand("reload", "Reload keybindings, extensions, skills, prompts, themes, and context files"),
    BuiltinSlashCommand("quit", f"Quit {APP_NAME}"),
)


def find_builtin_slash_command(name: str) -> BuiltinSlashCommand | None:
    return next((command for command in BUILTIN_SLASH_COMMANDS if command.name == name), None)

"""CLI argument parsing and help display.

Python port of `packages/coding-agent/src/cli/args.ts`. The TypeScript hand-rolls
a positional scanner rather than using a parser library, and the port keeps that
structure: the flag semantics (which flags consume a following value, how
unknown flags become extension flags, how `--print` optionally swallows the next
argument) are behaviour that extensions and scripts depend on, and a rewrite
onto ``argparse`` would change them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from ..core.config import APP_NAME

Mode = Literal["text", "json", "rpc"]
TuiMode = Literal["regular", "fullscreen"]
DiagnosticType = Literal["warning", "error"]

VALID_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
VALID_MODES = ("text", "json", "rpc")
VALID_TUI_MODES = ("regular", "fullscreen")


@dataclass
class Diagnostic:
    type: DiagnosticType
    message: str


@dataclass
class Args:
    """Parsed command line. Mirrors the TypeScript ``Args`` interface."""

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    system_prompt: str | None = None
    append_system_prompt: list[str] | None = None
    thinking: str | None = None
    continue_: bool | None = None
    resume: bool | None = None
    help: bool | None = None
    version: bool | None = None
    mode: Mode | None = None
    name: str | None = None
    no_session: bool | None = None
    session: str | None = None
    session_id: str | None = None
    fork: str | None = None
    session_dir: str | None = None
    models: list[str] | None = None
    tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    no_tools: bool | None = None
    no_builtin_tools: bool | None = None
    extensions: list[str] | None = None
    no_extensions: bool | None = None
    print: bool | None = None
    export: str | None = None
    no_skills: bool | None = None
    skills: list[str] | None = None
    prompt_templates: list[str] | None = None
    no_prompt_templates: bool | None = None
    themes: list[str] | None = None
    no_themes: bool | None = None
    no_context_files: bool | None = None
    list_models: str | bool | None = None
    offline: bool | None = None
    tui_mode: TuiMode | None = None
    verbose: bool | None = None
    project_trust_override: bool | None = None
    messages: list[str] = field(default_factory=list)
    file_args: list[str] = field(default_factory=list)
    unknown_flags: dict[str, bool | str] = field(default_factory=dict)
    """Unknown flags, which extensions may claim as their own."""
    diagnostics: list[Diagnostic] = field(default_factory=list)


def is_valid_thinking_level(level: str) -> bool:
    return level in VALID_THINKING_LEVELS


def _split_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",")]


def _split_non_empty_list(value: str) -> list[str]:
    return [part for part in (p.strip() for p in value.split(",")) if part]


def parse_args(argv: list[str]) -> Args:
    """Parse a command line into :class:`Args`. Never raises; issues become diagnostics."""
    result = Args()
    index = 0
    total = len(argv)

    while index < total:
        arg = argv[index]
        has_value = index + 1 < total
        following = argv[index + 1] if has_value else None

        if arg in ("--help", "-h"):
            result.help = True
        elif arg in ("--version", "-v"):
            result.version = True
        elif arg == "--mode" and has_value:
            index += 1
            if argv[index] in VALID_MODES:
                result.mode = argv[index]  # type: ignore[assignment]
        elif arg in ("--continue", "-c"):
            result.continue_ = True
        elif arg in ("--resume", "-r"):
            result.resume = True
        elif arg == "--provider" and has_value:
            index += 1
            result.provider = argv[index]
        elif arg == "--model" and has_value:
            index += 1
            result.model = argv[index]
        elif arg == "--api-key" and has_value:
            index += 1
            result.api_key = argv[index]
        elif arg == "--system-prompt" and has_value:
            index += 1
            result.system_prompt = argv[index]
        elif arg == "--append-system-prompt" and has_value:
            index += 1
            result.append_system_prompt = result.append_system_prompt or []
            result.append_system_prompt.append(argv[index])
        elif arg in ("--name", "-n"):
            if has_value:
                index += 1
                result.name = argv[index]
            else:
                result.diagnostics.append(Diagnostic("error", "--name requires a value"))
        elif arg == "--no-session":
            result.no_session = True
        elif arg == "--session" and has_value:
            index += 1
            result.session = argv[index]
        elif arg == "--session-id" and has_value:
            index += 1
            result.session_id = argv[index]
        elif arg == "--fork" and has_value:
            index += 1
            result.fork = argv[index]
        elif arg == "--session-dir" and has_value:
            index += 1
            result.session_dir = argv[index]
        elif arg == "--models" and has_value:
            index += 1
            result.models = _split_list(argv[index])
        elif arg in ("--no-tools", "-nt"):
            result.no_tools = True
        elif arg in ("--no-builtin-tools", "-nbt"):
            result.no_builtin_tools = True
        elif arg in ("--tools", "-t") and has_value:
            index += 1
            result.tools = _split_non_empty_list(argv[index])
        elif arg in ("--exclude-tools", "-xt") and has_value:
            index += 1
            result.exclude_tools = _split_non_empty_list(argv[index])
        elif arg == "--thinking" and has_value:
            index += 1
            level = argv[index]
            if is_valid_thinking_level(level):
                result.thinking = level
            else:
                result.diagnostics.append(
                    Diagnostic(
                        "warning",
                        f'Invalid thinking level "{level}". Valid values: {", ".join(VALID_THINKING_LEVELS)}',
                    )
                )
        elif arg in ("--print", "-p"):
            result.print = True
            # A prompt may follow -p, but not a file arg and not another flag
            # (a "---" prefixed value is treated as text, matching the TS).
            if (
                following is not None
                and not following.startswith("@")
                and (not following.startswith("-") or following.startswith("---"))
            ):
                result.messages.append(following)
                index += 1
        elif arg == "--export" and has_value:
            index += 1
            result.export = argv[index]
        elif arg in ("--extension", "-e") and has_value:
            index += 1
            result.extensions = result.extensions or []
            result.extensions.append(argv[index])
        elif arg in ("--no-extensions", "-ne"):
            result.no_extensions = True
        elif arg == "--skill" and has_value:
            index += 1
            result.skills = result.skills or []
            result.skills.append(argv[index])
        elif arg == "--prompt-template" and has_value:
            index += 1
            result.prompt_templates = result.prompt_templates or []
            result.prompt_templates.append(argv[index])
        elif arg == "--theme" and has_value:
            index += 1
            result.themes = result.themes or []
            result.themes.append(argv[index])
        elif arg in ("--no-skills", "-ns"):
            result.no_skills = True
        elif arg in ("--no-prompt-templates", "-np"):
            result.no_prompt_templates = True
        elif arg == "--no-themes":
            result.no_themes = True
        elif arg in ("--no-context-files", "-nc"):
            result.no_context_files = True
        elif arg == "--list-models":
            if following is not None and not following.startswith("-") and not following.startswith("@"):
                index += 1
                result.list_models = following
            else:
                result.list_models = True
        elif arg == "--tui-mode":
            if following in VALID_TUI_MODES:
                result.tui_mode = following  # type: ignore[assignment]
                index += 1
            elif following is None or following.startswith("-"):
                result.diagnostics.append(Diagnostic("error", "--tui-mode requires regular or fullscreen"))
            else:
                index += 1
                result.diagnostics.append(
                    Diagnostic("error", f'Invalid TUI mode "{following}". Valid values: regular, fullscreen')
                )
        elif arg == "--verbose":
            result.verbose = True
        elif arg in ("--approve", "-a"):
            result.project_trust_override = True
        elif arg in ("--no-approve", "-na"):
            result.project_trust_override = False
        elif arg == "--offline":
            result.offline = True
        elif arg.startswith("@"):
            result.file_args.append(arg[1:])
        elif arg.startswith("--"):
            equals_index = arg.find("=")
            if equals_index != -1:
                result.unknown_flags[arg[2:equals_index]] = arg[equals_index + 1 :]
            else:
                flag_name = arg[2:]
                if following is not None and not following.startswith("-") and not following.startswith("@"):
                    result.unknown_flags[flag_name] = following
                    index += 1
                else:
                    result.unknown_flags[flag_name] = True
        elif arg.startswith("-") and not arg.startswith("--"):
            result.diagnostics.append(Diagnostic("error", f"Unknown option: {arg}"))
        elif not arg.startswith("-"):
            result.messages.append(arg)

        index += 1

    return result


HELP_TEXT = f"""{APP_NAME} - AI coding assistant with read, bash, edit, write tools

Usage:
  {APP_NAME} [options] [@files...] [messages...]

Commands:
  {APP_NAME} install <source> [-l]     Install extension source and add to settings
  {APP_NAME} remove <source> [-l]      Remove extension source from settings
  {APP_NAME} uninstall <source> [-l]   Alias for remove
  {APP_NAME} update [source|self|pi]   Update pi, extensions, or model catalogs
  {APP_NAME} list                      List installed extensions from settings
  {APP_NAME} config [-l]               Open TUI to enable/disable package resources (Tab switches scope)
  {APP_NAME} auth <command>            Print credentials or check provider readiness
  {APP_NAME} <command> --help          Show help for install/remove/uninstall/update/list/config/auth

Options:
  --provider <name>              Provider name (default: google)
  --model <pattern>              Model pattern or ID (supports "provider/id" and optional ":<thinking>")
  --api-key <key>                API key (defaults to env vars)
  --system-prompt <text>         System prompt (default: coding assistant prompt)
  --append-system-prompt <text>  Append text or file contents to the system prompt (can be used multiple times)
  --mode <mode>                  Output mode: text (default), json, or rpc
  --print, -p                    Non-interactive mode: process prompt and exit
  --continue, -c                 Continue previous session
  --resume, -r                   Select a session to resume
  --session <path|id>            Use specific session file or partial UUID
  --session-id <id>              Use exact project session ID, creating it if missing
  --fork <path|id>               Fork specific session file or partial UUID into a new session
  --session-dir <dir>            Directory for session storage and lookup
  --no-session                   Don't save session (ephemeral)
  --name, -n <name>              Set session display name
  --models <patterns>            Comma-separated model patterns for Ctrl+P cycling
                                 Supports globs (anthropic/*, *sonnet*) and fuzzy matching
  --no-tools, -nt                Disable all tools by default (built-in and extension)
  --no-builtin-tools, -nbt       Disable built-in tools by default but keep extension/custom tools enabled
  --tools, -t <tools>            Comma-separated allowlist of tool names to enable
                                 Applies to built-in, extension, and custom tools
  --exclude-tools, -xt <tools>   Comma-separated denylist of tool names to disable
                                 Applies to built-in, extension, and custom tools
  --thinking <level>             Set thinking level: off, minimal, low, medium, high, xhigh, max
  --extension, -e <path>         Load an extension file (can be used multiple times)
  --no-extensions, -ne           Disable extension discovery (explicit -e paths still work)
  --skill <path>                 Load a skill file or directory (can be used multiple times)
  --no-skills, -ns               Disable skills discovery and loading
  --prompt-template <path>       Load a prompt template file or directory (can be used multiple times)
  --no-prompt-templates, -np     Disable prompt template discovery and loading
  --theme <path>                 Load a theme file or directory (can be used multiple times)
  --no-themes                    Disable theme discovery and loading
  --no-context-files, -nc        Disable AGENTS.md and CLAUDE.md discovery and loading
  --export <file>                Export session file to HTML and exit (not ported)
  --list-models [search]         List available models (with optional fuzzy search)
  --verbose                      Force verbose startup (overrides quietStartup setting)
  --tui-mode <mode>              TUI mode: regular (default) or fullscreen
  --approve, -a                  Trust project-local files for this run
  --no-approve, -na              Ignore project-local files for this run
  --offline                      Disable startup network operations (same as PI_OFFLINE=1)
  --help, -h                     Show this help
  --version, -v                  Show version number

Extensions can register additional flags (e.g., --plan from plan-mode extension).
"""


def print_help(write: Callable[[str], None] | None = None) -> None:
    """Print the CLI help text.

    TypeScript appends a section for extension-registered flags; the extension
    host is not ported, so there is nothing to append here.
    """
    (write or print)(HELP_TEXT)

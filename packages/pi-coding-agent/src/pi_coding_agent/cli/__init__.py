"""Coding agent command line.

`main.py` holds the entry point ported from `packages/coding-agent/src/cli.ts`;
`args.py` holds the argument scanner ported from
`packages/coding-agent/src/cli/args.ts`. This package re-exports the entry point
so `pi_coding_agent.cli.main` and `pi_coding_agent.cli.run_once` keep working.
"""

from __future__ import annotations

from .args import (
    VALID_MODES,
    VALID_THINKING_LEVELS,
    VALID_TUI_MODES,
    Args,
    Diagnostic,
    Mode,
    TuiMode,
    is_valid_thinking_level,
    parse_args,
)
from .entry import (
    DEFAULT_SYSTEM_PROMPT,
    build_models,
    build_parser,
    format_event,
    load_default_tools,
    main,
    resolve_model,
    run_once,
)
from .package_manager_cli import (
    PackageCommand,
    PackageCommandOptions,
    get_package_command_usage,
    handle_config_command,
    handle_package_command,
    parse_package_command,
    resolve_project_trust,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "VALID_MODES",
    "VALID_THINKING_LEVELS",
    "VALID_TUI_MODES",
    "Args",
    "Diagnostic",
    "Mode",
    "PackageCommand",
    "PackageCommandOptions",
    "TuiMode",
    "build_models",
    "build_parser",
    "format_event",
    "get_package_command_usage",
    "handle_config_command",
    "handle_package_command",
    "is_valid_thinking_level",
    "load_default_tools",
    "main",
    "parse_args",
    "parse_package_command",
    "resolve_model",
    "resolve_project_trust",
    "run_once",
]

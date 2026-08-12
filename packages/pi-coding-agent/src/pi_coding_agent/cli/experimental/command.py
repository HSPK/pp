"""The minimal command framework behind the experimental CLI.

Python port of `packages/coding-agent/src/cli/experimental/command.ts`.

The experimental CLI needs to peel off a few leading `--option value` pairs and
hand everything after them to the existing legacy argument parser untouched.
`argparse` cannot do that -- it owns the whole argv and errors on unknown flags
-- so upstream's hand-rolled parser is ported directly. Parsing stops at the
first `--` or the first argument that is not a registered option, and the rest
becomes `remaining_args`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

TValue = TypeVar("TValue")


class NamedCommandInvocation(Protocol):
    """Any parsed invocation. `command` names which command produced it."""

    command: str


@dataclass
class CommandOptionParseResult(Generic[TValue]):
    ok: bool
    value: TValue | None = None
    error: str = ""


@dataclass
class CommandOption(Generic[TValue]):
    """One `--name value` option and how to parse its value."""

    name: str
    parse: Callable[[str], CommandOptionParseResult[TValue]]


def value_option(name: str, parse: Callable[[str], CommandOptionParseResult[TValue]]) -> CommandOption[TValue]:
    return CommandOption(name=name, parse=parse)


def string_option(name: str) -> CommandOption[str]:
    return value_option(name, lambda value: CommandOptionParseResult(ok=True, value=value))


@dataclass
class CommandResult:
    """The outcome of parsing or executing a command.

    TypeScript models this as a discriminated union of `{ok: true, command}` and
    `{ok: false, errors}`; the flag plus two optional fields is the Python
    equivalent and keeps callers' `if not result.ok` checks identical.
    """

    ok: bool
    command: Any = None
    errors: list[str] = field(default_factory=list)


CommandParseResult = CommandResult
CommandExecutionResult = CommandResult
CommandBuildResult = CommandResult


@dataclass
class ParsedCommandInput:
    """The options a command matched, plus everything it did not consume."""

    remaining_args: list[str] = field(default_factory=list)
    _values: dict[str, list[Any]] = field(default_factory=dict)

    def value(self, option: CommandOption[TValue]) -> TValue | None:
        """The first occurrence of `option`, or `None`."""
        values = self._values.get(option.name)
        return values[0] if values else None

    def values(self, option: CommandOption[TValue]) -> list[TValue]:
        """Every occurrence of `option`, in order."""
        return list(self._values.get(option.name, []))


@dataclass
class _ParsedOptions:
    values: dict[str, list[Any]] = field(default_factory=dict)
    remaining_args: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class _RegisteredCommand:
    parse: Callable[[list[str]], CommandParseResult]
    execute: Callable[[list[str], Any], Awaitable[CommandExecutionResult]]


class Command:
    """One command: its options, its builder, its action, and its subcommands."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._options: dict[str, CommandOption[Any]] = {}
        self._subcommands: dict[str, _RegisteredCommand] = {}
        self._builder: Callable[[ParsedCommandInput], CommandBuildResult] | None = None
        self._action: Callable[[Any, Any], Awaitable[None] | None] | None = None

    def option(self, option: CommandOption[Any]) -> Command:
        if option.name in self._options:
            raise ValueError(f"Option {option.name} is already registered for {self.name}")
        self._options[option.name] = option
        return self

    def build(self, builder: Callable[[ParsedCommandInput], CommandBuildResult]) -> Command:
        self._builder = builder
        return self

    def action(self, action: Callable[[Any, Any], Awaitable[None] | None]) -> Command:
        self._action = action
        return self

    def command(self, command: Command) -> Command:
        if command.name in self._subcommands:
            raise ValueError(f"Command {command.name} is already registered")
        self._subcommands[command.name] = _RegisteredCommand(parse=command.parse, execute=command.execute)
        return self

    def parse(self, argv: list[str]) -> CommandParseResult:
        """Parse `argv`, dispatching to a subcommand when the first argument names one."""
        selected = self._select(argv)
        if selected is not None:
            registered, rest = selected
            return registered.parse(rest)
        return self._parse_own(argv)

    async def execute(self, argv: list[str], context: Any) -> CommandExecutionResult:
        """Parse `argv` and run the matched command's action against `context`."""
        selected = self._select(argv)
        if selected is not None:
            registered, rest = selected
            return await registered.execute(rest, context)

        parsed = self._parse_own(argv)
        if not parsed.ok:
            return parsed
        if self._action is None:
            raise ValueError(f"Command {self.name} does not define an action")
        result = self._action(parsed.command, context)
        if isinstance(result, Awaitable):
            await result
        return CommandResult(ok=True, command=parsed.command)

    def _select(self, argv: list[str]) -> tuple[_RegisteredCommand, list[str]] | None:
        if not argv:
            return None
        registered = self._subcommands.get(argv[0])
        return (registered, argv[1:]) if registered is not None else None

    def _parse_own(self, argv: list[str]) -> CommandParseResult:
        if self._builder is None:
            raise ValueError(f"Command {self.name} does not define a builder")
        parsed = self._parse_options(argv)
        built = self._builder(ParsedCommandInput(remaining_args=parsed.remaining_args, _values=parsed.values))
        errors = [*parsed.errors, *([] if built.ok else built.errors)]
        if errors:
            return CommandResult(ok=False, errors=errors)
        if not built.ok:
            raise ValueError(f"Command {self.name} failed without an error")
        return CommandResult(ok=True, command=built.command)

    def _parse_options(self, argv: list[str]) -> _ParsedOptions:
        parsed = _ParsedOptions()
        index = 0
        while index < len(argv):
            argument = argv[index]
            if argument == "--":
                parsed.remaining_args.extend(argv[index:])
                break

            equals = argument.find("=")
            name = argument if equals == -1 else argument[:equals]
            option = self._options.get(name)
            if option is None:
                parsed.remaining_args.extend(argv[index:])
                break

            value: str | None = None if equals == -1 else argument[equals + 1 :]
            if value is None and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                value = argv[index + 1]
                index += 1

            if not value:
                parsed.errors.append(f"{name} requires a value")
                index += 1
                continue

            existing = parsed.values.setdefault(name, [])
            if existing:
                parsed.errors.append(f"{name} may only be specified once")
                index += 1
                continue

            result = option.parse(value)
            if not result.ok:
                parsed.errors.append(result.error)
                index += 1
                continue

            existing.append(result.value)
            index += 1
        return parsed


__all__ = [
    "Command",
    "CommandBuildResult",
    "CommandExecutionResult",
    "CommandOption",
    "CommandOptionParseResult",
    "CommandParseResult",
    "CommandResult",
    "NamedCommandInvocation",
    "ParsedCommandInput",
    "string_option",
    "value_option",
]

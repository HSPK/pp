"""The experimental `pi` command.

Python port of `packages/coding-agent/src/cli/experimental/commands/pi.ts`.

`pi` is the default command: it keeps every legacy CLI flag working and adds
`--listen` so the same process can also serve RPC clients.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ...args import Args
from ..auth import AuthInput
from ..command import Command, CommandResult, ParsedCommandInput
from ..command_options import (
    auth_token_file_option,
    auth_token_option,
    parse_auth,
    parse_legacy_options,
    transport_option,
)
from ..transport_address import TransportAddress


@dataclass
class PiCommand:
    options: Args
    auth: AuthInput | None = None
    listen: list[TransportAddress] = field(default_factory=list)
    command: Literal["pi"] = "pi"


class PiCommandContext(Protocol):
    run_pi: Callable[[PiCommand], Awaitable[None] | None]


listen_option = transport_option("--listen")


def _build(input: ParsedCommandInput) -> CommandResult:
    auth = parse_auth(input)
    listen = input.values(listen_option)
    legacy = parse_legacy_options(input)
    errors = [*auth.errors, *legacy.errors]
    if "connect" in legacy.options.unknown_flags:
        errors.append("--connect is only valid for client mode")
    if errors:
        return CommandResult(ok=False, errors=errors)
    return CommandResult(ok=True, command=PiCommand(options=legacy.options, auth=auth.auth, listen=listen))


pi_command = (
    Command("pi")
    .option(listen_option)
    .option(auth_token_option)
    .option(auth_token_file_option)
    .build(_build)
    .action(lambda command, context: context.run_pi(command))
)


__all__ = ["PiCommand", "PiCommandContext", "pi_command"]

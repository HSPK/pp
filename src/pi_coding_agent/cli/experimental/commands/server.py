"""The experimental `server` command.

Python port of `packages/coding-agent/src/cli/experimental/commands/server.ts`.

`server` runs the RPC server alone. It deliberately rejects the legacy CLI
flags, since none of them apply to a headless server yet.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..auth import AuthInput
from ..command import Command, CommandResult, ParsedCommandInput
from ..command_options import (
    auth_token_file_option,
    auth_token_option,
    parse_auth,
    parse_legacy_options,
    transport_option,
    unsupported_legacy_options,
)
from ..transport_address import TransportAddress


@dataclass
class ServerCommand:
    auth: AuthInput | None = None
    listen: list[TransportAddress] = field(default_factory=list)
    command: Literal["server"] = "server"


class ServerCommandContext(Protocol):
    run_server: Callable[[ServerCommand], Awaitable[None] | None]


listen_option = transport_option("--listen")


def _build(input: ParsedCommandInput) -> CommandResult:
    auth = parse_auth(input)
    listen = input.values(listen_option)
    legacy = parse_legacy_options(input)
    errors = [*auth.errors, *legacy.errors, *unsupported_legacy_options("server", input)]
    if errors:
        return CommandResult(ok=False, errors=errors)
    return CommandResult(ok=True, command=ServerCommand(auth=auth.auth, listen=listen))


server_command = (
    Command("server")
    .option(listen_option)
    .option(auth_token_option)
    .option(auth_token_file_option)
    .build(_build)
    .action(lambda command, context: context.run_server(command))
)


__all__ = ["ServerCommand", "ServerCommandContext", "server_command"]

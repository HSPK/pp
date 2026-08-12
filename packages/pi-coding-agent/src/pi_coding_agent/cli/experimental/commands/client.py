"""The experimental `client` command.

Python port of `packages/coding-agent/src/cli/experimental/commands/client.ts`.

`client` attaches to a running server over `--connect`. Like `server`, it
rejects the legacy CLI flags.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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
class ClientCommand:
    auth: AuthInput | None = None
    connect: TransportAddress | None = None
    command: Literal["client"] = "client"


class ClientCommandContext(Protocol):
    run_client: Callable[[ClientCommand], Awaitable[None] | None]


connect_option = transport_option("--connect")


def _build(input: ParsedCommandInput) -> CommandResult:
    auth = parse_auth(input)
    connect = input.value(connect_option)
    legacy = parse_legacy_options(input)
    errors = [*auth.errors, *legacy.errors, *unsupported_legacy_options("client", input)]
    if errors:
        return CommandResult(ok=False, errors=errors)
    return CommandResult(ok=True, command=ClientCommand(auth=auth.auth, connect=connect))


client_command = (
    Command("client")
    .option(connect_option)
    .option(auth_token_option)
    .option(auth_token_file_option)
    .build(_build)
    .action(lambda command, context: context.run_client(command))
)


__all__ = ["ClientCommand", "ClientCommandContext", "client_command"]

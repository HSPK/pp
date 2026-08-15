"""The experimental `pi`/`server`/`client` command surface.

Python port of `packages/coding-agent/src/cli/experimental/cli.ts` and the
modules it composes (`command.ts`, `command-options.ts`, `transport-address.ts`,
`auth.ts`, `commands/`).

`experimental_cli` is the root command: bare arguments run `pi`, and the
`server` and `client` subcommands drive the ported `pi_server`/`pi_client`
socket stack. A caller supplies a context object with `run_pi`, `run_server`
and `run_client` and calls `await experimental_cli.execute(argv, context)`.
"""

from __future__ import annotations

from .auth import AuthInput, FileAuthInput, ParsedAuthInput, TokenAuthInput, parse_auth_input
from .command import (
    Command,
    CommandOption,
    CommandOptionParseResult,
    CommandResult,
    ParsedCommandInput,
    string_option,
    value_option,
)
from .command_options import (
    auth_token_file_option,
    auth_token_option,
    parse_auth,
    parse_legacy_options,
    transport_option,
    unsupported_legacy_options,
)
from .commands import (
    ClientCommand,
    ClientCommandContext,
    PiCommand,
    PiCommandContext,
    ServerCommand,
    ServerCommandContext,
    client_command,
    pi_command,
    server_command,
)
from .transport_address import (
    ParsedTransportAddress,
    TransportAddress,
    UnixTransportAddress,
    parse_transport_address,
)

experimental_cli = pi_command.command(server_command).command(client_command)
"""The composed experimental CLI: `pi` with `server` and `client` subcommands."""


__all__ = [
    "AuthInput",
    "ClientCommand",
    "ClientCommandContext",
    "Command",
    "CommandOption",
    "CommandOptionParseResult",
    "CommandResult",
    "FileAuthInput",
    "ParsedAuthInput",
    "ParsedCommandInput",
    "ParsedTransportAddress",
    "PiCommand",
    "PiCommandContext",
    "ServerCommand",
    "ServerCommandContext",
    "TokenAuthInput",
    "TransportAddress",
    "UnixTransportAddress",
    "auth_token_file_option",
    "auth_token_option",
    "client_command",
    "experimental_cli",
    "parse_auth",
    "parse_auth_input",
    "parse_legacy_options",
    "parse_transport_address",
    "pi_command",
    "server_command",
    "string_option",
    "transport_option",
    "unsupported_legacy_options",
    "value_option",
]

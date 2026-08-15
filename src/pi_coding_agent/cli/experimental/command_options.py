"""Shared options for the experimental commands.

Python port of `packages/coding-agent/src/cli/experimental/command-options.ts`.

Every experimental command takes the same auth pair and a transport address,
then forwards whatever it did not consume to the legacy `parse_args` so the
existing CLI flags keep working underneath the new command surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..args import Args, parse_args
from .auth import AuthInput, ParsedAuthInput, parse_auth_input
from .command import CommandOption, CommandOptionParseResult, ParsedCommandInput, string_option, value_option
from .transport_address import TransportAddress, parse_transport_address

auth_token_option = string_option("--auth-token")
auth_token_file_option = string_option("--auth-token-file")


def transport_option(name: str) -> CommandOption[TransportAddress]:
    """A `--listen`/`--connect` option that parses its value as a transport address."""

    def parse(value: str) -> CommandOptionParseResult[TransportAddress]:
        result = parse_transport_address(value, name)
        if result.address is not None:
            return CommandOptionParseResult(ok=True, value=result.address)
        return CommandOptionParseResult(ok=False, error=result.error or f'Invalid {name} address "{value}"')

    return value_option(name, parse)


def parse_auth(input: ParsedCommandInput) -> ParsedAuthInput:
    """Resolve the auth options this command matched."""
    return parse_auth_input(
        auth_token=input.value(auth_token_option),
        auth_token_file=input.value(auth_token_file_option),
    )


@dataclass
class ParsedLegacyOptions:
    options: Args
    errors: list[str] = field(default_factory=list)


def parse_legacy_options(input: ParsedCommandInput) -> ParsedLegacyOptions:
    """Run the unconsumed arguments through the existing CLI parser."""
    options = parse_args(list(input.remaining_args))
    return ParsedLegacyOptions(
        options=options,
        errors=[diagnostic.message for diagnostic in options.diagnostics if diagnostic.type == "error"],
    )


def unsupported_legacy_options(command: str, input: ParsedCommandInput) -> list[str]:
    """`server` and `client` reject legacy flags outright; `pi` accepts them."""
    if not input.remaining_args:
        return []
    return [f"The experimental {command} command does not support existing CLI options yet"]


__all__ = [
    "AuthInput",
    "ParsedLegacyOptions",
    "auth_token_file_option",
    "auth_token_option",
    "parse_auth",
    "parse_legacy_options",
    "transport_option",
    "unsupported_legacy_options",
]

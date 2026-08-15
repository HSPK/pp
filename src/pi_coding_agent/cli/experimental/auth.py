"""Auth input for the experimental commands.

Python port of `packages/coding-agent/src/cli/experimental/auth.ts`.

A caller supplies the bearer token either inline (`--auth-token`) or by path
(`--auth-token-file`), never both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TokenAuthInput:
    """A token supplied inline on the command line."""

    token: str
    type: Literal["token"] = "token"


@dataclass(frozen=True)
class FileAuthInput:
    """A path to a file holding the token."""

    path: str
    type: Literal["file"] = "file"


AuthInput = TokenAuthInput | FileAuthInput


@dataclass
class ParsedAuthInput:
    auth: AuthInput | None = None
    errors: list[str] = field(default_factory=list)


def parse_auth_input(auth_token: str | None = None, auth_token_file: str | None = None) -> ParsedAuthInput:
    """Resolve the two mutually exclusive auth options into one `AuthInput`."""
    if auth_token is not None and auth_token_file is not None:
        return ParsedAuthInput(errors=["--auth-token and --auth-token-file are mutually exclusive"])
    if auth_token is not None:
        return ParsedAuthInput(auth=TokenAuthInput(token=auth_token))
    if auth_token_file is not None:
        return ParsedAuthInput(auth=FileAuthInput(path=auth_token_file))
    return ParsedAuthInput()


__all__ = ["AuthInput", "FileAuthInput", "ParsedAuthInput", "TokenAuthInput", "parse_auth_input"]

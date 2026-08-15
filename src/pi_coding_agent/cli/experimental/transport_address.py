"""Transport addresses for the experimental server/client commands.

Python port of `packages/coding-agent/src/cli/experimental/transport-address.ts`.

Only `unix:///absolute/path` is accepted. The validation is deliberately strict
so an address that round-trips differently (an authority, a query, a fragment,
a percent-encoding the URL parser would normalise) is rejected rather than
silently pointing at a different socket.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

TransportOptionName = str


@dataclass(frozen=True)
class UnixTransportAddress:
    """A Unix domain socket path."""

    path: str
    transport: str = "unix"


TransportAddress = UnixTransportAddress


@dataclass
class ParsedTransportAddress:
    """The parsed address, or the reason it was rejected."""

    address: TransportAddress | None = None
    error: str | None = None


def parse_transport_address(value: str, option: TransportOptionName) -> ParsedTransportAddress:
    """Parse a `--listen`/`--connect` value into a transport address."""
    try:
        url = urlsplit(value)
    except ValueError:
        return ParsedTransportAddress(error=f'Invalid {option} address "{value}"')

    if not url.scheme:
        # JavaScript's `new URL(value)` throws on a relative reference; `urlsplit`
        # accepts it with an empty scheme, so reject it explicitly here.
        return ParsedTransportAddress(error=f'Invalid {option} address "{value}"')

    if url.scheme != "unix":
        # `urlsplit` leaves the scheme bare; TypeScript's `URL.protocol` keeps the colon.
        return ParsedTransportAddress(error=f'Unsupported {option} transport "{url.scheme}:"')

    if url.hostname or url.port or url.username or url.password:
        return ParsedTransportAddress(error="Unix transport address must not include an authority")

    if (
        not value.startswith("unix:///")
        or value.startswith("unix:////")
        or "?" in value
        or "#" in value
        or _normalized_href(url) != value
    ):
        return ParsedTransportAddress(error=f'Invalid {option} address "{value}"')

    try:
        path = unquote(url.path, errors="strict")
    except (UnicodeDecodeError, ValueError):
        return ParsedTransportAddress(error=f'Invalid {option} address "{value}"')

    if "\0" in path:
        return ParsedTransportAddress(error=f'Invalid {option} address "{value}"')

    if not posixpath.isabs(path):
        return ParsedTransportAddress(error="Unix transport address requires an absolute path")

    return ParsedTransportAddress(address=UnixTransportAddress(path=path))


def _normalized_href(url: object) -> str:
    """Rebuild the URL the way the parser understood it.

    Stands in for TypeScript's `url.href !== value` check: if reassembling the
    parsed parts does not reproduce the input byte for byte, the input relied on
    normalisation and is rejected.
    """
    return f"{url.scheme}://{url.netloc}{url.path}"  # type: ignore[attr-defined]


__all__ = [
    "ParsedTransportAddress",
    "TransportAddress",
    "UnixTransportAddress",
    "parse_transport_address",
]

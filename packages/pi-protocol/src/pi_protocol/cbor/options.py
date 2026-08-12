"""CBOR limits and shared errors.

Python port of `packages/protocol/src/cbor/options.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass

UINT32_BASE = 0x1_0000_0000
MAX_UINT32 = 0xFFFF_FFFF
MAX_CONFIGURED_DEPTH = 512

MAX_SAFE_INTEGER = 2**53 - 1
"""JavaScript's ``Number.MAX_SAFE_INTEGER``.

The protocol is defined in TypeScript, so integers outside this range cannot
round-trip. The Python port enforces the same bound rather than silently
producing values a TypeScript peer would reject.
"""

DEFAULT_MAX_CBOR_BYTE_LENGTH = 16 * 1024 * 1024
DEFAULT_MAX_CBOR_CONTAINER_LENGTH = 1_000_000
DEFAULT_MAX_CBOR_DEPTH = 64


class CborError(Exception):
    """A CBOR payload is malformed or exceeds a configured limit."""


@dataclass(frozen=True)
class ResolvedCborOptions:
    max_byte_length: int
    max_container_length: int
    max_depth: int


def _resolve_limit(name: str, value: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > maximum:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}")
    return value


def resolve_options(
    max_byte_length: int | None = None,
    max_container_length: int | None = None,
    max_depth: int | None = None,
) -> ResolvedCborOptions:
    """Resolve CBOR limits, applying the safe defaults for untrusted payloads."""
    return ResolvedCborOptions(
        max_byte_length=_resolve_limit(
            "max_byte_length",
            DEFAULT_MAX_CBOR_BYTE_LENGTH if max_byte_length is None else max_byte_length,
            MAX_UINT32,
        ),
        max_container_length=_resolve_limit(
            "max_container_length",
            DEFAULT_MAX_CBOR_CONTAINER_LENGTH if max_container_length is None else max_container_length,
            MAX_UINT32,
        ),
        max_depth=_resolve_limit(
            "max_depth",
            DEFAULT_MAX_CBOR_DEPTH if max_depth is None else max_depth,
            MAX_CONFIGURED_DEPTH,
        ),
    )

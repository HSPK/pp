"""Strict RFC 8949 CBOR subset (Python port of ``protocol/src/cbor``)."""

from __future__ import annotations

from .decoder import decode_cbor
from .encoder import encode_cbor
from .options import (
    DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
    MAX_SAFE_INTEGER,
    MAX_UINT32,
    CborError,
    ResolvedCborOptions,
    resolve_options,
)

__all__ = [
    "DEFAULT_MAX_CBOR_BYTE_LENGTH",
    "DEFAULT_MAX_CBOR_CONTAINER_LENGTH",
    "DEFAULT_MAX_CBOR_DEPTH",
    "MAX_SAFE_INTEGER",
    "MAX_UINT32",
    "CborError",
    "ResolvedCborOptions",
    "decode_cbor",
    "encode_cbor",
    "resolve_options",
]

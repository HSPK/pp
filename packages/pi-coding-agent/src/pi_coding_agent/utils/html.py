"""HTML entity decoding.

Python port of `packages/coding-agent/src/utils/html.ts`.

Only the five named XML entities plus numeric (`&#NN;`) and hex (`&#xNN;`)
references are decoded, matching upstream: this backs the ANSI/HTML round-trip
in the exporter, not general HTML parsing. Python's `html.unescape` would
accept the whole HTML5 named-entity table, which would silently diverge, so
the decode table is ported explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_CODE_POINT = 0x10FFFF

_NAMED_ENTITIES = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
}


@dataclass
class DecodedHtmlEntity:
    """A decoded entity plus how many source characters it spanned."""

    text: str
    length: int


def _decode_code_point(code_point: int | None) -> str | None:
    if code_point is None or code_point < 0 or code_point > MAX_CODE_POINT:
        return None
    try:
        return chr(code_point)
    except ValueError:
        return None


def _parse_int(text: str, base: int) -> int | None:
    """`int(text, base)` with JavaScript `parseInt`'s tolerance for a leading sign only.

    JavaScript's `parseInt` stops at the first invalid character and returns
    `NaN` for an empty prefix; Python raises. Returning `None` here is how this
    port spells `NaN`, which the caller rejects.
    """
    try:
        return int(text, base)
    except ValueError:
        return None


def decode_html_entity(entity: str) -> str | None:
    """Decode an entity body (the text between `&` and `;`). `None` if unknown."""
    named = _NAMED_ENTITIES.get(entity)
    if named is not None:
        return named

    if entity.startswith(("#x", "#X")):
        return _decode_code_point(_parse_int(entity[2:], 16))

    if entity.startswith("#"):
        return _decode_code_point(_parse_int(entity[1:], 10))

    return None


def decode_html_entity_at(html: str, index: int) -> DecodedHtmlEntity | None:
    """Decode the entity starting at `index` (which must be the `&`).

    Returns `None` when there is no terminating `;` within 16 characters or the
    body is not a recognised entity, so callers can treat the `&` as literal.
    """
    semicolon_index = html.find(";", index + 1)
    if semicolon_index == -1 or semicolon_index - index > 16:
        return None

    decoded = decode_html_entity(html[index + 1 : semicolon_index])
    if decoded is None:
        return None

    return DecodedHtmlEntity(text=decoded, length=semicolon_index - index + 1)


__all__ = ["DecodedHtmlEntity", "decode_html_entity", "decode_html_entity_at"]

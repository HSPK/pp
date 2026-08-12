"""Colour maths for HTML session export.

Python port of the colour helpers in
`packages/coding-agent/src/core/export-html/index.ts`: parsing a CSS colour,
computing relative luminance, adjusting brightness, and deriving the export
page/card/info background colours from a base colour.

These are split out from the exporter itself because they are pure functions
with exact expected outputs, and because the exporter proper depends on the
theme module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEX_PATTERN = re.compile(r"^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$")
_RGB_PATTERN = re.compile(r"^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$")

DEFAULT_EXPORT_COLORS = ("rgb(24, 24, 30)", "rgb(30, 30, 36)", "rgb(60, 55, 40)")
"""Fallback (page, card, info) backgrounds when the base colour cannot be parsed."""


@dataclass(frozen=True)
class Rgb:
    r: int
    g: int
    b: int


@dataclass(frozen=True)
class ExportColors:
    page_bg: str
    card_bg: str
    info_bg: str


def parse_color(color: str) -> Rgb | None:
    """Parse ``#RRGGBB`` or ``rgb(r, g, b)``. Returns None for anything else."""
    hex_match = _HEX_PATTERN.match(color)
    if hex_match:
        return Rgb(int(hex_match[1], 16), int(hex_match[2], 16), int(hex_match[3], 16))

    rgb_match = _RGB_PATTERN.match(color)
    if rgb_match:
        return Rgb(int(rgb_match[1]), int(rgb_match[2]), int(rgb_match[3]))

    return None


def get_luminance(r: int, g: int, b: int) -> float:
    """Relative luminance in 0..1; higher is lighter."""

    def to_linear(channel: int) -> float:
        s = channel / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    return 0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)


def _round_half_away_from_zero(value: float) -> int:
    """Round like JavaScript's ``Math.round``.

    Python's ``round`` uses banker's rounding, so ``round(0.5)`` is 0 while
    JavaScript's ``Math.round(0.5)`` is 1. Channel values must match the
    TypeScript exactly or exported colours drift by one.
    """
    import math

    return math.floor(value + 0.5)


def adjust_brightness(color: str, factor: float) -> str:
    """Scale each channel by ``factor``. Above 1 lightens, below 1 darkens."""
    parsed = parse_color(color)
    if parsed is None:
        return color

    def adjust(channel: int) -> int:
        return min(255, max(0, _round_half_away_from_zero(channel * factor)))

    return f"rgb({adjust(parsed.r)}, {adjust(parsed.g)}, {adjust(parsed.b)})"


def derive_export_colors(base_color: str) -> ExportColors:
    """Derive the export background colours from a base colour."""
    parsed = parse_color(base_color)
    if parsed is None:
        page, card, info = DEFAULT_EXPORT_COLORS
        return ExportColors(page_bg=page, card_bg=card, info_bg=info)

    is_light = get_luminance(parsed.r, parsed.g, parsed.b) > 0.5

    if is_light:
        return ExportColors(
            page_bg=adjust_brightness(base_color, 0.96),
            card_bg=base_color,
            info_bg=(f"rgb({min(255, parsed.r + 10)}, {min(255, parsed.g + 5)}, {max(0, parsed.b - 20)})"),
        )

    return ExportColors(
        page_bg=adjust_brightness(base_color, 0.7),
        card_bg=adjust_brightness(base_color, 0.85),
        info_bg=f"rgb({min(255, parsed.r + 20)}, {min(255, parsed.g + 15)}, {parsed.b})",
    )

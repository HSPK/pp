"""Tests for the HTML-export colour maths.

Expected values were produced by running the TypeScript implementation
(`core/export-html/index.ts`) under Node on the same inputs.
"""

from __future__ import annotations

import pytest
from pi_coding_agent.core.export_html.colors import (
    DEFAULT_EXPORT_COLORS,
    Rgb,
    adjust_brightness,
    derive_export_colors,
    get_luminance,
    parse_color,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#343541", Rgb(52, 53, 65)),
        ("#ffffff", Rgb(255, 255, 255)),
        ("#000000", Rgb(0, 0, 0)),
        ("#F0A1b2", Rgb(240, 161, 178)),
        ("rgb(24, 24, 30)", Rgb(24, 24, 30)),
        ("rgb(255,255,255)", Rgb(255, 255, 255)),
        ("rgb( 10 , 20 , 30 )", Rgb(10, 20, 30)),
    ],
)
def test_parse_color_accepts_hex_and_rgb(value, expected):
    assert parse_color(value) == expected


@pytest.mark.parametrize("value", ["not a color", "#fff", "#12345", "rgb(1,2)", "", "hsl(0,0%,0%)"])
def test_parse_color_rejects_unsupported_formats(value):
    assert parse_color(value) is None


def test_parse_color_does_not_clamp_out_of_range_channels():
    # The TypeScript parses the digits as-is; clamping happens later.
    assert parse_color("rgb(300,0,0)") == Rgb(300, 0, 0)


def test_luminance_extremes():
    assert get_luminance(0, 0, 0) == 0.0
    assert get_luminance(255, 255, 255) == pytest.approx(1.0)


def test_luminance_is_weighted_towards_green():
    assert get_luminance(0, 255, 0) > get_luminance(255, 0, 0)
    assert get_luminance(255, 0, 0) > get_luminance(0, 0, 255)


def test_luminance_uses_the_linear_segment_for_dark_channels():
    assert get_luminance(1, 1, 1) == pytest.approx(0.000303526983549, rel=1e-9)


@pytest.mark.parametrize(
    ("color", "factor", "expected"),
    [
        ("#343541", 0.7, "rgb(36, 37, 46)"),
        ("#343541", 0.85, "rgb(44, 45, 55)"),
        ("#ffffff", 0.96, "rgb(245, 245, 245)"),
        ("#000000", 0.7, "rgb(0, 0, 0)"),
        ("#010101", 0.7, "rgb(1, 1, 1)"),
    ],
)
def test_adjust_brightness(color, factor, expected):
    assert adjust_brightness(color, factor) == expected


def test_adjust_brightness_clamps_to_255():
    assert adjust_brightness("#ffffff", 2.0) == "rgb(255, 255, 255)"


def test_adjust_brightness_returns_unparseable_colors_unchanged():
    assert adjust_brightness("not a color", 0.5) == "not a color"


def test_adjust_brightness_rounds_half_away_from_zero():
    # JavaScript's Math.round(0.5) is 1; Python's round(0.5) would be 0.
    assert adjust_brightness("rgb(1, 1, 1)", 0.5) == "rgb(1, 1, 1)"


def test_derive_export_colors_for_a_dark_base():
    colors = derive_export_colors("#343541")
    assert colors.page_bg == "rgb(36, 37, 46)"
    assert colors.card_bg == "rgb(44, 45, 55)"
    assert colors.info_bg == "rgb(72, 68, 65)"


def test_derive_export_colors_for_a_light_base():
    colors = derive_export_colors("#e5e5e5")
    assert colors.page_bg == "rgb(220, 220, 220)"
    # A light theme keeps the base colour as the card background.
    assert colors.card_bg == "#e5e5e5"
    assert colors.info_bg == "rgb(239, 234, 209)"


def test_derive_export_colors_falls_back_for_an_unparseable_base():
    colors = derive_export_colors("not a color")
    assert (colors.page_bg, colors.card_bg, colors.info_bg) == DEFAULT_EXPORT_COLORS


def test_derive_export_colors_switches_at_the_luminance_midpoint():
    # sRGB luminance is non-linear: 50% grey is well below 0.5. The branch
    # actually flips between #bb and #bc.
    dark = derive_export_colors("#bbbbbb")
    light = derive_export_colors("#bcbcbc")
    # The light branch keeps the base as the card background; the dark one does not.
    assert dark.card_bg != "#bbbbbb"
    assert light.card_bg == "#bcbcbc"


def test_derive_export_colors_clamps_info_channels():
    colors = derive_export_colors("#ffffff")
    assert colors.info_bg == "rgb(255, 255, 235)"
    black = derive_export_colors("#000000")
    assert black.info_bg == "rgb(20, 15, 0)"

"""Tests for the ANSI-to-HTML converter.

The expected HTML in these tests was produced by running the TypeScript
implementation (`core/export-html/ansi-to-html.ts`) under Node on the same
inputs, so they pin the port to the original byte for byte.
"""

from __future__ import annotations

import pytest

from pi_coding_agent.core.export_html.ansi_to_html import (
    ANSI_COLORS,
    ansi_lines_to_html,
    ansi_to_html,
    color256_to_hex,
    escape_html,
)

E = "\x1b["


def test_plain_text_passes_through():
    assert ansi_to_html("plain text") == "plain text"
    assert ansi_to_html("") == ""


def test_standard_foreground_colour():
    assert ansi_to_html(f"{E}31mred{E}0m") == '<span style="color:#800000">red</span>'


def test_standard_background_colour():
    assert ansi_to_html(f"{E}41mred bg{E}49m") == '<span style="background-color:#800000">red bg</span>'


def test_bright_foreground_and_background():
    assert ansi_to_html(f"{E}90mbright{E}39m") == '<span style="color:#808080">bright</span>'
    assert ansi_to_html(f"{E}100mbright{E}0m") == '<span style="background-color:#808080">bright</span>'


def test_text_styles():
    assert ansi_to_html(f"{E}1mbold{E}22m") == '<span style="font-weight:bold">bold</span>'
    assert ansi_to_html(f"{E}2mdim{E}0m") == '<span style="opacity:0.6">dim</span>'
    assert ansi_to_html(f"{E}3mitalic{E}0m") == '<span style="font-style:italic">italic</span>'
    assert ansi_to_html(f"{E}4munder{E}0m") == '<span style="text-decoration:underline">under</span>'


def test_combined_styles_render_in_css_order():
    assert (
        ansi_to_html(f"{E}31;1;4mcombined{E}0m")
        == '<span style="color:#800000;font-weight:bold;text-decoration:underline">combined</span>'
    )


def test_bold_and_dim_share_a_reset_code():
    # Each escape closes the previous span and opens a new one, so the first
    # (bold-only) span is emitted empty.
    assert ansi_to_html(f"{E}1m{E}2mx{E}22my") == (
        '<span style="font-weight:bold"></span><span style="font-weight:bold;opacity:0.6">x</span>y'
    )


def test_individual_style_resets():
    assert ansi_to_html(f"{E}3m{E}4mx{E}23m{E}24my") == (
        '<span style="font-style:italic">'
        '</span><span style="font-style:italic;text-decoration:underline">x'
        '</span><span style="text-decoration:underline"></span>y'
    )


def test_256_colour_palette():
    assert ansi_to_html(f"{E}38;5;196m256{E}0m") == '<span style="color:#ff0000">256</span>'
    assert ansi_to_html(f"{E}48;5;21m bg{E}0m") == '<span style="background-color:#0000ff"> bg</span>'


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, "#000000"),
        (15, "#ffffff"),
        (16, "#000000"),
        (196, "#ff0000"),
        (231, "#ffffff"),
        (232, "#080808"),
        (255, "#eeeeee"),
    ],
)
def test_color256_to_hex(index, expected):
    assert color256_to_hex(index) == expected


def test_rgb_true_colour():
    assert ansi_to_html(f"{E}38;2;12;34;56mrgb{E}0m") == '<span style="color:rgb(12,34,56)">rgb</span>'
    assert ansi_to_html(f"{E}48;2;255;0;128mbg{E}0m") == '<span style="background-color:rgb(255,0,128)">bg</span>'


def test_truncated_extended_colour_sequences_fall_through_to_plain_codes():
    # `38;5` has no palette index, so nothing is applied.
    assert ansi_to_html(f"{E}38;5m truncated{E}0m") == " truncated"
    # `38;2;1;2` is too short for RGB, so 38 is skipped and the remaining
    # parameters are read as ordinary codes: 1 (bold) and 2 (dim).
    assert ansi_to_html(f"{E}38;2;1;2m truncated{E}0m") == (
        '<span style="font-weight:bold;opacity:0.6"> truncated</span>'
    )


def test_empty_parameters_mean_reset():
    assert ansi_to_html(f"{E}31mred{E}mplain") == '<span style="color:#800000">red</span>plain'


def test_unknown_codes_are_ignored():
    assert ansi_to_html(f"{E}999munknown{E}0m") == "unknown"


def test_default_colour_codes_clear_the_colour():
    assert ansi_to_html(f"{E}31m{E}39mplain") == '<span style="color:#800000"></span>plain'


def test_consecutive_colours_close_and_reopen_spans():
    assert ansi_to_html(f"a{E}31mb{E}32mc{E}0md") == (
        'a<span style="color:#800000">b</span><span style="color:#008000">c</span>d'
    )


def test_unterminated_span_is_closed():
    assert ansi_to_html(f"{E}31munterminated") == '<span style="color:#800000">unterminated</span>'


def test_trailing_escape_produces_an_empty_span():
    assert ansi_to_html(f"text{E}31m") == 'text<span style="color:#800000"></span>'


def test_html_is_escaped():
    assert ansi_to_html("<script>&\"'</script>") == ("&lt;script&gt;&amp;&quot;&#039;&lt;/script&gt;")


def test_html_inside_a_styled_span_is_escaped():
    assert ansi_to_html(f"{E}31m<b>&amp;</b>{E}0m") == (
        '<span style="color:#800000">&lt;b&gt;&amp;amp;&lt;/b&gt;</span>'
    )


def test_escape_html_orders_ampersand_first():
    # Escaping & last would double-escape the other entities.
    assert escape_html("&<>\"'") == "&amp;&lt;&gt;&quot;&#039;"


def test_non_ascii_text_is_preserved():
    assert ansi_to_html("你好 世界") == "你好 世界"
    assert ansi_to_html(f"{E}31m你好{E}0m") == '<span style="color:#800000">你好</span>'


def test_palette_has_sixteen_entries():
    assert len(ANSI_COLORS) == 16


# --------------------------------------------------------------------------
# line wrapping
# --------------------------------------------------------------------------


def test_lines_are_wrapped_in_divs():
    assert ansi_lines_to_html(["a", "b"]) == '<div class="ansi-line">a</div><div class="ansi-line">b</div>'


def test_empty_lines_become_a_non_breaking_space():
    assert ansi_lines_to_html([""]) == '<div class="ansi-line">&nbsp;</div>'


def test_styled_lines_keep_their_markup():
    assert ansi_lines_to_html([f"{E}31mred{E}0m"]) == (
        '<div class="ansi-line"><span style="color:#800000">red</span></div>'
    )


def test_no_lines_produces_empty_output():
    assert ansi_lines_to_html([]) == ""

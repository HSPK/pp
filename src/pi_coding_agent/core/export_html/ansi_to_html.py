"""ANSI escape code to HTML converter.

Python port of `packages/coding-agent/src/core/export-html/ansi-to-html.ts`.

Converts terminal ANSI colour/style codes to HTML with inline styles. Supports:

- standard foreground colours (30-37) and bright variants (90-97)
- standard background colours (40-47) and bright variants (100-107)
- the 256-colour palette (``38;5;N`` / ``48;5;N``)
- RGB true colour (``38;2;R;G;B`` / ``48;2;R;G;B``)
- bold (1), dim (2), italic (3), underline (4) and their resets
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ANSI_COLORS = (
    "#000000",  # 0: black
    "#800000",  # 1: red
    "#008000",  # 2: green
    "#808000",  # 3: yellow
    "#000080",  # 4: blue
    "#800080",  # 5: magenta
    "#008080",  # 6: cyan
    "#c0c0c0",  # 7: white
    "#808080",  # 8: bright black
    "#ff0000",  # 9: bright red
    "#00ff00",  # 10: bright green
    "#ffff00",  # 11: bright yellow
    "#0000ff",  # 12: bright blue
    "#ff00ff",  # 13: bright magenta
    "#00ffff",  # 14: bright cyan
    "#ffffff",  # 15: bright white
)

# ESC[ followed by parameters and terminated by 'm'.
ANSI_PATTERN = re.compile(r"\x1b\[([\d;]*)m")


def color256_to_hex(index: int) -> str:
    """Convert a 256-colour palette index to a hex colour."""
    if index < 16:
        return ANSI_COLORS[index]

    if index < 232:
        # 6x6x6 colour cube.
        cube_index = index - 16
        r = cube_index // 36
        g = (cube_index % 36) // 6
        b = cube_index % 6

        def component(value: int) -> int:
            return 0 if value == 0 else 55 + value * 40

        return f"#{component(r):02x}{component(g):02x}{component(b):02x}"

    # 24 shades of grey.
    gray = 8 + (index - 232) * 10
    return f"#{gray:02x}{gray:02x}{gray:02x}"


def escape_html(text: str) -> str:
    """Escape the HTML special characters, matching the TypeScript exactly."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


@dataclass
class TextStyle:
    fg: str | None = None
    bg: str | None = None
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False

    def reset(self) -> None:
        self.fg = None
        self.bg = None
        self.bold = False
        self.dim = False
        self.italic = False
        self.underline = False

    def has_style(self) -> bool:
        return self.fg is not None or self.bg is not None or self.bold or self.dim or self.italic or self.underline

    def to_inline_css(self) -> str:
        parts: list[str] = []
        if self.fg:
            parts.append(f"color:{self.fg}")
        if self.bg:
            parts.append(f"background-color:{self.bg}")
        if self.bold:
            parts.append("font-weight:bold")
        if self.dim:
            parts.append("opacity:0.6")
        if self.italic:
            parts.append("font-style:italic")
        if self.underline:
            parts.append("text-decoration:underline")
        return ";".join(parts)


def apply_sgr_code(params: list[int], style: TextStyle) -> None:
    """Apply SGR parameters to ``style`` in place. Unknown codes are ignored."""
    index = 0
    while index < len(params):
        code = params[index]

        if code == 0:
            style.reset()
        elif code == 1:
            style.bold = True
        elif code == 2:
            style.dim = True
        elif code == 3:
            style.italic = True
        elif code == 4:
            style.underline = True
        elif code == 22:
            style.bold = False
            style.dim = False
        elif code == 23:
            style.italic = False
        elif code == 24:
            style.underline = False
        elif 30 <= code <= 37:
            style.fg = ANSI_COLORS[code - 30]
        elif code == 38:
            if index + 2 < len(params) and params[index + 1] == 5:
                style.fg = color256_to_hex(params[index + 2])
                index += 2
            elif index + 4 < len(params) and params[index + 1] == 2:
                r, g, b = params[index + 2], params[index + 3], params[index + 4]
                style.fg = f"rgb({r},{g},{b})"
                index += 4
        elif code == 39:
            style.fg = None
        elif 40 <= code <= 47:
            style.bg = ANSI_COLORS[code - 40]
        elif code == 48:
            if index + 2 < len(params) and params[index + 1] == 5:
                style.bg = color256_to_hex(params[index + 2])
                index += 2
            elif index + 4 < len(params) and params[index + 1] == 2:
                r, g, b = params[index + 2], params[index + 3], params[index + 4]
                style.bg = f"rgb({r},{g},{b})"
                index += 4
        elif code == 49:
            style.bg = None
        elif 90 <= code <= 97:
            style.fg = ANSI_COLORS[code - 90 + 8]
        elif 100 <= code <= 107:
            style.bg = ANSI_COLORS[code - 100 + 8]

        index += 1


def _parse_params(param_string: str) -> list[int]:
    """Parse an SGR parameter list.

    An empty parameter list means "reset", and a non-numeric parameter becomes
    0, matching JavaScript's ``parseInt(p, 10) || 0``.
    """
    if not param_string:
        return [0]
    params: list[int] = []
    for part in param_string.split(";"):
        try:
            value = int(part)
        except ValueError:
            value = 0
        params.append(value)
    return params


def ansi_to_html(text: str) -> str:
    """Convert ANSI-escaped text to HTML with inline styles."""
    style = TextStyle()
    result: list[str] = []
    last_index = 0
    in_span = False

    for match in ANSI_PATTERN.finditer(text):
        before_text = text[last_index : match.start()]
        if before_text:
            result.append(escape_html(before_text))

        if in_span:
            result.append("</span>")
            in_span = False

        apply_sgr_code(_parse_params(match.group(1)), style)

        if style.has_style():
            result.append(f'<span style="{style.to_inline_css()}">')
            in_span = True

        last_index = match.end()

    remaining_text = text[last_index:]
    if remaining_text:
        result.append(escape_html(remaining_text))

    if in_span:
        result.append("</span>")

    return "".join(result)


def ansi_lines_to_html(lines: list[str]) -> str:
    """Convert ANSI-escaped lines to HTML, wrapping each line in a div."""
    return "".join(f'<div class="ansi-line">{ansi_to_html(line) or "&nbsp;"}</div>' for line in lines)

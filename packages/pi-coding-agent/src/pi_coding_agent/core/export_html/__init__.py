"""HTML session export.

Python port of `packages/coding-agent/src/core/export-html/`.

Ported here: the ANSI-to-HTML converter (`ansi_to_html.py`) and the colour
maths that derive the export palette (`colors.py`). Both are verified byte for
byte against the TypeScript implementation run under Node.

The exporter entry point itself (`index.ts`) additionally depends on the theme
module and on the vendored `marked`/`highlight.js` browser bundles that the
TypeScript build copies into the output; see the port status in the repository
README.
"""

from __future__ import annotations

from .ansi_to_html import (
    ANSI_COLORS,
    TextStyle,
    ansi_lines_to_html,
    ansi_to_html,
    color256_to_hex,
    escape_html,
)
from .colors import (
    DEFAULT_EXPORT_COLORS,
    ExportColors,
    Rgb,
    adjust_brightness,
    derive_export_colors,
    get_luminance,
    parse_color,
)

__all__ = [
    "ANSI_COLORS",
    "DEFAULT_EXPORT_COLORS",
    "ExportColors",
    "Rgb",
    "TextStyle",
    "adjust_brightness",
    "ansi_lines_to_html",
    "ansi_to_html",
    "color256_to_hex",
    "derive_export_colors",
    "escape_html",
    "get_luminance",
    "parse_color",
]

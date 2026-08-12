"""Formatting helpers for keybinding hints.

Ported from ``packages/coding-agent/src/modes/interactive/components/keybinding-hints.ts``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from pi_tui.keybindings import get_keybindings

from ..theme.theme import theme


@dataclass
class KeyTextFormatOptions:
    capitalize: bool = False


def _format_key_part(part: str, options: KeyTextFormatOptions) -> str:
    display_part = "option" if sys.platform == "darwin" and part.lower() == "alt" else part
    if not options.capitalize:
        return display_part
    # JS `charAt(0).toUpperCase() + slice(1)` only touches the first character,
    # unlike Python's `str.capitalize`, which also lowercases the rest.
    return display_part[:1].upper() + display_part[1:]


def format_key_text(key: str, options: KeyTextFormatOptions | None = None) -> str:
    options = options or KeyTextFormatOptions()
    return "/".join(
        "+".join(_format_key_part(part, options) for part in alternative.split("+")) for alternative in key.split("/")
    )


def _format_keys(keys: list[str], options: KeyTextFormatOptions | None = None) -> str:
    if len(keys) == 0:
        return ""
    return format_key_text("/".join(keys), options)


def key_text(keybinding: str) -> str:
    return _format_keys(get_keybindings().get_keys(keybinding))


def key_display_text(keybinding: str) -> str:
    return _format_keys(get_keybindings().get_keys(keybinding), KeyTextFormatOptions(capitalize=True))


def key_hint(keybinding: str, description: str) -> str:
    return theme.fg("dim", key_text(keybinding)) + theme.fg("muted", f" {description}")


def raw_key_hint(key: str, description: str) -> str:
    return theme.fg("dim", format_key_text(key)) + theme.fg("muted", f" {description}")

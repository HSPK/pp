"""Strip ANSI escape sequences from terminal output.

Port of `packages/coding-agent/src/utils/ansi.ts`, itself adapted from the
`ansi-regex`/`strip-ansi` npm packages (MIT licensed, Sindre Sorhus). Only the
`stripAnsi` export is used by the coding agent's bash executor.
"""

from __future__ import annotations

import re

_ST = r"(?:\u0007|\u001B\u005C|\u009C)"
_OSC = rf"(?:\u001B\][\s\S]*?{_ST})"
_CSI = r"[\u001B\u009B][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]"
_ANSI_RE = re.compile(f"{_OSC}|{_CSI}")


def strip_ansi(value: str) -> str:
    """Remove ANSI escape sequences (CSI and OSC) from `value`."""
    if not isinstance(value, str):
        # Upstream mirrors chalk's `strip-ansi`, which rejects non-strings
        # rather than coercing them. Without this guard a `dict` would fall
        # through the fast path below and be returned unchanged.
        raise TypeError(f"Expected a `str`, got `{type(value).__name__}`")

    # Fast path: ANSI codes require ESC (7-bit) or CSI (8-bit) introducer.
    if "\u001b" not in value and "\u009b" not in value:
        return value
    return _ANSI_RE.sub("", value)

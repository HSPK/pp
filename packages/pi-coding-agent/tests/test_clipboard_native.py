"""Python port of `packages/coding-agent/test/clipboard-native.test.ts`.

`src/utils/clipboard-native.ts` has no Python counterpart. It loads
`@mariozechner/clipboard`, a native Node addon, by walking a list of
`require` roots (bundled root first, then the installed package root) and
returning the first one that resolves. Python has no `require` chain, no
bundler root, and no such addon; `utils/clipboard.py` therefore implements
only the fallback chain the TypeScript uses when the addon is missing --
platform command line tools, then OSC 52 -- which is what actually runs on
Linux upstream anyway, because the addon is deliberately skipped there. See
that module's docstring.

The two TypeScript cases only test the require-root fallback logic, so there
is nothing behavioral left to pin here. They are recorded below.

The parts of clipboard behavior that *are* shared are covered by
`tests/test_clipboard.py` and `tests/test_clipboard_image.py`.
"""

from __future__ import annotations

import pytest
from pi_coding_agent.utils import clipboard

_REASON = (
    "`src/utils/clipboard-native.ts` is not ported: it resolves the "
    "`@mariozechner/clipboard` native Node addon through a `require`-root "
    "chain, which has no Python equivalent (see utils/clipboard.py)."
)


def test_clipboard_module_has_no_native_loader() -> None:
    """Pins the documented boundary: no `load_clipboard_native` seam exists.

    If one is ever added, this fails and the two cases below must be written.
    """
    assert not hasattr(clipboard, "load_clipboard_native")


@pytest.mark.skip(reason=_REASON)
def test_falls_back_to_the_next_require_root() -> None:
    """`test("falls back to the next require root")`.

    With `[primary, fallback]` where `primary` throws, asserts
    `loadClipboardNative(...)` returns the fallback's module, and that *both*
    roots were called with the literal id `"@mariozechner/clipboard"`.
    """


@pytest.mark.skip(reason=_REASON)
def test_returns_null_when_no_require_root_can_load_clipboard() -> None:
    """`test("returns null when no require root can load clipboard")`.

    With a single throwing root, `loadClipboardNative([missing])` is `null`.
    """

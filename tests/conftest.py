"""Shared fixtures for the ``pi-coding-agent`` tests."""

from __future__ import annotations

import pytest
from pi_tui.terminal_image import TerminalCapabilities, reset_capabilities_cache, set_capabilities

from pi_coding_agent.core import output_guard


@pytest.fixture(autouse=True)
def _pin_terminal_capabilities() -> object:
    """Pin terminal capabilities so results do not depend on who runs the tests.

    ``get_capabilities()`` sniffs the ambient environment once and caches it,
    so renderers behave differently depending on the terminal the suite was
    launched from. ``TERM_PROGRAM=vscode`` reports ``hyperlinks=True``, which
    makes the path renderers wrap paths in OSC 8 escapes; assertions looking
    for a bare path then fail in VS Code's integrated terminal while passing
    in CI, which sets no ``TERM_PROGRAM``.

    Pinning the least-capable terminal keeps the rendered output plain, which
    is what these assertions are written against. Tests that care about a
    specific capability still call ``set_capabilities`` themselves; this only
    establishes the default and clears the cache afterwards.
    """
    set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
    yield
    reset_capabilities_cache()


@pytest.fixture(autouse=True)
def _reset_output_guard() -> object:
    """Undo ``output_guard``'s process-global stdout takeover after each test.

    ``cli.entry.main`` calls ``take_over_stdout()`` for every non-interactive
    run and never restores it, which is correct for a real CLI process: it
    exits. Tests call ``main()`` in-process, so the takeover survives into
    every later test in the same worker and pins ``_raw_stdout()`` to the
    stdout object pytest captured for the test that leaked it. Once pytest
    closes that capture, ``flush_raw_stdout()`` raises
    ``ValueError: I/O operation on closed file``.

    ``_write_tail`` leaks the same way: it is a task owned by the finished
    test's event loop, so awaiting it from a later test touches a dead loop.
    """
    yield
    output_guard.restore_stdout()
    output_guard._write_tail = None

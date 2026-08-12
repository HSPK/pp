"""Shared fixtures for the ``pi-coding-agent`` tests."""

from __future__ import annotations

import pytest
from pi_coding_agent.core import output_guard


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

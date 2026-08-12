"""Port of `resolveEscapeTimeoutMs` (`packages/tui/src/terminal.ts:112`).

This port previously hard-coded a 10 ms escape-reassembly window. That is the
exact case that breaks over SSH: legacy Alt+key arrives as ESC followed by
another byte, and on a high-latency transport the second byte lands after the
window closes, so every Alt+key is delivered as a bare Escape.
"""

from __future__ import annotations

from pi_tui.terminal import (
    DEFAULT_ESCAPE_TIMEOUT_MS,
    DEFAULT_SSH_ESCAPE_TIMEOUT_MS,
    resolve_escape_timeout_ms,
)


def test_defaults_to_the_local_terminal_window():
    assert resolve_escape_timeout_ms({}) == DEFAULT_ESCAPE_TIMEOUT_MS


def test_ssh_gets_a_longer_window():
    assert resolve_escape_timeout_ms({"SSH_CONNECTION": "1.2.3.4 22"}) == DEFAULT_SSH_ESCAPE_TIMEOUT_MS
    assert resolve_escape_timeout_ms({"SSH_TTY": "/dev/pts/0"}) == DEFAULT_SSH_ESCAPE_TIMEOUT_MS


def test_an_explicit_setting_wins_over_the_ssh_default():
    assert resolve_escape_timeout_ms({"PI_TUI_ESC_TIMEOUT": "55", "SSH_TTY": "/dev/pts/0"}) == 55


def test_non_positive_and_unparsable_values_fall_through():
    """`Number.isFinite(configured) && configured > 0` upstream.

    A typo'd or zero setting must not disable escape reassembly entirely; it
    falls back to the default the environment would have chosen anyway.
    """
    for value in ("0", "-1", "abc", "", "NaN", "Infinity"):
        assert resolve_escape_timeout_ms({"PI_TUI_ESC_TIMEOUT": value}) == DEFAULT_ESCAPE_TIMEOUT_MS

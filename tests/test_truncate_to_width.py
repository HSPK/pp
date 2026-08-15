"""Python port of `packages/coding-agent/test/truncate-to-width.test.ts`.

Not to be confused with `packages/tui/test/truncate-to-width.test.ts` (ported in
`packages/pi-tui/tests/test_utils.py`), which shares the file name but tests
different inputs. This file is the coding-agent suite's regression coverage for
lines whose visible width exceeds their character count -- the crash that
prompted it came from a status line containing `U+2714` and `U+203A`.
"""

from __future__ import annotations

from pi_tui.utils import truncate_to_width, visible_width


def test_should_truncate_messages_with_unicode_characters_correctly():
    message = '\u2714 script to run \u203a dev $ concurrently "vite" "node --import tsx ./'
    width = 67
    max_msg_width = width - 2  # Account for cursor

    truncated = truncate_to_width(message, max_msg_width)

    assert visible_width(truncated) <= max_msg_width


def test_should_handle_emoji_characters():
    message = "\U0001f389 Celebration! \U0001f680 Launch \U0001f4e6 Package ready for deployment now"
    width = 40
    max_msg_width = width - 2

    truncated = truncate_to_width(message, max_msg_width)

    assert visible_width(truncated) <= max_msg_width


def test_should_handle_mixed_ascii_and_wide_characters():
    message = "Hello \u4e16\u754c Test \u4f60\u597d More text here that is long"
    width = 30
    max_msg_width = width - 2

    truncated = truncate_to_width(message, max_msg_width)

    assert visible_width(truncated) <= max_msg_width


def test_should_not_truncate_messages_that_fit():
    message = "Short message"
    width = 50
    max_msg_width = width - 2

    truncated = truncate_to_width(message, max_msg_width)

    assert truncated == message
    assert visible_width(truncated) <= max_msg_width


def test_should_add_ellipsis_when_truncating():
    message = "This is a very long message that needs to be truncated"
    width = 30
    max_msg_width = width - 2

    truncated = truncate_to_width(message, max_msg_width)

    assert "..." in truncated
    assert visible_width(truncated) <= max_msg_width


def test_should_handle_the_exact_crash_case_from_issue_report():
    # Terminal width was 67, line had visible width 68. The problematic text
    # contained the U+2714 and U+203A characters.
    message = '\u2714 script to run \u203a dev $ concurrently "vite" "node --import tsx ./server.ts"'
    terminal_width = 67
    cursor_width = 2  # "> " or "  "
    max_msg_width = terminal_width - cursor_width

    truncated = truncate_to_width(message, max_msg_width)

    # The final line (cursor + message) must not exceed terminal width.
    assert visible_width(truncated) + cursor_width <= terminal_width

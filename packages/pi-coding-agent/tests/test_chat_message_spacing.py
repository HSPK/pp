"""Blank-row separation and expansion state when replaying messages into the chat.

Port of `addMessageToChat` (`interactive-mode.ts:3459`). Three divergences lived
here and all three are visible to the user rather than internal:

- summaries were appended with no leading `Spacer(1)`, so they butted directly
  against the message above;
- summaries and custom messages never received `set_expanded`, so the
  "expand tool output" setting did not reach them on replay;
- a custom message with `display=False` was rendered anyway.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pi_tui.components.spacer import Spacer


def _spacer_count(container) -> int:
    return sum(1 for c in container.children if isinstance(c, Spacer))


@pytest.mark.asyncio
async def test_a_summary_is_separated_from_the_message_above(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`interactive-mode.ts:3490,3496` add a blank row before each summary."""
    from test_interactive_mode import _make_mode  # reuse the real harness

    mode, _terminal = await _make_mode(tmp_path, monkeypatch)
    try:
        await mode.init()
        before = _spacer_count(mode.chat_container)

        mode._add_message_to_chat(
            type("BranchSummary", (), {"role": "branchSummary", "summary": "s", "details": None})()
        )

        assert _spacer_count(mode.chat_container) == before + 1
    finally:
        await mode.shutdown()


@pytest.mark.asyncio
async def test_a_custom_message_that_opts_out_of_display_draws_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from test_interactive_mode import _make_mode

    mode, _terminal = await _make_mode(tmp_path, monkeypatch)
    try:
        await mode.init()
        before = len(mode.chat_container.children)

        mode._add_message_to_chat(
            type("Custom", (), {"role": "custom", "display": False, "customType": "x", "content": ""})()
        )

        assert len(mode.chat_container.children) == before
    finally:
        await mode.shutdown()

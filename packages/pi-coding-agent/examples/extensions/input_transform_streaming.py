"""Streaming-Aware Input Gate.

Python port of `packages/coding-agent/examples/extensions/input-transform-streaming.ts`.

Demonstrates `event.streaming_behavior` to skip expensive pre-processing
during mid-stream steering, where low latency matters.

This extension prepends `git diff --stat` output when the user mentions
file changes, giving the model immediate context. During steering the
exec call is skipped so the correction reaches the model without delay.

Start pi with this extension::

    pi -e ./examples/extensions/input_transform_streaming.py
"""

from __future__ import annotations

import re

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import (
    ExtensionContext,
    InputEvent,
    InputEventResult,
)

TRIGGER = re.compile(r"\b(changes?|diff|modified)\b", re.IGNORECASE)


def pi_extension(pi: ExtensionAPI) -> None:
    async def on_input(event: InputEvent, ctx: ExtensionContext) -> InputEventResult | None:
        # During steering, skip the exec call -- corrections should be fast.
        if event.streaming_behavior == "steer":
            return InputEventResult(action="continue")

        if not TRIGGER.search(event.text):
            return InputEventResult(action="continue")

        result = await pi.exec("git", ["diff", "--stat"])
        if result.code != 0 or not result.stdout.strip():
            return InputEventResult(action="continue")

        return InputEventResult(
            action="transform",
            text=f"{event.text}\n\nCurrent uncommitted changes:\n```\n{result.stdout.strip()}\n```",
        )

    pi.on("input", on_input)

"""Combine stdin, ``@file`` text and the first CLI message into one prompt.

Ported from ``packages/coding-agent/src/cli/initial-message.ts``.

Order matters and matches the TS: piped stdin first, then inlined file
contents, then the user's first message argument. The first message is
*consumed* (removed from ``parsed.messages``) so the caller does not send it
twice; the rest stay queued as follow-up turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pi_coding_agent.cli.args import Args


@dataclass
class InitialMessageResult:
    initial_message: str | None = None
    initial_images: list[Any] | None = None


def build_initial_message(
    parsed: Args,
    *,
    file_text: str | None = None,
    file_images: list[Any] | None = None,
    stdin_content: str | None = None,
) -> InitialMessageResult:
    parts: list[str] = []
    if stdin_content is not None:
        parts.append(stdin_content)
    if file_text:
        parts.append(file_text)

    if parsed.messages:
        parts.append(parsed.messages.pop(0))

    return InitialMessageResult(
        initial_message="".join(parts) if parts else None,
        initial_images=file_images if file_images else None,
    )


__all__ = ["InitialMessageResult", "build_initial_message"]

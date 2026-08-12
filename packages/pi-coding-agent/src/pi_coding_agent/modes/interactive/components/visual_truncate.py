"""Truncate text to a number of *visual* (wrapped) lines.

Ported from ``packages/coding-agent/src/modes/interactive/components/visual-truncate.ts``.
Shared by the tool-execution and bash-execution components so both truncate
identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pi_tui.components.text import Text


@dataclass
class VisualTruncateResult:
    visual_lines: list[str] = field(default_factory=list)
    skipped_count: int = 0


def truncate_to_visual_lines(
    text: str,
    max_visual_lines: int,
    width: int,
    padding_x: int = 0,
) -> VisualTruncateResult:
    """Keep the last ``max_visual_lines`` wrapped lines of ``text``.

    ``padding_x`` should be 0 when the result goes into a ``Box`` (which adds
    its own padding) and 1 for a plain ``Container``.
    """
    if not text:
        return VisualTruncateResult([], 0)

    all_visual_lines = Text(text, padding_x, 0).render(width)

    if len(all_visual_lines) <= max_visual_lines:
        return VisualTruncateResult(all_visual_lines, 0)

    return VisualTruncateResult(
        all_visual_lines[len(all_visual_lines) - max_visual_lines :],
        len(all_visual_lines) - max_visual_lines,
    )

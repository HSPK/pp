"""Apply extension-registered Markdown transformers.

Ported from ``packages/coding-agent/src/modes/interactive/components/markdown-transform.ts``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

MessageType = Literal["user", "assistant", "custom"]


@dataclass
class MarkdownTransformContext:
    message_type: MessageType
    is_streaming: bool
    available_width: int


MarkdownTransformer = Callable[[str, MarkdownTransformContext], Any]


def create_markdown_transform(
    message_type: MessageType,
    is_streaming: bool,
    transformers: Sequence[MarkdownTransformer],
) -> Callable[[str, int], str]:
    def transform(markdown: str, available_width: int) -> str:
        return apply_markdown_transformers(
            markdown,
            MarkdownTransformContext(message_type, is_streaming, available_width),
            transformers,
        )

    return transform


def apply_markdown_transformers(
    markdown: str,
    context: MarkdownTransformContext,
    transformers: Sequence[MarkdownTransformer],
) -> str:
    transformed_markdown = markdown
    for transformer in transformers:
        try:
            transformed = transformer(transformed_markdown, context)
        except Exception:
            # Keep the current Markdown and continue with the next transformer.
            continue
        if isinstance(transformed, str):
            transformed_markdown = transformed
    return transformed_markdown

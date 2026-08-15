"""Render highlight.js HTML output through a terminal formatter theme.

Python port of `packages/coding-agent/src/utils/syntax-highlight.ts`.

Only `render_highlighted_html` is ported. The `highlight` and
`supports_language` wrappers drive highlight.js, which has no Python
equivalent in this port; `render_highlighted_html` is pure and depends only on
`utils.html.decode_html_entity_at`, so it is ported as-is and can be fed by any
producer of highlight.js-shaped markup.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from pi_coding_agent.utils.html import decode_html_entity_at

HighlightFormatter = Callable[[str], str]
HighlightTheme = dict[str, HighlightFormatter]

_SPAN_CLOSE = "</span>"
_HIGHLIGHT_CLASS_PREFIX = "hljs-"
_CLASS_ATTRIBUTE = re.compile(r"\sclass\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")
_WHITESPACE = re.compile(r"\s+")
_SPAN_TAG_NAME_TERMINATORS = frozenset(">  \t\n\r")


def _get_scope_from_span_tag(tag: str) -> str | None:
    match = _CLASS_ATTRIBUTE.search(tag)
    if match is None:
        return None
    class_value = match.group(1) if match.group(1) is not None else match.group(2)
    if not class_value:
        return None

    for class_name in _WHITESPACE.split(class_value):
        if class_name.startswith(_HIGHLIGHT_CLASS_PREFIX):
            return class_name[len(_HIGHLIGHT_CLASS_PREFIX) :]

    return None


def _get_scope_formatter(scope: str, theme: HighlightTheme) -> HighlightFormatter | None:
    exact = theme.get(scope)
    if exact is not None:
        return exact

    dot_index = scope.find(".")
    if dot_index != -1:
        prefix_formatter = theme.get(scope[:dot_index])
        if prefix_formatter is not None:
            return prefix_formatter

    dash_index = scope.find("-")
    if dash_index != -1:
        prefix_formatter = theme.get(scope[:dash_index])
        if prefix_formatter is not None:
            return prefix_formatter

    return None


def _get_active_formatter(scopes: Sequence[str | None], theme: HighlightTheme) -> HighlightFormatter | None:
    for scope in reversed(scopes):
        if not scope:
            continue
        formatter = _get_scope_formatter(scope, theme)
        if formatter is not None:
            return formatter
    return theme.get("default")


def _is_span_open_tag_start(html: str, index: int) -> bool:
    if not html.startswith("<span", index):
        return False
    next_index = index + len("<span")
    if next_index >= len(html):
        return False
    return html[next_index] in _SPAN_TAG_NAME_TERMINATORS


def render_highlighted_html(html: str, theme: HighlightTheme | None = None) -> str:
    """Convert highlight.js markup to terminal text using ``theme``'s formatters."""
    theme = theme if theme is not None else {}
    output: list[str] = []
    text_buffer: list[str] = []
    scopes: list[str | None] = []

    def flush_text() -> None:
        if not text_buffer:
            return
        text = "".join(text_buffer)
        text_buffer.clear()
        formatter = _get_active_formatter(scopes, theme)
        output.append(formatter(text) if formatter is not None else text)

    index = 0
    length = len(html)
    while index < length:
        if _is_span_open_tag_start(html, index):
            tag_end_index = html.find(">", index + 5)
            if tag_end_index != -1:
                flush_text()
                scopes.append(_get_scope_from_span_tag(html[index : tag_end_index + 1]))
                index = tag_end_index + 1
                continue

        if html.startswith(_SPAN_CLOSE, index):
            flush_text()
            if scopes:
                scopes.pop()
            index += len(_SPAN_CLOSE)
            continue

        if html[index] == "&":
            decoded = decode_html_entity_at(html, index)
            if decoded is not None:
                text_buffer.append(decoded.text)
                index += decoded.length
                continue

        text_buffer.append(html[index])
        index += 1

    flush_text()
    return "".join(output)


__all__ = ["HighlightFormatter", "HighlightTheme", "render_highlighted_html"]

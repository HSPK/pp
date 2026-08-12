"""Tool HTML renderer for custom tools in HTML export.

Python port of `packages/coding-agent/src/core/export-html/tool-renderer.ts`.

A tool that ships TUI renderers should look the same in an exported HTML
transcript as it does in the terminal. This renderer invokes those same
`render_call`/`render_result` hooks off-screen at a fixed width and converts
their ANSI output to HTML. Tools without renderers return `None` so the
exporter falls back to its structured rendering.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...modes.interactive.components.tool_execution import (
    ToolRenderContext,
    ToolRenderResultOptions,
    ToolResult,
)
from .ansi_to_html import ansi_lines_to_html

DEFAULT_RENDER_WIDTH = 100

_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[\d;]*m")


def _is_blank_rendered_line(line: str) -> bool:
    return not _ANSI_ESCAPE_PATTERN.sub("", line).strip()


def _trim_rendered_result_lines(lines: list[str]) -> list[str]:
    """Drop leading and trailing lines that carry only styling, not text."""
    start = 0
    end = len(lines)
    while start < end and _is_blank_rendered_line(lines[start]):
        start += 1
    while end > start and _is_blank_rendered_line(lines[end - 1]):
        end -= 1
    return lines[start:end]


@dataclass
class RenderedToolResult:
    """A tool result rendered at both detail levels.

    `collapsed` is omitted when it would be identical to `expanded`, matching
    upstream's conditional spread.
    """

    expanded: str
    collapsed: str | None = None


@dataclass
class ToolHtmlRenderer:
    """Renders tool calls and results to HTML via their TUI renderers.

    `get_tool_definition` resolves a tool name to its definition; `theme` and
    `cwd` are handed to the renderers unchanged.
    """

    get_tool_definition: Callable[[str], Any]
    theme: Any
    cwd: str
    width: int = DEFAULT_RENDER_WIDTH
    _call_components: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _result_components: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _states: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _args: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _get_state(self, tool_call_id: str) -> dict[str, Any]:
        return self._states.setdefault(tool_call_id, {})

    def _create_render_context(
        self,
        tool_call_id: str,
        last_component: Any,
        expanded: bool,
        is_partial: bool,
        is_error: bool,
    ) -> ToolRenderContext:
        return ToolRenderContext(
            args=self._args.get(tool_call_id),
            tool_call_id=tool_call_id,
            invalidate=lambda: None,
            last_component=last_component,
            state=self._get_state(tool_call_id),
            cwd=self.cwd,
            execution_started=True,
            args_complete=True,
            is_partial=is_partial,
            expanded=expanded,
            show_images=False,
            is_error=is_error,
        )

    def render_call(self, tool_call_id: str, tool_name: str, args: Any) -> str | None:
        """Render a tool call. `None` when the tool has no custom renderer or rendering failed."""
        try:
            self._args[tool_call_id] = args
            definition = self.get_tool_definition(tool_name)
            renderer = getattr(definition, "render_call", None) if definition is not None else None
            if renderer is None:
                return None

            component = renderer(
                args,
                self.theme,
                self._create_render_context(tool_call_id, self._call_components.get(tool_call_id), False, True, False),
            )
            self._call_components[tool_call_id] = component
            return ansi_lines_to_html(component.render(self.width))
        except Exception:
            return None

    def render_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: list[Any],
        details: Any,
        is_error: bool,
    ) -> RenderedToolResult | None:
        """Render a tool result collapsed and expanded. `None` when there is no custom renderer."""
        try:
            definition = self.get_tool_definition(tool_name)
            renderer = getattr(definition, "render_result", None) if definition is not None else None
            if renderer is None:
                return None

            tool_result = ToolResult(content=result, details=details, is_error=is_error)

            collapsed_component = renderer(
                tool_result,
                ToolRenderResultOptions(expanded=False, is_partial=False),
                self.theme,
                self._create_render_context(
                    tool_call_id, self._result_components.get(tool_call_id), False, False, is_error
                ),
            )
            self._result_components[tool_call_id] = collapsed_component
            collapsed = ansi_lines_to_html(_trim_rendered_result_lines(collapsed_component.render(self.width)))

            expanded_component = renderer(
                tool_result,
                ToolRenderResultOptions(expanded=True, is_partial=False),
                self.theme,
                self._create_render_context(
                    tool_call_id, self._result_components.get(tool_call_id), True, False, is_error
                ),
            )
            self._result_components[tool_call_id] = expanded_component
            expanded = ansi_lines_to_html(_trim_rendered_result_lines(expanded_component.render(self.width)))

            return RenderedToolResult(
                expanded=expanded,
                collapsed=collapsed if collapsed and collapsed != expanded else None,
            )
        except Exception:
            return None


def create_tool_html_renderer(
    get_tool_definition: Callable[[str], Any],
    theme: Any,
    cwd: str,
    width: int = DEFAULT_RENDER_WIDTH,
) -> ToolHtmlRenderer:
    """Build a :class:`ToolHtmlRenderer`. Mirrors `createToolHtmlRenderer`."""
    return ToolHtmlRenderer(get_tool_definition=get_tool_definition, theme=theme, cwd=cwd, width=width)


__all__ = [
    "DEFAULT_RENDER_WIDTH",
    "RenderedToolResult",
    "ToolHtmlRenderer",
    "create_tool_html_renderer",
]

"""Tool call/result transcript row.

Ported from ``packages/coding-agent/src/modes/interactive/components/tool-execution.ts``.

Renderer hooks are supported: a ``ToolDefinition`` may carry ``render_call`` /
``render_result`` / ``render_shell`` attributes, supplied either by the
built-in per-tool renderers in ``tools/__init__.py`` or by extensions. A tool
with no renderers falls back to the generic composition path -- tool name,
pretty-printed arguments and text output -- exactly as the TypeScript version
does.

Kitty image conversion (``utils/image-convert.ts``) is ported: the Kitty
graphics protocol only accepts PNG, so non-PNG tool result images are decoded
and re-encoded in the background and swapped in once ready, exactly as
``maybeConvertImagesForKitty`` does. Until a conversion completes the block is
skipped, matching the TypeScript behaviour.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pi_agent.types import AgentToolResult
from pi_tui.component import Component, Container
from pi_tui.components.box import Box
from pi_tui.components.image import Image, ImageOptions, ImageTheme
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.tasks import spawn
from pi_tui.terminal_image import get_capabilities

from ....tools.render_utils import get_text_output
from ....utils.image_convert import convert_to_png
from ..theme.theme import theme

if TYPE_CHECKING:
    from pi_tui.tui import TuiBase


@dataclass
class ToolExecutionOptions:
    show_images: bool | None = None
    image_width_cells: int | None = None


@dataclass
class ToolRenderContext:
    args: Any
    tool_call_id: str
    invalidate: Callable[[], None]
    last_component: Component | None
    state: dict[str, Any]
    cwd: str
    execution_started: bool
    args_complete: bool
    is_partial: bool
    expanded: bool
    show_images: bool
    is_error: bool


@dataclass
class ToolRenderResultOptions:
    expanded: bool
    is_partial: bool


@dataclass
class ToolResult:
    content: list[Any] = field(default_factory=list)
    is_error: bool = False
    details: Any = None


def _block_get(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _block_mime_type(block: Any) -> Any:
    return _block_get(block, "mime_type") or _block_get(block, "mimeType")


class ToolExecutionComponent(Container):
    def __init__(
        self,
        tool_name: str,
        tool_call_id: str,
        args: Any,
        options: ToolExecutionOptions | None = None,
        tool_definition: Any = None,
        ui: TuiBase | None = None,
        cwd: str = ".",
        built_in_tool_definition: Any = None,
    ) -> None:
        super().__init__()
        options = options or ToolExecutionOptions()
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.args = args
        self.tool_definition = tool_definition
        if built_in_tool_definition is None:
            # `interactive-mode.ts`'s component looks the built-in renderers up
            # by tool name (`tool-execution.ts:57`) rather than having them
            # threaded in, so every call site gets them without changing.
            from pi_coding_agent.tools import create_all_tool_definitions

            built_in_tool_definition = create_all_tool_definitions(cwd).get(tool_name)
        self.built_in_tool_definition = built_in_tool_definition
        self.show_images = True if options.show_images is None else options.show_images
        self.image_width_cells = 60 if options.image_width_cells is None else options.image_width_cells
        self.ui = ui
        self.cwd = cwd

        self.renderer_state: dict[str, Any] = {}
        self._call_renderer_component: Component | None = None
        self._result_renderer_component: Component | None = None
        self._image_components: list[Image] = []
        self._image_spacers: list[Spacer] = []
        self._converted_images: dict[int, dict[str, str]] = {}
        self.expanded = False
        self.is_partial = True
        self.execution_started = False
        self.args_complete = False
        self.result: ToolResult | None = None
        self._hide_component = False

        self.add_child(Spacer(1))

        # All three shells are created up front: `content_box` for the default
        # renderer composition, `self_render_container` for tools that draw
        # their own framing, `content_text` for the generic fallback.
        self.content_box = Box(1, 1, lambda text: theme.bg("toolPendingBg", text))
        self.content_text = Text("", 1, 1, lambda text: theme.bg("toolPendingBg", text))
        self.self_render_container = Container()

        if self._has_renderer_definition():
            self.add_child(self.self_render_container if self._get_render_shell() == "self" else self.content_box)
        else:
            self.add_child(self.content_text)

        self._update_display()

    # -- renderer resolution ------------------------------------------------

    def _pick(self, attribute: str) -> Any:
        built_in = getattr(self.built_in_tool_definition, attribute, None)
        extension = getattr(self.tool_definition, attribute, None)
        if self.built_in_tool_definition is None:
            return extension
        if self.tool_definition is None:
            return built_in
        return extension if extension is not None else built_in

    def _get_call_renderer(self) -> Any:
        return self._pick("render_call")

    def _get_result_renderer(self) -> Any:
        return self._pick("render_result")

    def _has_renderer_definition(self) -> bool:
        return self.built_in_tool_definition is not None or self.tool_definition is not None

    def _get_render_shell(self) -> str:
        return self._pick("render_shell") or "default"

    def _get_render_context(self, last_component: Component | None) -> ToolRenderContext:
        def invalidate() -> None:
            self.invalidate()
            if self.ui is not None:
                self.ui.request_render()

        return ToolRenderContext(
            args=self.args,
            tool_call_id=self.tool_call_id,
            invalidate=invalidate,
            last_component=last_component,
            state=self.renderer_state,
            cwd=self.cwd,
            execution_started=self.execution_started,
            args_complete=self.args_complete,
            is_partial=self.is_partial,
            expanded=self.expanded,
            show_images=self.show_images,
            is_error=self.result.is_error if self.result is not None else False,
        )

    def _create_call_fallback(self) -> Component:
        return Text(theme.fg("toolTitle", theme.bold(self.tool_name)), 0, 0)

    def _create_result_fallback(self) -> Component | None:
        output = self._get_text_output()
        if not output:
            return None
        return Text(theme.fg("toolOutput", output), 0, 0)

    # -- state updates ------------------------------------------------------

    def update_args(self, args: Any) -> None:
        self.args = args
        self._update_display()

    def mark_execution_started(self) -> None:
        self.execution_started = True
        self._update_display()
        if self.ui is not None:
            self.ui.request_render()

    def set_args_complete(self) -> None:
        self.args_complete = True
        self._update_display()
        if self.ui is not None:
            self.ui.request_render()

    def update_result(self, result: ToolResult, is_partial: bool = False) -> None:
        self.result = result
        self.is_partial = is_partial
        self._update_display()
        self._maybe_convert_images_for_kitty()

    def _maybe_convert_images_for_kitty(self) -> None:
        """Port of `maybeConvertImagesForKitty`.

        The Kitty graphics protocol only accepts PNG, so a tool returning a JPEG
        would otherwise be dropped from the transcript. Conversion runs off the
        render path and each result is cached by block index, matching the
        TypeScript `convertToPng(...).then(...)` shape.
        """
        if get_capabilities().images != "kitty" or self.result is None:
            return

        image_blocks = [block for block in self.result.content if _block_get(block, "type") == "image"]
        for index, image in enumerate(image_blocks):
            data = _block_get(image, "data")
            mime_type = _block_mime_type(image)
            if not data or not mime_type or mime_type == "image/png" or index in self._converted_images:
                continue

            async def convert(index: int = index, data: str = data, mime_type: str = mime_type) -> None:
                converted = await asyncio.to_thread(convert_to_png, data, mime_type)
                if converted is None:
                    return
                self._converted_images[index] = converted
                self._update_display()
                if self.ui is not None:
                    self.ui.request_render()

            spawn(convert())

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self._update_display()

    def set_show_images(self, show: bool) -> None:
        self.show_images = show
        self._update_display()

    def set_image_width_cells(self, width: int) -> None:
        self.image_width_cells = max(1, math.floor(width))
        self._update_display()

    def invalidate(self) -> None:
        super().invalidate()
        self._update_display()

    # -- rendering ----------------------------------------------------------

    def render(self, width: int) -> list[str]:
        if self._hide_component:
            return []

        if self._has_renderer_definition() and self._get_render_shell() == "self":
            content_lines = self.self_render_container.render(width)
            if len(content_lines) == 0 and len(self._image_components) == 0:
                return []

            lines: list[str] = []
            if len(content_lines) > 0:
                lines.append("")
                lines.extend(content_lines)
            for index, image_component in enumerate(self._image_components):
                if index < len(self._image_spacers):
                    lines.extend(self._image_spacers[index].render(width))
                lines.extend(image_component.render(width))
            return lines

        return super().render(width)

    def _background_fn(self) -> Callable[[str], str]:
        if self.is_partial:
            return lambda text: theme.bg("toolPendingBg", text)
        if self.result is not None and self.result.is_error:
            return lambda text: theme.bg("toolErrorBg", text)
        return lambda text: theme.bg("toolSuccessBg", text)

    def _render_call_into(self, container: Container | Box) -> bool:
        call_renderer = self._get_call_renderer()
        if call_renderer is None:
            container.add_child(self._create_call_fallback())
            return True
        try:
            component = call_renderer(self.args, theme, self._get_render_context(self._call_renderer_component))
        except Exception:
            self._call_renderer_component = None
            container.add_child(self._create_call_fallback())
            return True
        self._call_renderer_component = component
        container.add_child(component)
        return True

    def _render_result_into(self, container: Container | Box, has_content: bool) -> bool:
        result_renderer = self._get_result_renderer()
        if result_renderer is None:
            component = self._create_result_fallback()
            if component is not None:
                container.add_child(component)
                return True
            return has_content
        try:
            # TypeScript passes only `{ content, details }` here (the shape of
            # `AgentToolResult<T>`), not the internal `isError` flag tracked by
            # this component -- renderers get error state via `context.isError`.
            component = result_renderer(
                AgentToolResult(content=self.result.content, details=self.result.details),
                ToolRenderResultOptions(expanded=self.expanded, is_partial=self.is_partial),
                theme,
                self._get_render_context(self._result_renderer_component),
            )
        except Exception:
            self._result_renderer_component = None
            fallback = self._create_result_fallback()
            if fallback is not None:
                container.add_child(fallback)
                return True
            return has_content
        self._result_renderer_component = component
        container.add_child(component)
        return True

    def _update_display(self) -> None:
        background_fn = self._background_fn()
        has_content = False
        self._hide_component = False

        if self._has_renderer_definition():
            render_container = self.self_render_container if self._get_render_shell() == "self" else self.content_box
            if isinstance(render_container, Box):
                render_container.set_bg_fn(background_fn)
            render_container.clear()

            has_content = self._render_call_into(render_container)
            if self.result is not None:
                has_content = self._render_result_into(render_container, has_content)
        else:
            self.content_text.set_custom_bg_fn(background_fn)
            self.content_text.set_text(self._format_tool_execution())
            has_content = True

        for image in self._image_components:
            self.remove_child(image)
        self._image_components = []
        for spacer in self._image_spacers:
            self.remove_child(spacer)
        self._image_spacers = []

        if self.result is not None:
            image_blocks = [block for block in self.result.content if _block_get(block, "type") == "image"]
            capabilities = get_capabilities()
            for index, image in enumerate(image_blocks):
                converted = self._converted_images.get(index)
                data = converted["data"] if converted else _block_get(image, "data")
                mime_type = converted["mimeType"] if converted else _block_mime_type(image)
                if not (capabilities.images and self.show_images and data and mime_type):
                    continue
                # Only PNG can go over the Kitty protocol. Non-PNG blocks are
                # skipped until `_maybe_convert_images_for_kitty` has produced
                # a converted copy.
                if capabilities.images == "kitty" and mime_type != "image/png":
                    continue

                spacer = Spacer(1)
                self.add_child(spacer)
                self._image_spacers.append(spacer)
                image_component = Image(
                    data,
                    mime_type,
                    ImageTheme(fallback_color=lambda s: theme.fg("toolOutput", s)),
                    ImageOptions(max_width_cells=self.image_width_cells),
                )
                self._image_components.append(image_component)
                self.add_child(image_component)

        if self._has_renderer_definition() and not has_content and len(self._image_components) == 0:
            self._hide_component = True

    def _get_text_output(self) -> str:
        return get_text_output(self.result, self.show_images)

    def _format_tool_execution(self) -> str:
        text = theme.fg("toolTitle", theme.bold(self.tool_name))
        content = json.dumps(self.args, indent=2, ensure_ascii=False) if self.args is not None else None
        if content:
            text += f"\n\n{content}"
        output = self._get_text_output()
        if output:
            text += f"\n{output}"
        return text


__all__ = [
    "ToolExecutionComponent",
    "ToolExecutionOptions",
    "ToolRenderContext",
    "ToolRenderResultOptions",
    "ToolResult",
]

"""Image component, ported from `packages/tui/src/components/image.ts`.

Renders a base64-encoded image inline (Kitty graphics protocol or iTerm2),
falling back to bracketed text (`imageFallback`) when the terminal does not
support either protocol. See `terminal_image.py` for the encoding boundary:
this component never decodes pixels, it only forwards base64 image bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pi_tui.component import Component
from pi_tui.terminal_image import (
    ImageDimensions,
    ImageRenderOptions,
    allocate_image_id,
    get_capabilities,
    get_cell_dimensions,
    get_image_dimensions,
    image_fallback,
    render_image,
)
from pi_tui.utils import truncate_to_width


@dataclass
class ImageTheme:
    fallback_color: Callable[[str], str]


@dataclass
class ImageOptions:
    max_width_cells: int | None = None
    max_height_cells: int | None = None
    filename: str | None = None
    """Kitty image ID. If provided, reuses this ID (for animations/updates)."""
    image_id: int | None = None


class Image(Component):
    def __init__(
        self,
        base64_data: str,
        mime_type: str,
        theme: ImageTheme,
        options: ImageOptions | None = None,
        dimensions: ImageDimensions | None = None,
    ) -> None:
        self._base64_data = base64_data
        self._mime_type = mime_type
        self._theme = theme
        self._options = options if options is not None else ImageOptions()
        self._dimensions = dimensions or get_image_dimensions(base64_data, mime_type) or ImageDimensions(800, 600)
        self._image_id = self._options.image_id

        self._cached_lines: list[str] | None = None
        self._cached_width: int | None = None

    def get_image_id(self) -> int | None:
        """Get the Kitty image ID used by this image (if any)."""
        return self._image_id

    def invalidate(self) -> None:
        self._cached_lines = None
        self._cached_width = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_width == width:
            return self._cached_lines

        max_width = max(1, min(width - 2, self._options.max_width_cells or 60))
        cell_dimensions = get_cell_dimensions()
        default_max_height = max(1, -(-(max_width * cell_dimensions.width_px) // cell_dimensions.height_px))
        max_height = (
            self._options.max_height_cells if self._options.max_height_cells is not None else default_max_height
        )

        caps = get_capabilities()
        lines: list[str]

        if caps.images:
            if caps.images == "kitty" and self._image_id is None:
                self._image_id = allocate_image_id()

            result = render_image(
                self._base64_data,
                self._dimensions,
                ImageRenderOptions(
                    max_width_cells=max_width,
                    max_height_cells=max_height,
                    image_id=self._image_id,
                    move_cursor=False,
                ),
            )

            if result:
                # Store the image ID for later cleanup.
                if result.image_id:
                    self._image_id = result.image_id

                if caps.images == "kitty":
                    # For Kitty: C=1 prevents cursor movement.
                    # Don't need the cursor movement.
                    lines = [result.sequence]

                    # Return `rows` lines so TUI accounts for image height.
                    for _ in range(result.rows - 1):
                        lines.append("")
                else:
                    # Return `rows` lines so TUI accounts for image height.
                    # First (rows-1) lines are empty and cleared before the image is drawn.
                    # Last line: move cursor back up, draw the image, then move back down
                    # so TUI cursor accounting stays inside the scroll area.
                    lines = []
                    for _ in range(result.rows - 1):
                        lines.append("")
                    row_offset = result.rows - 1
                    move_up = f"\x1b[{row_offset}A" if row_offset > 0 else ""
                    lines.append(move_up + result.sequence)
            else:
                fallback = image_fallback(self._mime_type, self._dimensions, self._options.filename)
                lines = [truncate_to_width(self._theme.fallback_color(fallback), width)]
        else:
            fallback = image_fallback(self._mime_type, self._dimensions, self._options.filename)
            lines = [truncate_to_width(self._theme.fallback_color(fallback), width)]

        self._cached_lines = lines
        self._cached_width = width

        return lines

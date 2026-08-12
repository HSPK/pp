"""Terminal image protocols, ported from `packages/tui/src/terminal-image.ts`.

This module implements the *encoding* side of inline terminal images: Kitty
graphics protocol framing, iTerm2 inline image framing, terminal capability
detection, and pure byte-level dimension parsing for PNG/JPEG/GIF/WebP
headers (no image-decoding library needed for those - they only read a
handful of bytes from the format's fixed header layout).

There is no pixel-decoding boundary to document here: unlike a renderer that
needs actual pixel data (e.g. for a sixel or half-block encoder), every
function in this module operates on already-base64-encoded image bytes
(read dimensions from headers, or pass the encoded bytes through to the
terminal's own protocol). Callers are responsible for supplying valid
base64-encoded image data; this module never decodes pixels itself.
"""

from __future__ import annotations

import base64
import binascii
import os
import random
import re
import struct
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ImageProtocol = Literal["kitty", "iterm2", None]


@dataclass
class TerminalCapabilities:
    images: ImageProtocol
    true_color: bool
    hyperlinks: bool


@dataclass(frozen=True)
class CellDimensions:
    width_px: int
    height_px: int


@dataclass(frozen=True)
class ImageDimensions:
    width_px: int
    height_px: int


@dataclass
class ImageRenderOptions:
    max_width_cells: int | None = None
    max_height_cells: int | None = None
    preserve_aspect_ratio: bool | None = None
    """Kitty image ID. If provided, reuses/replaces existing image with this ID."""
    image_id: int | None = None
    """Whether Kitty should apply its default cursor movement after placement."""
    move_cursor: bool | None = None


@dataclass
class RenderImageResult:
    sequence: str
    columns: int
    rows: int
    image_id: int | None = None


_cached_capabilities: TerminalCapabilities | None = None

# Default cell dimensions - updated by TUI when terminal responds to query.
_cell_dimensions = CellDimensions(width_px=9, height_px=18)


def get_cell_dimensions() -> CellDimensions:
    return _cell_dimensions


def set_cell_dimensions(dims: CellDimensions) -> None:
    global _cell_dimensions
    _cell_dimensions = dims


def _probe_tmux_hyperlinks() -> bool:
    """Checks whether the attached tmux client forwards OSC 8 hyperlinks to the
    outer terminal. tmux only re-emits them when its `client_termfeatures` lists
    `hyperlinks`, and strips them otherwise. On any error falls back to `False`.
    """
    try:
        result = subprocess.run(
            ["tmux", "display-message", "-p", "#{client_termfeatures}"],
            capture_output=True,
            text=True,
            timeout=0.25,
            check=False,
        )
        termfeatures = result.stdout
        return "hyperlinks" in [feature.strip() for feature in termfeatures.split(",")]
    except Exception:
        return False


def detect_capabilities(
    tmux_forwards_hyperlink: Callable[[], bool] = _probe_tmux_hyperlinks,
) -> TerminalCapabilities:
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    terminal_emulator = os.environ.get("TERMINAL_EMULATOR", "").lower()
    term = os.environ.get("TERM", "").lower()
    color_term = os.environ.get("COLORTERM", "").lower()
    has_true_color_hint = color_term in ("truecolor", "24bit")
    is_windows_console = sys.platform == "win32"

    # Emit OSC 8 hyperlinks only when tmux confirms it forwards.
    # Image protocols are unreliable under tmux, so leave `images: None`.
    if os.environ.get("TMUX") or term.startswith("tmux"):
        return TerminalCapabilities(images=None, true_color=has_true_color_hint, hyperlinks=tmux_forwards_hyperlink())

    # screen does not forward OSC 8 hyperlinks, so keep them off there.
    if term.startswith("screen"):
        return TerminalCapabilities(images=None, true_color=has_true_color_hint, hyperlinks=False)

    if os.environ.get("KITTY_WINDOW_ID") or term_program == "kitty":
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)

    if term_program == "ghostty" or "ghostty" in term or os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)

    if os.environ.get("WEZTERM_PANE") or term_program == "wezterm":
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)

    # Warp supports the Kitty graphics protocol and OSC 8 hyperlinks.
    if (
        term_program == "warpterminal"
        or os.environ.get("WARP_SESSION_ID")
        or os.environ.get("WARP_TERMINAL_SESSION_UUID")
    ):
        return TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)

    if os.environ.get("ITERM_SESSION_ID") or term_program == "iterm.app":
        return TerminalCapabilities(images="iterm2", true_color=True, hyperlinks=True)

    if os.environ.get("WT_SESSION"):
        return TerminalCapabilities(images=None, true_color=True, hyperlinks=True)

    if term_program == "vscode":
        return TerminalCapabilities(images=None, true_color=True, hyperlinks=True)

    if term_program == "alacritty":
        return TerminalCapabilities(images=None, true_color=True, hyperlinks=True)

    if terminal_emulator == "jetbrains-jediterm":
        return TerminalCapabilities(images=None, true_color=True, hyperlinks=False)

    # Windows Terminal does not always set WT_SESSION, for example when it hosts
    # a cmd.exe launched directly from Win+R. Modern Windows consoles support
    # truecolor; keep hyperlinks off unless we positively detected support above.
    if is_windows_console:
        return TerminalCapabilities(images=None, true_color=True, hyperlinks=False)

    # Unknown terminal: be conservative. OSC 8 is rendered invisibly as "just
    # text" on terminals that swallow it, which means the URL disappears from
    # the rendered output. Default to the legacy `text (url)` behavior unless we
    # have positively identified a hyperlink-capable terminal above.
    return TerminalCapabilities(images=None, true_color=has_true_color_hint, hyperlinks=False)


def get_capabilities() -> TerminalCapabilities:
    global _cached_capabilities
    if _cached_capabilities is None:
        _cached_capabilities = detect_capabilities()
    return _cached_capabilities


def reset_capabilities_cache() -> None:
    global _cached_capabilities
    _cached_capabilities = None


def set_capabilities(caps: TerminalCapabilities) -> None:
    """Override the cached capabilities. Useful in tests to exercise both code paths."""
    global _cached_capabilities
    _cached_capabilities = caps


KITTY_PREFIX = "\x1b_G"
ITERM2_PREFIX = "\x1b]1337;File="


def is_image_line(line: str) -> bool:
    # Fast path: sequence at line start (single-row images).
    if line.startswith(KITTY_PREFIX) or line.startswith(ITERM2_PREFIX):
        return True
    # Slow path: sequence elsewhere (multi-row images have cursor-up prefix).
    return KITTY_PREFIX in line or ITERM2_PREFIX in line


def allocate_image_id() -> int:
    """Generate a random image ID for the Kitty graphics protocol.

    Uses random IDs to avoid collisions between different module instances
    (e.g. main app vs extensions).
    """
    return random.randint(1, 0xFFFFFFFE)


def encode_kitty(
    base64_data: str,
    columns: int | None = None,
    rows: int | None = None,
    image_id: int | None = None,
    move_cursor: bool | None = None,
) -> str:
    """`move_cursor`: whether Kitty should apply its default cursor movement after
    placement. Default: `True`."""
    chunk_size = 4096

    params: list[str] = ["a=T", "f=100", "q=2"]
    if move_cursor is False:
        params.append("C=1")
    if columns:
        params.append(f"c={columns}")
    if rows:
        params.append(f"r={rows}")
    if image_id:
        params.append(f"i={image_id}")

    if len(base64_data) <= chunk_size:
        return f"\x1b_G{','.join(params)};{base64_data}\x1b\\"

    chunks: list[str] = []
    offset = 0
    is_first = True

    while offset < len(base64_data):
        chunk = base64_data[offset : offset + chunk_size]
        is_last = offset + chunk_size >= len(base64_data)

        if is_first:
            chunks.append(f"\x1b_G{','.join(params)},m=1;{chunk}\x1b\\")
            is_first = False
        elif is_last:
            chunks.append(f"\x1b_Gm=0;{chunk}\x1b\\")
        else:
            chunks.append(f"\x1b_Gm=1;{chunk}\x1b\\")

        offset += chunk_size

    return "".join(chunks)


def delete_kitty_image(image_id: int) -> str:
    """Delete a Kitty graphics image by ID.

    Uses uppercase 'I' to also free the image data.
    """
    return f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\"


def delete_all_kitty_images() -> str:
    """Delete all visible Kitty graphics images.

    Uses uppercase 'A' to also free the image data.
    """
    return "\x1b_Ga=d,d=A,q=2\x1b\\"


def delete_all_kitty_placements() -> str:
    """Delete all visible Kitty placements while retaining their uploaded image data."""
    return "\x1b_Ga=d,d=a,q=2\x1b\\"


def _decoded_byte_length(base64_data: str) -> int:
    padded = base64_data + "=" * (-len(base64_data) % 4)
    try:
        return len(base64.b64decode(padded))
    except (binascii.Error, ValueError):
        return 0


def encode_iterm2(
    base64_data: str,
    width: int | str | None = None,
    height: int | str | None = None,
    name: str | None = None,
    preserve_aspect_ratio: bool | None = None,
    inline: bool | None = None,
) -> str:
    params: list[str] = [
        f"inline={1 if inline is not False else 0}",
        f"size={_decoded_byte_length(base64_data)}",
    ]

    if width is not None:
        params.append(f"width={width}")
    if height is not None:
        params.append(f"height={height}")
    if name:
        name_base64 = base64.b64encode(name.encode()).decode()
        params.append(f"name={name_base64}")
    if preserve_aspect_ratio is False:
        params.append("preserveAspectRatio=0")

    return f"\x1b]1337;File={';'.join(params)}:{base64_data}\x07"


@dataclass
class ImageCellSize:
    columns: int
    rows: int


@dataclass
class KittyImageMetadata:
    image_id: int
    columns: int
    rows: int
    width_px: int
    height_px: int


@dataclass
class _RegisteredKittyImageMetadata(KittyImageMetadata):
    transmission_generation: int = 0


@dataclass
class KittyImagePlacement:
    image_id: int
    transmission_generation: int
    transmission_bytes: int
    estimated_decoded_bytes: int
    sequence: str
    replacement_line: str


_kitty_image_metadata: dict[int, _RegisteredKittyImageMetadata] = {}
_kitty_transmission_generation = 0


def register_kitty_image_metadata(metadata: KittyImageMetadata) -> None:
    global _kitty_transmission_generation
    _kitty_transmission_generation += 1
    _kitty_image_metadata.pop(metadata.image_id, None)
    _kitty_image_metadata[metadata.image_id] = _RegisteredKittyImageMetadata(
        image_id=metadata.image_id,
        columns=metadata.columns,
        rows=metadata.rows,
        width_px=metadata.width_px,
        height_px=metadata.height_px,
        transmission_generation=_kitty_transmission_generation,
    )
    if len(_kitty_image_metadata) > 1000:
        oldest_image_id = next(iter(_kitty_image_metadata))
        del _kitty_image_metadata[oldest_image_id]


def _get_registered_kitty_image_metadata(line: str) -> _RegisteredKittyImageMetadata | None:
    match = re.search(r"\x1b_G([^;]*);", line)
    if not match:
        return None
    controls = match.group(1)
    image_id_match = re.search(r"(?:^|,)i=(\d+)(?:,|$)", controls)
    if image_id_match is None:
        return None
    return _kitty_image_metadata.get(int(image_id_match.group(1)))


def get_kitty_image_metadata(line: str) -> KittyImageMetadata | None:
    metadata = _get_registered_kitty_image_metadata(line)
    if metadata is None:
        return None
    return KittyImageMetadata(
        image_id=metadata.image_id,
        columns=metadata.columns,
        rows=metadata.rows,
        width_px=metadata.width_px,
        height_px=metadata.height_px,
    )


_KITTY_PLACEMENT_CONTROL_KEYS = frozenset(
    {"i", "p", "x", "y", "w", "h", "X", "Y", "c", "r", "C", "U", "z", "P", "Q", "H", "V"}
)


def get_kitty_image_placement(line: str) -> KittyImagePlacement | None:
    """Build a placement-only command for an image line emitted by `render_image`."""
    match = re.search(r"\x1b_G([^;]*);", line)
    metadata = _get_registered_kitty_image_metadata(line)
    if not match or metadata is None:
        return None

    command_start = match.start()
    command_controls = match.group(1)
    transmission_end = 0
    while True:
        terminator = line.find("\x1b\\", command_start + len(KITTY_PREFIX))
        if terminator == -1:
            return None
        transmission_end = terminator + 2
        if not re.search(r"(?:^|,)m=1(?:,|$)", command_controls):
            break
        command_start = transmission_end
        if not line.startswith(KITTY_PREFIX, command_start):
            return None
        controls_end = line.find(";", command_start + len(KITTY_PREFIX))
        if controls_end == -1:
            return None
        command_controls = line[command_start + len(KITTY_PREFIX) : controls_end]

    controls = [
        control for control in match.group(1).split(",") if control.split("=", 1)[0] in _KITTY_PLACEMENT_CONTROL_KEYS
    ]
    sequence = f"\x1b_Ga=p,q=2,{','.join(controls)}\x1b\\"
    return KittyImagePlacement(
        image_id=metadata.image_id,
        transmission_generation=metadata.transmission_generation,
        transmission_bytes=transmission_end - match.start(),
        estimated_decoded_bytes=metadata.width_px * metadata.height_px * 4,
        sequence=sequence,
        replacement_line=f"{line[: match.start()]}{sequence}{line[transmission_end:]}",
    )


def crop_kitty_image_line(line: str, hidden_rows: int, visible_rows: int) -> str:
    metadata = get_kitty_image_metadata(line)
    match = re.search(r"\x1b_G([^;]*);", line)
    if metadata is None or not match or hidden_rows < 0 or hidden_rows >= metadata.rows or visible_rows <= 0:
        return line
    cropped_rows = min(visible_rows, metadata.rows - hidden_rows)
    if hidden_rows == 0 and cropped_rows == metadata.rows:
        return line
    source_y = (metadata.height_px * hidden_rows) // metadata.rows
    source_end = -(-(metadata.height_px * (hidden_rows + cropped_rows)) // metadata.rows)  # ceil division
    source_height = max(1, min(metadata.height_px, source_end) - source_y)
    controls = [control for control in match.group(1).split(",") if not re.match(r"^[yhr]=", control)]
    controls.extend([f"y={source_y}", f"h={source_height}", f"r={cropped_rows}"])
    return f"{line[: match.start()]}\x1b_G{','.join(controls)};{line[match.start() + len(match.group(0)) :]}"


def calculate_image_cell_size(
    image_dimensions: ImageDimensions,
    max_width_cells: int,
    max_height_cells: int | None = None,
    cell_dimensions: CellDimensions | None = None,
) -> ImageCellSize:
    cells = cell_dimensions if cell_dimensions is not None else CellDimensions(9, 18)
    max_width = max(1, int(max_width_cells))
    max_height = None if max_height_cells is None else max(1, int(max_height_cells))
    image_width = max(1, image_dimensions.width_px)
    image_height = max(1, image_dimensions.height_px)

    width_scale = (max_width * cells.width_px) / image_width
    height_scale = width_scale if max_height is None else (max_height * cells.height_px) / image_height
    scale = min(width_scale, height_scale)

    scaled_width_px = image_width * scale
    scaled_height_px = image_height * scale
    columns = -(-scaled_width_px // cells.width_px)  # ceil
    rows = -(-scaled_height_px // cells.height_px)  # ceil

    return ImageCellSize(
        columns=int(max(1, min(max_width, columns))),
        rows=int(max(1, rows if max_height is None else min(max_height, rows))),
    )


def calculate_image_rows(
    image_dimensions: ImageDimensions,
    target_width_cells: int,
    cell_dimensions: CellDimensions | None = None,
) -> int:
    return calculate_image_cell_size(image_dimensions, target_width_cells, None, cell_dimensions).rows


def get_png_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = base64.b64decode(base64_data + "=" * (-len(base64_data) % 4))
        if len(buffer) < 24:
            return None
        if buffer[0:4] != b"\x89PNG":
            return None
        width, height = struct.unpack(">II", buffer[16:24])
        return ImageDimensions(width_px=width, height_px=height)
    except Exception:
        return None


def get_jpeg_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = base64.b64decode(base64_data + "=" * (-len(base64_data) % 4))
        if len(buffer) < 2:
            return None
        if buffer[0] != 0xFF or buffer[1] != 0xD8:
            return None

        offset = 2
        while offset < len(buffer) - 9:
            if buffer[offset] != 0xFF:
                offset += 1
                continue

            marker = buffer[offset + 1]

            if 0xC0 <= marker <= 0xC2:
                (height,) = struct.unpack(">H", buffer[offset + 5 : offset + 7])
                (width,) = struct.unpack(">H", buffer[offset + 7 : offset + 9])
                return ImageDimensions(width_px=width, height_px=height)

            if offset + 3 >= len(buffer):
                return None
            (length,) = struct.unpack(">H", buffer[offset + 2 : offset + 4])
            if length < 2:
                return None
            offset += 2 + length

        return None
    except Exception:
        return None


def get_gif_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = base64.b64decode(base64_data + "=" * (-len(base64_data) % 4))
        if len(buffer) < 10:
            return None
        sig = buffer[0:6]
        if sig != b"GIF87a" and sig != b"GIF89a":
            return None
        width, height = struct.unpack("<HH", buffer[6:10])
        return ImageDimensions(width_px=width, height_px=height)
    except Exception:
        return None


def get_webp_dimensions(base64_data: str) -> ImageDimensions | None:
    try:
        buffer = base64.b64decode(base64_data + "=" * (-len(base64_data) % 4))
        if len(buffer) < 30:
            return None

        riff = buffer[0:4]
        webp = buffer[8:12]
        if riff != b"RIFF" or webp != b"WEBP":
            return None

        chunk = buffer[12:16]
        if chunk == b"VP8 ":
            if len(buffer) < 30:
                return None
            (raw_width,) = struct.unpack("<H", buffer[26:28])
            (raw_height,) = struct.unpack("<H", buffer[28:30])
            return ImageDimensions(width_px=raw_width & 0x3FFF, height_px=raw_height & 0x3FFF)
        elif chunk == b"VP8L":
            if len(buffer) < 25:
                return None
            (bits,) = struct.unpack("<I", buffer[21:25])
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return ImageDimensions(width_px=width, height_px=height)
        elif chunk == b"VP8X":
            if len(buffer) < 30:
                return None
            width = (buffer[24] | (buffer[25] << 8) | (buffer[26] << 16)) + 1
            height = (buffer[27] | (buffer[28] << 8) | (buffer[29] << 16)) + 1
            return ImageDimensions(width_px=width, height_px=height)

        return None
    except Exception:
        return None


def get_image_dimensions(base64_data: str, mime_type: str) -> ImageDimensions | None:
    if mime_type == "image/png":
        return get_png_dimensions(base64_data)
    if mime_type == "image/jpeg":
        return get_jpeg_dimensions(base64_data)
    if mime_type == "image/gif":
        return get_gif_dimensions(base64_data)
    if mime_type == "image/webp":
        return get_webp_dimensions(base64_data)
    return None


def render_image(
    base64_data: str,
    image_dimensions: ImageDimensions,
    options: ImageRenderOptions | None = None,
) -> RenderImageResult | None:
    opts = options if options is not None else ImageRenderOptions()
    caps = get_capabilities()

    if not caps.images:
        return None

    max_width = opts.max_width_cells if opts.max_width_cells is not None else 80
    size = calculate_image_cell_size(image_dimensions, max_width, opts.max_height_cells, get_cell_dimensions())

    if caps.images == "kitty":
        if opts.image_id is not None:
            register_kitty_image_metadata(
                KittyImageMetadata(
                    image_id=opts.image_id,
                    columns=size.columns,
                    rows=size.rows,
                    width_px=image_dimensions.width_px,
                    height_px=image_dimensions.height_px,
                )
            )
        sequence = encode_kitty(
            base64_data,
            columns=size.columns,
            rows=size.rows,
            image_id=opts.image_id,
            move_cursor=opts.move_cursor,
        )
        return RenderImageResult(sequence=sequence, columns=size.columns, rows=size.rows, image_id=opts.image_id)

    if caps.images == "iterm2":
        sequence = encode_iterm2(
            base64_data,
            width=size.columns,
            height="auto",
            preserve_aspect_ratio=opts.preserve_aspect_ratio if opts.preserve_aspect_ratio is not None else True,
        )
        return RenderImageResult(sequence=sequence, columns=size.columns, rows=size.rows)

    return None


def hyperlink(text: str, url: str) -> str:
    """Wrap text in an OSC 8 hyperlink sequence.

    The text is rendered as a clickable hyperlink in terminals that support OSC 8
    (Ghostty, Kitty, WezTerm, iTerm2, VSCode, and others).
    In terminals that do not support OSC 8, the escape sequences are ignored
    and only the plain text is displayed.
    """
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


def _shorten_image_path(filename: str) -> str:
    """Shorten home-prefixed absolute paths to ~/... for compact display."""
    home = str(Path.home())
    if home and (filename == home or filename.startswith(f"{home}/") or filename.startswith(f"{home}\\")):
        return f"~{filename[len(home) :]}"
    return filename


def image_fallback(mime_type: str, dimensions: ImageDimensions | None = None, filename: str | None = None) -> str:
    """Text fallback when the terminal cannot render inline images.

    Absolute paths are shown shortened (~/...) and, when OSC 8 hyperlinks are
    available, linked to file:// so the full path remains openable.
    """
    parts: list[str] = []
    if filename:
        display = _shorten_image_path(filename)
        if get_capabilities().hyperlinks and os.path.isabs(filename):
            parts.append(hyperlink(display, Path(filename).as_uri()))
        else:
            parts.append(display)
    parts.append(f"[{mime_type}]")
    if dimensions:
        parts.append(f"{dimensions.width_px}x{dimensions.height_px}")
    return f"[Image: {' '.join(parts)}]"

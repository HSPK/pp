"""Theme schema, loading, and ANSI color resolution.

Python port of
`packages/coding-agent/src/modes/interactive/theme/theme.ts`.

**Validation.** The TypeScript validates parsed theme JSON with a TypeBox
schema compiled by `Compile(ThemeJsonSchema)`. TypeBox 1.x objects are
"closed" (`additionalProperties: false`) by default, so every `Type.Object(...)`
in `ThemeJsonSchema` -- the top level, `colors`, and `export` -- rejects
unknown keys at runtime, even though none of them pass `additionalProperties`
explicitly. This port validates with :mod:`jsonschema` instead, against a
schema built from the shipped `theme-schema.json` asset (which already
encodes that same "closed object" shape, since it exists as an authored
JSON-Schema mirror of the TypeBox schema for editor tooling) with its `name`
`pattern` removed -- the TypeScript schema itself does not constrain `name`'s
character set; the `"/"` restriction is enforced separately by
`_assert_theme_name_is_valid`, exactly as the TypeScript's
`assertThemeNameIsValid` does after schema validation passes.

**Not ported (depend on subsystems this port does not have):**
- `highlightCode` / `getLanguageFromPath` / `getCliHighlightTheme` --
  depend on `highlight`/`supportsLanguage` in `utils/syntax-highlight.ts`,
  which wrap the `cli-highlight`/highlight.js pair. Nothing equivalent is
  bundled here, so `get_markdown_theme` leaves `MarkdownTheme.highlight_code`
  at `None` and fenced code blocks in assistant markdown render in the flat
  `mdCodeBlock` colour instead of being syntax-coloured.
- The theme file watcher (`startThemeWatcher` / `stopThemeWatcher`) --
  depends on `utils/fs-watch.ts`, which exists to drive live theme reload
  from disk edits. `onThemeChange` itself *is* ported (see below); only the
  filesystem-watching half is missing, so a theme file edited on disk is
  picked up by `/reload`, not automatically.

Everything else is ported: the color schema and built-in themes, merging a
partial theme's `vars`/`colors` over resolution, resolving color tokens
(hex/256-index/var-reference) to ANSI escape sequences in both truecolor and
256-color mode, the `Theme` class and its accessors, theme discovery/loading
(including the registered-theme and custom-theme-directory lookup paths),
terminal background/color-scheme auto-detection, `onThemeChange`, the
`getEditorTheme`/`getSelectListTheme`/`getSettingsListTheme` builders, the
global "current theme" instance, and the HTML-export color helpers.
`theme-controller.ts` is ported alongside this module as
`theme_controller.py`.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Literal, Protocol, runtime_checkable

from jsonschema import Draft7Validator

from pi_coding_agent.core.config import get_custom_themes_dir

ColorValue = str | int
ColorMode = Literal["truecolor", "256color"]
TerminalTheme = Literal["dark", "light"]

# ============================================================================
# Theme color/background token names
# ============================================================================

# Foreground color tokens (`ThemeColor` in the TypeScript).
THEME_COLOR_KEYS: tuple[str, ...] = (
    "accent",
    "border",
    "borderAccent",
    "borderMuted",
    "success",
    "error",
    "warning",
    "muted",
    "dim",
    "text",
    "thinkingText",
    "userMessageText",
    "customMessageText",
    "customMessageLabel",
    "toolTitle",
    "toolOutput",
    "mdHeading",
    "mdLink",
    "mdLinkUrl",
    "mdCode",
    "mdCodeBlock",
    "mdCodeBlockBorder",
    "mdQuote",
    "mdQuoteBorder",
    "mdHr",
    "mdListBullet",
    "toolDiffAdded",
    "toolDiffRemoved",
    "toolDiffContext",
    "syntaxComment",
    "syntaxKeyword",
    "syntaxFunction",
    "syntaxVariable",
    "syntaxString",
    "syntaxNumber",
    "syntaxType",
    "syntaxOperator",
    "syntaxPunctuation",
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    "thinkingMax",
    "bashMode",
)

# Background color tokens (`ThemeBg` in the TypeScript).
THEME_BG_KEYS: tuple[str, ...] = (
    "selectedBg",
    "scrollbarThumb",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
)

# ============================================================================
# Color utilities
# ============================================================================


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = hex_color.replace("#", "")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {hex_color}") from exc
    return r, g, b


# The 6x6x6 color cube channel values (indices 0-5).
CUBE_VALUES: tuple[int, ...] = (0, 95, 135, 175, 215, 255)

# Grayscale ramp values (indices 232-255, 24 grays from 8 to 238).
GRAY_VALUES: tuple[int, ...] = tuple(8 + i * 10 for i in range(24))


def _find_closest_cube_index(value: int) -> int:
    min_dist = float("inf")
    min_idx = 0
    for i, cube_value in enumerate(CUBE_VALUES):
        dist = abs(value - cube_value)
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    return min_idx


def _find_closest_gray_index(gray: int) -> int:
    min_dist = float("inf")
    min_idx = 0
    for i, gray_value in enumerate(GRAY_VALUES):
        dist = abs(gray - gray_value)
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    return min_idx


def _color_distance(r1: int, g1: int, b1: int, r2: int, g2: int, b2: int) -> float:
    # Weighted Euclidean distance (human eye is more sensitive to green).
    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2
    return dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114


def rgb_to_256(r: int, g: int, b: int) -> int:
    # Find closest color in the 6x6x6 cube.
    r_idx = _find_closest_cube_index(r)
    g_idx = _find_closest_cube_index(g)
    b_idx = _find_closest_cube_index(b)
    cube_r = CUBE_VALUES[r_idx]
    cube_g = CUBE_VALUES[g_idx]
    cube_b = CUBE_VALUES[b_idx]
    cube_index = 16 + 36 * r_idx + 6 * g_idx + b_idx
    cube_dist = _color_distance(r, g, b, cube_r, cube_g, cube_b)

    # Find closest grayscale.
    gray = round(0.299 * r + 0.587 * g + 0.114 * b)
    gray_idx = _find_closest_gray_index(gray)
    gray_value = GRAY_VALUES[gray_idx]
    gray_index = 232 + gray_idx
    gray_dist = _color_distance(r, g, b, gray_value, gray_value, gray_value)

    # Check if color has noticeable saturation (hue matters). If max-min
    # spread is significant, prefer cube to preserve tint. Only consider
    # grayscale if color is nearly neutral (spread < 10) AND grayscale is
    # actually closer.
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    spread = max_c - min_c
    if spread < 10 and gray_dist < cube_dist:
        return gray_index
    return cube_index


def hex_to_256(hex_color: str) -> int:
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_256(r, g, b)


def fg_ansi(color: ColorValue, mode: ColorMode) -> str:
    if color == "":
        return "\x1b[39m"
    if isinstance(color, int):
        return f"\x1b[38;5;{color}m"
    if color.startswith("#"):
        if mode == "truecolor":
            r, g, b = hex_to_rgb(color)
            return f"\x1b[38;2;{r};{g};{b}m"
        index = hex_to_256(color)
        return f"\x1b[38;5;{index}m"
    raise ValueError(f"Invalid color value: {color}")


def bg_ansi(color: ColorValue, mode: ColorMode) -> str:
    if color == "":
        return "\x1b[49m"
    if isinstance(color, int):
        return f"\x1b[48;5;{color}m"
    if color.startswith("#"):
        if mode == "truecolor":
            r, g, b = hex_to_rgb(color)
            return f"\x1b[48;2;{r};{g};{b}m"
        index = hex_to_256(color)
        return f"\x1b[48;5;{index}m"
    raise ValueError(f"Invalid color value: {color}")


def resolve_var_refs(
    value: ColorValue,
    variables: Mapping[str, ColorValue],
    visited: set[str] | None = None,
) -> ColorValue:
    if isinstance(value, int) or value == "" or value.startswith("#"):
        return value
    if visited is None:
        visited = set()
    if value in visited:
        raise ValueError(f"Circular variable reference detected: {value}")
    if value not in variables:
        raise ValueError(f"Variable reference not found: {value}")
    visited.add(value)
    return resolve_var_refs(variables[value], variables, visited)


def resolve_theme_colors(
    colors: Mapping[str, ColorValue],
    variables: Mapping[str, ColorValue] | None = None,
) -> dict[str, ColorValue]:
    variables = variables or {}
    return {key: resolve_var_refs(value, variables) for key, value in colors.items()}


def with_theme_color_fallbacks(colors: Mapping[str, ColorValue]) -> dict[str, ColorValue]:
    result = dict(colors)
    result.setdefault("thinkingMax", colors["thinkingXhigh"])
    result.setdefault("scrollbarThumb", colors["selectedBg"])
    return result


# ============================================================================
# Theme class
# ============================================================================


class Theme:
    """A fully resolved theme: ANSI escape sequences for every color token."""

    def __init__(
        self,
        fg_colors: Mapping[str, ColorValue],
        bg_colors: Mapping[str, ColorValue],
        mode: ColorMode,
        *,
        name: str | None = None,
        source_path: str | None = None,
        source_info: object | None = None,
    ) -> None:
        self.name = name
        self.source_path = source_path
        # `SourceInfo` (`core/source-info.ts`) is not ported; kept as an
        # untyped attribute for parity with the TypeScript field, which the
        # (unported) resource loader assigns after loading a theme file.
        self.source_info = source_info
        self._mode: ColorMode = mode

        colors = dict(fg_colors)
        colors.setdefault("thinkingMax", fg_colors["thinkingXhigh"])
        self._fg_colors: dict[str, str] = {key: fg_ansi(value, mode) for key, value in colors.items()}

        backgrounds = dict(bg_colors)
        backgrounds.setdefault("scrollbarThumb", bg_colors["selectedBg"])
        self._bg_colors: dict[str, str] = {key: bg_ansi(value, mode) for key, value in backgrounds.items()}

    def fg(self, color: str, text: str) -> str:
        ansi = self._fg_colors.get(color)
        if not ansi:
            raise ValueError(f"Unknown theme color: {color}")
        return f"{ansi}{text}\x1b[39m"  # Reset only foreground color.

    def bg(self, color: str, text: str) -> str:
        ansi = self._bg_colors.get(color)
        if not ansi:
            raise ValueError(f"Unknown theme background color: {color}")
        return f"{ansi}{text}\x1b[49m"  # Reset only background color.

    def bold(self, text: str) -> str:
        return f"\x1b[1m{text}\x1b[22m"

    def italic(self, text: str) -> str:
        return f"\x1b[3m{text}\x1b[23m"

    def underline(self, text: str) -> str:
        return f"\x1b[4m{text}\x1b[24m"

    def inverse(self, text: str) -> str:
        return f"\x1b[7m{text}\x1b[27m"

    def strikethrough(self, text: str) -> str:
        return f"\x1b[9m{text}\x1b[29m"

    def get_fg_ansi(self, color: str) -> str:
        ansi = self._fg_colors.get(color)
        if not ansi:
            raise ValueError(f"Unknown theme color: {color}")
        return ansi

    def get_bg_ansi(self, color: str) -> str:
        ansi = self._bg_colors.get(color)
        if not ansi:
            raise ValueError(f"Unknown theme background color: {color}")
        return ansi

    def get_color_mode(self) -> ColorMode:
        return self._mode

    def get_thinking_border_color(self, level: str) -> Callable[[str], str]:
        # Map thinking levels to dedicated theme colors.
        color_by_level = {
            "off": "thinkingOff",
            "minimal": "thinkingMinimal",
            "low": "thinkingLow",
            "medium": "thinkingMedium",
            "high": "thinkingHigh",
            "xhigh": "thinkingXhigh",
            "max": "thinkingMax",
        }
        color = color_by_level.get(level, "thinkingOff")
        return lambda text: self.fg(color, text)

    def get_bash_mode_border_color(self) -> Callable[[str], str]:
        return lambda text: self.fg("bashMode", text)


# ============================================================================
# Theme loading
# ============================================================================

_THEME_PACKAGE = resources.files(__package__)


def _read_asset(name: str) -> str:
    return (_THEME_PACKAGE / name).read_text(encoding="utf-8")


def _load_runtime_schema() -> dict:
    """Build the runtime validation schema from the shipped asset.

    `theme-schema.json` already models the "closed object" shape TypeBox 1.x
    compiles by default (`additionalProperties: false` at the top level, in
    `colors`, and in `export`). The only divergence from the TypeBox schema
    actually compiled by `theme.ts` is the `pattern` on `name`, which the
    TypeScript's `ThemeJsonSchema` does not have -- the `"/"` restriction is
    enforced separately there (and here) by a dedicated check with a
    friendlier error message. That pattern is stripped so this schema matches
    the TypeScript runtime validator exactly.
    """
    schema = json.loads(_read_asset("theme-schema.json"))
    schema = copy.deepcopy(schema)
    schema["properties"]["name"].pop("pattern", None)
    return schema


_THEME_JSON_SCHEMA = _load_runtime_schema()
_THEME_JSON_VALIDATOR = Draft7Validator(_THEME_JSON_SCHEMA)

_BUILTIN_THEME_NAMES: tuple[str, ...] = ("dark", "light")

_builtin_themes_cache: dict[str, dict] | None = None


def _get_builtin_themes() -> dict[str, dict]:
    global _builtin_themes_cache
    if _builtin_themes_cache is None:
        _builtin_themes_cache = {name: json.loads(_read_asset(f"{name}.json")) for name in _BUILTIN_THEME_NAMES}
    return _builtin_themes_cache


@dataclass(frozen=True)
class ThemeInfo:
    name: str
    path: str | None


_custom_theme_discovery_enabled = True


def set_custom_theme_discovery_enabled(enabled: bool) -> None:
    """Port of `DefaultResourceLoader`'s `noThemes` gate (`--no-themes`).

    TypeScript keeps CLI-supplied `--theme` paths and drops only the
    *discovered* ones (`resource-loader.ts:501`). This port has no package or
    extension theme discovery, so the equivalent set is the user's
    `~/.pi/agent/themes` directory: built-ins are always available because they
    ship inside the package and are not discovered at all.

    A module-level switch rather than a parameter because the theme registry is
    read from several call sites (selector, startup, `/theme`), and threading a
    flag through each one would leave whichever site was missed still
    discovering.
    """
    global _custom_theme_discovery_enabled
    _custom_theme_discovery_enabled = enabled


def get_available_themes(*, custom_themes_dir: str | None = None) -> list[str]:
    return [info.name for info in get_available_themes_with_paths(custom_themes_dir=custom_themes_dir)]


def get_available_themes_with_paths(*, custom_themes_dir: str | None = None) -> list[ThemeInfo]:
    themes_dir = custom_themes_dir if custom_themes_dir is not None else get_custom_themes_dir()
    result: list[ThemeInfo] = []
    if not _custom_theme_discovery_enabled:
        themes_dir = None  # type: ignore[assignment]
    seen: set[str] = set()

    def add_theme(theme_info: ThemeInfo) -> None:
        if theme_info.name in seen:
            return
        seen.add(theme_info.name)
        result.append(theme_info)

    # Built-in themes.
    for name in _get_builtin_themes():
        add_theme(ThemeInfo(name=name, path=str(_THEME_PACKAGE / f"{name}.json")))

    # Custom themes.
    if themes_dir is not None:
        for theme_info in _get_custom_theme_infos(themes_dir):
            add_theme(theme_info)

    for name, registered_theme in _registered_themes.items():
        add_theme(ThemeInfo(name=name, path=registered_theme.source_path))

    return sorted(result, key=lambda info: info.name)


def _get_custom_theme_infos(custom_themes_dir: str) -> list[ThemeInfo]:
    result: list[ThemeInfo] = []
    if not os.path.isdir(custom_themes_dir):
        return result
    for file_name in sorted(os.listdir(custom_themes_dir)):
        if not file_name.endswith(".json"):
            continue
        theme_path = os.path.join(custom_themes_dir, file_name)
        try:
            custom_theme = load_theme_from_path(theme_path)
        except Exception:
            # Invalid themes are ignored here; a resource loader would
            # report them during normal startup/reload (not ported; see the
            # module docstring).
            continue
        if custom_theme.name:
            result.append(ThemeInfo(name=custom_theme.name, path=theme_path))
    return result


def _assert_theme_name_is_valid(name: str) -> None:
    if "/" in name:
        raise ValueError(
            f'Invalid theme name "{name}": theme names cannot contain "/" because it is reserved for '
            "automatic light/dark theme settings."
        )


_REQUIRED_PROPERTY_PATTERN = re.compile(r"^'(.+)' is a required property$")


def _parse_theme_json(label: str, data: object) -> dict:
    errors = list(_THEME_JSON_VALIDATOR.iter_errors(data))
    if errors:
        missing_colors: set[str] = set()
        other_errors: list[str] = []
        for error in errors:
            if error.validator == "required" and list(error.path) == ["colors"]:
                match = _REQUIRED_PROPERTY_PATTERN.match(error.message)
                if match:
                    missing_colors.add(match.group(1))
                    continue
            path = "/" + "/".join(str(part) for part in error.path) if error.path else "/"
            other_errors.append(f"  - {path}: {error.message}")

        message = f'Invalid theme "{label}":\n'
        if missing_colors:
            message += "\nMissing required color tokens:\n"
            message += "\n".join(f"  - {color}" for color in sorted(missing_colors))
            message += '\n\nPlease add these colors to your theme\'s "colors" object.'
            message += "\nSee the built-in themes (dark.json, light.json) for reference values."
        if other_errors:
            message += "\n\nOther errors:\n" + "\n".join(other_errors)
        raise ValueError(message)

    assert isinstance(data, dict)
    _assert_theme_name_is_valid(data["name"])
    return data


def _parse_theme_json_content(label: str, content: str) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse theme {label}: {exc}") from exc
    return _parse_theme_json(label, data)


def _load_theme_json(name: str, *, custom_themes_dir: str | None = None) -> dict:
    builtin_themes = _get_builtin_themes()
    if name in builtin_themes:
        return builtin_themes[name]
    registered_theme = _registered_themes.get(name)
    if registered_theme is not None and registered_theme.source_path:
        with open(registered_theme.source_path, encoding="utf-8") as f:
            content = f.read()
        return _parse_theme_json_content(registered_theme.source_path, content)
    if registered_theme is not None:
        raise ValueError(f'Theme "{name}" does not have a source path for export')
    if not _custom_theme_discovery_enabled:
        # `--no-themes`: a saved `theme` setting naming a custom theme must not
        # silently keep working, or the flag would only hide themes from the
        # selector while still loading one at startup.
        raise ValueError(f"Theme not found: {name}")
    themes_dir = custom_themes_dir if custom_themes_dir is not None else get_custom_themes_dir()
    theme_path = os.path.join(themes_dir, f"{name}.json")
    if not os.path.exists(theme_path):
        raise ValueError(f"Theme not found: {name}")
    with open(theme_path, encoding="utf-8") as f:
        content = f.read()
    return _parse_theme_json_content(name, content)


_BG_COLOR_KEY_SET = frozenset(THEME_BG_KEYS)


def _create_theme(theme_json: dict, mode: ColorMode | None = None, source_path: str | None = None) -> Theme:
    from pi_tui import get_capabilities

    color_mode: ColorMode = mode if mode is not None else ("truecolor" if get_capabilities().true_color else "256color")
    resolved_colors = resolve_theme_colors(with_theme_color_fallbacks(theme_json["colors"]), theme_json.get("vars"))
    fg_colors: dict[str, ColorValue] = {}
    bg_colors: dict[str, ColorValue] = {}
    for key, value in resolved_colors.items():
        if key in _BG_COLOR_KEY_SET:
            bg_colors[key] = value
        else:
            fg_colors[key] = value
    return Theme(fg_colors, bg_colors, color_mode, name=theme_json.get("name"), source_path=source_path)


def load_theme_from_path(theme_path: str, mode: ColorMode | None = None) -> Theme:
    with open(theme_path, encoding="utf-8") as f:
        content = f.read()
    theme_json = _parse_theme_json_content(theme_path, content)
    return _create_theme(theme_json, mode, theme_path)


def load_theme(name: str, mode: ColorMode | None = None, *, custom_themes_dir: str | None = None) -> Theme:
    registered_theme = _registered_themes.get(name)
    if registered_theme is not None:
        return registered_theme
    theme_json = _load_theme_json(name, custom_themes_dir=custom_themes_dir)
    return _create_theme(theme_json, mode)


def get_theme_by_name(name: str, *, custom_themes_dir: str | None = None) -> Theme | None:
    try:
        return load_theme(name, custom_themes_dir=custom_themes_dir)
    except Exception:
        return None


@dataclass(frozen=True)
class AutoThemeSetting:
    light_theme: str
    dark_theme: str


def parse_auto_theme_setting(theme_setting: str | None) -> AutoThemeSetting | None:
    if not theme_setting:
        return None
    slash_index = theme_setting.find("/")
    if slash_index == -1 or theme_setting.find("/", slash_index + 1) != -1:
        return None
    light_theme = theme_setting[:slash_index].strip()
    dark_theme = theme_setting[slash_index + 1 :].strip()
    if not light_theme or not dark_theme:
        return None
    return AutoThemeSetting(light_theme=light_theme, dark_theme=dark_theme)


def resolve_theme_setting(theme_setting: str | None, terminal_theme: TerminalTheme) -> str | None:
    auto_theme = parse_auto_theme_setting(theme_setting)
    if auto_theme:
        return auto_theme.light_theme if terminal_theme == "light" else auto_theme.dark_theme
    if theme_setting is not None and "/" in theme_setting:
        return None
    if isinstance(theme_setting, str):
        return theme_setting
    return None


# ============================================================================
# Terminal theme detection
# ============================================================================


@dataclass(frozen=True)
class RgbColor:
    r: int
    g: int
    b: int


@dataclass(frozen=True)
class TerminalThemeDetection:
    theme: TerminalTheme
    source: Literal["terminal background", "COLORFGBG", "fallback"]
    detail: str
    confidence: Literal["high", "low"]


@runtime_checkable
class TerminalBackgroundThemeDetector(Protocol):
    async def query_terminal_background_color(self, timeout_ms: float) -> object | None: ...


@runtime_checkable
class TerminalAutoThemeDetector(TerminalBackgroundThemeDetector, Protocol):
    async def query_terminal_color_scheme(self, timeout_ms: float) -> TerminalTheme | None: ...


def _get_colorfgbg_background_index(colorfgbg: str) -> int | None:
    parts = colorfgbg.split(";")
    for part in reversed(parts):
        try:
            bg = int(part.strip())
        except ValueError:
            continue
        if 0 <= bg <= 255:
            return bg
    return None


def _get_rgb_color_luminance(rgb: object) -> float:
    r = rgb.r if hasattr(rgb, "r") else rgb["r"]  # type: ignore[index]
    g = rgb.g if hasattr(rgb, "g") else rgb["g"]  # type: ignore[index]
    b = rgb.b if hasattr(rgb, "b") else rgb["b"]  # type: ignore[index]

    def to_linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)


def _get_ansi_color_luminance(index: int) -> float:
    r, g, b = hex_to_rgb(ansi256_to_hex(index))
    return _get_rgb_color_luminance(RgbColor(r=r, g=g, b=b))


def get_theme_for_rgb_color(rgb: object) -> TerminalTheme:
    return "light" if _get_rgb_color_luminance(rgb) >= 0.5 else "dark"


def detect_terminal_background_from_env(*, env: Mapping[str, str] | None = None) -> TerminalThemeDetection:
    environ = env if env is not None else os.environ
    colorfgbg = environ.get("COLORFGBG") or ""
    bg = _get_colorfgbg_background_index(colorfgbg)
    if bg is not None:
        return TerminalThemeDetection(
            theme="light" if _get_ansi_color_luminance(bg) >= 0.5 else "dark",
            source="COLORFGBG",
            detail=f"background color index {bg}",
            confidence="high",
        )
    return TerminalThemeDetection(
        theme="dark",
        source="fallback",
        detail="no terminal background hint found",
        confidence="low",
    )


async def detect_terminal_background_theme(
    ui: TerminalBackgroundThemeDetector,
    timeout_ms: float,
    *,
    env: Mapping[str, str] | None = None,
) -> TerminalThemeDetection:
    try:
        rgb = await ui.query_terminal_background_color(timeout_ms)
        if rgb:
            r = rgb.r if hasattr(rgb, "r") else rgb["r"]  # type: ignore[index]
            g = rgb.g if hasattr(rgb, "g") else rgb["g"]  # type: ignore[index]
            b = rgb.b if hasattr(rgb, "b") else rgb["b"]  # type: ignore[index]
            return TerminalThemeDetection(
                theme=get_theme_for_rgb_color(rgb),
                source="terminal background",
                detail=f"OSC 11 background rgb({r}, {g}, {b})",
                confidence="high",
            )
    except Exception:
        # Fall back to environment-based detection when the terminal query fails.
        pass
    return detect_terminal_background_from_env(env=env)


async def detect_terminal_theme_for_auto(
    ui: TerminalAutoThemeDetector,
    timeout_ms: float,
    *,
    env: Mapping[str, str] | None = None,
) -> TerminalTheme:
    import asyncio

    color_scheme_task: asyncio.Task[TerminalTheme | None] | None = None
    query_color_scheme = getattr(ui, "query_terminal_color_scheme", None)
    if query_color_scheme is not None:
        try:
            color_scheme_task = asyncio.ensure_future(query_color_scheme(timeout_ms))
        except Exception:
            # Fall back to OSC 11 / COLORFGBG detection when starting the
            # color-scheme query fails.
            color_scheme_task = None

    background_theme_task = asyncio.ensure_future(detect_terminal_background_theme(ui, timeout_ms, env=env))

    if color_scheme_task is not None:
        try:
            color_scheme = await color_scheme_task
            if color_scheme:
                return color_scheme
        except Exception:
            # Fall back to the concurrently queried OSC 11 / COLORFGBG detection.
            pass
    return (await background_theme_task).theme


def get_default_theme(*, env: Mapping[str, str] | None = None) -> str:
    return detect_terminal_background_from_env(env=env).theme


# ============================================================================
# Global theme instance
# ============================================================================

_current_theme: Theme | None = None
_current_theme_name: str | None = None
_registered_themes: dict[str, Theme] = {}
_on_theme_change_callback: Callable[[], None] | None = None


class _CurrentThemeProxy:
    """Delegates attribute access to the current global theme.

    Mirrors the TypeScript `theme` export, a `Proxy` around the value stored
    on `globalThis` so all call sites (and multiple module loaders in the
    original dev setup) see the same live theme instance.
    """

    def __getattr__(self, name: str) -> object:
        if _current_theme is None:
            raise RuntimeError("Theme not initialized. Call init_theme() first.")
        return getattr(_current_theme, name)


theme = _CurrentThemeProxy()


def _set_global_theme(new_theme: Theme) -> None:
    global _current_theme
    _current_theme = new_theme


def set_registered_themes(themes: list[Theme]) -> None:
    _registered_themes.clear()
    for registered_theme in themes:
        if registered_theme.name:
            _assert_theme_name_is_valid(registered_theme.name)
            _registered_themes[registered_theme.name] = registered_theme


def init_theme(theme_name: str | None = None, *, custom_themes_dir: str | None = None) -> None:
    global _current_theme_name
    name = theme_name if theme_name is not None else get_default_theme()
    _current_theme_name = name
    try:
        _set_global_theme(load_theme(name, custom_themes_dir=custom_themes_dir))
    except Exception:
        # Theme is invalid - fall back to dark theme silently.
        _current_theme_name = "dark"
        _set_global_theme(load_theme("dark", custom_themes_dir=custom_themes_dir))


@dataclass(frozen=True)
class ThemeResult:
    success: bool
    error: str | None = None


def set_theme(name: str, *, custom_themes_dir: str | None = None) -> ThemeResult:
    global _current_theme_name
    _current_theme_name = name
    try:
        _set_global_theme(load_theme(name, custom_themes_dir=custom_themes_dir))
        if _on_theme_change_callback is not None:
            _on_theme_change_callback()
        return ThemeResult(success=True)
    except Exception as error:
        # Theme is invalid - fall back to dark theme.
        _current_theme_name = "dark"
        _set_global_theme(load_theme("dark", custom_themes_dir=custom_themes_dir))
        return ThemeResult(success=False, error=str(error))


def set_theme_instance(theme_instance: Theme) -> None:
    global _current_theme_name
    _set_global_theme(theme_instance)
    _current_theme_name = "<in-memory>"
    if _on_theme_change_callback is not None:
        _on_theme_change_callback()


def on_theme_change(callback: Callable[[], None]) -> None:
    global _on_theme_change_callback
    _on_theme_change_callback = callback


# ============================================================================
# HTML export helpers
# ============================================================================

_BASIC_ANSI_COLORS: tuple[str, ...] = (
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#c0c0c0",
    "#808080",
    "#ff0000",
    "#00ff00",
    "#ffff00",
    "#0000ff",
    "#ff00ff",
    "#00ffff",
    "#ffffff",
)


def ansi256_to_hex(index: int) -> str:
    """Convert a 256-color index to a hex string.

    Indices 0-15: basic colors (approximate). Indices 16-231: 6x6x6 color
    cube. Indices 232-255: grayscale ramp.
    """
    if index < 16:
        return _BASIC_ANSI_COLORS[index]

    if index < 232:
        cube_index = index - 16
        r = cube_index // 36
        g = (cube_index % 36) // 6
        b = cube_index % 6

        def to_hex(n: int) -> str:
            value = 0 if n == 0 else 55 + n * 40
            return format(value, "02x")

        return f"#{to_hex(r)}{to_hex(g)}{to_hex(b)}"

    gray = 8 + (index - 232) * 10
    gray_hex = format(gray, "02x")
    return f"#{gray_hex}{gray_hex}{gray_hex}"


def get_resolved_theme_colors(theme_name: str | None = None, *, custom_themes_dir: str | None = None) -> dict[str, str]:
    """Get resolved theme colors as CSS-compatible hex strings.

    Used by HTML export to generate CSS custom properties.
    """
    name = theme_name or _current_theme_name or get_default_theme()
    is_light = name == "light"
    theme_json = _load_theme_json(name, custom_themes_dir=custom_themes_dir)
    resolved = resolve_theme_colors(with_theme_color_fallbacks(theme_json["colors"]), theme_json.get("vars"))

    # Default text color for empty values (terminal uses default fg color).
    default_text = "#000000" if is_light else "#e5e5e7"

    css_colors: dict[str, str] = {}
    for key, value in resolved.items():
        if isinstance(value, int):
            css_colors[key] = ansi256_to_hex(value)
        elif value == "":
            # Empty means default terminal color - use sensible fallback for HTML.
            css_colors[key] = default_text
        else:
            css_colors[key] = value
    return css_colors


def is_light_theme(theme_name: str | None = None) -> bool:
    """Check if a theme is a "light" theme (for CSS that needs light/dark variants)."""
    return theme_name == "light"


@dataclass(frozen=True)
class ThemeExportColors:
    page_bg: str | None = None
    card_bg: str | None = None
    info_bg: str | None = None


def get_theme_export_colors(
    theme_name: str | None = None, *, custom_themes_dir: str | None = None
) -> ThemeExportColors:
    """Get explicit export colors from theme JSON, if specified.

    Fields are `None` for each color that isn't explicitly set.
    """
    name = theme_name or _current_theme_name or get_default_theme()
    # The TypeScript wraps the whole body, including the var-ref resolution, in
    # a try/catch that returns {}. The schema accepts any string here, so a
    # theme that validates cleanly can still carry an unresolvable reference,
    # and that must not blow up HTML export.
    try:
        theme_json = _load_theme_json(name, custom_themes_dir=custom_themes_dir)
        export_section = theme_json.get("export")
        if not export_section:
            return ThemeExportColors()

        variables = theme_json.get("vars") or {}

        def resolve(value: ColorValue | None) -> str | None:
            if value is None:
                return None
            resolved = resolve_var_refs(value, variables)
            if isinstance(resolved, int):
                return ansi256_to_hex(resolved)
            if resolved == "":
                return None
            return resolved

        return ThemeExportColors(
            page_bg=resolve(export_section.get("pageBg")),
            card_bg=resolve(export_section.get("cardBg")),
            info_bg=resolve(export_section.get("infoBg")),
        )
    except Exception:
        return ThemeExportColors()


# ============================================================================
# TUI helpers
# ============================================================================


def get_markdown_theme():
    """Build a `pi_tui.MarkdownTheme` from the current global theme.

    `highlightCode` is not ported (see the module docstring), so
    `highlight_code` is left at its `MarkdownTheme` default (`None`) and
    fenced code blocks render unhighlighted. `code_block_indent` is filled in
    by the interactive mode from the `codeBlockIndent` setting.
    """
    from pi_tui import MarkdownTheme

    return MarkdownTheme(
        heading=lambda text: theme.fg("mdHeading", text),
        link=lambda text: theme.fg("mdLink", text),
        link_url=lambda text: theme.fg("mdLinkUrl", text),
        code=lambda text: theme.fg("mdCode", text),
        code_block=lambda text: theme.fg("mdCodeBlock", text),
        code_block_border=lambda text: theme.fg("mdCodeBlockBorder", text),
        quote=lambda text: theme.fg("mdQuote", text),
        quote_border=lambda text: theme.fg("mdQuoteBorder", text),
        hr=lambda text: theme.fg("mdHr", text),
        list_bullet=lambda text: theme.fg("mdListBullet", text),
        bold=lambda text: theme.bold(text),
        italic=lambda text: theme.italic(text),
        underline=lambda text: theme.underline(text),
        strikethrough=lambda text: f"\x1b[9m{text}\x1b[29m",
    )


def get_select_list_theme():
    from pi_tui import SelectListTheme

    return SelectListTheme(
        selected_prefix=lambda text: theme.fg("accent", text),
        selected_text=lambda text: theme.fg("accent", text),
        description=lambda text: theme.fg("muted", text),
        scroll_info=lambda text: theme.fg("muted", text),
        no_match=lambda text: theme.fg("muted", text),
    )


def get_editor_theme():
    """Build a `pi_tui.EditorTheme` from the current global theme."""
    from pi_tui import EditorTheme

    return EditorTheme(
        border_color=lambda text: theme.fg("borderMuted", text),
        select_list=get_select_list_theme(),
    )


def get_settings_list_theme():
    from pi_tui import SettingsListTheme

    return SettingsListTheme(
        label=lambda text, selected: theme.fg("accent", text) if selected else text,
        value=lambda text, selected: theme.fg("accent", text) if selected else theme.fg("muted", text),
        description=lambda text: theme.fg("dim", text),
        cursor=theme.fg("accent", "\u2192 "),
        hint=lambda text: theme.fg("dim", text),
    )


__all__ = [
    "THEME_BG_KEYS",
    "THEME_COLOR_KEYS",
    "AutoThemeSetting",
    "ColorMode",
    "ColorValue",
    "RgbColor",
    "TerminalAutoThemeDetector",
    "TerminalBackgroundThemeDetector",
    "TerminalTheme",
    "TerminalThemeDetection",
    "Theme",
    "ThemeExportColors",
    "ThemeInfo",
    "ThemeResult",
    "ansi256_to_hex",
    "bg_ansi",
    "detect_terminal_background_from_env",
    "detect_terminal_background_theme",
    "detect_terminal_theme_for_auto",
    "fg_ansi",
    "get_available_themes",
    "get_available_themes_with_paths",
    "get_default_theme",
    "get_editor_theme",
    "get_markdown_theme",
    "get_resolved_theme_colors",
    "get_select_list_theme",
    "get_settings_list_theme",
    "get_theme_by_name",
    "get_theme_export_colors",
    "get_theme_for_rgb_color",
    "hex_to_256",
    "hex_to_rgb",
    "init_theme",
    "is_light_theme",
    "load_theme",
    "load_theme_from_path",
    "on_theme_change",
    "parse_auto_theme_setting",
    "resolve_theme_colors",
    "resolve_theme_setting",
    "resolve_var_refs",
    "rgb_to_256",
    "set_registered_themes",
    "set_theme",
    "set_theme_instance",
    "theme",
    "with_theme_color_fallbacks",
]


# --------------------------------------------------------------------------
# Syntax highlighting
#
# Port of `highlightCode` / `getLanguageFromPath` (`theme.ts:1179`, `:1203`).
# TypeScript highlights through `cli-highlight` (highlight.js); this port uses
# Pygments, which is already in the dependency tree. What matters for parity is
# not the tokenizer but that token colours come from the *active theme*'s
# `syntax*` entries rather than a library default palette -- that is what makes
# highlighted code match the rest of the UI when the user switches themes.
# --------------------------------------------------------------------------

_EXT_TO_LANG: dict[str, str] = {
    "ts": "typescript",
    "tsx": "typescript",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "py": "python",
    "rb": "ruby",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "kt": "kotlin",
    "swift": "swift",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "cs": "csharp",
    "php": "php",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "fish": "fish",
    "ps1": "powershell",
    "sql": "sql",
    "html": "html",
    "htm": "html",
    "css": "css",
    "scss": "scss",
    "sass": "sass",
    "less": "less",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "xml": "xml",
    "md": "markdown",
    "markdown": "markdown",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "cmake": "cmake",
    "lua": "lua",
    "perl": "perl",
    "r": "r",
    "scala": "scala",
    "clj": "clojure",
    "ex": "elixir",
    "exs": "elixir",
    "erl": "erlang",
    "hs": "haskell",
    "ml": "ocaml",
    "vim": "vim",
    "graphql": "graphql",
    "proto": "protobuf",
    "tf": "hcl",
    "hcl": "hcl",
}


def get_language_from_path(file_path: str) -> str | None:
    """Language id for a path's extension. Port of `getLanguageFromPath`."""
    if "." not in file_path:
        return None
    ext = file_path.split(".")[-1].lower()
    return _EXT_TO_LANG.get(ext)


def _theme_color_for_token(token: object) -> str | None:
    """Map a Pygments token to a theme colour key.

    Mirrors `buildCliHighlightTheme` (`theme.ts:1137`), which maps
    highlight.js scopes onto the same `syntax*` theme entries. Walks up the
    token hierarchy so specific tokens (`Token.Literal.String.Double`) inherit
    their parent's colour.
    """
    from pygments.token import Comment, Error, Keyword, Name, Number, Operator, Punctuation, String

    mapping = [
        (Comment, "syntaxComment"),
        (String, "syntaxString"),
        (Number, "syntaxNumber"),
        (Keyword.Type, "syntaxType"),
        (Name.Class, "syntaxType"),
        (Name.Builtin, "syntaxType"),
        (Name.Function, "syntaxFunction"),
        (Name.Decorator, "syntaxFunction"),
        (Keyword, "syntaxKeyword"),
        (Name.Variable, "syntaxVariable"),
        (Name.Attribute, "syntaxVariable"),
        (Operator, "syntaxOperator"),
        (Punctuation, "syntaxPunctuation"),
        (Error, "error"),
    ]
    for token_type, color in mapping:
        if token in token_type:
            return color
    return None


def highlight_code(code: str, lang: str | None = None) -> list[str]:
    """Highlight `code`, returning one string per line. Port of `highlightCode`.

    With no usable language the lines come back tinted `mdCodeBlock` rather
    than auto-detected: upstream disables auto-detection because it misreads
    prose as code and colours random English words as keywords.
    """
    if not lang:
        return [theme.fg("mdCodeBlock", line) for line in code.split("\n")]

    try:
        from pygments import lex
        from pygments.lexers import get_lexer_by_name
    except ImportError:  # pragma: no cover - pygments ships with the deps
        return [theme.fg("mdCodeBlock", line) for line in code.split("\n")]

    try:
        lexer = get_lexer_by_name(lang, stripnl=False, ensurenl=False)
    except Exception:
        return [theme.fg("mdCodeBlock", line) for line in code.split("\n")]

    try:
        out: list[str] = []
        current = ""
        for token_type, value in lex(code, lexer):
            color = _theme_color_for_token(token_type)
            # Style each line's fragment separately: a styled span must not
            # cross a newline or the reset lands on the wrong row.
            parts = value.split("\n")
            for index, part in enumerate(parts):
                if index > 0:
                    out.append(current)
                    current = ""
                if not part:
                    continue
                current += theme.fg(color, part) if color else part
        out.append(current)
        return out
    except Exception:
        return code.split("\n")

"""Tests for `modes/interactive/theme/theme.py`.

Ported from the TypeScript coding-agent package's theme tests:
`test/theme-detection.test.ts`, `test/scrollbar-theme.test.ts`,
`test/theme-picker.test.ts`, and `test/theme-export.test.ts`. All
filesystem/env work uses `tmp_path`/explicit `env=`/`custom_themes_dir=`
arguments; no test reads or writes the real `$HOME` or process environment.
"""

from __future__ import annotations

import json
import re

import pytest
from pi_tui.terminal_image import TerminalCapabilities, reset_capabilities_cache, set_capabilities

from pi_coding_agent.modes.interactive.theme.theme import (
    THEME_BG_KEYS,
    THEME_COLOR_KEYS,
    RgbColor,
    Theme,
    ThemeInfo,
    _get_builtin_themes,
    detect_terminal_background_from_env,
    detect_terminal_background_theme,
    detect_terminal_theme_for_auto,
    get_available_themes,
    get_available_themes_with_paths,
    get_theme_by_name,
    get_theme_export_colors,
    get_theme_for_rgb_color,
    hex_to_256,
    hex_to_rgb,
    load_theme,
    load_theme_from_path,
    parse_auto_theme_setting,
    resolve_theme_setting,
    set_registered_themes,
)


@pytest.fixture(autouse=True)
def _reset_theme_registry():
    yield
    set_registered_themes([])
    reset_capabilities_cache()


def _read_builtin_json(name: str) -> dict:
    from pi_coding_agent.modes.interactive.theme.theme import _THEME_PACKAGE

    return json.loads((_THEME_PACKAGE / f"{name}.json").read_text(encoding="utf-8"))


def _write_theme(tmp_path, theme_dict: dict, file_name: str | None = None) -> str:
    themes_dir = tmp_path / "themes"
    themes_dir.mkdir(exist_ok=True)
    path = themes_dir / (file_name or f"{theme_dict['name']}.json")
    path.write_text(json.dumps(theme_dict))
    return str(path)


# ============================================================================
# Built-in themes and every documented token resolves
# ============================================================================


def test_builtin_themes_load():
    themes = _get_builtin_themes()
    assert set(themes.keys()) == {"dark", "light"}
    assert themes["dark"]["name"] == "dark"
    assert themes["light"]["name"] == "light"


@pytest.mark.parametrize("name", ["dark", "light"])
def test_every_documented_token_resolves(name):
    t = load_theme(name, mode="truecolor")
    for color in THEME_COLOR_KEYS:
        ansi = t.get_fg_ansi(color)
        assert ansi.startswith("\x1b[38;")
        assert t.fg(color, "x") == f"{ansi}x\x1b[39m"
    for bg in THEME_BG_KEYS:
        ansi = t.get_bg_ansi(bg)
        assert ansi.startswith("\x1b[48;")
        assert t.bg(bg, "x") == f"{ansi}x\x1b[49m"


def test_style_helpers_use_standard_sgr_codes():
    t = load_theme("dark", mode="truecolor")
    assert t.bold("x") == "\x1b[1mx\x1b[22m"
    assert t.italic("x") == "\x1b[3mx\x1b[23m"
    assert t.underline("x") == "\x1b[4mx\x1b[24m"
    assert t.inverse("x") == "\x1b[7mx\x1b[27m"
    assert t.strikethrough("x") == "\x1b[9mx\x1b[29m"


def test_unknown_color_tokens_raise():
    t = load_theme("dark", mode="truecolor")
    with pytest.raises(ValueError, match="Unknown theme color"):
        t.fg("notAColor", "x")
    with pytest.raises(ValueError, match="Unknown theme background color"):
        t.bg("notABg", "x")


# ============================================================================
# User theme overrides only the keys it sets
# ============================================================================


def test_user_theme_overrides_only_set_keys(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "custom-accent"
    custom["colors"] = dict(dark["colors"])
    custom["colors"]["accent"] = "#123456"

    theme_path = _write_theme(tmp_path, custom)
    loaded = load_theme_from_path(theme_path, "truecolor")
    baseline = load_theme("dark", mode="truecolor")

    assert loaded.get_fg_ansi("accent") == "\x1b[38;2;18;52;86m"
    # Every other color token is unchanged relative to the base theme.
    for color in THEME_COLOR_KEYS:
        if color == "accent":
            continue
        assert loaded.get_fg_ansi(color) == baseline.get_fg_ansi(color)
    for bg in THEME_BG_KEYS:
        assert loaded.get_bg_ansi(bg) == baseline.get_bg_ansi(bg)


# ============================================================================
# Unknown key / malformed color rejection
# ============================================================================


def test_unknown_top_level_key_is_rejected(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "bad-top-level"
    custom["notAThemeField"] = True
    theme_path = _write_theme(tmp_path, custom)
    with pytest.raises(ValueError, match="Invalid theme"):
        load_theme_from_path(theme_path)


def test_unknown_color_key_is_rejected(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "bad-color-key"
    custom["colors"] = dict(dark["colors"])
    custom["colors"]["notARealColor"] = "#ffffff"
    theme_path = _write_theme(tmp_path, custom)
    with pytest.raises(ValueError, match="Invalid theme"):
        load_theme_from_path(theme_path)


def test_missing_required_color_is_rejected_with_clear_message(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "missing-color"
    custom["colors"] = dict(dark["colors"])
    del custom["colors"]["accent"]
    theme_path = _write_theme(tmp_path, custom)
    with pytest.raises(ValueError, match="Missing required color tokens") as exc_info:
        load_theme_from_path(theme_path)
    assert "accent" in str(exc_info.value)


def test_malformed_hex_color_is_rejected(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "malformed-hex"
    custom["colors"] = dict(dark["colors"])
    custom["colors"]["accent"] = "#zzzzzz"
    theme_path = _write_theme(tmp_path, custom)
    with pytest.raises(ValueError, match="Invalid hex color"):
        load_theme_from_path(theme_path)


def test_out_of_range_256_index_is_rejected(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "bad-256-index"
    custom["colors"] = dict(dark["colors"])
    custom["colors"]["accent"] = 300
    theme_path = _write_theme(tmp_path, custom)
    with pytest.raises(ValueError, match="Invalid theme"):
        load_theme_from_path(theme_path)


def test_dangling_var_reference_is_rejected(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "dangling-var"
    custom["colors"] = dict(dark["colors"])
    custom["colors"]["accent"] = "doesNotExist"
    theme_path = _write_theme(tmp_path, custom)
    with pytest.raises(ValueError, match="Variable reference not found"):
        load_theme_from_path(theme_path)


def test_theme_name_with_slash_is_rejected(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "light/dark"
    theme_path = _write_theme(tmp_path, custom, file_name="slash-name.json")
    with pytest.raises(ValueError, match='cannot contain "/"'):
        load_theme_from_path(theme_path)


# ============================================================================
# Missing theme file falls back to the default
# ============================================================================


def test_get_theme_by_name_returns_none_for_missing_theme(tmp_path):
    assert get_theme_by_name("does-not-exist", custom_themes_dir=str(tmp_path / "themes")) is None


def test_load_theme_raises_for_missing_theme(tmp_path):
    with pytest.raises(ValueError, match="Theme not found"):
        load_theme("does-not-exist", custom_themes_dir=str(tmp_path / "themes"))


# ============================================================================
# Scrollbar theme color fallback (ported from scrollbar-theme.test.ts)
# ============================================================================


def test_scrollbar_falls_back_to_selected_bg_when_omitted(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "legacy-scrollbar-theme"
    custom["colors"] = dict(dark["colors"])
    del custom["colors"]["scrollbarThumb"]

    theme_path = _write_theme(tmp_path, custom)
    loaded = load_theme_from_path(theme_path, "truecolor")
    assert loaded.get_bg_ansi("scrollbarThumb") == loaded.get_bg_ansi("selectedBg")


def test_scrollbar_uses_explicit_value(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "custom-scrollbar-theme"
    custom["colors"] = dict(dark["colors"])
    custom["colors"]["scrollbarThumb"] = "#123456"

    theme_path = _write_theme(tmp_path, custom)
    loaded = load_theme_from_path(theme_path, "truecolor")
    assert loaded.get_bg_ansi("scrollbarThumb") == "\x1b[48;2;18;52;86m"


def test_thinking_max_falls_back_to_thinking_xhigh(tmp_path):
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "legacy-thinking-max"
    custom["colors"] = dict(dark["colors"])
    del custom["colors"]["thinkingMax"]

    theme_path = _write_theme(tmp_path, custom)
    loaded = load_theme_from_path(theme_path, "truecolor")
    assert loaded.get_fg_ansi("thinkingMax") == loaded.get_fg_ansi("thinkingXhigh")


# ============================================================================
# Theme picker (ported from theme-picker.test.ts)
# ============================================================================


def test_picker_uses_custom_theme_content_name_instead_of_file_name(tmp_path):
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "bar"
    theme_path = themes_dir / "foo.json"
    theme_path.write_text(json.dumps(custom))

    names = get_available_themes(custom_themes_dir=str(themes_dir))
    assert "bar" in names
    assert "foo" not in names

    infos = get_available_themes_with_paths(custom_themes_dir=str(themes_dir))
    assert ThemeInfo(name="bar", path=str(theme_path)) in infos
    assert not any(info.name == "foo" for info in infos)


def test_picker_includes_builtin_themes(tmp_path):
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    names = get_available_themes(custom_themes_dir=str(themes_dir))
    assert names == sorted(names)
    assert "dark" in names
    assert "light" in names


# ============================================================================
# getThemeExportColors (ported from theme-export.test.ts)
# ============================================================================


def test_export_colors_resolve_var_references_like_regular_colors(tmp_path):
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "custom-export-vars"
    custom["vars"] = {
        **dark.get("vars", {}),
        "pageBgVar": "#112233",
        "pageBgAlias": "pageBgVar",
        "infoBgVar": "#445566",
        "cardBgVar": "#223344",
    }
    custom["export"] = {"pageBg": "pageBgAlias", "cardBg": "cardBgVar", "infoBg": "infoBgVar"}
    (themes_dir / "custom-export-vars.json").write_text(json.dumps(custom))

    result = get_theme_export_colors("custom-export-vars", custom_themes_dir=str(themes_dir))
    assert result.page_bg == "#112233"
    assert result.card_bg == "#223344"
    assert result.info_bg == "#445566"


def test_export_colors_resolve_recursive_vars_and_256_index(tmp_path):
    themes_dir = tmp_path / "agent" / "themes"
    themes_dir.mkdir(parents=True)
    dark = _read_builtin_json("dark")
    custom = dict(dark)
    custom["name"] = "custom-export-recursive"
    custom["vars"] = {
        **dark.get("vars", {}),
        "deepPageBg": "#abcdef",
        "pageBgAlias": "deepPageBg",
        "cardBgAnsi": 24,
    }
    custom["export"] = {"pageBg": "pageBgAlias", "cardBg": "cardBgAnsi", "infoBg": ""}
    (themes_dir / "custom-export-recursive.json").write_text(json.dumps(custom))

    result = get_theme_export_colors("custom-export-recursive", custom_themes_dir=str(themes_dir))
    assert result.page_bg == "#abcdef"
    assert result.card_bg == "#005f87"
    assert result.info_bg is None


def test_export_colors_empty_for_missing_theme(tmp_path):
    result = get_theme_export_colors("does-not-exist", custom_themes_dir=str(tmp_path / "themes"))
    assert result.page_bg is None
    assert result.card_bg is None
    assert result.info_bg is None


# ============================================================================
# Color mode / ANSI escape sequences (ported from theme-detection.test.ts)
# ============================================================================


def test_hex_to_rgb():
    assert hex_to_rgb("#123456") == (0x12, 0x34, 0x56)
    with pytest.raises(ValueError, match="Invalid hex color"):
        hex_to_rgb("#12345")
    with pytest.raises(ValueError, match="Invalid hex color"):
        hex_to_rgb("#zzzzzz")


def test_hex_to_256_matches_known_conversions():
    # Pure red maps into the color cube, not the grayscale ramp.
    assert hex_to_256("#ff0000") == 196
    # A neutral gray maps into the grayscale ramp.
    assert hex_to_256("#808080") == 244


def test_theme_color_mode_uses_terminal_capabilities():
    set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
    ansi256_theme = get_theme_by_name("dark")
    assert ansi256_theme is not None
    assert ansi256_theme.get_color_mode() == "256color"
    assert re.fullmatch(r"\x1b\[38;5;\d+m", ansi256_theme.get_fg_ansi("accent"))

    set_capabilities(TerminalCapabilities(images=None, true_color=True, hyperlinks=False))
    truecolor_theme = get_theme_by_name("dark")
    assert truecolor_theme is not None
    assert truecolor_theme.get_color_mode() == "truecolor"
    assert re.fullmatch(r"\x1b\[38;2;\d+;\d+;\d+m", truecolor_theme.get_fg_ansi("accent"))


def test_explicit_mode_overrides_capabilities():
    set_capabilities(TerminalCapabilities(images=None, true_color=True, hyperlinks=False))
    forced_256 = load_theme("dark", mode="256color")
    assert forced_256.get_color_mode() == "256color"


# ============================================================================
# Terminal theme detection (ported from theme-detection.test.ts)
# ============================================================================


def test_detect_from_env_uses_colorfgbg_background_index():
    assert detect_terminal_background_from_env(env={"COLORFGBG": "0;15"}).theme == "light"
    assert detect_terminal_background_from_env(env={"COLORFGBG": "0;15"}).source == "COLORFGBG"
    assert detect_terminal_background_from_env(env={"COLORFGBG": "0;15"}).confidence == "high"
    assert detect_terminal_background_from_env(env={"COLORFGBG": "15;0"}).theme == "dark"


def test_detect_from_env_uses_last_colorfgbg_field():
    assert detect_terminal_background_from_env(env={"COLORFGBG": "0;7;15"}).theme == "light"


def test_detect_from_env_defaults_to_dark_without_hints():
    detection = detect_terminal_background_from_env(env={})
    assert detection.theme == "dark"
    assert detection.source == "fallback"
    assert detection.confidence == "low"


class _FakeBackgroundUi:
    def __init__(self, rgb=None, raises=False):
        self._rgb = rgb
        self._raises = raises
        self.queried_timeout_ms = None

    async def query_terminal_background_color(self, timeout_ms):
        self.queried_timeout_ms = timeout_ms
        if self._raises:
            raise RuntimeError("terminal write failed")
        return self._rgb


async def test_detect_background_theme_prefers_terminal_query():
    ui = _FakeBackgroundUi(rgb=RgbColor(r=250, g=250, b=250))
    detection = await detect_terminal_background_theme(ui, 250, env={"COLORFGBG": "15;0"})
    assert ui.queried_timeout_ms == 250
    assert detection.theme == "light"
    assert detection.source == "terminal background"
    assert detection.confidence == "high"


async def test_detect_background_theme_falls_back_when_no_color():
    ui = _FakeBackgroundUi(rgb=None)
    detection = await detect_terminal_background_theme(ui, 250, env={"COLORFGBG": "15;0"})
    assert detection.theme == "dark"
    assert detection.source == "COLORFGBG"
    assert detection.confidence == "high"


async def test_detect_background_theme_falls_back_when_query_fails():
    ui = _FakeBackgroundUi(raises=True)
    detection = await detect_terminal_background_theme(ui, 250, env={"COLORFGBG": "0;15"})
    assert detection.theme == "light"
    assert detection.source == "COLORFGBG"
    assert detection.confidence == "high"


class _FakeAutoUi:
    def __init__(self, color_scheme_coro_factory, background_coro_factory):
        self._color_scheme_coro_factory = color_scheme_coro_factory
        self._background_coro_factory = background_coro_factory

    async def query_terminal_color_scheme(self, timeout_ms):
        return await self._color_scheme_coro_factory()

    async def query_terminal_background_color(self, timeout_ms):
        return await self._background_coro_factory()


async def test_auto_theme_returns_color_scheme_result_without_waiting():
    import asyncio

    color_scheme_future: asyncio.Future = asyncio.get_event_loop().create_future()
    background_started = asyncio.Event()

    async def color_scheme_coro():
        return await color_scheme_future

    async def background_coro():
        background_started.set()
        await asyncio.Future()  # Never resolves.

    ui = _FakeAutoUi(color_scheme_coro, background_coro)
    task = asyncio.ensure_future(detect_terminal_theme_for_auto(ui, 100))
    # TS asserts `backgroundQueryStarted === true` synchronously; the loop has
    # to run for the coroutine to start here, so the wait is bounded to keep a
    # never-started background query a failure rather than a hang.
    await asyncio.wait_for(background_started.wait(), timeout=5)
    color_scheme_future.set_result("dark")
    assert await asyncio.wait_for(task, timeout=5) == "dark"


async def test_auto_theme_uses_background_when_color_scheme_query_fails():
    async def color_scheme_coro():
        raise RuntimeError("color-scheme query failed")

    async def background_coro():
        return RgbColor(r=250, g=250, b=250)

    ui = _FakeAutoUi(color_scheme_coro, background_coro)
    result = await detect_terminal_theme_for_auto(ui, 100)
    assert result == "light"


def test_theme_for_rgb_color_classifies_by_luminance():
    assert get_theme_for_rgb_color(RgbColor(r=8, g=8, b=8)) == "dark"
    assert get_theme_for_rgb_color(RgbColor(r=250, g=250, b=250)) == "light"


def test_parses_and_resolves_automatic_theme_settings():
    setting = parse_auto_theme_setting("light/dark")
    assert setting is not None
    assert setting.light_theme == "light"
    assert setting.dark_theme == "dark"

    assert resolve_theme_setting("dark", "light") == "dark"
    assert resolve_theme_setting("light/dark", "light") == "light"
    assert resolve_theme_setting("light/dark", "dark") == "dark"
    assert resolve_theme_setting("light/dark/extra", "dark") is None


# ============================================================================
# Shipped JSON assets all load and validate
# ============================================================================


@pytest.mark.parametrize("name", ["dark", "light"])
def test_shipped_theme_asset_loads_and_validates(name):
    t = load_theme(name, mode="truecolor")
    assert isinstance(t, Theme)
    assert t.name == name


def test_theme_schema_asset_ships_and_is_valid_json():
    from pi_coding_agent.modes.interactive.theme.theme import _THEME_PACKAGE

    schema = json.loads((_THEME_PACKAGE / "theme-schema.json").read_text(encoding="utf-8"))
    assert schema["title"] == "Pi Coding Agent Theme"
    assert "colors" in schema["properties"]


# ============================================================================
# Registered (in-memory) themes
# ============================================================================


def _resolved_fg_bg_colors(dark: dict) -> tuple[dict, dict]:
    from pi_coding_agent.modes.interactive.theme.theme import (
        resolve_theme_colors,
        with_theme_color_fallbacks,
    )

    resolved = resolve_theme_colors(with_theme_color_fallbacks(dark["colors"]), dark.get("vars"))
    fg_colors = {key: resolved[key] for key in THEME_COLOR_KEYS}
    bg_colors = {key: resolved[key] for key in THEME_BG_KEYS}
    return fg_colors, bg_colors


def test_registered_theme_is_used_over_custom_directory(tmp_path):
    dark = _read_builtin_json("dark")
    fg_colors, bg_colors = _resolved_fg_bg_colors(dark)
    registered = Theme(fg_colors, bg_colors, "truecolor", name="registered-theme")
    set_registered_themes([registered])

    assert load_theme("registered-theme", custom_themes_dir=str(tmp_path / "themes")) is registered
    assert "registered-theme" in get_available_themes(custom_themes_dir=str(tmp_path / "themes"))


def test_set_registered_themes_rejects_slash_in_name():
    dark = _read_builtin_json("dark")
    fg_colors, bg_colors = _resolved_fg_bg_colors(dark)
    bad = Theme(fg_colors, bg_colors, "truecolor", name="light/dark")
    with pytest.raises(ValueError, match='cannot contain "/"'):
        set_registered_themes([bad])

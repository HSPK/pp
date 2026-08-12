"""Tests ported from packages/tui/test/terminal-image.test.ts."""

from __future__ import annotations

import base64
import os
import re
import struct
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pi_tui.components.image import Image, ImageOptions, ImageTheme
from pi_tui.terminal_image import (
    CellDimensions,
    ImageDimensions,
    ImageRenderOptions,
    KittyImageMetadata,
    TerminalCapabilities,
    calculate_image_cell_size,
    calculate_image_rows,
    crop_kitty_image_line,
    delete_all_kitty_images,
    delete_all_kitty_placements,
    delete_kitty_image,
    detect_capabilities,
    encode_iterm2,
    encode_kitty,
    get_capabilities,
    get_cell_dimensions,
    get_kitty_image_metadata,
    get_kitty_image_placement,
    hyperlink,
    image_fallback,
    is_image_line,
    register_kitty_image_metadata,
    render_image,
    reset_capabilities_cache,
    set_capabilities,
    set_cell_dimensions,
)
from pi_tui.utils import visible_width

_ENV_KEYS = [
    "TERM",
    "TERM_PROGRAM",
    "TERMINAL_EMULATOR",
    "COLORTERM",
    "TMUX",
    "KITTY_WINDOW_ID",
    "GHOSTTY_RESOURCES_DIR",
    "WEZTERM_PANE",
    "ITERM_SESSION_ID",
    "WT_SESSION",
    "CMUX_WORKSPACE_ID",
    "WARP_SESSION_ID",
    "WARP_TERMINAL_SESSION_UUID",
]


@contextmanager
def with_env(overrides: dict[str, str | None]) -> Iterator[None]:
    saved: dict[str, str | None] = {}
    for key in _ENV_KEYS:
        saved[key] = os.environ.pop(key, None)
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
            if saved[key] is not None:
                os.environ[key] = saved[key]


class TestIsImageLine:
    def test_detects_iterm2_sequence_at_start(self):
        line = "\x1b]1337;File=size=100,100;inline=1:base64encodeddata==\x07"
        assert is_image_line(line) is True

    def test_detects_iterm2_sequence_with_text_before(self):
        line = "Some text \x1b]1337;File=size=100,100;inline=1:base64data==\x07 more text"
        assert is_image_line(line) is True

    def test_detects_iterm2_sequence_in_middle_of_long_line(self):
        line = "Text before image..." + "\x1b]1337;File=inline=1:verylongbase64data==" + "...text after"
        assert is_image_line(line) is True

    def test_detects_iterm2_sequence_at_end(self):
        line = "Regular text ending with \x1b]1337;File=inline=1:base64data==\x07"
        assert is_image_line(line) is True

    def test_detects_minimal_iterm2_sequence(self):
        assert is_image_line("\x1b]1337;File=:\x07") is True

    def test_detects_kitty_sequence_at_start(self):
        line = "\x1b_Ga=T,f=100,t=f,d=base64data...\x1b\\\x1b_Gm=i=1;\x1b\\"
        assert is_image_line(line) is True

    def test_detects_kitty_sequence_with_text_before(self):
        line = "Output: \x1b_Ga=T,f=100;data...\x1b\\\x1b_Gm=i=1;\x1b\\"
        assert is_image_line(line) is True

    def test_detects_kitty_sequence_with_padding(self):
        line = "  \x1b_Ga=T,f=100...\x1b\\\x1b_Gm=i=1;\x1b\\  "
        assert is_image_line(line) is True

    def test_detects_image_sequences_in_very_long_lines(self):
        base64_char = "A" * 100
        image_sequence = "\x1b]1337;File=size=800,600;inline=1:"
        long_line = "Text prefix " + image_sequence + base64_char * 3000 + " suffix"
        assert len(long_line) > 300000
        assert is_image_line(long_line) is True

    def test_detects_when_terminal_doesnt_support_images(self):
        line = "Read image file [image/jpeg]\x1b]1337;File=inline=1:base64data==\x07"
        assert is_image_line(line) is True

    def test_detects_with_ansi_before(self):
        line = "\x1b[31mError output \x1b]1337;File=inline=1:image==\x07"
        assert is_image_line(line) is True

    def test_detects_with_ansi_after(self):
        line = "\x1b_Ga=T,f=100:data...\x1b\\\x1b_Gm=i=1;\x1b\\\x1b[0m reset"
        assert is_image_line(line) is True

    def test_plain_text_is_not_image(self):
        assert is_image_line("This is just a regular text line without any escape sequences") is False

    def test_only_ansi_codes_not_image(self):
        assert is_image_line("\x1b[31mRed text\x1b[0m and \x1b[32mgreen text\x1b[0m") is False

    def test_cursor_movement_codes_not_image(self):
        assert is_image_line("\x1b[1A\x1b[2KLine cleared and moved up") is False

    def test_partial_iterm2_sequence_not_image(self):
        assert is_image_line("Some text with ]1337;File but missing ESC at start") is False

    def test_partial_kitty_sequence_not_image(self):
        assert is_image_line("Some text with _G but missing ESC at start") is False

    def test_empty_line_not_image(self):
        assert is_image_line("") is False

    def test_newlines_only_not_image(self):
        assert is_image_line("\n") is False
        assert is_image_line("\n\n") is False

    def test_mixed_kitty_and_iterm2(self):
        line = "Kitty: \x1b_Ga=T...\x1b\\\x1b_Gm=i=1;\x1b\\ iTerm2: \x1b]1337;File=inline=1:data==\x07"
        assert is_image_line(line) is True

    def test_multiple_segments(self):
        line = "Start \x1b]1337;File=img1==\x07 middle \x1b]1337;File=img2==\x07 end"
        assert is_image_line(line) is True

    def test_file_path_with_keywords_not_image(self):
        assert is_image_line("/path/to/File_1337_backup/image.jpg") is False


class TestDetectCapabilities:
    def test_defaults_to_no_hyperlinks_for_unknown_terminals(self):
        with with_env({}):
            caps = detect_capabilities()
            assert caps.hyperlinks is False
            assert caps.images is None

    def test_enables_hyperlinks_under_tmux_when_forwarded(self):
        with with_env({"TMUX": "/tmp/tmux-1000/default,1234,0", "TERM_PROGRAM": "ghostty"}):
            caps = detect_capabilities(lambda: True)
            assert caps.hyperlinks is True
            assert caps.images is None

    def test_disables_hyperlinks_under_tmux_when_not_forwarded(self):
        with with_env({"TMUX": "/tmp/tmux-1000/default,1234,0", "TERM_PROGRAM": "ghostty"}):
            caps = detect_capabilities(lambda: False)
            assert caps.hyperlinks is False
            assert caps.images is None

    def test_checks_tmux_capability_when_term_starts_with_tmux(self):
        with with_env({"TERM": "tmux-256color", "TERM_PROGRAM": "iterm.app"}):
            caps = detect_capabilities(lambda: True)
            assert caps.hyperlinks is True
            assert caps.images is None

            caps2 = detect_capabilities(lambda: False)
            assert caps2.hyperlinks is False

    def test_forces_no_hyperlinks_for_screen(self):
        with with_env({"TERM": "screen-256color"}):
            caps = detect_capabilities()
            assert caps.hyperlinks is False
            assert caps.images is None

    def test_enables_hyperlinks_for_ghostty(self):
        with with_env({"TERM_PROGRAM": "ghostty"}):
            assert detect_capabilities().hyperlinks is True

    def test_ghostty_images_not_disabled_by_cmux(self):
        with with_env({"TERM_PROGRAM": "ghostty", "CMUX_WORKSPACE_ID": "workspace"}):
            caps = detect_capabilities()
            assert caps.images == "kitty"
            assert caps.hyperlinks is True

    def test_enables_hyperlinks_for_kitty(self):
        with with_env({"KITTY_WINDOW_ID": "1"}):
            assert detect_capabilities().hyperlinks is True

    def test_enables_hyperlinks_for_wezterm(self):
        with with_env({"WEZTERM_PANE": "0"}):
            assert detect_capabilities().hyperlinks is True

    def test_enables_images_and_hyperlinks_for_warp_via_term_program(self):
        with with_env({"TERM_PROGRAM": "WarpTerminal"}):
            caps = detect_capabilities()
            assert caps.images == "kitty"
            assert caps.true_color is True
            assert caps.hyperlinks is True

    def test_enables_images_and_hyperlinks_for_warp_via_session_id(self):
        with with_env({"WARP_SESSION_ID": "some-session-id"}):
            caps = detect_capabilities()
            assert caps.images == "kitty"
            assert caps.true_color is True
            assert caps.hyperlinks is True

    def test_enables_images_and_hyperlinks_for_warp_via_terminal_session_uuid(self):
        with with_env({"WARP_TERMINAL_SESSION_UUID": "d0e1a2e5-7ca7-44cd-9037-ac7222011161"}):
            caps = detect_capabilities()
            assert caps.images == "kitty"
            assert caps.true_color is True
            assert caps.hyperlinks is True

    def test_disables_images_for_warp_inside_tmux(self):
        with with_env(
            {
                "TERM_PROGRAM": "WarpTerminal",
                "TMUX": "/tmp/tmux-1000/default,1234,0",
                "TERM": "tmux-256color",
            }
        ):
            caps = detect_capabilities(lambda: True)
            assert caps.images is None
            assert caps.hyperlinks is True

    def test_enables_hyperlinks_for_iterm2(self):
        with with_env({"TERM_PROGRAM": "iterm.app"}):
            assert detect_capabilities().hyperlinks is True

    def test_enables_hyperlinks_for_vscode(self):
        with with_env({"TERM_PROGRAM": "vscode"}):
            assert detect_capabilities().hyperlinks is True

    def test_enables_truecolor_and_hyperlinks_for_windows_terminal(self):
        with with_env({"WT_SESSION": "session", "TERM": "xterm-256color"}):
            caps = detect_capabilities()
            assert caps.true_color is True
            assert caps.hyperlinks is True
            assert caps.images is None

    def test_enables_truecolor_without_hyperlinks_for_jetbrains(self):
        with with_env({"TERMINAL_EMULATOR": "JetBrains-JediTerm", "TERM": "xterm-256color"}):
            caps = detect_capabilities()
            assert caps.true_color is True
            assert caps.hyperlinks is False
            assert caps.images is None

    def test_does_not_inherit_windows_terminal_truecolor_through_tmux(self):
        with with_env({"WT_SESSION": "session", "TMUX": "/tmp/tmux-1000/default,1234,0", "TERM": "tmux-256color"}):
            caps = detect_capabilities(lambda: False)
            assert caps.true_color is False
            assert caps.hyperlinks is False
            assert caps.images is None

    def test_trusts_explicit_truecolor_hints_through_tmux(self):
        with with_env({"COLORTERM": "truecolor", "TMUX": "/tmp/tmux-1000/default,1234,0", "TERM": "tmux-256color"}):
            caps = detect_capabilities(lambda: False)
            assert caps.true_color is True
            assert caps.hyperlinks is False
            assert caps.images is None

    def test_enables_truecolor_and_hyperlinks_for_alacritty(self):
        with with_env({"TERM_PROGRAM": "alacritty"}):
            caps = detect_capabilities()
            assert caps.true_color is True
            assert caps.hyperlinks is True
            assert caps.images is None

    def test_windows_console_gets_truecolor_without_hyperlinks(self, monkeypatch: pytest.MonkeyPatch):
        import pi_tui.terminal_image as terminal_image_module

        monkeypatch.setattr(terminal_image_module.sys, "platform", "win32")
        with with_env({}):
            caps = detect_capabilities()
            assert caps.images is None
            assert caps.true_color is True
            assert caps.hyperlinks is False

    def test_get_capabilities_caches_detection_result(self):
        reset_capabilities_cache()
        try:
            with with_env({"TERM_PROGRAM": "ghostty"}):
                first = get_capabilities()
                # Env changes after the first call should not affect the cached result.
                with with_env({"TERM_PROGRAM": "iterm.app"}):
                    second = get_capabilities()
                assert first is second
                assert second.images == "kitty"
        finally:
            reset_capabilities_cache()


class TestProbeTmuxHyperlinks:
    def test_returns_true_when_client_termfeatures_lists_hyperlinks(self, monkeypatch: pytest.MonkeyPatch):
        from pi_tui.terminal_image import _probe_tmux_hyperlinks

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="256,hyperlinks,clipboard\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _probe_tmux_hyperlinks() is True

    def test_returns_false_when_hyperlinks_not_listed(self, monkeypatch: pytest.MonkeyPatch):
        from pi_tui.terminal_image import _probe_tmux_hyperlinks

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="256,clipboard\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _probe_tmux_hyperlinks() is False

    def test_returns_false_when_tmux_invocation_fails(self, monkeypatch: pytest.MonkeyPatch):
        from pi_tui.terminal_image import _probe_tmux_hyperlinks

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("tmux not found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _probe_tmux_hyperlinks() is False


class TestIterm2ImageEncoding:
    def test_includes_decoded_payload_size(self):
        sequence = encode_iterm2("AAAA", width=2, height="auto")
        assert sequence == "\x1b]1337;File=inline=1;size=3;width=2;height=auto:AAAA\x07"

    def test_omits_width_and_height_when_not_provided(self):
        sequence = encode_iterm2("AAAA")
        assert sequence == "\x1b]1337;File=inline=1;size=3:AAAA\x07"

    def test_encodes_name_as_base64(self):
        sequence = encode_iterm2("AAAA", name="shot.png")
        expected_name = base64.b64encode(b"shot.png").decode()
        assert f"name={expected_name}" in sequence

    def test_can_disable_preserve_aspect_ratio(self):
        sequence = encode_iterm2("AAAA", preserve_aspect_ratio=False)
        assert "preserveAspectRatio=0" in sequence

    def test_can_disable_inline_display(self):
        sequence = encode_iterm2("AAAA", inline=False)
        assert sequence.startswith("\x1b]1337;File=inline=0;")

    def test_decoded_byte_length_falls_back_to_zero_for_invalid_base64(self):
        from pi_tui.terminal_image import _decoded_byte_length

        assert _decoded_byte_length("not-valid-base64!!!") == 0


class TestKittyImageCursorMovement:
    def test_can_request_no_cursor_movement(self):
        sequence = encode_kitty("AAAA", columns=2, rows=2, move_cursor=False)
        assert sequence.startswith("\x1b_Ga=T,f=100,q=2,C=1,c=2,r=2;")

    def test_omits_optional_params_when_not_provided(self):
        sequence = encode_kitty("AAAA")
        assert sequence == "\x1b_Ga=T,f=100,q=2;AAAA\x1b\\"

    def test_splits_large_payloads_into_chunks_with_middle_frames(self):
        # 3 chunks: first (m=1), a middle chunk (m=1, no params), and the last (m=0).
        payload = "A" * (4096 * 2 + 10)
        sequence = encode_kitty(payload, columns=5, rows=5, image_id=7)
        assert sequence.count("\x1b_G") == 3
        assert sequence.startswith("\x1b_Ga=T,f=100,q=2,c=5,r=5,i=7,m=1;")
        middle_index = sequence.index("\x1b_Gm=1;")
        assert middle_index > 0
        assert "\x1b_Gm=0;" in sequence
        assert sequence.endswith("\x1b\\")

    def test_suppresses_replies_for_delete_commands(self):
        assert delete_kitty_image(42) == "\x1b_Ga=d,d=I,i=42,q=2\x1b\\"
        assert delete_all_kitty_images() == "\x1b_Ga=d,d=A,q=2\x1b\\"
        assert delete_all_kitty_placements() == "\x1b_Ga=d,d=a,q=2\x1b\\"

    def test_preserves_render_image_default_cursor_movement(self):
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            result = render_image("AAAA", ImageDimensions(20, 20), ImageRenderOptions(max_width_cells=2))
            assert result is not None
            assert ",C=1," not in result.sequence
            assert result.rows == 2
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))

    def test_can_opt_render_image_into_no_cursor_movement(self):
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            result = render_image(
                "AAAA", ImageDimensions(20, 20), ImageRenderOptions(max_width_cells=2, move_cursor=False)
            )
            assert result is not None
            assert ",C=1," in result.sequence
            assert result.rows == 2
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))

    def test_registers_metadata_and_crops_partial_placement(self):
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            result = render_image(
                "AAAA",
                ImageDimensions(100, 100),
                ImageRenderOptions(max_width_cells=3, image_id=42, move_cursor=False),
            )
            assert result is not None
            metadata = get_kitty_image_metadata(result.sequence)
            assert metadata is not None
            assert metadata.image_id == 42
            assert metadata.columns == 3
            assert metadata.rows == 3
            assert metadata.width_px == 100
            assert metadata.height_px == 100
            assert "y=66,h=34,r=1" in crop_kitty_image_line(result.sequence, 2, 1)
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))

    def test_creates_placement_only_commands(self):
        register_kitty_image_metadata(KittyImageMetadata(image_id=42, columns=3, rows=3, width_px=100, height_px=100))
        transmission = encode_kitty("A" * 8192, columns=3, rows=3, image_id=42, move_cursor=False)
        line = f"left {crop_kitty_image_line(transmission, 2, 1)} right"
        placement = get_kitty_image_placement(line)
        assert placement is not None
        assert placement.transmission_bytes == len(line) - len("left ") - len(" right")
        assert placement.estimated_decoded_bytes == 100 * 100 * 4
        assert placement.sequence == "\x1b_Ga=p,q=2,C=1,c=3,i=42,y=66,h=34,r=1\x1b\\"
        assert placement.replacement_line == f"left {placement.sequence} right"
        assert "AAAA" not in placement.replacement_line

    def test_honors_max_height_cells(self):
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            result = render_image(
                "AAAA", ImageDimensions(10, 100), ImageRenderOptions(max_width_cells=10, max_height_cells=5)
            )
            assert result is not None
            assert result.rows == 5
            assert ",c=1,r=5" in result.sequence
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))

    def test_caps_image_component_height_to_square_pixel_box(self):
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 20))
        try:
            image = Image(
                "AAAA",
                "image/png",
                ImageTheme(fallback_color=lambda value: value),
                ImageOptions(max_width_cells=10),
                ImageDimensions(10, 100),
            )
            lines = image.render(12)
            assert len(lines) == 5
            assert ",c=1,r=5" in lines[0]
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))

    def test_places_image_sequence_on_first_line_with_empty_padding(self):
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            image = Image(
                "AAAA",
                "image/png",
                ImageTheme(fallback_color=lambda value: value),
                ImageOptions(max_width_cells=2),
                ImageDimensions(20, 20),
            )
            lines = image.render(4)
            image_id = image.get_image_id()
            assert isinstance(image_id, int)
            assert lines[0].startswith("\x1b_G")
            assert ",C=1," in lines[0]
            assert f",i={image_id}" in lines[0]
            assert lines[0].endswith("\x1b\\")
            assert lines[1:] == [""]
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))

    def test_truncates_long_fallback_lines_to_render_width(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        try:
            long_path = str(Path.home() / "images" / (("generated-image-with-a-very-long-absolute-path" * 4) + ".png"))
            width = 40
            image = Image(
                "AAAA",
                "image/png",
                ImageTheme(fallback_color=lambda value: f"\x1b[33m{value}\x1b[0m"),
                ImageOptions(filename=long_path),
                ImageDimensions(1280, 720),
            )
            lines = image.render(width)
            assert len(lines) == 1
            assert visible_width(lines[0]) <= width
            assert "..." in lines[0]
            assert "~" in lines[0]
        finally:
            reset_capabilities_cache()

    def test_reuses_cached_lines_for_same_render_width(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        try:
            calls = 0

            def fallback_color(value: str) -> str:
                nonlocal calls
                calls += 1
                return value

            image = Image(
                "AAAA", "image/png", ImageTheme(fallback_color=fallback_color), dimensions=ImageDimensions(10, 10)
            )
            first = image.render(40)
            second = image.render(40)
            assert second is first
            assert calls == 1
        finally:
            reset_capabilities_cache()

    def test_invalidate_clears_cache_and_forces_rerender(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        try:
            calls = 0

            def fallback_color(value: str) -> str:
                nonlocal calls
                calls += 1
                return value

            image = Image(
                "AAAA", "image/png", ImageTheme(fallback_color=fallback_color), dimensions=ImageDimensions(10, 10)
            )
            first = image.render(40)
            image.invalidate()
            second = image.render(40)
            assert second is not first
            assert second == first
            assert calls == 2
        finally:
            reset_capabilities_cache()

    def test_reuses_provided_kitty_image_id_without_allocating(self):
        set_capabilities(TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            image = Image(
                "AAAA",
                "image/png",
                ImageTheme(fallback_color=lambda value: value),
                ImageOptions(max_width_cells=2, image_id=99),
                ImageDimensions(20, 20),
            )
            lines = image.render(4)
            assert image.get_image_id() == 99
            assert ",i=99" in lines[0]
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))

    def test_renders_iterm2_sequence_with_cursor_move_up_on_last_line(self):
        set_capabilities(TerminalCapabilities(images="iterm2", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            image = Image(
                "AAAA",
                "image/png",
                ImageTheme(fallback_color=lambda value: value),
                ImageOptions(max_width_cells=5, max_height_cells=3),
                ImageDimensions(100, 100),
            )
            lines = image.render(20)
            assert image.get_image_id() is None
            assert len(lines) == 3
            assert lines[0] == ""
            assert lines[1] == ""
            assert lines[2].startswith("\x1b[2A")
            assert "\x1b]1337;File=" in lines[2]
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))

    def test_renders_iterm2_sequence_without_cursor_move_when_single_row(self):
        set_capabilities(TerminalCapabilities(images="iterm2", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            image = Image(
                "AAAA",
                "image/png",
                ImageTheme(fallback_color=lambda value: value),
                ImageOptions(max_width_cells=5, max_height_cells=1),
                ImageDimensions(100, 100),
            )
            lines = image.render(20)
            assert len(lines) == 1
            assert not lines[0].startswith("\x1b[")
            assert "\x1b]1337;File=" in lines[0]
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))


class TestKittyImageMetadataRegistration:
    def test_evicts_oldest_entry_once_over_one_thousand_registrations(self):
        import pi_tui.terminal_image as terminal_image_module

        terminal_image_module._kitty_image_metadata.clear()
        try:
            for image_id in range(1, 1002):
                register_kitty_image_metadata(
                    KittyImageMetadata(image_id=image_id, columns=1, rows=1, width_px=1, height_px=1)
                )
            assert len(terminal_image_module._kitty_image_metadata) == 1000
            # The very first registered image (id=1) should have been evicted.
            assert get_kitty_image_metadata("\x1b_Gi=1;AAAA\x1b\\") is None
            # The most recent registration should still be present.
            assert get_kitty_image_metadata("\x1b_Gi=1001;AAAA\x1b\\") is not None
        finally:
            terminal_image_module._kitty_image_metadata.clear()


class TestGetKittyImagePlacementInvalid:
    def test_returns_none_when_line_has_no_kitty_sequence(self):
        assert get_kitty_image_placement("plain text, no image here") is None

    def test_returns_none_when_metadata_was_never_registered(self):
        assert get_kitty_image_placement("\x1b_Gi=999999,c=1,r=1;AAAA\x1b\\") is None

    def test_returns_none_when_transmission_terminator_missing(self):
        register_kitty_image_metadata(KittyImageMetadata(image_id=55, columns=1, rows=1, width_px=1, height_px=1))
        try:
            assert get_kitty_image_placement("\x1b_Gi=55;AAAA") is None
        finally:
            import pi_tui.terminal_image as terminal_image_module

            terminal_image_module._kitty_image_metadata.pop(55, None)

    def test_follows_chained_multipart_transmission_to_find_placement(self):
        register_kitty_image_metadata(KittyImageMetadata(image_id=66, columns=2, rows=2, width_px=10, height_px=10))
        try:
            transmission = encode_kitty("A" * 8192, columns=2, rows=2, image_id=66)
            placement = get_kitty_image_placement(transmission)
            assert placement is not None
            assert placement.image_id == 66
        finally:
            import pi_tui.terminal_image as terminal_image_module

            terminal_image_module._kitty_image_metadata.pop(66, None)


class TestCropKittyImageLineInvalid:
    def test_returns_line_unchanged_without_registered_metadata(self):
        line = "\x1b_Gi=123456,c=1,r=1;AAAA\x1b\\"
        assert crop_kitty_image_line(line, 0, 1) == line

    def test_returns_line_unchanged_for_negative_hidden_rows(self):
        register_kitty_image_metadata(KittyImageMetadata(image_id=77, columns=1, rows=4, width_px=10, height_px=40))
        try:
            transmission = encode_kitty("AAAA", columns=1, rows=4, image_id=77)
            assert crop_kitty_image_line(transmission, -1, 2) == transmission
        finally:
            import pi_tui.terminal_image as terminal_image_module

            terminal_image_module._kitty_image_metadata.pop(77, None)

    def test_returns_line_unchanged_when_hidden_rows_meets_or_exceeds_total(self):
        register_kitty_image_metadata(KittyImageMetadata(image_id=78, columns=1, rows=4, width_px=10, height_px=40))
        try:
            transmission = encode_kitty("AAAA", columns=1, rows=4, image_id=78)
            assert crop_kitty_image_line(transmission, 4, 2) == transmission
        finally:
            import pi_tui.terminal_image as terminal_image_module

            terminal_image_module._kitty_image_metadata.pop(78, None)

    def test_returns_line_unchanged_for_non_positive_visible_rows(self):
        register_kitty_image_metadata(KittyImageMetadata(image_id=79, columns=1, rows=4, width_px=10, height_px=40))
        try:
            transmission = encode_kitty("AAAA", columns=1, rows=4, image_id=79)
            assert crop_kitty_image_line(transmission, 1, 0) == transmission
        finally:
            import pi_tui.terminal_image as terminal_image_module

            terminal_image_module._kitty_image_metadata.pop(79, None)

    def test_returns_line_unchanged_when_all_rows_stay_visible(self):
        register_kitty_image_metadata(KittyImageMetadata(image_id=80, columns=1, rows=4, width_px=10, height_px=40))
        try:
            transmission = encode_kitty("AAAA", columns=1, rows=4, image_id=80)
            assert crop_kitty_image_line(transmission, 0, 4) == transmission
        finally:
            import pi_tui.terminal_image as terminal_image_module

            terminal_image_module._kitty_image_metadata.pop(80, None)


class TestCalculateImageCellSize:
    def test_scales_down_to_fit_max_width_only(self):
        size = calculate_image_cell_size(ImageDimensions(200, 100), 10, None, CellDimensions(10, 10))
        assert size.columns == 10
        assert size.rows == 5

    def test_scales_down_to_fit_both_width_and_height_bounds(self):
        size = calculate_image_cell_size(ImageDimensions(100, 100), 10, 3, CellDimensions(10, 10))
        assert size.rows == 3
        assert size.columns <= 10

    def test_uses_default_cell_dimensions_when_not_provided(self):
        default_dims = get_cell_dimensions()
        with_explicit = calculate_image_cell_size(ImageDimensions(100, 100), 10, None, default_dims)
        with_default = calculate_image_cell_size(ImageDimensions(100, 100), 10)
        assert with_explicit == with_default

    def test_never_returns_less_than_one_column_or_row(self):
        size = calculate_image_cell_size(ImageDimensions(1, 1), 0, 0, CellDimensions(10, 10))
        assert size.columns >= 1
        assert size.rows >= 1

    def test_calculate_image_rows_matches_cell_size_rows(self):
        rows = calculate_image_rows(ImageDimensions(200, 100), 10, CellDimensions(10, 10))
        assert rows == calculate_image_cell_size(ImageDimensions(200, 100), 10, None, CellDimensions(10, 10)).rows


class TestRenderImage:
    def test_returns_none_when_terminal_has_no_image_support(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        try:
            assert render_image("AAAA", ImageDimensions(10, 10)) is None
        finally:
            reset_capabilities_cache()

    def test_renders_iterm2_sequence_with_preserved_aspect_ratio_by_default(self):
        set_capabilities(TerminalCapabilities(images="iterm2", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            result = render_image("AAAA", ImageDimensions(100, 100), ImageRenderOptions(max_width_cells=5))
            assert result is not None
            assert result.image_id is None
            assert "\x1b]1337;File=" in result.sequence
            assert "preserveAspectRatio=0" not in result.sequence
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))

    def test_iterm2_can_disable_preserve_aspect_ratio(self):
        set_capabilities(TerminalCapabilities(images="iterm2", true_color=True, hyperlinks=True))
        set_cell_dimensions(CellDimensions(10, 10))
        try:
            result = render_image(
                "AAAA",
                ImageDimensions(100, 100),
                ImageRenderOptions(max_width_cells=5, preserve_aspect_ratio=False),
            )
            assert result is not None
            assert "preserveAspectRatio=0" in result.sequence
        finally:
            reset_capabilities_cache()
            set_cell_dimensions(CellDimensions(9, 18))


class TestImageFallback:
    def test_shortens_home_prefixed_absolute_paths_without_hyperlinks(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        try:
            abs_path = str(Path.home() / ".pi" / "agent" / "shot.png")
            result = image_fallback("image/png", ImageDimensions(1280, 720), abs_path)
            assert result == "[Image: ~/.pi/agent/shot.png [image/png] 1280x720]"
        finally:
            reset_capabilities_cache()

    def test_wraps_shortened_paths_in_osc8_when_hyperlinks_enabled(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=True))
        try:
            abs_path = str(Path.home() / ".pi" / "agent" / "shot.png")
            result = image_fallback("image/png", ImageDimensions(10, 10), abs_path)
            assert "\x1b]8;;file://" in result
            assert abs_path.replace("\\", "/") in result or abs_path in result

            visible = re.sub(r"\x1b\]8;;.*?\x1b\\", "", result)
            assert visible == "[Image: ~/.pi/agent/shot.png [image/png] 10x10]"
        finally:
            reset_capabilities_cache()

    def test_leaves_bare_basenames_unchanged(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=True))
        try:
            result = image_fallback("image/png", ImageDimensions(1, 1), "clankolas.png")
            assert result == "[Image: clankolas.png [image/png] 1x1]"
            assert "\x1b]8;" not in result
        finally:
            reset_capabilities_cache()

    def test_omits_filename_segment_when_not_provided(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        try:
            assert image_fallback("image/png", ImageDimensions(8, 6)) == "[Image: [image/png] 8x6]"
        finally:
            reset_capabilities_cache()

    def test_omits_dimensions_segment_when_not_provided(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=False))
        try:
            assert image_fallback("image/png") == "[Image: [image/png]]"
        finally:
            reset_capabilities_cache()

    def test_does_not_hyperlink_relative_paths_even_when_supported(self):
        set_capabilities(TerminalCapabilities(images=None, true_color=False, hyperlinks=True))
        try:
            result = image_fallback("image/png", ImageDimensions(1, 1), "relative/shot.png")
            assert result == "[Image: relative/shot.png [image/png] 1x1]"
            assert "\x1b]8;" not in result
        finally:
            reset_capabilities_cache()


class TestHyperlink:
    def test_wraps_text_in_osc8_sequences(self):
        result = hyperlink("click me", "https://example.com")
        assert result == "\x1b]8;;https://example.com\x1b\\click me\x1b]8;;\x1b\\"

    def test_preserves_ansi_styling_inside_hyperlink(self):
        styled = "\x1b[4m\x1b[34mclick me\x1b[0m"
        result = hyperlink(styled, "https://example.com")
        assert result.startswith("\x1b]8;;https://example.com\x1b\\")
        assert styled in result
        assert result.endswith("\x1b]8;;\x1b\\")

    def test_works_with_empty_text(self):
        assert hyperlink("", "https://example.com") == "\x1b]8;;https://example.com\x1b\\\x1b]8;;\x1b\\"

    def test_works_with_file_uris(self):
        result = hyperlink("README.md", "file:///home/user/README.md")
        assert "file:///home/user/README.md" in result
        assert "README.md" in result


class TestImageDimensionParsing:
    def test_parses_png_dimensions(self):
        from pi_tui.terminal_image import get_png_dimensions

        # 2x1 PNG (minimal valid header + IHDR chunk we control).
        header = b"\x89PNG\r\n\x1a\n"
        ihdr_len = (13).to_bytes(4, "big")
        ihdr_type = b"IHDR"
        width_height = (2).to_bytes(4, "big") + (1).to_bytes(4, "big")
        rest = b"\x08\x02\x00\x00\x00" + b"\x00\x00\x00\x00"  # bit depth/etc + fake CRC
        data = header + ihdr_len + ihdr_type + width_height + rest
        b64 = base64.b64encode(data).decode()
        dims = get_png_dimensions(b64)
        assert dims == ImageDimensions(2, 1)

    def test_returns_none_for_invalid_png(self):
        from pi_tui.terminal_image import get_png_dimensions

        assert get_png_dimensions(base64.b64encode(b"not a png").decode()) is None

    def test_returns_none_for_garbage_base64(self):
        from pi_tui.terminal_image import get_png_dimensions

        assert get_png_dimensions("!!!not-base64!!!") is None

    def test_get_image_dimensions_dispatches_by_mime(self):
        from pi_tui.terminal_image import get_image_dimensions

        assert get_image_dimensions("AAAA", "image/unknown") is None

    def _make_jpeg(self, width: int, height: int, *, extra_segment: bool = False) -> bytes:
        soi = b"\xff\xd8"
        prefix = b""
        if extra_segment:
            # A harmless comment segment (0xFFFE) that get_jpeg_dimensions must skip over.
            prefix = b"\xff\xfe" + struct.pack(">H", 4) + b"hi"
        sof = (
            b"\xff\xc0"
            + struct.pack(">H", 8)
            + b"\x08"
            + struct.pack(">H", height)
            + struct.pack(">H", width)
            + b"\x01"
        )
        return soi + prefix + sof

    def test_parses_jpeg_dimensions_from_sof0_marker(self):
        from pi_tui.terminal_image import get_jpeg_dimensions

        data = self._make_jpeg(4, 3)
        assert get_jpeg_dimensions(base64.b64encode(data).decode()) == ImageDimensions(4, 3)

    def test_parses_jpeg_dimensions_after_skipping_other_segments(self):
        from pi_tui.terminal_image import get_jpeg_dimensions

        data = self._make_jpeg(15, 20, extra_segment=True)
        assert get_jpeg_dimensions(base64.b64encode(data).decode()) == ImageDimensions(15, 20)

    def test_returns_none_for_jpeg_missing_sof_marker(self):
        from pi_tui.terminal_image import get_jpeg_dimensions

        truncated = b"\xff\xd8\xff\xfe" + struct.pack(">H", 4) + b"hi"
        assert get_jpeg_dimensions(base64.b64encode(truncated).decode()) is None

    def test_returns_none_for_non_jpeg_signature(self):
        from pi_tui.terminal_image import get_jpeg_dimensions

        assert get_jpeg_dimensions(base64.b64encode(b"\x00\x01" + b"\x00" * 10).decode()) is None

    def test_returns_none_for_too_short_jpeg_buffer(self):
        from pi_tui.terminal_image import get_jpeg_dimensions

        assert get_jpeg_dimensions(base64.b64encode(b"\xff").decode()) is None

    def test_parses_gif87a_dimensions(self):
        from pi_tui.terminal_image import get_gif_dimensions

        data = b"GIF87a" + struct.pack("<HH", 5, 6) + b"\x00\x00\x00"
        assert get_gif_dimensions(base64.b64encode(data).decode()) == ImageDimensions(5, 6)

    def test_parses_gif89a_dimensions(self):
        from pi_tui.terminal_image import get_gif_dimensions

        data = b"GIF89a" + struct.pack("<HH", 7, 8) + b"\x00\x00\x00"
        assert get_gif_dimensions(base64.b64encode(data).decode()) == ImageDimensions(7, 8)

    def test_returns_none_for_invalid_gif_signature(self):
        from pi_tui.terminal_image import get_gif_dimensions

        data = b"NOTAGIF!!" + b"\x00"
        assert get_gif_dimensions(base64.b64encode(data).decode()) is None

    def test_returns_none_for_too_short_gif_buffer(self):
        from pi_tui.terminal_image import get_gif_dimensions

        assert get_gif_dimensions(base64.b64encode(b"GIF89a").decode()) is None

    def _make_webp_vp8(self, width: int, height: int) -> bytes:
        payload = b"\x00" * 6 + struct.pack("<H", width & 0x3FFF) + struct.pack("<H", height & 0x3FFF)
        chunk = b"VP8 " + struct.pack("<I", len(payload)) + payload
        return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk

    def _make_webp_vp8l(self, width: int, height: int) -> bytes:
        bits = ((width - 1) & 0x3FFF) | (((height - 1) & 0x3FFF) << 14)
        # Padded so the overall buffer clears the format-agnostic 30-byte minimum.
        payload = b"\x2f" + struct.pack("<I", bits) + b"\x00" * 5
        chunk = b"VP8L" + struct.pack("<I", len(payload)) + payload
        return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk

    def _make_webp_vp8x(self, width: int, height: int) -> bytes:
        w = width - 1
        h = height - 1
        payload = b"\x00" * 4 + bytes([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF])
        payload += bytes([h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF])
        chunk = b"VP8X" + struct.pack("<I", len(payload)) + payload
        return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk

    def test_parses_webp_vp8_lossy_dimensions(self):
        from pi_tui.terminal_image import get_webp_dimensions

        data = self._make_webp_vp8(7, 9)
        assert get_webp_dimensions(base64.b64encode(data).decode()) == ImageDimensions(7, 9)

    def test_parses_webp_vp8l_lossless_dimensions(self):
        from pi_tui.terminal_image import get_webp_dimensions

        data = self._make_webp_vp8l(10, 12)
        assert get_webp_dimensions(base64.b64encode(data).decode()) == ImageDimensions(10, 12)

    def test_parses_webp_vp8x_extended_dimensions(self):
        from pi_tui.terminal_image import get_webp_dimensions

        data = self._make_webp_vp8x(20, 30)
        assert get_webp_dimensions(base64.b64encode(data).decode()) == ImageDimensions(20, 30)

    def test_returns_none_for_unknown_webp_chunk_type(self):
        from pi_tui.terminal_image import get_webp_dimensions

        chunk = b"XXXX" + struct.pack("<I", 8) + b"\x00" * 8
        data = b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk
        assert get_webp_dimensions(base64.b64encode(data).decode()) is None

    def test_returns_none_for_non_riff_or_non_webp_signature(self):
        from pi_tui.terminal_image import get_webp_dimensions

        data = b"NOPE" + b"\x00" * 26
        assert get_webp_dimensions(base64.b64encode(data).decode()) is None

    def test_returns_none_for_too_short_webp_buffer(self):
        from pi_tui.terminal_image import get_webp_dimensions

        assert get_webp_dimensions(base64.b64encode(b"RIFF" + b"\x00" * 10).decode()) is None

    def test_get_image_dimensions_dispatches_jpeg_gif_and_webp(self):
        from pi_tui.terminal_image import get_image_dimensions

        jpeg = base64.b64encode(self._make_jpeg(4, 3)).decode()
        gif = base64.b64encode(b"GIF89a" + struct.pack("<HH", 5, 6) + b"\x00\x00\x00").decode()
        webp = base64.b64encode(self._make_webp_vp8(7, 9)).decode()

        assert get_image_dimensions(jpeg, "image/jpeg") == ImageDimensions(4, 3)
        assert get_image_dimensions(gif, "image/gif") == ImageDimensions(5, 6)
        assert get_image_dimensions(webp, "image/webp") == ImageDimensions(7, 9)

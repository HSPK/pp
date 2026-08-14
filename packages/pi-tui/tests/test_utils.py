"""Tests ported from packages/tui/test/truncate-to-width.test.ts,
tab-width.test.ts, wrap-ansi.test.ts, overlay-options.test.ts,
tui-alt-screen.test.ts, terminal-image.test.ts, and
regression-regional-indicator-width.test.ts.
"""

import asyncio

import pi_tui.utils as utils_module
import pytest
from pi_tui.testing import FakeTerminal, MiniTerminalModel
from pi_tui.tui import OverlayOptions
from pi_tui.tui_main_screen import TuiMainScreen
from pi_tui.utils import (
    apply_background_to_line,
    extract_ansi_code,
    extract_segments,
    get_grapheme_cell_range,
    get_osc8_link_at_column,
    normalize_terminal_output,
    slice_by_column,
    slice_with_width,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
)


class _FullViewportContent:
    def render(self, width: int) -> list[str]:
        return [line.ljust(width) for line in ("base 0", "base 1", "base 2")]

    def invalidate(self) -> None:
        return None


class _TabStatusOverlay:
    def render(self, _width: int) -> list[str]:
        return ["\tX"]

    def invalidate(self) -> None:
        return None


class TestTruncateToWidth:
    def test_keeps_output_within_width_for_very_large_unicode_input(self):
        text = "🙂界" * 100_000
        truncated = truncate_to_width(text, 40, "…")

        assert visible_width(truncated) <= 40
        assert truncated.endswith("…\x1b[0m")

    def test_preserves_ansi_styling_for_kept_text(self):
        text = f"\x1b[31m{'hello ' * 1000}\x1b[0m"
        truncated = truncate_to_width(text, 20, "…")

        assert visible_width(truncated) <= 20
        assert "\x1b[31m" in truncated
        assert truncated.endswith("\x1b[0m…\x1b[0m")

    def test_closes_bel_terminated_osc8_link_when_truncating_label(self):
        open_seq = "\x1b]8;;https://example.com\x07"
        close_seq = "\x1b]8;;\x07"
        text = f"{open_seq}some-longer-label-here{close_seq}"

        assert truncate_to_width(text, 15) == f"{open_seq}some-longer-{close_seq}\x1b[0m...\x1b[0m"

    def test_handles_malformed_ansi_escape_prefixes_without_hanging(self):
        text = f"abc\x1bnot-ansi {'🙂' * 1000}"
        truncated = truncate_to_width(text, 20, "…")

        assert visible_width(truncated) <= 20

    def test_clips_wide_ellipsis_safely_and_brackets_with_resets(self):
        assert truncate_to_width("abcdef", 1, "🙂") == ""
        assert truncate_to_width("abcdef", 2, "🙂") == "\x1b[0m🙂\x1b[0m"
        assert visible_width(truncate_to_width("abcdef", 2, "🙂")) <= 2

    def test_returns_original_text_when_it_fits_even_if_ellipsis_too_wide(self):
        assert truncate_to_width("a", 2, "🙂") == "a"
        assert truncate_to_width("界", 2, "🙂") == "界"

    def test_pads_truncated_output_to_requested_width(self):
        truncated = truncate_to_width("🙂界🙂界🙂界", 8, "…", True)
        assert visible_width(truncated) == 8

    def test_adds_trailing_reset_when_truncating_without_ellipsis(self):
        truncated = truncate_to_width(f"\x1b[31m{'hello' * 100}", 10, "")
        assert visible_width(truncated) <= 10
        assert truncated.endswith("\x1b[0m")

    def test_keeps_contiguous_prefix_instead_of_skipping_wide_grapheme(self):
        truncated = truncate_to_width("🙂\t界 \x1b_abc\x07", 7, "…", True)
        assert truncated == "🙂\t\x1b[0m…\x1b[0m "


class TestVisibleWidth:
    def test_counts_tabs_inline_and_skips_ansi_inline(self):
        assert visible_width("\t\x1b[31m界\x1b[0m") == 5

    def test_counts_indic_conjunct_spacing_code_points(self):
        assert visible_width("र्क") == 2
        assert visible_width("नेटवर्क") == 5
        assert visible_width("सर्वाधिकार सुरक्षित। ऑर्डर पर क्लिक करें") == 33
        assert visible_width("র্ক") == 2
        assert visible_width("ર્ક") == 2
        assert visible_width("ର୍କ") == 2
        assert visible_width("ర్క") == 2
        assert visible_width("ര്‍ക") == 2

    def test_keeps_ordinary_combining_marks_zero_width(self):
        assert visible_width("e\u0301") == 1
        assert visible_width("čřžůú") == 5
        assert visible_width("שָׁ") == 1
        assert visible_width("بّ") == 1
        assert visible_width("རྐ") == 1
        assert visible_width("ᜠ᜴") == 1
        assert visible_width("가〮") == 2
        assert visible_width("가〯") == 2

    def test_keeps_cjk_and_japanese_width_accounting_unchanged(self):
        assert visible_width("网络") == 4
        assert visible_width("ネットワーク") == 12
        assert visible_width("が") == 2
        assert visible_width("か\u3099") == 2

    def test_counts_myanmar_marks_that_terminals_allocate_cells_for(self):
        assert visible_width("ကာ") == 2
        assert visible_width("ကေ") == 2
        assert visible_width("က်") == 2
        assert visible_width("ကျ") == 2
        assert visible_width("ကြ") == 2
        assert visible_width("ကဳ") == 2
        assert visible_width("ကဴ") == 2
        assert visible_width("ကဵ") == 2
        assert visible_width("ကး") == 2
        assert visible_width("ကို") == 1
        assert visible_width("က္") == 1

    def test_keeps_thai_and_lao_am_clusters_at_normal_cell_width(self):
        assert visible_width("ำ") == 1
        assert visible_width("ຳ") == 1
        assert visible_width("กำ") == 2
        assert visible_width("ກຳ") == 2

    def test_normalizes_thai_and_lao_am_vowels_only_for_terminal_output(self):
        assert normalize_terminal_output("ำ") == "ํา"
        assert normalize_terminal_output("ຳ") == "ໍາ"
        assert visible_width(normalize_terminal_output("ำabc")) == visible_width("ำabc")
        assert visible_width(normalize_terminal_output("ຳabc")) == visible_width("ຳabc")


class TestTabWidthAccounting:
    def test_keeps_slice_helper_widths_consistent_with_visible_width(self):
        text = "out 192M\t.pi/skill-tests/results-ha"
        text_out, width = slice_with_width(text, 0, 10, True)

        assert text_out == "out 192M"
        assert width == 8
        assert visible_width(text_out) == width

    def test_keeps_overlay_segment_widths_consistent_with_visible_width(self):
        text = "out 192M\t.pi/skill-tests/results-ha"
        before, before_width, _after, _after_width = extract_segments(text, 10, 13, 10, True)

        assert before == "out 192M"
        assert before_width == 8
        assert visible_width(before) == before_width

        tab_before, tab_before_width, _a, _aw = extract_segments(text, 11, 13, 10, True)
        assert tab_before == "out 192M\t"
        assert tab_before_width == 11
        assert visible_width(tab_before) == tab_before_width

    def test_keeps_tabs_inside_terminal_control_sequences_byte_identical(self):
        control_sequences = [
            "\x1b]8;;https://example.test/a\tb\x07",
            "\x1b]0;window\ttitle\x1b\\",
            "\x1b_payload\tdata\x1b\\",
        ]

        for control_sequence in control_sequences:
            assert normalize_terminal_output(f"{control_sequence}label\ttext") == f"{control_sequence}label   text"

    @pytest.mark.asyncio
    async def test_keeps_tab_containing_overlays_on_one_physical_terminal_row(self):
        terminal = FakeTerminal(16, 3)
        tui = TuiMainScreen(terminal)
        tui.add_child(_FullViewportContent())
        tui.show_overlay(_TabStatusOverlay(), OverlayOptions(width=4, row=1, col=4))
        tui.start()

        try:
            await asyncio.sleep(0.03)
            model = MiniTerminalModel(16, 3)
            model.feed("".join(terminal.writes))
            assert model.viewport() == ["base 0          ", "base   X        ", "base 2          "]
            assert "\t" not in "".join(terminal.writes)
        finally:
            tui.stop()


class TestRegionalIndicatorWidthRegression:
    def test_treats_partial_flag_grapheme_as_full_width(self):
        partial_flag = "🇨"
        list_line = "      - 🇨"

        assert visible_width(partial_flag) == 2
        assert visible_width(list_line) == 10

    def test_wraps_intermediate_partial_flag_list_line_before_overflow(self):
        wrapped = wrap_text_with_ansi("      - 🇨", 9)

        assert len(wrapped) == 2
        assert visible_width(wrapped[0] or "") == 7
        assert visible_width(wrapped[1] or "") == 2

    def test_treats_all_regional_indicator_singleton_graphemes_as_width_2(self):
        for cp in range(0x1F1E6, 0x1F1FF + 1):
            regional_indicator = chr(cp)
            assert visible_width(regional_indicator) == 2, f"Expected U+{cp:X} to be width 2"

    def test_keeps_full_flag_pairs_at_width_2(self):
        samples = ["🇯🇵", "🇺🇸", "🇬🇧", "🇨🇳", "🇩🇪", "🇫🇷"]
        for flag in samples:
            assert visible_width(flag) == 2, f"Expected {flag} to be width 2"

    def test_keeps_common_streaming_emoji_intermediates_at_stable_width(self):
        samples = ["👍", "👍🏻", "✅", "⚡", "⚡️", "👨", "👨‍💻", "🏳️‍🌈"]
        for sample in samples:
            assert visible_width(sample) == 2, f"Expected {sample} to be width 2"


class TestWrapTextWithAnsi:
    def test_should_not_apply_underline_before_styled_text(self):
        underline_on = "\x1b[4m"
        underline_off = "\x1b[24m"
        url = "https://example.com/very/long/path/that/will/wrap"
        text = f"read this thread {underline_on}{url}{underline_off}"

        wrapped = wrap_text_with_ansi(text, 40)

        assert wrapped[0] == "read this thread"
        assert wrapped[1].startswith(underline_on)
        assert "https://" in wrapped[1]

    def test_should_not_have_whitespace_before_underline_reset_code(self):
        underline_on = "\x1b[4m"
        underline_off = "\x1b[24m"
        text_with_underlined_trailing_space = f"{underline_on}underlined text here {underline_off}more"

        wrapped = wrap_text_with_ansi(text_with_underlined_trailing_space, 18)

        assert f" {underline_off}" not in wrapped[0]

    def test_should_not_bleed_underline_to_padding(self):
        underline_on = "\x1b[4m"
        underline_off = "\x1b[24m"
        url = "https://example.com/very/long/path/that/will/definitely/wrap"
        text = f"prefix {underline_on}{url}{underline_off} suffix"

        wrapped = wrap_text_with_ansi(text, 30)

        for line in wrapped[1:-1]:
            if underline_on in line:
                assert line.endswith(underline_off)
                assert not line.endswith("\x1b[0m")

    def test_should_preserve_background_color_across_wrapped_lines(self):
        bg_blue = "\x1b[44m"
        reset = "\x1b[0m"
        text = f"{bg_blue}hello world this is blue background text{reset}"

        wrapped = wrap_text_with_ansi(text, 15)

        for line in wrapped:
            assert bg_blue in line

        for line in wrapped[:-1]:
            assert not line.endswith("\x1b[0m")

    def test_should_reset_underline_but_preserve_background(self):
        underline_on = "\x1b[4m"
        underline_off = "\x1b[24m"
        reset = "\x1b[0m"

        text = f"\x1b[41mprefix {underline_on}UNDERLINED_CONTENT_THAT_WRAPS{underline_off} suffix{reset}"

        wrapped = wrap_text_with_ansi(text, 20)

        for line in wrapped:
            has_bg_color = "[41m" in line or ";41m" in line or "[41;" in line
            assert has_bg_color

        for line in wrapped[:-1]:
            if ("[4m" in line or "[4;" in line or ";4m" in line) and underline_off not in line:
                assert line.endswith(underline_off)
                assert not line.endswith("\x1b[0m")

    def test_should_handle_lf_crlf_and_cr_line_endings(self):
        assert wrap_text_with_ansi("first\nsecond\r\nthird\rfourth", 80) == [
            "first",
            "second",
            "third",
            "fourth",
        ]

    def test_should_preserve_ansi_state_across_crlf_and_cr_line_endings(self):
        red = "\x1b[31m"
        reset = "\x1b[0m"

        assert wrap_text_with_ansi(f"{red}first\r\nsecond\rthird{reset}", 80) == [
            f"{red}first",
            f"{red}second",
            f"{red}third{reset}",
        ]

    def test_should_wrap_plain_text_correctly(self):
        text = "hello world this is a test"
        wrapped = wrap_text_with_ansi(text, 10)

        assert len(wrapped) > 1
        for line in wrapped:
            assert visible_width(line) <= 10

    def test_should_break_cjk_runs_at_grapheme_boundaries_after_latin_text(self):
        text = "This is an example 中文汉字测试段落内容中文汉字测试段落内容."
        wrapped = wrap_text_with_ansi(text, 40)

        assert wrapped == ["This is an example 中文汉字测试段落内容", "中文汉字测试段落内容."]
        for line in wrapped:
            assert visible_width(line) <= 40

    def test_should_preserve_color_codes_when_wrapping_cjk_runs(self):
        red = "\x1b[31m"
        reset = "\x1b[0m"
        text = f"{red}This is an example 中文汉字测试段落内容中文汉字测试段落内容.{reset}"
        wrapped = wrap_text_with_ansi(text, 40)

        assert len(wrapped) == 2
        assert wrapped[0] == f"{red}This is an example 中文汉字测试段落内容"
        assert wrapped[1] == f"{red}中文汉字测试段落内容.{reset}"
        for line in wrapped:
            assert visible_width(line) <= 40

    def test_should_ignore_osc133_semantic_markers_in_visible_width(self):
        text = "\x1b]133;A\x07hello\x1b]133;B\x07"
        assert visible_width(text) == 5

    def test_should_ignore_osc_sequences_terminated_with_st(self):
        text = "\x1b]133;A\x1b\\hello\x1b]133;B\x1b\\"
        assert visible_width(text) == 5

    def test_should_treat_isolated_regional_indicators_as_width_2(self):
        assert visible_width("🇨") == 2
        assert visible_width("🇨🇳") == 2

    def test_should_truncate_trailing_whitespace_that_exceeds_width(self):
        two_spaces_wrapped_to_width1 = wrap_text_with_ansi("  ", 1)
        assert visible_width(two_spaces_wrapped_to_width1[0]) <= 1

    def test_should_preserve_color_codes_across_wraps(self):
        red = "\x1b[31m"
        reset = "\x1b[0m"
        text = f"{red}hello world this is red{reset}"

        wrapped = wrap_text_with_ansi(text, 10)

        for line in wrapped[1:]:
            assert line.startswith(red)

        for line in wrapped[:-1]:
            assert not line.endswith("\x1b[0m")


class TestWrapTextWithAnsiOsc8Hyperlinks:
    def test_reemits_osc8_open_at_start_of_continuation_lines(self):
        import re

        url = "https://example.com"
        input_text = f"\x1b]8;;{url}\x1b\\0123456789\x1b]8;;\x1b\\"
        lines = wrap_text_with_ansi(input_text, 6)

        for line in lines:
            stripped = re.sub(r"\x1b\][8]?;;[^\x1b\x07]*\x1b\\", "", line)
            stripped = re.sub(r"\x1b\[[0-9;]*m", "", stripped)
            if stripped.strip():
                assert line.startswith(f"\x1b]8;;{url}\x1b\\") or f"\x1b]8;;{url}\x1b\\" in line

    def test_closes_osc8_before_each_line_break(self):
        url = "https://example.com"
        input_text = f"\x1b]8;;{url}\x1b\\0123456789\x1b]8;;\x1b\\"
        lines = wrap_text_with_ansi(input_text, 6)

        for line in lines[:-1]:
            if f"\x1b]8;;{url}\x1b\\" in line:
                assert line.endswith("\x1b]8;;\x1b\\")

    def test_preserves_bel_terminators_when_wrapping_oauth_style_hyperlinks(self):
        url = f"https://example.com/oauth/{'a' * 32}"
        input_text = f"\x1b]8;;{url}\x07{url}\x1b]8;;\x07"
        lines = wrap_text_with_ansi(input_text, 20)

        assert len(lines) > 1
        for line in lines:
            assert f"\x1b]8;;{url}\x07" in line
            assert f"\x1b]8;;{url}\x1b\\" not in line
        for line in lines[:-1]:
            assert line.endswith("\x1b]8;;\x07")

    def test_does_not_emit_osc8_on_lines_outside_hyperlink(self):
        import re

        url = "https://example.com"
        input_text = f"before \x1b]8;;{url}\x1b\\link\x1b]8;;\x1b\\ after"
        lines = wrap_text_with_ansi(input_text, 80)

        assert len(lines) == 1
        open_count = len(re.findall(r"\x1b\]8;;https:[^\x1b]+\x1b\\", lines[0]))
        close_count = len(re.findall(r"\x1b\]8;;\x1b\\", lines[0]))
        assert open_count == 1
        assert close_count == 1


class TestOverlayCjkBoundaryRegression:
    def test_excludes_wide_grapheme_from_before_when_overlay_starts_inside_it(self):
        before, before_width, after, after_width = extract_segments("abcd让EFGH", 5, 9, 11, True)

        assert before == "abcd"
        assert before_width == 4
        assert visible_width(before) == before_width
        assert after == "H"
        assert after_width == 1

    def test_keeps_ascii_before_segment_behavior_at_same_boundary(self):
        before, before_width, _after, _after_width = extract_segments("abcdG EFGH", 5, 9, 11, True)

        assert before == "abcdG"
        assert before_width == 5
        assert visible_width(before) == before_width


class TestSliceByColumn:
    def test_basic_slicing(self):
        assert slice_by_column("hello world", 0, 5) == "hello"
        assert slice_by_column("hello world", 6, 5) == "world"

    def test_slice_with_wide_chars_strict(self):
        text = "a界b"
        assert slice_by_column(text, 0, 2, True) == "a"
        assert slice_by_column(text, 0, 3, True) == "a界"


class TestTruncateToWidthAdditional:
    def test_handles_zero_width_and_empty_text(self):
        assert truncate_to_width("abcdef", 0) == ""
        assert truncate_to_width("", 3) == ""
        assert truncate_to_width("", 3, pad=True) == "   "


class TestVisibleWidthCache:
    def test_evicts_oldest_non_ascii_cache_entry_when_cache_is_full(self):
        original_cache = dict(utils_module._width_cache)
        try:
            utils_module._width_cache.clear()
            keys = [f"界{i}" for i in range(utils_module.WIDTH_CACHE_SIZE)]
            for key in keys:
                visible_width(key)

            visible_width("🙂new")

            assert len(utils_module._width_cache) == utils_module.WIDTH_CACHE_SIZE
            assert keys[0] not in utils_module._width_cache
            assert "🙂new" in utils_module._width_cache
        finally:
            utils_module._width_cache.clear()
            utils_module._width_cache.update(original_cache)


class TestStripTerminalSequences:
    def test_strips_ansi_osc_and_apc_sequences(self):
        text = "A\x1b[31mB\x1b[0m\x1b]8;;https://example.com\x07C\x1b]8;;\x07\x1b_payload\x1b\\D"

        assert utils_module.strip_terminal_sequences(text) == "ABCD"
        assert utils_module.strip_terminal_sequences("plain") == "plain"


class TestGraphemeCellRange:
    def test_returns_cell_ranges_for_combining_wide_and_emoji_graphemes(self):
        line = "a界e\u0301👨\u200d💻"

        assert get_grapheme_cell_range(line, 0) == (0, 1)
        assert get_grapheme_cell_range(line, 1) == (1, 3)
        assert get_grapheme_cell_range(line, 2) == (1, 3)
        assert get_grapheme_cell_range(line, 3) == (3, 4)
        assert get_grapheme_cell_range(line, 4) == (4, 6)
        assert get_grapheme_cell_range(line, 5) == (4, 6)
        assert get_grapheme_cell_range(line, 6) is None

    def test_ignores_ansi_sequences_when_locating_graphemes(self):
        line = "\x1b[31ma界\x1b[0m"

        assert get_grapheme_cell_range(line, 0) == (0, 1)
        assert get_grapheme_cell_range(line, 1) == (1, 3)
        assert get_grapheme_cell_range(line, 2) == (1, 3)
        assert get_grapheme_cell_range(line, 3) is None


class TestOsc8HyperlinkHelpers:
    def test_parses_and_formats_osc8_hyperlinks(self):
        st_code = "\x1b]8;id=1;https://example.com\x1b\\"
        bel_code = "\x1b]8;;https://example.com\x07"

        assert utils_module._parse_osc8_hyperlink(st_code) == utils_module._ActiveHyperlink(
            "id=1", "https://example.com", "\x1b\\"
        )
        assert utils_module._parse_osc8_hyperlink(bel_code) == utils_module._ActiveHyperlink(
            "", "https://example.com", "\x07"
        )
        assert utils_module._format_osc8_hyperlink(utils_module._parse_osc8_hyperlink(st_code)) == st_code
        assert utils_module._format_osc8_hyperlink(utils_module._parse_osc8_hyperlink(bel_code)) == bel_code
        assert utils_module._format_osc8_close("\x1b\\") == "\x1b]8;;\x1b\\"
        assert utils_module._format_osc8_close("\x07") == "\x1b]8;;\x07"

    def test_distinguishes_close_and_non_osc8_sequences(self):
        assert utils_module._parse_osc8_hyperlink("\x1b]8;;\x07") is None
        assert utils_module._parse_osc8_hyperlink("\x1b]8bad") is Ellipsis
        assert utils_module._parse_osc8_hyperlink("\x1b[31m") is Ellipsis

    def test_finds_the_matching_close_for_the_last_open_link_in_a_prefix(self):
        open_bel = "\x1b]8;;https://example.com\x07"
        close_bel = "\x1b]8;;\x07"

        assert utils_module._get_active_osc8_close("plain text") == ""
        assert utils_module._get_active_osc8_close(f"{open_bel}label") == close_bel
        assert utils_module._get_active_osc8_close(f"{open_bel}label{close_bel}") == ""


class TestGetOsc8LinkAtColumn:
    def test_returns_link_only_inside_hyperlink_boundaries(self):
        url = "https://example.com"
        line = f"before \x1b]8;;{url}\x1b\\link\x1b]8;;\x1b\\ after"

        assert get_osc8_link_at_column(line, 6) is None
        for column in range(7, 11):
            assert get_osc8_link_at_column(line, column) == url
        assert get_osc8_link_at_column(line, 11) is None

    def test_handles_bel_terminated_links_over_wide_graphemes(self):
        url = "https://example.com/emoji"
        line = f"\x1b]8;;{url}\x07🙂\x1b]8;;\x07"

        assert get_osc8_link_at_column(line, 0) == url
        assert get_osc8_link_at_column(line, 1) == url
        assert get_osc8_link_at_column(line, 2) is None

    def test_treats_tabs_inside_hyperlinks_as_three_columns(self):
        url = "https://example.com/tab"
        line = f"\x1b]8;;{url}\x1b\\\tX\x1b]8;;\x1b\\"

        for column in range(4):
            assert get_osc8_link_at_column(line, column) == url
        assert get_osc8_link_at_column(line, 4) is None


class TestNormalizeTerminalOutputAdditional:
    def test_decomposes_thai_and_lao_am_vowels_while_expanding_visible_tabs(self):
        assert normalize_terminal_output("ำ\tຳ") == "ํา   ໍາ"


class TestExtractAnsiCode:
    def test_extracts_supported_csi_osc_and_apc_sequences(self):
        samples = [
            "\x1b[38;5;240m",
            "\x1b[12G",
            "\x1b[2K",
            "\x1b[1H",
            "\x1b[2J",
            "\x1b]8;;https://example.com\x1b\\",
            "\x1b]0;window title\x07",
            "\x1b_payload\x1b\\",
            "\x1b_payload\x07",
        ]

        for sample in samples:
            assert extract_ansi_code(sample + "X", 0) == (sample, len(sample))

        text = "a\x1b[31m"
        assert extract_ansi_code(text, 1) == ("\x1b[31m", 5)

    def test_returns_none_for_non_escape_unsupported_and_incomplete_sequences(self):
        assert extract_ansi_code("plain", 0) is None
        assert extract_ansi_code("\x1bX", 0) is None
        assert extract_ansi_code("\x1b[31", 0) is None
        assert extract_ansi_code("\x1b]8;;https://example.com", 0) is None
        assert extract_ansi_code("\x1b_payload", 0) is None


class TestAnsiCodeTracker:
    def test_tracks_sgr_attributes_colors_and_hyperlinks(self):
        tracker = utils_module._AnsiCodeTracker()
        tracker.process("\x1b[1;2;3;4;5;7;8;9;31;41;38;5;240;48;2;1;2;3m")
        tracker.process("\x1b]8;;https://example.com\x07")

        assert tracker.get_active_codes() == (
            "\x1b[1;2;3;4;5;7;8;9;38;5;240;48;2;1;2;3m\x1b]8;;https://example.com\x07"
        )
        assert tracker.get_line_end_reset() == "\x1b[24m\x1b]8;;\x07"
        assert tracker.has_active_codes() is True

        tracker.process("\x1b[24m")
        assert tracker.get_active_codes() == ("\x1b[1;2;3;5;7;8;9;38;5;240;48;2;1;2;3m\x1b]8;;https://example.com\x07")
        assert tracker.get_line_end_reset() == "\x1b]8;;\x07"

        tracker.process("\x1b[22;23;25;27;28;29;39;49m")
        assert tracker.get_active_codes() == "\x1b]8;;https://example.com\x07"

        tracker.process("\x1b]8;;\x07")
        assert tracker.get_active_codes() == ""
        assert tracker.has_active_codes() is False

    def test_handles_resets_and_ignores_non_sgr_sequences(self):
        tracker = utils_module._AnsiCodeTracker()
        tracker.process("\x1b[1m")
        tracker.process("\x1b[21m")
        assert tracker.get_active_codes() == ""

        tracker.process("\x1b[1;0;31m")
        assert tracker.get_active_codes() == "\x1b[31m"

        tracker.process("\x1b]8;;https://example.com\x07")
        tracker.process("\x1b[0m")
        assert tracker.get_active_codes() == "\x1b]8;;https://example.com\x07"

        tracker.process("\x1b[?25m")
        tracker.process("\x1b[2J")
        assert tracker.get_active_codes() == "\x1b]8;;https://example.com\x07"

        tracker.clear()
        assert tracker.get_active_codes() == ""
        assert tracker.has_active_codes() is False


class TestSplitIntoTokensWithAnsi:
    def test_keeps_pending_ansi_attached_to_visible_tokens(self):
        assert utils_module._split_into_tokens_with_ansi("A\x1b[31m B") == ["A", "\x1b[31m ", "B"]
        assert utils_module._split_into_tokens_with_ansi("界\x1b[31m") == ["界\x1b[31m"]
        assert utils_module._split_into_tokens_with_ansi("\x1b[31m") == ["\x1b[31m"]


class TestBreakLongWord:
    def test_breaks_long_words_without_losing_underline_or_osc8_state(self):
        tracker = utils_module._AnsiCodeTracker()
        tracker.process("\x1b[4m")
        tracker.process("\x1b]8;;https://example.com\x1b\\")

        assert utils_module._break_long_word("abc界def", 4, tracker) == [
            "\x1b[4m\x1b]8;;https://example.com\x1b\\abc\x1b[24m\x1b]8;;\x1b\\",
            "\x1b[4m\x1b]8;;https://example.com\x1b\\界de\x1b[24m\x1b]8;;\x1b\\",
            "\x1b[4m\x1b]8;;https://example.com\x1b\\f",
        ]

    def test_returns_a_single_empty_line_for_empty_input(self):
        assert utils_module._break_long_word("", 4, utils_module._AnsiCodeTracker()) == [""]


class TestApplyBackgroundToLine:
    def test_preserves_styled_and_hyperlinked_content_while_padding(self):
        def bg_fn(text: str) -> str:
            return f"\x1b[44m{text}\x1b[49m"

        styled = "\x1b[31mred\x1b[39m"

        assert apply_background_to_line(styled, 6, bg_fn) == "\x1b[44m\x1b[31mred\x1b[39m   \x1b[49m"
        assert visible_width(apply_background_to_line(styled, 6, bg_fn)) == 6
        assert apply_background_to_line("", 3, bg_fn) == "\x1b[44m   \x1b[49m"

        hyperlink = "\x1b]8;;https://example.com\x1b\\go\x1b]8;;\x1b\\"
        result = apply_background_to_line(hyperlink, 4, bg_fn)
        assert hyperlink in result
        assert result.endswith("  \x1b[49m")
        assert visible_width(result) == 4


class TestTruncateFragmentToWidth:
    def test_truncates_plain_unicode_and_ansi_fragments_by_visible_width(self):
        assert utils_module._truncate_fragment_to_width("abc", 0) == ("", 0)
        assert utils_module._truncate_fragment_to_width("", 3) == ("", 0)
        assert utils_module._truncate_fragment_to_width("abc", 2) == ("ab", 2)
        assert utils_module._truncate_fragment_to_width("🙂界", 3) == ("🙂", 2)
        assert utils_module._truncate_fragment_to_width("\x1b[31mab\t界\x1b[0m", 5) == ("\x1b[31mab\t", 5)
        assert utils_module._truncate_fragment_to_width("\x1b[31mab界\x1b[0m", 3) == ("\x1b[31mab", 2)


class TestSliceWithWidthAdditional:
    def test_handles_zero_length_and_preserves_pending_ansi_prefixes(self):
        assert slice_with_width("hello", 0, 0) == ("", 0)
        assert slice_with_width("\x1b[31mhello\x1b[0m", 1, 3) == ("\x1b[31mell", 3)


class TestExtractSegmentsAdditional:
    def test_replays_active_styling_and_hyperlinks_into_after_segment(self):
        line = "\x1b[31mabcdef\x1b[0m"
        assert extract_segments(line, 3, 3, 3) == ("\x1b[31mabc", 3, "\x1b[31mdef", 3)

        hyperlink_line = "\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\ tail"
        assert extract_segments(hyperlink_line, 2, 2, 3) == (
            "\x1b]8;;https://example.com\x1b\\li",
            2,
            "\x1b]8;;https://example.com\x1b\\nk\x1b]8;;\x1b\\ ",
            3,
        )

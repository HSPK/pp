"""Python port of `packages/tui/test/bug-regression-isimageline-startswith-bug.test.ts`.

Regression for `is_image_line`: it used to test the line with a `startswith`
prefix check against the *active* terminal's image escape prefix. When the
terminal had no image support that prefix was `None`, so a line carrying a
300KB inline-image sequence was classified as ordinary text and put through the
width check, which crashed with "Rendered line exceeds terminal width". The fix
is a substring check for either protocol prefix, anywhere in the line.
"""

from __future__ import annotations

from pi_tui.terminal_image import is_image_line


class TestBugScenarioTerminalWithoutImageSupport:
    def test_old_implementation_would_return_false_causing_crash(self) -> None:
        def old_is_image_line(line: str, image_escape_prefix: str | None) -> bool:
            return image_escape_prefix is not None and line.startswith(image_escape_prefix)

        # When the terminal does not support images, the prefix is None.
        terminal_without_image_support = None

        line_with_image_sequence = "Read image file [image/jpeg]\x1b]1337;File=size=800,600;inline=1:base64data...\x07"

        assert old_is_image_line(line_with_image_sequence, terminal_without_image_support) is False, (
            "Bug: old implementation returns false for line containing image sequence "
            "when terminal has no image support"
        )

    def test_new_implementation_returns_true_correctly(self) -> None:
        line_with_image_sequence = "Read image file [image/jpeg]\x1b]1337;File=size=800,600;inline=1:base64data...\x07"

        assert is_image_line(line_with_image_sequence) is True

    def test_new_implementation_detects_kitty_sequences_in_any_position(self) -> None:
        scenarios = [
            "At start: \x1b_Ga=T,f=100,data...\x1b\\",
            "Prefix \x1b_Ga=T,data...\x1b\\",
            "Suffix text \x1b_Ga=T,data...\x1b\\ suffix",
            "Middle \x1b_Ga=T,data...\x1b\\ more text",
            # Very long line (simulating the 300KB+ crash scenario).
            f"Text before \x1b_Ga=T,f=100{'A' * 300000} text after",
        ]

        for line in scenarios:
            assert is_image_line(line) is True, f"Should detect Kitty sequence in: {line[:50]}..."

    def test_new_implementation_detects_iterm2_sequences_in_any_position(self) -> None:
        scenarios = [
            "At start: \x1b]1337;File=size=100,100:base64...\x07",
            "Prefix \x1b]1337;File=inline=1:data==\x07",
            "Suffix text \x1b]1337;File=inline=1:data==\x07 suffix",
            "Middle \x1b]1337;File=inline=1:data==\x07 more text",
            # Very long line (simulating the 304KB crash scenario).
            f"Text before \x1b]1337;File=size=800,600;inline=1:{'B' * 300000} text after",
        ]

        for line in scenarios:
            assert is_image_line(line) is True, f"Should detect iTerm2 sequence in: {line[:50]}..."


class TestIntegrationToolExecutionScenario:
    def test_detects_image_sequences_in_read_tool_output(self) -> None:
        tool_output_line = "Read image file [image/jpeg]\x1b]1337;File=size=800,600;inline=1:base64image...\x07"

        assert is_image_line(tool_output_line) is True

    def test_detects_kitty_sequences_from_image_component(self) -> None:
        kitty_line = "\x1b_Ga=T,f=100,t=f,d=base64data...\x1b\\\x1b_Gm=i=1;\x1b\\"

        assert is_image_line(kitty_line) is True

    def test_handles_ansi_codes_before_image_sequences(self) -> None:
        lines = [
            "\x1b[31mError\x1b[0m: \x1b]1337;File=inline=1:base64==\x07",
            "\x1b[33mWarning\x1b[0m: \x1b_Ga=T,data...\x1b\\",
            "\x1b[1mBold\x1b[0m \x1b]1337;File=:base64==\x07\x1b[0m",
        ]

        for line in lines:
            assert is_image_line(line) is True, f"Should detect image sequence after ANSI codes: {line[:30]}..."


class TestCrashScenarioSimulation:
    def test_does_not_crash_on_very_long_lines_with_image_sequences(self) -> None:
        base64_char = "A" * 100
        iterm2_sequence = "\x1b]1337;File=size=800,600;inline=1:"

        crash_line = "Output: " + iterm2_sequence + base64_char * 3040 + " end of output"

        assert len(crash_line) > 300000, "Test line should be > 300KB"
        assert is_image_line(crash_line) is True

    def test_handles_lines_exactly_matching_crash_log_dimensions(self) -> None:
        target_width = 58649
        prefix = "Text"
        sequence = "\x1b_Ga=T,f=100"
        suffix = "End"
        padding = "A" * (target_width - len(prefix) - len(sequence) - len(suffix))
        line = f"{prefix}{sequence}{padding}{suffix}"

        assert len(line) == 58649
        assert is_image_line(line) is True


class TestNegativeCasesDoNotFalsePositive:
    def test_does_not_detect_images_in_regular_long_text(self) -> None:
        long_text = "A" * 100000

        assert is_image_line(long_text) is False

    def test_does_not_detect_images_in_lines_with_file_paths(self) -> None:
        file_paths = [
            "/path/to/1337/image.jpg",
            "/usr/local/bin/File_converter",
            "~/Documents/1337File_backup.png",
            "./_G_test_file.txt",
        ]

        for path in file_paths:
            assert is_image_line(path) is False, f"Should not falsely detect image sequence in path: {path}"

"""Python port of `packages/coding-agent/test/path-utils.test.ts`."""

from __future__ import annotations

import os
from pathlib import Path

from pi_coding_agent.tools.path_utils import expand_path, resolve_read_path, resolve_to_cwd
from pi_coding_agent.utils.paths import resolve_path


def test_expand_path_expands_tilde_to_home_directory() -> None:
    assert "~" not in expand_path("~")


def test_expand_path_expands_tilde_slash_path_to_home_directory() -> None:
    assert "~/" not in expand_path("~/Documents/file.txt")


def test_expand_path_keeps_tilde_prefixed_filenames_literal() -> None:
    assert expand_path("~draft.md") == "~draft.md"
    assert expand_path("@~draft.md") == "~draft.md"


def test_expand_path_normalizes_unicode_spaces() -> None:
    assert expand_path("file\u00a0name.txt") == "file name.txt"


def test_resolve_to_cwd_resolves_absolute_paths_as_is(tmp_path: Path) -> None:
    absolute_path = str(tmp_path / "absolute" / "path" / "file.txt")
    assert resolve_to_cwd(absolute_path, str(tmp_path / "some" / "cwd")) == absolute_path


def test_resolve_to_cwd_resolves_relative_paths_against_cwd() -> None:
    assert resolve_to_cwd("relative/file.txt", "/some/cwd") == resolve_path("/some/cwd/relative/file.txt")


def test_resolve_to_cwd_resolves_tilde_prefixed_filenames_against_cwd(tmp_path: Path) -> None:
    cwd = str(tmp_path / "pi-path-utils-cwd")
    assert resolve_to_cwd("~draft.md", cwd) == os.path.join(cwd, "~draft.md")
    assert resolve_to_cwd("@~draft.md", cwd) == os.path.join(cwd, "~draft.md")


def test_resolve_read_path_resolves_existing_file_path(tmp_path: Path) -> None:
    (tmp_path / "test-file.txt").write_text("content")
    assert resolve_read_path("test-file.txt", str(tmp_path)) == str(tmp_path / "test-file.txt")


def test_resolve_read_path_handles_nfc_vs_nfd_normalization(tmp_path: Path) -> None:
    nfd_file_name = "file\u0065\u0301.txt"
    nfc_file_name = "file\u00e9.txt"
    assert nfd_file_name != nfc_file_name
    assert nfd_file_name.encode() != nfc_file_name.encode()

    (tmp_path / nfd_file_name).write_text("content")

    result = resolve_read_path(nfc_file_name, str(tmp_path))
    assert str(tmp_path) in result
    assert result.startswith(str(tmp_path / "file"))
    assert result.endswith(".txt")


def test_resolve_read_path_handles_curly_quotes(tmp_path: Path) -> None:
    curly_quote_name = "Capture d\u2019cran.txt"
    straight_quote_name = "Capture d'cran.txt"
    assert curly_quote_name != straight_quote_name

    (tmp_path / curly_quote_name).write_text("content")

    assert resolve_read_path(straight_quote_name, str(tmp_path)) == str(tmp_path / curly_quote_name)


def test_resolve_read_path_handles_combined_nfc_and_curly_quote(tmp_path: Path) -> None:
    nfc_curly_name = "Capture d\u2019\u00e9cran.txt"
    nfc_straight_name = "Capture d'\u00e9cran.txt"
    assert nfc_curly_name != nfc_straight_name

    (tmp_path / nfc_curly_name).write_text("content")

    assert resolve_read_path(nfc_straight_name, str(tmp_path)) == str(tmp_path / nfc_curly_name)


def test_resolve_read_path_handles_macos_screenshot_narrow_no_break_space(tmp_path: Path) -> None:
    macos_name = "Screenshot 2024-01-01 at 10.00.00\u202fAM.png"
    user_name = "Screenshot 2024-01-01 at 10.00.00 AM.png"

    (tmp_path / macos_name).write_text("content")

    assert resolve_read_path(user_name, str(tmp_path)) == str(tmp_path / macos_name)


def test_resolve_read_path_handles_macos_screenshot_lowercase_am_pm(tmp_path: Path) -> None:
    macos_name = "Screenshot 2024-01-01 at 10.00.00\u202fam.png"
    user_name = "Screenshot 2024-01-01 at 10.00.00 am.png"

    (tmp_path / macos_name).write_text("content")

    assert resolve_read_path(user_name, str(tmp_path)) == str(tmp_path / macos_name)

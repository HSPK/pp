"""Tests for path normalization helpers shared by the built-in tools.

Includes the Python port of `packages/coding-agent/test/paths.test.ts`.
"""

from __future__ import annotations

import ntpath
import os
import re
import sys
from pathlib import Path

import pytest
from pi_coding_agent.utils import paths
from pi_coding_agent.utils.paths import (
    canonicalize_path,
    get_cwd_relative_path,
    is_local_path,
    normalize_path,
    resolve_path,
)


def test_normalize_path_supports_trim_and_unicode_space_normalization():
    value = "\u00a0foo\u00a0bar\u2003baz\u202fqux\u3000"

    normalized = paths.normalize_path(
        value,
        paths.PathInputOptions(trim=True, expand_tilde=False, normalize_unicode_spaces=True),
    )

    assert normalized == "foo bar baz qux"


def test_normalize_path_strips_a_leading_at_prefix_only_when_present():
    options = paths.PathInputOptions(expand_tilde=False, strip_at_prefix=True)

    assert paths.normalize_path("@notes.txt", options) == "notes.txt"
    assert paths.normalize_path("notes.txt", options) == "notes.txt"


def test_normalize_path_expands_tilde_against_a_custom_home_dir():
    options = paths.PathInputOptions(home_dir="/home/tester")

    assert paths.normalize_path("~", options) == "/home/tester"
    assert paths.normalize_path("~/subpath", options) == "/home/tester/subpath"


def test_normalize_path_converts_file_urls_and_preserves_plain_relative_paths():
    options = paths.PathInputOptions(expand_tilde=False)

    assert paths.normalize_path("file:///workspace/a%20b.txt", options) == "/workspace/a b.txt"
    assert paths.normalize_path("docs/readme.md", options) == "docs/readme.md"


def test_resolve_path_keeps_absolute_inputs_absolute():
    resolved = paths.resolve_path("/workspace/project/../README.md", base_dir="/ignored/base")

    assert resolved == "/workspace/README.md"


def test_resolve_path_joins_relative_inputs_against_base_dir():
    resolved = paths.resolve_path("src/../README.md", base_dir="/workspace/project")

    assert resolved == "/workspace/project/README.md"


def test_resolve_path_defaults_to_the_current_working_directory(monkeypatch):
    monkeypatch.setattr(paths.Path, "cwd", classmethod(lambda cls: Path("/workspace/current")))

    assert paths.resolve_path("docs/guide.md") == "/workspace/current/docs/guide.md"


def test_resolve_path_applies_multiple_normalization_options_together():
    options = paths.PathInputOptions(trim=True, strip_at_prefix=True, home_dir="/home/tester")

    resolved = paths.resolve_path("  @~/notes.txt  ", base_dir="/ignored/base", options=options)

    assert resolved == "/home/tester/notes.txt"


def _file_url(path: str) -> str:
    return Path(path).as_uri()


def test_canonicalize_path_returns_real_path_for_regular_file(tmp_path: Path) -> None:
    file = tmp_path / "file.txt"
    file.write_text("hello")
    assert canonicalize_path(str(file)) == os.path.realpath(str(file))


def test_canonicalize_path_resolves_symlinks_to_targets(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_text("hello")
    link.symlink_to(target)
    assert canonicalize_path(str(link)) == os.path.realpath(str(target))


def test_canonicalize_path_resolves_directory_symlinks(tmp_path: Path) -> None:
    target_dir = tmp_path / "target-dir"
    link_dir = tmp_path / "link-dir"
    target_dir.mkdir()
    link_dir.symlink_to(target_dir, target_is_directory=True)
    assert canonicalize_path(str(link_dir)) == os.path.realpath(str(target_dir))


def test_canonicalize_path_falls_back_when_target_missing(tmp_path: Path) -> None:
    nonexistent = str(tmp_path / "no-such-file")
    assert canonicalize_path(nonexistent) == nonexistent


def test_canonicalize_path_falls_back_for_dangling_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    assert canonicalize_path(str(link)) == str(link)


def test_get_cwd_relative_path_keeps_names_starting_with_dots(tmp_path: Path) -> None:
    cwd = str(tmp_path / "pi-paths-cwd")
    assert get_cwd_relative_path(os.path.join(cwd, "..config", "AGENTS.md"), cwd) == os.path.join(
        "..config", "AGENTS.md"
    )


def test_get_cwd_relative_path_rejects_parent_directory_traversals(tmp_path: Path) -> None:
    cwd = str(tmp_path / "pi-paths-cwd")
    assert get_cwd_relative_path(os.path.normpath(os.path.join(cwd, "..", "AGENTS.md")), cwd) is None


def test_resolve_path_expands_only_home_tilde_shortcuts(tmp_path: Path) -> None:
    cwd = str(tmp_path / "pi-paths-cwd")
    home = str(Path.home())
    assert normalize_path("~") == home
    assert normalize_path("~/file.txt") == os.path.join(home, "file.txt")
    assert resolve_path("~draft.md", cwd) == os.path.join(cwd, "~draft.md")
    assert normalize_path("~draft.md") == "~draft.md"


def test_resolve_path_resolves_relative_paths_against_base_directory(tmp_path: Path) -> None:
    cwd = str(tmp_path / "pi-paths-cwd")
    expected = os.path.join(cwd, "subdir", "file.txt")
    assert resolve_path("subdir/file.txt", cwd) == expected
    assert resolve_path("subdir/file.txt", _file_url(cwd)) == expected


def test_resolve_path_accepts_file_urls(tmp_path: Path) -> None:
    file_path = str(tmp_path / "file with spaces.txt")
    assert resolve_path(_file_url(file_path), str(tmp_path / "base")) == file_path


def test_resolve_path_raises_for_invalid_file_urls() -> None:
    with pytest.raises(ValueError):
        resolve_path("file:///%E0%A4%A")


def test_resolve_path_preserves_literal_percent_sequences(tmp_path: Path) -> None:
    for name in ["report%2026.md", "foo%2Fbar", "malformed%A.md"]:
        file_path = str(tmp_path / name)
        assert resolve_path(file_path, str(tmp_path / "base")) == file_path


def test_resolve_path_does_not_treat_windows_file_url_pathname_as_native_path(monkeypatch) -> None:
    """TS: "does not treat Windows file URL pathname strings as native paths"
    (`it.runIf(process.platform === "win32")`).

    On Windows, Node's `pathToFileURL(...).pathname` yields a POSIX-shaped
    string like ``/E:/project/dir/SKILL.md``. `resolvePath` must resolve that
    through Windows path semantics (`path.win32.isAbsolute`/`resolve`), not
    POSIX ones. `resolve_path` delegates the absolute-path check and
    normalization to `os.path`, which is `posixpath` on Linux -- unlike
    `normalize_windows_shell_path` (a pure string transform tested
    unconditionally above), this one genuinely depends on which path module
    `os.path` aliases. It is still fakeable without a real Windows machine:
    `ntpath` is a pure-Python module always importable, and gives the same
    split/join semantics as Node's `path.win32` (verified against `node -e
    "require('node:path').win32.resolve(...)"`, which returns the same
    `\\E:\\project\\dir\\SKILL.md` for this input). The module-local `os` name
    inside `pi_coding_agent.utils.paths` is rebound to a stand-in whose `.path`
    is `ntpath`, which does not touch the process-wide `os` module.
    """
    pathname = "/E:/project/dir/SKILL.md"
    assert re.match(r"^/[A-Za-z]:", pathname)

    class _FakeOs:
        path = ntpath

    monkeypatch.setattr(paths, "os", _FakeOs())
    assert resolve_path(pathname, "E:\\project") == ntpath.normpath(pathname)


def test_normalize_windows_shell_path_converts_git_bash_msys_cygwin_wsl_paths() -> None:
    # This is a pure string transform (not gated on the host OS in TypeScript
    # either -- normalizeWindowsShellPath itself never checks process.platform),
    # so it is exercised directly regardless of the platform running the tests.
    assert paths.normalize_windows_shell_path("/c/Users/example/project") == "C:\\Users\\example\\project"
    assert paths.normalize_windows_shell_path("/cygdrive/d/work") == "D:\\work"
    assert paths.normalize_windows_shell_path("/mnt/e/source") == "E:\\source"
    assert paths.normalize_windows_shell_path("/c") == "C:\\"


def test_normalize_windows_shell_path_leaves_other_path_forms_unchanged() -> None:
    for path in [
        "C:/Users/example",
        "C:\\Users\\example",
        "//server/share/file",
        "/c/Users\\example",
        "relative/file",
        "/tmp/file",
    ]:
        assert paths.normalize_windows_shell_path(path) == path


@pytest.mark.skipif(sys.platform != "win32", reason="only applies on Windows, matching the TS `runIf(win32)` gate")
def test_normalize_windows_shell_path_is_applied_by_normal_path_handling_on_windows() -> None:
    assert normalize_path("/c/Users/example") == "C:\\Users\\example"
    assert resolve_path("/mnt/c/Users/example", "D:\\work") == os.path.normpath("C:/Users/example")


def test_is_local_path_returns_true_for_bare_names() -> None:
    assert is_local_path("my-package") is True


def test_is_local_path_returns_true_for_relative_paths() -> None:
    assert is_local_path("./foo") is True


def test_is_local_path_returns_true_for_file_urls() -> None:
    assert is_local_path("file:///tmp/foo") is True


def test_is_local_path_returns_false_for_npm_protocol() -> None:
    assert is_local_path("npm:package") is False


def test_is_local_path_returns_false_for_git_protocol() -> None:
    assert is_local_path("git://repo") is False


def test_is_local_path_returns_false_for_https_protocol() -> None:
    assert is_local_path("https://example.com") is False

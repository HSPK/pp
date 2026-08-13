"""Tests for `pi_coding_agent.utils.tools_manager`.

No dedicated TS test file exists for `packages/coding-agent/src/utils/tools-manager.ts`.
These cover the pure logic (asset naming, platform mapping, resolution order,
archive extraction, the skip paths) offline: nothing here downloads anything.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

import pytest
from pi_coding_agent.utils import tools_manager
from pi_coding_agent.utils.tools_manager import (
    TOOLS,
    ensure_tool,
    extract_archive,
    find_binary_recursively,
    get_tool_path,
    is_offline_mode_enabled,
)


@pytest.mark.parametrize(
    ("tool", "plat", "architecture", "expected"),
    [
        ("fd", "darwin", "arm64", "fd-v10.0.0-aarch64-apple-darwin.tar.gz"),
        ("fd", "darwin", "x64", "fd-v10.0.0-x86_64-apple-darwin.tar.gz"),
        ("fd", "linux", "arm64", "fd-v10.0.0-aarch64-unknown-linux-gnu.tar.gz"),
        ("fd", "win32", "x64", "fd-v10.0.0-x86_64-pc-windows-msvc.zip"),
        ("fd", "freebsd", "x64", None),
        ("rg", "darwin", "arm64", "ripgrep-10.0.0-aarch64-apple-darwin.tar.gz"),
        # ripgrep ships musl for linux x86_64 but gnu for aarch64.
        ("rg", "linux", "x64", "ripgrep-10.0.0-x86_64-unknown-linux-musl.tar.gz"),
        ("rg", "linux", "arm64", "ripgrep-10.0.0-aarch64-unknown-linux-gnu.tar.gz"),
        ("rg", "win32", "arm64", "ripgrep-10.0.0-aarch64-pc-windows-msvc.zip"),
        ("rg", "freebsd", "x64", None),
    ],
)
def test_asset_names_match_the_published_release_assets(tool, plat, architecture, expected):
    assert TOOLS[tool].get_asset_name("10.0.0", plat, architecture) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("TRUE", True), ("yes", True), ("0", False), ("", False), ("no", False)],
)
def test_offline_mode_flag(monkeypatch, value, expected):
    monkeypatch.setenv("PI_OFFLINE", value)
    assert is_offline_mode_enabled() is expected


def test_offline_mode_is_off_when_unset(monkeypatch):
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    assert is_offline_mode_enabled() is False


def test_get_tool_path_prefers_the_managed_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "linux")
    managed = tmp_path / "rg"
    managed.write_text("#!/bin/sh\n")

    assert get_tool_path("rg", str(tmp_path)) == str(managed)


def test_get_tool_path_falls_back_to_path(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "linux")
    monkeypatch.setattr(tools_manager, "command_exists", lambda name: name == "fdfind")

    # `fd` declares `fdfind` as an alternative name, which Debian uses.
    assert get_tool_path("fd", str(tmp_path)) == "fdfind"


def test_get_tool_path_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "linux")
    monkeypatch.setattr(tools_manager, "command_exists", lambda name: False)

    assert get_tool_path("rg", str(tmp_path)) is None


def test_get_tool_path_rejects_an_unknown_tool(tmp_path):
    assert get_tool_path("nope", str(tmp_path)) is None


def test_find_binary_recursively_locates_a_nested_binary(tmp_path):
    nested = tmp_path / "ripgrep-14.1.0-x86_64" / "inner"
    nested.mkdir(parents=True)
    target = nested / "rg"
    target.write_text("binary")

    assert find_binary_recursively(tmp_path, "rg") == target
    assert find_binary_recursively(tmp_path, "fd") is None


def test_extract_archive_unpacks_a_tar_gz(tmp_path):
    payload = tmp_path / "rg"
    payload.write_text("binary")
    archive = tmp_path / "ripgrep-1.0.0-x86_64.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="ripgrep-1.0.0-x86_64/rg")

    destination = tmp_path / "out"
    destination.mkdir()
    extract_archive(archive, destination, archive.name)

    assert (destination / "ripgrep-1.0.0-x86_64" / "rg").read_text() == "binary"


def test_extract_archive_unpacks_a_zip(tmp_path):
    archive = tmp_path / "ripgrep-1.0.0-x86_64.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("rg.exe", "binary")

    destination = tmp_path / "out"
    destination.mkdir()
    extract_archive(archive, destination, archive.name)

    assert (destination / "rg.exe").read_text() == "binary"


def test_extract_archive_skips_entries_that_escape_the_destination(tmp_path):
    payload = tmp_path / "evil"
    payload.write_text("owned")
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../escaped")

    destination = tmp_path / "out"
    destination.mkdir()
    extract_archive(archive, destination, archive.name)

    assert not (tmp_path / "escaped").exists()


def test_extract_archive_rejects_an_unknown_format(tmp_path):
    archive = tmp_path / "tool.rar"
    archive.write_text("nope")

    with pytest.raises(RuntimeError, match="Unsupported archive format"):
        extract_archive(archive, tmp_path, archive.name)


@pytest.mark.asyncio
async def test_ensure_tool_returns_an_already_installed_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "linux")
    managed = tmp_path / "rg"
    managed.write_text("#!/bin/sh\n")

    async def must_not_download(tool, bin_dir=None):
        raise AssertionError("must not download when the tool is already present")

    monkeypatch.setattr(tools_manager, "download_tool", must_not_download)

    assert await ensure_tool("rg", bin_dir=str(tmp_path)) == str(managed)


@pytest.mark.asyncio
async def test_ensure_tool_skips_the_download_in_offline_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", "1")
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "linux")
    monkeypatch.setattr(tools_manager, "command_exists", lambda name: False)

    seen: list[tools_manager.ToolStatus] = []
    assert await ensure_tool("rg", seen.append, bin_dir=str(tmp_path)) is None
    assert [(s.type, "Offline mode enabled" in s.message) for s in seen] == [("warning", True)]


@pytest.mark.asyncio
async def test_ensure_tool_points_android_users_at_the_termux_package(tmp_path, monkeypatch):
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "android")
    monkeypatch.setattr(tools_manager, "command_exists", lambda name: False)

    seen: list[tools_manager.ToolStatus] = []
    assert await ensure_tool("rg", seen.append, bin_dir=str(tmp_path)) is None
    assert [(s.type, "pkg install ripgrep" in s.message) for s in seen] == [("warning", True)]


@pytest.mark.asyncio
async def test_ensure_tool_returns_none_when_the_download_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "linux")
    monkeypatch.setattr(tools_manager, "command_exists", lambda name: False)

    async def failing_download(tool, bin_dir=None):
        raise RuntimeError("GitHub API error: 503")

    monkeypatch.setattr(tools_manager, "download_tool", failing_download)

    seen: list[tools_manager.ToolStatus] = []
    assert await ensure_tool("rg", seen.append, bin_dir=str(tmp_path)) is None
    # The download attempt is announced first, then its failure.
    assert [s.type for s in seen] == ["info", "warning"]
    assert "Failed to download ripgrep" in seen[-1].message


@pytest.mark.asyncio
async def test_ensure_tool_installs_and_reports_the_path(tmp_path, monkeypatch):
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "linux")
    monkeypatch.setattr(tools_manager, "command_exists", lambda name: False)

    async def fake_download(tool, bin_dir=None):
        return os.path.join(bin_dir or "", tool)

    monkeypatch.setattr(tools_manager, "download_tool", fake_download)

    seen: list[tools_manager.ToolStatus] = []
    assert await ensure_tool("rg", seen.append, bin_dir=str(tmp_path)) == str(Path(tmp_path) / "rg")
    assert [s.type for s in seen] == ["info", "info"]
    assert "installed to" in seen[-1].message


@pytest.mark.asyncio
async def test_ensure_tool_rejects_an_unknown_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_manager, "command_exists", lambda name: False)

    assert await ensure_tool("nope", bin_dir=str(tmp_path)) is None


@pytest.mark.asyncio
async def test_ensure_tool_reports_status_without_writing_to_stdout(tmp_path, monkeypatch, capsys):
    """The TUI calls this after mounting, so a stray print draws over the frame."""
    monkeypatch.setenv("PI_OFFLINE", "1")
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "linux")
    monkeypatch.setattr(tools_manager, "command_exists", lambda name: False)

    seen: list[tools_manager.ToolStatus] = []
    assert await ensure_tool("rg", seen.append, bin_dir=str(tmp_path)) is None

    assert [s.type for s in seen] == ["warning"]
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_ensure_tool_is_silent_when_the_tool_is_already_present(tmp_path, monkeypatch):
    """Nothing to report: no download, no status."""
    monkeypatch.setattr(tools_manager, "current_platform", lambda: "linux")
    (tmp_path / "rg").write_text("#!/bin/sh\n")

    seen: list[tools_manager.ToolStatus] = []
    assert await ensure_tool("rg", seen.append, bin_dir=str(tmp_path)) == str(tmp_path / "rg")
    assert seen == []

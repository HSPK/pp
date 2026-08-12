"""Locate and, if needed, download the managed `fd` and `rg` binaries.

Python port of `packages/coding-agent/src/utils/tools-manager.ts`.

The grep and glob tools are dramatically faster when `rg` and `fd` are
available. Rather than require users to install them, this resolves each tool
in order -- the agent's own `bin/` directory, then `PATH` -- and downloads the
current GitHub release into `bin/` when neither has it.

Extraction uses Python's `tarfile`/`zipfile` instead of shelling out to
`tar`/`unzip`/`powershell` as TypeScript does, so it needs no external
commands and behaves the same on every platform.
"""

from __future__ import annotations

import os
import platform
import random
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from pi_coding_agent.core.config import APP_NAME, get_bin_dir
from pi_coding_agent.utils.management_http import FetchRetryOptions, fetch_with_retry

ManagedTool = Literal["fd", "rg"]

NETWORK_TIMEOUT_MS = 10_000
DOWNLOAD_TIMEOUT_MS = 120_000

# `fd` stopped publishing x86_64 macOS builds after this release.
_FD_DARWIN_X64_PINNED_VERSION = "10.3.0"

TERMUX_PACKAGES: dict[str, str] = {"fd": "fd", "rg": "ripgrep"}
"""Termux package names. Android needs `pkg install`; Linux binaries do not run there."""


def is_offline_mode_enabled() -> bool:
    value = os.environ.get("PI_OFFLINE")
    if not value:
        return False
    return value == "1" or value.lower() in ("true", "yes")


@dataclass(frozen=True)
class ToolConfig:
    """How to find, download and unpack one managed tool."""

    name: str
    repo: str
    """GitHub repository, e.g. `sharkdp/fd`."""
    binary_name: str
    """Name of the binary inside the archive."""
    tag_prefix: str
    """Release tag prefix: `v` for `v1.0.0`, empty for `1.0.0`."""
    get_asset_name: Callable[[str, str, str], str | None]
    system_binary_names: tuple[str, ...] = ()
    """Alternative `PATH` command names to try before downloading."""


def _fd_asset_name(version: str, plat: str, architecture: str) -> str | None:
    arch_str = "aarch64" if architecture == "arm64" else "x86_64"
    if plat == "darwin":
        return f"fd-v{version}-{arch_str}-apple-darwin.tar.gz"
    if plat == "linux":
        return f"fd-v{version}-{arch_str}-unknown-linux-gnu.tar.gz"
    if plat == "win32":
        return f"fd-v{version}-{arch_str}-pc-windows-msvc.zip"
    return None


def _rg_asset_name(version: str, plat: str, architecture: str) -> str | None:
    arch_str = "aarch64" if architecture == "arm64" else "x86_64"
    if plat == "darwin":
        return f"ripgrep-{version}-{arch_str}-apple-darwin.tar.gz"
    if plat == "linux":
        if architecture == "arm64":
            return f"ripgrep-{version}-aarch64-unknown-linux-gnu.tar.gz"
        return f"ripgrep-{version}-x86_64-unknown-linux-musl.tar.gz"
    if plat == "win32":
        return f"ripgrep-{version}-{arch_str}-pc-windows-msvc.zip"
    return None


TOOLS: dict[str, ToolConfig] = {
    "fd": ToolConfig(
        name="fd",
        repo="sharkdp/fd",
        binary_name="fd",
        system_binary_names=("fd", "fdfind"),
        tag_prefix="v",
        get_asset_name=_fd_asset_name,
    ),
    "rg": ToolConfig(
        name="ripgrep",
        repo="BurntSushi/ripgrep",
        binary_name="rg",
        tag_prefix="",
        get_asset_name=_rg_asset_name,
    ),
}


def current_platform() -> str:
    """The platform name in Node's vocabulary (`darwin`, `linux`, `win32`, `android`)."""
    if hasattr(sys, "getandroidapilevel") or "ANDROID_ROOT" in os.environ:
        return "android"
    system = platform.system().lower()
    if system == "windows":
        return "win32"
    if system == "darwin":
        return "darwin"
    return system


def current_arch() -> str:
    """The architecture in Node's vocabulary (`arm64`, `x64`)."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "x64"
    return machine


def command_exists(command: str) -> bool:
    """Whether `command` can actually be run, not merely resolved on `PATH`."""
    if shutil.which(command) is None:
        return False
    try:
        subprocess.run([command, "--version"], capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def get_tool_path(tool: ManagedTool, bin_dir: str | None = None) -> str | None:
    """Resolve `tool` to a managed binary path or a `PATH` command name. `None` if absent."""
    config = TOOLS.get(tool)
    if config is None:
        return None

    suffix = ".exe" if current_platform() == "win32" else ""
    local_path = Path(bin_dir or get_bin_dir()) / (config.binary_name + suffix)
    if local_path.exists():
        return str(local_path)

    for name in config.system_binary_names or (config.binary_name,):
        if command_exists(name):
            return name

    return None


async def get_latest_version(repo: str) -> str:
    """The newest release tag of `repo`, with any leading `v` stripped."""
    response = await fetch_with_retry(
        f"https://api.github.com/repos/{repo}/releases/latest",
        headers={"User-Agent": f"{APP_NAME}-coding-agent"},
        options=FetchRetryOptions(timeout_ms=NETWORK_TIMEOUT_MS),
    )
    if response.status_code >= 400:
        raise RuntimeError(f"GitHub API error: {response.status_code}")
    tag_name = response.json()["tag_name"]
    return tag_name[1:] if tag_name.startswith("v") else tag_name


async def download_file(url: str, dest: Path) -> None:
    """Stream `url` to `dest`."""
    timeout = httpx.Timeout(DOWNLOAD_TIMEOUT_MS / 1000)
    async with (
        httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client,
        client.stream("GET", url) as response,
    ):
        if response.status_code >= 400:
            raise RuntimeError(f"Failed to download: {response.status_code}")
        with dest.open("wb") as file:
            async for chunk in response.aiter_bytes():
                file.write(chunk)


def find_binary_recursively(root_dir: Path, binary_file_name: str) -> Path | None:
    """Locate `binary_file_name` anywhere under `root_dir`.

    Release archives differ: some put the binary at the root, others nest it
    under a versioned directory.
    """
    stack = [root_dir]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_file() and entry.name == binary_file_name:
                return entry
            if entry.is_dir():
                stack.append(entry)
    return None


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def extract_archive(archive_path: Path, extract_dir: Path, asset_name: str) -> None:
    """Unpack a `.tar.gz` or `.zip` release archive, rejecting paths that escape `extract_dir`."""
    try:
        if asset_name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as archive:
                members = [
                    member for member in archive.getmembers() if _is_within(extract_dir, extract_dir / member.name)
                ]
                archive.extractall(extract_dir, members=members, filter="data")
        elif asset_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as archive:
                names = [name for name in archive.namelist() if _is_within(extract_dir, extract_dir / name)]
                archive.extractall(extract_dir, members=names)
        else:
            raise RuntimeError(f"Unsupported archive format: {asset_name}")
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as error:
        raise RuntimeError(f"Failed to extract {asset_name}: {error}") from error


async def download_tool(tool: ManagedTool, bin_dir: str | None = None) -> str:
    """Download, unpack and install `tool`. Returns the installed binary path."""
    config = TOOLS.get(tool)
    if config is None:
        raise RuntimeError(f"Unknown tool: {tool}")

    plat = current_platform()
    architecture = current_arch()

    version = await get_latest_version(config.repo)
    if tool == "fd" and plat == "darwin" and architecture == "x64":
        version = _FD_DARWIN_X64_PINNED_VERSION

    asset_name = config.get_asset_name(version, plat, architecture)
    if asset_name is None:
        raise RuntimeError(f"Unsupported platform: {plat}/{architecture}")

    tools_dir = Path(bin_dir or get_bin_dir())
    tools_dir.mkdir(parents=True, exist_ok=True)

    download_url = f"https://github.com/{config.repo}/releases/download/{config.tag_prefix}{version}/{asset_name}"
    archive_path = tools_dir / asset_name
    binary_ext = ".exe" if plat == "win32" else ""
    binary_path = tools_dir / (config.binary_name + binary_ext)

    await download_file(download_url, archive_path)

    # `fd` and `rg` can download concurrently at startup, so each extraction
    # needs its own directory.
    suffix = f"_{config.binary_name}_{os.getpid()}_{int(time.time() * 1000)}_{random.randbytes(4).hex()}"
    extract_dir = Path(tempfile.mkdtemp(prefix="extract_tmp", suffix=suffix, dir=tools_dir))

    try:
        extract_archive(archive_path, extract_dir, asset_name)

        binary_file_name = config.binary_name + binary_ext
        stem = asset_name.removesuffix(".tar.gz").removesuffix(".zip")
        candidates = [extract_dir / stem / binary_file_name, extract_dir / binary_file_name]
        extracted = next((candidate for candidate in candidates if candidate.exists()), None)
        if extracted is None:
            extracted = find_binary_recursively(extract_dir, binary_file_name)
        if extracted is None:
            raise RuntimeError(f"Binary not found in archive: expected {binary_file_name} under {extract_dir}")

        binary_path.unlink(missing_ok=True)
        shutil.move(str(extracted), str(binary_path))

        if plat != "win32":
            binary_path.chmod(binary_path.stat().st_mode | stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    finally:
        archive_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

    return str(binary_path)


async def ensure_tool(tool: ManagedTool, silent: bool = False, bin_dir: str | None = None) -> str | None:
    """Return a usable path for `tool`, downloading it if necessary.

    Returns `None` when the tool is unavailable and cannot be installed --
    offline mode, Android, an unsupported platform, or a failed download. Never
    raises; callers fall back to their slower path.
    """
    existing = get_tool_path(tool, bin_dir)
    if existing:
        return existing

    config = TOOLS.get(tool)
    if config is None:
        return None

    if is_offline_mode_enabled():
        if not silent:
            print(f"{config.name} not found. Offline mode enabled, skipping download.")
        return None

    if current_platform() == "android":
        package = TERMUX_PACKAGES.get(tool, tool)
        if not silent:
            print(f"{config.name} not found. Install with: pkg install {package}")
        return None

    if not silent:
        print(f"{config.name} not found. Downloading...")

    try:
        path = await download_tool(tool, bin_dir)
    except Exception as error:
        if not silent:
            print(f"Failed to download {config.name}: {error}")
        return None

    if not silent:
        print(f"{config.name} installed to {path}")
    return path


__all__ = [
    "DOWNLOAD_TIMEOUT_MS",
    "NETWORK_TIMEOUT_MS",
    "TERMUX_PACKAGES",
    "TOOLS",
    "ManagedTool",
    "ToolConfig",
    "command_exists",
    "current_arch",
    "current_platform",
    "download_file",
    "download_tool",
    "ensure_tool",
    "extract_archive",
    "find_binary_recursively",
    "get_latest_version",
    "get_tool_path",
    "is_offline_mode_enabled",
]

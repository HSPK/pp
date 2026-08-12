"""Python port of `packages/server/test/unix.test.ts`.

`tests/test_unix_listener_coverage.py` already exists in the port, but it is a
port-only coverage suite written against the Python listener's internals; this
file is the direct translation of the upstream filesystem-lifecycle test. It is
named `test_server_unix` rather than `test_unix` because pytest's prepend
import mode puts every package's `tests/` directory on `sys.path`, so equal
basenames across packages collide.

The upstream "genuinely stale socket" case forks
`test/fixtures/stale-socket-server.mjs` and `SIGKILL`s it. The Python analogue
spawns a child interpreter that binds and listens on the path and kills it the
same way, so the socket inode is left behind with nothing accepting on it.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pi_server.server import PiServer
from pi_server.testing.client import ProtocolTestClient, connect_unix_test_client
from pi_server.testing.service import TestServerService
from pi_server.transports.unix import UnixServerOptions, create_unix_server

DEFAULT_TIMEOUT = 5.0

_STALE_SOCKET_SERVER = (
    "import socket, sys, time\n"
    "server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
    "server.bind(sys.argv[1])\n"
    "server.listen(1)\n"
    "sys.stdout.write('listening\\n')\n"
    "sys.stdout.flush()\n"
    "while True:\n"
    "    time.sleep(1)\n"
)


class UnixHarness:
    """Tracks servers, clients, children and temp dirs, like upstream's `afterEach`."""

    def __init__(self) -> None:
        self._servers: list[PiServer] = []
        self._clients: list[ProtocolTestClient] = []
        self._children: list[subprocess.Popen[str]] = []
        self._directories: list[Path] = []

    def socket_path(self, nested: bool = False) -> str:
        # A Unix socket path is capped at ~107 bytes, so these live in a short
        # private temp directory rather than under pytest's `tmp_path`.
        directory = Path(tempfile.mkdtemp(prefix="ps-"))
        self._directories.append(directory)
        return str(directory / "p" / "n" / "server.sock") if nested else str(directory / "server.sock")

    def make_server(self, path: str) -> PiServer:
        server = create_unix_server(TestServerService(), UnixServerOptions(path=path))
        self._servers.append(server)
        return server

    async def connect(self, path: str) -> ProtocolTestClient:
        client = await asyncio.wait_for(connect_unix_test_client(path), DEFAULT_TIMEOUT)
        self._clients.append(client)
        return client

    def spawn_stale_socket_server(self, path: str) -> subprocess.Popen[str]:
        child = subprocess.Popen(
            [sys.executable, "-c", _STALE_SOCKET_SERVER, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._children.append(child)
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "listening"
        return child

    def kill(self, child: subprocess.Popen[str]) -> None:
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=DEFAULT_TIMEOUT)
        self._children.remove(child)

    async def aclose(self) -> None:
        for child in self._children:
            if child.poll() is None:
                child.send_signal(signal.SIGKILL)
                child.wait(timeout=DEFAULT_TIMEOUT)
        self._children.clear()
        await asyncio.gather(*(client.close() for client in self._clients), return_exceptions=True)
        self._clients.clear()
        await asyncio.gather(*(server.close() for server in self._servers), return_exceptions=True)
        self._servers.clear()
        for directory in self._directories:
            shutil.rmtree(directory, ignore_errors=True)
        self._directories.clear()


@pytest.fixture
async def unix_harness() -> AsyncIterator[UnixHarness]:
    harness = UnixHarness()
    try:
        yield harness
    finally:
        await asyncio.wait_for(harness.aclose(), timeout=10.0)


def _identity(path: str) -> tuple[int, int]:
    info = os.lstat(path)
    return (info.st_dev, info.st_ino)


async def test_rejects_a_live_listener_without_unlinking_it(unix_harness: UnixHarness) -> None:
    path = unix_harness.socket_path()
    first = unix_harness.make_server(path)
    await first.start()
    first_identity = _identity(path)

    second = unix_harness.make_server(path)
    with pytest.raises(RuntimeError, match=r"already running"):
        await second.start()

    assert stat.S_ISSOCK(os.lstat(path).st_mode)
    assert _identity(path) == first_identity

    client = await unix_harness.connect(path)
    hello = await asyncio.wait_for(client.hello(), DEFAULT_TIMEOUT)
    assert hello["type"] == "hello"


async def test_never_unlinks_a_regular_file_at_the_configured_path(unix_harness: UnixHarness) -> None:
    path = unix_harness.socket_path()
    Path(path).write_text("do not remove")
    os.chmod(path, 0o640)

    server = unix_harness.make_server(path)
    with pytest.raises(RuntimeError, match=r"non-socket"):
        await server.start()

    assert Path(path).read_text() == "do not remove"


async def test_creates_nested_parents_restricts_permissions_and_removes_its_own_socket(
    unix_harness: UnixHarness,
) -> None:
    path = unix_harness.socket_path(nested=True)
    server = unix_harness.make_server(path)
    await server.start()

    info = os.lstat(path)
    assert stat.S_ISSOCK(info.st_mode)
    assert info.st_mode & 0o777 == 0o600

    await server.close()
    with pytest.raises(FileNotFoundError):
        os.lstat(path)


async def test_does_not_remove_a_replacement_inode_during_shutdown(unix_harness: UnixHarness) -> None:
    path = unix_harness.socket_path()
    server = unix_harness.make_server(path)
    await server.start()
    os.unlink(path)
    Path(path).write_text("replacement")

    closing = asyncio.ensure_future(server.close())
    assert Path(path).read_text() == "replacement"
    await asyncio.wait_for(closing, DEFAULT_TIMEOUT)
    assert Path(path).read_text() == "replacement"


async def test_removes_a_genuinely_stale_socket_before_binding(unix_harness: UnixHarness) -> None:
    path = unix_harness.socket_path()
    child = unix_harness.spawn_stale_socket_server(path)
    assert stat.S_ISSOCK(os.lstat(path).st_mode)
    unix_harness.kill(child)

    server = unix_harness.make_server(path)
    await server.start()
    assert stat.S_ISSOCK(os.lstat(path).st_mode)

    client = await unix_harness.connect(path)
    hello = await asyncio.wait_for(client.hello(), DEFAULT_TIMEOUT)
    assert hello["type"] == "hello"

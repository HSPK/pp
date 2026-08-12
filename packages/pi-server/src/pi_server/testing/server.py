"""Deterministic unstarted `PiServer` factory for transport conformance tests.

Python port of `packages/server/src/testing/server.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..server import PiServer
from ..types import PiServerOptions, PiServerService
from .service import TestServerService


@dataclass
class TestServer:
    server: PiServer
    service: PiServerService

    __test__ = False  # not a pytest test class despite the name matching TS `TestServer`


def create_test_server(options: PiServerOptions, service: PiServerService | None = None) -> TestServer:
    """Creates an unstarted `PiServer` wired to `service` (a fresh `TestServerService` by default)."""
    resolved_service = service if service is not None else TestServerService()
    return TestServer(server=PiServer(resolved_service, options), service=resolved_service)

"""Naive substring session search.

Python port of `packages/agent/src/harness/session/search.ts`.
`ScanningSessionSearch` scans every session's entries and matches a
case-insensitive substring against each entry's JSON serialization -- there is
no index, so this is only suitable for small session stores.

`getFileSystemResultOrThrow` (unwraps the injected `FileSystem`'s
`Result<T, FileError>`) is omitted: this port has no `FileSystem`/`Result`
abstraction (see `jsonl/types.py`), so callers of this module simply let
`OSError`/`SessionError` propagate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from .jsonl.codec import entry_to_wire
from .session import Session
from .types import EntryQuery, SessionMetadata


@dataclass(kw_only=True)
class SessionSearchOptions:
    text: str
    cwd: str | None = None


@dataclass(kw_only=True)
class SessionSearchHit:
    metadata: SessionMetadata
    entry_id: str
    timestamp: str
    snippet: str | None = None
    score: float | None = None


class SessionSearch(Protocol):
    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]: ...


class ScanningSessionSearchSource(Protocol):
    async def list(self) -> list[SessionMetadata]: ...
    async def open(self, metadata: SessionMetadata) -> Session: ...


class ScanningSessionSearch:
    def __init__(self, source: ScanningSessionSearchSource) -> None:
        self._source = source

    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]:
        normalized_text = options.text.strip().lower()
        if not normalized_text:
            return []
        hits: list[SessionSearchHit] = []
        for metadata in await self._source.list():
            cwd = getattr(metadata, "cwd", None)
            if options.cwd is not None and cwd != options.cwd:
                continue
            session = await self._source.open(metadata)
            for entry in await session.find_entries(EntryQuery(order="oldestFirst")):
                payload = json.dumps(entry_to_wire(entry))
                if normalized_text not in payload.lower():
                    continue
                hits.append(
                    SessionSearchHit(
                        metadata=metadata,
                        entry_id=entry.id,
                        timestamp=_iso_timestamp(entry.timestamp),
                        snippet=payload,
                    )
                )
        return hits


def _iso_timestamp(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def create_scanning_session_search(source: ScanningSessionSearchSource) -> SessionSearch:
    return ScanningSessionSearch(source)


__all__ = [
    "ScanningSessionSearch",
    "SessionSearch",
    "SessionSearchHit",
    "SessionSearchOptions",
    "create_scanning_session_search",
]

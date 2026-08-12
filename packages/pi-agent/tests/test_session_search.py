"""Tests for `pi_agent.harness.session.search`.

Ported behaviour from `packages/agent/src/harness/session/search.ts`, which has
no dedicated TypeScript suite; the assertions below follow that source line by
line: an empty/whitespace query short-circuits, the `cwd` option filters on the
metadata's `cwd` field, and every entry whose JSON serialization contains the
lower-cased query becomes a hit carrying the serialization as the snippet.

One deliberate port difference is asserted here: TypeScript snippets are
`JSON.stringify(entry)` of the in-memory entry, which already is the wire shape.
This port's entries are snake_case dataclasses, so `search.py` serializes
`entry_to_wire(entry)` instead, which produces the same camelCase wire JSON.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pi_agent.harness.session import InMemorySessionRepo, JsonlSessionCreateOptions, JsonlSessionRepo
from pi_agent.harness.session.jsonl.types import JsonlSessionRepoOptions
from pi_agent.harness.session.search import (
    ScanningSessionSearch,
    SessionSearchOptions,
    create_scanning_session_search,
)
from pi_agent.harness.session.types import SessionCreateOptions, SessionMetadata
from session_conformance_helpers import create_assistant_message, create_user_message

TIMEOUT = 5.0


class _ExplodingSource:
    """Source that fails if it is touched: proves the empty-query short circuit."""

    async def list(self):
        raise AssertionError("list() must not be called for an empty query")

    async def open(self, metadata):
        raise AssertionError("open() must not be called for an empty query")


@pytest.fixture
def repo() -> InMemorySessionRepo:
    return InMemorySessionRepo()


async def _seed(repo: InMemorySessionRepo, session_id: str, texts: list[str]) -> list[str]:
    session = await repo.create(SessionCreateOptions(id=session_id))
    return [await session.append_message(create_user_message(text)) for text in texts]


async def test_returns_no_hits_for_an_empty_query():
    search = ScanningSessionSearch(_ExplodingSource())

    assert await asyncio.wait_for(search.search(SessionSearchOptions(text="")), timeout=TIMEOUT) == []


async def test_returns_no_hits_for_a_whitespace_only_query():
    search = ScanningSessionSearch(_ExplodingSource())

    assert await asyncio.wait_for(search.search(SessionSearchOptions(text="   \t\n")), timeout=TIMEOUT) == []


async def test_finds_matching_entries_across_sessions_in_oldest_first_order(repo: InMemorySessionRepo):
    first_ids = await _seed(repo, "session-a", ["needle in the haystack", "unrelated"])
    second_ids = await _seed(repo, "session-b", ["also a needle here"])
    search = ScanningSessionSearch(repo)

    hits = await asyncio.wait_for(search.search(SessionSearchOptions(text="needle")), timeout=TIMEOUT)

    assert [hit.entry_id for hit in hits] == [first_ids[0], second_ids[0]]
    assert [hit.metadata.id for hit in hits] == ["session-a", "session-b"]


async def test_matches_case_insensitively_and_trims_the_query(repo: InMemorySessionRepo):
    ids = await _seed(repo, "session", ["The Quick Brown Fox"])
    search = ScanningSessionSearch(repo)

    hits = await asyncio.wait_for(search.search(SessionSearchOptions(text="  QUICK bROWN  ")), timeout=TIMEOUT)

    assert [hit.entry_id for hit in hits] == ids


async def test_returns_no_hits_when_nothing_matches(repo: InMemorySessionRepo):
    await _seed(repo, "session", ["hello world"])
    search = ScanningSessionSearch(repo)

    assert await asyncio.wait_for(search.search(SessionSearchOptions(text="goodbye")), timeout=TIMEOUT) == []


async def test_snippet_is_the_wire_json_of_the_entry(repo: InMemorySessionRepo):
    await _seed(repo, "session", ["snippet payload"])
    search = ScanningSessionSearch(repo)

    hit = (await asyncio.wait_for(search.search(SessionSearchOptions(text="snippet")), timeout=TIMEOUT))[0]

    payload = json.loads(hit.snippet)
    assert payload["type"] == "message"
    assert payload["message"]["role"] == "user"
    assert payload["message"]["content"][0]["text"] == "snippet payload"
    # Wire keys stay camelCase, so a query can match them the way it would
    # against a session written by the TypeScript CLI.
    assert "parentId" in payload
    assert hit.score is None


async def test_matches_against_wire_field_names_and_non_message_entries(repo: InMemorySessionRepo):
    session = await repo.create(SessionCreateOptions(id="session"))
    custom_id = await session.append_custom_entry("note", {"remember": "this"})
    search = ScanningSessionSearch(repo)

    by_value = await asyncio.wait_for(search.search(SessionSearchOptions(text="remember")), timeout=TIMEOUT)
    by_wire_key = await asyncio.wait_for(search.search(SessionSearchOptions(text="customType")), timeout=TIMEOUT)

    assert [hit.entry_id for hit in by_value] == [custom_id]
    assert [hit.entry_id for hit in by_wire_key] == [custom_id]


async def test_reports_entry_timestamps_as_utc_iso_strings_with_milliseconds(repo: InMemorySessionRepo):
    await _seed(repo, "session", ["timestamped"])
    search = ScanningSessionSearch(repo)

    hit = (await asyncio.wait_for(search.search(SessionSearchOptions(text="timestamped")), timeout=TIMEOUT))[0]

    entry = await (await repo.open(SessionMetadata(id="session", created_at=0))).get_entry(hit.entry_id)
    assert hit.timestamp.endswith("Z")
    # `new Date(ms).toISOString()`: millisecond precision, no offset suffix.
    assert len(hit.timestamp) == len("1970-01-01T00:00:00.000Z")
    assert hit.timestamp.startswith("20")
    assert entry.timestamp > 0


async def test_matches_multiple_entries_within_one_session(repo: InMemorySessionRepo):
    session = await repo.create(SessionCreateOptions(id="session"))
    first = await session.append_message(create_user_message("match one"))
    await session.append_message(create_assistant_message("nothing here"))
    second = await session.append_message(create_user_message("match two"))
    search = ScanningSessionSearch(repo)

    hits = await asyncio.wait_for(search.search(SessionSearchOptions(text="match")), timeout=TIMEOUT)

    assert [hit.entry_id for hit in hits] == [first, second]


async def test_skips_sessions_whose_metadata_has_no_cwd_when_a_cwd_filter_is_given(repo: InMemorySessionRepo):
    # `InMemorySessionRepo` metadata has no `cwd` at all, so TS's
    # `(metadata as { cwd?: unknown }).cwd` is undefined and never equals the
    # requested cwd; the Python `getattr(metadata, "cwd", None)` matches that.
    await _seed(repo, "session", ["needle"])
    search = ScanningSessionSearch(repo)

    hits = await asyncio.wait_for(search.search(SessionSearchOptions(text="needle", cwd="/workspace")), timeout=TIMEOUT)

    assert hits == []


async def test_filters_by_cwd_against_a_jsonl_repo(tmp_path):
    repo = JsonlSessionRepo(JsonlSessionRepoOptions(sessions_root=tmp_path))
    inside = await repo.create(JsonlSessionCreateOptions(id="inside", cwd=str(tmp_path / "project")))
    inside_id = await inside.append_message(create_user_message("needle inside"))
    outside = await repo.create(JsonlSessionCreateOptions(id="outside", cwd=str(tmp_path / "other")))
    await outside.append_message(create_user_message("needle outside"))
    search = create_scanning_session_search(repo)

    filtered = await asyncio.wait_for(
        search.search(SessionSearchOptions(text="needle", cwd=str(tmp_path / "project"))), timeout=TIMEOUT
    )
    unfiltered = await asyncio.wait_for(search.search(SessionSearchOptions(text="needle")), timeout=TIMEOUT)

    assert [hit.entry_id for hit in filtered] == [inside_id]
    assert filtered[0].metadata.cwd == str(tmp_path / "project")
    assert len(unfiltered) == 2


async def test_create_scanning_session_search_returns_a_working_search(repo: InMemorySessionRepo):
    ids = await _seed(repo, "session", ["factory built"])

    search = create_scanning_session_search(repo)
    hits = await asyncio.wait_for(search.search(SessionSearchOptions(text="factory")), timeout=TIMEOUT)

    assert isinstance(search, ScanningSessionSearch)
    assert [hit.entry_id for hit in hits] == ids

"""Python port of `packages/coding-agent/test/session-selector-search.test.ts`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

from pi_coding_agent.core.session_manager import SessionInfo
from pi_coding_agent.modes.interactive.components.session_selector_search import filter_and_sort_sessions

_EPOCH = datetime.fromtimestamp(0, tz=UTC)


def make_session(
    session_id: str,
    modified: str,
    all_messages_text: str,
    name: str | None = None,
) -> SessionInfo:
    return SessionInfo(
        path=f"/tmp/{session_id}.jsonl",
        id=session_id,
        cwd="",
        name=name,
        created=_EPOCH,
        modified=datetime.fromisoformat(modified.replace("Z", "+00:00")),
        message_count=1,
        first_message="(no messages)",
        all_messages_text=all_messages_text,
    )


def _ids(sessions: list[SessionInfo]) -> list[str]:
    return [session.id for session in sessions]


class TestSessionSelectorSearch:
    def test_filters_by_quoted_phrase_with_whitespace_normalization(self) -> None:
        sessions = [
            make_session("a", "2026-01-01T00:00:00.000Z", "node\n\n   cve was discussed"),
            make_session("b", "2026-01-02T00:00:00.000Z", "node something else"),
        ]

        assert _ids(filter_and_sort_sessions(sessions, '"node cve"', "recent")) == ["a"]

    def test_filters_by_regex_and_is_case_insensitive(self) -> None:
        sessions = [
            make_session("a", "2026-01-02T00:00:00.000Z", "Brave is great"),
            make_session("b", "2026-01-03T00:00:00.000Z", "bravery is not the same"),
        ]

        assert _ids(filter_and_sort_sessions(sessions, r"re:\bbrave\b", "recent")) == ["a"]

    def test_recent_sort_preserves_input_order(self) -> None:
        sessions = [
            make_session("newer", "2026-01-03T00:00:00.000Z", "brave"),
            make_session("older", "2026-01-01T00:00:00.000Z", "brave"),
            make_session("nomatch", "2026-01-04T00:00:00.000Z", "something else"),
        ]

        assert _ids(filter_and_sort_sessions(sessions, '"brave"', "recent")) == ["newer", "older"]

    def test_relevance_sort_orders_by_score_and_tie_breaks_by_modified_desc(self) -> None:
        sessions = [
            make_session("late", "2026-01-03T00:00:00.000Z", "xxxx brave"),
            make_session("early", "2026-01-01T00:00:00.000Z", "brave xxxx"),
        ]

        assert _ids(filter_and_sort_sessions(sessions, '"brave"', "relevance")) == ["early", "late"]

        tie_sessions = [
            make_session("newer", "2026-01-03T00:00:00.000Z", "brave"),
            make_session("older", "2026-01-01T00:00:00.000Z", "brave"),
        ]

        assert _ids(filter_and_sort_sessions(tie_sessions, '"brave"', "relevance")) == ["newer", "older"]

    def test_returns_empty_list_for_invalid_regex(self) -> None:
        sessions = [make_session("a", "2026-01-01T00:00:00.000Z", "brave")]

        assert filter_and_sort_sessions(sessions, "re:(", "recent") == []


class TestNameFilter:
    sessions: ClassVar[list[SessionInfo]] = [
        make_session("named1", "2026-01-03T00:00:00.000Z", "blueberry", name="My Project"),
        make_session("named2", "2026-01-02T00:00:00.000Z", "blueberry", name="Another Named"),
        make_session("other1", "2026-01-04T00:00:00.000Z", "blueberry"),
        make_session("other2", "2026-01-01T00:00:00.000Z", "blueberry"),
    ]

    def test_returns_all_sessions_when_name_filter_is_all(self) -> None:
        result = filter_and_sort_sessions(self.sessions, "", "recent", "all")
        assert _ids(result) == ["named1", "named2", "other1", "other2"]

    def test_returns_only_named_sessions_when_name_filter_is_named(self) -> None:
        result = filter_and_sort_sessions(self.sessions, "", "recent", "named")
        assert _ids(result) == ["named1", "named2"]

    def test_applies_name_filter_before_search_query(self) -> None:
        result = filter_and_sort_sessions(self.sessions, "blueberry", "recent", "named")
        assert _ids(result) == ["named1", "named2"]

    def test_excludes_whitespace_only_names_from_named_filter(self) -> None:
        sessions_with_whitespace = [
            make_session("whitespace", "2026-01-01T00:00:00.000Z", "test", name="   "),
            make_session("empty", "2026-01-02T00:00:00.000Z", "test", name=""),
            make_session("named", "2026-01-03T00:00:00.000Z", "test", name="Real Name"),
        ]

        result = filter_and_sort_sessions(sessions_with_whitespace, "", "recent", "named")
        assert _ids(result) == ["named"]

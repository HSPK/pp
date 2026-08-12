"""Python port of `packages/coding-agent/test/suite/regressions/5996-session-name-newlines.test.ts`."""

from __future__ import annotations

from pathlib import Path

from harness import create_harness, drain_session_tasks
from pi_coding_agent.core.extensions.loader import ExtensionAPI


async def test_filters_newlines_when_agent_session_set_session_name_is_called(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.session.set_session_name("hello\nworld\r\nagain")
        await drain_session_tasks(harness.session)

        assert harness.session_manager.get_session_name() == "hello world again"
        assert [event.name for event in harness.events_of_type("session_info_changed")] == ["hello world again"]
    finally:
        harness.cleanup()


async def test_filters_newlines_when_an_extension_calls_pi_set_session_name(tmp_path: Path) -> None:
    captured: list[ExtensionAPI] = []

    def factory(pi: ExtensionAPI) -> None:
        captured.append(pi)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        assert captured
        captured[0].set_session_name("from\nextension")
        await drain_session_tasks(harness.session)

        assert harness.session_manager.get_session_name() == "from extension"
        assert [event.name for event in harness.events_of_type("session_info_changed")] == ["from extension"]
    finally:
        harness.cleanup()

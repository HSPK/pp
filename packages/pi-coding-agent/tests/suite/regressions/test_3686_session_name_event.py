"""Python port of `packages/coding-agent/test/suite/regressions/3686-session-name-event.test.ts`."""

from __future__ import annotations

from pathlib import Path

from harness import create_harness, drain_session_tasks
from pi_coding_agent.core.extensions.loader import ExtensionAPI


async def test_emits_session_info_changed_when_set_session_name_is_called(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.session.set_session_name("hello world")

        assert harness.session_manager.get_session_name() == "hello world"
        assert [event.name for event in harness.events_of_type("session_info_changed")] == ["hello world"]
    finally:
        harness.cleanup()


async def test_emits_session_info_changed_when_an_extension_sets_the_name(tmp_path: Path) -> None:
    api: dict[str, ExtensionAPI] = {}

    def factory(pi: ExtensionAPI) -> None:
        api["pi"] = pi

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        api["pi"].set_session_name("from extension")

        assert harness.session_manager.get_session_name() == "from extension"
        assert [event.name for event in harness.events_of_type("session_info_changed")] == ["from extension"]
    finally:
        harness.cleanup()


async def test_emits_session_info_changed_to_extensions(tmp_path: Path) -> None:
    api: dict[str, ExtensionAPI] = {}
    events: list[str | None] = []

    def factory(pi: ExtensionAPI) -> None:
        api["pi"] = pi

        def on_session_info_changed(event, ctx) -> None:
            events.append(event.name)

        pi.on("session_info_changed", on_session_info_changed)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        api["pi"].set_session_name("first")
        await drain_session_tasks(harness.session)
        harness.session.set_session_name("second")
        await drain_session_tasks(harness.session)

        assert events == ["first", "second"]
    finally:
        harness.cleanup()

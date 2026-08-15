"""Python port of `packages/coding-agent/test/suite/regressions/7193-event-bus-lifecycle.test.ts`.

Extension-owned `pi.events` subscriptions used to outlive the session that
made them: the bus is created by the host and reused, so after the session was
replaced or disposed its extension handlers still ran. These cases pin that a
disposed session's handlers come off the bus while the host's own handler on
the same channel keeps firing.
"""

from __future__ import annotations

from pathlib import Path

from harness import create_harness

from pi_coding_agent.core.event_bus import create_event_bus
from pi_coding_agent.core.extensions.loader import (
    ExtensionAPI,
    ExtensionRuntimeActions,
    NamedInlineExtension,
    load_extension_factories,
)


async def test_removes_extension_owned_event_bus_listeners_on_dispose(
    tmp_path: Path,
) -> None:
    event_bus = create_event_bus()
    counts = {"extension": 0, "host": 0}
    captured: dict[str, ExtensionAPI] = {}

    def factory(pi: ExtensionAPI) -> None:
        captured.setdefault("first", pi)

        def handler(_data: object) -> None:
            counts["extension"] += 1

        pi.events.on("reload:test", handler)

    event_bus.on("reload:test", lambda _data: counts.__setitem__("host", counts["host"] + 1))

    loaded = await load_extension_factories(
        [NamedInlineExtension(name="events", factory=factory)],
        str(tmp_path),
        ExtensionRuntimeActions(event_bus=event_bus),
    )
    assert loaded.errors == []

    harness = await create_harness(tmp_path, extensions=loaded.extensions)
    try:

        def emit() -> dict[str, int]:
            before = dict(counts)
            event_bus.emit("reload:test", None)
            return {
                "extension": counts["extension"] - before["extension"],
                "host": counts["host"] - before["host"],
            }

        assert emit() == {"extension": 1, "host": 1}

        # The TypeScript test calls `harness.session.reload()` twice here and
        # asserts the counts stay at 1 each time (each reload swaps in a freshly
        # loaded extension whose subscription replaces the old one), plus
        #
        #     expect(() => firstApi?.getCommands()).toThrow("stale after session replacement or reload");
        #
        # Session replacement (`reload`/`newSession`/`fork`/`switchSession`) and
        # the `runtime.invalidate()` staleness flag it needs are deliberately
        # not ported -- see `core/agent_session.py`'s `dispose()` docstring --
        # so there is no `session.reload()` to call and no stale `pi` to poke.
        # The dispose half below is the part this port can and must honour.

        harness.session.dispose()
        assert emit() == {"extension": 0, "host": 1}
    finally:
        harness.cleanup()


async def test_unsubscribing_twice_is_a_no_op(tmp_path: Path) -> None:
    """`trackEventBusSubscription`'s `active` latch: the returned unsubscribe is
    idempotent and drops itself from the tracked set, so a later dispose does
    not call the underlying unsubscribe again."""
    event_bus = create_event_bus()
    calls: list[object] = []
    unsubscribes: list[object] = []

    def factory(pi: ExtensionAPI) -> None:
        unsubscribes.append(pi.events.on("channel", lambda data: calls.append(data)))

    loaded = await load_extension_factories(
        [NamedInlineExtension(name="events", factory=factory)],
        str(tmp_path),
        ExtensionRuntimeActions(event_bus=event_bus),
    )
    extension = loaded.extensions[0]
    assert len(extension.event_bus_unsubscribers) == 1

    unsubscribe = unsubscribes[0]
    unsubscribe()
    assert extension.event_bus_unsubscribers == []
    unsubscribe()

    event_bus.emit("channel", "x")
    assert calls == []


async def test_events_emit_reaches_other_subscribers(tmp_path: Path) -> None:
    event_bus = create_event_bus()
    seen: list[object] = []
    event_bus.on("from-extension", seen.append)

    emitters: list[ExtensionAPI] = []

    def factory(pi: ExtensionAPI) -> None:
        emitters.append(pi)

    loaded = await load_extension_factories(
        [NamedInlineExtension(name="emitter", factory=factory)],
        str(tmp_path),
        ExtensionRuntimeActions(event_bus=event_bus),
    )
    assert loaded.errors == []

    emitters[0].events.emit("from-extension", {"n": 1})
    assert seen == [{"n": 1}]


async def test_events_without_a_bus_are_inert(tmp_path: Path) -> None:
    """A host that never created a bus must not crash an extension using
    `pi.events` -- `ExtensionRuntimeActions.event_bus` defaults to `None`."""
    apis: list[ExtensionAPI] = []

    def factory(pi: ExtensionAPI) -> None:
        apis.append(pi)
        pi.events.on("channel", lambda _data: None)
        pi.events.emit("channel", None)

    loaded = await load_extension_factories([NamedInlineExtension(name="inert", factory=factory)], str(tmp_path))

    assert loaded.errors == []
    assert loaded.extensions[0].event_bus_unsubscribers == []

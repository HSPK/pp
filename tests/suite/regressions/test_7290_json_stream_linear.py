"""Python port of `packages/coding-agent/test/suite/regressions/7290-json-stream-linear.test.ts`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness import Harness, create_harness
from pi_ai.providers.faux import faux_assistant_message

from pi_coding_agent.modes.json_event import to_json_event


@pytest.fixture
def harnesses() -> list[Harness]:
    created: list[Harness] = []
    yield created
    while created:
        created.pop().cleanup()


async def _measure_update_bytes(tmp_path: Path, harnesses: list[Harness], text: str) -> int:
    harness = await create_harness(tmp_path)
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message(text)])

    await harness.session.prompt("respond")

    session_updates = harness.events_of_type("message_update")
    for update in session_updates:
        assert hasattr(update, "message")
        assert hasattr(update.assistant_message_event, "partial")

    updates = [to_json_event(event) for event in session_updates]
    assert len(updates) > 0
    for update in updates:
        assert "message" not in update
        assert "partial" not in update["assistantMessageEvent"]
    return sum(len(json.dumps(event).encode("utf-8")) for event in updates)


async def test_emits_delta_only_message_updates_whose_size_scales_linearly(
    tmp_path: Path, harnesses: list[Harness]
) -> None:
    small_bytes = await _measure_update_bytes(tmp_path / "small", harnesses, "x" * 2_000)
    large_bytes = await _measure_update_bytes(tmp_path / "large", harnesses, "x" * 4_000)

    assert large_bytes > small_bytes
    assert large_bytes / small_bytes < 2.2

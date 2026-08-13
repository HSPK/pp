"""Python port of `packages/coding-agent/test/suite/regressions/7911-json-stream-usage.test.ts`.

#7290 introduced a delta-only wire projection that stripped cumulative
assistant snapshots from `message_update`. It took the cumulative `usage` with
them, which is fixed-size metadata a streaming consumer needs: without it there
are no token counts at all until `message_end`.
"""

from __future__ import annotations

from pathlib import Path

from harness import create_harness
from pi_ai.providers.faux import faux_assistant_message
from pi_coding_agent.modes.json_event import to_json_event


async def test_includes_cumulative_usage_without_cumulative_message_snapshots(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path)
    try:
        harness.set_responses([faux_assistant_message("hello")])

        await harness.session.prompt("respond")

        update = next(
            (
                event
                for event in harness.events_of_type("message_update")
                if getattr(event.message, "role", None) == "assistant"
                and getattr(event.message.usage, "total_tokens", 0) > 0
            ),
            None,
        )
        assert update is not None, "Expected an assistant update with populated usage"

        wire_update = to_json_event(update)

        assert wire_update["usage"] is not None
        assert "message" not in wire_update
        assert "partial" not in wire_update["assistantMessageEvent"]
    finally:
        harness.cleanup()

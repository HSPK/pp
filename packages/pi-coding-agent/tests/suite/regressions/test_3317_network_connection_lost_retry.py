"""Python port of `packages/coding-agent/test/suite/regressions/3317-network-connection-lost-retry.test.ts`."""

from __future__ import annotations

from pathlib import Path

from harness import create_harness, get_assistant_texts
from pi_ai.providers.faux import faux_assistant_message


async def test_retries_transient_network_connection_lost_failures(tmp_path: Path) -> None:
    harness = await create_harness(
        tmp_path,
        settings={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}},
    )
    try:
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message="Network connection lost."),
                faux_assistant_message("recovered after reconnect"),
            ]
        )

        await harness.session.prompt("test")

        assert harness.faux.state.call_count == 2
        assert [event.error_message for event in harness.events_of_type("auto_retry_start")] == [
            "Network connection lost."
        ]
        assert [event.success for event in harness.events_of_type("auto_retry_end")] == [True]
        assert "recovered after reconnect" in get_assistant_texts(harness)
    finally:
        harness.cleanup()

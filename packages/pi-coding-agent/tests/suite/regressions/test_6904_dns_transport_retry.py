"""Python port of `packages/coding-agent/test/suite/regressions/6904-dns-transport-retry.test.ts`."""

from __future__ import annotations

from pathlib import Path

from harness import create_harness
from pi_ai.providers.faux import faux_assistant_message

WRAPPED_DNS_LOOKUP_ERROR = (
    "The pending stream has been canceled (caused by: getaddrinfo ENOTFOUND bedrock-runtime.us-east-1.amazonaws.com)"
)


async def test_retries_a_transient_dns_lookup_failure(tmp_path: Path) -> None:
    harness = await create_harness(
        tmp_path,
        settings={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}},
    )
    try:
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message=WRAPPED_DNS_LOOKUP_ERROR),
                faux_assistant_message("recovered after DNS retry"),
            ]
        )

        await harness.session.prompt("test")

        assert harness.faux.state.call_count == 2
        assert [event.error_message for event in harness.events_of_type("auto_retry_start")] == [
            WRAPPED_DNS_LOOKUP_ERROR
        ]
        assert [event.success for event in harness.events_of_type("auto_retry_end")] == [True]
    finally:
        harness.cleanup()

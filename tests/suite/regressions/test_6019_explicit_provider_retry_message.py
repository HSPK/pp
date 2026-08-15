"""Python port of `packages/coding-agent/test/suite/regressions/6019-explicit-provider-retry-message.test.ts`."""

from __future__ import annotations

from pathlib import Path

import pytest
from harness import create_harness
from pi_ai.providers.faux import faux_assistant_message

OPENAI_EXPLICIT_RETRY_MESSAGE = (
    "An error occurred while processing your request. You can retry your request, or contact us through our "
    "help center at help.openai.com if the error persists. Please include the request ID req_******** in your "
    "message."
)
BEDROCK_EXPLICIT_RETRY_MESSAGE = (
    '{"message":"The system encountered an unexpected error during processing. Try your request again."}'
)


@pytest.mark.parametrize(
    ("provider", "error_message"),
    [("openai", OPENAI_EXPLICIT_RETRY_MESSAGE), ("bedrock", BEDROCK_EXPLICIT_RETRY_MESSAGE)],
)
async def test_retries_explicit_provider_retry_guidance(tmp_path: Path, provider: str, error_message: str) -> None:
    harness = await create_harness(
        tmp_path / provider,
        settings={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}},
    )
    try:
        harness.set_responses(
            [
                faux_assistant_message("", stop_reason="error", error_message=error_message),
                faux_assistant_message("recovered"),
            ]
        )

        await harness.session.prompt("test")

        assert harness.faux.state.call_count == 2
        assert [event.error_message for event in harness.events_of_type("auto_retry_start")] == [error_message]
        assert [event.success for event in harness.events_of_type("auto_retry_end")] == [True]
    finally:
        harness.cleanup()

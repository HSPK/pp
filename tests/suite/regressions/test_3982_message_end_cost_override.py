"""Python port of `packages/coding-agent/test/suite/regressions/3982-message-end-cost-override.test.ts`."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from harness import create_harness
from pi_ai.providers.faux import faux_assistant_message

from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import MessageEndEventResult


async def test_allows_extensions_to_replace_finalized_assistant_usage_cost(tmp_path: Path) -> None:
    def factory(pi: ExtensionAPI) -> None:
        def on_message_end(event, ctx) -> MessageEndEventResult | None:
            message = event.message
            if getattr(message, "role", None) != "assistant":
                return None
            usage = dataclasses.replace(
                message.usage,
                cost=dataclasses.replace(message.usage.cost, total=0.123),
            )
            return MessageEndEventResult(message=dataclasses.replace(message, usage=usage))

        pi.on("message_end", on_message_end)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        harness.set_responses([faux_assistant_message("hello")])

        await harness.session.prompt("hi")

        assistant_message = next(
            (message for message in harness.session.messages if getattr(message, "role", None) == "assistant"),
            None,
        )
        assert assistant_message is not None
        assert assistant_message.usage.cost.total == 0.123

        message_end = next(
            (
                event
                for event in harness.events_of_type("message_end")
                if getattr(event.message, "role", None) == "assistant"
            ),
            None,
        )
        assert message_end is not None
        assert message_end.message.usage.cost.total == 0.123
    finally:
        harness.cleanup()

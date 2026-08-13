"""Selecting a model and thinking level.

Port of `examples/sdk/02-custom-model.ts`.

    uv run python packages/pi-coding-agent/examples/sdk/02_custom_model.py
"""

import asyncio
import sys

from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def main() -> None:
    model_runtime = await ModelRuntime.create()

    # Option 1: a specific built-in model, by provider and id.
    opus = model_runtime.get_model("anthropic", "claude-opus-4-5")
    if opus is not None:
        print(f"Found model: {opus.provider}/{opus.id}")

    # Option 2: the same call also resolves custom models from models.json.
    custom = model_runtime.get_model("my-provider", "my-model")
    if custom is not None:
        print(f"Found custom model: {custom.provider}/{custom.id}")

    # Option 3: only the models whose provider has usable credentials.
    available = await model_runtime.get_available()
    print("Available models:", [f"{model.provider}/{model.id}" for model in available])
    if not available:
        print("No configured provider; run `pp` once to log in.")
        return

    result = await create_agent_session(
        CreateAgentSessionOptions(
            model=available[0],
            thinking_level="medium",  # off, minimal, low, medium, high, xhigh, max
            model_runtime=model_runtime,
        )
    )
    session = result.session

    try:

        def on_event(event: object) -> None:
            if getattr(event, "type", None) != "message_update":
                return
            update = event.assistant_message_event
            if update.type == "text_delta":
                sys.stdout.write(update.delta)
                sys.stdout.flush()

        session.subscribe(on_event)
        await session.prompt("Say hello in one sentence.")
        print()
    finally:
        session.dispose()


if __name__ == "__main__":
    asyncio.run(main())

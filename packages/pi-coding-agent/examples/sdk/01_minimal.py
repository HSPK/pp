"""Minimal SDK usage.

Port of `examples/sdk/01-minimal.ts`.

Uses defaults throughout: skills, extensions, tools and context files are
discovered from the cwd and `~/.pi/agent`, and the model comes from settings
or the first available provider.

    uv run python packages/pi-coding-agent/examples/sdk/01_minimal.py
"""

import asyncio
import sys

from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def main() -> None:
    result = await create_agent_session(CreateAgentSessionOptions())
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

        await session.prompt("What files are in the current directory?")
        print()
        for message in session.state.messages:
            print(message)
    finally:
        # Releases the session's background tasks and file handles.
        session.dispose()


if __name__ == "__main__":
    asyncio.run(main())

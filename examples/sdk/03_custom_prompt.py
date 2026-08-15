"""Replacing or extending the system prompt.

Port of `examples/sdk/03-custom-prompt.ts`.

TypeScript passes `systemPromptOverride`/`appendSystemPromptOverride`
callbacks to `DefaultResourceLoader`; this port takes the resolved strings
directly as `ResourceLoaderOptions.system_prompt` / `append_system_prompt`.

    uv run python packages/pi-coding-agent/examples/sdk/03_custom_prompt.py
"""

import asyncio
import sys

from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager


def _stream_text(session: object) -> None:
    def on_event(event: object) -> None:
        if getattr(event, "type", None) != "message_update":
            return
        update = event.assistant_message_event
        if update.type == "text_delta":
            sys.stdout.write(update.delta)
            sys.stdout.flush()

    session.subscribe(on_event)


async def main() -> None:
    cwd = "."
    agent_dir = get_agent_dir()

    # Option 1: replace the prompt entirely.
    #
    # `append_system_prompt=[]` is required, not cosmetic: left unset, the
    # loader still appends APPEND_SYSTEM.md from ~/.pi/agent or <cwd>/.pi, and
    # the "replacement" would silently carry that text along.
    replacing = ResourceLoader(
        ResourceLoaderOptions(
            cwd=cwd,
            agent_dir=agent_dir,
            system_prompt=('You are a helpful assistant that speaks like a pirate.\nAlways end responses with "Arrr!"'),
            append_system_prompt=[],
        )
    )
    replacing.reload()

    result = await create_agent_session(
        CreateAgentSessionOptions(resource_loader=replacing, session_manager=SessionManager.in_memory())
    )
    session = result.session
    try:
        _stream_text(session)
        print("=== Replace prompt ===")
        await session.prompt("What is 2 + 2?")
        print("\n")
    finally:
        session.dispose()

    # Option 2: keep the built-in prompt and append to it.
    appending = ResourceLoader(
        ResourceLoaderOptions(
            cwd=cwd,
            agent_dir=agent_dir,
            append_system_prompt=["Always mention the current date in your first response."],
        )
    )
    appending.reload()

    result = await create_agent_session(
        CreateAgentSessionOptions(resource_loader=appending, session_manager=SessionManager.in_memory())
    )
    session = result.session
    try:
        _stream_text(session)
        print("=== Append to prompt ===")
        await session.prompt("Say hello.")
        print()
    finally:
        session.dispose()


if __name__ == "__main__":
    asyncio.run(main())

"""Python port of `packages/coding-agent/test/suite/regressions/2860-replaced-session-context.test.ts`.

Regression #2860 is about the *extension* command context surviving a session
replacement: `ctx.newSession/fork/switchSession` take a `withSession` callback
that must run against the replacement session, after the runtime has rebound
extensions, and the stale `pi`/`ctx` captured by the old session must start
throwing.

None of that machinery exists here. `ExtensionCommandContext` deliberately
drops `new_session`/`fork`/`switch_session`/`reload` (see its docstring in
`core/extensions/types.py`), `ReplacedSessionContext` is defined for type
parity but never constructed, and `AgentSessionRuntime` has no
`set_rebind_session` (see `core/agent_session_runtime.py`'s "Dropped: the
extension system" note). So the extension-facing half of each case is skipped
with the reason at that exact spot.

The *session-replacement* half is portable and is pinned below: the runtime's
`fork`/`switch_session` really do swap in the branch/session the TypeScript
expects, with the transcript the TypeScript asserts minus the `withSession`
message. That is the part a divergence in `AgentSessionRuntime` would break.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pi_ai.auth.types import Credential
from pi_ai.providers.faux import (
    FauxModelDefinition,
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_provider,
)

from pi_coding_agent.core.agent_session_runtime import create_agent_session_runtime
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.extensions.types import ExtensionCommandContext, ReplacedSessionContext
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager


def get_text(message: Any) -> str:
    """Port of the TypeScript `getText` helper."""
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content if getattr(part, "type", None) == "text")


def transcript(runtime: Any) -> list[str]:
    return [f"{message.role}:{get_text(message)}" for message in runtime.session.messages]


async def create_runtime_for_test(tmp_path: Path, responses: list[str]):
    """Port of the TypeScript `createRuntimeForTest`, minus extensions/rebinding."""
    temp_dir = tmp_path / "agent"
    temp_dir.mkdir(parents=True, exist_ok=True)

    faux = faux_provider(RegisterFauxProviderOptions(models=[FauxModelDefinition(id="faux-1", reasoning=False)]))
    faux.set_responses([faux_assistant_message(response) for response in responses])
    model = faux.get_model()
    assert model is not None

    auth_storage = AuthStorage.in_memory()
    await auth_storage.set(faux.provider.id, Credential(type="api_key", key="faux-key"))
    model_runtime = await ModelRuntime.create(
        agent_dir=str(temp_dir),
        credentials=auth_storage,
        models_path=str(temp_dir / "models.json"),
        providers=[faux.provider],
    )
    await model_runtime.set_runtime_api_key(faux.provider.id, "faux-key")

    settings_manager = SettingsManager.in_memory({})
    resource_loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=str(temp_dir),
            agent_dir=str(temp_dir),
            no_skills=True,
            no_prompt_templates=True,
            no_context_files=True,
        )
    )
    resource_loader.reload()

    async def create_runtime(*, cwd: str, agent_dir: str, session_manager: SessionManager, **_ignored):
        return await create_agent_session(
            CreateAgentSessionOptions(
                cwd=cwd,
                agent_dir=agent_dir,
                model=model,
                settings_manager=settings_manager,
                session_manager=session_manager,
                model_runtime=model_runtime,
                resource_loader=resource_loader,
            )
        )

    runtime = await create_agent_session_runtime(
        create_runtime,
        cwd=str(temp_dir),
        agent_dir=str(temp_dir),
        session_manager=SessionManager.create(str(temp_dir)),
    )
    return runtime, faux


async def test_supports_fork(tmp_path: Path) -> None:
    """`it("supports withSession for fork")`, minus the `withSession` callback."""
    runtime, _faux = await create_runtime_for_test(tmp_path, ["seed reply", "fork reply"])
    try:
        await runtime.session.prompt("seed")
        leaf_id = runtime.session.session_manager.get_leaf_id()
        assert leaf_id is not None

        result = await runtime.fork(leaf_id, "at")
        assert result["cancelled"] is False

        # TypeScript sends "fork callback message" from the `withSession`
        # callback; this port has no such callback, so the equivalent prompt is
        # issued directly against the replacement session.
        await runtime.session.prompt("fork callback message")

        assert transcript(runtime) == [
            "user:seed",
            "assistant:seed reply",
            "user:fork callback message",
            "assistant:fork reply",
        ]
    finally:
        await runtime.dispose()


async def test_supports_switch_session(tmp_path: Path) -> None:
    """`it("supports withSession for switchSession")`, minus the `withSession` callback."""
    runtime, _faux = await create_runtime_for_test(tmp_path, ["root reply", "target reply", "switch reply"])
    try:
        await runtime.session.prompt("root")
        original_session_path = runtime.session.session_file
        assert original_session_path is not None

        new_session_result = await runtime.new_session()
        assert new_session_result["cancelled"] is False

        await runtime.session.prompt("target")
        target_session_path = runtime.session.session_file
        assert target_session_path is not None

        await runtime.switch_session(original_session_path)
        await runtime.switch_session(target_session_path)

        await runtime.session.prompt("switch callback message")

        assert runtime.session.session_file == target_session_path
        assert transcript(runtime) == [
            "user:target",
            "assistant:target reply",
            "user:switch callback message",
            "assistant:switch reply",
        ]
    finally:
        await runtime.dispose()


def test_command_context_does_not_expose_session_replacement() -> None:
    """Pins the documented boundary the three skips below rely on.

    If this port ever grows `ctx.new_session`/`fork`/`switch_session`, this
    test fails and the skipped cases must be written for real.
    """
    fields = set(ExtensionCommandContext.__dataclass_fields__)
    assert "new_session" not in fields
    assert "fork" not in fields
    assert "switch_session" not in fields
    assert "reload" not in fields
    # `ReplacedSessionContext` exists for type parity but is never constructed.
    assert issubclass(ReplacedSessionContext, ExtensionCommandContext)


@pytest.mark.skip(
    reason="`ctx.newSession({ withSession })`, extension rebinding and stale-`pi`/`ctx` invalidation "
    "are not ported: ExtensionCommandContext drops new_session/fork/switch_session and "
    "AgentSessionRuntime has no setRebindSession (see both modules' docstrings). The portable half "
    "of the event ordering (session_shutdown of the outgoing session before session_start of the "
    "replacement) is pinned by tests/test_agent_session_runtime_events.py."
)
def test_rebinds_before_with_session_and_invalidates_stale_pi_and_ctx() -> None:
    """`it("rebinds before withSession, targets the replacement session, and invalidates stale pi/ctx")`.

    Asserts, in order: `events == ["start:1"]` before the command; after
    `prompt("/repro")`, `events == ["start:1", "shutdown:1", "start:2",
    "with:1"]` (so extensions are rebound *before* `withSession` runs); the
    replacement session file is defined and differs from the old one; the
    captured old `ctx.sessionManager.getSessionFile()` throws; the captured old
    `pi.sendUserMessage(...)` throws; and the replacement session's transcript
    is exactly `["user:Hello from the new session!", "assistant:hello reply"]`.
    """

"""Offline end-to-end test of the Pi `AgentSession` harness.

Exercises `pi_evals.pi_harness.create_pi_coding_agent_harness` (the port of
`packages/evals/src/pi-harness.ts`) against `pi_ai`'s scripted `faux`
provider, so a real `AgentSession` runs in the isolated temporary workspace
with no network access and no provider credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_evals import pi_harness
from pi_evals.harness import HarnessContext, JsonValue
from pi_evals.pi_harness import (
    PiCodingAgentHarnessOptions,
    PiCodingAgentModelSelection,
    PiCodingAgentOutputContext,
    PromptStep,
    ReloadStep,
    create_pi_coding_agent_harness,
)
from pi_evals.vitest_evals.artifacts import PI_SESSION_SNAPSHOT_ARTIFACT


@pytest.fixture
def faux_model_runtime(monkeypatch, tmp_path: Path):
    """Point the harness at an authenticated `faux` provider instead of the user's agent dir."""
    faux = faux_provider()
    agent_dir = tmp_path / "runtime-agent"
    agent_dir.mkdir()

    async def create_runtime() -> ModelRuntime:
        runtime = await ModelRuntime.create(agent_dir=str(agent_dir), providers=[faux.provider])
        await runtime.login(faux.provider.id, "faux-key")
        return runtime

    class _RuntimeStub:
        @staticmethod
        async def create() -> ModelRuntime:
            return await create_runtime()

    monkeypatch.setattr(pi_harness, "ModelRuntime", _RuntimeStub)
    return faux


async def test_runs_a_prompt_in_an_isolated_workspace(faux_model_runtime) -> None:
    faux_model_runtime.set_responses([faux_assistant_message("Paris")])
    harness = create_pi_coding_agent_harness(
        PiCodingAgentHarnessOptions(
            no_tools="all",
            model=PiCodingAgentModelSelection(provider=faux_model_runtime.provider.id, id="faux-1"),
        )
    )
    context = HarnessContext()

    result = await harness.run("What is the capital of France?", context)

    assert result.output == "Paris"
    assert result.usage.provider == faux_model_runtime.provider.id
    assert result.usage.model == "faux-1"
    assert (result.usage.total_tokens or 0) > 0
    assert result.timings is not None and result.timings.total_ms >= 0
    assert [event.type for event in result.events] == ["message", "message"]
    assert context.artifacts["runId"]
    snapshot = context.artifacts[PI_SESSION_SNAPSHOT_ARTIFACT]
    assert isinstance(snapshot, str)
    assert json.loads(snapshot.splitlines()[0])


async def test_runs_prompt_and_reload_steps_and_transforms_the_output(faux_model_runtime) -> None:
    faux_model_runtime.set_responses([faux_assistant_message("wrote it"), faux_assistant_message("used it")])
    seen_prompts: list[str] = []

    def transform_system_prompt(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "You are a test agent."

    def output(context: PiCodingAgentOutputContext) -> JsonValue:
        return {
            "response": context.response,
            "activeTools": context.session.get_active_tool_names(),
            "systemPromptStart": context.session.system_prompt.split("\n")[0],
            "workspaceIsEmpty": not list(Path(context.cwd).iterdir()),
        }

    harness = create_pi_coding_agent_harness(
        PiCodingAgentHarnessOptions(
            name="scripted",
            no_tools="all",
            model=PiCodingAgentModelSelection(provider=faux_model_runtime.provider.id, id="faux-1"),
            transform_system_prompt=transform_system_prompt,
            output=output,
        )
    )

    result = await harness.run(
        [PromptStep(content="Write the file."), ReloadStep(), PromptStep(content="Use the file.")],
        HarnessContext(),
    )

    assert harness.name == "scripted"
    assert seen_prompts and seen_prompts[0].strip()
    assert result.output == {
        "response": "used it",
        "activeTools": [],
        "systemPromptStart": "You are a test agent.",
        "workspaceIsEmpty": True,
    }


async def test_rejects_an_empty_transformed_system_prompt(faux_model_runtime) -> None:
    harness = create_pi_coding_agent_harness(
        PiCodingAgentHarnessOptions(
            no_tools="all",
            model=PiCodingAgentModelSelection(provider=faux_model_runtime.provider.id, id="faux-1"),
            transform_system_prompt=lambda _prompt: "   ",
        )
    )

    with pytest.raises(ValueError, match="must not be empty"):
        await harness.run("anything", HarnessContext())


async def test_reports_a_missing_model(faux_model_runtime) -> None:
    harness = create_pi_coding_agent_harness(
        PiCodingAgentHarnessOptions(model=PiCodingAgentModelSelection(provider="faux", id="absent"))
    )

    with pytest.raises(RuntimeError, match="Eval model not found: faux/absent"):
        await harness.run("anything", HarnessContext())

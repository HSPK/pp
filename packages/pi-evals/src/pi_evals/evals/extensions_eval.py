"""The Pi extension-authoring comparative eval.

Python port of `packages/evals/src/extensions.eval.ts`.

Two harnesses run the same three-step task -- write a Pi extension, reload,
use it -- against two system prompts: one with the guidelines and Pi
documentation sections removed (baseline) and the full default prompt
(candidate). A deterministic judge scores whether the model produced a
loadable extension registering a `hello` tool, called it, and answered with
exactly its greeting.

**Differences from the TypeScript.**

- Extensions here are Python files (`.pi/extensions/hello.py` exporting
  `pi_extension`), not TypeScript modules, so the judge checks for a
  `pi_coding_agent` import instead of `@earendil-works/pi-coding-agent`. The
  two legacy-import checks (`@mariozechner/*` and `@sinclair/typebox`) have
  no Python counterpart and are dropped.
- `session.resourceLoader.getExtensions()` has no counterpart: this port's
  `ResourceLoader` does not load extensions and `create_agent_session` never
  attaches them (a documented boundary of `pi_coding_agent.core.sdk`). The
  output transform therefore loads the extensions itself from the isolated
  workspace with `discover_and_load_extensions`, which reports the same
  errors and tool registrations the TypeScript read off the session. For the
  same reason the "successful `hello` tool call" judge check can never pass
  in this port: nothing activates an extension the model just wrote, so the
  candidate cannot call it. That check is kept, and its failure is a true
  observation about this port's capability, not a scoring bug.
- The two harnesses differ exactly as upstream intends: the default prompt
  carries both the Guidelines section and the "Pi documentation" paragraph,
  while the baseline transform truncates at `Guidelines:` and so strips both.
  `system_prompt_has_pi_docs` therefore tracks `default-system-prompt`. (This
  comparison was inert while the port shipped no `docs/` tree and omitted the
  paragraph; the docs are ported now, so it measures what it was built to
  measure.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pi_coding_agent.core.extensions.loader import discover_and_load_extensions

from pi_evals.harness import (
    EvalCase,
    EvalOptions,
    JudgeContext,
    JudgeResult,
    TestArtifactAttachment,
    create_judge,
    describe_eval,
)
from pi_evals.pi_harness import (
    PiCodingAgentHarnessOptions,
    PiCodingAgentOutputContext,
    PromptStep,
    ReloadStep,
    create_pi_coding_agent_harness,
)
from pi_evals.vitest_evals.artifacts import record_eval_source_artifact
from pi_evals.vitest_evals.harness_table import eval_harness_table

_IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", re.MULTILINE)


@dataclass
class ExtensionAuthoringOutput:
    """Port of `ExtensionAuthoringOutput`."""

    response: str
    system_prompt_has_guidelines: bool
    system_prompt_has_pi_docs: bool
    extension_errors: list[dict[str, str]] = field(default_factory=list)
    loaded_extensions: list[dict[str, object]] = field(default_factory=list)
    extension_source: str | None = None


async def _extension_authoring_output(context: PiCodingAgentOutputContext) -> ExtensionAuthoringOutput:
    extensions = await discover_and_load_extensions([], context.cwd, agent_dir=context.agent_dir)
    extension_path = Path(context.cwd) / ".pi" / "extensions" / "hello.py"
    extension_source = extension_path.read_text(encoding="utf-8") if extension_path.exists() else None
    return ExtensionAuthoringOutput(
        response=context.response,
        system_prompt_has_guidelines="\nGuidelines:\n" in context.session.system_prompt,
        system_prompt_has_pi_docs="\nPi documentation (read only" in context.session.system_prompt,
        extension_errors=list(extensions.errors),
        loaded_extensions=[
            {"path": extension.path, "tools": list(extension.tools.keys())} for extension in extensions.extensions
        ],
        extension_source=extension_source,
    )


def create_extension_authoring_harness(name: str, transform_system_prompt=None):
    """Port of `createExtensionAuthoringHarness`."""
    return create_pi_coding_agent_harness(
        PiCodingAgentHarnessOptions(
            name=name,
            transform_system_prompt=transform_system_prompt,
            output=_extension_authoring_output,
        )
    )


def exclude_guidelines_and_documentation(default_prompt: str) -> str:
    """Port of `excludeGuidelinesAndDocumentation`."""
    guidelines_start = default_prompt.find("\nGuidelines:\n")
    if guidelines_start == -1:
        raise ValueError("Default Pi system prompt has no Guidelines section.")
    return default_prompt[:guidelines_start]


def prepare_default_prompt_override(default_prompt: str) -> str:
    """Port of `prepareDefaultPromptOverride`."""
    cwd_start = default_prompt.rfind("\nCurrent working directory: ")
    if cwd_start == -1:
        raise ValueError("Default Pi system prompt has no working-directory section.")
    return default_prompt[:cwd_start]


def _judge_extension_authoring(context: JudgeContext) -> JudgeResult:
    output = context.output
    assert isinstance(output, ExtensionAuthoringOutput)
    failures: list[str] = []
    if output.extension_source is None:
        failures.append("generated extension source is unavailable")
    else:
        imports = _IMPORT_PATTERN.findall(output.extension_source)
        if not any(module.split(".")[0] == "pi_coding_agent" for module in imports):
            failures.append("extension does not import the canonical pi_coding_agent package")
    if output.extension_errors:
        failures.append("extension loader reported errors")
    if not any("hello" in extension["tools"] for extension in output.loaded_extensions):
        failures.append('no loaded extension registered the "hello" tool')
    if not any(
        call.name == "hello"
        and call.status == "ok"
        and call.arguments.get("name") == "Bob"
        and call.result == "Hello, Bob!"
        for call in context.tool_calls
    ):
        failures.append('no successful hello({ name: "Bob" }) call returned "Hello, Bob!"')
    if output.response != "Hello, Bob!":
        failures.append('final response was not exactly "Hello, Bob!"')

    return JudgeResult(
        score=1 if not failures else 0,
        metadata={
            "rationale": "Extension authoring workflow completed." if not failures else "; ".join(failures),
        },
    )


ExtensionAuthoringJudge = create_judge("ExtensionAuthoringJudge", _judge_extension_authoring)

extension_harness_table = eval_harness_table(
    "Pi extension authoring system prompt",
    baseline=create_extension_authoring_harness("system-prompt-without-docs", exclude_guidelines_and_documentation),
    candidate=create_extension_authoring_harness("default-system-prompt", prepare_default_prompt_override),
)


def _define_row(row) -> None:
    def define(it) -> None:
        async def creates_reloads_and_uses_a_hello_extension(case: EvalCase) -> None:
            result = await case.run(
                [
                    PromptStep(
                        content=(
                            "Create a Pi extension with a hello tool that takes a name and returns a greeting. "
                            "For example, passing Bob should return `Hello, Bob!`."
                        )
                    ),
                    ReloadStep(),
                    PromptStep(
                        content=(
                            "Use the hello tool to greet Bob. "
                            "Respond with exactly the tool's greeting and nothing else."
                        )
                    ),
                ]
            )
            if result.output.extension_source is not None:
                run_id = result.artifacts.get("runId")
                if not isinstance(run_id, str):
                    raise RuntimeError("Pi eval run did not record a run ID.")
                record_eval_source_artifact(
                    case.task,
                    run_id,
                    TestArtifactAttachment(
                        name="hello.py",
                        content_type="text/x-python",
                        body=result.output.extension_source,
                    ),
                )
            expects_full_prompt = row.name == "default-system-prompt"
            assert result.output.system_prompt_has_guidelines is expects_full_prompt
            # The baseline transform truncates at `Guidelines:`, which also
            # removes the documentation paragraph that follows it.
            assert result.output.system_prompt_has_pi_docs is expects_full_prompt

        it("creates, reloads, and uses a hello extension", creates_reloads_and_uses_a_hello_extension)

    describe_eval(
        "Pi extension authoring system prompt",
        EvalOptions(harness=row.harness, judges=[ExtensionAuthoringJudge], judge_threshold=None),
        define,
        suffix=f"{row.name} repetition {row.repetition}",
        namespace=globals(),
    )


for _row in extension_harness_table:
    _define_row(_row)

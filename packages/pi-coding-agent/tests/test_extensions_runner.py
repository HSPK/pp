"""Tests for pi_coding_agent.core.extensions.runner.

Ported from packages/coding-agent/test/extensions-runner.test.ts, narrowed to
this port's actual `ExtensionRunner` surface (see runner.py's module
docstring for what was dropped: shortcuts, flags, message/entry/markdown
renderers, provider registration -- none of those exist here). Extension
fixtures are written into `tmp_path` as real Python files and loaded through
the real loader (`discover_and_load_extensions`); nothing is ever loaded from
outside `tmp_path`.

Individually unported cases from the TypeScript file, with the reason:

- The eight `shortcut conflicts` cases: this port has no keybinding/shortcut
  registry (`getShortcuts`/`buildBuiltinKeybindings`), so there is nothing to
  detect a conflict against.
- The three `flags` cases (`collects flags`, `keeps first flag`,
  `can set flag values`): `pi.registerFlag` and `ExtensionRuntime.flagValues`
  are not part of this port's extension API.
- The three `message and entry renderers` cases: message/entry/markdown
  renderer registries belong to the extension UI host, which is out of scope.
- The three `provider registration` cases: no `ModelRegistry` provider
  registration exists here.
- `command context > passes fork options through to the bound handler`:
  `ExtensionCommandContext.fork` is deliberately dropped (see its docstring in
  `types.py` -- it needs the session-replacement machinery this port omits).
"""

import os

import pytest
from pi_ai.types import Model, TextContent
from pi_ai.utils.abort import AbortController
from pi_coding_agent.core.extensions.loader import discover_and_load_extensions
from pi_coding_agent.core.extensions.runner import (
    ExtensionContextActions,
    ExtensionRunner,
    emit_project_trust_event,
    wrap_registered_tools,
)
from pi_coding_agent.core.extensions.types import (
    BeforeAgentStartEventResult,
    ContextEvent,
    Extension,
    ExtensionUIContext,
    InputEventResult,
    NullExtensionUIContext,
    ProjectTrustContext,
    ProjectTrustEvent,
    ProjectTrustEventResult,
    ToolCallEvent,
    ToolResultEvent,
    ToolResultEventResult,
    UserBashEvent,
)
from pi_coding_agent.core.model_resolver import ScopedModel
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.system_prompt import BuildSystemPromptOptions

_SCOPED_MODEL = Model(
    id="scoped-test",
    name="Scoped Test",
    api="openai-completions",
    provider="test",
    base_url="https://fake.example.com",
    context_window=1000,
    max_tokens=100,
)


def _write(path, content: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def _tool_extension(name: str, description: str = "Test tool") -> str:
    return f"""
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ToolDefinition


async def _execute(tool_call_id, params, signal, on_update, ctx):
    from pi_agent.types import AgentToolResult

    return AgentToolResult(content=[TextContent(text="ok")], details={{}})


def pi_extension(pi):
    pi.register_tool(
        ToolDefinition(name="{name}", label="{name}", description="{description}", execute=_execute)
    )
"""


def _command_extension(name: str, description: str = "Test command") -> str:
    return f"""
def pi_extension(pi):
    async def _handler(args, ctx):
        return None

    pi.register_command("{name}", handler=_handler, description="{description}")
"""


async def _load(tmp_path, *files: tuple[str, str]) -> list[Extension]:
    ext_dir = tmp_path / ".pi" / "extensions"
    for filename, content in files:
        _write(str(ext_dir / filename), content)
    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path / "agent-empty"))
    assert result.errors == []
    return result.extensions


# ---------------------------------------------------------------------------
# Tool collection
# ---------------------------------------------------------------------------


async def test_collects_tools_from_multiple_extensions(tmp_path):
    extensions = await _load(
        tmp_path, ("tool-a.py", _tool_extension("tool_a")), ("tool-b.py", _tool_extension("tool_b"))
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    tools = runner.get_all_registered_tools()

    assert sorted(t.definition.name for t in tools) == ["tool_a", "tool_b"]


async def test_keeps_first_tool_when_two_extensions_register_the_same_name(tmp_path):
    extensions = await _load(
        tmp_path,
        ("a-first.py", _tool_extension("shared", description="first")),
        ("b-second.py", _tool_extension("shared", description="second")),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    tools = runner.get_all_registered_tools()

    assert len(tools) == 1
    assert tools[0].definition.description == "first"


async def test_reports_a_duplicate_tool_as_a_conflict_not_a_load_error(tmp_path):
    """A name collision is advisory; both extensions still load successfully.

    Upstream keeps the two channels apart: `discoverAndLoadExtensions` reports
    only load failures, while `resource-loader.ts`'s `detectConflicts()`
    returns its own list. Folding conflicts into `errors` made a duplicate tool
    name look like a failed extension load.
    """
    ext_dir = tmp_path / ".pi" / "extensions"
    _write(str(ext_dir / "a-first.py"), _tool_extension("shared", description="first"))
    _write(str(ext_dir / "b-second.py"), _tool_extension("shared", description="second"))

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path / "agent-empty"))

    assert result.errors == []
    assert len(result.extensions) == 2
    assert [conflict["error"] for conflict in result.conflicts] == [
        f'Tool "shared" conflicts with {ext_dir / "a-first.py"}'
    ]


# ---------------------------------------------------------------------------
# Command collection
# ---------------------------------------------------------------------------


async def test_collects_commands_from_multiple_extensions(tmp_path):
    extensions = await _load(
        tmp_path, ("cmd-a.py", _command_extension("cmd-a")), ("cmd-b.py", _command_extension("cmd-b"))
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    commands = runner.get_registered_commands()

    assert len(commands) == 2
    # TS asserts the declared `name`s and the `invocationName`s separately;
    # this port returns `(invocation_name, command)` pairs rather than
    # attaching `invocationName` to the command, so both live in one list.
    assert sorted(name for name, _ in commands) == ["cmd-a", "cmd-b"]
    assert sorted(command.name for _, command in commands) == ["cmd-a", "cmd-b"]


async def test_gets_command_by_invocation_name(tmp_path):
    extensions = await _load(tmp_path, ("cmd.py", _command_extension("my-cmd", description="My command")))
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    command = runner.get_command("my-cmd")
    assert command is not None
    assert command.name == "my-cmd"
    assert command.description == "My command"
    # TS also asserts `cmd.invocationName === "my-cmd"`. `RegisteredCommand`
    # carries no invocation name here, so the pair list is the equivalent.
    assert [name for name, _ in runner.get_registered_commands()] == ["my-cmd"]

    assert runner.get_command("not-exists") is None


async def test_suffixes_duplicate_extension_commands_in_insertion_order(tmp_path):
    extensions = await _load(
        tmp_path,
        ("cmd-a.py", _command_extension("shared-cmd", description="First command")),
        ("cmd-b.py", _command_extension("shared-cmd", description="Second command")),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    resolved = runner.get_registered_commands()

    assert len(resolved) == 2
    assert [name for name, _ in resolved] == ["shared-cmd:1", "shared-cmd:2"]
    # Suffixing must rename only the invocation, never the declared name.
    assert [command.name for _, command in resolved] == ["shared-cmd", "shared-cmd"]
    assert [cmd.description for _, cmd in resolved] == ["First command", "Second command"]
    assert runner.get_command("shared-cmd:1").description == "First command"
    assert runner.get_command("shared-cmd:2").description == "Second command"
    # TS additionally asserts `runner.getCommandDiagnostics()` is `[]`. That
    # field exists in TypeScript but is reset by `getRegisteredCommands()` and
    # never populated anywhere, so the assertion is vacuous there; this port
    # has no `get_command_diagnostics` rather than a dead one to assert on.


# ---------------------------------------------------------------------------
# Context creation
# ---------------------------------------------------------------------------


def test_create_context_reflects_bound_actions():
    runner = ExtensionRunner([], cwd="/does/not/matter")
    runner.bind_core(ExtensionContextActions(is_project_trusted=lambda: False, get_model=lambda: "sentinel-model"))

    ctx = runner.create_context()

    assert ctx.mode == "print"
    assert ctx.has_ui is False
    assert ctx.is_project_trusted() is False
    assert ctx.model == "sentinel-model"


# ---------------------------------------------------------------------------
# Error handling / isolation
# ---------------------------------------------------------------------------


async def test_calls_error_listeners_when_handler_raises_and_does_not_crash(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "throws.py",
            """
def pi_extension(pi):
    async def _on_context(event, ctx):
        raise RuntimeError("Handler error!")

    pi.on("context", _on_context)
""",
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))
    errors = []
    runner.on_error(errors.append)

    result = await runner.emit(ContextEvent(messages=[]))

    assert result is None
    assert len(errors) == 1
    assert "Handler error!" in errors[0].error
    assert errors[0].event == "context"


async def test_hook_ordering_across_multiple_extensions():
    """Handlers run in extension-list order, then per-extension registration order."""
    order: list[str] = []

    def _make(tag):
        def _handler(event, ctx):
            order.append(tag)

        return _handler

    ext1 = Extension(path="ext1", resolved_path="ext1")
    ext1.handlers["context"] = [_make("ext1-a"), _make("ext1-b")]
    ext2 = Extension(path="ext2", resolved_path="ext2")
    ext2.handlers["context"] = [_make("ext2-a")]

    runner = ExtensionRunner([ext1, ext2], cwd="/tmp-not-used")
    await runner.emit(ContextEvent(messages=[]))

    assert order == ["ext1-a", "ext1-b", "ext2-a"]


async def test_a_raising_handler_does_not_block_later_handlers():
    order: list[str] = []

    def _raising(event, ctx):
        order.append("raising")
        raise RuntimeError("boom")

    def _after(event, ctx):
        order.append("after")

    def _ext2(event, ctx):
        order.append("ext2")

    ext1 = Extension(path="ext1", resolved_path="ext1")
    ext1.handlers["context"] = [_raising, _after]
    ext2 = Extension(path="ext2", resolved_path="ext2")
    ext2.handlers["context"] = [_ext2]

    runner = ExtensionRunner([ext1, ext2], cwd="/tmp-not-used")
    errors = []
    runner.on_error(errors.append)

    await runner.emit(ContextEvent(messages=[]))

    assert order == ["raising", "after", "ext2"]
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# tool_call / tool_result
# ---------------------------------------------------------------------------


async def test_emit_tool_call_is_not_error_isolated():
    """Port of the documented exception in runner.py's module docstring: a
    `tool_call` handler exception propagates out of `emit_tool_call` instead
    of being caught, unlike every other event type."""
    ext = Extension(path="ext", resolved_path="ext")

    async def _raises(event, ctx):
        raise RuntimeError("tool_call handler exploded")

    ext.handlers["tool_call"] = [_raises]
    runner = ExtensionRunner([ext], cwd="/tmp-not-used")

    try:
        await runner.emit_tool_call(ToolCallEvent(tool_call_id="1", tool_name="t", input={}))
    except RuntimeError as err:
        assert "tool_call handler exploded" in str(err)
    else:
        raise AssertionError("expected emit_tool_call to propagate the handler's exception")


async def test_emit_tool_result_chains_content_modifications_across_handlers():
    async def _handler1(event, ctx):
        return ToolResultEventResult(content=[*event.content, TextContent(text="ext1")])

    async def _handler2(event, ctx):
        return ToolResultEventResult(content=[*event.content, TextContent(text="ext2")])

    ext1 = Extension(path="ext1", resolved_path="ext1")
    ext1.handlers["tool_result"] = [_handler1]
    ext2 = Extension(path="ext2", resolved_path="ext2")
    ext2.handlers["tool_result"] = [_handler2]

    runner = ExtensionRunner([ext1, ext2], cwd="/tmp-not-used")
    chained = await runner.emit_tool_result(
        ToolResultEvent(
            tool_call_id="call-1",
            tool_name="my_tool",
            input={},
            content=[TextContent(text="base")],
            is_error=False,
            details={"initial": True},
        )
    )

    assert chained is not None
    assert chained.content[0] == TextContent(text="base")
    assert len(chained.content) == 3
    appended = sorted(item.text for item in chained.content[1:])
    assert appended == ["ext1", "ext2"]


async def test_emit_tool_result_preserves_previous_modifications_with_partial_patches():
    ext1 = Extension(path="ext1", resolved_path="ext1")
    ext1.handlers["tool_result"] = [
        lambda event, ctx: ToolResultEventResult(content=[TextContent(text="first")], details={"source": "ext1"})
    ]
    ext2 = Extension(path="ext2", resolved_path="ext2")
    ext2.handlers["tool_result"] = [lambda event, ctx: ToolResultEventResult(is_error=True)]

    runner = ExtensionRunner([ext1, ext2], cwd="/tmp-not-used")
    chained = await runner.emit_tool_result(
        ToolResultEvent(
            tool_call_id="call-2",
            tool_name="my_tool",
            input={},
            content=[TextContent(text="base")],
            is_error=False,
            details={"initial": True},
        )
    )

    assert chained is not None
    assert chained.content == [TextContent(text="first")]
    assert chained.details == {"source": "ext1"}
    assert chained.is_error is True


# ---------------------------------------------------------------------------
# before_agent_start / input
# ---------------------------------------------------------------------------


async def test_emit_before_agent_start_chains_system_prompt_and_collects_messages():
    ext1 = Extension(path="ext1", resolved_path="ext1")
    ext1.handlers["before_agent_start"] = [
        lambda event, ctx: BeforeAgentStartEventResult(system_prompt=event.system_prompt + " [ext1]")
    ]
    ext2 = Extension(path="ext2", resolved_path="ext2")
    ext2.handlers["before_agent_start"] = [
        lambda event, ctx: BeforeAgentStartEventResult(system_prompt=event.system_prompt + " [ext2]", message="hi")
    ]

    runner = ExtensionRunner([ext1, ext2], cwd="/tmp-not-used")
    result = await runner.emit_before_agent_start(
        "prompt text", None, "base prompt", BuildSystemPromptOptions(cwd="/tmp-not-used")
    )

    assert result is not None
    messages, system_prompt_override = result
    assert messages == ["hi"]
    assert system_prompt_override == "base prompt [ext1] [ext2]"


async def test_keeps_ctx_get_system_prompt_in_sync_with_chained_system_prompt_updates():
    """Port of "keeps ctx.getSystemPrompt() in sync with chained system prompt updates"."""
    ext1 = Extension(path="ext1", resolved_path="ext1")
    ext1.handlers["before_agent_start"] = [
        lambda event, ctx: BeforeAgentStartEventResult(system_prompt=ctx.get_system_prompt() + "\nfirst")
    ]
    ext2 = Extension(path="ext2", resolved_path="ext2")
    ext2.handlers["before_agent_start"] = [
        lambda event, ctx: BeforeAgentStartEventResult(system_prompt=ctx.get_system_prompt() + "\nsecond")
    ]

    runner = ExtensionRunner([ext1, ext2], cwd="/tmp-not-used")
    errors: list[str] = []
    runner.on_error(lambda error: errors.append(error.error))

    result = await runner.emit_before_agent_start("hello", None, "base", BuildSystemPromptOptions(cwd="/tmp-not-used"))

    assert errors == []
    assert result is not None
    messages, system_prompt_override = result
    # TS asserts `{ messages: undefined, systemPrompt: "base\nfirst\nsecond" }`;
    # this port returns an empty list where TS leaves `messages` undefined.
    assert messages == []
    assert system_prompt_override == "base\nfirst\nsecond"


async def test_emit_input_transform_chaining_and_handled_short_circuit():
    ext1 = Extension(path="ext1", resolved_path="ext1")
    ext1.handlers["input"] = [lambda event, ctx: InputEventResult(action="transform", text=event.text + " [ext1]")]
    ext2 = Extension(path="ext2", resolved_path="ext2")
    ext2.handlers["input"] = [lambda event, ctx: InputEventResult(action="handled")]

    runner = ExtensionRunner([ext1, ext2], cwd="/tmp-not-used")
    result = await runner.emit_input("hello", None, "interactive")

    assert result.action == "handled"


async def test_emit_input_transform_without_handled():
    ext1 = Extension(path="ext1", resolved_path="ext1")
    ext1.handlers["input"] = [lambda event, ctx: InputEventResult(action="transform", text=event.text + " [ext1]")]

    runner = ExtensionRunner([ext1], cwd="/tmp-not-used")
    result = await runner.emit_input("hello", None, "interactive")

    assert result.action == "transform"
    assert result.text == "hello [ext1]"


async def test_has_handlers():
    ext = Extension(path="ext", resolved_path="ext")
    ext.handlers["agent_start"] = [lambda event, ctx: None]
    runner = ExtensionRunner([ext], cwd="/tmp-not-used")

    assert runner.has_handlers("agent_start") is True
    assert runner.has_handlers("agent_end") is False


# ---------------------------------------------------------------------------
# Tool wrapping
# ---------------------------------------------------------------------------


async def test_wrap_registered_tools_produces_executable_agent_tools(tmp_path):
    extensions = await _load(tmp_path, ("with-tool.py", _tool_extension("wrapped_tool")))
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    registered = runner.get_all_registered_tools()
    agent_tools = wrap_registered_tools(registered, runner)

    assert len(agent_tools) == 1
    tool = agent_tools[0]
    assert tool.name == "wrapped_tool"

    result = await tool.execute("call-id", {}, None, None)
    assert result.content == [TextContent(text="ok")]


async def test_get_tool_definition_returns_none_for_unknown_tool():
    runner = ExtensionRunner([], cwd="/tmp-not-used")
    assert runner.get_tool_definition("does-not-exist") is None


# -------------------------------------------------------------------------
# project_trust
# -------------------------------------------------------------------------


async def test_project_trust_returns_no_result_when_every_handler_is_undecided(tmp_path):
    undecided = """
from pi_coding_agent.core.extensions.types import ProjectTrustEventResult


def pi_extension(pi):
    pi.on("project_trust", lambda event, ctx: ProjectTrustEventResult(trusted="undecided"))
"""
    extensions = await _load(tmp_path, ("only_undecided.py", undecided))

    emitted = await emit_project_trust_event(
        extensions,
        ProjectTrustEvent(cwd=str(tmp_path)),
        ProjectTrustContext(cwd=str(tmp_path), mode="tui", has_ui=False, ui=NullExtensionUIContext()),
    )

    assert emitted.result is None
    assert emitted.errors == []


async def test_project_trust_records_a_raising_handler_and_keeps_scanning(tmp_path):
    throwing = """
def _boom(event, ctx):
    raise RuntimeError("trust handler boom")


def pi_extension(pi):
    pi.on("project_trust", _boom)
"""
    decided = """
from pi_coding_agent.core.extensions.types import ProjectTrustEventResult


def pi_extension(pi):
    pi.on("project_trust", lambda event, ctx: ProjectTrustEventResult(trusted="yes", remember=False))
"""
    extensions = await _load(tmp_path, ("a_throwing.py", throwing), ("b_decided.py", decided))

    emitted = await emit_project_trust_event(
        extensions,
        ProjectTrustEvent(cwd=str(tmp_path)),
        ProjectTrustContext(cwd=str(tmp_path), mode="tui", has_ui=False, ui=NullExtensionUIContext()),
    )

    assert emitted.result == ProjectTrustEventResult(trusted="yes", remember=False)
    assert len(emitted.errors) == 1
    assert "trust handler boom" in emitted.errors[0].error
    assert emitted.errors[0].event == "project_trust"


# ---------------------------------------------------------------------------
# Deliberately-omitted surfaces
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Shortcut registration and conflict detection against KeybindingsManager are not "
        "ported: runner.py's docstring lists 'no keybinding/shortcut resolution "
        "(getShortcuts/buildBuiltinKeybindings)' and there is no pi.register_shortcut(). "
        "Covers 8 TypeScript cases under 'shortcut conflicts'."
    )
)
def test_shortcut_conflict_detection() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "CLI flags (pi.register_flag/getFlag/setFlag) are not ported: runner.py's docstring "
        "lists 'no CLI flags'. Covers 3 TypeScript cases under 'flags'."
    )
)
def test_extension_registered_flags() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "Message/markdown/entry renderer registries are not ported: runner.py's docstring "
        "lists 'no message/markdown/entry renderer registries' (per-tool TUI renderers are "
        "on the top-level README's 'Not ported, by decision' list). Covers 3 TypeScript "
        "cases under 'message and entry renderers'."
    )
)
def test_extension_registered_renderers() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "Extension provider registration is not ported: runner.py's docstring lists 'no "
        "provider registration/ModelRegistry', and core/model_runtime.py records that "
        "ModelRegistry and extension-supplied providers are omitted. Covers 3 TypeScript "
        "cases under 'provider registration'."
    )
)
def test_extension_provider_registration() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "ExtensionCommandContext.fork is not ported: types.py's ExtensionCommandContext "
        "docstring records that new_session/fork/switch_session/reload need the "
        "multi-session-file/reload machinery agent_session.py documents as out of scope. "
        "Covers 1 TypeScript case, 'passes fork options through to the bound handler'."
    )
)
def test_command_context_fork_options() -> None:
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# scoped_models
# ---------------------------------------------------------------------------


async def test_scoped_models_reflects_the_get_scoped_models_context_action(tmp_path):
    extensions = await _load(tmp_path)
    runner = ExtensionRunner(extensions, cwd=str(tmp_path), session_manager=SessionManager.in_memory(str(tmp_path)))

    # Before bind_core the default is an empty tuple (never None).
    assert runner.create_context().scoped_models == ()

    # TypeScript casts a partial literal (`as unknown as ScopedModel[]`); a
    # real `ScopedModel` is offline and free here, so use one rather than a
    # dict that would satisfy the assertion more easily than production does.
    scoped = (ScopedModel(model=_SCOPED_MODEL, thinking_level="high"),)
    runner.bind_core(ExtensionContextActions(get_scoped_models=lambda: scoped))
    context_scoped = runner.create_context().scoped_models
    assert context_scoped == scoped
    assert context_scoped[0].model.id == "scoped-test"
    assert context_scoped[0].thinking_level == "high"
    # TS asserts `toBe(scoped)` -- identity, because `ctx.scopedModels` is a
    # live getter returning the caller's array. `ExtensionContext` here is a
    # snapshot dataclass whose `scoped_models` is a `tuple[...]` rebuilt by
    # `create_context()` (see its docstring), so identity cannot hold and
    # equality of the same elements is the strongest available form.


# ---------------------------------------------------------------------------
# project_trust
# ---------------------------------------------------------------------------


async def test_project_trust_continues_past_undecided_and_returns_first_decision(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "a-undecided.py",
            "from pi_coding_agent.core.extensions.types import ProjectTrustEventResult\n\n\n"
            "def pi_extension(pi):\n"
            '    pi.on("project_trust", lambda event, ctx: '
            'ProjectTrustEventResult(trusted="undecided", remember=True))\n',
        ),
        (
            "b-decided.py",
            "from pi_coding_agent.core.extensions.types import ProjectTrustEventResult\n\n\n"
            "def pi_extension(pi):\n"
            '    pi.on("project_trust", lambda event, ctx: '
            'ProjectTrustEventResult(trusted="no", remember=True))\n',
        ),
    )

    result = await emit_project_trust_event(
        extensions,
        ProjectTrustEvent(cwd=str(tmp_path)),
        ProjectTrustContext(cwd=str(tmp_path), mode="tui", has_ui=False, ui=NullExtensionUIContext()),
    )

    assert result.result == ProjectTrustEventResult(trusted="no", remember=True)
    assert result.errors == []


# ---------------------------------------------------------------------------
# Context creation
# ---------------------------------------------------------------------------


async def test_context_exposes_the_current_abort_signal(tmp_path):
    extensions = await _load(tmp_path)
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))
    controller = AbortController()

    runner.bind_core(ExtensionContextActions(get_signal=lambda: controller.signal))

    ctx = runner.create_context()
    assert ctx.signal is controller.signal
    assert ctx.signal is not None
    assert ctx.signal.aborted is False

    controller.abort()
    assert ctx.signal.aborted is True


async def test_context_exposes_print_mode_and_has_ui_false_by_default(tmp_path):
    extensions = await _load(tmp_path)
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))
    runner.bind_core(ExtensionContextActions())

    ctx = runner.create_context()
    assert ctx.mode == "print"
    assert ctx.has_ui is False


async def test_context_exposes_project_trust_state(tmp_path):
    extensions = await _load(tmp_path)
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))
    runner.bind_core(ExtensionContextActions(is_project_trusted=lambda: False))

    ctx = runner.create_context()
    assert ctx.is_project_trusted() is False


async def test_context_exposes_the_session_manager_the_runner_was_built_with(tmp_path):
    # TypeScript's `ExtensionRunner` takes a real `SessionManager` as a required
    # constructor argument and every test in `extensions-runner.test.ts` passes
    # `SessionManager.inMemory()`. The Python port defaults it to `None`, so
    # the threading into `ExtensionContext` is untested unless a real one is
    # supplied; drive the real object here rather than a fake.
    session_manager = SessionManager.in_memory(str(tmp_path))
    extensions = await _load(tmp_path)
    runner = ExtensionRunner(extensions, cwd=str(tmp_path), session_manager=session_manager)
    runner.bind_core(ExtensionContextActions())

    assert runner.create_context().session_manager is session_manager
    assert ExtensionRunner(extensions, cwd=str(tmp_path)).create_context().session_manager is None


async def test_context_exposes_rpc_mode_with_has_ui_true(tmp_path):
    class _FakeUI(ExtensionUIContext):
        pass

    extensions = await _load(tmp_path)
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))
    runner.bind_core(ExtensionContextActions())
    runner.set_ui_context(_FakeUI(), "rpc")

    ctx = runner.create_context()
    assert ctx.mode == "rpc"
    assert ctx.has_ui is True


async def test_context_exposes_tui_mode_with_has_ui_true(tmp_path):
    class _FakeUI(ExtensionUIContext):
        pass

    extensions = await _load(tmp_path)
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))
    runner.bind_core(ExtensionContextActions())
    runner.set_ui_context(_FakeUI(), "tui")

    ctx = runner.create_context()
    assert ctx.mode == "tui"
    assert ctx.has_ui is True


# ---------------------------------------------------------------------------
# user_bash
# ---------------------------------------------------------------------------


async def test_user_bash_returns_the_first_handler_result(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "a-passes.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            "        return None\n\n"
            '    pi.on("user_bash", _handler)\n',
        ),
        (
            "b-intercepts.py",
            "from pi_coding_agent.core.extensions.types import UserBashEventResult\n\n"
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        return UserBashEventResult(result={"command": event.command})\n\n'
            '    pi.on("user_bash", _handler)\n',
        ),
        (
            "c-never-runs.py",
            "from pi_coding_agent.core.extensions.types import UserBashEventResult\n\n"
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        return UserBashEventResult(result={"command": "wrong"})\n\n'
            '    pi.on("user_bash", _handler)\n',
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    result = await runner.emit_user_bash(UserBashEvent(command="ls", exclude_from_context=False, cwd=str(tmp_path)))

    assert result is not None
    assert result.result == {"command": "ls"}


async def test_user_bash_returns_none_when_no_handler_intercepts(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "observer.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            "        return None\n\n"
            '    pi.on("user_bash", _handler)\n',
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    assert (
        await runner.emit_user_bash(UserBashEvent(command="ls", exclude_from_context=True, cwd=str(tmp_path))) is None
    )


async def test_user_bash_isolates_a_throwing_handler(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "a-throwing.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        raise RuntimeError("bash handler boom")\n\n'
            '    pi.on("user_bash", _handler)\n',
        ),
        (
            "b-good.py",
            "from pi_coding_agent.core.extensions.types import UserBashEventResult\n\n"
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        return UserBashEventResult(result="ok")\n\n'
            '    pi.on("user_bash", _handler)\n',
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))
    errors = []
    runner.on_error(errors.append)

    result = await runner.emit_user_bash(UserBashEvent(command="ls", exclude_from_context=False, cwd=str(tmp_path)))

    assert result is not None
    assert result.result == "ok"
    assert len(errors) == 1
    assert errors[0].event == "user_bash"
    assert "bash handler boom" in errors[0].error


# ---------------------------------------------------------------------------
# before_provider_request
# ---------------------------------------------------------------------------


async def test_before_provider_request_chains_handler_replacements(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "a-first.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        return {**event.payload, "seen_by": ["a"]}\n\n'
            '    pi.on("before_provider_request", _handler)\n',
        ),
        (
            "b-second.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        return {**event.payload, "seen_by": [*event.payload["seen_by"], "b"]}\n\n'
            '    pi.on("before_provider_request", _handler)\n',
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    assert runner.has_handlers("before_provider_request") is True

    payload = await runner.emit_before_provider_request({"model": "m"})

    assert payload == {"model": "m", "seen_by": ["a", "b"]}


async def test_before_provider_request_keeps_the_payload_when_a_handler_returns_nothing(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "observer.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            "        return None\n\n"
            '    pi.on("before_provider_request", _handler)\n',
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    payload = {"model": "m"}

    assert await runner.emit_before_provider_request(payload) is payload


async def test_before_provider_request_isolates_a_throwing_handler(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "a-throwing.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        raise RuntimeError("payload handler boom")\n\n'
            '    pi.on("before_provider_request", _handler)\n',
        ),
        (
            "b-good.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        return {**event.payload, "patched": True}\n\n'
            '    pi.on("before_provider_request", _handler)\n',
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))
    errors = []
    runner.on_error(errors.append)

    payload = await runner.emit_before_provider_request({"model": "m"})

    assert payload == {"model": "m", "patched": True}
    assert len(errors) == 1
    assert errors[0].event == "before_provider_request"
    assert "payload handler boom" in errors[0].error


# ---------------------------------------------------------------------------
# before_provider_headers
# ---------------------------------------------------------------------------


async def test_before_provider_headers_handler_mutates_headers_in_place(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "headers.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        event.headers["X-Turn-Index"] = "3"\n\n'
            '    pi.on("before_provider_headers", _handler)\n',
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    assert runner.has_handlers("before_provider_headers") is True

    headers = await runner.emit_before_provider_headers({"User-Agent": "kimchi/1.0"})
    assert headers["X-Turn-Index"] == "3"
    assert headers["User-Agent"] == "kimchi/1.0"


async def test_before_provider_headers_isolates_a_throwing_handler(tmp_path):
    extensions = await _load(
        tmp_path,
        (
            "a-throwing.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        raise RuntimeError("header handler boom")\n\n'
            '    pi.on("before_provider_headers", _handler)\n',
        ),
        (
            "b-good.py",
            "def pi_extension(pi):\n"
            "    def _handler(event, ctx):\n"
            '        event.headers["X-Good"] = "yes"\n\n'
            '    pi.on("before_provider_headers", _handler)\n',
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))
    errors = []
    runner.on_error(errors.append)

    headers = await runner.emit_before_provider_headers({"User-Agent": "x"})

    assert headers["X-Good"] == "yes"
    assert headers["User-Agent"] == "x"
    assert len(errors) == 1
    assert errors[0].event == "before_provider_headers"
    assert "header handler boom" in errors[0].error


async def test_emit_context_chains_each_handlers_replacement_list(tmp_path):
    """`emit_context` feeds each handler the previous handler's result.

    Port of `emitContext`'s chaining. This is the wrapper `core/sdk.py` wires
    into the agent loop as `transform_context`, so it is the path production
    actually takes -- the `emit(ContextEvent(...))` form covered elsewhere
    exercises dispatch but not the chaining or the pass-through of a handler
    that returns nothing.
    """
    extensions = await _load(
        tmp_path,
        (
            "first.py",
            """
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ContextEventResult


def pi_extension(pi):
    async def _on_context(event, ctx):
        return ContextEventResult(
            messages=[*event.messages, {"role": "user", "content": [TextContent(text="first")]}]
        )

    pi.on("context", _on_context)
""",
        ),
        (
            "second.py",
            """
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ContextEventResult


def pi_extension(pi):
    async def _on_context(event, ctx):
        # Sees `first`'s output, which is what proves the chaining.
        return ContextEventResult(
            messages=[*event.messages, {"role": "user", "content": [TextContent(text="second")]}]
        )

    pi.on("context", _on_context)
""",
        ),
        (
            "observer.py",
            """
def pi_extension(pi):
    async def _on_context(event, ctx):
        return None

    pi.on("context", _on_context)
""",
        ),
    )
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    result = await runner.emit_context([])

    texts = [c.text for message in result for c in message["content"]]
    assert texts == ["first", "second"]


async def test_emit_context_returns_the_input_when_no_extension_handles_it(tmp_path):
    """A no-op run must hand back the messages unchanged, not an empty list."""
    extensions = await _load(tmp_path, ("noop.py", "def pi_extension(pi):\n    pass\n"))
    runner = ExtensionRunner(extensions, cwd=str(tmp_path))

    messages = [{"role": "user", "content": [TextContent(text="keep me")]}]
    result = await runner.emit_context(messages)

    assert [c.text for m in result for c in m["content"]] == ["keep me"]

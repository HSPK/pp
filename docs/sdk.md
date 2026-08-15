> pi can help you use the SDK. Ask it to build an integration for your use case.

# SDK

The Python SDK provides programmatic access to pi's agent session runtime. Use it to embed pi in Python applications, build custom interfaces, or integrate with automated workflows.

**Example use cases:**

- Build a custom UI.
- Integrate agent capabilities into existing Python applications.
- Create automated pipelines with agent reasoning.
- Build custom tools that spawn or coordinate sessions.
- Test agent behavior programmatically.

The main SDK entry point is [`src/pi_coding_agent/core/sdk.py`](../src/pi_coding_agent/core/sdk.py): `create_agent_session()` and `CreateAgentSessionOptions`.

See [examples/sdk/](../examples/sdk/) for runnable sibling examples:

- [`01_minimal.py`](../examples/sdk/01_minimal.py) — all defaults.
- [`02_custom_model.py`](../examples/sdk/02_custom_model.py) — selecting a model and thinking level.
- [`03_custom_prompt.py`](../examples/sdk/03_custom_prompt.py) — replacing or appending to the system prompt.

More are ported as the surfaces they need land.

## Quick Start

```python
import asyncio
import sys

from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager


async def main() -> None:
    model_runtime = await ModelRuntime.create()
    result = await create_agent_session(
        CreateAgentSessionOptions(
            model_runtime=model_runtime,
            session_manager=SessionManager.in_memory(),
        )
    )
    session = result.session

    def on_event(event: object) -> None:
        if getattr(event, "type", None) == "message_update":
            update = event.assistant_message_event
            if update.type == "text_delta":
                sys.stdout.write(update.delta)
                sys.stdout.flush()

    session.subscribe(on_event)
    await session.prompt("What files are in the current directory?")


if __name__ == "__main__":
    asyncio.run(main())
```

## Installation

From the Python workspace:

```bash
uv sync --all-packages
uv run pp --help
```

Package console scripts are defined in each package's `pyproject.toml`:

- `pp = "pi_coding_agent.cli:main"`
- `pp-ai = "pi_ai.cli:main"`
- `pp-evals = "pi_evals.run_evals:main"`

There is no separate SDK package; import from the workspace packages.

## Core Concepts

### createAgentSession()

Python name: `create_agent_session()`.

`create_agent_session()` wires together a `ModelRuntime`, `SessionManager`, `SettingsManager`, `ResourceLoader`, built-in tools, custom tools, and already-loaded extensions into an `AgentSession`.

Unlike the TypeScript SDK, this port's `ResourceLoader` does **not** load extensions. Load them with `discover_and_load_extensions()` and pass the resulting list to `CreateAgentSessionOptions.extensions`.

```python
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager


async def create_minimal_session() -> object:
    result = await create_agent_session()
    return result.session


async def create_read_only_session() -> object:
    result = await create_agent_session(
        CreateAgentSessionOptions(
            tools=["read", "grep", "find", "ls"],
            session_manager=SessionManager.in_memory(),
        )
    )
    return result.session
```

### AgentSession

The session manages agent lifecycle, message history, model state, compaction, tools, settings, and event streaming.

Common API surface:

```python
async def use_session(session: object) -> None:
    await session.prompt("What files are here?")
    await session.steer("Focus on the failing test")
    await session.follow_up("Summarize what changed")

    unsubscribe = session.subscribe(lambda event: print(getattr(event, "type", None)))
    unsubscribe()

    print(session.session_file)
    print(session.session_id)
    print(session.model)
    print(session.thinking_level)
    print(session.messages)
    print(session.is_streaming)

    await session.set_model(session.model)
    session.set_thinking_level("medium")
    await session.cycle_model()
    session.cycle_thinking_level()

    await session.compact("Focus on code changes")
    session.abort_compaction()

    await session.abort()
    session.dispose()
```

Session replacement APIs such as new-session, resume, fork, clone, and import live on `AgentSessionRuntime`, not on `AgentSession`.

### createAgentSessionRuntime() and AgentSessionRuntime

Python names: `create_agent_session_runtime()` and `AgentSessionRuntime`.

Use the runtime API when you need to replace the active session and rebuild session-bound state. This is the same layer used by print and interactive modes.

```python
import os

from pi_coding_agent.core.agent_session_runtime import create_agent_session_runtime
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager


async def create_runtime() -> object:
    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    session_manager = SessionManager.create(cwd)

    async def factory(**kwargs: object) -> object:
        return await create_agent_session(
            CreateAgentSessionOptions(
                cwd=kwargs.get("cwd", cwd),
                agent_dir=kwargs.get("agent_dir", agent_dir),
                session_manager=kwargs.get("session_manager", session_manager),
            )
        )

    return await create_agent_session_runtime(
        factory,
        cwd=cwd,
        agent_dir=agent_dir,
        session_manager=session_manager,
    )
```

`AgentSessionRuntime` owns replacement of the active runtime across:

- `new_session()`
- `switch_session()`
- `fork()`
- clone flows via `fork(entry_id, position="at")`
- `import_from_jsonl()`

Important behavior:

- `runtime.session` changes after replacement.
- Event subscriptions are attached to a specific `AgentSession`, so re-subscribe after replacement.
- If you use extensions, call `runtime.session.bind_extensions()` for the new session; modes do this through `set_rebind_session()`.
- If runtime creation or replacement fails, the method raises and the caller decides how to handle it.

```python
async def replace_session(runtime: object) -> None:
    session = runtime.session
    unsubscribe = session.subscribe(lambda event: None)

    await runtime.new_session()

    unsubscribe()
    session = runtime.session
    session.subscribe(lambda event: None)
```

### Prompting and Message Queueing

`AgentSession.prompt()` is asynchronous and uses keyword options:

```python
async def prompt_examples(session: object) -> None:
    await session.prompt("What files are here?")

    await session.prompt(
        "Stop and do this instead",
        streaming_behavior="steer",
    )

    await session.prompt(
        "After you're done, also check X",
        streaming_behavior="followUp",
    )
```

Relevant options:

- `expand_prompt_templates: bool = True`
- `images: list[ImageContent] | None = None`
- `streaming_behavior: "steer" | "followUp" | None = None`
- `source: "interactive" | "rpc" | "extension" = "interactive"`
- `preflight_result: Callable[[bool], None] | None = None`

Behavior:

- Extension commands execute immediately, even while streaming.
- Skill commands and file-based prompt templates expand by default.
- During streaming without `streaming_behavior`, `prompt()` raises.
- `preflight_result(True)` means the prompt was accepted, queued, or handled immediately.
- `preflight_result(False)` means preflight rejected before acceptance.

Explicit queueing:

```python
async def queue_examples(session: object) -> None:
    await session.steer("New instruction")
    await session.follow_up("After you're done, also do this")
```

Both `steer()` and `follow_up()` expand skill commands and prompt templates, but reject registered extension commands.

### Agent and AgentState

`AgentSession.agent` is the low-level `pi_agent.agent.Agent` instance.

```python
async def inspect_agent_state(session: object) -> None:
    state = session.agent.state

    messages = state.messages
    model = state.model
    thinking_level = state.thinking_level
    system_prompt = state.system_prompt
    tools = state.tools
    streaming_message = getattr(state, "streaming_message", None)
    error_message = getattr(state, "error_message", None)

    state.messages = list(messages)
    state.tools = list(tools)
    await session.agent.wait_for_idle()
    print(model, thinking_level, system_prompt, streaming_message, error_message)
```

### Events

Subscribe to session events to receive streaming output and lifecycle notifications. Events are dataclass instances with a `type` attribute. Python attribute names are snake_case.

```python
def on_event(event: object) -> None:
    event_type = getattr(event, "type", None)
    if event_type == "message_update":
        update = event.assistant_message_event
        if update.type == "text_delta":
            print(update.delta, end="", flush=True)
        if update.type == "thinking_delta":
            pass
    elif event_type == "tool_execution_start":
        print(f"Tool: {event.tool_name}")
    elif event_type == "tool_execution_end":
        print("error" if event.is_error else "success")
    elif event_type == "queue_update":
        print(event.steering, event.follow_up)
    elif event_type in {
        "agent_start",
        "agent_end",
        "agent_settled",
        "turn_start",
        "turn_end",
        "message_start",
        "message_end",
        "compaction_start",
        "compaction_end",
        "auto_retry_start",
        "auto_retry_end",
        "summarization_retry_scheduled",
        "summarization_retry_attempt_start",
        "summarization_retry_finished",
        "thinking_level_changed",
        "session_info_changed",
    }:
        pass


unsubscribe = session.subscribe(on_event)
```

## Options Reference

### Directories

```python
import os

from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def create_with_directories() -> object:
    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    loader = ResourceLoader(ResourceLoaderOptions(cwd=cwd, agent_dir=agent_dir))
    loader.reload()
    return await create_agent_session(CreateAgentSessionOptions(cwd=cwd, agent_dir=agent_dir, resource_loader=loader))
```

`cwd` is used for project resources and tool path resolution. `agent_dir` defaults to `~/.pi/agent` and is used for global settings, credentials, models, prompts, skills, and sessions.

`ResourceLoader` discovers:

- Project skills: `<cwd>/.pi/skills/` when trusted.
- User skills: `<agent_dir>/skills/`.
- Project prompts: `<cwd>/.pi/prompts/` when trusted.
- User prompts: `<agent_dir>/prompts/`.
- Context files: `AGENTS.override.md`, `AGENTS.md`, `AGENTS.MD`, `CLAUDE.md`, `CLAUDE.MD`.
- System prompt files: `.pi/SYSTEM.md` or `<agent_dir>/SYSTEM.md`.
- Append prompt files: `.pi/APPEND_SYSTEM.md` or `<agent_dir>/APPEND_SYSTEM.md`.

Themes are not loaded by this Python `ResourceLoader`. Extensions are loaded separately.

### Model

```python
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def create_with_model() -> object:
    model_runtime = await ModelRuntime.create()
    model = model_runtime.get_model("openai", "gpt-5.6-sol")
    if model is None:
        raise RuntimeError("Model not found")

    available = await model_runtime.get_available()
    print(len(available))

    return await create_agent_session(
        CreateAgentSessionOptions(
            model=model,
            thinking_level="medium",
            model_runtime=model_runtime,
        )
    )
```

If no model is provided:

1. The session's saved model is restored when possible.
2. Settings defaults are used.
3. The first available authenticated model is used.

CLI-style model parsing helpers are ported with snake_case names:

```python
from pi_coding_agent.core.model_resolver import resolve_cli_model, resolve_model_scope_with_diagnostics


async def resolve_models(model_runtime: object) -> None:
    cli_model = resolve_cli_model(
        model_runtime,
        cli_model="openai/gpt-5.6-sol:high",
    )
    if cli_model.error:
        raise RuntimeError(cli_model.error)

    scope = await resolve_model_scope_with_diagnostics(
        ["openai/*:high", "gpt-5"],
        model_runtime,
    )
    scoped_models = scope.scoped_models
    for diagnostic in scope.diagnostics:
        print(diagnostic.message)
```

### API Keys and OAuth

Authentication resolution is handled by `ModelRuntime`:

1. Runtime overrides via `set_runtime_api_key()`.
2. Stored credentials in `auth.json`.
3. Provider environment variables.
4. Configured model/provider auth from `models.json`.

```python
import os

from pi_coding_agent.core.model_runtime import ModelRuntime


async def configure_auth() -> None:
    model_runtime = await ModelRuntime.create()

    for provider in model_runtime.get_providers():
        status = await model_runtime.check_auth(provider.id)
        print(provider.name, status)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        await model_runtime.set_runtime_api_key("anthropic", api_key)

    custom_runtime = await ModelRuntime.create(
        auth_path="app-state/auth.json",
        models_path="app-state/models.json",
    )
    print(custom_runtime.get_error())
```

OAuth login and credential storage are partially ported. Stored OAuth credentials are reported, but request signing through OAuth is not fully wired in `pi_ai`; API-key auth is the reliable path.

Remote model catalog refresh and `CredentialSynchronizationError` are not ported. `ModelRuntime.refresh()` reloads local `models.json` only.

### System Prompt

Use `ResourceLoaderOptions.system_prompt` and `append_system_prompt` to override prompt text.

```python
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def create_with_prompt(cwd: str) -> object:
    loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=cwd,
            agent_dir=get_agent_dir(),
            system_prompt="You are a helpful assistant.",
            append_system_prompt=["Be concise."],
        )
    )
    loader.reload()
    return await create_agent_session(CreateAgentSessionOptions(cwd=cwd, resource_loader=loader))
```

### Tools

Built-in tool names: `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`.

Default active built-ins: `read`, `bash`, `edit`, `write`.

- `no_tools="all"` disables all tools.
- `no_tools="builtin"` disables default built-ins while keeping extension and custom tools available.
- `tools=[...]` is an allowlist for built-in, extension, and custom tools.
- `exclude_tools=[...]` disables specific tool names after any allowlist is applied.

```python
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def create_tool_variants() -> None:
    read_only = await create_agent_session(CreateAgentSessionOptions(tools=["read", "grep", "find", "ls"]))
    selected = await create_agent_session(CreateAgentSessionOptions(tools=["read", "bash", "grep"]))
    without_bash = await create_agent_session(CreateAgentSessionOptions(exclude_tools=["bash"]))
    print(read_only.session, selected.session, without_bash.session)
```

#### Tools with Custom cwd

When you pass `cwd`, built-in tools are created for that directory.

```python
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager


async def create_for_project(cwd: str) -> object:
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=cwd,
            tools=["read", "bash", "grep"],
            session_manager=SessionManager.in_memory(cwd),
        )
    )
    return result.session
```

### Custom Tools

Pass custom tools as a dictionary keyed by tool name. Use `pi_agent.types.AgentTool` and `AgentToolResult`.

```python
import time
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def status_tool_execute(
    tool_call_id: str,
    args: Any,
    signal: object | None = None,
    update: object | None = None,
) -> AgentToolResult:
    del tool_call_id, args, signal, update
    return AgentToolResult(content=[TextContent(text=f"Uptime: {time.monotonic():.0f}s")])


status_tool = AgentTool(
    name="status",
    label="Status",
    description="Get process status",
    parameters={"type": "object", "properties": {}},
    execute=status_tool_execute,
)


async def create_with_custom_tool() -> object:
    result = await create_agent_session(
        CreateAgentSessionOptions(custom_tools={"status": status_tool}, tools=["read", "status"])
    )
    return result.session
```

`define_tool()` also exists for extension-style `ToolDefinition` objects, but SDK `custom_tools` expects runnable `AgentTool` instances.

### Extensions

Extensions are ported, but they are loaded outside `ResourceLoader` in Python. Load them first, then pass the already-loaded list to `CreateAgentSessionOptions.extensions`.

```python
from pi_coding_agent.core.extensions import discover_and_load_extensions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def create_with_extensions(cwd: str) -> object:
    loaded = await discover_and_load_extensions(
        configured_paths=[],
        cwd=cwd,
        project_trusted=True,
    )
    for error in loaded.errors:
        print(error)

    result = await create_agent_session(CreateAgentSessionOptions(cwd=cwd, extensions=loaded.extensions))
    return result.session
```

Extensions can register tools, subscribe to events, add commands, and handle session lifecycle hooks. The extension UI is narrowed to a headless-safe subset; TUI widgets, themes, custom renderers, provider registration, and extension CLI flags are not ported.

See [extensions.md](extensions.md) for the full API.

### Skills

Use `ResourceLoaderOptions.additional_skill_paths` to add skill files or directories.

```python
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def create_with_skills(cwd: str) -> object:
    loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=cwd,
            agent_dir=get_agent_dir(),
            additional_skill_paths=["docs/skills"],
        )
    )
    loader.reload()
    result = await create_agent_session(CreateAgentSessionOptions(cwd=cwd, resource_loader=loader))
    return result.session
```

Synthetic in-memory skill overrides from the TypeScript `DefaultResourceLoader` are not ported.

### Context Files

`ResourceLoader` discovers context files from disk. In-memory `agentsFilesOverride` is not ported.

```python
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions


def load_context(cwd: str) -> list[dict[str, str]]:
    loader = ResourceLoader(ResourceLoaderOptions(cwd=cwd, agent_dir=get_agent_dir()))
    loader.reload()
    return loader.get_agents_files()
```

### Slash Commands

Prompt templates are slash commands. Use `ResourceLoaderOptions.additional_prompt_template_paths` to add template files or directories.

```python
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session


async def create_with_prompts(cwd: str) -> object:
    loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=cwd,
            agent_dir=get_agent_dir(),
            additional_prompt_template_paths=["prompts"],
        )
    )
    loader.reload()
    result = await create_agent_session(CreateAgentSessionOptions(cwd=cwd, resource_loader=loader))
    return result.session
```

### Session Management

Sessions use append-only JSONL trees with `id`/`parent_id` links and a current leaf.

```python
import os

from pi_coding_agent.core.agent_session_runtime import create_agent_session_runtime
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager


async def session_examples() -> None:
    cwd = os.getcwd()

    in_memory = await create_agent_session(CreateAgentSessionOptions(session_manager=SessionManager.in_memory(cwd)))

    persisted = await create_agent_session(CreateAgentSessionOptions(session_manager=SessionManager.create(cwd)))

    continued = await create_agent_session(
        CreateAgentSessionOptions(session_manager=SessionManager.continue_recent(cwd))
    )

    opened = await create_agent_session(CreateAgentSessionOptions(session_manager=SessionManager.open("session.jsonl")))

    current_project_sessions = await SessionManager.list(cwd)
    all_sessions = await SessionManager.list_all()

    print(in_memory.session, persisted.session, continued.session, opened.session)
    print(current_project_sessions, all_sessions)
```

Runtime replacement:

```python
import os

from pi_coding_agent.core.agent_session_runtime import create_agent_session_runtime
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager


async def runtime_session_examples() -> None:
    cwd = os.getcwd()
    agent_dir = get_agent_dir()
    session_manager = SessionManager.create(cwd)

    async def factory(**kwargs: object) -> object:
        return await create_agent_session(
            CreateAgentSessionOptions(
                cwd=kwargs.get("cwd", cwd),
                agent_dir=kwargs.get("agent_dir", agent_dir),
                session_manager=kwargs.get("session_manager", session_manager),
            )
        )

    runtime = await create_agent_session_runtime(
        factory,
        cwd=cwd,
        agent_dir=agent_dir,
        session_manager=session_manager,
    )

    await runtime.new_session()
    await runtime.switch_session("session.jsonl")
    await runtime.fork("entry-id")
    await runtime.fork("entry-id", position="at")
```

`SessionManager` tree API:

```python
from pi_coding_agent.core.session_manager import SessionManager


manager = SessionManager.open("session.jsonl")
entries = manager.get_entries()
tree = manager.get_tree()
branch = manager.get_branch()
leaf = manager.get_leaf_entry()
entry = manager.get_entry("entry-id")
children = manager.get_children("entry-id")
label = manager.get_label("entry-id")
manager.append_label_change("entry-id", "checkpoint")
manager.branch("entry-id")
manager.branch_with_summary("entry-id", "Summary")
manager.create_branched_session("entry-id")
print(entries, tree, branch, leaf, entry, children, label)
```

See [Session Format](session-format.md).

### Settings Management

```python
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager


async def settings_examples(cwd: str) -> None:
    from_files = SettingsManager.create(cwd)

    with_overrides = SettingsManager.create(cwd)
    with_overrides.apply_overrides({"compaction": {"enabled": False}})

    in_memory = SettingsManager.in_memory({"compaction": {"enabled": False}})

    result = await create_agent_session(
        CreateAgentSessionOptions(
            settings_manager=in_memory,
            session_manager=SessionManager.in_memory(cwd),
        )
    )

    await with_overrides.flush()
    errors = with_overrides.drain_errors()
    print(from_files, result.session, errors)
```

Static factories:

- `SettingsManager.create(cwd, agent_dir=None, options=None)` loads files.
- `SettingsManager.in_memory(settings=None, options=None)` avoids file I/O.
- `SettingsManager.from_storage(storage, options=None)` uses a custom backend.

Settings load from two locations and merge:

1. Global: `<agent_dir>/settings.json`.
2. Project: `<cwd>/.pi/settings.json` when the project is trusted.

Project values override global values. Nested dictionaries merge by key. Writes are synchronous in this port; `flush()` is a no-op kept for API parity.

## ResourceLoader

Use `ResourceLoader` to discover skills, prompt templates, context files, and system prompts. It does not discover extensions or themes in this Python port.

```python
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions


cwd = "."
loader = ResourceLoader(ResourceLoaderOptions(cwd=cwd, agent_dir=get_agent_dir()))
loader.reload()

skills = loader.get_skills().skills
prompts, prompt_diagnostics = loader.get_prompts()
context_files = loader.get_agents_files()
system_prompt = loader.get_system_prompt()
append_system_prompt = loader.get_append_system_prompt()
print(skills, prompts, prompt_diagnostics, context_files, system_prompt, append_system_prompt)
```

## Return Value

`create_agent_session()` returns `CreateAgentSessionResult`:

```python
from dataclasses import fields

from pi_coding_agent.core.sdk import CreateAgentSessionResult


result_fields = [field.name for field in fields(CreateAgentSessionResult)]
print(result_fields)
```

Fields:

- `session: AgentSession`
- `model_fallback_message: str | None`

The TypeScript `extensionsResult` field is not present. This port keeps extension loading outside `ResourceLoader`; pass already-loaded extensions through `CreateAgentSessionOptions.extensions`.

## Complete Example

```python
import asyncio
import os
import sys
import time
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager


async def status_execute(
    tool_call_id: str,
    args: Any,
    signal: object | None = None,
    update: object | None = None,
) -> AgentToolResult:
    del tool_call_id, args, signal, update
    return AgentToolResult(content=[TextContent(text=f"Uptime: {time.monotonic():.0f}s")])


async def main() -> None:
    cwd = os.getcwd()
    agent_dir = get_agent_dir()

    model_runtime = await ModelRuntime.create(
        auth_path="app-state/auth.json",
        models_path="app-state/models.json",
    )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        await model_runtime.set_runtime_api_key("anthropic", api_key)

    model = model_runtime.get_model("anthropic", "claude-opus-4-5")

    settings_manager = SettingsManager.in_memory(
        {"compaction": {"enabled": False}, "retry": {"enabled": True, "maxRetries": 2}}
    )

    loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=cwd,
            agent_dir=agent_dir,
            system_prompt="You are a minimal assistant. Be concise.",
        )
    )
    loader.reload()

    status_tool = AgentTool(
        name="status",
        label="Status",
        description="Get system status",
        parameters={"type": "object", "properties": {}},
        execute=status_execute,
    )

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=cwd,
            agent_dir=agent_dir,
            model=model,
            thinking_level="off",
            model_runtime=model_runtime,
            tools=["read", "bash", "status"],
            custom_tools={"status": status_tool},
            resource_loader=loader,
            session_manager=SessionManager.in_memory(cwd),
            settings_manager=settings_manager,
        )
    )
    session = result.session

    def on_event(event: object) -> None:
        if getattr(event, "type", None) == "message_update":
            update = event.assistant_message_event
            if update.type == "text_delta":
                sys.stdout.write(update.delta)
                sys.stdout.flush()

    session.subscribe(on_event)
    await session.prompt("Get status and list files.")


if __name__ == "__main__":
    asyncio.run(main())
```

## Run Modes

The SDK exports utilities for building interfaces on top of `create_agent_session()`.

### InteractiveMode

Full TUI interactive mode is ported in `pi_coding_agent.modes.interactive.interactive_mode.InteractiveMode`. The CLI starts it when `pp` runs on a TTY without `-p` or `--mode json`.

```python
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode, InteractiveModeOptions


async def run_interactive(runtime: object) -> None:
    mode = InteractiveMode(
        runtime,
        InteractiveModeOptions(
            model_fallback_message=None,
            initial_message="Hello",
            initial_images=[],
            initial_messages=[],
        ),
    )
    await mode.run()
```

### runPrintMode

Python name: `run_print_mode()`.

```python
from pi_coding_agent.modes.print_mode import PrintModeOptions, run_print_mode


async def run_print(runtime: object) -> int:
    return await run_print_mode(
        runtime,
        PrintModeOptions(
            mode="text",
            initial_message="Hello",
            initial_images=[],
            messages=["Follow up"],
        ),
    )
```

### runRpcMode

Unavailable in this Python port. Legacy stdio RPC is superseded by the `pi_server`/`pi_client` socket stack; see [RPC documentation](rpc.md).

## RPC Mode Alternative

For subprocess-based integration, use `pp --mode rpc`; see [rpc-stdio.md](rpc-stdio.md).

Use one of these instead:

- `uv run pp --mode json "prompt"` for one-shot JSON event output.
- `pi_server` + `pi_client` over a Unix-domain socket for durable sessions and multi-client attachment.
- The in-process SDK when your host is Python.

The SDK is preferred when:

- You are in the same Python process.
- You need direct access to agent state.
- You want to customize tools/extensions programmatically.

The socket stack is preferred when:

- You want process isolation.
- You need multiple clients to attach/detach.
- You are building a language-agnostic client around `pi_protocol`.

## Exports

The top-level `pi_coding_agent` package does not re-export the SDK. Import from the modules that define the symbols.

Common imports:

```python
from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime, create_agent_session_runtime
from pi_coding_agent.core.config import CONFIG_DIR_NAME, get_agent_dir
from pi_coding_agent.core.extensions import define_tool, discover_and_load_extensions
from pi_coding_agent.core.model_resolver import resolve_cli_model, resolve_model_scope_with_diagnostics
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.tools import create_all_tools, create_coding_tools, create_read_only_tools
```

For extension types, see [extensions.md](extensions.md). For socket RPC, see [rpc.md](rpc.md).

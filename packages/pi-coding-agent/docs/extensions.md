> pi can create extensions. Ask it to build one for your use case.

# Extensions

Extensions are Python modules that extend pi's behavior. They can subscribe to lifecycle events, register custom tools callable by the LLM, and add slash commands.

This document describes the Python port. Python extensions are not TypeScript extensions: use `.py` files, snake_case API names, JSON Schema dictionaries, and the `pp` CLI entry point. Several TypeScript extension features are intentionally not ported; those sections say so explicitly.

> **Placement for startup discovery:** Put extensions in `~/.pi/agent/extensions/` (global) or `.pi/extensions/` (project-local). Project-local extensions load only after project trust is granted. Use `uv run pp -e ./path.py` for quick tests. The Python `/reload` command does not hot-reload extension code.

**Key capabilities:**
- **Custom tools** - Register tools the LLM can call via `pi.register_tool()`
- **Event interception** - Block or modify tool calls, inject context, customize compaction
- **Custom commands** - Register commands like `/mycommand` via `pi.register_command()`
- **Session access** - Read session state from `ctx.session_manager`
- **Custom tool state** - Store branch-aware state in tool result `details`

**Not available in the Python port:** custom header/footer components, extension-driven renderers, terminal input listeners, extension shortcuts, extension CLI flags, dynamic provider registration, `resources_discover`, `user_bash`, and extension hot reload. Widgets and the `ctx.ui` dialogs *are* available in interactive mode (see [UI Context](#ui-context)).

See [examples/extensions/](../examples/extensions/) for working Python implementations.

## Table of Contents

- [Quick Start](#quick-start)
- [Extension Locations](#extension-locations)
- [Available Imports](#available-imports)
- [Writing an Extension](#writing-an-extension)
  - [Extension Styles](#extension-styles)
- [Events](#events)
  - [Lifecycle Overview](#lifecycle-overview)
  - [Resource Events](#resource-events)
  - [Session Events](#session-events)
  - [Agent Events](#agent-events)
  - [Model Events](#model-events)
  - [Tool Events](#tool-events)
- [ExtensionContext](#extensioncontext)
- [ExtensionCommandContext](#extensioncommandcontext)
- [ExtensionAPI Methods](#extensionapi-methods)
- [State Management](#state-management)
- [Custom Tools](#custom-tools)
  - [Dynamic Tool Loading](#dynamic-tool-loading)
- [Custom UI](#custom-ui)
- [Error Handling](#error-handling)
- [Mode Behavior](#mode-behavior)
- [Examples Reference](#examples-reference)

## Quick Start

Create `~/.pi/agent/extensions/my_extension.py`:

```python
from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import (
    ExtensionContext,
    ToolCallEvent,
    ToolCallEventResult,
    ToolDefinition,
)


async def greet(tool_call_id, params, signal, on_update, ctx):
    return AgentToolResult(
        content=[TextContent(text=f"Hello, {params['name']}!")],
        details={},
    )


def pi_extension(pi: ExtensionAPI) -> None:
    def on_session_start(event, ctx: ExtensionContext) -> None:
        if ctx.has_ui:
            ctx.ui.notify("Extension loaded", "info")

    async def on_tool_call(event: ToolCallEvent, ctx: ExtensionContext) -> ToolCallEventResult | None:
        command = event.input.get("command")
        if event.tool_name == "bash" and isinstance(command, str) and "rm -rf" in command:
            return ToolCallEventResult(block=True, reason="Blocked by extension", terminate=True)
        return None

    async def hello(args: str, ctx) -> None:
        if ctx.has_ui:
            ctx.ui.notify(f"Hello {args.strip() or 'world'}", "info")

    pi.on("session_start", on_session_start)
    pi.on("tool_call", on_tool_call)
    pi.register_tool(
        ToolDefinition(
            name="greet",
            label="Greet",
            description="Greet someone by name",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Name to greet"}},
                "required": ["name"],
            },
            execute=greet,
        )
    )
    pi.register_command("hello", handler=hello, description="Say hello")
```

Test with `--extension` (or `-e`):

```bash
uv run pp -e ./my_extension.py
```

## Extension Locations

> **Security:** Extensions run with your full system permissions and can execute arbitrary Python code. Only install extensions from sources you trust.

Extensions are auto-discovered from trusted locations. Project-local `.pi/extensions` entries load only after the project is trusted.

| Location | Scope |
|----------|-------|
| `~/.pi/agent/extensions/*.py` | Global (all projects) |
| `~/.pi/agent/extensions/*/__init__.py` | Global (subdirectory) |
| `.pi/extensions/*.py` | Project-local |
| `.pi/extensions/*/__init__.py` | Project-local (subdirectory) |

Additional paths via `settings.json`:

```json
{
  "packages": [
    "git:github.com/user/repo@v1",
    "./local-package"
  ],
  "extensions": [
    "/path/to/local/extension.py",
    "/path/to/local/extension/dir"
  ]
}
```

Global settings live at `~/.pi/agent/settings.json`. Project settings live at `.pi/settings.json` and require project trust.

To share extensions as pi packages, use git or local-path packages with a top-level `pi.json` manifest. npm-sourced packages are not available in the Python port. See [packages.md](packages.md).

## Available Imports

| Package/module | Purpose |
|---------|---------|
| `pi_coding_agent.core.extensions.loader` | `ExtensionAPI`, extension loading helpers |
| `pi_coding_agent.core.extensions.types` | Extension events, result dataclasses, `ToolDefinition`, contexts |
| `pi_agent.types` | `AgentToolResult`, `AgentToolUpdateCallback`, tool execution types |
| `pi_ai.types` | `TextContent`, `ImageContent`, `Model`, `Usage` |
| `pi_coding_agent.tools.*` | Built-in tool factories and helpers such as truncation and file mutation queue |
| `pi_coding_agent.core.config` | `CONFIG_DIR_NAME`, `get_agent_dir()` |
| `pi_tui` | TUI primitives; extension renderer/UI-host hooks are not wired in the Python port |

Dependencies work like normal Python dependencies. Install them into the environment that runs `pp` (for example with `uv add` for a workspace or `pip install` for a virtual environment).

Python standard-library imports are available.

## Writing an Extension

An extension defines a module-level callable named `pi_extension`. It receives an `ExtensionAPI`. The callable can be synchronous or asynchronous:

```python
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import ExtensionContext


def pi_extension(pi: ExtensionAPI) -> None:
    def on_event(event, ctx: ExtensionContext) -> None:
        if ctx.has_ui:
            ctx.ui.notify("Done", "info")

    async def command_handler(args: str, ctx) -> None:
        await ctx.wait_for_idle()

    pi.on("session_start", on_event)
    pi.register_command("name", handler=command_handler, description="Run a command")
```

Extension modules are loaded with `importlib`, so they must be valid Python. The entry point must be named `pi_extension`; there is no default export.

If `pi_extension` returns an awaitable, pi awaits it before startup continues. Do one-time setup there, but do not start long-lived resources from the factory.

### Async factory functions

Use an async factory for one-time setup that must finish before handlers are registered.

```python
from pi_coding_agent.core.extensions.loader import ExtensionAPI


async def pi_extension(pi: ExtensionAPI) -> None:
    result = await pi.exec("git", ["rev-parse", "--show-toplevel"])
    root = result.stdout.strip() if result.code == 0 else "unknown"

    async def show_root(args: str, ctx) -> None:
        if ctx.has_ui:
            ctx.ui.notify(f"Git root: {root}", "info")

    pi.register_command("git-root", handler=show_root, description="Show git root")
```

Dynamic provider registration from an extension is not available in the Python port.

### Long-lived resources and shutdown

Do not start processes, sockets, file watchers, or timers from the factory. Start them from `session_start`, a command, a tool, or another event that needs them, and clean them up in `session_shutdown`.

```python
from pi_coding_agent.core.extensions.loader import ExtensionAPI


def pi_extension(pi: ExtensionAPI) -> None:
    state = {"open": False}

    def on_session_start(event, ctx) -> None:
        state["open"] = True

    def on_session_shutdown(event, ctx) -> None:
        state["open"] = False

    pi.on("session_start", on_session_start)
    pi.on("session_shutdown", on_session_shutdown)
```

### Extension Styles

**Single file** - simplest, for small extensions:

```
~/.pi/agent/extensions/
└── my_extension.py
```

**Directory with `__init__.py`** - for multi-file extensions:

```
~/.pi/agent/extensions/
└── my_extension/
    ├── __init__.py      # Entry point: defines pi_extension(pi)
    ├── tools.py         # Helper module
    └── utils.py         # Helper module
```

**Package with `pi.json`** - for extensions that need non-default resource paths:

```
~/.pi/agent/extensions/
└── my_extension/
    ├── pi.json
    └── src/
        └── main.py
```

```json
{
  "extensions": ["./src/main.py"],
  "skills": [],
  "prompts": [],
  "themes": []
}
```

Use `uv` or `pip` to install third-party dependencies into the Python environment running `pp`.

## Events

### Lifecycle Overview

```
pp starts
  │
  ├─► project_trust (user/global and CLI extensions only)
  ├─► session_start { reason: "startup" }
  └─► user sends prompt
      │
      ├─► extension commands checked first
      ├─► input
      ├─► before_agent_start
      ├─► agent_start
      ├─► message_start / message_update / message_end
      │
      ├─► turn_start
      ├─► context
      ├─► before_provider_request
      ├─► after_provider_response
      ├─► tool_execution_start
      ├─► tool_call
      ├─► tool_execution_update
      ├─► tool_result
      ├─► tool_execution_end
      └─► turn_end

session replacement through AgentSessionRuntime
  ├─► session_before_switch or session_before_fork
  ├─► session_shutdown
  └─► session_start { reason: "new" | "resume" | "fork" }

compaction
  ├─► session_before_compact
  └─► session_compact

tree navigation
  ├─► session_before_tree
  └─► session_tree

model/thinking changes
  ├─► model_select
  └─► thinking_level_select

exit
  └─► session_shutdown
```

`resources_discover`, `user_bash`, and extension reload events are not wired in the Python port.

### Startup Events

#### project_trust

Fired before pi decides whether to trust a project with dynamic configs. Only user/global and CLI `-e` extensions participate; project-local extensions are not loaded until after trust is resolved.

```python
from pi_coding_agent.core.extensions.types import ProjectTrustEventResult


def pi_extension(pi) -> None:
    async def on_project_trust(event, ctx):
        if ctx.has_ui and await ctx.ui.confirm("Trust project?", event.cwd):
            return ProjectTrustEventResult(trusted="yes", remember=True)
        return ProjectTrustEventResult(trusted="undecided")

    pi.on("project_trust", on_project_trust)
```

Return `ProjectTrustEventResult(trusted="yes" | "no" | "undecided")`. Use `remember=True` to persist a yes/no decision.

### Resource Events

#### resources_discover

`ResourcesDiscoverEvent` and `ResourcesDiscoverResult` exist in `core.extensions.types`, but `resources_discover` is not emitted by the Python port. Extension-contributed skill, prompt, and theme paths are not available through this hook.

### Session Events

See [Session Format](session-format.md) for session storage internals and the SessionManager API.

#### session_start

Fired when a session starts or when `AgentSessionRuntime` replaces a session.

```python
def pi_extension(pi) -> None:
    def on_session_start(event, ctx) -> None:
        session_file = ctx.session_manager.get_session_file()
        if ctx.has_ui:
            ctx.ui.notify(f"Session: {session_file or 'ephemeral'}", "info")

    pi.on("session_start", on_session_start)
```

`event.reason` is one of `"startup"`, `"new"`, `"resume"`, or `"fork"` in normal Python-port execution. The `"reload"` reason is defined but not produced.

#### session_info_changed

Fired when the current session display name is changed.

```python
def pi_extension(pi) -> None:
    def on_session_info_changed(event, ctx) -> None:
        if ctx.has_ui:
            ctx.ui.notify(f"Session renamed: {event.name or '(none)'}", "info")

    pi.on("session_info_changed", on_session_info_changed)
```

#### session_before_switch

Fired before `AgentSessionRuntime.new_session()` or `switch_session()` replaces the current session.

```python
from pi_coding_agent.core.extensions.types import SessionBeforeSwitchResult


def pi_extension(pi) -> None:
    async def on_before_switch(event, ctx):
        if event.reason == "new" and ctx.has_ui:
            ok = await ctx.ui.confirm("New session?", "Switch away from the current session?")
            if not ok:
                return SessionBeforeSwitchResult(cancel=True)
        return None

    pi.on("session_before_switch", on_before_switch)
```

#### session_before_fork

Fired before `AgentSessionRuntime.fork()`.

```python
from pi_coding_agent.core.extensions.types import SessionBeforeForkResult


def pi_extension(pi) -> None:
    def on_before_fork(event, ctx):
        if event.position == "before":
            return None
        return SessionBeforeForkResult(cancel=False)

    pi.on("session_before_fork", on_before_fork)
```

#### session_before_compact / session_compact

Fired on manual and automatic compaction. See [compaction.md](compaction.md).

```python
from pi_coding_agent.core.compaction import CompactionResult
from pi_coding_agent.core.extensions.types import SessionBeforeCompactResult


def pi_extension(pi) -> None:
    def on_before_compact(event, ctx):
        if event.reason == "manual":
            return None
        return SessionBeforeCompactResult(cancel=False)

    def on_compact(event, ctx) -> None:
        if ctx.has_ui:
            ctx.ui.notify(f"Compacted: {event.reason}", "info")

    pi.on("session_before_compact", on_before_compact)
    pi.on("session_compact", on_compact)
```

An extension may cancel compaction or return a `CompactionResult` through `SessionBeforeCompactResult(compaction=...)`.

#### session_before_tree / session_tree

Fired on session tree navigation. See [Sessions](sessions.md).

```python
from pi_coding_agent.core.extensions.types import SessionBeforeTreeResult


def pi_extension(pi) -> None:
    def on_before_tree(event, ctx):
        if event.preparation.label:
            return SessionBeforeTreeResult(label=event.preparation.label)
        return None

    def on_tree(event, ctx) -> None:
        if ctx.has_ui:
            ctx.ui.notify(f"Tree leaf: {event.new_leaf_id}", "info")

    pi.on("session_before_tree", on_before_tree)
    pi.on("session_tree", on_tree)
```

#### session_shutdown

Fired before a session runtime is torn down.

```python
def pi_extension(pi) -> None:
    def on_shutdown(event, ctx) -> None:
        reason = event.reason
        print(f"extension shutdown: {reason}")

    pi.on("session_shutdown", on_shutdown)
```

### Agent Events

#### before_agent_start

Fired after user input is accepted and before the agent loop starts. It can inject a custom message and/or replace the system prompt for that turn.

```python
from pi_coding_agent.core.extensions.types import BeforeAgentStartEventResult


def pi_extension(pi) -> None:
    def on_before_agent_start(event, ctx):
        return BeforeAgentStartEventResult(system_prompt=event.system_prompt + "\n\nFor this turn, answer concisely.")

    pi.on("before_agent_start", on_before_agent_start)
```

`event.system_prompt_options` exposes the base inputs used to build the system prompt.

#### agent_start / agent_end / agent_settled

`agent_start` fires when a low-level agent run begins. `agent_end` fires when it ends. `agent_settled` fires after retries, compaction retries, and queued continuations are done.

```python
def pi_extension(pi) -> None:
    def on_agent_settled(event, ctx) -> None:
        if ctx.is_idle() and ctx.has_ui:
            ctx.ui.notify("Agent idle", "info")

    pi.on("agent_settled", on_agent_settled)
```

#### turn_start / turn_end

Fired for each turn.

```python
def pi_extension(pi) -> None:
    def on_turn_end(event, ctx) -> None:
        usage = ctx.get_context_usage()
        if usage is not None and ctx.has_ui:
            ctx.ui.notify(f"Context tokens: {usage.tokens}", "info")

    pi.on("turn_end", on_turn_end)
```

#### message_start / message_update / message_end

Fired for user, assistant, and tool-result messages. `message_end` handlers can return `MessageEndEventResult(message=...)` to replace the finalized message. The replacement must keep the same role.

```python
from pi_coding_agent.core.extensions.types import MessageEndEventResult


def pi_extension(pi) -> None:
    def on_message_end(event, ctx):
        if getattr(event.message, "role", None) != "assistant":
            return None
        return MessageEndEventResult(message=event.message)

    pi.on("message_end", on_message_end)
```

#### tool_execution_start / tool_execution_update / tool_execution_end

Fired for tool execution lifecycle updates.

```python
def pi_extension(pi) -> None:
    def on_tool_execution_start(event, ctx) -> None:
        print(f"tool starting: {event.tool_name} {event.tool_call_id}")

    pi.on("tool_execution_start", on_tool_execution_start)
```

#### context

Fired before each LLM call. Return `ContextEventResult(messages=...)` to replace the context messages for that request.

```python
from pi_coding_agent.core.extensions.types import ContextEventResult


def pi_extension(pi) -> None:
    def on_context(event, ctx):
        filtered = [message for message in event.messages if getattr(message, "role", None) != "debug"]
        return ContextEventResult(messages=filtered)

    pi.on("context", on_context)
```

#### before_provider_headers

The event type and runner method exist, but the Python `pi_ai` request path does not expose assembled provider headers to the coding-agent wrapper. This hook is not emitted by the stock Python port.

#### before_provider_request

Fired after the provider-specific payload is built and before it is sent. Returning `None` keeps the payload unchanged; returning any other value replaces it.

```python
def pi_extension(pi) -> None:
    def on_before_provider_request(event, ctx):
        print(event.payload)
        return None

    pi.on("before_provider_request", on_before_provider_request)
```

#### after_provider_response

Fired after an HTTP response is received and before its stream body is consumed.

```python
def pi_extension(pi) -> None:
    def on_after_provider_response(event, ctx) -> None:
        if event.status == 429:
            print("rate limited", event.headers.get("retry-after"))

    pi.on("after_provider_response", on_after_provider_response)
```

### Model Events

#### model_select

Fired when the model changes through `AgentSession.set_model()` or model cycling.

```python
def pi_extension(pi) -> None:
    def on_model_select(event, ctx) -> None:
        previous = f"{event.previous_model.provider}/{event.previous_model.id}" if event.previous_model else "none"
        current = f"{event.model.provider}/{event.model.id}"
        if ctx.has_ui:
            ctx.ui.notify(f"Model changed: {previous} -> {current}", "info")

    pi.on("model_select", on_model_select)
```

#### thinking_level_select

Fired when the thinking level changes.

```python
def pi_extension(pi) -> None:
    def on_thinking_level_select(event, ctx) -> None:
        if ctx.has_ui:
            ctx.ui.set_status("thinking", f"thinking: {event.level}")

    pi.on("thinking_level_select", on_thinking_level_select)
```

### Tool Events

#### tool_call

Fired before a tool executes. It can mutate `event.input` in place or block the call.

```python
from pi_coding_agent.core.extensions.types import ToolCallEventResult


def pi_extension(pi) -> None:
    def on_tool_call(event, ctx):
        if event.tool_name == "bash":
            command = event.input.get("command")
            if isinstance(command, str):
                event.input["command"] = f"source ~/.profile\n{command}"
                if "rm -rf" in command:
                    return ToolCallEventResult(
                        block=True,
                        reason="Dangerous command",
                        terminate=True,
                    )
        return None

    pi.on("tool_call", on_tool_call)
```

The TypeScript helper `isToolCallEventType` is not available. Check `event.tool_name` and validate the argument shape yourself.

#### Typing custom tool input

Python extensions use normal type annotations and JSON Schema dictionaries. There is no TypeBox `Static` type or TypeScript generic narrowing.

```python
from typing import TypedDict


class MyToolInput(TypedDict):
    action: str
```

#### tool_result

Fired after tool execution and before the final result is persisted. Return `ToolResultEventResult` with partial patches.

```python
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ToolResultEventResult


def pi_extension(pi) -> None:
    def on_tool_result(event, ctx):
        if event.tool_name != "bash" or event.is_error:
            return None
        return ToolResultEventResult(
            content=[TextContent(text="bash completed")],
            details=event.details,
            is_error=False,
        )

    pi.on("tool_result", on_tool_result)
```

### User Bash Events

#### user_bash

`UserBashEvent` and `UserBashEventResult` exist in `core.extensions.types`, but `user_bash` is not emitted by the Python port. The `!` and `!!` interactive bash feature is ported, but the extension interception hook is not wired.

### Input Events

#### input

Fired when user input is received, after extension commands are checked and before skill/template expansion.

```python
from pi_coding_agent.core.extensions.types import InputEventResult


def pi_extension(pi) -> None:
    def on_input(event, ctx):
        if event.streaming_behavior == "steer":
            return InputEventResult(action="continue")
        if event.text.startswith("?quick "):
            return InputEventResult(action="transform", text=f"Respond briefly: {event.text[7:]}")
        if event.text == "ping":
            if ctx.has_ui:
                ctx.ui.notify("pong", "info")
            return InputEventResult(action="handled")
        return InputEventResult(action="continue")

    pi.on("input", on_input)
```

Results:
- `continue` - pass through unchanged
- `transform` - modify text/images, then continue to expansion
- `handled` - skip agent processing

See [input_transform_streaming.py](../examples/extensions/input_transform_streaming.py).

## ExtensionContext

All handlers receive `ctx: ExtensionContext`.

### ctx.ui

`ctx.ui` has the protocol methods `select`, `confirm`, `input`, `notify`, `set_status`, `set_title`, `get_tools_expanded`, `set_tools_expanded`, and `set_widget`.

Interactive mode binds a real context, so `ctx.has_ui` is `True` there. Print, JSON and RPC modes have no UI, so `ctx.ui` is a no-op context and `ctx.has_ui` is `False`. Guard UI calls with `if ctx.has_ui:` so an extension works in every mode.

#### ctx.ui.set_widget

```python
ctx.ui.set_widget("build-status", ["Build: passing"])
ctx.ui.set_widget("build-status", ["Build: passing"], "belowEditor")
ctx.ui.set_widget("build-status", None)  # remove
```

Widgets are keyed. Setting an existing key replaces the widget, including when the placement changes, so a key never appears twice. `None` removes it. `placement` is `"aboveEditor"` (default) or `"belowEditor"`.

Content is either a list of lines or a factory `(tui, theme) -> Component` for a live component. A line list is capped at 10 lines and the excess is replaced by a `... (widget truncated)` marker, so a runaway widget cannot push the prompt off screen.

Widgets belong to the session that created them: switching sessions clears them.

### ctx.mode

Current run mode: `"tui"`, `"rpc"`, `"json"`, or `"print"`. Interactive mode sets `"tui"`; the other modes leave the default runner mode unless a custom host sets it.

### ctx.hasUI

Python name: `ctx.has_ui`. It is `True` only when a host supplied a real `ExtensionUIContext`.

### ctx.cwd

Current working directory.

Use `CONFIG_DIR_NAME` instead of hardcoding `.pi` when constructing project-local config paths.

```python
from pathlib import Path

from pi_coding_agent.core.config import CONFIG_DIR_NAME
from pi_coding_agent.core.extensions.loader import ExtensionAPI


def pi_extension(pi: ExtensionAPI) -> None:
    def on_session_start(event, ctx) -> None:
        project_config_path = Path(ctx.cwd) / CONFIG_DIR_NAME / "my-extension.json"
        print(project_config_path)

    pi.on("session_start", on_session_start)
```

### ctx.isProjectTrusted()

Python name: `ctx.is_project_trusted()`. Returns whether project-local trust is active for the current session context.

### ctx.sessionManager

Python name: `ctx.session_manager`. It gives read access to session state. See [Session Format](session-format.md).

```python
def read_session(ctx) -> None:
    entries = ctx.session_manager.get_entries()
    branch = ctx.session_manager.get_branch()
    leaf_id = ctx.session_manager.get_leaf_id()
    print(len(entries), len(branch), leaf_id)
```

### ctx.modelRegistry / ctx.model / ctx.thinkingLevel / ctx.scopedModels

`ctx.model_registry` is not available in the Python port. Use `ctx.model`, `ctx.thinking_level`, and `ctx.scoped_models`.

### ctx.signal

The current agent abort signal, or `None` when no agent turn is active. Pass it to helpers that accept `AbortSignal`.

```python
async def run_git(pi, ctx):
    return await pi.exec("git", ["status"], None)
```

### ctx.isIdle() / ctx.abort() / ctx.hasPendingMessages()

Python names: `ctx.is_idle()`, `ctx.abort()`, and `ctx.has_pending_messages()`.

### ctx.shutdown()

Requests that the host exit. Interactive mode registers a handler, so this leaves the CLI; it waits for the current turn to finish rather than cutting off a response in flight.

A host that registers no handler (print or JSON mode, or an SDK embedder) leaves this a no-op: only the host knows what shutting down means for it.

### ctx.getContextUsage()

Python name: `ctx.get_context_usage()`.

```python
def maybe_report_usage(ctx) -> None:
    usage = ctx.get_context_usage()
    if usage is not None and usage.tokens is not None:
        print(f"tokens: {usage.tokens}")
```

### ctx.compact()

Trigger compaction without awaiting completion. Pass `CompactOptions`.

```python
from pi_coding_agent.core.extensions.types import CompactOptions


def trigger(ctx) -> None:
    ctx.compact(CompactOptions(custom_instructions="Focus on recent changes"))
```

See [trigger_compact.py](../examples/extensions/trigger_compact.py).

### ctx.getSystemPrompt()

Python name: `ctx.get_system_prompt()`.

```python
def print_prompt_length(ctx) -> None:
    prompt = ctx.get_system_prompt()
    print(f"System prompt length: {len(prompt)}")
```

## ExtensionCommandContext

Command handlers receive `ExtensionCommandContext`, which extends `ExtensionContext`.

### ctx.getSystemPromptOptions()

Python name: `ctx.get_system_prompt_options()`.

```python
def context_paths(ctx) -> list[str]:
    options = ctx.get_system_prompt_options()
    return [file.path for file in (options.context_files or [])]
```

Treat this as sensitive extension-local data.

### ctx.waitForIdle()

Python name: `ctx.wait_for_idle()`.

```python
def pi_extension(pi) -> None:
    async def handler(args: str, ctx) -> None:
        await ctx.wait_for_idle()
        if ctx.has_ui:
            ctx.ui.notify("Agent is idle", "info")

    pi.register_command("after-idle", handler=handler, description="Wait until idle")
```

### ctx.newSession(options?)

Not available on `ExtensionCommandContext` in the Python port.

### ctx.fork(entryId, options?)

Not available on `ExtensionCommandContext` in the Python port.

### ctx.navigateTree(targetId, options?)

Not available on `ExtensionCommandContext` in the Python port.

### ctx.switchSession(sessionPath, options?)

Not available on `ExtensionCommandContext` in the Python port.

### Session replacement lifecycle and footguns

`AgentSessionRuntime` emits the session replacement events, but command contexts do not expose `new_session`, `fork`, `switch_session`, `navigate_tree`, or `withSession` helpers. Do not document or depend on TypeScript's `withSession` pattern in Python extensions.

### ctx.reload()

Not available on `ExtensionCommandContext` in the Python port. The interactive `/reload` command reloads keybindings, settings, resources, and theme selection; it does not reload extension code.

## ExtensionAPI Methods

### pi.on(event, handler)

Subscribe to events.

```python
def pi_extension(pi) -> None:
    pi.on("agent_start", lambda event, ctx: print("started"))
```

### pi.registerTool(definition)

Python name: `pi.register_tool(definition)`.

```python
from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ToolDefinition


async def execute(tool_call_id, params, signal, on_update, ctx):
    if on_update is not None:
        on_update(AgentToolResult(content=[TextContent(text="Working...")]))
    return AgentToolResult(content=[TextContent(text="Done")], details={"ok": True})


def register(pi) -> None:
    pi.register_tool(
        ToolDefinition(
            name="my_tool",
            label="My Tool",
            description="What this tool does",
            prompt_snippet="Summarize or transform text according to action",
            prompt_guidelines=["Use my_tool when the user asks to summarize previously generated text."],
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "add"]},
                    "text": {"type": "string"},
                },
                "required": ["action"],
            },
            execute=execute,
        )
    )
```

`parameters` is JSON Schema, not TypeBox. `prompt_guidelines` bullets are appended flat, so each guideline must name the tool.

### pi.sendMessage(message, options?)

Python name: `pi.send_message(...)`. Bound by the CLI, so it reaches the session. A custom host that loads extensions itself must supply `ExtensionRuntimeActions.send_message`, or the call is a no-op.

### pi.sendUserMessage(content, options?)

Python name: `pi.send_user_message(...)`. Bound by the CLI, like `send_message`. Delivery mode is either a keyword (`deliver_as="followUp"`) or TypeScript's options dict (`{"deliverAs": "followUp"}`).

```python
def request_follow_up(pi) -> None:
    pi.send_user_message("Summarize the last result", deliver_as="followUp")
```

### pi.appendEntry(customType, data?)

Python name: `pi.append_entry(custom_type, data=None)`. Bound by the CLI.

### pi.setSessionName(name)

Python name: `pi.set_session_name(name)`. Bound by the CLI.

### pi.getSessionName()

Python name: `pi.get_session_name()`. Bound by the CLI; returns `None` before a session exists.

### pi.setLabel(entryId, label)

Not available on `ExtensionAPI` in the Python port.

### pi.registerCommand(name, options)

Python name: `pi.register_command(name, handler=..., description=..., get_argument_completions=...)`.

```python
def pi_extension(pi) -> None:
    async def stats(args: str, ctx) -> None:
        count = len(ctx.session_manager.get_entries())
        if ctx.has_ui:
            ctx.ui.notify(f"{count} entries", "info")
        else:
            print(f"{count} entries")

    pi.register_command("stats", handler=stats, description="Show session statistics")
```

If multiple extensions register the same command name, pi disambiguates them as `name:1`, `name:2`, and so on.

### pi.getCommands()

Not available on `ExtensionAPI` in the Python port.

### pi.registerMessageRenderer(customType, renderer)

Not available in the Python port.

### pi.registerMarkdownTransformer(transformer)

Not available in the Python port.

### pi.registerEntryRenderer(customType, renderer)

Not available in the Python port.

### pi.registerShortcut(shortcut, options)

Not available in the Python port.

### pi.registerFlag(name, options)

Not available in the Python port. Unknown CLI flags are parsed, but extensions cannot claim them through `register_flag`.

### pi.exec(command, args, options?)

Execute a command. `options` is `ExecOptions` from `pi_coding_agent.core.exec`.

```python
async def git_status(pi):
    result = await pi.exec("git", ["status", "--short"])
    print(result.stdout, result.stderr, result.code)
```

### pi.getActiveTools() / pi.getAllTools() / pi.setActiveTools(names)

Python has `pi.get_active_tools()` and `pi.set_active_tools(names)`. They require host-provided runtime actions to be effective. `pi.get_all_tools()` is not available on `ExtensionAPI`.

### pi.setModel(model)

Not available on `ExtensionAPI` in the Python port. `AgentSession.set_model()` exists for host code.

### pi.getThinkingLevel() / pi.setThinkingLevel(level)

Not available on `ExtensionAPI` in the Python port. Use `ctx.thinking_level` for read-only access.

### pi.events

`pi.events` works out of the box: a bus is created when the host supplies none, so extensions loaded together can coordinate through it. A host that wants to own the bus (to bridge it elsewhere, or to clear it) passes one as `ExtensionRuntimeActions.event_bus`.

```python
def pi_extension(pi) -> None:
    unsubscribe = pi.events.on("my:event", lambda data: print(data))
    pi.events.emit("my:event", {"ok": True})
    unsubscribe()
```

### pi.registerProvider(name, config)

Not available in the Python port.

### pi.unregisterProvider(name)

Not available in the Python port.

## State Management

Extensions with state should store branch-aware snapshots in tool result `details` and reconstruct from `ctx.session_manager.get_branch()`.

```python
from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ToolDefinition


def pi_extension(pi) -> None:
    items: list[str] = []

    def on_session_start(event, ctx) -> None:
        items.clear()
        for entry in ctx.session_manager.get_branch():
            message = getattr(entry, "message", None)
            if getattr(message, "role", None) == "toolResult" and getattr(message, "tool_name", None) == "my_tool":
                details = getattr(message, "details", None) or {}
                items[:] = list(details.get("items", []))

    async def execute(tool_call_id, params, signal, on_update, ctx):
        items.append(str(params.get("text", "new item")))
        return AgentToolResult(
            content=[TextContent(text="Added")],
            details={"items": list(items)},
        )

    pi.on("session_start", on_session_start)
    pi.register_tool(
        ToolDefinition(
            name="my_tool",
            label="My Tool",
            description="Store an item",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            execute=execute,
        )
    )
```

## Custom Tools

Register tools the LLM can call via `pi.register_tool()`. Tools appear in the model tool list and can contribute prompt snippets/guidelines.

Use `prompt_snippet` for a one-line entry in the default system prompt. Use `prompt_guidelines` to add tool-specific bullets while the tool is active.

If your custom tool accepts a path, normalize a leading `@` if needed, matching built-in tool behavior.

If your custom tool mutates files, use `with_file_mutation_queue()` so it participates in the same per-file queue as built-in `edit` and `write`.

```python
from pathlib import Path

from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.tools.file_mutation_queue import with_file_mutation_queue


async def replace_text(params, ctx):
    absolute_path = Path(ctx.cwd, params["path"]).resolve()

    async def mutate():
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        current = absolute_path.read_text(encoding="utf-8")
        next_text = current.replace(params["oldText"], params["newText"])
        absolute_path.write_text(next_text, encoding="utf-8")
        return AgentToolResult(content=[TextContent(text=f"Updated {params['path']}")], details={})

    return await with_file_mutation_queue(str(absolute_path), mutate)
```

### Tool Definition

```python
from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ToolDefinition


async def execute(tool_call_id, params, signal, on_update, ctx):
    if signal is not None and signal.aborted:
        return AgentToolResult(content=[TextContent(text="Cancelled")])

    if on_update is not None:
        on_update(AgentToolResult(content=[TextContent(text="Working...")], details={"progress": 50}))

    result = await ctx.ui.input("Optional input") if ctx.has_ui else None
    return AgentToolResult(
        content=[TextContent(text="Done")],
        details={"input": result},
        terminate=True,
    )


def make_tool() -> ToolDefinition:
    return ToolDefinition(
        name="my_tool",
        label="My Tool",
        description="What this tool does",
        prompt_snippet="List or add items in the project todo list",
        prompt_guidelines=[
            "Use my_tool for todo planning instead of direct file edits when the user asks for a task list."
        ],
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "add"]},
                "text": {"type": "string"},
            },
            "required": ["action"],
        },
        prepare_arguments=lambda args: args,
        execute=execute,
    )
```

**Usage accounting:** If a tool makes nested LLM calls, return their combined `Usage` as `usage` on `AgentToolResult`.

**Signaling errors:** Throw an exception from `execute` to mark tool execution as failed.

**Early termination:** Return `terminate=True` on `AgentToolResult` to hint that the automatic follow-up LLM call should be skipped after the current tool batch. This only takes effect when every finalized tool result in that batch is terminating.

**Argument preparation:** `prepare_arguments(args)` is optional. It runs before schema validation and `execute()`.

```python
def prepare_edit_arguments(args):
    if not isinstance(args, dict):
        return args
    old_text = args.get("oldText")
    new_text = args.get("newText")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return args
    edits = list(args.get("edits") or [])
    return {**args, "edits": [*edits, {"oldText": old_text, "newText": new_text}]}
```

### Overriding Built-in Tools

Extensions can override built-in tools (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`) by registering a tool with the same name. First registration per name wins in the runner, so explicit caller tools and load order matter.

```bash
uv run pp -e ./tool_override.py
```

Alternatively, use `--no-builtin-tools` to start without built-in tools while keeping extension tools enabled:

```bash
uv run pp --no-builtin-tools -e ./my_extension.py
```

Built-in per-tool bespoke renderers are not ported; built-in tools use generic rendering.

Built-in tool implementations:
- [read.py](../src/pi_coding_agent/tools/read.py) - `ReadToolDetails`
- [bash.py](../src/pi_coding_agent/tools/bash.py) - `BashToolDetails`
- [edit.py](../src/pi_coding_agent/tools/edit.py) - `EditToolDetails`
- [write.py](../src/pi_coding_agent/tools/write.py)
- [grep.py](../src/pi_coding_agent/tools/grep.py)
- [find.py](../src/pi_coding_agent/tools/find.py)
- [ls.py](../src/pi_coding_agent/tools/ls.py)

### Remote Execution

The full TypeScript remote-operations extension pattern is not ported. Python keeps operations dataclasses for `write` and `edit`, and `create_bash_tool()` supports `command_prefix`, `session_environment`, `expose_session_environment`, and `spawn_hook`.

```python
from pi_coding_agent.tools.bash import BashSpawnContext, create_bash_tool


def prefix_spawn(context: BashSpawnContext) -> BashSpawnContext:
    return BashSpawnContext(
        command=f"source ~/.profile\n{context.command}",
        cwd=context.cwd,
        env={**context.env, "CI": "1"},
    )


def make_bash(cwd: str):
    return create_bash_tool(cwd, spawn_hook=prefix_spawn)
```

`create_bash_tool()` can expose `PI_SESSION_ID`, `PI_SESSION_FILE`, `PI_PROVIDER`, `PI_MODEL`, and `PI_REASONING_LEVEL` when a host supplies `session_environment`.

### Output Truncation

**Tools MUST truncate their output** to avoid overwhelming the LLM context. The built-in limit is 50KB and 2000 lines. Use the exported truncation utilities.

```python
from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head


def make_result(output: str) -> AgentToolResult:
    truncation = truncate_head(output, max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES)
    result = truncation.content
    if truncation.truncated:
        result += (
            f"\n\n[Output truncated: {truncation.output_lines} of {truncation.total_lines} lines "
            f"({format_size(truncation.output_bytes)} of {format_size(truncation.total_bytes)}).]"
        )
    return AgentToolResult(content=[TextContent(text=result)], details={"truncation": truncation})
```

Use `truncate_head` for content where the beginning matters and `truncate_tail` for logs or command output.

### Multiple Tools

One extension can register multiple tools with shared state.

```python
def pi_extension(pi) -> None:
    connection = {"open": False}

    async def connect(tool_call_id, params, signal, on_update, ctx):
        connection["open"] = True
        from pi_agent.types import AgentToolResult
        from pi_ai.types import TextContent

        return AgentToolResult(content=[TextContent(text="connected")])

    pi.on("session_shutdown", lambda event, ctx: connection.update(open=False))
```

### Custom Rendering

Tool `render_call`, `render_result`, `renderShell`, message renderers, entry renderers, and markdown transformers are not available in the Python extension API. Built-in TUI components exist in `pi_tui`, but the extension renderer host is not wired.

#### renderCall

Not available in the Python port.

#### renderResult

Not available in the Python port.

#### Keybinding Hints

`key_hint`, `key_text`, and `raw_key_hint` exist in `pi_coding_agent.modes.interactive.components.keybinding_hints`, but custom extension renderers and custom components are not wired.

#### Best Practices

For Python tools, keep returned text compact, include enough detail for the LLM, store structured data in `details`, and truncate large output.

#### Fallback

When no custom renderer exists, the Python port uses generic tool result rendering.

### Dynamic Tool Loading

Dynamic tool loading through `pi.set_active_tools()` is bound by the CLI. A custom host that loads extensions itself must supply the action through `ExtensionRuntimeActions`.

#### Models with native deferred loading

Native deferred tool loading (`tool_reference`, `tool_search_call`, `tool_search_output`) is not implemented in the Python extension API.

#### Fallback behavior

When a host supports `set_active_tools`, newly active tools are applied through the normal active tool list on later requests.

#### Search tool example

```python
from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ToolDefinition

SEARCHABLE_TOOL_NAMES = {"lookup_weather", "search_issues"}


async def lookup_weather(tool_call_id, params, signal, on_update, ctx):
    return AgentToolResult(content=[TextContent(text=f"Weather for {params['city']}: sunny")], details={})


async def search_issues(tool_call_id, params, signal, on_update, ctx):
    return AgentToolResult(content=[TextContent(text=f"No issues matching {params['query']}")], details={})


def pi_extension(pi) -> None:
    async def search_tools(tool_call_id, params, signal, on_update, ctx):
        active = pi.get_active_tools()
        matches = [name for name in SEARCHABLE_TOOL_NAMES if name not in active]
        pi.set_active_tools([*active, *matches])
        return AgentToolResult(
            content=[TextContent(text=f"Loaded tools: {', '.join(matches) or 'none'}")],
            details={"matches": matches},
        )

    pi.register_tool(
        ToolDefinition(
            name="lookup_weather",
            label="Lookup Weather",
            description="Look up the current weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            execute=lookup_weather,
        )
    )
    pi.register_tool(
        ToolDefinition(
            name="search_issues",
            label="Search Issues",
            description="Search project issues by keyword",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            execute=search_issues,
        )
    )
    pi.register_tool(
        ToolDefinition(
            name="search_tools",
            label="Search Tools",
            description="Search for and enable tools relevant to a task",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            execute=search_tools,
        )
    )
```

## Custom UI

Interactive mode binds a real `ExtensionUIContext`, so dialogs and `set_widget` work there. In print, JSON and RPC modes the UI methods are no-ops.

Not ported: custom header/footer components, editor control, terminal input listeners, autocomplete providers and theme accessors.

**For custom components, see [tui.md](tui.md)** for the underlying TUI component library. A widget factory can return any of those components.

### Dialogs

`select`, `confirm`, `input`, and `notify` are wired in interactive mode and open the corresponding dialog in place of the prompt until the user answers.

```python
async def maybe_confirm(ctx) -> bool:
    if not ctx.has_ui:
        return False
    return await ctx.ui.confirm("Confirm", "Continue?")
```

#### Timed Dialogs with Countdown

Dialog timeout options are not part of the Python `ExtensionUIContext` protocol.

#### Manual Dismissal with AbortSignal

The Python `ExtensionUIContext` protocol takes no `AbortSignal`, so a dialog is dismissed by the user, not by the extension.

### Widgets, Status, and Footer

`set_status`, `set_title` and `set_widget` are available (see [ctx.ui.set_widget](#ctxuiset_widget)). Custom headers, custom footers, working indicators, editor text, paste handling, autocomplete providers, editor replacement, and theme switching through extensions are not ported.

### Autocomplete Providers

Extension-provided autocomplete providers are not available in the Python port.

### Custom Components

`ctx.ui.custom()` is not available in the Python port.

#### Overlay Mode (Experimental)

Extension overlays are not available in the Python port.

### Custom Editor

Extension-provided custom editors are not available in the Python port.

### Message and Entry Rendering

`register_message_renderer` and `register_entry_renderer` are not available in the Python port.

### Theme Colors

Theme objects exist under `pi_coding_agent.modes.interactive.theme`, but extensions cannot currently register custom renderers that receive them.

## Error Handling

- Extension handler errors are logged on the runner and later handlers continue.
- `tool_call` errors are fail-safe and block the tool through the agent loop.
- Tool `execute` errors must be signaled by raising an exception.

## Mode Behavior

| Mode | `ctx.mode` | `ctx.has_ui` | Notes |
|------|------------|--------------|-------|
| Interactive | `"tui"` | `True` | Widgets, dialogs, status, title and tools-expanded are wired |
| RPC | `"rpc"` if a host binds it | Host-dependent | Legacy stdio RPC mode is not ported |
| JSON | `"json"` if a host binds it | `False` unless supplied | Event stream to stdout |
| Print | `"print"` | `False` | Default runner context |

Use `ctx.has_ui` before any UI method.

## Examples Reference

All Python examples in [examples/extensions/](../examples/extensions/).

| Example | Description | Key APIs |
|---------|-------------|----------|
| `git_merge_and_resolve.py` | Fetch and merge upstream after agent turns, then send conflict follow-up messages | `on("agent_end")`, `exec`, `send_user_message` |
| `input_transform_streaming.py` | Transform input unless it is mid-stream steering | `on("input")`, `InputEventResult`, `streaming_behavior`, `exec` |
| `trigger_compact.py` | Trigger compaction by threshold or slash command | `on("turn_end")`, `register_command`, `CompactOptions`, `ctx.compact()` |
| `plan_mode/utils.py` | Pure plan-mode helpers only | No extension entry point; full plan-mode UI/shortcut extension is not ported |

TypeScript examples such as `snake.ts`, `space-invaders.ts`, `overlay-*`, `custom-provider-*`, `ssh.ts`, `tool-override.ts`, `truncated-tool.ts`, and most UI examples have no Python example counterpart yet.

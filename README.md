# pi-coding-agent

Python port of pi's terminal coding harness. The installed CLI entry point is
`pp`; the lower-level OAuth helper is `pp-ai`; evals run with `pp-evals`.
The package requires Python >= 3.11 and is developed as a uv workspace.

Pi is a minimal terminal coding harness. Adapt pi to your workflows without
forking internals. Extend it with Python [Extensions](#extensions), [Skills](#skills),
[Prompt Templates](#prompt-templates), and [Themes](#themes). Bundle those
resources in [Pi Packages](#pi-packages) and share them by git or local path.

Pi ships with useful defaults but keeps the core small. It runs in interactive
TUI mode, print mode, JSON event mode, a Python SDK, the stdio JSONL RPC mode,
and a Unix-socket RPC stack.

## Share your OSS coding agent sessions

Sessions are stored as JSONL files, so external tooling can process them. The
TypeScript README's `pi-share-hf` workflow is not part of this Python port.

## Table of Contents

- [Quick Start](#quick-start)
- [Providers & Models](#providers--models)
- [Interactive Mode](#interactive-mode)
  - [Editor](#editor)
  - [Commands](#commands)
  - [Keyboard Shortcuts](#keyboard-shortcuts)
  - [Message Queue](#message-queue)
- [Sessions](#sessions)
  - [Branching](#branching)
  - [Compaction](#compaction)
- [Settings](#settings)
- [Context Files](#context-files)
- [Customization](#customization)
  - [Prompt Templates](#prompt-templates)
  - [Skills](#skills)
  - [Extensions](#extensions)
  - [Themes](#themes)
  - [Pi Packages](#pi-packages)
- [Programmatic Usage](#programmatic-usage)
- [Philosophy](#philosophy)
- [CLI Reference](#cli-reference)

---

## Quick Start

From a source checkout:

```bash
uv sync --all-packages
uv run pp --help
```

From an installed wheel or package index:

```bash
python -m pip install pi-coding-agent
pp --help
```

Authenticate with an API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run pp
```

Or use interactive login:

```text
/login  # Then select provider
```

Then talk to pi. By default, pi gives the model four tools: `read`, `bash`,
`edit`, and `write`. Additional built-in tools (`grep`, `find`, `ls`) can be
enabled with `--tools`.

**Platform notes:** [Windows](docs/windows.md) | [Termux (Android)](docs/termux.md) | [tmux](docs/tmux.md) | [Terminal setup](docs/terminal-setup.md) | [Shell aliases](docs/shell-aliases.md)

---

## Providers & Models

The Python port ships a static built-in model catalog and reads custom overlays
from `~/.pi/agent/models.json`. Remote model-catalog refresh and `pp update
--models` are not ported.

**OAuth login is ported for:**

- Anthropic
- GitHub Copilot
- Kimi For Coding
- OpenRouter
- Radius
- xAI

**API-key providers in the catalog:**

- Amazon Bedrock
- Ant Ling
- Anthropic
- Azure OpenAI
- Baseten
- Cerebras
- Cloudflare AI Gateway
- Cloudflare Workers AI
- DeepSeek
- Fireworks
- GitHub Copilot
- Google
- Google Vertex AI
- Groq
- Hugging Face
- Kimi For Coding
- MiniMax
- MiniMax CN
- Mistral
- Moonshot AI
- Moonshot AI CN
- NVIDIA
- OpenAI
- OpenAI Codex
- OpenCode Zen
- OpenCode Go
- OpenRouter
- Qwen Token Plan
- Qwen Token Plan CN
- Qwen Token Plan Individual
- Radius
- Together
- Vercel AI Gateway
- xAI
- Xiaomi
- Xiaomi Token Plan AMS
- Xiaomi Token Plan CN
- Xiaomi Token Plan SGP
- Z.AI
- Z.AI Coding CN

Amazon Bedrock and OpenAI Codex models remain discoverable, but their streaming
APIs are not ported and raise `NotImplementedError` when used.

Select a model with `/model` in the TUI, or use CLI options:

```bash
uv run pp --provider anthropic --model claude-sonnet-4-5
uv run pp --model anthropic/claude-sonnet-4-5
uv run pp --model claude-sonnet-4-5:high
uv run pp --list-models sonnet
```

Custom providers can be added in `~/.pi/agent/models.json` when they use a
ported API shape (`openai-completions`, `openai-responses`,
`anthropic-messages`, or `google-generative-ai`). See [docs/models.md](docs/models.md)
and [docs/custom-provider.md](docs/custom-provider.md).

The llama.cpp local-model extension from the TypeScript package is not ported.

---

## Interactive Mode

Run `pp` on a TTY with no `-p` flag to start the interactive TUI.

The interface from top to bottom:

- **Startup header** - Shows shortcuts, context files, prompt templates,
  skills, and extensions
- **Messages** - User messages, assistant responses, tool calls and results,
  notifications, errors, and generic custom entries
- **Editor** - Where you type; border color indicates thinking level
- **Footer** - Working directory, branch, token usage, context usage, current
  model, and extension status text

Both TUI modes are ported: `regular` and `fullscreen`. Runtime switching between
them is not ported; changing the setting applies on the next start.

Extension widgets, custom header/footer/editor replacement, extension-driven
dialogs, and terminal input listeners from the TypeScript UI host are not ported.

### Editor

| Feature | How |
|---------|-----|
| File reference | Type `@` to fuzzy-search project files |
| Path completion | Tab completion is provided by the TUI editor |
| Multi-line | Use the configured TUI newline keybinding |
| External editor | Ctrl+G opens `externalEditor`, `$VISUAL`, `$EDITOR`, Notepad on Windows, or `nano` elsewhere |
| Clipboard | Ctrl+V pastes image/text on Unix-like systems; Alt+V on Windows |
| Bash commands | `!command` runs and sends output to the model, `!!command` runs without sending |

Standard editing keybindings are loaded from `~/.pi/agent/keybindings.json`.
See [docs/keybindings.md](docs/keybindings.md).

### Commands

Type `/` in the editor to trigger commands. Extensions can register custom
commands, skills are available as `/skill:name`, and prompt templates expand as
`/templatename`.

| Command | Description |
|---------|-------------|
| `/login`, `/logout` | Manage provider credentials |
| `/model` | Switch models |
| `/scoped-models` | Enable/disable models for Ctrl+P cycling |
| `/settings` | Thinking level, theme, message delivery, transport |
| `/resume` | Pick from previous sessions |
| `/new` | Start a new session |
| `/name <name>` | Set session display name |
| `/session` | Show session info |
| `/tree` | Jump to any point in the session and continue from there |
| `/trust` | Save project trust decision for future sessions |
| `/fork` | Create a new session from a previous user message |
| `/clone` | Duplicate the current active branch into a new session |
| `/compact [prompt]` | Manually compact context, optional custom instructions |
| `/copy` | Copy last assistant message to clipboard |
| `/export [file]` | Export session; JSONL works, HTML export is not ported |
| `/import <file>` | Import and resume a session from a JSONL file |
| `/share` | Starts gist sharing, but fails because HTML document assembly is not ported |
| `/reload` | Reload keybindings, extensions, skills, prompts, themes, and context files |
| `/hotkeys` | Show all keyboard shortcuts |
| `/changelog` | Display version history |
| `/quit` | Quit pi |

### Keyboard Shortcuts

See `/hotkeys` for the full list. Customize via `~/.pi/agent/keybindings.json`.

**Commonly used:**

| Key | Action |
|-----|--------|
| Ctrl+C | Clear editor |
| Ctrl+D | Exit when editor is empty |
| Escape | Cancel/abort |
| Ctrl+L | Open model selector |
| Ctrl+P / Shift+Ctrl+P | Cycle scoped models forward/backward |
| Shift+Tab | Cycle thinking level |
| Ctrl+O | Collapse/expand tool output |
| Ctrl+T | Collapse/expand thinking blocks |
| Ctrl+X | Copy message to clipboard |
| Alt+Enter | Queue follow-up message |
| Alt+Up | Restore queued messages |

### Message Queue

Submit messages while the agent is working:

- **Enter** queues a steering message, delivered after the current assistant
  turn finishes its tool calls
- **Alt+Enter** queues a follow-up message, delivered after the agent finishes
  all current work
- **Escape** aborts and restores queued messages to the editor
- **Alt+Up** retrieves queued messages back to the editor

Configure delivery in [settings](docs/settings.md): `steeringMode` and
`followUpMode` can be `"one-at-a-time"` or `"all"`. `transport` selects
provider transport preference (`"sse"`, `"websocket"`, `"websocket-cached"`,
or `"auto"`) where supported.

---

## Sessions

Sessions are JSONL files with a tree structure. Each entry has an `id` and
`parentId`, enabling in-place branching without creating new files. See
[docs/session-format.md](docs/session-format.md).

### Management

Sessions auto-save to `~/.pi/agent/sessions/` organized by working directory.

```bash
uv run pp -c                  # Continue most recent session
uv run pp -r                  # Browse and select from past sessions
uv run pp --no-session        # Ephemeral mode
uv run pp --name "my task"    # Set session display name at startup
uv run pp --session <path|id> # Use specific session file or ID
uv run pp --fork <path|id>    # Fork a session file or ID into a new session
```

Use `/session` in interactive mode to see the current session ID before reusing
it with `--session <id>` or `--fork <id>`.

### Branching

**`/tree`** - Navigate the session tree in-place. Select any previous point,
continue from there, and switch between branches. All history is preserved in a
single file.

- Search by typing, fold/unfold and jump between branches with Ctrl+Left/Ctrl+Right or Alt+Left/Alt+Right
- Filter modes: default, no-tools, user-only, labeled-only, all
- Press Ctrl+X to copy the selected message
- Press Shift+L to label entries and Shift+T to toggle label timestamps

**`/fork`** creates a new session file from a previous user message on the
active branch.

**`/clone`** duplicates the current active branch into a new session file at the
current position.

**`--fork <path|id>`** forks an existing session file or partial session UUID
from the CLI.

### Compaction

Long sessions can exhaust context windows. Compaction summarizes older messages
while keeping recent ones.

**Manual:** `/compact` or `/compact <custom instructions>`

**Automatic:** Enabled by default. It triggers on context overflow or near the
context limit. Configure via `/settings` or `settings.json`.

Compaction is lossy. The full history remains in the JSONL file; use `/tree` to
revisit it. Extension hooks for custom compaction are ported. See
[docs/compaction.md](docs/compaction.md).

---

## Settings

Use `/settings` to modify common options, or edit JSON files directly:

| Location | Scope |
|----------|-------|
| `~/.pi/agent/settings.json` | Global |
| `.pi/settings.json` | Project, loaded only when the project is trusted |

Important settings keys include `defaultProvider`, `defaultModel`,
`defaultThinkingLevel`, `steeringMode`, `followUpMode`, `transport`,
`externalEditor`, `defaultProjectTrust`, `enableInstallTelemetry`,
`versionCheckPackage`, `tuiMode`, `fullscreenExitOutput`, `fullscreenScrollbar`,
`enabledModels`, `images.autoResize`, and `images.blockImages`.

The `markdown.mermaid` setting exists, but Mermaid diagram rendering is not
ported.

See [docs/settings.md](docs/settings.md) for all options.

### Project Trust

On interactive startup, pi asks before trusting a project folder that contains
project-local settings, resources, or project `.agents/skills` and has no saved
decision for the folder or a parent folder in `~/.pi/agent/trust.json`.
Trusting a project allows pi to load `.pi/settings.json`, `.pi` resources, and
project extensions.

Non-interactive modes (`-p` and `--mode json`) do not show a trust prompt.
Without a saved trust decision, they use `defaultProjectTrust`: `ask` and
`never` ignore project resources; `always` trusts them. Pass `--approve` or
`--no-approve` to override project trust for one run.

Use `/trust` in interactive mode to save a project trust decision for future
sessions. It writes `~/.pi/agent/trust.json`; restart pi for newly trusted
project resources to load.

### Telemetry and update checks

Pi has two separate startup features:

- **Update check:** checks PyPI, not `pi.dev`. Configure the distribution
  with `PI_VERSION_CHECK_PACKAGE` or `versionCheckPackage`. Disable with
  `PI_SKIP_VERSION_CHECK=1`.
- **Install/update telemetry:** controlled by `enableInstallTelemetry` or
  `PI_TELEMETRY`. This also controls optional provider attribution headers for
  OpenRouter, Cloudflare, and direct NVIDIA requests.

Use `--offline` or `PI_OFFLINE=1` to disable startup network operations,
including update checks, package update checks, and install/update telemetry.

---

## Context Files

Pi loads `AGENTS.md` or `CLAUDE.md` at startup from:

- `~/.pi/agent/AGENTS.md`
- Parent directories, walking up from cwd
- Current directory

If a directory contains `AGENTS.override.md`, pi loads it instead of
`AGENTS.md` or `CLAUDE.md` from that directory. Context files from other
directories are still concatenated.

Disable context file loading with `--no-context-files` or `-nc`.

### System Prompt

Replace the default system prompt with `.pi/SYSTEM.md` or
`~/.pi/agent/SYSTEM.md`. Append without replacing via `APPEND_SYSTEM.md`.

---

## Customization

### Prompt Templates

Reusable prompts as Markdown files. Type `/name` to expand.

```markdown
<!-- ~/.pi/agent/prompts/review.md -->
Review this code for bugs, security issues, and performance problems.
Focus on: $ARGUMENTS
```

Place in `~/.pi/agent/prompts/`, `.pi/prompts/`, or a [pi package](#pi-packages).
See [docs/prompt-templates.md](docs/prompt-templates.md).

### Skills

On-demand capability packages following the Agent Skills layout. Invoke via
`/skill:name` or let the agent load them automatically.

```markdown
---
name: my-skill
description: Use this skill when the user asks about X.
---

# My Skill

## Steps
1. Do this
2. Then that
```

Place in `~/.pi/agent/skills/`, `~/.agents/skills/`, `.pi/skills/`, or
`.agents/skills/` from `cwd` up through parent directories, or in a pi package.
See [docs/skills.md](docs/skills.md).

### Extensions

Python modules that extend pi with custom tools, commands, and event handlers.
An extension file exports a callable named `pi_extension`.

```python
from typing import Any

from pi_agent.types import AgentToolResult, AgentToolUpdateCallback
from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal
from pi_coding_agent.core.extensions import ExtensionAPI, ExtensionContext, ToolDefinition


async def echo_tool(
    tool_call_id: str,
    params: dict[str, Any],
    signal: AbortSignal | None,
    on_update: AgentToolUpdateCallback | None,
    ctx: ExtensionContext,
) -> AgentToolResult:
    text = str(params.get("text", ""))
    return AgentToolResult(content=[TextContent(text=text)])


def pi_extension(pi: ExtensionAPI) -> None:
    pi.register_tool(
        ToolDefinition(
            name="echo",
            label="Echo",
            description="Echo text back.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
            execute=echo_tool,
        )
    )
```

The ported extension API supports tool registration, command registration,
event handlers, extension events, and headless-safe UI methods (`select`,
`confirm`, `input`, `notify`, `set_status`, `set_title`). Provider registration,
custom CLI flags, shortcut registration, markdown/message/entry renderer
registration, and the TypeScript extension UI host are not ported.

Place extensions in `~/.pi/agent/extensions/`, `.pi/extensions/`, or a pi
package. Python package manifests use `pi.json`, not `package.json`. See
[docs/extensions.md](docs/extensions.md) and [examples/extensions/](examples/extensions/).

### Themes

Built-in themes: `dark`, `light`. Custom theme loading is ported; live
file-watcher hot reload is not ported.

Place themes in `~/.pi/agent/themes/` or provide them through a package. Project
`.pi/themes/` and package themes are resolved by the package manager; the
interactive runtime currently initializes built-in and global custom-directory
themes directly. See [docs/themes.md](docs/themes.md).

### Pi Packages

Bundle and share extensions, skills, prompts, and themes by git repository or
local path.

> **Security:** Pi packages run with full system access. Extensions execute
> arbitrary Python code, and skills can instruct the model to perform actions.
> Review third-party code before installing it.

```bash
uv run pp install git:github.com/user/repo
uv run pp install git:github.com/user/repo@v1
uv run pp install https://github.com/user/repo
uv run pp install ssh://git@github.com/user/repo
uv run pp install ./local/path
uv run pp remove git:github.com/user/repo
uv run pp uninstall ./local/path
uv run pp list
uv run pp update
uv run pp update git:github.com/user/repo
uv run pp config
```

Use `-l` or `--local` with `install`, `remove`, or `uninstall` for project-local
settings. Git packages install to `~/.pi/agent/git/` or `.pi/git/`; local-path
packages are referenced in place. `npm:` sources, package dependency install,
self-update, `update --all`, `update --self`, `update --models`, and
`update --extension` are not ported.

Create a package with a `pi.json` manifest:

```json
{
  "extensions": ["./extensions"],
  "skills": ["./skills"],
  "prompts": ["./prompts"],
  "themes": ["./themes"]
}
```

Without a manifest, pi auto-discovers conventional directories:
`extensions/`, `skills/`, `prompts/`, and `themes/`. See [docs/packages.md](docs/packages.md).

---

## Programmatic Usage

### SDK

```python
import asyncio

from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager


async def main() -> None:
    result = await create_agent_session(CreateAgentSessionOptions(session_manager=SessionManager.in_memory()))
    try:
        await result.session.prompt("What files are in the current directory?")
        print(result.session.get_last_assistant_text())
    finally:
        result.session.dispose()


asyncio.run(main())
```

For custom setup, pass `model_runtime`, `settings_manager`, `resource_loader`,
`custom_tools`, or loaded `extensions` in `CreateAgentSessionOptions`. See
[docs/sdk.md](docs/sdk.md) and [examples/sdk/](examples/sdk/).

### RPC Mode

`pp --mode rpc` runs the agent headless behind a JSON protocol on
stdin/stdout: a host writes one command per line and reads responses and events
back. See [docs/rpc-stdio.md](docs/rpc-stdio.md).

For multi-client access over a Unix socket, the `pi_server` / `pi_client` /
`pi_protocol` stack is documented in [docs/rpc.md](docs/rpc.md).

---

## Philosophy

Pi is extensible so it does not have to dictate your workflow. Features that
other tools bake in can be built with extensions, skills, or packages.

**No MCP by default.** Build CLI tools with READMEs or add MCP support through
an extension.

**No built-in sub-agents.** Spawn pi instances yourself or build the workflow as
an extension or package.

**No permission popups.** Run in a container or add a confirmation flow through
an extension.

**No plan mode.** Write plans to files or build plan mode as an extension.

**No built-in to-dos.** Use a TODO file or build a task system as an extension.

**No background bash.** Use tmux for direct observability and control.

---

## CLI Reference

```bash
pp [options] [@files...] [messages...]
```

### Package Commands

```bash
pp install <source> [-l]
pp remove <source> [-l]
pp uninstall <source> [-l]
pp update [source]
pp list
pp config
```

`pp config` and project-local package commands accept `--approve` and
`--no-approve` for one-command project trust overrides. `pp update` never
prompts for project trust.

### Modes

| Flag | Description |
|------|-------------|
| (default) | Interactive mode on a TTY |
| `-p`, `--print` | Print response and exit |
| `--mode json` | Output events as JSON lines |
| `--mode rpc` | Headless JSON-line protocol on stdin/stdout |
| `--export <file>` | Parsed; the non-interactive export driver is not wired |

In print mode, pi also reads piped stdin and merges it into the initial prompt:

```bash
cat README.md | pp -p "Summarize this text"
```

### Model Options

| Option | Description |
|--------|-------------|
| `--provider <name>` | Provider name |
| `--model <pattern>` | Model pattern or ID; supports `provider/id` and optional `:<thinking>` |
| `--api-key <key>` | API key for this process |
| `--thinking <level>` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `--models <patterns>` | Comma-separated patterns for Ctrl+P cycling |
| `--list-models [search]` | List available models |

### Session Options

| Option | Description |
|--------|-------------|
| `-c`, `--continue` | Continue most recent session |
| `-r`, `--resume` | Browse and select session |
| `--session <path\|id>` | Use specific session file or partial UUID |
| `--session-id <id>` | Use exact project session ID, creating it if missing |
| `--fork <path\|id>` | Fork session file or partial UUID into a new session |
| `--session-dir <dir>` | Custom session storage directory |
| `--no-session` | Ephemeral mode |
| `--name <name>`, `-n <name>` | Set session display name at startup |

### Tool Options

| Option | Description |
|--------|-------------|
| `--tools <list>`, `-t <list>` | Allowlist tool names |
| `--exclude-tools <list>`, `-xt <list>` | Disable tool names |
| `--no-builtin-tools`, `-nbt` | Disable built-in tools by default but keep extension/custom tools enabled |
| `--no-tools`, `-nt` | Disable all tools by default |

Available built-in tools: `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`.

### Resource Options

| Option | Description |
|--------|-------------|
| `-e`, `--extension <path>` | Load an extension file; repeatable |
| `--no-extensions`, `-ne` | Disable extension discovery |
| `--skill <path>` | Load skill file or directory; repeatable |
| `--no-skills`, `-ns` | Disable skill discovery |
| `--prompt-template <path>` | Load prompt template file or directory; repeatable |
| `--no-prompt-templates`, `-np` | Disable prompt template discovery |
| `--theme <path>` | Load theme file or directory; repeatable |
| `--no-themes` | Disable theme discovery |
| `--no-context-files`, `-nc` | Disable context file discovery |

### Other Options

| Option | Description |
|--------|-------------|
| `--system-prompt <text>` | Replace default prompt |
| `--append-system-prompt <text>` | Append text or file contents to the system prompt; repeatable |
| `--tui-mode <mode>` | `regular` or `fullscreen` |
| `--verbose` | Force verbose startup |
| `-a`, `--approve` | Trust project-local files for this run |
| `-na`, `--no-approve` | Ignore project-local files for this run |
| `--offline` | Disable startup network operations |
| `-h`, `--help` | Show help |
| `-v`, `--version` | Show version |

### File Arguments

Prefix files with `@` to include them in the message:

```bash
pp @prompt.md "Answer this"
pp -p @screenshot.png "What's in this image?"
pp @code.py @test_code.py "Review these files"
```

### Examples

```bash
# Interactive with initial prompt
pp "List all .py files in src/"

# Non-interactive
pp -p "Summarize this codebase"

# Non-interactive with piped stdin
cat README.md | pp -p "Summarize this text"

# Named one-shot session
pp --name "release audit" -p "Audit this repository"

# Different model
pp --provider openai --model gpt-4o "Help me refactor"

# Model with provider prefix
pp --model openai/gpt-4o "Help me refactor"

# Model with thinking level shorthand
pp --model claude-sonnet-4-5:high "Solve this complex problem"

# Limit model cycling
pp --models "anthropic/*,*sonnet*"

# Read-only mode
pp --tools read,grep,find,ls -p "Review the code"

# Disable one tool while keeping the rest available
pp --exclude-tools bash

# High thinking level
pp --thinking high "Solve this complex problem"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PI_CODING_AGENT_DIR` | Override config directory; default `~/.pi/agent` |
| `PI_PACKAGE_DIR` | Override package directory for README/docs/examples resolution |
| `PI_OFFLINE` | Disable startup network operations |
| `PI_SKIP_VERSION_CHECK` | Skip GitHub release update check |
| `PI_VERSION_CHECK_PACKAGE` | PyPI distribution used for update checks |
| `PI_TELEMETRY` | Override install/update telemetry and provider attribution headers |
| `PI_SHARE_VIEWER_URL` | Override share viewer URL prefix |
| `PI_CLEAR_ON_SHRINK` | Clear terminal when it shrinks if terminal setting is unset |
| `PI_HARDWARE_CURSOR` | Show hardware cursor if setting is unset |
| `PI_TIMING` | Print startup timing diagnostics when set to `1` |
| `VISUAL`, `EDITOR` | External editor fallback |

Commands run by the LLM-callable bash tool also receive current session
metadata:

| Variable | Description |
|----------|-------------|
| `PI_SESSION_ID` | Current session ID |
| `PI_SESSION_FILE` | Absolute session JSONL path when available |
| `PI_PROVIDER` | Current model provider |
| `PI_MODEL` | Current model ID |
| `PI_REASONING_LEVEL` | Current reasoning level |

See [Environment Variables](docs/environment-variables.md#bash-tool-session-environment).

---

## Contributing & Development

See the repository root README for port status and verification commands. Do
not confuse that root document with this package-level product manual.

## License

MIT

## See Also

- `pi-ai`: provider registry and LLM API layer; CLI entry point `pp-ai`
- `pi-agent`: agent loop and session-independent agent types
- `pi-tui`: terminal UI components
- `pi-protocol`, `pi-server`, `pi-client`: socket RPC stack
- `pp-evals`: eval harness; CLI entry point `pp-evals`

---

`pp-coding-agent` is developed in [HSPK/pp](https://github.com/HSPK/pp). It was split out of the `pp` monorepo; sibling packages (`pp-ai`, `pp-agent-core`, `pp-tui`, `pp-coding-agent`, ...) each live in their own
repository and are consumed from PyPI.

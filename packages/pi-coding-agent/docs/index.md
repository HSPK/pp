# Pi Documentation

Pi is a minimal terminal coding harness. This Python port keeps the core small while it is being brought to parity with the TypeScript project. It is extended through Python extensions, skills, prompt templates, themes, and pi resource packages.

## Quick start

Install the Python workspace with uv:

```bash
cd /path/to/pp
uv sync --all-packages
```

Then run pi from the workspace:

```bash
uv run pp
```

To run the workspace checkout while keeping another project as pi's working directory, use uv's project selector:

```bash
cd /path/to/project
uv run --project /path/to/pp pp
```

Requires Python >=3.11. The console entry points provided by the workspace are `pp`, `pp-ai`, and `pp-evals`.

Authenticate with `/login` for providers supported by the Python port, or set an API key such as `ANTHROPIC_API_KEY` before starting pi.

For the full first-run flow, see [Quickstart](quickstart.md).

## Start here

- [Quickstart](quickstart.md) - install, authenticate, and run a first session.
- [Using Pi](usage.md) - interactive mode, slash commands, context files, and CLI reference.
- [Providers](providers.md) - subscription and API-key setup for built-in providers.
- [llama.cpp](llama-cpp.md) - status of the TypeScript llama.cpp extension in this Python port.
- [Security](security.md) - project trust, sandbox boundaries, and vulnerability reporting.
- [Containerization](containerization.md) - sandbox pi with Docker or OpenShell.
- [Settings](settings.md) - global and project settings.
- [Keybindings](keybindings.md) - default shortcuts and custom keybindings.
- [Sessions](sessions.md) - session management, branching, and tree navigation.
- [Compaction](compaction.md) - context compaction and branch summarization.

## Customization

- [Extensions](extensions.md) - Python modules for tools, commands, events, and custom behavior.
- [Skills](skills.md) - Agent Skills for reusable on-demand capabilities.
- [Prompt templates](prompt-templates.md) - reusable prompts that expand from slash commands.
- [Themes](themes.md) - built-in and custom terminal themes.
- [Pi packages](packages.md) - bundle and share extensions, skills, prompts, and themes.
- [Custom models](models.md) - add model entries for supported provider APIs.
- [Custom providers](custom-provider.md) - implement custom APIs and OAuth flows.

## Programmatic usage

- [SDK](sdk.md) - embed pi in Python applications.
- [Stdio RPC mode](rpc-stdio.md) - drive the agent as a subprocess over JSON lines on stdin/stdout.
- [Socket RPC mode](rpc.md) - integrate with the Unix-socket `pi_server` / `pi_client` stack.
- [JSON event stream mode](json.md) - print mode with structured events.
- [TUI components](tui.md) - build custom terminal UI for extensions.

## Reference

- [Environment variables](environment-variables.md) - Pi process configuration and session metadata available to bash tools.
- [Session format](session-format.md) - JSONL session file format, entry types, and SessionManager API.

## Platform setup

- [Windows](windows.md)
- [Termux on Android](termux.md)
- [tmux](tmux.md)
- [Terminal setup](terminal-setup.md)
- [Shell aliases](shell-aliases.md)

## Development

- [Development](development.md) - local setup, project structure, and debugging.

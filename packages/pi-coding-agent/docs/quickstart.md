# Quickstart

This page gets you from a source checkout to a useful first pi session.

## Install

The Python port is a uv workspace. From the workspace root:

```bash
cd /path/to/pp
uv sync --all-packages
```

Requires Python >=3.11. The workspace installs the console entry points `pp`, `pp-ai`, and `pp-evals` into uv's managed environment.

Check the CLI starts:

```bash
uv run pp --help
uv run pp --version
```

### Uninstall

For a source checkout, remove the checkout and uv environment when you no longer need them. User settings, credentials, sessions, and installed pi packages live under `~/.pi/agent/` and are not removed automatically.

```bash
cd /path/to/pp
rm -rf .venv
```

If you installed a built wheel with pip, uninstall the Python package with the same Python environment that installed it:

```bash
python3 -m pip uninstall pi-coding-agent
```

Then start pi in the project directory you want it to work on:

```bash
cd /path/to/project
uv run --project /path/to/pp pp
```

If you are already in the Python workspace and want pi to work on that workspace, run:

```bash
uv run pp
```

## Authenticate

Pi can use supported subscription providers through `/login`, or API-key providers through environment variables or the auth file.

### Option 1: subscription login

Start pi and run:

```text
/login
```

Then select a provider. The Python port has the OAuth and API-key auth storage used by `/login`, but individual providers that are listed as not ported in the repository README will still fail when used.

### Option 2: API key

Set an API key before launching pi:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run --project /path/to/pp pp
```

You can also run `/login` and select an API-key provider to store the key in `~/.pi/agent/auth.json`.

See [Providers](providers.md) for supported providers, environment variables, and cloud-provider setup.

## First session

Once pi starts, type a request and press Enter:

```text
Summarize this repository and tell me how to run its checks.
```

By default, pi gives the model four tools:

- `read` - read files
- `write` - create or overwrite files
- `edit` - patch files
- `bash` - run shell commands

Additional built-in read-only tools (`grep`, `find`, `ls`) are available through tool options. Pi runs in your current working directory and can modify files there. Use git or another checkpointing workflow if you want easy rollback.

## Give pi project instructions

Pi loads context files at startup. Add an `AGENTS.md` file to tell it how to work in a project:

```markdown
# Project Instructions

- Run `uv run pytest` after code changes.
- Do not run production migrations locally.
- Keep responses concise.
```

Pi loads:

- `~/.pi/agent/AGENTS.md` for global instructions
- `AGENTS.md` or `CLAUDE.md` from parent directories and the current directory

If a directory contains `AGENTS.override.md`, Pi loads it instead of `AGENTS.md` or `CLAUDE.md` from that directory.

Restart pi, or run `/reload`, after changing context files.

## Common things to try

### Reference files

Type `@` in the editor to fuzzy-search files, or pass files on the command line:

```bash
uv run --project /path/to/pp pp @README.md "Summarize this"
uv run --project /path/to/pp pp @src/app.py @tests/test_app.py "Review these together"
```

Images or text can be pasted with Ctrl+V (Alt+V on Windows); images can also be dragged into supported terminals.

### Run shell commands

In interactive mode:

```text
!uv run ruff check
```

The command output is sent to the model. Use `!!command` to run a command without adding its output to the model context.

### Switch models

Use `/model` or Ctrl+L to choose a model. Use Shift+Tab to cycle thinking level. Use Ctrl+P / Shift+Ctrl+P to cycle through scoped models.

### Continue later

Sessions are saved automatically:

```bash
uv run --project /path/to/pp pp -c                  # Continue most recent session
uv run --project /path/to/pp pp -r                  # Browse previous sessions
uv run --project /path/to/pp pp --name "my task"    # Set session display name at startup
uv run --project /path/to/pp pp --session <path|id> # Open a specific session
```

Inside pi, use `/resume`, `/new`, `/tree`, `/fork`, and `/clone` to manage sessions.

### Non-interactive mode

For one-shot prompts:

```bash
uv run --project /path/to/pp pp -p "Summarize this codebase"
cat README.md | uv run --project /path/to/pp pp -p "Summarize this text"
uv run --project /path/to/pp pp -p @screenshot.png "What's in this image?"
```

Use `--mode json` for a one-way JSON event stream, or `--mode rpc` for the bidirectional [stdio RPC protocol](rpc-stdio.md).

## Next steps

- [Using Pi](usage.md) - interactive mode, slash commands, sessions, context files, and CLI reference.
- [Providers](providers.md) - authentication and model setup.
- [Settings](settings.md) - global and project configuration.
- [Keybindings](keybindings.md) - shortcuts and customization.
- [Pi Packages](packages.md) - install shared extensions, skills, prompts, and themes.

Platform notes: [Windows](windows.md), [Termux](termux.md), [tmux](tmux.md), [Terminal setup](terminal-setup.md), [Shell aliases](shell-aliases.md).

# Using Pi

This page collects day-to-day usage details that do not fit on the quickstart page.

## Interactive Mode

The Python port starts the interactive TUI when `pp` is run on a TTY without `-p` or `--mode json`.

The interface has four main areas:

- **Startup header** - shortcuts, loaded context files, prompt templates, skills, and extensions
- **Messages** - user messages, assistant responses, tool calls, tool results, notifications, and errors
- **Editor** - where you type; border color indicates the current thinking level
- **Footer** - working directory, session name, token/cache usage, cost, context usage, and current model. Totals include assistant responses, usage reported by tools, and summary generation.

The editor can be replaced temporarily by built-in UI such as `/settings` or by selector dialogs.

### Editor Features

| Feature | How |
|---------|-----|
| File reference | Type `@` to fuzzy-search project files |
| Path completion | Press Tab to complete paths |
| Multi-line input | Shift+Enter, or Ctrl+Enter on Windows Terminal |
| Copy response | Ctrl+X copies the last assistant message; in `/tree`, it copies the selected message |
| Images | Paste with Ctrl+V, Alt+V on Windows, or drag into the terminal |
| Shell command | `!command` runs and sends output to the model |
| Hidden shell command | `!!command` runs without sending output to the model |
| External editor | Ctrl+G opens `externalEditor`, `$VISUAL`, `$EDITOR`, Notepad on Windows, or `nano` elsewhere |

See [Keybindings](keybindings.md) for all shortcuts and customization.

## Slash Commands

Type `/` in the editor to open command completion. Skills are available as `/skill:name`, and prompt templates expand via `/templatename`.

| Command | Description |
|---------|-------------|
| `/login`, `/logout` | Manage OAuth or API-key credentials |
| `/model` | Switch models |
| `/scoped-models` | Enable/disable models for Ctrl+P cycling |
| `/settings` | Thinking level, theme, message delivery, transport, and TUI options |
| `/resume` | Pick from previous sessions |
| `/new` | Start a new session |
| `/name <name>` | Set session display name |
| `/session` | Show session file, ID, messages, tokens, and cost |
| `/tree` | Jump to any point in the session and continue from there |
| `/trust` | Save project trust decision for future sessions |
| `/fork` | Create a new session from a previous user message |
| `/clone` | Duplicate the current active branch into a new session |
| `/compact [prompt]` | Manually compact context, optionally with custom instructions |
| `/copy` | Copy last assistant message to clipboard |
| `/export <file.jsonl>` | Export the current branch to JSONL. HTML export is not ported |
| `/import <file.jsonl>` | Import and resume a session from a JSONL file |
| `/share` | Not currently usable: it depends on the unported HTML exporter |
| `/reload` | Reload keybindings, extensions, skills, prompts, themes, and context files |
| `/hotkeys` | Show all keyboard shortcuts |
| `/changelog` | Display version history |
| `/quit` | Quit pi |

The llama.cpp local-model extension is not ported, so `/llama` is unavailable.

## Message Queue

You can submit messages while the agent is still working:

- **Enter** queues a steering message, delivered after the current assistant turn finishes executing its tool calls.
- **Alt+Enter** queues a follow-up message, delivered after the agent finishes all work.
- **Escape** aborts and restores queued messages to the editor.
- **Alt+Up** retrieves queued messages back to the editor.

On Windows Terminal, Alt+Enter is fullscreen by default. Remap it as described in [Terminal setup](terminal-setup.md) if you want pi to receive the shortcut.

Configure delivery in [Settings](settings.md) with `steeringMode` and `followUpMode`.

## Sessions

Sessions are saved automatically to `~/.pi/agent/sessions/`, organized by working directory.

```bash
pp -c                       # Continue most recent session
pp -r                       # Browse and select a session
pp --no-session             # Ephemeral mode; do not save
pp --name "my task"         # Set session display name at startup
pp --session <path|id>      # Use a specific session file or session ID
pp --session-id <uuid>      # Use or create an exact project session ID
pp --fork <path|id>         # Fork a session into a new session file
pp --session-dir <dir>      # Use a custom session storage directory
```

Useful session commands:

- `/session` shows the current session file and ID.
- `/tree` navigates the in-file session tree and can summarize abandoned branches.
- `/fork` creates a new session from an earlier user message.
- `/clone` duplicates the current active branch into a new session file.
- `/compact` summarizes older messages to free context.

See [Sessions](sessions.md) and [Compaction](compaction.md) for details.

## Context Files

Pi loads `AGENTS.md` or `CLAUDE.md` at startup from:

- `~/.pi/agent/AGENTS.md` for global instructions
- parent directories, walking up from the current working directory
- the current directory

If a directory contains `AGENTS.override.md`, Pi loads it instead of `AGENTS.md` or `CLAUDE.md` from that directory. Context files from other directories still layer normally.

Use context files for project conventions, commands, safety rules, and preferences. `--no-context-files` / `-nc` is parsed but not currently wired into the session resource loader.

### System Prompt Files

Replace the default system prompt with:

- `.pi/SYSTEM.md` for a project
- `~/.pi/agent/SYSTEM.md` globally

Append to the default prompt without replacing it with `APPEND_SYSTEM.md` in either location.

### Project Trust

On interactive startup, pi asks before trusting a project folder that contains project-local settings, resources, or project `.agents/skills` and has no saved decision for the folder or a parent folder in `~/.pi/agent/trust.json`. Trusting a project allows pi to load `.pi/settings.json`, `.pi` resources, read trusted project resource directories and execute project extensions.

Before the trust decision, pi loads only context files, user/global extensions, and CLI `-e` extensions. Project-local extensions, project package-managed extensions, and project settings are loaded only after the project is trusted. This split also applies when switching to a session from a different cwd whose trust has not been resolved in the current process.

Non-interactive modes (`-p` and `--mode json`) do not show a trust prompt. Without an applicable saved trust decision, they use `defaultProjectTrust` from global settings: `ask` (default) and `never` ignore those project resources, while `always` trusts them. Pass `--approve`/`-a` or `--no-approve`/`-na` to override project trust for one run.

The legacy stdio `--mode rpc` is accepted by the argument parser but is not ported; use the `pi_server` / `pi_client` socket stack instead.

`pp config` and package commands use saved/default project trust without an interactive prompt. Pass `--approve` to trust project-local settings for one command or `--no-approve` to ignore them. `pp update` uses only an explicit flag or a saved trust decision.

Use `/trust` in interactive mode to save a project trust decision for future sessions, including trust for the immediate parent folder. It writes `~/.pi/agent/trust.json` only; the current session is not reloaded, so restart pi for changes to take effect.

## Exporting and Sharing Sessions

Use `/export path.jsonl` to write the current branch to JSONL.

HTML export and `/share` are not ported because the HTML document assembly is unavailable.

## CLI Reference

```bash
pp [options] [@files...] [messages...]
```

### Package Commands

```bash
pp install <source> [-l]       # Install a git or local package, -l for project-local
pp remove <source> [-l]        # Remove package
pp uninstall <source> [-l]     # Alias for remove
pp update [source]             # Update all git packages, or one git package source
pp list                        # List installed packages
pp config [-l]                 # Enable/disable package resources
```

Package commands accept `--approve`/`--no-approve` to trust or ignore project-local settings for one command. `pp update` never prompts for project trust. Self-update, npm package sources, model-catalog refresh, and `update --all` / `--extensions` / `--models` / `--self` / `--extension` are not supported in this port.

See [Pi Packages](packages.md) for package sources and security notes.

### Modes

| Flag | Description |
|------|-------------|
| default | Interactive mode |
| `-p`, `--print` | Print response and exit |
| `--mode json` | Output all events as JSON lines; see [JSON mode](json.md) |
| `--mode rpc` | Accepted but not ported; exits with an error |
| `--export <file>` | Parsed but not currently handled by the CLI |

In print mode, pi also reads piped stdin and merges it into the initial prompt:

```bash
cat README.md | pp -p "Summarize this text"
```

### Model Options

| Option | Description |
|--------|-------------|
| `--provider <name>` | Provider, such as `anthropic`, `openai`, or `google` |
| `--model <pattern>` | Model pattern or ID; supports `provider/id` and optional `:<thinking>` |
| `--api-key <key>` | API key for this process only; requires a selected model/provider |
| `--thinking <level>` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `--models <patterns>` | Comma-separated patterns for Ctrl+P cycling |
| `--list-models [search]` | List available models |

### Session Options

| Option | Description |
|--------|-------------|
| `-c`, `--continue` | Continue the most recent session |
| `-r`, `--resume` | Browse and select a session; without a TTY, continue the most recent session |
| `--session <path\|id>` | Use a specific session file or partial UUID |
| `--session-id <id>` | Use an exact project session ID, creating it if missing |
| `--fork <path\|id>` | Fork a session file or partial UUID into a new session |
| `--session-dir <dir>` | Custom session storage directory |
| `--no-session` | Ephemeral mode; do not save |
| `--name <name>`, `-n <name>` | Set session display name at startup |

### Tool Options

| Option | Description |
|--------|-------------|
| `--tools <list>`, `-t <list>` | Allowlist specific built-in, extension, and custom tools |
| `--exclude-tools <list>`, `-xt <list>` | Disable specific built-in, extension, and custom tools |
| `--no-builtin-tools`, `-nbt` | Disable built-in tools but keep extension/custom tools enabled |
| `--no-tools`, `-nt` | Disable all tools |

Built-in tools: `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`.

### Resource Options

| Option | Description |
|--------|-------------|
| `-e`, `--extension <path>` | Load a local Python extension file or directory; repeatable |
| `--no-extensions`, `-ne` | Disable extension discovery; explicit `-e` paths still load |
| `--skill <path>` | Parsed but not currently passed to the session resource loader |
| `--no-skills`, `-ns` | Parsed but not currently passed to the session resource loader |
| `--prompt-template <path>` | Parsed but not currently passed to the session resource loader |
| `--no-prompt-templates`, `-np` | Parsed but not currently passed to the session resource loader |
| `--theme <path>` | Parsed but not currently passed to startup theme loading |
| `--no-themes` | Parsed but not currently passed to startup theme loading |
| `--no-context-files`, `-nc` | Parsed but not currently passed to the session resource loader |

For extensions, combine `--no-extensions` with explicit `-e` flags to load exactly what you need, ignoring discovered extension directories. Example:

```bash
pp --no-extensions -e ./my_extension.py
```

### Other Options

| Option | Description |
|--------|-------------|
| `--system-prompt <text>` | Parsed but not currently passed to the session resource loader |
| `--append-system-prompt <text>` | Parsed but not currently passed to the session resource loader |
| `--tui-mode <mode>` | TUI mode: `regular` (default) or experimental `fullscreen` |
| `--verbose` | Force verbose startup |
| `-a`, `--approve` | Trust project-local files for this run |
| `-na`, `--no-approve` | Ignore project-local files for this run |
| `--offline` | Disable startup network operations; same as `PI_OFFLINE=1` |
| `-h`, `--help` | Show help |
| `-v`, `--version` | Show version |

In `fullscreen` mode, the transcript scrolls inside the terminal viewport while queued messages, working status, editor, and footer remain fixed at the bottom. Mouse/trackpad input scrolls the region under the pointer; keyboard viewport actions remain available. Inline images work in terminals that support the Kitty graphics protocol, including Kitty and Ghostty.

Set **TUI mode** in `/settings` to choose the default for future sessions. Runtime switching is not ported; the new mode applies on next start.

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
pp --model sonnet:high "Solve this complex problem"

# Limit model cycling
pp --models "claude-*,gpt-4o"

# Read-only mode
pp --tools read,grep,find,ls -p "Review the code"

# Disable one extension or built-in tool while keeping the rest available
pp --exclude-tools ask_question
```

## Design Principles

Pi keeps the core small and pushes workflow-specific behavior into extensions, skills, prompt templates, and packages.

It intentionally does not include built-in MCP, sub-agents, permission popups, plan mode, to-dos, or background bash. You can build or install those workflows as extensions or packages, or use external tools such as containers and tmux.

For the full rationale, read the [blog post](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/).

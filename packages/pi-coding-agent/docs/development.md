# Development

See [README.md](../../README.md) for additional porting guidelines and current porting status.

## Setup

```bash
cd /path/to/pp
uv sync --all-packages
```

Run from source:

```bash
uv run pp --help
uv run pp --version
uv run pp --list-models
```

Run from another project while using this checkout as the installed project:

```bash
cd /path/to/project
uv run --project /path/to/pp pp
```

The command keeps the caller's current working directory.

## Forking / Rebranding

The Python port does not read `package.json`. App identity and path constants live in `packages/pi-coding-agent/src/pi_coding_agent/core/config.py`:

```text
APP_NAME = "pi"
APP_TITLE = "pi"
PACKAGE_NAME = "pi-coding-agent"
```

Changing these affects CLI banner text, config-path environment variable names, and package asset resolution. The console script name is declared in `packages/pi-coding-agent/pyproject.toml` under `[project.scripts]`.

## Path Resolution

The self-documentation paths are resolved by `core/config.py`:

```text
get_package_dir()
get_readme_path()
get_docs_path()
get_examples_path()
```

`PI_PACKAGE_DIR` overrides the package directory. It must point at the directory holding `README.md`, `docs/`, and `examples/`.

## Debug Command

`/debug` (hidden) writes to `~/.pi/agent/pi-debug.log`:
- Rendered TUI lines with ANSI codes
- Last messages sent to the LLM

Set `PI_TIMING=1` for startup timing diagnostics.

## Testing

```bash
uv run pytest                      # Run tests
uv run pytest packages/pi-ai       # Run one package's tests
uv run ruff check packages/        # Lint
uv run ruff format --check packages/ # Format check
./check.sh                         # Lint + format check + tests + coverage + parity report
```

Do not run real-provider tests with live credentials unless you intend to spend provider quota.

## Project Structure

```
packages/
  pi-ai/           # LLM provider abstraction
  pi-agent/        # Agent loop, harness, transcript layer
  pi-tui/          # Terminal UI components
  pi-protocol/     # CBOR/framing/wire protocol
  pi-client/       # Unix-socket client
  pi-server/       # Unix-socket server
  pi-coding-agent/ # CLI, tools, interactive mode
  pi-evals/        # Behavioral eval runner
```

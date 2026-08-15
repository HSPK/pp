# Environment Variables

Pi uses environment variables in three ways:

- Variables such as `PI_OFFLINE` configure the Pi process.
- Commands run by the LLM-callable bash tool receive `PI_*` variables describing the current session.
- Provider API-key variables configure model providers.

Provider API-key variables are documented separately in [Providers](providers.md#environment-variables-or-auth-file).

## Process Marker

The TypeScript CLI sets `PI_CODING_AGENT=true`. The Python CLI does not currently set that marker. If it is already present in the parent environment, child commands inherit it like any other ambient variable.

## Bash Tool Session Environment

Commands run by the LLM-callable bash tool receive the current Pi session state:

| Variable | Description |
|----------|-------------|
| `PI_SESSION_ID` | Current session ID |
| `PI_SESSION_FILE` | Absolute path to the current session JSONL file; unset for ephemeral sessions in the CLI harness |
| `PI_PROVIDER` | Currently selected model provider |
| `PI_MODEL` | Currently selected model ID |
| `PI_REASONING_LEVEL` | Current effective reasoning level: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` |

The values are resolved when each command starts. Switching models or changing the reasoning level therefore affects the next bash command without restarting Pi. `PI_PROVIDER` and `PI_MODEL` identify the selected Pi model, not a different upstream model that a router may choose internally.

When asked which model or provider is running, inspect these variables instead of inferring the answer from the system prompt:

```bash
printf '%s/%s\n' "$PI_PROVIDER" "$PI_MODEL"
printf 'reasoning=%s session=%s\n' "$PI_REASONING_LEVEL" "$PI_SESSION_ID"
```

The session file can be inspected directly when the session is persistent:

```bash
if [ -n "$PI_SESSION_FILE" ]; then
  tail -n 1 "$PI_SESSION_FILE"
fi
```

These variables are injected into the LLM-callable bash tool. They are not injected into user-entered `!` or `!!` commands.

### Custom Bash Tools

Python bash tools created with `create_bash_tool()` expose the session environment by default when given a `session_environment` callback. Injection happens before `spawn_hook`, so a hook receives the variables in `ctx.env`.

Disable session metadata with `expose_session_environment=False`. When disabled, Pi removes inherited values for these five variables so nested Pi processes do not expose stale parent-session metadata.

## Pi Process Configuration

These variables are read by the Python port itself:

| Variable | Description |
|----------|-------------|
| `PI_CODING_AGENT_DIR` | Override the config directory; default is `~/.pi/agent` |
| `PI_PACKAGE_DIR` | Override the package directory holding `README.md`, `docs/`, and `examples/`; used by self-documentation path resolution |
| `PI_OFFLINE` | Disable startup network operations, package-manager network work, managed tool downloads, and version checks |
| `PI_SKIP_VERSION_CHECK` | Disable the latest-release version request |
| `PI_VERSION_CHECK_PACKAGE` | PyPI distribution used for version checks; Python-specific replacement for the TypeScript `pi.dev` version API |
| `PI_TELEMETRY` | Override telemetry and provider attribution behavior: `1`/`true`/`yes` or `0`/`false`/`no` |
| `PI_CACHE_RETENTION` | Set to `long` for extended provider prompt caching where supported |
| `PI_SHARE_VIEWER_URL` | Override the base URL used by `/share` URL construction |
| `PI_HARDWARE_CURSOR` | Set to `1` to show the hardware cursor unless `showHardwareCursor` is set; see [Terminal setup](terminal-setup.md) |
| `PI_CLEAR_ON_SHRINK` | Diagnostic TUI behavior: clear when the terminal shrinks |
| `PI_EXPERIMENTAL` | Enable experimental CLI surfaces guarded by the Python port |
| `PI_TIMING` | Set to `1` to print internal startup timing diagnostics |
| `PI_DEBUG_REDRAW` | Set to `1` to disable normal redraw coalescing in the TUI main screen |
| `PI_TUI_WRITE_LOG` | Low-level terminal write logging used by `pi_tui` |
| `VISUAL`, `EDITOR` | External editor fallback when `externalEditor` is unset |
| `HTTP_PROXY`, `HTTPS_PROXY` | Proxy outbound HTTP requests |

`PI_CODING_AGENT_SESSION_DIR` is defined as a constant but is not currently read by the Python session manager. Use `--session-dir` instead.

## Evals Environment

The `pp-evals` entry point also reads:

| Variable | Description |
|----------|-------------|
| `PI_EVAL_ARTIFACT_DIR` | Override the eval artifact output directory |
| `PI_SESSION_SNAPSHOT_ARTIFACT` | Session snapshot artifact path used by the eval reporter |

Provider credentials such as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, cloud-provider variables, and provider-scoped model-list variables are listed in [Providers](providers.md#environment-variables-or-auth-file).

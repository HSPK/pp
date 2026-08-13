# pp

A Python port of [pi](https://github.com/earendil-works/pi), a coding agent that
runs in your terminal.

## What pi is

pi is a coding agent harness. You give it a prompt; it reads files, runs
commands, edits code and writes new files to carry it out. It ships as a
terminal UI with a scrollback or fullscreen mode, a session store that records
every turn as an append-only log you can branch and resume, and an extension
system for adding your own commands, tools and themes.

It talks to models through a provider layer rather than a single vendor SDK, so
Anthropic, OpenAI, Google, GitHub Copilot, OpenRouter and OpenAI-compatible
gateways all work through the same interface, with OAuth or API-key auth per
provider.

One detail worth knowing, because it explains a design choice throughout: pi can
answer questions about itself. There is no retrieval index behind that. The
system prompt names absolute paths to this package's `README.md`, `docs/` and
`examples/` plus a map from topic to filename, and the agent opens those real
files with its ordinary read tool. The documentation is a functional part of the
product, not decoration.

## This port

This repository is a Python implementation of that TypeScript project. It is a
port, not a rewrite: the goal is that a given input produces the same observable
behaviour as upstream, down to error strings and on-disk formats.

| TypeScript package | PyPI distribution | Directory | Import module |
| --- | --- | --- | --- |
| `@earendil-works/pi-telemetry` | `pp-telemetry` | `packages/pi-telemetry` | `pi_telemetry` |
| `@earendil-works/pi-ai` | `pp-ai` | `packages/pi-ai` | `pi_ai` |
| `@earendil-works/pi-agent-core` | `pp-agent-core` | `packages/pi-agent` | `pi_agent` |
| `@earendil-works/pi-tui` | `pp-tui` | `packages/pi-tui` | `pi_tui` |
| `@earendil-works/pi-protocol` | `pp-rpc-protocol` | `packages/pi-protocol` | `pi_protocol` |
| `@earendil-works/pi-client` | `pp-rpc-client` | `packages/pi-client` | `pi_client` |
| `@earendil-works/pi-server` | `pp-rpc-server` | `packages/pi-server` | `pi_server` |
| `@earendil-works/pi` (coding agent) | `pp-coding-agent` | `packages/pi-coding-agent` | `pi_coding_agent` |
| `@earendil-works/pi-evals` | `pp-evals` | `packages/pi-evals` | `pi_evals` |

The three names that differ from a mechanical translation do so because the
obvious name was already taken on PyPI by an unrelated project: `pi-agent` and
`pi-coding-agent` belong to other authors, and `pp-server` is
[`pp.server`](https://pypi.org/project/pp.server/), an actively maintained
package. `pp-rpc-protocol`/`pp-rpc-client`/`pp-rpc-server` are named as a set
because they are one stack: framed CBOR over a Unix socket.

Import modules deliberately keep their `pi_*` spelling. They name what the code
*is* — a port of pi — and every module docstring, test and porting convention
refers to them.

## Install

```bash
pip install pp-coding-agent   # or: uv tool install pp-coding-agent
pp                            # start the agent
```

The nine distributions are released in lockstep: they always share one version,
and each pins its siblings with `==`. Installing `pp-coding-agent` pulls the
whole stack.

Requires Python 3.11+. The workspace is managed with
[uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-packages                # install workspace + dev tools
uv run pp                             # start the agent (interactive)
uv run pp "explain this repo"         # run a single prompt
uv run pp --list-models               # what the provider layer can reach
uv run pp-ai login anthropic          # authenticate a provider
uv run pp-evals --provider openai --model gpt-5.6-sol
```

Documentation lives in `packages/pi-coding-agent/docs/` — extensions, custom
providers, session format, settings, keybindings, the SDK and the TUI.

## Development

```bash
uv run pytest                         # the whole suite
uv run pytest packages/pi-ai          # one package
uv run ruff check packages/           # lint
uv run ruff format packages/          # format
./check.sh                            # lint + format check + tests + coverage
```

`scripts/check_test_parity.py` indexes which upstream test files have a Python
counterpart. Treat it as an index, not a verdict — it matches filenames in
docstrings and cannot tell a faithful port from a weakened one. Reading the two
files side by side is the only real check.

## Releasing

All nine distributions share one version and pin each other with `==`. Two
checks in `check.sh` enforce the properties that a workspace install otherwise
hides:

- `scripts/set_version.py --check` — every package declares the same version,
  and no internal dependency is left unpinned. An unpinned sibling is not a
  style problem: `[tool.uv.sources]` does not survive into a wheel, so
  `Requires-Dist: pp-ai` would be resolved from PyPI at install time.
- `scripts/check_dependencies.py` — every cross-package import is declared.
  `uv sync --all-packages` puts all nine on `sys.path`, so an undeclared
  sibling import passes the whole test suite and only fails for the end user.

To cut a release:

```bash
python scripts/set_version.py 0.2.0   # rewrites all 9 versions and every == pin
uv lock && uv sync --all-packages
./check.sh
git commit -am "release 0.2.0" && git tag v0.2.0 && git push --tags
```

Pushing the tag runs `.github/workflows/release.yml`, which builds all nine
distributions and publishes them to PyPI through Trusted Publishing (OIDC), in
dependency order. No API token is stored in this repository: PyPI is configured
to trust that one workflow file, and mints a short-lived token per upload. The
upload order is not cosmetic — the packages pin each other with `==`, so a
dependency has to land before the package requiring it.

Version numbering is the port's own. `UPSTREAM_VERSION` in
`pi_coding_agent.core.config` records which upstream release the behaviour is
aligned with, because this port deliberately omits the features listed below.

## What is not ported

Behaviour that upstream has and this port does not. Each entry is a deliberate
decision with a reason, not an oversight:

- **Node's `fs.watch`**, which has no standard-library equivalent; the git
  branch watcher polls instead, with the same debounce and callback contract.
- **Mermaid diagram rendering**, which renders through the `grok-mermaid` npm
  package.
- **The `/arminsayshi` and `/dementedelves` easter eggs** and the announcement
  banner, which are bundled ASCII-art animations and a PNG asset.
- **Most of the extension UI host**: widgets and the select/confirm/input
  dialogs, footer statuses, terminal title and the tools-expanded toggle *are*
  wired; custom header/footer components, terminal input listeners,
  working-indicator control, editor control, autocomplete providers and the
  theme accessors are not. The startup resource/diagnostic report is not
  ported either.
- **Extension provider registration** (`registerProvider`/`unregisterProvider`)
  and the **remote model catalog** (`refreshModels`, `ModelsPublication`), which
  together form a dynamic catalog layer this port does not implement. Providers
  come from the built-in set plus `models.json`; see `docs/custom-provider.md`.
- **AWS Bedrock** (`bedrock-converse-stream`) needs SigV4 signing and the Smithy
  stack; **OpenAI Codex** (`openai-codex-responses`) needs its OAuth/WebSocket
  transport. Both providers are still in the model catalog, so their models are
  discoverable; streaming raises `NotImplementedError`.
- **The legacy stdio RPC mode.** Superseded by the `pi_server`/`pi_client`
  socket stack, which is ported. Its strict LF-only JSONL framing *is* ported and
  tested. `--mode rpc` reports the incompatibility and exits.
- **The HTML exporter's document assembly**, which embeds vendored
  `marked`/`highlight.js` browser bundles. Its ANSI-to-HTML converter and colour
  maths are ported and cross-checked. `--export` reports this and exits.
- **npm-sourced packages** in the package manager (no Python registry
  equivalent); git and local-path sources are ported.
- **Install-method detection and self-update.** Upstream inspects its own
  npm/pnpm/yarn/bun install layout to update itself; a `uv`/`pip` install has no
  equivalent story.

## How the port is verified

Verification is anchored on upstream artifacts wherever they exist rather than
on the port agreeing with itself:

- TypeScript test cases are ported alongside the code they cover.
- The session layer runs the upstream conformance suite against both the
  in-memory and JSONL stores.
- `short_hash` was compared byte for byte against the JavaScript implementation.
- The CBOR codec is tested against the RFC 8949 Appendix A vectors.
- `diff_words` is diff-tested against the real `diff` npm package at the exact
  version upstream pins (8.0.4), and `render_diff` against the upstream
  `diff.ts` executed under Node; both corpora are checked in under
  `packages/pi-coding-agent/tests/data/`.
- `format_tokens`, `format_cwd_for_footer`, `sanitize_status_text` and the JS
  `toFixed` emulation were compared against the TypeScript running under Node.
- The provider adapters are driven by SSE fixtures over a real loopback socket,
  not only by mocked transports.
- The CLI argument scanner and the JSONL framing were cross-checked by running
  the actual TypeScript implementations under Node and comparing every field of
  the output, case by case.
- The RPC stack is exercised end to end: the real server, driven by the real
  coding-agent session runtime, against the real client over a Unix socket.
- Extension loading is tested against real extension files written to a temp
  directory, including the refusal to load project-local extensions from an
  untrusted project.
- `scripts/fake_openai_server.py` serves a scripted OpenAI-compatible endpoint so
  the installed `pp` binary can be smoke tested end to end without an API key.



## Porting conventions

These rules keep independently ported modules consistent.

**File layout.** A TypeScript file `packages/<pkg>/src/a/b-c.ts` becomes
`packages/pi-<pkg>/src/pi_<pkg>/a/b_c.py`. Tests live in
`packages/pi-<pkg>/tests/test_<b_c>.py`.

**Naming.** `camelCase` becomes `snake_case` for functions, methods, fields and
locals. Types keep their `PascalCase` names. String literal values that cross
the wire (event `type` tags, provider field names, JSON keys) keep their exact
TypeScript spelling — for example the event tag stays `"toolcall_start"` and the
message role stays `"toolResult"`.

**Discriminated unions.** A TypeScript union such as
`{ type: "text" } | { type: "image" }` becomes one `@dataclass` per variant with
a `type: Literal[...]` field that has a default, plus a `Union` alias. Dispatch
on `value.type`, not on `isinstance`, so behaviour matches the original.

**Optional fields.** TypeScript `field?: T` becomes `field: T | None = None`.
Optional collections that the TypeScript code always treats as "empty when
absent" become `field(default_factory=list/dict)` instead.

**Async.** `Promise<T>` becomes `async def -> T`. `AsyncIterable` becomes an
async generator. `AbortSignal` becomes an `asyncio.Event`-backed
`pi_ai.utils.abort.AbortSignal`; cancellation of the whole request uses
`asyncio.CancelledError` semantics on top of it.

**Errors.** A thrown `Error` becomes a domain exception (`ModelsError`,
`ToolValidationError`, ...) or `ValueError`/`RuntimeError` when the TypeScript
code threw a bare `Error`. Never swallow an exception that the original
propagated.

**No JS-only behaviour.** Don't reproduce JavaScript quirks that have no meaning
in Python (`undefined` vs missing keys, `Object.freeze`). Do reproduce
behaviour that callers depend on, such as `Date.now()` millisecond timestamps
(`pi_ai.types.now_ms`).

**Docstrings.** Every ported module starts with a docstring naming its
TypeScript source file. Port the explanatory comments that state *why* code is
written a certain way; drop comments that only restate the code.

**Dependencies.** Prefer the standard library. Current third-party use:
`httpx` (HTTP/SSE), `jsonschema` (tool schema validation), `pyyaml`,
`wcwidth` (terminal width). Do not add a dependency without a concrete need.

## Testing

Every ported module needs tests. Tests are written against the Python API and
must assert real behaviour, not just that a function is callable. Provider
network calls are exercised against an in-process fake transport, never a real
endpoint.

## License and attribution

MIT, the same licence as upstream.

This project is a derivative work of [pi](https://github.com/earendil-works/pi)
by Mario Zechner and Earendil Works. The original copyright notice is preserved
in [LICENSE](LICENSE) alongside the notice for the Python port, and each
published distribution carries that file and credits the original author in its
`authors` metadata.

It is an independent port and is not affiliated with or endorsed by the
upstream project. Bugs found here should be reported to
[this repository](https://github.com/HSPK/pp/issues), not to upstream, unless
they are reproducible against the TypeScript implementation.

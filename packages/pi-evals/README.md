# pp-evals

Python port of the TypeScript `@earendil-works/pi-evals` package
(`../../../pp/packages/evals`).

Pi evals are behavioral, model-backed checks for Pi workflows. They adapt a
real `AgentSession` to an eval harness, run it in isolated temporary project
and agent directories, and attach native Pi session artifacts. Use them to
measure end-to-end behavior and compare prompts, tools, skills, models, or
other harness configurations.

## The `vitest-evals` substitution

The TypeScript package is built on the npm library
[`vitest-evals`](https://github.com/getsentry/vitest-evals) plus a Vitest
reporter. There is no Python equivalent, and no `node_modules` to read the real
types from, so this port reimplements the slice of that interface `pp-evals`
actually uses, on top of pytest, in `pi_evals.harness`:

| `vitest-evals` / Vitest | `pi_evals` |
| --- | --- |
| `createHarness({ name, run })` | `create_harness(name=..., run=...)` |
| `Harness`, `HarnessContext`, `HarnessRun` | `Harness` (Protocol), `HarnessContext`, `HarnessRun` |
| `SimpleHarnessResult` | `SimpleHarnessResult` |
| `TranscriptEvent`, `toolCalls(...)` | `TranscriptEvent` union, `tool_calls(...)` |
| `normalizeRecord`, `toJsonValue` | `normalize_record`, `to_json_value` |
| `isHarnessRun`, `attachHarnessRunToError` | `is_harness_run`, `attach_harness_run_to_error` |
| `createJudge(name, fn)` | `create_judge(name, fn)` |
| `describeEval(name, options, define)` | `describe_eval(name, options, define)` |
| Vitest `TestArtifact` / `task.meta` | `TestArtifact`, `EvalMeta` in a pytest stash |
| Vitest `Reporter` | a pytest plugin (`pi_evals.vitest_evals.reporter`) |
| Vitest `describe.for(table)` | a plain `for` loop over `eval_harness_table(...)` |
| `npm run eval` (`scripts/run-evals.mjs`) | the `pp-evals` console script |

`describe_eval(...)` generates one `async def test_...` per declared case and
injects it into the calling module's globals, so pytest collects them
normally. Judges run after the body; the mean score is recorded on the test's
`EvalMeta`. `judge_threshold=None` records a low score as an observation
instead of failing the test, exactly like the TypeScript.

The on-disk artifact format is unchanged, so both implementations' `.eval/`
directories are comparable: `runs.jsonl` records keep the TypeScript key
spelling (`schemaVersion`, `runId`, `fullName`, `totalTokens`, ...) and
attachments land under `sessions/` and `sources/`. Records are appended in
completion order, as upstream appends them from `onTestCaseResult`; consumers
must key by `harness`/`runId` rather than by line number. `summary.py` is the
layer that imposes order, sorting eval sets, candidates and groups itself.

Under `pytest-xdist` (`-n auto`) the plugin is loaded in every worker. The
append takes an exclusive `flock` so concurrent workers cannot interleave
partial lines, and each worker ships its observations to the controller over
xdist's `workeroutput` channel -- `pytest_terminal_summary` runs only in the
controller, which never executed the tests. `pytest_testnodedown` is an xdist
hookspec, so the controller-side collector is registered from
`pytest_configure` only when xdist is loaded; otherwise pytest would reject
the whole plugin with `PluginValidationError: unknown hook`. Vitest needs none
of this: its reporters run in the main process.

## Running evals

```bash
uv run pp-evals --provider openai --model gpt-5.6-sol
```

The equivalent environment variables are:

```bash
PI_PROVIDER=openai PI_MODEL=gpt-5.6-sol uv run pp-evals
```

CLI values take precedence and become defaults for harnesses that do not
select a model explicitly. Provider and model must be supplied together. The
runner also allows no default when every executed harness configures its own
model. Authentication comes from Pi's normal `ModelRuntime`, including Pi
subscription credentials and provider API-key environment variables.

Additional arguments are forwarded to pytest:

```bash
uv run pp-evals src/pi_evals/evals/extensions_eval.py
uv run pp-evals -k "creates_reloads_and_uses"
```

Each invocation prints an ignored `.eval/` artifact directory and exports it as
`PI_EVAL_ARTIFACT_DIR`. `runs.jsonl` indexes completed harness runs and their
native Pi session JSONL attachments under `sessions/`. These files may contain
prompts, responses, source code, and tool output.

## Writing evals

Pi-specific evals use `create_pi_coding_agent_harness(...)` from
`pi_evals.pi_harness`, with one harness bound to each `describe_eval(...)`
suite:

```python
from pi_evals.harness import EvalCase, EvalOptions, describe_eval
from pi_evals.pi_harness import PiCodingAgentHarnessOptions, create_pi_coding_agent_harness

harness = create_pi_coding_agent_harness(PiCodingAgentHarnessOptions(no_tools="all"))


def _define(it) -> None:
    async def answers_a_factual_question(case: EvalCase) -> None:
        result = await case.run("What is the capital of France? Reply with only the city name.")
        assert result.output == "Paris"

    it("answers a factual question", answers_a_factual_question)


describe_eval("Pi smoke", EvalOptions(harness=harness), _define)
```

The `define` callback receives `it`, which registers one case by name; each
case body takes a single `EvalCase` (TypeScript destructures `{ run, task }`
from the same thing).

### Configuring the Pi harness

`PiCodingAgentHarnessOptions` accepts:

- `name`: stable harness identity used by reports and comparisons.
- `model`: optional `PiCodingAgentModelSelection(provider, id)`. It overrides
  the runner's default model.
- `no_tools`: Pi's tool-disable configuration (`"all"` or `"builtin"`).
- `transform_system_prompt`: transforms the complete default prompt before the
  eval starts.
- `output`: transforms the final response and `AgentSession` into a JSON-safe
  domain result.

An explicitly selected model makes model-comparison harnesses independent of
the runner default:

```python
harness = create_pi_coding_agent_harness(
    PiCodingAgentHarnessOptions(
        name="claude-opus-4-6",
        model=PiCodingAgentModelSelection(provider="anthropic", id="claude-opus-4-6"),
    )
)
```

A run accepts either one prompt or a sequence of prompt and reload steps.
Reload steps are useful when the preceding prompt creates or changes Pi
resources:

```python
result = await case.run(
    [
        PromptStep(content="Create a Pi extension."),
        ReloadStep(),
        PromptStep(content="Use the extension."),
    ]
)
```

### Transforming harness output

Use `output` to expose scenario-specific, JSON-safe behavior without adding
that behavior to the generic Pi adapter:

```python
harness = create_pi_coding_agent_harness(
    PiCodingAgentHarnessOptions(
        output=lambda context: {
            "response": context.response,
            "activeTools": context.session.get_active_tool_names(),
        }
    )
)
```

Assert application behavior on `result.output`. Assert model and tool traces on
`result.events`, using `pi_evals.harness.tool_calls(...)`.

### Writing comparative eval sets

Use `eval_harness_table(...)` to run the same inputs against multiple
harnesses. Harnesses may differ by prompt, tools, skills, model, or any other
Pi configuration. TypeScript wraps this in Vitest's `describe.for(...)`; here a
plain loop declares one suite per row:

```python
TargetTaskJudge = create_judge(
    "TargetTaskJudge",
    lambda context: JudgeResult(score=1 if context.output == "expected result" else 0),
)

for row in eval_harness_table(
    "target skill effectiveness",
    baseline=without_target_skill_harness,
    candidate=with_target_skill_harness,
    repetitions=6,
):

    def _define(it, row=row) -> None:
        async def completes_the_target_task(case: EvalCase) -> None:
            await case.run("Complete the target task.")

        it("completes the target task", completes_the_target_task)

    describe_eval(
        "target skill effectiveness",
        EvalOptions(harness=row.harness, judges=[TargetTaskJudge], judge_threshold=None),
        _define,
        suffix=f"{row.name} repetition {row.repetition}",
    )
```

Comparative suites should record correctness with deterministic or model-backed
judges and set `judge_threshold=None`. This keeps a low score as an observation
instead of failing the pytest invocation. Use hard assertions only for suite
invariants and infrastructure contracts.

The Pi harness snapshots native session JSONL before deleting its temporary
workspace. The reporter's `pytest_runtest_teardown` hook (the port of the
eval-only Vitest `afterEach` in `setup.ts`) registers that snapshot against the
test before persisting artifacts.

Harness names must be stable and unique within an eval set. The grouping key
combines repetition with a non-empty string `input["id"]` when available,
otherwise with a SHA-256 hash of strict canonical JSON input. Use `candidate`
for one treatment or `candidates` for multiple treatments. Each candidate is
compared only with the declared baseline. For each matched input and
repetition, the reporter computes pass-rate lift from each run's recorded
average judge score, treating a score of at least `1` as passing. Lift is the
candidate pass rate minus the baseline pass rate, in percentage points. Missing
judge scores are reported as incomplete observations. Tokens, latency, and
estimated cost remain separate candidate-minus-baseline paired deltas; missing
telemetry remains unavailable.

See the [`skill-eval-harness`](https://github.com/adewale/skill-eval-harness/)
guidance for comparative-eval methodology, repetition strategy, trustworthy
judges, and telemetry interpretation.

## Layout

| Python module | TypeScript source |
| --- | --- |
| `pi_evals.harness` | the used slice of the `vitest-evals` npm package |
| `pi_evals.pi_harness` | `src/pi-harness.ts` |
| `pi_evals.vitest_evals.summary` | `src/vitest-evals/summary.ts` |
| `pi_evals.vitest_evals.harness_table` | `src/vitest-evals/harness-table.ts` |
| `pi_evals.vitest_evals.artifacts` | `src/vitest-evals/artifacts.ts` |
| `pi_evals.vitest_evals.reporter` | `src/vitest-evals/reporter.ts`, `src/vitest-evals/setup.ts` |
| `pi_evals.evals.smoke_eval` | `src/smoke.eval.ts` |
| `pi_evals.evals.extensions_eval` | `src/extensions.eval.ts` |
| `pi_evals.run_evals` | `scripts/run-evals.mjs` |

## Deliberate omissions

- **`vitest-evals` itself.** Only the interface `pp-evals` uses is ported; its
  scorers, its own reporters, and its Vitest integration are not.
- **`AgentSession.reload()`.** Not ported by `pi_coding_agent`, and
  `create_agent_session` never loads extensions, so a `ReloadStep` reloads
  settings, resources, the system prompt and the tool registry but cannot
  activate an extension the model just wrote. `pi_evals.pi_harness` documents
  the substitute (`reload_eval_session`), and `extensions_eval` documents the
  judge check that consequently cannot pass in this port.
- **`createAgentSessionServices` / `createAgentSessionFromServices`.** The
  per-cwd services bundle was dropped by `pi_coding_agent`; this port calls
  `create_agent_session` with the same pieces directly.
- **The "Pi documentation" system-prompt section.**
  `pi_coding_agent.core.system_prompt` deliberately omits it (no `docs/`
  tree ships with the Python port), so `extensions_eval`'s baseline transform
  only strips Guidelines and both harnesses report
  `system_prompt_has_pi_docs` as `False`.
- **The TypeScript-only extension judge checks** (`@mariozechner/*`,
  `@sinclair/typebox` imports) have no Python counterpart; the Python judge
  checks for a `pi_coding_agent` import instead.
- **`vitest.config.ts` / `vitest.test.config.ts`.** Runner configuration is
  expressed by the root `pyproject.toml` pytest settings and the `pp-evals`
  console script.

## Tests

`tests/` ports the four pure-logic TypeScript test files (artifacts,
harness-table, summary, pi-harness model resolution) and adds coverage for the
pytest-specific substitutions: the reporter plugin, the runner CLI, the
`AgentSession` harness, and the runner end to end over the real eval modules.
Everything runs offline -- the session-level tests drive `pi_ai`'s scripted
`faux` provider, so there are no provider calls:

```bash
uv run pytest packages/pp-evals -q
```

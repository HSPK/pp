"""Minimal Python stand-in for the npm `vitest-evals` package, on top of pytest.

The TypeScript package `packages/evals` is built on the npm library
`vitest-evals` (imported as `vitest-evals` and `vitest-evals/harness` by
`packages/evals/src/pi-harness.ts`,
`packages/evals/src/vitest-evals/harness-table.ts`,
`packages/evals/src/vitest-evals/reporter.ts`,
`packages/evals/src/vitest-evals/setup.ts`,
`packages/evals/src/smoke.eval.ts` and
`packages/evals/src/extensions.eval.ts`). **There is no Python equivalent of
`vitest-evals`**, so this module ports the slice of its interface that
`packages/evals` actually uses, and nothing else:

| `vitest-evals` export | Python |
| --- | --- |
| `createHarness` | `create_harness` |
| `Harness` | `Harness` |
| `HarnessContext` | `HarnessContext` |
| `HarnessRun` | `HarnessRun` |
| `SimpleHarnessResult` | `SimpleHarnessResult` |
| `TranscriptEvent` | `TranscriptEvent` (union of the three event dataclasses) |
| `JsonValue` | `JsonValue` |
| `normalizeRecord` | `normalize_record` |
| `toJsonValue` | `to_json_value` |
| `isHarnessRun` | `is_harness_run` |
| `attachHarnessRunToError` / `getHarnessRunFromError` | `attach_harness_run_to_error` / `get_harness_run_from_error` |
| `describeEval` | `describe_eval` |
| `createJudge` | `create_judge` |
| `toolCalls(...)` trace helper | `tool_calls` |

Everything else `vitest-evals` offers (its own reporter, scorer library,
matchers, dataset loaders) is out of scope: `packages/evals` never touches it.

**How declaration works.** `describeEval(name, options, (it) => ...)` is a
Vitest `describe` block whose `it(...)` callbacks receive a `run` function
bound to the suite's harness. The pytest equivalent here generates one async
test function per `it(...)` call and injects it into the calling module's
globals, so plain `pytest` collects it. The test body receives a single
`EvalCase` argument carrying `run`, `task` and the case name, instead of
TypeScript's destructured `{ run, task }` context object.

**How scoring works.** After the body returns, the recorded `HarnessRun` is
scored by every configured judge; the mean score is stored as
`EvalMeta.eval.avg_score`, which is exactly what
`packages/evals/src/vitest-evals/reporter.ts` reads via
`test.meta().eval?.avgScore`. `judge_threshold=None` keeps a low score as an
observation instead of failing the test, matching the TypeScript default
comparative-eval setup.

**How results reach the reporter.** Vitest exposes per-test metadata through
`test.meta()`; here the equivalent `EvalMeta` is stashed on the pytest item
(`EVAL_META_KEY`) and read back by `pi_evals.vitest_evals.reporter`.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Literal, Protocol

import pytest

JsonValue = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
"""Port of `vitest-evals`' `JsonValue`."""


# ==========================================================================
# JSON helpers (`normalizeRecord`, `toJsonValue`)
# ==========================================================================


def to_json_value(value: object) -> JsonValue:
    """Best-effort conversion of an arbitrary value to a `JsonValue`.

    `pi-harness.ts` uses this for tool-result content that is not plain text.
    Dataclasses (this port's stand-in for TypeScript's structural object
    literals) are converted field by field; anything else that is not
    JSON-native falls back to its string form, as `vitest-evals` does.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # Non-finite numbers have no JSON form; JSON.stringify emits `null` for them.
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {name: to_json_value(getattr(value, name)) for name in value.__dataclass_fields__}
    return str(value)


def normalize_record(value: object) -> dict[str, JsonValue]:
    """Port of `normalizeRecord`: coerce a value to a JSON-safe string-keyed record."""
    converted = to_json_value(value)
    if isinstance(converted, dict):
        return converted
    return {}


# ==========================================================================
# Transcript events
# ==========================================================================


@dataclass
class TranscriptMessageEvent:
    role: Literal["user", "assistant", "system"]
    content: str
    type: Literal["message"] = "message"


@dataclass
class TranscriptToolCallEvent:
    id: str
    name: str
    arguments: dict[str, JsonValue] = field(default_factory=dict)
    type: Literal["tool_call"] = "tool_call"


@dataclass
class TranscriptToolResultError:
    message: str


@dataclass
class TranscriptToolResultEvent:
    tool_call_id: str
    name: str
    content: JsonValue = None
    error: TranscriptToolResultError | None = None
    type: Literal["tool_result"] = "tool_result"


TranscriptEvent = TranscriptMessageEvent | TranscriptToolCallEvent | TranscriptToolResultEvent


@dataclass
class ToolCallTrace:
    """One normalized tool call, as `vitest-evals`' `toolCalls(...)` helper yields it."""

    name: str
    status: Literal["ok", "error", "pending"]
    arguments: dict[str, JsonValue] = field(default_factory=dict)
    result: JsonValue = None


def tool_calls(events: Sequence[TranscriptEvent]) -> list[ToolCallTrace]:
    """Port of `vitest-evals`' `toolCalls(...)` trace helper.

    Pairs every `tool_call` event with its matching `tool_result` by id; a
    call with no result stays `"pending"`.
    """
    results: dict[str, TranscriptToolResultEvent] = {
        event.tool_call_id: event for event in events if event.type == "tool_result"
    }
    traces: list[ToolCallTrace] = []
    for event in events:
        if event.type != "tool_call":
            continue
        result = results.get(event.id)
        if result is None:
            traces.append(ToolCallTrace(name=event.name, status="pending", arguments=event.arguments))
            continue
        traces.append(
            ToolCallTrace(
                name=event.name,
                status="error" if result.error else "ok",
                arguments=event.arguments,
                result=result.content,
            )
        )
    return traces


# ==========================================================================
# Harness results and runs
# ==========================================================================


@dataclass
class HarnessUsage:
    """Port of `vitest-evals`' harness `usage`. Field names are serialized to
    `runs.jsonl` in their TypeScript spelling by
    `pi_evals.vitest_evals.reporter`, so they keep their camelCase JSON keys
    there while staying snake_case in Python."""

    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tool_calls: int | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JsonValue]:
        record: dict[str, JsonValue] = {}
        if self.provider is not None:
            record["provider"] = self.provider
        if self.model is not None:
            record["model"] = self.model
        if self.input_tokens is not None:
            record["inputTokens"] = self.input_tokens
        if self.output_tokens is not None:
            record["outputTokens"] = self.output_tokens
        if self.total_tokens is not None:
            record["totalTokens"] = self.total_tokens
        if self.tool_calls is not None:
            record["toolCalls"] = self.tool_calls
        if self.metadata:
            record["metadata"] = dict(self.metadata)
        return record


@dataclass
class HarnessTimings:
    total_ms: float

    def to_json(self) -> dict[str, JsonValue]:
        return {"totalMs": self.total_ms}


@dataclass
class SimpleHarnessResult:
    """Port of `vitest-evals`' `SimpleHarnessResult<TOutput>`."""

    output: object
    events: list[TranscriptEvent] = field(default_factory=list)
    usage: HarnessUsage = field(default_factory=HarnessUsage)
    timings: HarnessTimings | None = None


@dataclass
class HarnessRun:
    """Port of `vitest-evals`' `HarnessRun`: a completed (or failed) harness run."""

    output: object = None
    events: list[TranscriptEvent] = field(default_factory=list)
    usage: HarnessUsage = field(default_factory=HarnessUsage)
    timings: HarnessTimings | None = None
    artifacts: dict[str, JsonValue] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def is_harness_run(value: object) -> bool:
    """Port of `isHarnessRun`."""
    return isinstance(value, HarnessRun)


_HARNESS_RUN_ATTRIBUTE = "__pi_evals_harness_run__"


def attach_harness_run_to_error(error: BaseException, run: HarnessRun) -> BaseException:
    """Port of `attachHarnessRunToError`: carry a partial run on a thrown error."""
    setattr(error, _HARNESS_RUN_ATTRIBUTE, run)
    return error


def get_harness_run_from_error(error: BaseException) -> HarnessRun | None:
    """Port of `getHarnessRunFromError`."""
    run = getattr(error, _HARNESS_RUN_ATTRIBUTE, None)
    return run if isinstance(run, HarnessRun) else None


# ==========================================================================
# Harness
# ==========================================================================


@dataclass
class HarnessContext:
    """Port of `vitest-evals`' `HarnessContext`.

    `signal` is this port's `AbortSignal` analogue (`pi_ai.utils.abort.AbortSignal`
    duck-typed here as anything with `throw_if_aborted()`), matching the
    monorepo's documented `AbortSignal` porting convention.
    """

    artifacts: dict[str, JsonValue] = field(default_factory=dict)
    signal: Any = None

    def set_artifact(self, name: str, value: JsonValue) -> None:
        self.artifacts[name] = value


class Harness(Protocol):
    """Port of `vitest-evals`' `Harness<TInput, TOutput>`."""

    name: str

    async def run(self, input: object, context: HarnessContext) -> HarnessRun: ...


HarnessRunFunction = Callable[..., SimpleHarnessResult | Awaitable[SimpleHarnessResult]]


@dataclass
class _CreatedHarness:
    name: str
    _run: HarnessRunFunction

    async def run(self, input: object, context: HarnessContext) -> HarnessRun:
        result = self._run(input=input, signal=context.signal, set_artifact=context.set_artifact)
        if inspect.isawaitable(result):
            result = await result
        return HarnessRun(
            output=result.output,
            events=list(result.events),
            usage=result.usage,
            timings=result.timings,
            artifacts=dict(context.artifacts),
        )


def create_harness(*, name: str, run: HarnessRunFunction) -> Harness:
    """Port of `createHarness({ name, run })`.

    `run` is called with the keyword arguments `input`, `signal` and
    `set_artifact` (TypeScript destructures the same three from one object)
    and may be sync or async.
    """
    return _CreatedHarness(name=name, _run=run)


# ==========================================================================
# Judges
# ==========================================================================


@dataclass
class JudgeContext:
    """What a judge function receives. TypeScript destructures `{ output, toolCalls }`."""

    input: object
    output: object
    events: list[TranscriptEvent] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCallTrace]:
        return tool_calls(self.events)


@dataclass
class JudgeResult:
    score: float
    metadata: dict[str, JsonValue] = field(default_factory=dict)


JudgeFunction = Callable[[JudgeContext], JudgeResult | Awaitable[JudgeResult]]


@dataclass
class Judge:
    """Port of `vitest-evals`' `Judge`."""

    name: str
    judge: JudgeFunction

    async def score(self, context: JudgeContext) -> JudgeResult:
        result = self.judge(context)
        if inspect.isawaitable(result):
            result = await result
        return result


def create_judge(name: str, judge: JudgeFunction) -> Judge:
    """Port of `createJudge(name, fn)`."""
    return Judge(name=name, judge=judge)


# ==========================================================================
# Test declaration (`describeEval`)
# ==========================================================================


@dataclass
class TestArtifactAttachment:
    """Port of Vitest's `TestAttachment` for the two artifact kinds this package records."""

    __test__ = False
    """Not a pytest test class, despite the TypeScript-derived name."""

    name: str
    content_type: str
    body: str
    body_encoding: Literal["utf-8"] = "utf-8"


@dataclass
class TestArtifact:
    """Port of Vitest's `TestArtifact` for `@earendil-works/pi-evals:*` artifacts."""

    __test__ = False
    """Not a pytest test class, despite the TypeScript-derived name."""

    type: str
    run_id: str | None = None
    attachments: list[TestArtifactAttachment] = field(default_factory=list)
    annotation: dict[str, JsonValue] | None = None


@dataclass
class EvalTask:
    """Port of Vitest's `RunnerTestCase` as far as this package uses it.

    `packages/evals` only ever reads `task.artifacts` and `task.meta.harness`,
    so those are the only members here.
    """

    name: str = ""
    artifacts: list[TestArtifact] = field(default_factory=list)


@dataclass
class EvalScore:
    """Port of the `eval` slice of Vitest's per-test meta: `test.meta().eval`."""

    avg_score: float | None = None
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalHarnessMeta:
    """Port of the `harness` slice of Vitest's per-test meta: `test.meta().harness`."""

    name: str
    run: HarnessRun | None = None


@dataclass
class EvalMeta:
    """Everything `pi_evals.vitest_evals.reporter` needs from one eval test."""

    eval_name: str
    case_name: str
    task: EvalTask
    file: str = ""
    harness: EvalHarnessMeta | None = None
    eval: EvalScore | None = None
    status: Literal["passed", "failed", "skipped"] = "passed"


EVAL_META_KEY: pytest.StashKey[EvalMeta] = pytest.StashKey()
"""Stash key under which each eval test's `EvalMeta` is attached to its pytest item."""


@dataclass
class EvalRunResult:
    """What `run(...)` returns inside an eval body.

    Mirrors the object `vitest-evals` hands back: the harness output plus the
    run's `errors`, `usage`, `timings` and `artifacts`.
    """

    output: Any
    events: list[TranscriptEvent] = field(default_factory=list)
    usage: HarnessUsage = field(default_factory=HarnessUsage)
    timings: HarnessTimings | None = None
    artifacts: dict[str, JsonValue] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class EvalCase:
    """The single argument an eval body receives, replacing `{ run, task }`."""

    def __init__(self, name: str, harness: Harness, meta: EvalMeta) -> None:
        self.name = name
        self.harness = harness
        self.meta = meta
        self.task = meta.task
        self._last_input: object = None

    async def run(self, input: object) -> EvalRunResult:
        context = HarnessContext()
        self.meta.harness = EvalHarnessMeta(name=self.harness.name)
        self._last_input = input
        try:
            run = await self.harness.run(input, context)
        except Exception as error:
            # A failed run still carries telemetry the reporter needs, so record
            # it (with the error message) before propagating, as `vitest-evals`
            # does through `attachHarnessRunToError`.
            partial = get_harness_run_from_error(error)
            run = partial if partial is not None else HarnessRun(artifacts=dict(context.artifacts))
            run.errors = [*run.errors, str(error) or type(error).__name__]
            self.meta.harness.run = run
            raise
        self.meta.harness.run = run
        return EvalRunResult(
            output=run.output,
            events=list(run.events),
            usage=run.usage,
            timings=run.timings,
            artifacts=dict(run.artifacts),
            errors=list(run.errors),
        )


@dataclass
class EvalOptions:
    """Port of `describeEval`'s options object."""

    harness: Harness
    judges: list[Judge] = field(default_factory=list)
    judge_threshold: float | None = 1.0
    """`None` records a low score as an observation instead of failing the test."""


EvalBody = Callable[[EvalCase], Awaitable[None]]


class EvalRegistrar:
    """The `it` callback `describeEval`'s definition function receives."""

    def __init__(self, eval_name: str, options: EvalOptions, namespace: dict[str, Any], suffix: str) -> None:
        self._eval_name = eval_name
        self._options = options
        self._namespace = namespace
        self._suffix = suffix

    def __call__(self, case_name: str, body: EvalBody) -> None:
        eval_name = self._eval_name
        options = self._options

        async def test_function(request: pytest.FixtureRequest) -> None:
            meta = EvalMeta(eval_name=eval_name, case_name=case_name, task=EvalTask(name=case_name))
            request.node.stash[EVAL_META_KEY] = meta
            case = EvalCase(case_name, options.harness, meta)
            await body(case)
            await _score_eval_case(case, options)

        test_function.__name__ = _test_function_name(eval_name, case_name, self._suffix)
        test_function.__qualname__ = test_function.__name__
        test_function.__doc__ = f"{eval_name} > {case_name}"
        self._namespace[test_function.__name__] = test_function


async def _score_eval_case(case: EvalCase, options: EvalOptions) -> None:
    if not options.judges:
        return
    harness_meta = case.meta.harness
    run = harness_meta.run if harness_meta else None
    if run is None:
        return
    scores: dict[str, float] = {}
    for judge in options.judges:
        context = JudgeContext(input=case._last_input, output=run.output, events=run.events)
        result = await judge.score(context)
        scores[judge.name] = result.score
    average = sum(scores.values()) / len(scores)
    case.meta.eval = EvalScore(avg_score=average, scores=scores)
    if options.judge_threshold is not None and average < options.judge_threshold:
        raise AssertionError(
            f"{case.meta.eval_name} > {case.name}: average judge score {average} is below threshold "
            f"{options.judge_threshold} ({scores})"
        )


def _test_function_name(eval_name: str, case_name: str, suffix: str) -> str:
    parts = [_slugify(eval_name), _slugify(case_name)]
    if suffix:
        parts.append(_slugify(suffix))
    return "test_" + "__".join(part for part in parts if part)


def _slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", value)).strip("_").lower()


def describe_eval(
    eval_name: str,
    options: EvalOptions,
    define: Callable[[EvalRegistrar], None],
    *,
    suffix: str = "",
    namespace: dict[str, Any] | None = None,
) -> None:
    """Port of `describeEval(name, options, (it) => ...)`.

    Generates one pytest test function per `it(...)` call and injects it into
    the calling module's globals (`namespace` overrides that, which keeps the
    generator testable). `suffix` disambiguates the generated names when the
    same eval is declared once per harness-table row -- the equivalent of
    Vitest's `describe.for(table)("$name repetition $repetition", ...)`.
    """
    if namespace is None:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is None:
            raise RuntimeError("describe_eval could not determine the calling module namespace.")
        namespace = caller.f_globals
    define(EvalRegistrar(eval_name, options, namespace, suffix))


def canonical_json(value: JsonValue) -> str:
    """`JSON.stringify` with no whitespace, for keys that must match the TypeScript byte for byte."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


__all__ = [
    "EVAL_META_KEY",
    "EvalCase",
    "EvalHarnessMeta",
    "EvalMeta",
    "EvalOptions",
    "EvalRegistrar",
    "EvalRunResult",
    "EvalScore",
    "EvalTask",
    "Harness",
    "HarnessContext",
    "HarnessRun",
    "HarnessTimings",
    "HarnessUsage",
    "JsonValue",
    "Judge",
    "JudgeContext",
    "JudgeResult",
    "SimpleHarnessResult",
    "TestArtifact",
    "TestArtifactAttachment",
    "ToolCallTrace",
    "TranscriptEvent",
    "TranscriptMessageEvent",
    "TranscriptToolCallEvent",
    "TranscriptToolResultError",
    "TranscriptToolResultEvent",
    "attach_harness_run_to_error",
    "canonical_json",
    "create_harness",
    "create_judge",
    "describe_eval",
    "get_harness_run_from_error",
    "is_harness_run",
    "normalize_record",
    "to_json_value",
    "tool_calls",
]

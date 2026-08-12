"""Tests for `pi_evals.harness`.

This module has no TypeScript test to port: it stands in for the npm package
`vitest-evals`, whose interface `packages/evals` builds on. That substitution
is exactly why it needs direct tests -- nothing upstream pins its behavior, so
these tests are the only thing keeping it honest.

The end-to-end coverage in `test_run_evals_end_to_end.py` drives this module
through a pytest subprocess, which means coverage tooling cannot see it. These
tests exercise the same code in-process.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pi_evals.harness import (
    EvalCase,
    EvalMeta,
    EvalOptions,
    EvalTask,
    HarnessContext,
    HarnessRun,
    HarnessTimings,
    HarnessUsage,
    JudgeResult,
    SimpleHarnessResult,
    TranscriptMessageEvent,
    TranscriptToolCallEvent,
    TranscriptToolResultError,
    TranscriptToolResultEvent,
    attach_harness_run_to_error,
    canonical_json,
    create_harness,
    create_judge,
    describe_eval,
    get_harness_run_from_error,
    is_harness_run,
    normalize_record,
    to_json_value,
    tool_calls,
)

# --------------------------------------------------------------------------
# to_json_value / normalize_record
# --------------------------------------------------------------------------


@dataclass
class Point:
    x: int
    y: str


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        ("text", "text"),
        (7, 7),
        (1.5, 1.5),
        ([1, "a"], [1, "a"]),
        ((1, 2), [1, 2]),
        ({"k": 1}, {"k": 1}),
        ({1: "a"}, {"1": "a"}),
    ],
)
def test_to_json_value_passes_json_native_values_through(value, expected):
    assert to_json_value(value) == expected


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_to_json_value_maps_non_finite_numbers_to_none(value):
    # JSON has no representation for these; `JSON.stringify` emits `null`.
    assert to_json_value(value) is None


def test_to_json_value_converts_dataclasses_field_by_field():
    assert to_json_value(Point(x=1, y="two")) == {"x": 1, "y": "two"}


def test_to_json_value_recurses_through_containers():
    assert to_json_value({"points": [Point(x=1, y="a")]}) == {"points": [{"x": 1, "y": "a"}]}


def test_to_json_value_falls_back_to_the_string_form():
    class Opaque:
        def __str__(self) -> str:
            return "opaque!"

    assert to_json_value(Opaque()) == "opaque!"


def test_to_json_value_converts_a_dataclass_type_itself_by_string():
    # The class object is not an instance, so it takes the fallback path.
    assert isinstance(to_json_value(Point), str)


def test_normalize_record_keeps_records_and_drops_everything_else():
    assert normalize_record({"a": 1}) == {"a": 1}
    assert normalize_record(Point(x=1, y="a")) == {"x": 1, "y": "a"}
    assert normalize_record([1, 2]) == {}
    assert normalize_record("text") == {}
    assert normalize_record(None) == {}


# --------------------------------------------------------------------------
# tool_calls
# --------------------------------------------------------------------------


def test_tool_calls_pairs_each_call_with_its_result():
    events = [
        TranscriptMessageEvent(role="user", content="hi"),
        TranscriptToolCallEvent(id="c1", name="read", arguments={"path": "a.txt"}),
        TranscriptToolResultEvent(tool_call_id="c1", name="read", content="file body"),
    ]

    traces = tool_calls(events)

    assert len(traces) == 1
    assert traces[0].name == "read"
    assert traces[0].status == "ok"
    assert traces[0].arguments == {"path": "a.txt"}
    assert traces[0].result == "file body"


def test_tool_calls_marks_a_failed_result_as_error():
    events = [
        TranscriptToolCallEvent(id="c1", name="bash"),
        TranscriptToolResultEvent(
            tool_call_id="c1", name="bash", content="boom", error=TranscriptToolResultError(message="boom")
        ),
    ]

    assert tool_calls(events)[0].status == "error"


def test_tool_calls_marks_a_call_with_no_result_as_pending():
    traces = tool_calls([TranscriptToolCallEvent(id="c1", name="read", arguments={"path": "a"})])

    assert traces[0].status == "pending"
    assert traces[0].result is None
    assert traces[0].arguments == {"path": "a"}


def test_tool_calls_ignores_unrelated_events_and_preserves_call_order():
    events = [
        TranscriptToolCallEvent(id="c1", name="first"),
        TranscriptMessageEvent(role="assistant", content="thinking"),
        TranscriptToolCallEvent(id="c2", name="second"),
        TranscriptToolResultEvent(tool_call_id="c2", name="second", content=2),
        TranscriptToolResultEvent(tool_call_id="c1", name="first", content=1),
    ]

    assert [trace.name for trace in tool_calls(events)] == ["first", "second"]
    assert [trace.result for trace in tool_calls(events)] == [1, 2]


def test_tool_calls_of_an_empty_transcript_is_empty():
    assert tool_calls([]) == []


# --------------------------------------------------------------------------
# HarnessRun error carrying
# --------------------------------------------------------------------------


def test_is_harness_run_identifies_runs():
    assert is_harness_run(HarnessRun()) is True
    assert is_harness_run({"output": "x"}) is False
    assert is_harness_run(None) is False


def test_a_harness_run_round_trips_through_an_error():
    run = HarnessRun(output="partial")
    error = attach_harness_run_to_error(RuntimeError("boom"), run)

    assert get_harness_run_from_error(error) is run


def test_an_error_without_an_attached_run_yields_none():
    assert get_harness_run_from_error(RuntimeError("boom")) is None


# --------------------------------------------------------------------------
# create_harness
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_harness_runs_an_async_function_and_collects_artifacts():
    async def run(*, input, signal, set_artifact):
        set_artifact("runId", "abc")
        return SimpleHarnessResult(
            output=f"saw {input}",
            events=[TranscriptMessageEvent(role="user", content=str(input))],
            usage=HarnessUsage(provider="faux", model="m", total_tokens=3),
            timings=HarnessTimings(total_ms=1.5),
        )

    harness = create_harness(name="h", run=run)
    result = await harness.run("hello", HarnessContext())

    assert harness.name == "h"
    assert result.output == "saw hello"
    assert result.usage.total_tokens == 3
    assert result.timings.total_ms == 1.5
    assert result.artifacts == {"runId": "abc"}


@pytest.mark.asyncio
async def test_create_harness_also_accepts_a_sync_run_function():
    def run(*, input, signal, set_artifact):
        return SimpleHarnessResult(output=input)

    result = await create_harness(name="sync", run=run).run("value", HarnessContext())

    assert result.output == "value"


@pytest.mark.asyncio
async def test_create_harness_passes_the_signal_through():
    seen: dict[str, object] = {}

    def run(*, input, signal, set_artifact):
        seen["signal"] = signal
        return SimpleHarnessResult(output=None)

    sentinel = object()
    await create_harness(name="h", run=run).run(None, HarnessContext(signal=sentinel))

    assert seen["signal"] is sentinel


@pytest.mark.asyncio
async def test_harness_run_artifacts_are_copied_not_shared():
    def run(*, input, signal, set_artifact):
        set_artifact("a", 1)
        return SimpleHarnessResult(output=None)

    context = HarnessContext()
    result = await create_harness(name="h", run=run).run(None, context)
    context.set_artifact("b", 2)

    assert result.artifacts == {"a": 1}


# --------------------------------------------------------------------------
# EvalCase
# --------------------------------------------------------------------------


def make_meta() -> EvalMeta:
    return EvalMeta(eval_name="Suite", case_name="case", task=EvalTask(name="case"))


@pytest.mark.asyncio
async def test_eval_case_run_records_the_harness_run_on_the_meta():
    harness = create_harness(name="h", run=lambda **_: SimpleHarnessResult(output="out"))
    meta = make_meta()

    result = await EvalCase("case", harness, meta).run("in")

    assert result.output == "out"
    assert meta.harness.name == "h"
    assert meta.harness.run.output == "out"


@pytest.mark.asyncio
async def test_eval_case_run_records_telemetry_then_reraises_on_failure():
    def run(**_):
        raise RuntimeError("harness exploded")

    meta = make_meta()

    with pytest.raises(RuntimeError, match="harness exploded"):
        await EvalCase("case", create_harness(name="h", run=run), meta).run("in")

    # The reporter still needs the failed run, so it must be recorded.
    assert meta.harness.run.errors == ["harness exploded"]


@pytest.mark.asyncio
async def test_eval_case_run_prefers_a_partial_run_attached_to_the_error():
    def run(**_):
        raise attach_harness_run_to_error(RuntimeError("late failure"), HarnessRun(output="partial"))

    meta = make_meta()

    with pytest.raises(RuntimeError):
        await EvalCase("case", create_harness(name="h", run=run), meta).run("in")

    assert meta.harness.run.output == "partial"
    assert meta.harness.run.errors == ["late failure"]


@pytest.mark.asyncio
async def test_eval_case_records_a_bare_error_type_when_there_is_no_message():
    def run(**_):
        raise RuntimeError

    meta = make_meta()
    with pytest.raises(RuntimeError):
        await EvalCase("case", create_harness(name="h", run=run), meta).run("in")

    assert meta.harness.run.errors == ["RuntimeError"]


# --------------------------------------------------------------------------
# Judges and describe_eval
# --------------------------------------------------------------------------


def always(score: float):
    """A judge that always returns `score`. Judges return a `JudgeResult`, not a bare float."""
    return create_judge(f"judge-{score}", lambda context: JudgeResult(score=score))


@pytest.mark.asyncio
async def test_describe_eval_generates_a_test_per_case():
    namespace: dict[str, object] = {}
    harness = create_harness(name="h", run=lambda **_: SimpleHarnessResult(output="ok"))

    def define(it):
        async def body(case):
            await case.run("input")

        it("first case", body)
        it("second case", body)

    describe_eval("My Suite", EvalOptions(harness=harness), define, namespace=namespace)

    assert sorted(namespace) == ["test_my_suite__first_case", "test_my_suite__second_case"]
    assert namespace["test_my_suite__first_case"].__doc__ == "My Suite > first case"


def test_describe_eval_suffix_disambiguates_repeated_declarations():
    namespace: dict[str, object] = {}
    harness = create_harness(name="h", run=lambda **_: SimpleHarnessResult(output="ok"))

    for suffix in ("baseline", "candidate"):
        describe_eval(
            "Suite",
            EvalOptions(harness=harness),
            lambda it: it("case", lambda case: None),
            suffix=suffix,
            namespace=namespace,
        )

    assert sorted(namespace) == ["test_suite__case__baseline", "test_suite__case__candidate"]


def test_describe_eval_slugifies_punctuation_into_single_underscores():
    namespace: dict[str, object] = {}
    describe_eval(
        "Pi: extension  authoring!",
        EvalOptions(harness=create_harness(name="h", run=lambda **_: SimpleHarnessResult())),
        lambda it: it("creates, reloads & uses", lambda case: None),
        namespace=namespace,
    )

    assert list(namespace) == ["test_pi_extension_authoring__creates_reloads_uses"]


@pytest.mark.asyncio
async def test_judges_average_and_pass_above_the_threshold():
    namespace: dict[str, object] = {}
    harness = create_harness(name="h", run=lambda **_: SimpleHarnessResult(output="ok"))
    options = EvalOptions(harness=harness, judges=[always(1.0), always(1.0)])

    describe_eval("S", options, lambda it: it("c", lambda case: case.run("in")), namespace=namespace)
    await _invoke(namespace["test_s__c"])


@pytest.mark.asyncio
async def test_a_below_threshold_average_fails_the_test():
    namespace: dict[str, object] = {}
    harness = create_harness(name="h", run=lambda **_: SimpleHarnessResult(output="ok"))
    options = EvalOptions(harness=harness, judges=[always(1.0), always(0.0)])

    describe_eval("S", options, lambda it: it("c", lambda case: case.run("in")), namespace=namespace)

    with pytest.raises(AssertionError, match="below threshold"):
        await _invoke(namespace["test_s__c"])


@pytest.mark.asyncio
async def test_a_none_threshold_records_a_low_score_without_failing():
    namespace: dict[str, object] = {}
    harness = create_harness(name="h", run=lambda **_: SimpleHarnessResult(output="ok"))
    options = EvalOptions(harness=harness, judges=[always(0.0)], judge_threshold=None)

    describe_eval("S", options, lambda it: it("c", lambda case: case.run("in")), namespace=namespace)
    meta = await _invoke(namespace["test_s__c"])

    assert meta.eval.avg_score == 0.0


@pytest.mark.asyncio
async def test_no_judges_leaves_the_score_unset():
    namespace: dict[str, object] = {}
    harness = create_harness(name="h", run=lambda **_: SimpleHarnessResult(output="ok"))

    describe_eval(
        "S", EvalOptions(harness=harness), lambda it: it("c", lambda case: case.run("in")), namespace=namespace
    )
    meta = await _invoke(namespace["test_s__c"])

    assert meta.eval is None


@pytest.mark.asyncio
async def test_a_judge_sees_the_output_and_tool_calls():
    seen: dict[str, object] = {}

    def judge(context):
        seen["output"] = context.output
        seen["tool_calls"] = context.tool_calls
        return JudgeResult(score=1.0)

    events = [
        TranscriptToolCallEvent(id="c1", name="read"),
        TranscriptToolResultEvent(tool_call_id="c1", name="read", content="body"),
    ]
    harness = create_harness(name="h", run=lambda **_: SimpleHarnessResult(output="answer", events=events))
    namespace: dict[str, object] = {}
    options = EvalOptions(harness=harness, judges=[create_judge("j", judge)])

    describe_eval("S", options, lambda it: it("c", lambda case: case.run("in")), namespace=namespace)
    await _invoke(namespace["test_s__c"])

    assert seen["output"] == "answer"
    assert [trace.name for trace in seen["tool_calls"]] == ["read"]


# --------------------------------------------------------------------------
# canonical_json
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"b": 1, "a": 2}, '{"b":1,"a":2}'),
        ([1, "two", None, True], '[1,"two",null,true]'),
        ("汉字", '"汉字"'),
    ],
)
def test_canonical_json_matches_json_stringify(value, expected):
    # No whitespace and no ASCII escaping, so the bytes match the TypeScript.
    assert canonical_json(value) == expected


# --------------------------------------------------------------------------


class _StubStash(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)


async def _invoke(test_function) -> EvalMeta:
    """Run a generated eval test outside pytest and return its recorded meta."""

    class Node:
        def __init__(self) -> None:
            self.stash: dict[object, EvalMeta] = _StubStash()

    class Request:
        def __init__(self) -> None:
            self.node = Node()

    request = Request()
    await test_function(request)
    return next(iter(request.node.stash.values()))

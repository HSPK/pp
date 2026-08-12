"""Tests for the small core-utility ports: cache-stats, exec, event-bus,
timings, tool-definition-wrapper, and deprecation.

Ported behaviour reference:
- `packages/coding-agent/src/core/cache-stats.ts`
- `packages/coding-agent/src/core/exec.ts`
- `packages/coding-agent/src/core/event-bus.ts`
- `packages/coding-agent/src/core/timings.ts`
- `packages/coding-agent/src/core/tools/tool-definition-wrapper.ts`
- `packages/coding-agent/src/utils/deprecation.ts`
"""

from __future__ import annotations

import asyncio
import time

import pytest
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import AssistantMessage, Cost, Model, ModelCost, TextContent, Usage
from pi_ai.utils.abort import AbortController
from pi_coding_agent.core import timings as timings_module
from pi_coding_agent.core.cache_stats import (
    CACHE_TTL_MS,
    CacheMiss,
    CacheWasteTotals,
    collect_cache_misses,
    compute_cache_waste,
    detect_cache_miss,
)
from pi_coding_agent.core.event_bus import create_event_bus
from pi_coding_agent.core.exec import ExecOptions, exec_command
from pi_coding_agent.core.extensions.types import ToolDefinition
from pi_coding_agent.core.session_manager import (
    BranchSummaryEntry,
    CompactionEntry,
    SessionMessageEntry,
)
from pi_coding_agent.tools.tool_definition_wrapper import (
    create_tool_definition_from_agent_tool,
    wrap_tool_definition,
    wrap_tool_definitions,
)
from pi_coding_agent.utils.deprecation import (
    clear_deprecation_warnings_for_tests,
    warn_deprecation,
)

# ---------------------------------------------------------------------------
# cache_stats
# ---------------------------------------------------------------------------


def _usage(
    input_tokens: int = 0,
    output: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    cost_input: float = 0.0,
    cost_cache_read: float = 0.0,
    cost_cache_write: float = 0.0,
) -> Usage:
    return Usage(
        input=input_tokens,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output + cache_read + cache_write,
        cost=Cost(input=cost_input, cache_read=cost_cache_read, cache_write=cost_cache_write),
    )


def _assistant(
    usage: Usage,
    timestamp: int,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-5",
) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(text="hi")],
        usage=usage,
        stop_reason="stop",
        timestamp=timestamp,
        api="anthropic-messages",
        provider=provider,
        model=model,
    )


def _msg_entry(entry_id: str, parent_id: str | None, message: AssistantMessage) -> SessionMessageEntry:
    return SessionMessageEntry(id=entry_id, parent_id=parent_id, timestamp="2025-01-01T00:00:00Z", message=message)


class _FakeModels:
    """Minimal `ModelPriceSource` stand-in."""

    def __init__(self, cache_read_rate: float) -> None:
        self._cache_read_rate = cache_read_rate

    def get_model(self, provider: str, model_id: str) -> Model | None:
        return Model(id=model_id, provider=provider, cost=ModelCost(cache_read=self._cache_read_rate))


def test_cache_ttl_ms_matches_five_minutes() -> None:
    assert CACHE_TTL_MS == 5 * 60 * 1000


def test_no_miss_on_first_turn() -> None:
    entries = [_msg_entry("1", None, _assistant(_usage(input_tokens=5000), timestamp=0))]
    totals = compute_cache_waste(entries, _FakeModels(1.0))
    assert totals == CacheWasteTotals(missed_tokens=0, missed_cost=0.0, missed_count=0)


def test_no_miss_when_provider_never_reports_cache() -> None:
    """Two turns with cache_read=cache_write=0 throughout: a provider that has
    no cache support at all should never be flagged, even though every token
    is technically "re-billed"."""
    entries = [
        _msg_entry("1", None, _assistant(_usage(input_tokens=5000), timestamp=0)),
        _msg_entry("2", "1", _assistant(_usage(input_tokens=5000), timestamp=1000)),
    ]
    totals = compute_cache_waste(entries, _FakeModels(1.0))
    assert totals.missed_count == 0


def test_miss_below_noise_floor_is_not_counted() -> None:
    entries = [
        _msg_entry("1", None, _assistant(_usage(cache_write=1024, cost_cache_write=0.01), timestamp=0)),
        _msg_entry("2", "1", _assistant(_usage(cache_write=1024, cost_cache_write=0.01), timestamp=1000)),
    ]
    totals = compute_cache_waste(entries, _FakeModels(1.0))
    assert totals.missed_count == 0


def test_miss_above_noise_floor_is_counted_with_cost() -> None:
    entries = [
        _msg_entry("1", None, _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=1000)),
        _msg_entry(
            "2",
            "1",
            _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=1000 + CACHE_TTL_MS + 1000),
        ),
    ]
    misses = collect_cache_misses(entries, _FakeModels(1.0))
    assert len(misses) == 1
    miss = next(iter(misses.values()))
    assert miss.missed_tokens == 2000
    # paid_per_token = 0.01 / 2000 = 5e-6 ; read_per_token = 1.0 / 1e6 = 1e-6
    assert miss.missed_cost == pytest.approx(2000 * (5e-6 - 1e-6))
    assert miss.idle_ms == CACHE_TTL_MS + 1000
    assert miss.model_changed is False

    totals = compute_cache_waste(entries, _FakeModels(1.0))
    assert totals.missed_tokens == 2000
    assert totals.missed_count == 1
    assert totals.missed_cost == pytest.approx(miss.missed_cost)


def test_model_change_is_still_counted_and_flagged() -> None:
    entries = [
        _msg_entry(
            "1", None, _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=0, model="model-a")
        ),
        _msg_entry(
            "2", "1", _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=1000, model="model-b")
        ),
    ]
    misses = collect_cache_misses(entries, _FakeModels(1.0))
    assert len(misses) == 1
    assert next(iter(misses.values())).model_changed is True


def test_compaction_resets_the_scan() -> None:
    entries = [
        _msg_entry("1", None, _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=0)),
        CompactionEntry(
            id="c1",
            parent_id="1",
            timestamp="2025-01-01T00:00:00Z",
            summary="s",
            first_kept_entry_id="2",
            tokens_before=1,
        ),
        _msg_entry("2", "c1", _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=1000)),
    ]
    totals = compute_cache_waste(entries, _FakeModels(1.0))
    assert totals.missed_count == 0


def test_branch_summary_resets_the_scan() -> None:
    entries = [
        _msg_entry("1", None, _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=0)),
        BranchSummaryEntry(id="b1", parent_id="1", timestamp="2025-01-01T00:00:00Z", from_id="1", summary="s"),
        _msg_entry("2", "b1", _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=1000)),
    ]
    totals = compute_cache_waste(entries, _FakeModels(1.0))
    assert totals.missed_count == 0


def test_cache_read_only_provider_total_miss_is_counted() -> None:
    """A provider that only ever reports `cache_read` (never `cache_write`, e.g.
    OpenAI-style): once cache activity has been reported, a turn with zero
    cache_read/cache_write is a total miss, not "no cache support"."""
    entries = [
        _msg_entry("1", None, _assistant(_usage(input_tokens=100, cache_read=5000), timestamp=0)),
        _msg_entry("2", "1", _assistant(_usage(input_tokens=5000, cost_input=0.025), timestamp=1000)),
    ]
    totals = compute_cache_waste(entries, _FakeModels(1.0))
    assert totals.missed_count == 1
    assert totals.missed_tokens == 5000


def test_detect_cache_miss_on_message_not_yet_in_entries() -> None:
    entries = [_msg_entry("1", None, _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=0))]
    new_message = _assistant(_usage(cache_write=2000, cost_cache_write=0.01), timestamp=1000)
    miss = detect_cache_miss(entries, new_message, _FakeModels(1.0))
    assert miss is not None
    assert miss.missed_tokens == 2000

    # Sanity: appending it and re-scanning agrees.
    full_entries = [*entries, _msg_entry("2", "1", new_message)]
    misses = collect_cache_misses(full_entries, _FakeModels(1.0))
    assert len(misses) == 1
    assert next(iter(misses.values())) == miss


def test_cache_miss_dataclass_fields() -> None:
    miss = CacheMiss(missed_tokens=1, missed_cost=0.1, idle_ms=2, model_changed=True)
    assert miss.missed_tokens == 1
    assert miss.model_changed is True


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_command_captures_stdout_and_exit_code() -> None:
    result = await exec_command("echo", ["hello world"], ".")
    assert result.stdout == "hello world\n"
    assert result.stderr == ""
    assert result.code == 0
    assert result.killed is False


@pytest.mark.asyncio
async def test_exec_command_captures_stderr_and_nonzero_exit_code() -> None:
    result = await exec_command("bash", ["-c", "echo oops 1>&2; exit 7"], ".")
    assert result.stdout == ""
    assert result.stderr == "oops\n"
    assert result.code == 7


@pytest.mark.asyncio
async def test_exec_command_respects_cwd() -> None:
    result = await exec_command("pwd", [], "/tmp")
    assert result.stdout.strip() == "/tmp"


@pytest.mark.asyncio
async def test_exec_command_times_out_and_kills_process() -> None:
    """A signal-terminated child reports code 0, as Node's `code ?? 0` does.

    Node resolves `null` for a signal death and `execCommand` maps that to 0,
    so `killed` is the flag that says the command did not finish - not `code`.
    """
    result = await exec_command("sleep", ["5"], ".", ExecOptions(timeout=100))
    assert result.killed is True
    assert result.code == 0


@pytest.mark.asyncio
async def test_exec_command_aborts_via_signal() -> None:
    controller = AbortController()

    async def _abort_soon() -> None:
        await asyncio.sleep(0.05)
        controller.abort()

    task = asyncio.ensure_future(_abort_soon())
    result = await exec_command("sleep", ["5"], ".", ExecOptions(signal=controller.signal))
    await task
    assert result.killed is True


@pytest.mark.asyncio
async def test_exec_command_already_aborted_signal_kills_immediately() -> None:
    controller = AbortController()
    controller.abort()
    result = await exec_command("sleep", ["5"], ".", ExecOptions(signal=controller.signal))
    assert result.killed is True


# ---------------------------------------------------------------------------
# event_bus
# ---------------------------------------------------------------------------


async def _settle(ticks: int = 5) -> None:
    """Yield a fixed number of event-loop ticks so queued handlers can run.

    The event-bus cases below need "the loop has had its chance" -- for the
    positive ones so an async handler completes, for the negative ones so an
    unsubscribed handler would have had every opportunity to fire. Spelling
    that as `await asyncio.sleep(0.01)` ties the claim to wall-clock time,
    which is unreliable now that the suite runs in parallel; a fixed tick count
    is the same claim, deterministic and instant.
    """
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_event_bus_emit_and_on() -> None:
    bus = create_event_bus()
    received: list[object] = []
    bus.on("channel", lambda data: received.append(data))
    bus.emit("channel", {"foo": "bar"})
    # Node's EventEmitter dispatches synchronously; no yield should be needed.
    assert received == [{"foo": "bar"}]


@pytest.mark.asyncio
async def test_event_bus_unsubscribe_stops_delivery() -> None:
    bus = create_event_bus()
    received: list[object] = []
    unsubscribe = bus.on("channel", lambda data: received.append(data))
    unsubscribe()
    bus.emit("channel", "should not arrive")
    await _settle()
    assert received == []


@pytest.mark.asyncio
async def test_event_bus_async_handler_runs_up_to_its_first_await() -> None:
    """Matches JS: an async function body runs synchronously until it awaits."""
    bus = create_event_bus()
    order: list[str] = []

    async def handler(data: object) -> None:
        order.append("start")
        await asyncio.sleep(0)
        order.append("end")

    bus.on("channel", lambda data: order.append("sync"))
    bus.on("channel", handler)
    bus.emit("channel", 1)

    assert order == ["sync", "start"]
    await _settle()
    assert order == ["sync", "start", "end"]


def test_event_bus_delivers_without_a_running_event_loop() -> None:
    """`emit` from sync code must not silently drop the event."""
    bus = create_event_bus()
    received: list[object] = []
    bus.on("channel", lambda data: received.append(data))
    bus.emit("channel", "sync-context")
    assert received == ["sync-context"]


@pytest.mark.asyncio
async def test_event_bus_supports_async_handlers() -> None:
    bus = create_event_bus()
    received: list[object] = []

    async def handler(data: object) -> None:
        await asyncio.sleep(0)
        received.append(data)

    bus.on("channel", handler)
    bus.emit("channel", 42)
    await _settle()
    assert received == [42]


@pytest.mark.asyncio
async def test_event_bus_handler_error_is_isolated(capsys: pytest.CaptureFixture[str]) -> None:
    bus = create_event_bus()
    received: list[object] = []

    def bad_handler(_data: object) -> None:
        raise ValueError("boom")

    bus.on("channel", bad_handler)
    bus.on("channel", lambda data: received.append(data))
    bus.emit("channel", "value")
    await _settle()
    assert received == ["value"]
    captured = capsys.readouterr()
    assert "Event handler error (channel)" in captured.err


@pytest.mark.asyncio
async def test_event_bus_clear_removes_all_handlers() -> None:
    bus = create_event_bus()
    received: list[object] = []
    bus.on("channel", lambda data: received.append(data))
    bus.clear()
    bus.emit("channel", "value")
    await _settle()
    assert received == []


@pytest.mark.asyncio
async def test_event_bus_channels_are_independent() -> None:
    bus = create_event_bus()
    received_a: list[object] = []
    received_b: list[object] = []
    bus.on("a", lambda data: received_a.append(data))
    bus.on("b", lambda data: received_b.append(data))
    bus.emit("a", 1)
    await _settle()
    assert received_a == [1]
    assert received_b == []


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_timings_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(timings_module, "_timing_namespaces", {})
    yield
    timings_module._timing_namespaces.clear()


def test_timings_are_no_ops_when_disabled(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(timings_module, "_ENABLED", False)
    timings_module.reset_timings()
    timings_module.time("step")
    timings_module.print_timings()
    assert timings_module._timing_namespaces == {}
    assert capsys.readouterr().err == ""


def test_timings_record_and_print_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(timings_module, "_ENABLED", True)
    timings_module.reset_timings()
    timings_module.time("step1")
    timings_module.time("step2")
    timings_module.print_timings()
    err = capsys.readouterr().err
    assert "Startup Timings: main" in err
    assert "step1:" in err
    assert "step2:" in err
    assert "TOTAL:" in err


def test_timings_use_separate_namespaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timings_module, "_ENABLED", True)
    timings_module.reset_timings("main")
    timings_module.time("a", "main")
    timings_module.time("b", "extensions")
    assert set(timings_module._timing_namespaces.keys()) == {"main", "extensions"}
    assert timings_module._timing_namespaces["main"].timings[0].label == "a"
    assert timings_module._timing_namespaces["extensions"].timings[0].label == "b"


def test_time_auto_resets_namespace_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timings_module, "_ENABLED", True)
    timings_module.time("first", "custom")
    assert "custom" in timings_module._timing_namespaces
    assert len(timings_module._timing_namespaces["custom"].timings) == 1


# ---------------------------------------------------------------------------
# tools.tool_definition_wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrap_tool_definition_defaults_context_from_factory() -> None:
    seen_ctx: list[object] = []

    async def execute(tool_call_id, params, signal, on_update, ctx):
        seen_ctx.append(ctx)
        return AgentToolResult(content=[TextContent(text=f"{tool_call_id}:{params}")])

    definition = ToolDefinition(name="t", label="T", description="d", execute=execute)
    tool = wrap_tool_definition(definition, ctx_factory=lambda: "factory-ctx")

    result = await tool.execute("call-1", {"a": 1}, None, None)
    assert seen_ctx == ["factory-ctx"]
    assert result.content[0].text == "call-1:{'a': 1}"
    assert tool.name == "t"
    assert tool.label == "T"


@pytest.mark.asyncio
async def test_wrap_tool_definition_prefers_explicit_ctx_over_factory() -> None:
    seen_ctx: list[object] = []

    async def execute(tool_call_id, params, signal, on_update, ctx):
        seen_ctx.append(ctx)
        return AgentToolResult(content=[])

    definition = ToolDefinition(name="t", label="T", description="d", execute=execute)
    tool = wrap_tool_definition(definition, ctx_factory=lambda: "factory-ctx")
    await tool.execute("call-1", {}, None, None, "explicit-ctx")
    assert seen_ctx == ["explicit-ctx"]


@pytest.mark.asyncio
async def test_wrap_tool_definition_without_ctx_factory_passes_none() -> None:
    seen_ctx: list[object] = []

    async def execute(tool_call_id, params, signal, on_update, ctx):
        seen_ctx.append(ctx)
        return AgentToolResult(content=[])

    definition = ToolDefinition(name="t", label="T", description="d", execute=execute)
    tool = wrap_tool_definition(definition)
    await tool.execute("call-1", {}, None, None)
    assert seen_ctx == [None]


@pytest.mark.asyncio
async def test_wrap_tool_definitions_wraps_each_one() -> None:
    async def execute(tool_call_id, params, signal, on_update, ctx):
        return AgentToolResult(content=[])

    definitions = [
        ToolDefinition(name="a", label="A", description="d", execute=execute),
        ToolDefinition(name="b", label="B", description="d", execute=execute),
    ]
    tools = wrap_tool_definitions(definitions)
    assert [tool.name for tool in tools] == ["a", "b"]
    for tool in tools:
        result = await tool.execute("id", {}, None, None)
        assert result.content == []


@pytest.mark.asyncio
async def test_create_tool_definition_from_agent_tool_round_trips_execute() -> None:
    async def agent_execute(tool_call_id, params, signal, on_update):
        return AgentToolResult(content=[TextContent(text="agent-result")])

    tool = AgentTool(name="a", description="desc", execute=agent_execute)
    definition = create_tool_definition_from_agent_tool(tool)
    assert definition.name == "a"
    assert definition.label == tool.label

    result = await definition.execute("id", {}, None, None, None)
    assert result.content[0].text == "agent-result"


# ---------------------------------------------------------------------------
# utils.deprecation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_deprecation_state():
    clear_deprecation_warnings_for_tests()
    yield
    clear_deprecation_warnings_for_tests()


def test_warn_deprecation_prints_message(capsys: pytest.CaptureFixture[str]) -> None:
    warn_deprecation("old flag")
    captured = capsys.readouterr()
    assert "Deprecation warning: old flag" in captured.err


def test_warn_deprecation_only_prints_once_per_message(capsys: pytest.CaptureFixture[str]) -> None:
    warn_deprecation("dup")
    warn_deprecation("dup")
    warn_deprecation("dup")
    captured = capsys.readouterr()
    assert captured.err.count("dup") == 1


def test_warn_deprecation_prints_distinct_messages_separately(capsys: pytest.CaptureFixture[str]) -> None:
    warn_deprecation("one")
    warn_deprecation("two")
    captured = capsys.readouterr()
    assert "one" in captured.err
    assert "two" in captured.err


def test_clear_deprecation_warnings_allows_message_again(capsys: pytest.CaptureFixture[str]) -> None:
    warn_deprecation("reused")
    capsys.readouterr()
    clear_deprecation_warnings_for_tests()
    warn_deprecation("reused")
    captured = capsys.readouterr()
    assert "reused" in captured.err


# ---------------------------------------------------------------------------
# exec: process termination edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_command_missing_binary_reports_code_one() -> None:
    """TS surfaces a spawn failure as code 1; it must never raise at callers."""
    result = await exec_command("definitely-not-a-real-binary-xyz", [], ".")
    assert result.code == 1
    assert result.killed is False
    assert result.stdout == ""


@pytest.mark.asyncio
async def test_exec_command_bad_cwd_reports_code_one() -> None:
    result = await exec_command("echo", ["x"], "/no/such/directory/anywhere")
    assert result.code == 1


@pytest.mark.asyncio
async def test_exec_command_returns_when_a_descendant_holds_the_pipe_open() -> None:
    """A backgrounded grandchild inherits stdout; waiting for EOF would hang.

    `bash -c "echo hi; sleep 20 &"` exits immediately but its child keeps the
    stdout pipe open for 20s. The call must return promptly with the output
    that was produced.
    """
    started = time.monotonic()
    result = await exec_command("bash", ["-c", "echo hi; sleep 20 &"], ".")
    elapsed = time.monotonic() - started

    assert result.stdout.strip() == "hi"
    assert elapsed < 5, f"returned after {elapsed:.1f}s; the descendant blocked the read"


@pytest.mark.asyncio
async def test_exec_command_timeout_is_honoured_despite_a_held_pipe() -> None:
    started = time.monotonic()
    result = await exec_command("bash", ["-c", "sleep 30 & sleep 30"], ".", ExecOptions(timeout=200))
    elapsed = time.monotonic() - started

    assert result.killed is True
    assert elapsed < 5, f"timeout took {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_exec_command_keeps_output_written_slowly() -> None:
    """Output still arriving must not be truncated by the idle grace period."""
    result = await exec_command("bash", ["-c", "for i in 1 2 3; do echo line$i; sleep 0.05; done"], ".")
    assert result.stdout == "line1\nline2\nline3\n"


# ---------------------------------------------------------------------------
# the wrapper on its live path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrap_registered_tool_uses_the_shared_wrapper_and_diffs_active_tools() -> None:
    """The live extension path must default the context AND report new tools.

    `wrap_registered_tool` is what actually runs for every extension tool, so
    it is asserted here directly rather than only through the standalone
    wrapper: a tool that activates others via `pi.setActiveTools()` has to
    surface them in `added_tool_names`.
    """
    from dataclasses import dataclass as _dataclass

    from pi_agent.types import AgentToolResult
    from pi_coding_agent.core.extensions.runner import wrap_registered_tool
    from pi_coding_agent.core.extensions.types import ToolDefinition

    seen_ctx: list[object] = []
    active: list[str] = ["alpha"]

    async def execute(tool_call_id, params, signal, on_update, ctx):
        seen_ctx.append(ctx)
        active.append("beta")
        return AgentToolResult()

    definition = ToolDefinition(
        name="activator",
        label="Activator",
        description="activates another tool",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    @_dataclass
    class _Registered:
        definition: ToolDefinition

    sentinel = object()

    class _Runner:
        def create_context(self):
            return sentinel

        def get_active_tool_names(self):
            return list(active)

    tool = wrap_registered_tool(_Registered(definition), _Runner())

    assert tool.name == "activator"
    result = await tool.execute("call-1", {}, None, None)

    assert seen_ctx == [sentinel], "the context must be defaulted from the runner"
    assert result.added_tool_names == ["beta"]

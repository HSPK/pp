"""RPC mode dispatch, ported alongside `modes/rpc/dispatcher.py`.

Covers `packages/coding-agent/src/modes/rpc/rpc-mode.ts`'s `handleCommand` and
`handleInputLine`. The dispatcher takes its `output` as a callable, so every
case here drives a real `AgentSession` and reads the protocol lines back out of
a list -- no subprocess, no stdout takeover.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from harness import Harness, create_harness, wait_until
from pi_ai.providers.faux import faux_assistant_message
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.modes.rpc import RPC_COMMAND_TYPES, RpcDispatcher, RpcExtensionUIContext
from pi_coding_agent.modes.rpc.types import make_success


class FakeRuntimeHost:
    """The slice of `AgentSessionRuntime` the dispatcher actually calls.

    A real runtime would replace the session by rebuilding it from disk, which
    these cases do not need; what they need to observe is *whether* a
    replacement was attempted and whether a rebind followed it.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self.calls: list[tuple[str, Any]] = []
        self.cancelled = False
        self.disposed = False

    @property
    def session(self) -> Any:
        return self._session

    async def new_session(self, parent_session: str | None = None) -> dict[str, Any]:
        self.calls.append(("new_session", parent_session))
        return {"cancelled": self.cancelled}

    async def fork(self, entry_id: str, position: str = "before") -> dict[str, Any]:
        self.calls.append(("fork", (entry_id, position)))
        return {"cancelled": self.cancelled, "selected_text": "forked text"}

    async def switch_session(self, session_path: str, cwd_override: str | None = None) -> dict[str, Any]:
        self.calls.append(("switch_session", session_path))
        return {"cancelled": self.cancelled}

    async def dispose(self) -> None:
        self.disposed = True

    def set_rebind_session(self, _rebind: Any) -> None:
        pass


@pytest.fixture
async def rpc(tmp_path: Path):
    harness = await create_harness(tmp_path)
    outputs: list[dict[str, Any]] = []
    host = FakeRuntimeHost(harness.session)
    rebinds: list[int] = []

    async def rebind() -> None:
        rebinds.append(1)

    dispatcher = RpcDispatcher(host, outputs.append, RpcExtensionUIContext(outputs.append), rebind)
    try:
        yield harness, dispatcher, outputs, host, rebinds
    finally:
        harness.cleanup()


# ---------------------------------------------------------------------------
# Command table
# ---------------------------------------------------------------------------


def test_command_table_matches_handlers() -> None:
    """`RPC_COMMAND_TYPES` is what dispatch accepts, so the two must agree.

    A command added to the table with no handler raises `AttributeError` at
    request time instead of answering `Unknown command`; a handler missing from
    the table is unreachable.
    """
    handlers = {name[len("_cmd_") :] for name in dir(RpcDispatcher) if name.startswith("_cmd_")}
    assert handlers == set(RPC_COMMAND_TYPES)


def test_command_table_covers_the_typescript_protocol() -> None:
    """Pinned to the union in `rpc-types.ts`, so a dropped command is visible."""
    expected = {
        "prompt",
        "steer",
        "follow_up",
        "abort",
        "new_session",
        "get_state",
        "set_model",
        "cycle_model",
        "get_available_models",
        "set_thinking_level",
        "cycle_thinking_level",
        "get_available_thinking_levels",
        "set_steering_mode",
        "set_follow_up_mode",
        "compact",
        "set_auto_compaction",
        "set_auto_retry",
        "abort_retry",
        "bash",
        "abort_bash",
        "get_session_stats",
        "export_html",
        "switch_session",
        "fork",
        "clone",
        "get_fork_messages",
        "get_entries",
        "get_tree",
        "get_last_assistant_text",
        "set_session_name",
        "get_messages",
        "get_commands",
    }
    assert set(RPC_COMMAND_TYPES) == expected


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


async def test_malformed_line_answers_parse_error_and_keeps_serving(rpc) -> None:
    _harness, dispatcher, outputs, _host, _rebinds = rpc

    await dispatcher.handle_input_line("{not json")
    assert outputs[0]["success"] is False
    assert outputs[0]["command"] == "parse"
    assert outputs[0]["id"] is None

    # The session is still usable: a bad line must not end the loop.
    await dispatcher.handle_input_line('{"id": "1", "type": "get_state"}')
    assert outputs[1]["success"] is True


async def test_non_object_line_is_rejected(rpc) -> None:
    _harness, dispatcher, outputs, _host, _rebinds = rpc
    await dispatcher.handle_input_line("[1, 2, 3]")
    assert outputs[0]["command"] == "parse"
    assert "JSON object" in outputs[0]["error"]


async def test_unknown_command_answers_by_name(rpc) -> None:
    _harness, dispatcher, outputs, _host, _rebinds = rpc
    await dispatcher.handle_input_line('{"id": "7", "type": "teleport"}')
    assert outputs[0] == {
        "id": "7",
        "type": "response",
        "command": "teleport",
        "success": False,
        "error": "Unknown command: teleport",
    }


async def test_handler_exception_becomes_an_error_response(rpc) -> None:
    """A raising command answers instead of killing the reader loop."""
    _harness, dispatcher, outputs, _host, _rebinds = rpc

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("bash unavailable")

    dispatcher.session.abort_bash = boom
    await dispatcher.handle_input_line('{"id": "9", "type": "abort_bash"}')
    assert outputs[0]["success"] is False
    assert outputs[0]["error"] == "bash unavailable"
    assert outputs[0]["command"] == "abort_bash"


# ---------------------------------------------------------------------------
# State and model
# ---------------------------------------------------------------------------


async def test_get_state_reports_the_live_session(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", "type": "get_state"})
    data = response["data"]

    assert data["sessionId"] == harness.session.session_id
    assert data["isStreaming"] is False
    assert data["messageCount"] == len(harness.session.messages)
    assert data["pendingMessageCount"] == 0
    assert data["steeringMode"] == harness.session.steering_mode
    assert data["model"]["id"] == harness.session.model.id


async def test_set_model_reports_an_unknown_model_by_name(rpc) -> None:
    _harness, dispatcher, _outputs, _host, _rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", "type": "set_model", "provider": "nope", "modelId": "ghost"})
    assert response["success"] is False
    assert response["error"] == "Model not found: nope/ghost"


async def test_set_model_switches_to_an_available_model(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    model = harness.session.model_runtime.get_available_snapshot()[0]
    response = await dispatcher.handle_command(
        {"id": "1", "type": "set_model", "provider": model.provider, "modelId": model.id}
    )
    assert response["success"] is True
    assert response["data"]["id"] == model.id
    assert harness.session.model.id == model.id


async def test_get_available_models_lists_the_snapshot(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", "type": "get_available_models"})
    expected = [model.id for model in harness.session.model_runtime.get_available_snapshot()]
    assert [model["id"] for model in response["data"]["models"]] == expected


async def test_cycle_with_nothing_to_cycle_to_answers_null_data(rpc) -> None:
    """`data: null` and no `data` key are different responses.

    `abort` carries no `data`; `cycle_thinking_level` with nothing to cycle to
    carries an explicit null. Collapsing them would make a host unable to tell
    "not applicable" from "not answered".
    """
    _harness, dispatcher, _outputs, _host, _rebinds = rpc
    dispatcher.session.cycle_thinking_level = lambda: None

    cycled = await dispatcher.handle_command({"id": "1", "type": "cycle_thinking_level"})
    aborted = await dispatcher.handle_command({"id": "2", "type": "abort"})

    assert "data" in cycled and cycled["data"] is None
    assert "data" not in aborted


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


async def test_set_session_name_rejects_blank_names(rpc) -> None:
    _harness, dispatcher, _outputs, _host, _rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", "type": "set_session_name", "name": "   "})
    assert response["success"] is False
    assert response["error"] == "Session name cannot be empty"


async def test_set_session_name_trims_and_applies(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", "type": "set_session_name", "name": "  triage  "})
    assert response["success"] is True
    assert harness.session.session_name == "triage"


async def test_get_entries_rejects_an_unknown_since_marker(rpc) -> None:
    _harness, dispatcher, _outputs, _host, _rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", "type": "get_entries", "since": "missing"})
    assert response["success"] is False
    assert response["error"] == "Entry not found: missing"


async def test_get_entries_returns_only_what_follows_since(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    harness.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])
    await harness.session.prompt("first")
    await harness.session.prompt("second")

    entries = harness.session.session_manager.get_entries()
    assert len(entries) > 2
    marker = entries[0].id

    response = await dispatcher.handle_command({"id": "1", "type": "get_entries", "since": marker})
    returned = [entry["id"] for entry in response["data"]["entries"]]
    assert returned == [entry.id for entry in entries[1:]]
    assert response["data"]["leafId"] == harness.session.session_manager.get_leaf_id()


async def test_get_tree_reports_the_leaf(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    harness.set_responses([faux_assistant_message("hi")])
    await harness.session.prompt("hello")

    response = await dispatcher.handle_command({"id": "1", "type": "get_tree"})
    assert response["data"]["leafId"] == harness.session.session_manager.get_leaf_id()
    assert isinstance(response["data"]["tree"], list)


async def test_a_null_leaf_is_omitted_rather_than_sent_as_null(rpc) -> None:
    """`to_wire` drops `None`, so a fresh session answers without `leafId`.

    TypeScript emits `"leafId": null` here. The port follows its own JSON
    convention instead -- the same one print mode and `--mode json` already use
    -- so a host reading this protocol handles one shape, not two.
    """
    _harness, dispatcher, _outputs, _host, _rebinds = rpc
    dispatcher.session.session_manager.get_leaf_id = lambda: None
    response = await dispatcher.handle_command({"id": "1", "type": "get_tree"})
    assert "leafId" not in response["data"]


async def test_clone_without_a_current_entry_is_refused(rpc) -> None:
    _harness, dispatcher, _outputs, _host, _rebinds = rpc
    dispatcher.session.session_manager.get_leaf_id = lambda: None
    response = await dispatcher.handle_command({"id": "1", "type": "clone"})
    assert response["success"] is False
    assert response["error"] == "Cannot clone session: no current entry selected"


async def test_clone_forks_at_the_leaf(rpc) -> None:
    harness, dispatcher, _outputs, host, rebinds = rpc
    harness.set_responses([faux_assistant_message("hi")])
    await harness.session.prompt("hello")

    response = await dispatcher.handle_command({"id": "1", "type": "clone"})
    assert response["data"] == {"cancelled": False}
    assert host.calls == [("fork", (harness.session.session_manager.get_leaf_id(), "at"))]
    assert rebinds == [1]


async def test_get_last_assistant_text(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    harness.set_responses([faux_assistant_message("the answer")])
    await harness.session.prompt("question")
    response = await dispatcher.handle_command({"id": "1", "type": "get_last_assistant_text"})
    assert response["data"]["text"] == "the answer"


# ---------------------------------------------------------------------------
# Session replacement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected_call"),
    [
        ({"type": "new_session"}, "new_session"),
        ({"type": "fork", "entryId": "e1"}, "fork"),
        ({"type": "switch_session", "sessionPath": "/tmp/s.jsonl"}, "switch_session"),
    ],
)
async def test_replacement_commands_rebind_when_they_succeed(rpc, command, expected_call) -> None:
    _harness, dispatcher, _outputs, host, rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", **command})
    assert response["success"] is True
    assert response["data"]["cancelled"] is False
    assert host.calls[0][0] == expected_call
    assert rebinds == [1]


@pytest.mark.parametrize(
    "command",
    [
        {"type": "new_session"},
        {"type": "fork", "entryId": "e1"},
        {"type": "switch_session", "sessionPath": "/tmp/s.jsonl"},
    ],
)
async def test_a_cancelled_replacement_does_not_rebind(rpc, command) -> None:
    """An extension can veto a replacement. Rebinding anyway would re-emit
    `session_start` and re-subscribe to a session that never changed.
    """
    _harness, dispatcher, _outputs, host, rebinds = rpc
    host.cancelled = True
    response = await dispatcher.handle_command({"id": "1", **command})
    assert response["data"]["cancelled"] is True
    assert rebinds == []


async def test_fork_returns_the_selected_text(rpc) -> None:
    _harness, dispatcher, _outputs, _host, _rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", "type": "fork", "entryId": "e1"})
    assert response["data"] == {"text": "forked text", "cancelled": False}


async def test_new_session_forwards_the_parent(rpc) -> None:
    _harness, dispatcher, _outputs, host, _rebinds = rpc
    await dispatcher.handle_command({"id": "1", "type": "new_session", "parentSession": "/tmp/parent.jsonl"})
    assert host.calls == [("new_session", "/tmp/parent.jsonl")]


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


async def test_prompt_answers_on_preflight_not_on_completion(rpc) -> None:
    """A host needs to know its prompt was accepted before the turn finishes.

    So `prompt` returns nothing from dispatch and the success line is emitted
    from the preflight callback while the turn runs in the background.
    """
    harness, dispatcher, outputs, _host, _rebinds = rpc
    harness.set_responses([faux_assistant_message("done")])

    immediate = await dispatcher.handle_command({"id": "42", "type": "prompt", "message": "hello"})
    assert immediate is None

    await wait_until(lambda: bool(outputs), what="prompt response")
    assert outputs[0] == make_success("42", "prompt")

    await wait_until(lambda: harness.session.is_idle, what="turn to settle")


async def test_prompt_failure_before_preflight_answers_an_error(rpc) -> None:
    _harness, dispatcher, outputs, _host, _rebinds = rpc

    async def failing_prompt(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("no model configured")

    dispatcher.session.prompt = failing_prompt
    await dispatcher.handle_command({"id": "3", "type": "prompt", "message": "hi"})

    await wait_until(lambda: bool(outputs), what="prompt error")
    assert outputs[0]["success"] is False
    assert outputs[0]["error"] == "no model configured"


async def test_steer_and_follow_up_queue_while_streaming(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    harness.set_responses([faux_assistant_message("ok")])

    steered = await dispatcher.handle_command({"id": "1", "type": "steer", "message": "wait"})
    followed = await dispatcher.handle_command({"id": "2", "type": "follow_up", "message": "then this"})

    assert steered["success"] is True and followed["success"] is True
    assert harness.session.get_steering_messages() == ["wait"]
    assert harness.session.get_follow_up_messages() == ["then this"]


# ---------------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------------


async def test_toggle_commands_reach_the_session(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc

    await dispatcher.handle_command({"id": "1", "type": "set_auto_compaction", "enabled": False})
    assert harness.session.auto_compaction_enabled is False

    await dispatcher.handle_command({"id": "2", "type": "set_auto_retry", "enabled": False})
    assert harness.session.auto_retry_enabled is False

    await dispatcher.handle_command({"id": "3", "type": "set_steering_mode", "mode": "one-at-a-time"})
    assert harness.session.steering_mode == "one-at-a-time"

    await dispatcher.handle_command({"id": "4", "type": "set_follow_up_mode", "mode": "one-at-a-time"})
    assert harness.session.follow_up_mode == "one-at-a-time"


async def test_get_available_thinking_levels(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", "type": "get_available_thinking_levels"})
    assert response["data"]["levels"] == list(harness.session.get_available_thinking_levels())


# ---------------------------------------------------------------------------
# Bash
# ---------------------------------------------------------------------------


async def test_bash_runs_the_command_and_records_it(rpc) -> None:
    harness, dispatcher, _outputs, _host, _rebinds = rpc
    response = await dispatcher.handle_command({"id": "1", "type": "bash", "command": "echo rpc-mode"})
    assert response["success"] is True
    assert "rpc-mode" in response["data"]["output"]
    assert any("echo rpc-mode" in str(entry) for entry in harness.session.session_manager.get_entries())


async def test_a_user_bash_handler_result_is_recorded_without_re_running(tmp_path: Path) -> None:
    """A `user_bash` handler that returns a result has already run the command.

    Executing it again here would run it twice, which for a command with side
    effects is the difference between one commit and two.
    """
    ran: list[str] = []
    marker = tmp_path / "side-effect.txt"

    def extension(pi: ExtensionAPI) -> None:
        def on_user_bash(event: Any, _ctx: Any) -> Any:
            from pi_coding_agent.core.bash_executor import BashResult
            from pi_coding_agent.core.extensions.types import UserBashEventResult

            ran.append(event.command)
            return UserBashEventResult(
                result=BashResult(output="handled", exit_code=0, cancelled=False, truncated=False)
            )

        pi.on("user_bash", on_user_bash)

    harness = await create_harness(tmp_path, extension_factories=[extension])
    try:
        dispatcher = RpcDispatcher(FakeRuntimeHost(harness.session), lambda _payload: None)
        response = await dispatcher.handle_command({"id": "1", "type": "bash", "command": f"touch {marker}"})
        assert response["data"]["output"] == "handled"
        assert ran == [f"touch {marker}"]
        assert not marker.exists(), "the command was executed despite the handler answering"
    finally:
        harness.cleanup()


# ---------------------------------------------------------------------------
# Commands listing
# ---------------------------------------------------------------------------


async def test_get_commands_reports_extension_commands_with_their_source(tmp_path: Path) -> None:
    def extension(pi: ExtensionAPI) -> None:
        async def handler(_args: str, _ctx: Any) -> None:
            return None

        pi.register_command("deploy", handler=handler, description="Ship it")

    harness = await create_harness(tmp_path, extension_factories=[extension])
    try:
        dispatcher = RpcDispatcher(FakeRuntimeHost(harness.session), lambda _payload: None)
        response = await dispatcher.handle_command({"id": "1", "type": "get_commands"})
        entries = {command["name"]: command for command in response["data"]["commands"]}
        assert entries["deploy"]["source"] == "extension"
        assert entries["deploy"]["description"] == "Ship it"
    finally:
        harness.cleanup()


# ---------------------------------------------------------------------------
# Extension UI bridge
# ---------------------------------------------------------------------------


async def test_extension_ui_response_is_routed_to_the_waiting_dialog(rpc) -> None:
    _harness, dispatcher, outputs, _host, _rebinds = rpc
    ui = dispatcher._ui_context

    answer = asyncio.ensure_future(ui.select("Pick one", ["a", "b"]))
    await wait_until(lambda: bool(outputs), what="ui request")

    request = outputs[0]
    assert request["type"] == "extension_ui_request"
    assert request["method"] == "select"
    assert request["options"] == ["a", "b"]

    await dispatcher.handle_input_line(json.dumps({"type": "extension_ui_response", "id": request["id"], "value": "b"}))
    assert await answer == "b"


async def test_a_cancelled_dialog_answers_none(rpc) -> None:
    _harness, dispatcher, outputs, _host, _rebinds = rpc
    ui = dispatcher._ui_context

    answer = asyncio.ensure_future(ui.input("Name?"))
    await wait_until(lambda: bool(outputs), what="ui request")
    await dispatcher.handle_input_line(
        json.dumps({"type": "extension_ui_response", "id": outputs[0]["id"], "cancelled": True})
    )
    assert await answer is None


async def test_confirm_defaults_to_false_when_cancelled(rpc) -> None:
    _harness, dispatcher, outputs, _host, _rebinds = rpc
    ui = dispatcher._ui_context

    answer = asyncio.ensure_future(ui.confirm("Delete?", "This cannot be undone"))
    await wait_until(lambda: bool(outputs), what="ui request")
    await dispatcher.handle_input_line(
        json.dumps({"type": "extension_ui_response", "id": outputs[0]["id"], "cancelled": True})
    )
    assert await answer is False


async def test_an_unmatched_ui_response_is_dropped(rpc) -> None:
    """A host answering twice, or answering after its own timeout, must not
    raise here -- the second answer has nobody left to deliver it to.
    """
    _harness, dispatcher, outputs, _host, _rebinds = rpc
    await dispatcher.handle_input_line('{"type": "extension_ui_response", "id": "ghost", "value": "x"}')
    assert outputs == []


async def test_ui_response_does_not_produce_a_command_response(rpc) -> None:
    _harness, dispatcher, outputs, _host, _rebinds = rpc
    ui = dispatcher._ui_context
    answer = asyncio.ensure_future(ui.select("Pick", ["a"]))
    await wait_until(lambda: bool(outputs), what="ui request")
    request_id = outputs[0]["id"]
    outputs.clear()

    await dispatcher.handle_input_line(json.dumps({"type": "extension_ui_response", "id": request_id, "value": "a"}))
    assert await answer == "a"
    assert outputs == []


def test_fire_and_forget_ui_calls_emit_immediately() -> None:
    outputs: list[dict[str, Any]] = []
    ui = RpcExtensionUIContext(outputs.append)

    ui.notify("heads up", "warning")
    ui.set_status("build", "running")
    ui.set_title("pi")
    ui.set_widget("stats", ["a", "b"], "belowEditor")

    assert [payload["method"] for payload in outputs] == ["notify", "setStatus", "setTitle", "setWidget"]
    assert outputs[0]["notifyType"] == "warning"
    assert outputs[3]["widgetLines"] == ["a", "b"]
    assert outputs[3]["widgetPlacement"] == "belowEditor"
    assert len({payload["id"] for payload in outputs}) == 4


def test_a_widget_factory_is_dropped_rather_than_serialized() -> None:
    """A factory builds a TUI component, and RPC mode has no TUI to build it
    against. Emitting it would put an unserializable object on the wire.
    """
    outputs: list[dict[str, Any]] = []
    ui = RpcExtensionUIContext(outputs.append)
    ui.set_widget("stats", lambda _tui, _theme: object())
    assert outputs == []


def test_clearing_the_widget_still_reaches_the_host() -> None:
    outputs: list[dict[str, Any]] = []
    ui = RpcExtensionUIContext(outputs.append)
    ui.set_widget("stats", None)
    assert outputs[0]["widgetLines"] is None


async def test_cancel_all_releases_dialogs_waiting_on_a_closed_host() -> None:
    """When stdin ends, an extension awaiting an answer can never get one.

    Leaving the future pending would hang shutdown behind a reply that is not
    coming.
    """
    outputs: list[dict[str, Any]] = []
    ui = RpcExtensionUIContext(outputs.append)
    answer = asyncio.ensure_future(ui.select("Pick", ["a"]))
    await wait_until(lambda: bool(outputs), what="ui request")

    ui.cancel_all()
    assert await answer is None


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


async def test_an_extension_can_request_shutdown(tmp_path: Path) -> None:
    harness: Harness = await create_harness(tmp_path)
    try:
        dispatcher = RpcDispatcher(FakeRuntimeHost(harness.session), lambda _payload: None)
        assert dispatcher.shutdown_requested is False
        harness.session.set_extension_shutdown_handler(dispatcher.request_shutdown)
        harness.session.extension_runner.create_context().shutdown()
        assert dispatcher.shutdown_requested is True
    finally:
        harness.cleanup()

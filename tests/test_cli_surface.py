"""Tests for the CLI surface: arg dispatch, auth commands, list-models, wire encoding."""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pi_ai.auth.types import CredentialInfo

from pi_coding_agent.cli import entry
from pi_coding_agent.cli.args import HELP_TEXT, parse_args, print_help
from pi_coding_agent.cli.auth_command import (
    AuthCommandError,
    get_auth_command_name,
    get_auth_command_usage,
    get_auth_credential,
    handle_auth_command,
    is_auth_command_help,
    parse_auth_command,
    validate_auth_command_args,
)
from pi_coding_agent.cli.entry import resolve_app_mode, to_print_output_mode
from pi_coding_agent.cli.list_models import format_token_count, list_models
from pi_coding_agent.cli.session_selection import (
    SessionSelectionError,
    resolve_session_path,
    validate_fork_flags,
    validate_session_id_flags,
)
from pi_coding_agent.core.output_guard import (
    restore_stdout,
    take_over_stdout,
)
from pi_coding_agent.modes.json_event import to_json_event
from pi_coding_agent.utils.js_number import js_round, to_fixed
from pi_coding_agent.utils.wire import snake_to_camel, to_wire

# ---------------------------------------------------------------------------
# resolve_app_mode
# ---------------------------------------------------------------------------


def test_interactive_when_both_streams_are_ttys() -> None:
    assert resolve_app_mode(parse_args([]), True, True) == "interactive"


def test_print_flag_forces_print_mode() -> None:
    assert resolve_app_mode(parse_args(["-p", "hi"]), True, True) == "print"
    assert resolve_app_mode(parse_args(["--print", "hi"]), True, True) == "print"


def test_non_tty_stdin_forces_print_mode() -> None:
    assert resolve_app_mode(parse_args([]), False, True) == "print"


def test_non_tty_stdout_forces_print_mode() -> None:
    assert resolve_app_mode(parse_args([]), True, False) == "print"


def test_mode_json_wins_over_tty_detection() -> None:
    assert resolve_app_mode(parse_args(["--mode", "json"]), True, True) == "json"
    assert resolve_app_mode(parse_args(["--mode", "json"]), False, False) == "json"


def test_mode_rpc_wins_over_print_flag() -> None:
    assert resolve_app_mode(parse_args(["--mode", "rpc", "-p"]), True, True) == "rpc"


def test_print_output_mode_mapping() -> None:
    assert to_print_output_mode("json") == "json"
    assert to_print_output_mode("print") == "text"
    assert to_print_output_mode("interactive") == "text"


# ---------------------------------------------------------------------------
# help text
# ---------------------------------------------------------------------------


def test_help_text_lists_every_documented_command(capsys: Any) -> None:
    print_help()
    output = capsys.readouterr().out
    for command in ("install", "remove", "uninstall", "update", "list", "config", "auth"):
        assert f"pi {command}" in output


def test_help_text_documents_mode_flag() -> None:
    assert "--mode <mode>" in HELP_TEXT
    assert "rpc" in HELP_TEXT


# ---------------------------------------------------------------------------
# auth command parsing
# ---------------------------------------------------------------------------


def test_parse_auth_command_returns_none_for_non_auth_args() -> None:
    assert parse_auth_command([]) is None
    assert parse_auth_command(["--help"]) is None
    assert parse_auth_command(["hello"]) is None


def test_parse_auth_command_kinds() -> None:
    assert parse_auth_command(["auth", "check"]).kind == "check"
    assert parse_auth_command(["auth", "print-api-key"]).kind == "api_key"
    assert parse_auth_command(["auth", "print-bearer-token"]).kind == "bearer_token"


def test_parse_auth_command_rejects_unknown_subcommand() -> None:
    with pytest.raises(AuthCommandError, match="Unknown auth command"):
        parse_auth_command(["auth", "bogus"])


def test_check_only_flags_rejected_on_other_kinds() -> None:
    for flag in ("--json", "--credentials", "--no-refresh"):
        with pytest.raises(AuthCommandError, match="only supported by auth check"):
            parse_auth_command(["auth", "print-api-key", flag])


def test_min_expiry_only_on_bearer_token() -> None:
    with pytest.raises(AuthCommandError, match="only supported by print-bearer-token"):
        parse_auth_command(["auth", "check", "--min-expiry", "1h"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [("500ms", 500), ("30s", 30_000), ("30m", 1_800_000), ("1h", 3_600_000), ("2H", 7_200_000)],
)
def test_min_expiry_duration_units(value: str, expected: int) -> None:
    command = parse_auth_command(["auth", "print-bearer-token", "--min-expiry", value])
    assert command.min_expiry_ms == expected


@pytest.mark.parametrize("value", ["1", "1d", "abc", "-1h", ""])
def test_min_expiry_rejects_bad_durations(value: str) -> None:
    with pytest.raises(AuthCommandError, match="must use a duration"):
        parse_auth_command(["auth", "print-bearer-token", "--min-expiry", value])


def test_min_expiry_rejects_missing_value() -> None:
    with pytest.raises(AuthCommandError, match="must use a duration"):
        parse_auth_command(["auth", "print-bearer-token", "--min-expiry"])


def test_auth_command_forwards_remaining_args() -> None:
    command = parse_auth_command(["auth", "check", "--provider", "anthropic", "--json"])
    assert command.args == ["--provider", "anthropic"]
    assert command.json is True


def test_is_auth_command_help() -> None:
    assert is_auth_command_help(["auth"]) is True
    assert is_auth_command_help(["auth", "help"]) is True
    assert is_auth_command_help(["auth", "check", "--help"]) is True
    assert is_auth_command_help(["auth", "check", "-h"]) is True
    assert is_auth_command_help(["auth", "check"]) is False
    assert is_auth_command_help(["hello"]) is False


def test_auth_command_names_and_usage() -> None:
    assert get_auth_command_name("check") == "auth check"
    assert get_auth_command_name("api_key") == "auth print-api-key"
    assert get_auth_command_name("bearer_token") == "auth print-bearer-token"
    for kind in ("check", "api_key", "bearer_token"):
        assert get_auth_command_usage(kind).startswith("pi auth ")


def test_validate_auth_command_args_requires_provider_or_model() -> None:
    with pytest.raises(AuthCommandError, match="Auth checks require"):
        validate_auth_command_args(parse_args([]), "check")
    with pytest.raises(AuthCommandError, match="Credential printing requires"):
        validate_auth_command_args(parse_args([]), "api_key")


def test_validate_auth_command_args_rejects_extra_input() -> None:
    with pytest.raises(AuthCommandError, match="only accept --provider and --model"):
        validate_auth_command_args(parse_args(["--provider", "anthropic", "hello"]), "check")


def test_validate_auth_command_args_rejects_unknown_flags() -> None:
    args = parse_args(["--provider", "anthropic", "--bogus"])
    with pytest.raises(AuthCommandError, match="Unknown option --bogus"):
        validate_auth_command_args(args, "check")


def test_validate_auth_command_args_treats_blank_as_missing() -> None:
    with pytest.raises(AuthCommandError, match="Auth checks require"):
        validate_auth_command_args(parse_args(["--provider", "   "]), "check")


# ---------------------------------------------------------------------------
# credential extraction
# ---------------------------------------------------------------------------


@dataclass
class _Auth:
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class _AuthResult:
    auth: _Auth


def test_get_auth_credential_prefers_api_key() -> None:
    result = _AuthResult(_Auth(api_key="sk-1", headers={"authorization": "Bearer tok"}))
    assert get_auth_credential(result) == "sk-1"


def test_get_auth_credential_reads_bearer_header_case_insensitively() -> None:
    assert get_auth_credential(_AuthResult(_Auth(headers={"Authorization": "Bearer tok"}))) == "tok"
    assert get_auth_credential(_AuthResult(_Auth(headers={"AUTHORIZATION": "bearer x"}))) == "x"


def test_get_auth_credential_returns_none_without_credentials() -> None:
    assert get_auth_credential(None) is None
    assert get_auth_credential(_AuthResult(_Auth())) is None
    assert get_auth_credential(_AuthResult(_Auth(headers={"x-api-key": "nope"}))) is None


# ---------------------------------------------------------------------------
# auth command dispatch
# ---------------------------------------------------------------------------


@dataclass
class _AuthStatus:
    configured: bool
    type: str | None = None


class _FakeRuntime:
    def __init__(
        self,
        *,
        configured: bool = True,
        auth: Any = None,
        provider: Any = object(),
        credentials: list[CredentialInfo] | None = None,
    ) -> None:
        self._configured = configured
        self._auth = auth
        self._provider = provider
        self._credentials = credentials or []

    def get_error(self) -> str | None:
        return None

    def get_provider(self, provider_id: str) -> Any:
        return SimpleNamespace(id=provider_id) if self._provider is not None else None

    def get_providers(self) -> list[Any]:
        return [SimpleNamespace(id=info.provider_id) for info in self._credentials]

    async def list_credentials(self) -> list[CredentialInfo]:
        return list(self._credentials)

    def get_provider_auth_status(self, provider_id: str) -> _AuthStatus:
        return _AuthStatus(configured=self._configured, type="api_key")

    async def check_auth(self, provider_id: str) -> _AuthStatus:
        return _AuthStatus(configured=self._configured, type="api_key")

    def find_model(self, reference: str) -> Any:
        return None

    async def get_auth(self, target: Any, *, min_oauth_validity_ms: int | None = None) -> Any:
        return self._auth


@pytest.mark.asyncio
async def test_handle_auth_command_ignores_other_commands() -> None:
    assert await handle_auth_command(["install", "x"]) is None
    assert await handle_auth_command([]) is None


@pytest.mark.asyncio
async def test_handle_auth_command_prints_help() -> None:
    lines: list[str] = []
    assert await handle_auth_command(["auth"], write=lines.append) == 0
    assert "pi auth print-api-key" in "\n".join(lines)


@pytest.mark.asyncio
async def test_auth_check_ready_exits_zero() -> None:
    lines: list[str] = []
    runtime = _FakeRuntime(auth=_AuthResult(_Auth(api_key="sk-1")))
    code = await handle_auth_command(
        ["auth", "check", "--provider", "anthropic"], model_runtime=runtime, write=lines.append
    )
    assert code == 0
    assert lines == ["anthropic: ready"]


@pytest.mark.asyncio
async def test_auth_check_not_configured_exits_one() -> None:
    lines: list[str] = []
    code = await handle_auth_command(
        ["auth", "check", "--provider", "anthropic"],
        model_runtime=_FakeRuntime(configured=False),
        write=lines.append,
    )
    assert code == 1
    assert lines == ["anthropic: not_ready (credentials_not_configured)"]


@pytest.mark.asyncio
async def test_auth_check_missing_provider_exits_one() -> None:
    lines: list[str] = []
    code = await handle_auth_command(
        ["auth", "check", "--provider", "anthropic"],
        model_runtime=_FakeRuntime(provider=None),
        write=lines.append,
    )
    assert code == 1
    assert "provider_not_found" in lines[0]


@pytest.mark.asyncio
async def test_auth_check_json_output() -> None:
    lines: list[str] = []
    runtime = _FakeRuntime(auth=_AuthResult(_Auth(api_key="sk-1")))
    await handle_auth_command(
        ["auth", "check", "--provider", "anthropic", "--json"],
        model_runtime=runtime,
        write=lines.append,
    )
    assert json.loads(lines[0]) == {
        "status": "ready",
        "provider": "anthropic",
        "authType": "api_key",
    }


@pytest.mark.asyncio
async def test_auth_check_json_with_credentials() -> None:
    lines: list[str] = []
    runtime = _FakeRuntime(auth=_AuthResult(_Auth(api_key="sk-1")))
    await handle_auth_command(
        ["auth", "check", "--provider", "anthropic", "--json", "--credentials"],
        model_runtime=runtime,
        write=lines.append,
    )
    assert json.loads(lines[0])["credential"] == "sk-1"


@pytest.mark.asyncio
async def test_auth_check_no_refresh_skips_credential_resolution() -> None:
    class _Exploding(_FakeRuntime):
        async def get_auth(self, target: Any) -> Any:
            raise AssertionError("--no-refresh must not resolve credentials")

    lines: list[str] = []
    code = await handle_auth_command(
        ["auth", "check", "--provider", "anthropic", "--no-refresh"],
        model_runtime=_Exploding(),
        write=lines.append,
    )
    assert code == 0


@pytest.mark.asyncio
async def test_auth_check_refresh_failure_is_not_ready() -> None:
    lines: list[str] = []
    code = await handle_auth_command(
        ["auth", "check", "--provider", "anthropic"],
        model_runtime=_FakeRuntime(auth=None),
        write=lines.append,
    )
    assert code == 1
    assert "credentials_not_configured" in lines[0]


@pytest.mark.asyncio
async def test_print_api_key() -> None:
    lines: list[str] = []
    runtime = _FakeRuntime(auth=_AuthResult(_Auth(api_key="sk-1")))
    code = await handle_auth_command(
        ["auth", "print-api-key", "--provider", "anthropic"],
        model_runtime=runtime,
        write=lines.append,
    )
    assert code == 0
    assert lines == ["sk-1"]


@pytest.mark.asyncio
async def test_print_bearer_token() -> None:
    lines: list[str] = []
    runtime = _FakeRuntime(
        auth=_AuthResult(_Auth(headers={"authorization": "Bearer tok"})),
        credentials=[CredentialInfo(provider_id="anthropic", type="oauth")],
    )
    code = await handle_auth_command(
        ["auth", "print-bearer-token", "--provider", "anthropic"],
        model_runtime=runtime,
        write=lines.append,
    )
    assert code == 0
    assert lines == ["tok"]


@pytest.mark.asyncio
async def test_print_credential_without_credential_exits_one() -> None:
    lines: list[str] = []
    code = await handle_auth_command(
        ["auth", "print-api-key", "--provider", "anthropic"],
        model_runtime=_FakeRuntime(auth=None),
        write=lines.append,
    )
    assert code == 1
    assert lines == []


@pytest.mark.asyncio
async def test_auth_command_validation_error_prints_usage(capsys: Any) -> None:
    code = await handle_auth_command(["auth", "check"], model_runtime=_FakeRuntime())
    assert code == 1
    assert "Usage: pi auth check" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# list-models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1K"),
        (1_500, "1.5K"),
        (200_000, "200K"),
        (1_000_000, "1M"),
        (1_250_000, "1.3M"),
        (1_500_000, "1.5M"),
        (2_000_000, "2M"),
    ],
)
def test_format_token_count(count: int, expected: str) -> None:
    assert format_token_count(count) == expected


@dataclass
class _Model:
    provider: str
    id: str
    context_window: int = 200_000
    max_tokens: int = 64_000
    reasoning: bool = False
    input: list[str] = field(default_factory=lambda: ["text"])


class _ModelsRuntime:
    def __init__(self, models: list[_Model]) -> None:
        self._models = models

    def get_error(self) -> str | None:
        return None

    async def get_available(self, provider_id: str | None = None) -> list[_Model]:
        return list(self._models)


@pytest.mark.asyncio
async def test_list_models_reports_no_models() -> None:
    lines: list[str] = []
    await list_models(_ModelsRuntime([]), None, lines.append)
    assert "No models available" in lines[0]


@pytest.mark.asyncio
async def test_list_models_sorts_and_aligns() -> None:
    models = [
        _Model("zebra", "b-model"),
        _Model("alpha", "z-model", reasoning=True, input=["text", "image"]),
        _Model("alpha", "a-model"),
    ]
    lines: list[str] = []
    await list_models(_ModelsRuntime(models), None, lines.append)

    assert lines[0].split() == ["provider", "model", "context", "max-out", "thinking", "images"]
    assert [line.split()[0:2] for line in lines[1:]] == [
        ["alpha", "a-model"],
        ["alpha", "z-model"],
        ["zebra", "b-model"],
    ]
    # Every column starts at the same offset on every row.
    assert len({len(line.rstrip()) - len(line.split()[-1]) for line in lines}) == 1


@pytest.mark.asyncio
async def test_list_models_reports_thinking_and_images() -> None:
    models = [_Model("alpha", "m", reasoning=True, input=["text", "image"])]
    lines: list[str] = []
    await list_models(_ModelsRuntime(models), None, lines.append)
    assert lines[1].split()[-2:] == ["yes", "yes"]


@pytest.mark.asyncio
async def test_list_models_fuzzy_filters() -> None:
    models = [_Model("anthropic", "claude-sonnet-5"), _Model("google", "gemini-3")]
    lines: list[str] = []
    await list_models(_ModelsRuntime(models), "sonnet", lines.append)
    assert len(lines) == 2
    assert "claude-sonnet-5" in lines[1]


@pytest.mark.asyncio
async def test_list_models_reports_empty_search() -> None:
    lines: list[str] = []
    await list_models(_ModelsRuntime([_Model("anthropic", "claude")]), "zzzz", lines.append)
    assert lines == ['No models matching "zzzz"']


# ---------------------------------------------------------------------------
# session selection
# ---------------------------------------------------------------------------


def test_validate_fork_flags_rejects_conflicts() -> None:
    for flags in (["--session", "x"], ["--continue"], ["--resume"], ["--no-session"]):
        args = parse_args(["--fork", "abc", *flags])
        with pytest.raises(SessionSelectionError, match="--fork cannot be combined"):
            validate_fork_flags(args)


def test_validate_fork_flags_allows_fork_alone() -> None:
    validate_fork_flags(parse_args(["--fork", "abc"]))
    validate_fork_flags(parse_args([]))


def test_validate_session_id_flags_rejects_conflicts() -> None:
    for flags in (["--session", "x"], ["--continue"], ["--resume"]):
        args = parse_args(["--session-id", "01234567-89ab-7def-8123-456789abcdef", *flags])
        with pytest.raises(SessionSelectionError, match="--session-id cannot be combined"):
            validate_session_id_flags(args)


def test_validate_session_id_flags_rejects_invalid_id() -> None:
    with pytest.raises(SessionSelectionError, match="Error:"):
        validate_session_id_flags(parse_args(["--session-id", "not a uuid"]))


@pytest.mark.asyncio
async def test_resolve_session_path_treats_paths_as_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for arg in ("./a.jsonl", "sub/dir/x", "x.jsonl"):
            resolved = await resolve_session_path(arg, tmp)
            assert resolved.type == "path"
            assert os.path.isabs(resolved.path)


@pytest.mark.asyncio
async def test_resolve_session_path_reports_not_found() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        resolved = await resolve_session_path("deadbeef", tmp, tmp)
        assert resolved.type == "not_found"
        assert resolved.arg == "deadbeef"


# ---------------------------------------------------------------------------
# wire encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("id", "id"),
        ("stop_reason", "stopReason"),
        ("assistant_message_event", "assistantMessageEvent"),
        ("continue_", "continue"),
        ("a_b_c_d", "aBCD"),
        ("", ""),
    ],
)
def test_snake_to_camel(name: str, expected: str) -> None:
    assert snake_to_camel(name) == expected


@dataclass
class _Nested:
    total_tokens: int = 5


@dataclass
class _WireSample:
    stop_reason: str = "stop"
    parent_session: str | None = None
    nested: _Nested = field(default_factory=_Nested)
    items: list[_Nested] = field(default_factory=list)
    arguments: dict[str, Any] = field(default_factory=dict)


def test_to_wire_camel_cases_dataclass_fields() -> None:
    assert to_wire(_WireSample())["stopReason"] == "stop"


def test_to_wire_drops_none_like_json_stringify_drops_undefined() -> None:
    assert "parentSession" not in to_wire(_WireSample())


def test_to_wire_recurses_into_nested_dataclasses_and_lists() -> None:
    payload = to_wire(_WireSample(items=[_Nested(7)]))
    assert payload["nested"] == {"totalTokens": 5}
    assert payload["items"] == [{"totalTokens": 7}]


def test_to_wire_leaves_plain_dict_keys_alone() -> None:
    payload = to_wire(_WireSample(arguments={"file_path": "a.py", "old_str": "x"}))
    assert payload["arguments"] == {"file_path": "a.py", "old_str": "x"}


def test_to_wire_passes_scalars_through() -> None:
    assert to_wire(1) == 1
    assert to_wire("x") == "x"
    assert to_wire(True) is True
    assert to_wire(None) is None


# ---------------------------------------------------------------------------
# json events
# ---------------------------------------------------------------------------


@dataclass
class _DeltaEvent:
    type: str = "text_delta"
    delta: str = "hi"
    partial: Any = None


@dataclass
class _MessageUpdate:
    type: str = "message_update"
    assistant_message_event: Any = None


def test_to_json_event_strips_partial_snapshot() -> None:
    event = _MessageUpdate(assistant_message_event=_DeltaEvent(partial={"content": []}))
    payload = to_json_event(event)
    assert payload["type"] == "message_update"
    assert "partial" not in payload["assistantMessageEvent"]
    assert payload["assistantMessageEvent"]["delta"] == "hi"


def test_to_json_event_passes_other_events_through() -> None:
    @dataclass
    class _Start:
        type: str = "agent_start"

    assert to_json_event(_Start()) == {"type": "agent_start"}


def test_to_json_event_camel_cases_keys() -> None:
    @dataclass
    class _TurnEnd:
        type: str = "turn_end"
        tool_results: list[Any] = field(default_factory=list)

    assert to_json_event(_TurnEnd()) == {"type": "turn_end", "toolResults": []}


# ---------------------------------------------------------------------------
# output guard
# ---------------------------------------------------------------------------


def test_take_over_stdout_redirects_print_to_stderr() -> None:
    real_stdout, real_stderr = os.sys.stdout, os.sys.stderr
    fake_out, fake_err = io.StringIO(), io.StringIO()
    os.sys.stdout, os.sys.stderr = fake_out, fake_err
    try:
        take_over_stdout()
        print("noise")
        restore_stdout()
    finally:
        os.sys.stdout, os.sys.stderr = real_stdout, real_stderr

    assert "noise" in fake_err.getvalue()
    assert fake_out.getvalue() == ""


# ---------------------------------------------------------------------------
# JS number semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [(1.25, 1, "1.3"), (1.35, 1, "1.4"), (1.95, 1, "1.9"), (0.5, 0, "1"), (2.675, 2, "2.67")],
)
def test_to_fixed_matches_js(value: float, digits: int, expected: str) -> None:
    assert to_fixed(value, digits) == expected


@pytest.mark.parametrize(("value", "expected"), [(0.5, 1), (1.5, 2), (2.5, 3), (-0.5, 0), (-1.5, -1)])
def test_js_round_ties_towards_positive_infinity(value: float, expected: int) -> None:
    assert js_round(value) == expected


# ---------------------------------------------------------------------------
# piped stdin
# ---------------------------------------------------------------------------


class _FakeStdin:
    def __init__(self, data: str = "", *, tty: bool = False, error: Exception | None = None) -> None:
        self._data = data
        self._tty = tty
        self._error = error

    def isatty(self) -> bool:
        return self._tty

    def read(self) -> str:
        if self._error is not None:
            raise self._error
        return self._data


def test_read_piped_stdin_returns_none_on_a_tty(monkeypatch: Any) -> None:
    monkeypatch.setattr(entry.sys, "stdin", _FakeStdin("ignored", tty=True))
    assert entry.read_piped_stdin(True) is None


def test_read_piped_stdin_trims_content(monkeypatch: Any) -> None:
    monkeypatch.setattr(entry.sys, "stdin", _FakeStdin("  hello \n"))
    assert entry.read_piped_stdin(False) == "hello"


def test_read_piped_stdin_treats_whitespace_as_empty(monkeypatch: Any) -> None:
    monkeypatch.setattr(entry.sys, "stdin", _FakeStdin("   \n\t "))
    assert entry.read_piped_stdin(False) is None


def test_read_piped_stdin_tolerates_unreadable_stdin(monkeypatch: Any) -> None:
    """A closed stdin must not crash startup; pytest supplies one of these."""
    monkeypatch.setattr(entry.sys, "stdin", _FakeStdin(error=OSError("closed")))
    assert entry.read_piped_stdin(False) is None
    monkeypatch.setattr(entry.sys, "stdin", _FakeStdin(error=ValueError("detached")))
    assert entry.read_piped_stdin(False) is None


def test_piped_stdin_downgrades_interactive_to_print(monkeypatch: Any) -> None:
    """Content on stdin means there is nobody to interact with."""
    seen: dict[str, Any] = {}

    async def fake_run_app(parsed, app_mode, processed_files=None, stdin_content=None):
        seen["app_mode"] = app_mode
        seen["stdin_content"] = stdin_content
        return 0

    monkeypatch.setattr(entry, "run_app", fake_run_app)
    monkeypatch.setattr(entry.sys, "stdin", _FakeStdin("do the thing", tty=True))
    monkeypatch.setattr(entry.sys.stdout, "isatty", lambda: True)
    # `resolve_app_mode` sees a TTY, but `read_piped_stdin` is asked directly.
    monkeypatch.setattr(entry, "read_piped_stdin", lambda tty: "do the thing")

    assert entry.main([]) == 0
    assert seen == {"app_mode": "print", "stdin_content": "do the thing"}

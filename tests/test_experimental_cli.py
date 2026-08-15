"""Tests for `pi_coding_agent.cli.experimental`.

Python port of `packages/coding-agent/test/experimental-cli-command.test.ts`.

The experimental CLI's job is to peel off its own options and hand everything
else to the legacy parser untouched, so most of these cases pin exactly where
that handoff happens.
"""

from __future__ import annotations

import pytest

from pi_coding_agent.cli.experimental import experimental_cli
from pi_coding_agent.cli.experimental.auth import FileAuthInput, TokenAuthInput
from pi_coding_agent.cli.experimental.transport_address import UnixTransportAddress


def test_selects_pi_mode_and_parses_existing_cli_arguments():
    result = experimental_cli.parse(
        ["--provider", "anthropic", "--model", "claude-sonnet", "--thinking", "high", "inspect", "the project"]
    )

    assert result.ok
    assert result.command.command == "pi"
    assert result.command.options.provider == "anthropic"
    assert result.command.options.model == "claude-sonnet"
    assert result.command.options.thinking == "high"
    assert result.command.options.messages == ["inspect", "the project"]


def test_parses_a_server_listener():
    result = experimental_cli.parse(["server", "--listen", "unix:///tmp/pi.sock"])

    assert result.ok
    assert result.command.command == "server"
    assert result.command.listen == [UnixTransportAddress(path="/tmp/pi.sock")]
    assert result.command.auth is None


def test_parses_a_client_transport_address():
    result = experimental_cli.parse(["client", "--connect", "unix:///tmp/pi.sock"])

    assert result.ok
    assert result.command.command == "client"
    assert result.command.connect == UnixTransportAddress(path="/tmp/pi.sock")


def test_leaves_experimental_looking_existing_option_values_with_the_existing_parser():
    result = experimental_cli.parse(["--system-prompt", "--listen", "unix:///tmp/pi.sock"])

    assert result.ok
    assert result.command.command == "pi"
    assert result.command.options.system_prompt == "--listen"
    assert result.command.options.messages == ["unix:///tmp/pi.sock"]


def test_stops_parsing_command_options_when_existing_cli_arguments_begin():
    result = experimental_cli.parse(["--model", "claude-sonnet", "--listen=unix:///tmp/second.sock"])

    assert result.ok
    assert result.command.options.model == "claude-sonnet"
    # `--model` is not an experimental option, so everything from it onward went
    # to the legacy parser and the later `--listen` never became a listener.
    assert result.command.listen == []
    assert result.command.options.unknown_flags["listen"] == "unix:///tmp/second.sock"


@pytest.mark.parametrize(
    ("argv", "auth"),
    [
        (["--auth-token", "secret"], TokenAuthInput(token="secret")),
        (["--auth-token-file", "/tmp/token"], FileAuthInput(path="/tmp/token")),
    ],
)
def test_parses_authentication_source(argv, auth):
    result = experimental_cli.parse(argv)

    assert result.ok
    assert result.command.command == "pi"
    assert result.command.auth == auth


@pytest.mark.parametrize("argv", [[], ["server"], ["client"]])
def test_permits_omitted_authentication(argv):
    result = experimental_cli.parse(argv)

    assert result.ok
    assert result.command.command == (argv[0] if argv else "pi")
    assert result.command.auth is None


def test_passes_unknown_options_file_arguments_and_the_separator_to_the_existing_parser():
    result = experimental_cli.parse(["--unknown", "@prompt.md", "--", "--listen", "unix:///tmp/pi.sock"])

    assert result.ok
    assert result.command.options.file_args == ["prompt.md"]
    assert result.command.options.unknown_flags["unknown"] is True
    assert result.command.options.unknown_flags["listen"] == "unix:///tmp/pi.sock"


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (
            ["--listen", "unix:///tmp/pi.sock", "--listen", "unix:///tmp/pi-admin.sock"],
            "--listen may only be specified once",
        ),
        (
            ["--auth-token", "secret", "--auth-token-file", "/tmp/token"],
            "--auth-token and --auth-token-file are mutually exclusive",
        ),
        (["--auth-token", "first", "--auth-token", "second"], "--auth-token may only be specified once"),
        (
            ["--auth-token-file", "/tmp/first", "--auth-token-file=/tmp/second"],
            "--auth-token-file may only be specified once",
        ),
        (["--listen", "/tmp/pi.sock"], 'Invalid --listen address "/tmp/pi.sock"'),
        (["--listen", "ws://localhost:8080"], 'Unsupported --listen transport "ws:"'),
        (["--listen", "unix://relative.sock"], "Unix transport address must not include an authority"),
        (["--listen", "unix:///tmp/pi.sock?wrong=value"], 'Invalid --listen address "unix:///tmp/pi.sock?wrong=value"'),
        (["--listen", "unix:///tmp/pi.sock#fragment"], 'Invalid --listen address "unix:///tmp/pi.sock#fragment"'),
        (["--listen", "unix:/tmp/pi.sock"], 'Invalid --listen address "unix:/tmp/pi.sock"'),
        (["--listen", "unix:///tmp/%00pi.sock"], 'Invalid --listen address "unix:///tmp/%00pi.sock"'),
        (
            ["client", "--listen", "unix:///tmp/pi.sock"],
            "The experimental client command does not support existing CLI options yet",
        ),
        (
            ["server", "--connect", "unix:///tmp/pi.sock"],
            "The experimental server command does not support existing CLI options yet",
        ),
        (["client", "--connect", "ws://localhost:8080"], 'Unsupported --connect transport "ws:"'),
        (["--listen"], "--listen requires a value"),
        (["--connect="], "--connect is only valid for client mode"),
    ],
)
def test_rejects_invalid_experimental_input(argv, error):
    result = experimental_cli.parse(argv)

    assert not result.ok
    assert any(error in message for message in result.errors)


def test_rejects_unsupported_options_without_parsing_them():
    result = experimental_cli.parse(
        ["client", "--listen", "ws://localhost:8080", "--auth-token", "secret", "--auth-token-file", "/tmp/token"]
    )

    assert not result.ok
    assert result.errors == ["The experimental client command does not support existing CLI options yet"]


def test_treats_command_names_after_the_first_argument_as_existing_cli_arguments():
    result = experimental_cli.parse(["--cwd", "/workspace", "server"])

    assert result.ok
    assert result.command.command == "pi"
    assert result.command.options.messages == ["server"]


@pytest.mark.asyncio
async def test_execute_dispatches_to_the_context_action():
    class Context:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run_pi(self, command):
            self.calls.append(f"pi:{command.command}")

        def run_server(self, command):
            self.calls.append(f"server:{command.listen[0].path}")

        def run_client(self, command):
            self.calls.append(f"client:{command.connect.path}")

    context = Context()
    assert (await experimental_cli.execute([], context)).ok
    assert (await experimental_cli.execute(["server", "--listen", "unix:///tmp/s.sock"], context)).ok
    assert (await experimental_cli.execute(["client", "--connect", "unix:///tmp/c.sock"], context)).ok

    assert context.calls == ["pi:pi", "server:/tmp/s.sock", "client:/tmp/c.sock"]


@pytest.mark.asyncio
async def test_execute_returns_errors_without_running_an_action():
    class Context:
        def run_pi(self, command):
            raise AssertionError("action must not run for invalid input")

    result = await experimental_cli.execute(["--listen", "unix:/bad"], Context())

    assert not result.ok
    assert result.errors

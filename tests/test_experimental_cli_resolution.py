"""Python port of `packages/coding-agent/test/experimental-cli-resolution.test.ts`.

Pins how the experimental CLI composes the `pi`/`server`/`client` commands with
the legacy argument parser: which options each command still refuses, and the
order errors are reported in.
"""

from __future__ import annotations

from typing import Any

import pytest

from pi_coding_agent.cli.experimental import experimental_cli
from pi_coding_agent.cli.experimental.auth import TokenAuthInput
from pi_coding_agent.cli.experimental.transport_address import UnixTransportAddress

UNSUPPORTED_SERVER_OPTIONS = "The experimental server command does not support existing CLI options yet"
UNSUPPORTED_CLIENT_OPTIONS = "The experimental client command does not support existing CLI options yet"


def test_composes_pi_command_options_with_the_existing_parser() -> None:
    result = experimental_cli.parse(
        [
            "--listen",
            "unix:///tmp/pi.sock",
            "--auth-token",
            "secret",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet",
            "--thinking",
            "high",
            "inspect",
        ]
    )

    assert result.ok
    assert result.command.command == "pi"
    assert result.command.listen == [UnixTransportAddress(path="/tmp/pi.sock")]
    assert result.command.auth == TokenAuthInput(token="secret")
    assert result.command.options.provider == "anthropic"
    assert result.command.options.model == "claude-sonnet"
    assert result.command.options.thinking == "high"
    assert result.command.options.messages == ["inspect"]


@pytest.mark.parametrize(("option", "attribute"), [("--help", "help"), ("--version", "version")])
def test_keeps_pi_help_and_version_handling_in_existing_cli_options(option: str, attribute: str) -> None:
    result = experimental_cli.parse([option])

    assert result.ok
    assert result.command.command == "pi"
    assert getattr(result.command.options, attribute) is True


@pytest.mark.parametrize(
    ("command", "option", "error"),
    [
        ("server", "--help", UNSUPPORTED_SERVER_OPTIONS),
        ("server", "--version", UNSUPPORTED_SERVER_OPTIONS),
        ("client", "--help", UNSUPPORTED_CLIENT_OPTIONS),
        ("client", "--version", UNSUPPORTED_CLIENT_OPTIONS),
    ],
)
def test_rejects_deferred_help_and_version_handling(command: str, option: str, error: str) -> None:
    result = experimental_cli.parse([command, option])

    assert not result.ok
    assert result.errors == [error]


def test_rejects_existing_options_that_the_server_command_does_not_support_yet() -> None:
    result = experimental_cli.parse(["server", "--model", "claude-sonnet", "prompt"])

    assert not result.ok
    assert result.errors == [UNSUPPORTED_SERVER_OPTIONS]


def test_rejects_existing_options_that_the_client_command_does_not_support_yet() -> None:
    result = experimental_cli.parse(["client", "--tui-mode", "fullscreen", "@prompt.md"])

    assert not result.ok
    assert result.errors == [UNSUPPORTED_CLIENT_OPTIONS]


def test_reports_existing_parser_errors_before_capability_errors() -> None:
    result = experimental_cli.parse(["client", "--tui-mode", "wrong", "--model", "claude-sonnet"])

    assert not result.ok
    assert result.errors == [
        'Invalid TUI mode "wrong". Valid values: regular, fullscreen',
        UNSUPPORTED_CLIENT_OPTIONS,
    ]


def test_parses_an_empty_server_command() -> None:
    result = experimental_cli.parse(["server"])

    assert result.ok
    assert result.command.command == "server"
    # TypeScript asserts `toEqual({ ok: true, command: { command: "server" } })`
    # and only spreads `auth`/`listen` into the object when they are set, so the
    # equivalent here is that both stay at their empty defaults.
    assert result.command.auth is None
    assert result.command.listen == []
    assert result.errors == []


@pytest.mark.parametrize("name", ["pi", "server", "client"])
async def test_executes_the_parsed_command(name: str) -> None:
    class Context:
        def __init__(self) -> None:
            self.calls: dict[str, int] = {"pi": 0, "server": 0, "client": 0}

        def run_pi(self, command: Any) -> None:
            self.calls["pi"] += 1

        def run_server(self, command: Any) -> None:
            self.calls["server"] += 1

        def run_client(self, command: Any) -> None:
            self.calls["client"] += 1

    context = Context()
    result = await experimental_cli.execute([] if name == "pi" else [name], context)

    assert result.ok
    assert result.command.command == name
    assert context.calls == {key: (1 if key == name else 0) for key in ("pi", "server", "client")}

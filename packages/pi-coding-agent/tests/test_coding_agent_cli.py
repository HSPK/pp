"""Focused tests for CLI helpers without provider network requests."""

from __future__ import annotations

import io
import os
from types import SimpleNamespace

import pi_coding_agent.cli.entry as cli
import pytest
from pi_agent import AgentTool, MessageEndEvent, ToolExecutionStartEvent
from pi_ai import AssistantMessage, Model, TextContent, ToolResultMessage
from pi_ai.providers import openai_compatible_provider
from pi_ai.registry import Models
from pi_coding_agent.tools import ALL_TOOL_NAMES


class NonTtyInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def make_model(model_id: str, name: str) -> Model:
    return Model(
        id=model_id,
        name=name,
        api="openai-completions",
        context_window=8_000,
        max_tokens=1_000,
    )


def make_sample_models() -> Models:
    return Models(
        [
            openai_compatible_provider(
                "fake",
                "Fake",
                "https://fake.invalid/v1",
                ["FAKE_API_KEY"],
                [make_model("alpha", "Alpha")],
            ),
            openai_compatible_provider(
                "alt",
                "Alt",
                "https://alt.invalid/v1",
                ["ALT_API_KEY"],
                [make_model("beta", "Beta")],
            ),
        ]
    )


def clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in (
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "PI_API_KEY",
        "FAKE_API_KEY",
        "ALT_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)


def test_build_parser_parses_supported_arguments():
    args = cli.build_parser().parse_args(
        [
            "--list-models",
            "--base-url",
            "https://example.invalid/v1",
            "--model",
            "fake/alpha",
            "hello",
            "world",
        ]
    )

    assert args.list_models is True
    assert args.base_url == "https://example.invalid/v1"
    assert args.model == "fake/alpha"
    assert args.prompt == ["hello", "world"]


def test_build_models_adds_custom_provider_only_when_base_url_is_given():
    default_models = cli.build_models()
    custom_models = cli.build_models("https://custom.invalid/v1", "custom-model")

    assert default_models.get_provider("custom") is None

    custom_provider = custom_models.get_provider("custom")
    assert custom_provider is not None
    assert len(custom_models.get_providers()) == len(default_models.get_providers()) + 1
    assert custom_provider.base_url == "https://custom.invalid/v1"
    assert tuple(custom_provider.auth.api_key.env_vars) == ("PI_API_KEY", "OPENAI_API_KEY")

    custom_model = custom_models.get_model("custom", "custom-model")
    assert custom_model is not None
    assert custom_model.provider == "custom"
    assert custom_model.base_url == "https://custom.invalid/v1"


def test_resolve_model_supports_explicit_and_bare_references():
    models = make_sample_models()

    explicit = cli.resolve_model(models, "fake/alpha")
    bare = cli.resolve_model(models, "beta")

    assert explicit.provider == "fake"
    assert explicit.id == "alpha"
    assert bare.provider == "alt"
    assert bare.id == "beta"


def test_resolve_model_reports_unknown_models():
    models = make_sample_models()

    with pytest.raises(SystemExit) as excinfo:
        cli.resolve_model(models, "missing")

    message = str(excinfo.value)
    assert "Unknown model: missing" in message
    assert "Known models: fake/alpha, alt/beta" in message


def test_resolve_model_skips_unset_env_vars_before_a_match(monkeypatch):
    # First provider has an env var list where the first entry is unset and
    # the second is set: the inner loop must continue past the miss instead
    # of stopping at the first candidate. A second, fully-unconfigured
    # provider precedes it so the outer loop also continues past a full miss.
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("ALT_API_KEY", "secret")
    models = Models(
        [
            openai_compatible_provider(
                "unset",
                "Unset",
                "https://unset.invalid/v1",
                ["UNSET_API_KEY"],
                [make_model("gamma", "Gamma")],
            ),
            openai_compatible_provider(
                "fake",
                "Fake",
                "https://fake.invalid/v1",
                ["FAKE_API_KEY", "ALT_API_KEY"],
                [make_model("alpha", "Alpha")],
            ),
        ]
    )

    model = cli.resolve_model(models, None)

    assert model.provider == "fake"
    assert model.id == "alpha"


def test_resolve_model_uses_first_configured_provider(monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    models = cli.build_models()

    model = cli.resolve_model(models, None)

    assert model.provider == "deepseek"
    assert model.id in {m.id for m in models.get_models("deepseek")}


def test_resolve_model_requires_a_selection_or_configured_provider(monkeypatch):
    clear_provider_env(monkeypatch)
    models = cli.build_models()

    with pytest.raises(SystemExit) as excinfo:
        cli.resolve_model(models, None)

    message = str(excinfo.value)
    assert "No model selected and no provider API key found in the environment." in message
    assert "OPENAI_API_KEY" in message
    assert "DEEPSEEK_API_KEY" in message


def test_format_event_renders_assistant_output_and_errors():
    output = cli.format_event(MessageEndEvent(message=AssistantMessage(content=[TextContent(text="All done.")])))
    error = cli.format_event(
        MessageEndEvent(
            message=AssistantMessage(
                content=[TextContent(text="ignored")],
                error_message="rate limited",
            )
        )
    )

    assert output == "All done."
    assert error == "error: rate limited"


def test_format_event_renders_tool_results_and_tool_starts():
    ok = cli.format_event(
        MessageEndEvent(
            message=ToolResultMessage(
                tool_name="read",
                content=[TextContent(text="line one\nline two")],
            )
        )
    )
    error = cli.format_event(
        MessageEndEvent(
            message=ToolResultMessage(
                tool_name="read",
                content=[TextContent(text="bad path")],
                is_error=True,
            )
        )
    )
    start = cli.format_event(
        ToolExecutionStartEvent(
            tool_call_id="call-1",
            tool_name="read",
            args={"path": "a.txt"},
        )
    )

    assert ok == "  [read ok] line one"
    assert error == "  [read error] bad path"
    assert start == "  -> read({'path': 'a.txt'})"


def test_format_event_ignores_unhandled_events():
    assert cli.format_event(SimpleNamespace(type="turn_start")) is None


def test_main_lists_models(monkeypatch):
    """`--list-models` routes to the TS-parity table printer, not the agent loop."""
    calls: dict[str, object] = {}

    async def fake_handle(search=None, *, agent_dir=None, **kwargs):
        calls["search"] = search
        return 0

    monkeypatch.setattr(cli, "handle_list_models", fake_handle)

    assert cli.main(["--list-models"]) == 0
    assert calls == {"search": None}


def test_main_list_models_forwards_search_pattern(monkeypatch):
    calls: dict[str, object] = {}

    async def fake_handle(search=None, *, agent_dir=None, **kwargs):
        calls["search"] = search
        return 0

    monkeypatch.setattr(cli, "handle_list_models", fake_handle)

    assert cli.main(["--list-models", "sonnet"]) == 0
    assert calls == {"search": "sonnet"}


def test_main_requires_a_non_empty_prompt(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_models", lambda base_url=None, model_id=None: make_sample_models())
    monkeypatch.setattr(cli.sys, "stdin", NonTtyInput(""))

    with pytest.raises(SystemExit) as excinfo:
        raise SystemExit(cli.main([]))

    assert excinfo.value.code == 2
    assert "No prompt given." in capsys.readouterr().err


def test_main_runs_a_prompt_through_print_mode(monkeypatch):
    """A prompt on a non-TTY stdin runs print mode and returns its exit code."""
    calls: dict[str, object] = {}

    async def fake_build(parsed, cwd, agent_dir, project_trusted=True):
        calls["messages"] = list(parsed.messages)
        calls["model"] = parsed.model
        return object()

    async def fake_print_mode(runtime, options):
        calls["mode"] = options.mode
        calls["initial_message"] = options.initial_message
        calls["remaining"] = list(options.messages)
        return 7

    monkeypatch.setattr(cli, "build_session_runtime", fake_build)
    monkeypatch.setattr(cli, "run_print_mode", fake_print_mode)
    monkeypatch.setattr(cli.sys, "stdin", NonTtyInput(""))

    exit_code = cli.main(["--model", "alpha", "say", "hi"])

    assert exit_code == 7
    assert calls == {
        "messages": ["say", "hi"],
        "model": "alpha",
        "mode": "text",
        "initial_message": "say",
        "remaining": ["hi"],
    }


def test_main_json_mode_uses_json_output(monkeypatch):
    calls: dict[str, object] = {}

    async def fake_build(parsed, cwd, agent_dir, project_trusted=True):
        return object()

    async def fake_print_mode(runtime, options):
        calls["mode"] = options.mode
        return 0

    monkeypatch.setattr(cli, "build_session_runtime", fake_build)
    monkeypatch.setattr(cli, "run_print_mode", fake_print_mode)

    assert cli.main(["--mode", "json", "hi"]) == 0
    assert calls == {"mode": "json"}


def test_main_reports_unported_rpc_mode(capsys):
    assert cli.main(["--mode", "rpc"]) == 2
    assert "RPC mode is not ported" in capsys.readouterr().err


def test_main_prints_version(capsys):
    from pi_coding_agent.core.config import VERSION

    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == VERSION


def test_main_prints_help(capsys):
    assert cli.main(["--help"]) == 0
    assert "pi - AI coding assistant" in capsys.readouterr().out


def test_main_offline_flag_sets_environment(monkeypatch):
    """`--offline` sets PI_OFFLINE for every downstream network gate.

    `main` mutates the real environment, so the original value is restored
    explicitly; `monkeypatch.delenv` records nothing to restore when the
    variable was already unset, and the leaked "1" would put unrelated tests
    into offline mode.
    """
    original = os.environ.get("PI_OFFLINE")

    async def fake_handle(search=None, *, agent_dir=None, **kwargs):
        return 0

    monkeypatch.setattr(cli, "handle_list_models", fake_handle)
    try:
        cli.main(["--offline", "--list-models"])
        assert os.environ["PI_OFFLINE"] == "1"
    finally:
        if original is None:
            os.environ.pop("PI_OFFLINE", None)
        else:
            os.environ["PI_OFFLINE"] = original


def test_load_default_tools_returns_agent_tools(tmp_path):
    tools = cli.load_default_tools(str(tmp_path))

    assert len(tools) == len(ALL_TOOL_NAMES)
    assert {tool.name for tool in tools} == ALL_TOOL_NAMES
    assert all(isinstance(tool, AgentTool) for tool in tools)

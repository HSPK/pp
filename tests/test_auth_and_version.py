"""Tests for auth commands, fullscreen mode and the GitHub version check."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from pi_tui.testing import FakeTerminal
from pi_tui.tui_alt_screen import TuiAltScreen
from pi_tui.tui_main_screen import TuiMainScreen

from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.modes.interactive.interactive_mode import (
    InteractiveMode,
    InteractiveModeOptions,
    create_interactive_tui,
)
from pi_coding_agent.utils.version_check import (
    DEFAULT_VERSION_CHECK_PACKAGE,
    check_for_new_pi_version,
    compare_package_versions,
    format_version_check_error,
    get_latest_pi_release,
    get_pi_user_agent,
    is_newer_package_version,
    resolve_version_check_package,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[=>]")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _run(coro: Any, timeout: float = 30.0) -> Any:
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


async def _wait_for_selector(mode: InteractiveMode, timeout_s: float = 5.0) -> Any:
    """Wait until a fire-and-forget login task has installed its dialog.

    The alternative -- a fixed `asyncio.sleep(0.05)` -- ties a positive
    assertion to wall-clock time, so under the parallel suite's load the task
    can simply not have got there yet. Polling the actual condition on
    zero-delay ticks pins the same thing without a clock.
    """
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    while mode._active_selector is None:
        if loop.time() - started_at > timeout_s:
            raise AssertionError("Timed out waiting for the login dialog")
        await asyncio.sleep(0)
    return mode._active_selector


async def _make_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tui_mode: str | None = None
) -> InteractiveMode:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("PI_OFFLINE", "1")

    from pi_ai.providers.faux import faux_provider

    from pi_coding_agent.core.model_runtime import ModelRuntime

    faux = faux_provider()
    model_runtime = await ModelRuntime.create(agent_dir=str(agent_dir), providers=[faux.provider])
    await model_runtime.login(faux.provider.id, "faux-key")
    options = CreateAgentSessionOptions(
        cwd=str(cwd), agent_dir=str(agent_dir), model=faux.models[0], model_runtime=model_runtime
    )
    result = await create_agent_session(options)

    async def create_runtime(**_kwargs: Any) -> Any:
        return await create_agent_session(options)

    runtime = AgentSessionRuntime(result.session, str(agent_dir), create_runtime, result.model_fallback_message)
    mode_options = InteractiveModeOptions(tui_mode=tui_mode)
    return InteractiveMode(runtime, mode_options, terminal=FakeTerminal(100, 30))


# --------------------------------------------------------------------------
# TUI mode selection
# --------------------------------------------------------------------------


def test_create_interactive_tui_selects_the_renderer():
    assert isinstance(create_interactive_tui(terminal=FakeTerminal()), TuiMainScreen)
    assert isinstance(create_interactive_tui(tui_mode="fullscreen", terminal=FakeTerminal()), TuiAltScreen)


def test_regular_mode_is_the_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            assert mode.tui_mode == "regular"
            assert isinstance(mode.renderer, TuiMainScreen)
            await mode.init()
            assert mode.transcript_scroll_view is None
        finally:
            await mode.shutdown()

    _run(scenario())


def test_fullscreen_from_the_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "settings.json").write_text(json.dumps({"tuiMode": "fullscreen"}), encoding="utf-8")
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            assert mode.tui_mode == "fullscreen"
            assert isinstance(mode.renderer, TuiAltScreen)
        finally:
            await mode.shutdown()

    _run(scenario())


def test_fullscreen_option_overrides_the_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch, tui_mode="fullscreen")
        try:
            assert isinstance(mode.renderer, TuiAltScreen)
            await mode.init()
            # Fullscreen pins the dock and scrolls only the transcript.
            assert mode.transcript_scroll_view is not None
            assert mode.renderer._layout_root is not None
        finally:
            await mode.shutdown()

    _run(scenario())


def test_fullscreen_wraps_the_transcript_in_a_scroll_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch, tui_mode="fullscreen")
        try:
            await mode.init()
            # The alt screen renders through its layout engine (which needs a
            # viewport height), so assert the wiring and the transcript content
            # rather than the composed frame; the frame itself is covered by
            # the pi-tui alt-screen tests.
            assert mode.transcript_scroll_view is not None
            assert mode.document_container in mode.transcript_scroll_view.children
            assert mode.renderer.get_mounted_roots() == [mode.renderer._layout_root]
            body = _strip("\n".join(mode.document_container.render(90)))
            assert "pi v" in body
        finally:
            await mode.shutdown()

    _run(scenario())


def test_regular_mode_mounts_containers_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            roots = mode.renderer.get_mounted_roots()
            assert mode.document_container in roots
            assert mode.editor_container in roots
            assert mode.footer_container in roots
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# login / logout
# --------------------------------------------------------------------------


def test_login_provider_options_cover_every_auth_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            options = mode.get_login_provider_options()
            assert options
            assert all(option.auth_type in ("oauth", "api_key") for option in options)
            assert [option.name for option in options] == sorted(o.name for o in options)
        finally:
            await mode.shutdown()

    _run(scenario())


def test_login_command_opens_the_provider_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/login")
            assert mode._active_selector is not None
            rendered = _strip("\n".join(mode._active_selector.render(80)))
            assert "Select provider to configure" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_login_with_a_provider_name_opens_the_dialog_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            provider_id = mode.get_login_provider_options()[0].id
            # The login dialog blocks until the user answers, so the submit
            # handler is fire-and-forget here exactly as the editor drives it.
            task = asyncio.ensure_future(mode._handle_submit(f"/login {provider_id}"))
            await _wait_for_selector(mode)
            assert mode._active_selector is not None
            rendered = _strip("\n".join(mode._active_selector.render(80)))
            assert "Login to" in rendered
            mode._active_selector.cancel()
            await asyncio.wait_for(task, timeout=5)
        finally:
            await mode.shutdown()

    _run(scenario())


def test_api_key_login_persists_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            option = next(o for o in mode.get_login_provider_options() if o.auth_type == "api_key")
            task = asyncio.ensure_future(mode._start_provider_login(option))
            dialog = await _wait_for_selector(mode)

            assert dialog is not None
            dialog.input.set_value("sk-test-key")
            dialog.input.on_submit("sk-test-key")
            await asyncio.wait_for(task, timeout=10)

            auth = json.loads((tmp_path / "agent" / "auth.json").read_text(encoding="utf-8"))
            assert auth[option.id]["key"] == "sk-test-key"
            rendered = _strip("\n".join(mode.chat_container.render(90)))
            assert f"Logged in to {option.name}" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_login_cancel_reports_and_restores_the_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            option = next(o for o in mode.get_login_provider_options() if o.auth_type == "api_key")
            task = asyncio.ensure_future(mode._start_provider_login(option))
            dialog = await _wait_for_selector(mode)

            assert dialog is not None
            dialog.cancel()
            await asyncio.wait_for(task, timeout=10)

            assert mode._active_selector is None
            assert mode.editor in mode.editor_container.children
            rendered = _strip("\n".join(mode.chat_container.render(80)))
            assert "Login cancelled" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_logout_lists_only_configured_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.show_oauth_selector("logout")
            selector = mode._active_selector
            assert selector is not None
            # The faux provider was logged in by the harness.
            assert len(selector.all_providers) >= 1
            assert all(
                provider.status is not None and provider.status.configured for provider in selector.all_providers
            )
        finally:
            await mode.shutdown()

    _run(scenario())


def test_logout_removes_the_credential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            configured = [
                option
                for option in mode.get_login_provider_options()
                if option.status is not None and option.status.configured
            ]
            assert configured
            await mode._logout_provider(configured[0])

            auth = json.loads((tmp_path / "agent" / "auth.json").read_text(encoding="utf-8"))
            assert configured[0].id not in auth
            rendered = _strip("\n".join(mode.chat_container.render(80)))
            assert f"Logged out of {configured[0].name}" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_provider_auth_status_reports_stored_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            runtime = mode.session.model_runtime
            provider_id = mode.session.model.provider
            status = runtime.get_provider_auth_status(provider_id)
            assert status.configured is True
            assert status.type == "api_key"
            assert runtime.is_using_oauth(provider_id) is False
            assert runtime.get_provider_auth_status("does-not-exist").configured is False
        finally:
            await mode.shutdown()

    _run(scenario())


def test_session_command_reports_session_details(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/session")
            rendered = _strip("\n".join(mode.chat_container.render(110)))
            assert "Session" in rendered
            assert mode.session_manager.get_session_id() in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# version check
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("1.0.0", "1.0.0", 0),
        ("1.0.1", "1.0.0", 1),
        ("1.0.0", "1.0.1", -1),
        ("2.0.0", "1.9.9", 1),
        ("1.0.0-alpha", "1.0.0", -1),
        ("1.0.0", "1.0.0-alpha", 1),
        ("1.0.0-alpha.1", "1.0.0-alpha", 1),
        ("1.0.0-alpha.2", "1.0.0-alpha.10", -1),
        ("1.0.0-alpha", "1.0.0-beta", -1),
        ("v1.2.3", "1.2.3", 0),
        ("not-semver", "1.0.0", None),
    ],
)
def test_compare_package_versions(left: str, right: str, expected: int | None):
    assert compare_package_versions(left, right) == expected


def test_an_unparseable_version_is_not_an_upgrade():
    """Upstream's "any difference means newer" cannot be used on PyPI.

    PyPI serves PEP 440, so a fallback that treats every difference as an
    upgrade offers `0.2.0rc1` over the stable `0.2.0` it precedes.
    """
    assert is_newer_package_version("nightly-2", "nightly-1") is False
    assert is_newer_package_version("nightly", "nightly") is False


def test_resolve_package_precedence(tmp_path: Path):
    class FakeSettings:
        def get_version_check_package(self) -> str:
            return "from-settings"

    assert resolve_version_check_package(env={}) == DEFAULT_VERSION_CHECK_PACKAGE
    assert resolve_version_check_package(settings_manager=FakeSettings(), env={}) == "from-settings"
    assert (
        resolve_version_check_package(settings_manager=FakeSettings(), env={"PI_VERSION_CHECK_PACKAGE": "from-env"})
        == "from-env"
    )
    assert (
        resolve_version_check_package(
            "from-arg", settings_manager=FakeSettings(), env={"PI_VERSION_CHECK_PACKAGE": "from-env"}
        )
        == "from-arg"
    )


def test_user_agent_shape():
    agent = get_pi_user_agent("1.2.3")
    assert agent.startswith("pi/1.2.3 (")
    assert "python/" in agent


def test_format_version_check_error_includes_errno():
    error = RuntimeError("fetch failed")
    error.__cause__ = OSError(111, "Connection refused")
    assert "111" in format_version_check_error(error)


def _pypi_client(payload: Any, status_code: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "pypi.org/pypi/" in str(request.url)
        return httpx.Response(status_code, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_get_latest_release_reads_the_pypi_version():
    async def scenario() -> None:
        async with _pypi_client({"info": {"version": "9.9.9"}}) as client:
            release = await get_latest_pi_release("1.0.0", client=client, env={})
        assert release is not None
        assert release.version == "9.9.9"
        assert release.url == "https://pypi.org/project/pp-coding-agent/9.9.9/"

    _run(scenario())


def test_a_yanked_release_is_not_offered():
    """Installers skip a yanked release, so offering it is a dead end."""

    async def scenario() -> None:
        async with _pypi_client({"info": {"version": "9.9.9", "yanked": True}}) as client:
            assert await get_latest_pi_release("1.0.0", client=client, env={}) is None

    _run(scenario())


def test_get_latest_release_uses_the_configured_package():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"info": {"version": "1.0.0"}})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await get_latest_pi_release("1.0.0", client=client, env={"PI_VERSION_CHECK_PACKAGE": "my-dist"})

    _run(scenario())
    assert seen[0] == "https://pypi.org/pypi/my-dist/json"


def test_offline_skips_the_version_check():
    async def scenario() -> None:
        result = await get_latest_pi_release("1.0.0", env={"PI_OFFLINE": "1"})
        assert result is None

    _run(scenario())


def test_check_for_new_version_only_reports_newer():
    async def scenario() -> None:
        async with _pypi_client({"info": {"version": "2.0.0"}}) as client:
            newer = await check_for_new_pi_version("1.0.0", client=client, env={})
        assert newer is not None and newer.version == "2.0.0"

        async with _pypi_client({"info": {"version": "0.9.0"}}) as client:
            older = await check_for_new_pi_version("1.0.0", client=client, env={})
        assert older is None

    _run(scenario())


def test_check_for_new_version_swallows_errors():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await check_for_new_pi_version("1.0.0", client=client, env={}) is None

    _run(scenario())


def test_check_respects_skip_env():
    async def scenario() -> None:
        assert await check_for_new_pi_version("1.0.0", env={"PI_SKIP_VERSION_CHECK": "1"}) is None

    _run(scenario())


def test_non_ok_response_yields_no_release():
    async def scenario() -> None:
        async with _pypi_client({}, status_code=404) as client:
            assert await get_latest_pi_release("1.0.0", client=client, env={}) is None

    _run(scenario())


def test_missing_version_yields_no_release():
    async def scenario() -> None:
        async with _pypi_client({"info": {}}) as client:
            assert await get_latest_pi_release("1.0.0", client=client, env={}) is None

    _run(scenario())

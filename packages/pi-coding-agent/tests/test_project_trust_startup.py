"""Tests for project-trust resolution and startup UI gating.

Project trust decides whether a folder's ``.pi`` settings, extensions and
packages may be loaded and executed, so these tests assert against real
`SettingsManager` loads rather than only the decision helpers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pi_coding_agent.cli.args import parse_args
from pi_coding_agent.cli.startup_ui import (
    OFFICIAL_APP_NAME,
    OFFICIAL_CONFIG_DIR_NAME,
    OFFICIAL_PACKAGE_NAME,
    is_official_distribution,
    load_themes,
    should_run_first_time_setup,
)
from pi_coding_agent.core.config import ENV_AGENT_DIR
from pi_coding_agent.core.project_trust import (
    ExtensionTrustDecision,
    create_project_trust_context,
    format_project_trust_prompt,
    resolve_project_trusted,
)
from pi_coding_agent.core.settings_manager import (
    SettingsManager,
    SettingsManagerCreateOptions,
)
from pi_coding_agent.core.trust_manager import ProjectTrustStore


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project folder with trust-requiring resources and a project setting."""
    config = tmp_path / "project" / ".pi"
    (config / "extensions").mkdir(parents=True)
    (config / "extensions" / "ext.js").write_text("// extension")
    (config / "settings.json").write_text(json.dumps({"defaultModel": "from-project"}))
    return tmp_path / "project"


@pytest.fixture
def agent_dir(tmp_path: Path) -> str:
    path = tmp_path / "agent"
    path.mkdir()
    return str(path)


class _RecordingUI:
    def __init__(self, answer: str | None = None) -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[str]]] = []

    async def select(self, title: str, options: Sequence[str]) -> str | None:
        self.calls.append((title, list(options)))
        return self.answer


# ---------------------------------------------------------------------------
# what trust actually controls
# ---------------------------------------------------------------------------


def test_untrusted_project_settings_are_not_loaded(project: Path, agent_dir: str) -> None:
    trusted = SettingsManager.create(str(project), agent_dir, SettingsManagerCreateOptions(project_trusted=True))
    untrusted = SettingsManager.create(str(project), agent_dir, SettingsManagerCreateOptions(project_trusted=False))
    assert trusted.get_default_model() == "from-project"
    assert untrusted.get_default_model() is None


# ---------------------------------------------------------------------------
# resolve_project_trusted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_override_short_circuits_everything(project: Path, agent_dir: str) -> None:
    ui = _RecordingUI("Trust")
    context = create_project_trust_context(cwd=str(project), mode="interactive", has_ui=True, ui=ui)
    store = ProjectTrustStore(agent_dir)

    for override in (True, False):
        result = await resolve_project_trusted(
            cwd=str(project),
            trust_store=store,
            project_trust_context=context,
            trust_override=override,
        )
        assert result is override
    assert ui.calls == []


@pytest.mark.asyncio
async def test_project_without_trust_requiring_resources_is_trusted(tmp_path: Path, agent_dir: str) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    context = create_project_trust_context(cwd=str(plain), mode="print", has_ui=False)
    result = await resolve_project_trusted(
        cwd=str(plain), trust_store=ProjectTrustStore(agent_dir), project_trust_context=context
    )
    assert result is True


@pytest.mark.asyncio
async def test_remembered_decision_is_reused(project: Path, agent_dir: str) -> None:
    store = ProjectTrustStore(agent_dir)
    store.set(str(project), True)
    ui = _RecordingUI("Trust")
    context = create_project_trust_context(cwd=str(project), mode="interactive", has_ui=True, ui=ui)

    assert await resolve_project_trusted(cwd=str(project), trust_store=store, project_trust_context=context) is True
    assert ui.calls == [], "a remembered decision must not re-prompt"


@pytest.mark.asyncio
async def test_remembered_denial_is_reused(project: Path, agent_dir: str) -> None:
    store = ProjectTrustStore(agent_dir)
    store.set(str(project), False)
    context = create_project_trust_context(cwd=str(project), mode="print", has_ui=False)
    assert await resolve_project_trusted(cwd=str(project), trust_store=store, project_trust_context=context) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("setting", "expected"), [("always", True), ("never", False)])
async def test_default_project_trust_setting_decides_without_prompting(
    project: Path, agent_dir: str, setting: str, expected: bool
) -> None:
    ui = _RecordingUI("Trust")
    context = create_project_trust_context(cwd=str(project), mode="interactive", has_ui=True, ui=ui)
    result = await resolve_project_trusted(
        cwd=str(project),
        trust_store=ProjectTrustStore(agent_dir),
        project_trust_context=context,
        default_project_trust=setting,
    )
    assert result is expected
    assert ui.calls == []


@pytest.mark.asyncio
async def test_no_ui_denies_rather_than_prompting(project: Path, agent_dir: str) -> None:
    """Without a way to ask, the safe answer is "do not trust"."""
    context = create_project_trust_context(cwd=str(project), mode="print", has_ui=True)
    result = await resolve_project_trusted(
        cwd=str(project),
        trust_store=ProjectTrustStore(agent_dir),
        project_trust_context=context,
        default_project_trust="ask",
    )
    assert result is False


@pytest.mark.asyncio
async def test_prompt_answer_is_honoured_and_remembered(project: Path, agent_dir: str) -> None:
    store = ProjectTrustStore(agent_dir)
    ui = _RecordingUI("Trust")
    context = create_project_trust_context(cwd=str(project), mode="interactive", has_ui=True, ui=ui)

    result = await resolve_project_trusted(
        cwd=str(project),
        trust_store=store,
        project_trust_context=context,
        default_project_trust="ask",
    )
    assert result is True
    assert len(ui.calls) == 1
    assert str(project) in ui.calls[0][0]
    assert store.get(str(project)) is True


@pytest.mark.asyncio
async def test_cancelled_prompt_denies(project: Path, agent_dir: str) -> None:
    store = ProjectTrustStore(agent_dir)
    ui = _RecordingUI(None)
    context = create_project_trust_context(cwd=str(project), mode="interactive", has_ui=True, ui=ui)

    result = await resolve_project_trusted(
        cwd=str(project),
        trust_store=store,
        project_trust_context=context,
        default_project_trust="ask",
    )
    assert result is False
    assert store.get(str(project)) is None, "cancelling must not persist a decision"


@pytest.mark.asyncio
async def test_extension_decision_wins_over_the_prompt(project: Path, agent_dir: str) -> None:
    store = ProjectTrustStore(agent_dir)
    ui = _RecordingUI("Trust")
    context = create_project_trust_context(cwd=str(project), mode="interactive", has_ui=True, ui=ui)

    async def decider(cwd: str) -> ExtensionTrustDecision:
        return ExtensionTrustDecision(trusted=False, remember=True)

    result = await resolve_project_trusted(
        cwd=str(project),
        trust_store=store,
        project_trust_context=context,
        trust_decider=decider,
    )
    assert result is False
    assert ui.calls == []
    assert store.get(str(project)) is False


@pytest.mark.asyncio
async def test_extension_decision_without_remember_is_not_persisted(project: Path, agent_dir: str) -> None:
    store = ProjectTrustStore(agent_dir)
    context = create_project_trust_context(cwd=str(project), mode="print", has_ui=False)

    async def decider(cwd: str) -> ExtensionTrustDecision:
        return ExtensionTrustDecision(trusted=True, remember=False)

    assert (
        await resolve_project_trusted(
            cwd=str(project),
            trust_store=store,
            project_trust_context=context,
            trust_decider=decider,
        )
        is True
    )
    assert store.get(str(project)) is None


@pytest.mark.asyncio
async def test_extension_abstention_falls_through(project: Path, agent_dir: str) -> None:
    store = ProjectTrustStore(agent_dir)
    ui = _RecordingUI("Trust")
    context = create_project_trust_context(cwd=str(project), mode="interactive", has_ui=True, ui=ui)

    async def decider(cwd: str) -> None:
        return None

    result = await resolve_project_trusted(
        cwd=str(project),
        trust_store=store,
        project_trust_context=context,
        trust_decider=decider,
        default_project_trust="ask",
    )
    assert result is True
    assert len(ui.calls) == 1


def test_prompt_text_names_the_folder_and_the_risk() -> None:
    prompt = format_project_trust_prompt("/some/project")
    assert "/some/project" in prompt
    assert "execute project extensions" in prompt


# ---------------------------------------------------------------------------
# trust context construction
# ---------------------------------------------------------------------------


def test_only_interactive_mode_gets_a_ui() -> None:
    ui = _RecordingUI()
    for mode in ("print", "json", "rpc"):
        context = create_project_trust_context(cwd="/x", mode=mode, has_ui=True, ui=ui)
        assert context.has_ui is False
        assert context.ui is None
        assert context.mode == mode

    context = create_project_trust_context(cwd="/x", mode="interactive", has_ui=True, ui=ui)
    assert context.has_ui is True
    assert context.ui is ui
    assert context.mode == "tui"


def test_interactive_without_a_terminal_has_no_ui() -> None:
    context = create_project_trust_context(cwd="/x", mode="interactive", has_ui=False)
    assert context.has_ui is False


# ---------------------------------------------------------------------------
# startup UI gating
# ---------------------------------------------------------------------------


def test_official_distribution_detection() -> None:
    assert is_official_distribution(
        package_name=OFFICIAL_PACKAGE_NAME,
        app_name=OFFICIAL_APP_NAME,
        config_dir_name=OFFICIAL_CONFIG_DIR_NAME,
    )
    assert not is_official_distribution(
        package_name="forked", app_name=OFFICIAL_APP_NAME, config_dir_name=OFFICIAL_CONFIG_DIR_NAME
    )
    assert not is_official_distribution(
        package_name=OFFICIAL_PACKAGE_NAME, app_name="mypi", config_dir_name=OFFICIAL_CONFIG_DIR_NAME
    )
    assert not is_official_distribution(
        package_name=OFFICIAL_PACKAGE_NAME, app_name=OFFICIAL_APP_NAME, config_dir_name=".mypi"
    )


def test_first_time_setup_requires_experimental_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_EXPERIMENTAL", raising=False)
    monkeypatch.delenv(ENV_AGENT_DIR, raising=False)
    assert should_run_first_time_setup(str(tmp_path / "settings.json")) is False


def test_first_time_setup_skipped_when_settings_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    monkeypatch.delenv(ENV_AGENT_DIR, raising=False)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    assert should_run_first_time_setup(str(settings)) is False


def test_first_time_setup_skipped_with_custom_agent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    monkeypatch.setenv(ENV_AGENT_DIR, str(tmp_path))
    assert should_run_first_time_setup(str(tmp_path / "settings.json")) is False


def test_first_time_setup_runs_on_a_fresh_experimental_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    monkeypatch.delenv(ENV_AGENT_DIR, raising=False)
    assert should_run_first_time_setup(str(tmp_path / "settings.json")) is True


# ---------------------------------------------------------------------------
# startup theme loading
# ---------------------------------------------------------------------------


class _Resource:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled


def test_load_themes_skips_disabled_resources() -> None:
    assert load_themes([_Resource("/nope.json", enabled=False)]) == []


def test_load_themes_survives_a_broken_theme() -> None:
    """A malformed theme must not stop a trust prompt from appearing."""
    assert load_themes([_Resource("/definitely/missing/theme.json")]) == []


def test_load_themes_deduplicates_by_name(tmp_path: Path) -> None:
    theme = {"name": "dup", "colors": {}}
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(theme))
    second.write_text(json.dumps(theme))
    loaded = load_themes([_Resource(str(first)), _Resource(str(second))])
    assert len(loaded) <= 1


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),
        (["--approve"], True),
        (["-a"], True),
        (["--no-approve"], False),
        (["-na"], False),
    ],
)
def test_trust_override_flags(argv: list[str], expected: bool | None) -> None:
    assert parse_args(argv).project_trust_override is expected

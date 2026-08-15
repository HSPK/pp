"""Tests for the interactive-mode dependency layer.

Covers `core/app_keybindings.py`, `core/telemetry.py`, `core/trust_manager.py`,
`core/http_dispatcher.py`, `core/footer_data_provider.py`,
`utils/changelog.py`, `utils/clipboard.py` and `utils/open_browser.py`.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest
from pi_tui.keybindings import TUI_KEYBINDINGS

from pi_coding_agent.core.app_keybindings import (
    APP_KEYBINDINGS,
    KEYBINDING_NAME_MIGRATIONS,
    KEYBINDINGS,
    KeybindingsManager,
    migrate_keybindings_config,
)
from pi_coding_agent.core.footer_data_provider import (
    FooterDataProvider,
    is_windows_mounted_repo_path,
    is_wsl_environment,
    should_poll_git_head,
)
from pi_coding_agent.core.http_dispatcher import (
    DEFAULT_HTTP_IDLE_TIMEOUT_MS,
    apply_http_proxy_settings,
    configure_http_dispatcher,
    format_http_idle_timeout_ms,
    parse_http_idle_timeout_ms,
)
from pi_coding_agent.core.telemetry import is_install_telemetry_enabled
from pi_coding_agent.core.trust_manager import (
    ProjectTrustStore,
    ProjectTrustUpdate,
    get_project_trust_options,
    get_project_trust_parent_path,
    has_trust_requiring_project_resources,
)
from pi_coding_agent.utils import clipboard as clipboard_module
from pi_coding_agent.utils import open_browser as open_browser_module
from pi_coding_agent.utils.changelog import (
    ChangelogEntry,
    compare_versions,
    get_new_entries,
    normalize_changelog_links,
    parse_changelog,
)

# --------------------------------------------------------------------------
# app_keybindings
# --------------------------------------------------------------------------


def test_keybindings_merges_tui_and_app_definitions():
    assert KEYBINDINGS == {**TUI_KEYBINDINGS, **APP_KEYBINDINGS}
    assert len(KEYBINDINGS) == len(TUI_KEYBINDINGS) + len(APP_KEYBINDINGS)


def test_every_app_keybinding_is_namespaced_and_described():
    for name, definition in APP_KEYBINDINGS.items():
        assert name.startswith("app."), name
        assert definition.description, name


def test_default_manager_resolves_app_defaults():
    manager = KeybindingsManager()
    assert manager.get_keys("app.interrupt") == ["escape"]
    assert manager.get_keys("app.model.select") == ["ctrl+l"]


def test_user_bindings_override_defaults():
    manager = KeybindingsManager({"app.interrupt": ["ctrl+q", "ctrl+c"]})
    assert manager.get_keys("app.interrupt") == ["ctrl+q", "ctrl+c"]


def test_migrate_renames_legacy_names():
    config, migrated = migrate_keybindings_config({"interrupt": "ctrl+q", "cursorUp": "up"})
    assert migrated is True
    assert config == {"tui.editor.cursorUp": "up", "app.interrupt": "ctrl+q"}


def test_migrate_leaves_current_names_untouched():
    config, migrated = migrate_keybindings_config({"app.interrupt": "ctrl+q"})
    assert migrated is False
    assert config == {"app.interrupt": "ctrl+q"}


def test_migrate_prefers_existing_current_name_on_collision():
    legacy_name = next(iter(KEYBINDING_NAME_MIGRATIONS))
    current_name = KEYBINDING_NAME_MIGRATIONS[legacy_name]
    config, migrated = migrate_keybindings_config({legacy_name: "ctrl+a", current_name: "ctrl+b"})
    assert config[current_name] == "ctrl+b"
    assert legacy_name not in config
    assert migrated is True


def test_migrate_keeps_unknown_names():
    # TS keeps unknown keys; `toKeybindingsConfig` later drops non-string values.
    config, migrated = migrate_keybindings_config({"totallyUnknown": "ctrl+z"})
    assert config == {"totallyUnknown": "ctrl+z"}
    assert migrated is False


def test_migrate_orders_known_bindings_by_definition_order():
    config, _ = migrate_keybindings_config({"app.model.select": "ctrl+l", "app.interrupt": "escape", "zzz": "x"})
    known = [k for k in config if k in KEYBINDINGS]
    assert known == [k for k in KEYBINDINGS if k in config]
    assert list(config)[-1] == "zzz"


def test_migrate_keeps_null_unbind_values():
    config, _ = migrate_keybindings_config({"app.interrupt": None})
    assert config == {"app.interrupt": None}


def test_manager_create_reads_user_config(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "keybindings.json").write_text(json.dumps({"interrupt": "ctrl+q"}), encoding="utf-8")

    manager = KeybindingsManager.create(str(agent_dir))
    assert manager.get_keys("app.interrupt") == ["ctrl+q"]
    # Migration happens in memory only; the file on disk is left alone.
    assert json.loads((agent_dir / "keybindings.json").read_text(encoding="utf-8")) == {"interrupt": "ctrl+q"}


def test_manager_create_ignores_non_string_bindings(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "keybindings.json").write_text(
        json.dumps({"app.interrupt": 42, "app.model.select": ["ctrl+m", 7]}), encoding="utf-8"
    )
    manager = KeybindingsManager.create(str(agent_dir))
    assert manager.get_keys("app.interrupt") == ["escape"]
    assert manager.get_keys("app.model.select") == ["ctrl+l"]


def test_manager_create_tolerates_missing_and_malformed_config(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    assert KeybindingsManager.create(str(agent_dir)).get_keys("app.interrupt") == ["escape"]

    (agent_dir / "keybindings.json").write_text("{not json", encoding="utf-8")
    assert KeybindingsManager.create(str(agent_dir)).get_keys("app.interrupt") == ["escape"]

    (agent_dir / "keybindings.json").write_text("[1,2,3]", encoding="utf-8")
    assert KeybindingsManager.create(str(agent_dir)).get_keys("app.interrupt") == ["escape"]


def test_manager_reload_picks_up_config_changes(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    manager = KeybindingsManager.create(str(agent_dir))
    assert manager.get_keys("app.interrupt") == ["escape"]

    (agent_dir / "keybindings.json").write_text(json.dumps({"app.interrupt": "ctrl+q"}), encoding="utf-8")
    manager.reload()
    assert manager.get_keys("app.interrupt") == ["ctrl+q"]


def test_effective_config_lists_every_binding():
    config = KeybindingsManager().get_effective_config()
    assert set(config) == set(KEYBINDINGS)
    # A single key collapses to a bare string; multiple keys stay a list.
    assert config["app.interrupt"] == "escape"
    assert config["tui.editor.cursorLeft"] == ["left", "ctrl+b"]


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------


class _FakeSettingsManager:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled

    def get_enable_install_telemetry(self) -> bool:
        return self._enabled


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("YES", True),
        ("0", False),
        ("", False),
        ("no", False),
    ],
)
def test_telemetry_env_flag_parsing(env: str, expected: bool):
    assert is_install_telemetry_enabled(_FakeSettingsManager(True), env) is expected
    assert is_install_telemetry_enabled(_FakeSettingsManager(False), env) is expected


def test_telemetry_falls_back_to_setting_when_env_absent():
    assert is_install_telemetry_enabled(_FakeSettingsManager(True), None) is True
    assert is_install_telemetry_enabled(_FakeSettingsManager(False), None) is False


def test_telemetry_reads_process_env_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PI_TELEMETRY", "0")
    assert is_install_telemetry_enabled(_FakeSettingsManager(True)) is False
    monkeypatch.delenv("PI_TELEMETRY")
    assert is_install_telemetry_enabled(_FakeSettingsManager(True)) is True


# --------------------------------------------------------------------------
# trust_manager
# --------------------------------------------------------------------------


def test_trust_store_roundtrip(tmp_path: Path):
    store = ProjectTrustStore(str(tmp_path / "agent"))
    project = tmp_path / "project"
    project.mkdir()

    assert store.get(str(project)) is None
    store.set(str(project), True)
    assert store.get(str(project)) is True
    store.set(str(project), False)
    assert store.get(str(project)) is False
    store.set(str(project), None)
    assert store.get(str(project)) is None


def test_trust_decision_is_inherited_from_nearest_ancestor(tmp_path: Path):
    store = ProjectTrustStore(str(tmp_path / "agent"))
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)

    store.set(str(parent), True)
    entry = store.get_entry(str(child))
    assert entry is not None
    assert entry.decision is True
    assert entry.path == os.path.realpath(str(parent))

    store.set(str(child), False)
    assert store.get(str(child)) is False
    assert store.get(str(parent)) is True


def test_trust_file_is_sorted_json_with_trailing_newline(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    store = ProjectTrustStore(str(agent_dir))
    for name in ("zeta", "alpha", "mid"):
        (tmp_path / name).mkdir()
    store.set_many([ProjectTrustUpdate(path=str(tmp_path / name), decision=True) for name in ("zeta", "alpha", "mid")])

    raw = (agent_dir / "trust.json").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    keys = list(json.loads(raw).keys())
    assert keys == sorted(keys)


def test_trust_store_rejects_malformed_files(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    trust_path = agent_dir / "trust.json"
    store = ProjectTrustStore(str(agent_dir))

    trust_path.write_text("{oops", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to read trust store"):
        store.get(str(tmp_path))

    trust_path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected an object"):
        store.get(str(tmp_path))

    trust_path.write_text('{"/x": "yes"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be true, false, or null"):
        store.get(str(tmp_path))


def test_trust_store_releases_lock_after_failure(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "trust.json").write_text("{oops", encoding="utf-8")
    store = ProjectTrustStore(str(agent_dir))

    with pytest.raises(RuntimeError):
        store.get(str(tmp_path))
    assert not (agent_dir / "trust.json.lock").exists()


def test_trust_options_include_parent_and_session_only(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()

    options = get_project_trust_options(str(project))
    assert [o.label for o in options] == [
        "Trust",
        f"Trust parent folder ({get_project_trust_parent_path(str(project))})",
        "Do not trust",
    ]

    options = get_project_trust_options(str(project), include_session_only=True)
    assert [o.label for o in options] == [
        "Trust",
        f"Trust parent folder ({get_project_trust_parent_path(str(project))})",
        "Trust (this session only)",
        "Do not trust",
        "Do not trust (this session only)",
    ]
    assert options[2].updates == []
    assert options[-1].updates == []


def test_trust_parent_option_clears_the_child_entry(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    parent_option = get_project_trust_options(str(project))[1]
    decisions = {u.path: u.decision for u in parent_option.updates}
    assert decisions[os.path.realpath(str(project))] is None
    assert decisions[get_project_trust_parent_path(str(project))] is True


def test_trust_parent_path_is_none_at_filesystem_root():
    assert get_project_trust_parent_path("/") is None


@pytest.mark.parametrize(
    "resource",
    ["settings.json", "extensions", "skills", "prompts", "themes", "SYSTEM.md", "APPEND_SYSTEM.md"],
)
def test_project_config_resources_require_trust(tmp_path: Path, resource: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    project = tmp_path / "project"
    (project / ".pi").mkdir(parents=True)
    assert has_trust_requiring_project_resources(str(project)) is False

    target = project / ".pi" / resource
    if "." in resource:
        target.write_text("x", encoding="utf-8")
    else:
        target.mkdir()
    assert has_trust_requiring_project_resources(str(project)) is True


def test_agents_skills_in_ancestor_requires_trust(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "parent" / "project"
    project.mkdir(parents=True)
    assert has_trust_requiring_project_resources(str(project)) is False

    (tmp_path / "parent" / ".agents" / "skills").mkdir(parents=True)
    assert has_trust_requiring_project_resources(str(project)) is True


def test_user_global_agents_skills_never_requires_trust(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    (home / ".agents" / "skills").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    assert has_trust_requiring_project_resources(str(home)) is False


# --------------------------------------------------------------------------
# http_dispatcher
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("disabled", 0),
        ("DISABLED", 0),
        ("  disabled  ", 0),
        ("", None),
        ("   ", None),
        ("30000", 30_000),
        ("30000.7", 30_000),
        ("abc", None),
        (30_000, 30_000),
        (30_000.9, 30_000),
        (0, 0),
        (-1, None),
        (True, None),
        (None, None),
        (float("inf"), None),
        (float("nan"), None),
    ],
)
def test_parse_http_idle_timeout(value: object, expected: int | None):
    assert parse_http_idle_timeout_ms(value) == expected


@pytest.mark.parametrize(
    ("timeout_ms", "label"),
    [
        (30_000, "30 sec"),
        (60_000, "1 min"),
        (120_000, "2 min"),
        (300_000, "5 min"),
        (0, "disabled"),
        (45_000, "45 sec"),
    ],
)
def test_format_http_idle_timeout(timeout_ms: int, label: str):
    assert format_http_idle_timeout_ms(timeout_ms) == label


def test_configure_http_dispatcher_sets_global_idle_timeout():
    from pi_ai.utils.http import build_timeout, get_idle_timeout_ms, set_idle_timeout_ms

    previous = get_idle_timeout_ms()
    try:
        configure_http_dispatcher(DEFAULT_HTTP_IDLE_TIMEOUT_MS)
        assert get_idle_timeout_ms() == DEFAULT_HTTP_IDLE_TIMEOUT_MS
        assert build_timeout(600_000).read == 300.0

        configure_http_dispatcher(0)
        assert build_timeout(600_000).read is None

        with pytest.raises(ValueError, match="Invalid HTTP idle timeout"):
            configure_http_dispatcher(-5)
    finally:
        set_idle_timeout_ms(previous)


def test_apply_http_proxy_settings_does_not_clobber_existing():
    env: dict[str, str] = {}
    apply_http_proxy_settings("  http://proxy:8080  ", env)
    assert env == {"HTTP_PROXY": "http://proxy:8080", "HTTPS_PROXY": "http://proxy:8080"}

    apply_http_proxy_settings("http://other:9090", env)
    assert env["HTTP_PROXY"] == "http://proxy:8080"


def test_apply_http_proxy_settings_ignores_blank():
    env: dict[str, str] = {}
    apply_http_proxy_settings("   ", env)
    apply_http_proxy_settings(None, env)
    assert env == {}


def test_apply_http_proxy_settings_defaults_to_process_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HTTP_PROXY", "http://existing:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://existing:1")
    apply_http_proxy_settings("http://other:2")
    assert os.environ["HTTP_PROXY"] == "http://existing:1"


# --------------------------------------------------------------------------
# changelog
# --------------------------------------------------------------------------


def test_parse_changelog_collects_versioned_sections(tmp_path: Path):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n- ignored\n\n"
        "## [1.2.3] - 2024-01-01\n### Added\n- thing\n\n"
        "## [1.2.2]\n- older\n",
        encoding="utf-8",
    )
    entries = parse_changelog(str(path))
    assert [(e.major, e.minor, e.patch) for e in entries] == [(1, 2, 3), (1, 2, 2)]
    assert entries[0].content.startswith("## [1.2.3]")
    assert "- thing" in entries[0].content
    assert "ignored" not in entries[0].content


def test_parse_changelog_handles_missing_file(tmp_path: Path):
    assert parse_changelog(str(tmp_path / "nope.md")) == []


def test_parse_changelog_ignores_header_only_section(tmp_path: Path):
    path = tmp_path / "CHANGELOG.md"
    path.write_text("## [1.0.0]\n", encoding="utf-8")
    # A header with no following lines still counts: currentLines == [line].
    assert [(e.major, e.minor, e.patch) for e in parse_changelog(str(path))] == [(1, 0, 0)]


def test_compare_versions_orders_by_component():
    a = ChangelogEntry(1, 2, 3, "")
    assert compare_versions(a, ChangelogEntry(1, 2, 3, "")) == 0
    assert compare_versions(a, ChangelogEntry(0, 9, 9, "")) > 0
    assert compare_versions(a, ChangelogEntry(1, 3, 0, "")) < 0
    assert compare_versions(a, ChangelogEntry(1, 2, 4, "")) < 0


def test_get_new_entries_filters_older_versions():
    entries = [ChangelogEntry(1, 2, 3, ""), ChangelogEntry(1, 2, 2, ""), ChangelogEntry(2, 0, 0, "")]
    assert [(e.major, e.minor, e.patch) for e in get_new_entries(entries, "1.2.2")] == [(1, 2, 3), (2, 0, 0)]
    assert get_new_entries(entries, "2.0.0") == []


def test_get_new_entries_tolerates_partial_and_junk_versions():
    entries = [ChangelogEntry(0, 0, 1, "")]
    assert len(get_new_entries(entries, "0")) == 1
    assert len(get_new_entries(entries, "x.y.z")) == 1


def test_normalize_changelog_links_rewrites_relative_targets():
    out = normalize_changelog_links("[a](./docs/guide.md)", "1.2.3")
    assert out == "[a](https://github.com/earendil-works/pi/blob/v1.2.3/packages/coding-agent/docs/guide.md)"


def test_normalize_changelog_links_uses_tree_route_for_directories():
    out = normalize_changelog_links("[a](docs/)", "v1.2.3")
    assert out == "[a](https://github.com/earendil-works/pi/tree/v1.2.3/packages/coding-agent/docs/)"


def test_normalize_changelog_links_pins_floating_refs_and_legacy_repo():
    out = normalize_changelog_links("[a](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/x.ts)", "1.2.3")
    assert out == "[a](https://github.com/earendil-works/pi/blob/v1.2.3/packages/ai/src/x.ts)"


@pytest.mark.parametrize("target", ["#anchor", "//cdn.example/x", "https://example.com/a", "mailto:a@b.c"])
def test_normalize_changelog_links_leaves_non_local_targets(target: str):
    assert normalize_changelog_links(f"[a]({target})", "1.2.3") == f"[a]({target})"


def test_normalize_changelog_links_preserves_query_and_fragment():
    out = normalize_changelog_links("[a](docs/a.md?plain=1#L4)", "1.2.3")
    assert out.endswith("/packages/coding-agent/docs/a.md?plain=1#L4)")


def test_normalize_changelog_links_handles_images_and_titles():
    out = normalize_changelog_links('![img](docs/x.png "Title")', "1.2.3")
    assert out == '![img](https://github.com/earendil-works/pi/blob/v1.2.3/packages/coding-agent/docs/x.png "Title")'


def test_normalize_changelog_links_accepts_entry_objects():
    out = normalize_changelog_links("[a](docs/x.md)", ChangelogEntry(1, 2, 3, ""))
    assert "/v1.2.3/" in out


# --------------------------------------------------------------------------
# clipboard / open_browser
# --------------------------------------------------------------------------


class _FakeStdout:
    def __init__(self) -> None:
        self.written = ""
        self.flushed = 0

    def write(self, text: str) -> None:
        self.written += text

    def flush(self) -> None:
        self.flushed += 1


def test_emit_osc52_writes_escape_sequence():
    out = _FakeStdout()
    assert clipboard_module.emit_osc52("hi", stream=out) is True
    assert out.written == "\x1b]52;c;aGk=\x07"
    assert out.flushed == 1


def test_emit_osc52_refuses_oversized_payloads():
    out = _FakeStdout()
    huge = "a" * (clipboard_module.MAX_OSC52_ENCODED_LENGTH + 10)
    assert clipboard_module.emit_osc52(huge, stream=out) is False
    assert out.written == ""


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"WAYLAND_DISPLAY": "wayland-0"}, True),
        ({"XDG_SESSION_TYPE": "wayland"}, True),
        ({"XDG_SESSION_TYPE": "x11"}, False),
    ],
)
def test_is_wayland_session(env: dict[str, str], expected: bool):
    assert clipboard_module.is_wayland_session(env) is expected


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"SSH_CONNECTION": "1 2 3 4"}, True),
        ({"SSH_CLIENT": "1 2 3"}, True),
        ({"MOSH_CONNECTION": "x"}, True),
    ],
)
def test_is_remote_session(env: dict[str, str], expected: bool):
    assert clipboard_module.is_remote_session(env) is expected


def test_copy_to_clipboard_uses_platform_tool(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(clipboard_module, "_run_with_input", lambda argv, text: calls.append((argv, text)))
    monkeypatch.setattr(clipboard_module, "emit_osc52", lambda *a, **k: pytest.fail("should not fall back"))

    asyncio.run(clipboard_module.copy_to_clipboard("hello", {"DISPLAY": ":0"}))
    assert calls == [(["xclip", "-selection", "clipboard"], "hello")]


def test_copy_to_clipboard_falls_back_from_xclip_to_xsel(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run(argv: list[str], text: str) -> None:
        calls.append(argv)
        if argv[0] == "xclip":
            raise FileNotFoundError("xclip")

    monkeypatch.setattr(clipboard_module, "_run_with_input", fake_run)
    asyncio.run(clipboard_module.copy_to_clipboard("hello", {"DISPLAY": ":0"}))
    assert [c[0] for c in calls] == ["xclip", "xsel"]


def test_copy_to_clipboard_falls_back_to_osc52(monkeypatch: pytest.MonkeyPatch):
    emitted: list[str] = []
    monkeypatch.setattr(clipboard_module, "_run_with_input", lambda argv, text: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(clipboard_module, "emit_osc52", lambda text: emitted.append(text) or True)

    asyncio.run(clipboard_module.copy_to_clipboard("hello", {}))
    assert emitted == ["hello"]


def test_copy_to_clipboard_always_emits_osc52_when_remote(monkeypatch: pytest.MonkeyPatch):
    emitted: list[str] = []
    monkeypatch.setattr(clipboard_module, "_run_with_input", lambda argv, text: None)
    monkeypatch.setattr(clipboard_module, "emit_osc52", lambda text: emitted.append(text) or True)

    asyncio.run(clipboard_module.copy_to_clipboard("hello", {"DISPLAY": ":0", "SSH_CONNECTION": "x"}))
    assert emitted == ["hello"]


def test_copy_to_clipboard_raises_when_nothing_worked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(clipboard_module, "_run_with_input", lambda argv, text: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(clipboard_module, "emit_osc52", lambda text: False)

    with pytest.raises(RuntimeError, match="Failed to copy to clipboard"):
        asyncio.run(clipboard_module.copy_to_clipboard("hello", {}))


def test_copy_to_clipboard_prefers_termux(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(clipboard_module, "_run_with_input", lambda argv, text: calls.append(argv))
    asyncio.run(clipboard_module.copy_to_clipboard("x", {"TERMUX_VERSION": "1", "DISPLAY": ":0"}))
    assert calls == [["termux-clipboard-set"]]


def test_copy_to_clipboard_prefers_wl_copy_on_wayland(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(clipboard_module, "_run_with_input", lambda argv, text: calls.append(argv))
    asyncio.run(clipboard_module.copy_to_clipboard("x", {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}))
    assert calls == [["wl-copy"]]


def test_copy_to_clipboard_falls_back_to_x11_when_wl_copy_fails(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run(argv: list[str], text: str) -> None:
        calls.append(argv)
        if argv[0] == "wl-copy":
            raise FileNotFoundError("wl-copy")

    monkeypatch.setattr(clipboard_module, "_run_with_input", fake_run)
    asyncio.run(clipboard_module.copy_to_clipboard("x", {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}))
    assert [c[0] for c in calls] == ["wl-copy", "xclip"]


def test_read_clipboard_text_uses_wl_paste_on_wayland(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(clipboard_module.sys, "platform", "linux")
    monkeypatch.setattr(clipboard_module, "_read_command", lambda argv: "text" if argv[0] == "wl-paste" else None)
    result = asyncio.run(clipboard_module.read_clipboard_text({"WAYLAND_DISPLAY": "wayland-0"}))
    assert result == "text"


def test_read_clipboard_text_returns_none_without_tools(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(clipboard_module.sys, "platform", "linux")
    monkeypatch.setattr(clipboard_module, "_read_command", lambda argv: None)
    assert asyncio.run(clipboard_module.read_clipboard_text({"DISPLAY": ":0"})) is None


def test_read_clipboard_text_maps_empty_output_to_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(clipboard_module.sys, "platform", "linux")
    monkeypatch.setattr(clipboard_module, "_read_command", lambda argv: "")
    assert asyncio.run(clipboard_module.read_clipboard_text({"DISPLAY": ":0"})) is None


def test_read_command_returns_none_on_failure():
    assert clipboard_module._read_command(["false"]) is None
    assert clipboard_module._read_command(["definitely-not-a-real-binary-xyz"]) is None


def test_read_command_returns_stdout():
    assert clipboard_module._read_command(["printf", "hi"]) == "hi"


def test_open_browser_spawns_platform_handler(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> object:
        calls.append(cmd)
        assert kwargs["start_new_session"] is True
        return object()

    monkeypatch.setattr(open_browser_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(open_browser_module.sys, "platform", "linux")
    open_browser_module.open_browser("https://example.com")
    assert calls == [["xdg-open", "https://example.com"]]

    calls.clear()
    monkeypatch.setattr(open_browser_module.sys, "platform", "darwin")
    open_browser_module.open_browser("https://example.com")
    assert calls == [["open", "https://example.com"]]

    calls.clear()
    monkeypatch.setattr(open_browser_module.sys, "platform", "win32")
    open_browser_module.open_browser("https://example.com&x")
    assert calls == [["rundll32", "url.dll,FileProtocolHandler", "https://example.com&x"]]


def test_open_browser_swallows_launcher_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_popen(cmd: list[str], **kwargs: object) -> object:
        raise FileNotFoundError("xdg-open")

    monkeypatch.setattr(open_browser_module.subprocess, "Popen", fake_popen)
    open_browser_module.open_browser("https://example.com")


# --------------------------------------------------------------------------
# footer_data_provider
# --------------------------------------------------------------------------


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True, capture_output=True)


def test_footer_reports_none_outside_a_repo(tmp_path: Path):
    provider = FooterDataProvider(str(tmp_path))
    try:
        assert provider.git_paths is None
        assert provider.get_git_branch() is None
    finally:
        provider.dispose()


def test_footer_reads_branch_from_head(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = FooterDataProvider(str(repo))
    try:
        assert provider.get_git_branch() == "main"
    finally:
        provider.dispose()


def test_footer_reports_detached_head(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".git" / "HEAD").write_text("deadbeef" * 5 + "\n", encoding="utf-8")
    provider = FooterDataProvider(str(repo))
    try:
        assert provider.get_git_branch() == "detached"
    finally:
        provider.dispose()


def test_footer_shells_out_for_invalid_head_ref(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/.invalid\n", encoding="utf-8")

    provider = FooterDataProvider(str(repo), resolve_branch=lambda repo_dir: "from-git")
    try:
        assert provider.get_git_branch() == "from-git"
    finally:
        provider.dispose()

    provider = FooterDataProvider(str(repo), resolve_branch=lambda repo_dir: None)
    try:
        assert provider.get_git_branch() == "detached"
    finally:
        provider.dispose()


def test_footer_refresh_notifies_only_on_change(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = FooterDataProvider(str(repo))
    changes: list[str | None] = []
    provider.on_branch_change(lambda: changes.append(provider.get_git_branch()))
    try:
        assert provider.get_git_branch() == "main"
        assert provider.refresh_git_branch() is False
        assert changes == []

        (repo / ".git" / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")
        assert provider.refresh_git_branch() is True
        assert changes == ["feature"]
    finally:
        provider.dispose()


def test_footer_first_refresh_only_seeds_the_cache(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = FooterDataProvider(str(repo))
    changes: list[None] = []
    provider.on_branch_change(lambda: changes.append(None))
    try:
        assert provider.refresh_git_branch() is False
        assert changes == []
        assert provider.get_git_branch() == "main"
    finally:
        provider.dispose()


def test_footer_unsubscribe_stops_notifications(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = FooterDataProvider(str(repo))
    changes: list[None] = []
    unsubscribe = provider.on_branch_change(lambda: changes.append(None))
    try:
        provider.get_git_branch()
        unsubscribe()
        unsubscribe()  # idempotent
        (repo / ".git" / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")
        provider.refresh_git_branch()
        assert changes == []
    finally:
        provider.dispose()


def test_footer_set_cwd_reresolves_and_notifies(tmp_path: Path):
    repo_a = tmp_path / "a"
    repo_b = tmp_path / "b"
    _init_repo(repo_a)
    _init_repo(repo_b)
    subprocess.run(["git", "-C", str(repo_b), "symbolic-ref", "HEAD", "refs/heads/other"], check=True)

    provider = FooterDataProvider(str(repo_a))
    changes: list[None] = []
    provider.on_branch_change(lambda: changes.append(None))
    try:
        assert provider.get_git_branch() == "main"
        provider.set_cwd(str(repo_a))
        assert changes == []

        provider.set_cwd(str(repo_b))
        assert changes == [None]
        assert provider.get_git_branch() == "other"
    finally:
        provider.dispose()


def test_footer_extension_statuses_and_provider_count(tmp_path: Path):
    provider = FooterDataProvider(str(tmp_path))
    try:
        assert provider.get_extension_statuses() == {}
        provider.set_extension_status("ext", "busy")
        assert provider.get_extension_statuses() == {"ext": "busy"}
        provider.set_extension_status("ext", None)
        assert provider.get_extension_statuses() == {}

        provider.set_extension_status("a", "1")
        provider.clear_extension_statuses()
        assert provider.get_extension_statuses() == {}

        assert provider.get_available_provider_count() == 0
        provider.set_available_provider_count(3)
        assert provider.get_available_provider_count() == 3
    finally:
        provider.dispose()


def test_footer_dispose_stops_refreshes(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    provider = FooterDataProvider(str(repo))
    changes: list[None] = []
    provider.on_branch_change(lambda: changes.append(None))
    provider.get_git_branch()
    provider.dispose()

    (repo / ".git" / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")
    assert provider.refresh_git_branch() is False
    assert changes == []


def test_footer_watch_loop_publishes_branch_changes(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)

    async def scenario() -> list[str | None]:
        provider = FooterDataProvider(str(repo))
        seen: list[str | None] = []
        provider.on_branch_change(lambda: seen.append(provider.get_git_branch()))
        try:
            assert provider.get_git_branch() == "main"
            provider.WATCH_DEBOUNCE_MS = 10
            provider.start_watching(poll_interval_ms=5)
            provider.start_watching()  # second call is a no-op
            (repo / ".git" / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")
            for _ in range(200):
                await asyncio.sleep(0.01)
                if seen:
                    break
            return seen
        finally:
            provider.dispose()

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=10)) == ["feature"]


def test_footer_watch_loop_stops_on_dispose(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)

    async def scenario() -> None:
        provider = FooterDataProvider(str(repo))
        provider.start_watching(poll_interval_ms=5)
        task = provider._watch_task
        assert task is not None
        provider.dispose()
        # Waiting on the task itself, rather than sleeping and hoping, keeps
        # this independent of how loaded the machine is.
        await asyncio.wait([task], timeout=5)
        assert task.cancelled() or task.done()

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_footer_works_in_a_git_worktree(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "f.txt").write_text("x", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e",
    }
    subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True, capture_output=True, env=env)
    worktree = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-q", "-b", "wtbranch", str(worktree)],
        check=True,
        capture_output=True,
        env=env,
    )

    provider = FooterDataProvider(str(worktree))
    try:
        assert provider.get_git_branch() == "wtbranch"
    finally:
        provider.dispose()


@pytest.mark.parametrize(
    ("env", "expected"),
    [({}, False), ({"WSL_DISTRO_NAME": "Ubuntu"}, True), ({"WSL_INTEROP": "/run/x"}, True)],
)
def test_is_wsl_environment(env: dict[str, str], expected: bool):
    assert is_wsl_environment(env) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [("/mnt/c/foo", True), ("/mnt/c", True), ("/MNT/C/foo", True), ("/mnt/cc/foo", False), ("/home/x", False)],
)
def test_is_windows_mounted_repo_path(path: str, expected: bool):
    assert is_windows_mounted_repo_path(path) is expected


def test_should_poll_git_head_requires_both_conditions():
    assert should_poll_git_head("/mnt/c/repo", {"WSL_DISTRO_NAME": "Ubuntu"}) is True
    assert should_poll_git_head("/home/x/repo", {"WSL_DISTRO_NAME": "Ubuntu"}) is False
    assert should_poll_git_head("/mnt/c/repo", {}) is False

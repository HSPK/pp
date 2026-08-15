"""Python port of `packages/coding-agent/test/interactive-mode-status.test.ts`.

The TypeScript test calls `InteractiveMode.prototype.<private>` bound to a
hand-built `this`. The Python analogue is calling the unbound function with a
stand-in object, the same technique
`tests/test_interactive_mode_compaction.py` uses.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from pi_tui.autocomplete import AutocompleteItem
from pi_tui.component import Component, Container
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text

from pi_coding_agent.modes.interactive.components.oauth_selector import AuthSelectorProvider
from pi_coding_agent.modes.interactive.interactive_mode import (
    InteractiveMode,
    format_login_provider_completion_description,
    get_login_provider_completion_options,
    get_login_provider_search_text,
)
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi


@pytest.fixture(autouse=True, scope="module")
def _dark_theme() -> None:
    # `show_status` colours through the global theme instance.
    init_theme("dark")


class _Recorder:
    def __init__(self, result: Any = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._result = result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self._result


def _render_last_line(container: Container, width: int = 120) -> str:
    if not container.children:
        return ""
    return "\n".join(container.children[-1].render(width))


class _OtherComponent(Component):
    def render(self, _width: int) -> list[str]:
        return ["OTHER"]

    def invalidate(self) -> None:
        return None


class _StatusThis:
    """Stand-in `self` for `show_status`."""

    show_status = InteractiveMode.show_status

    def __init__(self) -> None:
        self.chat_container = Container()
        self.ui = type("_Ui", (), {"request_render": _Recorder()})()
        self.last_status_spacer: Spacer | None = None
        self.last_status_text: Text | None = None


def test_coalesces_immediately_sequential_status_messages() -> None:
    fake = _StatusThis()

    fake.show_status("STATUS_ONE")
    assert len(fake.chat_container.children) == 2
    assert "STATUS_ONE" in _render_last_line(fake.chat_container)

    fake.show_status("STATUS_TWO")
    # The second status updates the previous line instead of appending.
    assert len(fake.chat_container.children) == 2
    assert "STATUS_TWO" in _render_last_line(fake.chat_container)
    assert "STATUS_ONE" not in _render_last_line(fake.chat_container)


def test_appends_a_new_status_line_if_something_else_was_added_in_between() -> None:
    fake = _StatusThis()

    fake.show_status("STATUS_ONE")
    assert len(fake.chat_container.children) == 2

    fake.chat_container.add_child(_OtherComponent())
    assert len(fake.chat_container.children) == 3

    fake.show_status("STATUS_TWO")
    # Adds spacer + text.
    assert len(fake.chat_container.children) == 5
    assert "STATUS_TWO" in _render_last_line(fake.chat_container)


class _Expandable(Component):
    def __init__(self) -> None:
        self.set_expanded = _Recorder()

    def render(self, _width: int) -> list[str]:
        return [""]

    def invalidate(self) -> None:
        return None


class _ToolsExpandedThis:
    """Stand-in `self` for `set_tools_expanded`."""

    set_tools_expanded = InteractiveMode.set_tools_expanded
    _iter_components = InteractiveMode._iter_components

    def __init__(self) -> None:
        self.tool_output_expanded = False
        self.loaded_resources_container = Container()
        self.chat_container = Container()
        self.loaded_resources_child = _Expandable()
        self.chat_child = _Expandable()
        self.loaded_resources_container.add_child(self.loaded_resources_child)
        self.chat_container.add_child(self.chat_child)
        self.ui = type("_Ui", (), {"request_render": _Recorder()})()
        self.show_status = _Recorder()


def test_set_tools_expanded_applies_expansion_state_to_chat_entries() -> None:
    fake = _ToolsExpandedThis()

    fake.set_tools_expanded(True)

    assert fake.tool_output_expanded is True
    # The TypeScript test also asserts `builtInHeader.setExpanded(true)`. This
    # port's startup header is a plain `Text`, not an `ExpandableText`, because
    # the startup resource/diagnostic report it expands is not ported (see the
    # repository README); there is no header to expand here.
    assert fake.loaded_resources_child.set_expanded.calls == [((True,), {})]
    assert fake.chat_child.set_expanded.calls == [((True,), {})]
    assert fake.show_status.calls == [(("Tool output: expanded",), {})]


def test_set_tools_expanded_is_a_no_op_when_already_in_that_state() -> None:
    fake = _ToolsExpandedThis()

    fake.set_tools_expanded(False)

    assert fake.chat_child.set_expanded.calls == []
    assert fake.show_status.calls == []


def test_set_tools_expanded_does_not_descend_into_nested_containers() -> None:
    """TypeScript's `setToolsExpanded` iterates `container.children` only.

    An earlier revision of this port walked the tree recursively, so an
    expandable nested inside another container was toggled where TypeScript
    leaves it alone. Both ports build these containers flat, so nothing nested
    is a transcript entry.
    """
    fake = _ToolsExpandedThis()
    nested_parent = Container()
    nested_child = _Expandable()
    nested_parent.add_child(nested_child)
    fake.chat_container.add_child(nested_parent)

    fake.set_tools_expanded(True)

    assert fake.chat_child.set_expanded.calls == [((True,), {})]
    assert nested_child.set_expanded.calls == []


# The TypeScript file's `createExtensionUIContext setTheme`,
# `showExtensionCustom` and `createExtensionUIContext addAutocompleteProvider`
# blocks all drive the extension UI host (extension-supplied dialogs, custom
# components and autocomplete wrappers). That host is not ported -- see the
# "Not ported, by decision" list in the repository README -- so there is no
# `create_extension_ui_context`, no `show_extension_custom` and no
# `autocomplete_provider_wrappers` to assert against.
@pytest.mark.skip(reason="extension UI host (createExtensionUIContext/showExtensionCustom) is not ported")
def test_extension_ui_context_set_theme_persists_theme_changes() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason="extension UI host (showExtensionCustom overlay focus) is not ported")
def test_overlay_custom_ui_reclaims_input_after_non_overlay_custom_ui_closes() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason="extension UI host (createExtensionUIContext/setTheme) is not ported")
def test_extension_ui_context_does_not_persist_invalid_theme_names() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason="extension UI host (createExtensionUIContext/addAutocompleteProvider) is not ported")
def test_extension_ui_context_stores_wrapper_factories_and_rebuilds_autocomplete() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason="extension UI host (autocompleteProviderWrappers) is not ported")
def test_setup_autocomplete_provider_stacks_wrapper_factories_over_a_fresh_base_provider() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason="extension UI host (autocompleteProviderWrappers) is not ported")
def test_setup_autocomplete_provider_merges_trigger_characters_from_wrapper_factories() -> None:
    raise AssertionError("unreachable")


# `setupAutocompleteProvider`'s two TypeScript tests stack
# `autocompleteProviderWrappers` (registered through
# `uiContext.addAutocompleteProvider`) over the base provider and merge their
# `triggerCharacters`. Both the registration API and the wrapper list belong to
# the unported extension UI host, so this port's `_setup_autocomplete_provider`
# installs the base provider directly. What is portable is that it installs the
# *same* provider instance on both editors, which is asserted here.
class _SetupAutocompleteThis:
    _setup_autocomplete_provider = InteractiveMode._setup_autocomplete_provider

    def __init__(self) -> None:
        self.autocomplete_provider: Any = None
        self.default_editor = type("_Editor", (), {"set_autocomplete_provider": _Recorder()})()
        self.editor = type("_Editor", (), {"set_autocomplete_provider": _Recorder()})()
        self.base = object()
        self._create_base_autocomplete_provider = lambda: self.base


def test_setup_autocomplete_provider_installs_one_provider_on_both_editors() -> None:
    fake = _SetupAutocompleteThis()

    fake._setup_autocomplete_provider()

    assert fake.default_editor.set_autocomplete_provider.calls == [((fake.base,), {})]
    assert fake.editor.set_autocomplete_provider.calls == [((fake.base,), {})]


@dataclass
class _FakeModel:
    id: str
    provider: str
    name: str


class _FakeModelRuntime:
    def __init__(self, models: list[_FakeModel]) -> None:
        self._models = models

    def get_available_snapshot(self) -> list[_FakeModel]:
        return self._models


class _FakeResourceLoader:
    def get_skills(self) -> Any:
        return type("_Skills", (), {"skills": []})()


class _FakeSession:
    def __init__(self, models: list[_FakeModel]) -> None:
        self.scoped_models: list[Any] = []
        self.model_runtime = _FakeModelRuntime(models)
        self.prompt_templates: list[Any] = []
        self.resource_loader = _FakeResourceLoader()


class _FakeSettingsManager:
    def get_enable_skill_commands(self) -> bool:
        return False


class _BaseAutocompleteThis:
    """Stand-in `self` for `_create_base_autocomplete_provider`."""

    _create_base_autocomplete_provider = InteractiveMode._create_base_autocomplete_provider
    _model_argument_completions = InteractiveMode._model_argument_completions
    _login_argument_completions = InteractiveMode._login_argument_completions
    _prefix_autocomplete_description = InteractiveMode._prefix_autocomplete_description
    _get_autocomplete_source_tag = InteractiveMode._get_autocomplete_source_tag

    def __init__(
        self,
        models: list[_FakeModel] | None = None,
        login_options: list[AuthSelectorProvider] | None = None,
    ) -> None:
        self.session = _FakeSession(models or [])
        self.settings_manager = _FakeSettingsManager()
        self.skill_commands: dict[str, str] = {}
        self.session_manager = type("_SessionManager", (), {"get_cwd": lambda _self: "/tmp"})()
        self.fd_path: str | None = None
        self._login_options = login_options or []

    def get_login_provider_options(self, auth_type: str | None = None) -> list[AuthSelectorProvider]:
        return list(self._login_options)


async def _suggest(provider: Any, line: str) -> Any:
    return await provider.get_suggestions([line], 0, len(line), signal=asyncio.Event())


def test_matches_model_command_arguments_across_provider_model_order() -> None:
    fake = _BaseAutocompleteThis(
        models=[
            _FakeModel(id="gpt-5.2-codex", provider="github-copilot", name="GPT-5.2 Codex"),
            _FakeModel(id="gpt-5.5", provider="openai-codex", name="GPT-5.5"),
        ]
    )
    provider = fake._create_base_autocomplete_provider()

    suggestions = asyncio.run(_suggest(provider, "/model codexgpt"))

    assert [item.value for item in suggestions.items] == [
        "openai-codex/gpt-5.5",
        "github-copilot/gpt-5.2-codex",
    ]


def test_matches_login_command_arguments_by_provider_id_and_name() -> None:
    fake = _BaseAutocompleteThis(
        login_options=[
            AuthSelectorProvider(id="anthropic", name="Anthropic", auth_type="oauth"),
            AuthSelectorProvider(id="anthropic", name="Anthropic", auth_type="api_key"),
            AuthSelectorProvider(id="openai", name="OpenAI", auth_type="api_key"),
        ]
    )
    provider = fake._create_base_autocomplete_provider()

    suggestions = asyncio.run(_suggest(provider, "/login subscription anthrop"))

    assert suggestions.items == [
        AutocompleteItem(
            value="anthropic",
            label="anthropic",
            description="Anthropic · subscription/API key",
        )
    ]


def test_login_completion_options_merge_auth_types_and_sort_by_name() -> None:
    options = get_login_provider_completion_options(
        [
            AuthSelectorProvider(id="openai", name="OpenAI", auth_type="api_key"),
            AuthSelectorProvider(id="anthropic", name="Anthropic", auth_type="api_key"),
            AuthSelectorProvider(id="anthropic", name="Anthropic", auth_type="oauth"),
        ]
    )

    assert [(option.id, option.auth_types) for option in options] == [
        ("anthropic", ["oauth", "api_key"]),
        ("openai", ["api_key"]),
    ]
    assert get_login_provider_search_text(options[0]) == "anthropic Anthropic oauth subscription api_key API key"
    assert format_login_provider_completion_description(options[0]) == "Anthropic · subscription/API key"


def test_login_completion_description_omits_a_name_equal_to_the_id() -> None:
    options = get_login_provider_completion_options(
        [AuthSelectorProvider(id="local-proxy", name="local-proxy", auth_type="api_key")]
    )

    assert format_login_provider_completion_description(options[0]) == "API key"


# `showLoadedResources` has 21 TypeScript tests covering the compact/expanded
# startup listing and the extension-label disambiguation rules. That whole
# startup resource/diagnostic report is not ported (repository README, "Not
# ported, by decision"): this port keeps `loaded_resources_container` for
# layout, but nothing fills it, and none of the helpers those tests reach
# through the fake `this` -- `formatDisplayPath`, `formatExtensionDisplayPath`,
# `formatContextPath`, `getShortPath`, `isPackageSource`,
# `getCompactExtensionLabel(s)`, `getCompactPathLabel`,
# `getCompactPackageSourceLabel`, `getCompactDisplayPathSegments`,
# `getCompactNonPackageExtensionLabel`, `getScopeGroup`, `buildScopeGroups`,
# `formatScopeGroups`, `getStartupExpansionState` -- exists here either. They
# are listed one per TypeScript case so the port-depth measurement is not
# flattered by a single blanket skip.
_NO_LOADED_RESOURCES = "startup resource/diagnostic report (showLoadedResources) is not ported"


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_shows_a_compact_resource_listing_by_default() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_shows_full_resource_listing_when_expanded() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_shows_full_resource_listing_on_verbose_startup_even_when_tool_output_is_collapsed() -> (
    None
):
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_abbreviates_extensions_in_compact_listing() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_captures_mixed_extension_layouts_in_compact_output() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_adds_more_parent_folders_until_local_extension_labels_are_unique() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_strips_index_ts_from_local_extension_label_showing_parent_dir() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_strips_index_js_from_local_extension_label_showing_parent_dir() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_mixed_single_file_and_subdirectory_index_ts_extensions_strip_index_ts() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_multiple_index_ts_with_unique_parent_dirs_need_no_disambiguation() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_multiple_index_ts_with_same_parent_dir_name_disambiguated_with_grandparent() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_non_index_file_in_subdirectory_stays_as_filename() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_package_extensions_still_strip_index_ts_correctly() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_labels_npm_sibling_extensions_relative_to_the_declaring_package() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_labels_windows_npm_sibling_extensions_relative_to_the_declaring_package() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_captures_mixed_extension_layouts_in_expanded_output() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_shows_context_paths_relative_to_cwd_while_preserving_full_external_paths() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_shows_system_prompt_context_paths_before_project_context_files() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_shows_full_context_paths_when_expanded() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_does_not_show_verbose_listing_on_quiet_startup_during_reload() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=_NO_LOADED_RESOURCES)
def test_show_loaded_resources_still_shows_diagnostics_on_quiet_startup_when_requested() -> None:
    raise AssertionError("unreachable")


def test_status_text_is_dim_coloured() -> None:
    """`show_status` uses the theme's `dim` colour, as the TypeScript does."""
    fake = _StatusThis()

    fake.show_status("HELLO")

    rendered = _render_last_line(fake.chat_container)
    assert strip_ansi(rendered).strip() == "HELLO"
    assert rendered != strip_ansi(rendered)

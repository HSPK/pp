"""The interactive TUI mode.

Ported from ``packages/coding-agent/src/modes/interactive/interactive-mode.ts``.

Scope of this port
------------------
The TypeScript class is 6399 lines and fans out into the whole product surface.
This module ports the parts that make the interactive terminal actually work:

* the composition root (`create_interactive_tui`, container/editor/footer
  layout, keybindings, theme registration),
* `init` / `run` and the main input loop,
* the app key handlers and the editor submit handler with slash-command
  dispatch,
* the agent event subscription and transcript rendering (user, assistant,
  thinking, tool execution, custom entries, bash execution),
* selector overlay management for the ported dialogs,
* `!`/`!!` bash execution and the pending-message display,
* signal handling and shutdown.

Not ported yet, and *documented rather than silently missing*: the extension UI
host (widgets, custom header/footer, extension dialogs, terminal input
listeners), the resource/diagnostic startup report, the startup changelog
banner (``showStartupNoticesIfNeeded``) and update notifications, tmux
keyboard detection, and fullscreen mode switching (``switchTuiMode``).

The only slash commands with no implementation are the two easter eggs
``/arminsayshi`` and ``/dementedelves``; each has a ``_unsupported_command``
entry so the user gets a clear message instead of a silent no-op. ``/export``,
``/import``, ``/share``, ``/clone``, ``/tree`` and ``/changelog`` *are*
implemented below in the slash-command dispatch.

Because the startup changelog banner is missing, ``settings.changelog`` /
``getLastChangelogVersion`` and the ``collapse-changelog`` setting persist
correctly but have no reader in this port.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass, field, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pi_agent.harness.session.context import create_compaction_summary_message
from pi_ai.auth.types import AuthCheck, AuthEvent, AuthInteraction, AuthPrompt
from pi_ai.registry import Model
from pi_ai.types import AssistantMessage

# `race_with_abort_signal` also covers `packages/coding-agent/src/utils/abort.ts`'s
# `raceWithAbortSignal` -- see `pi_ai.utils.abort`'s module docstring for why this
# package reuses that implementation instead of duplicating a second copy.
from pi_ai.utils.abort import race_with_abort_signal
from pi_tui.autocomplete import (
    AutocompleteItem,
    CombinedAutocompleteProvider,
    SlashCommand,
)
from pi_tui.component import Component, Container
from pi_tui.components.markdown import Markdown, MarkdownTheme
from pi_tui.components.scroll_view import ScrollView, ScrollViewOptions
from pi_tui.components.spacer import Spacer
from pi_tui.components.stack import StackEntry
from pi_tui.components.text import Text
from pi_tui.components.v_stack import VStack
from pi_tui.fuzzy import fuzzy_filter
from pi_tui.keybindings import set_keybindings
from pi_tui.tasks import spawn
from pi_tui.terminal import ProcessTerminal, Terminal
from pi_tui.tui import TuiStopOptions
from pi_tui.tui_alt_screen import TuiAltScreen, TuiAltScreenOptions
from pi_tui.tui_main_screen import TuiMainScreen
from pi_tui.utils import visible_width

from ...core.agent_session_runtime import SessionImportFileNotFoundError
from ...core.app_keybindings import KeybindingsManager
from ...core.config import (
    APP_NAME,
    APP_TITLE,
    VERSION,
    get_agent_dir,
    get_bin_dir,
    get_changelog_path,
    get_debug_log_path,
    get_share_viewer_url,
)
from ...core.extensions.types import UserBashEvent
from ...core.footer_data_provider import FooterDataProvider
from ...core.http_dispatcher import configure_http_dispatcher, format_http_idle_timeout_ms
from ...core.model_resolver import ScopedModel, find_exact_model_reference_match, resolve_model_scope_from_models
from ...core.session_cwd import MissingSessionCwdError, format_missing_session_cwd_prompt
from ...core.slash_commands import BUILTIN_SLASH_COMMANDS, find_builtin_slash_command
from ...core.trust_manager import ProjectTrustStore
from ...utils.changelog import normalize_changelog_links, parse_changelog
from ...utils.clipboard import copy_to_clipboard, read_clipboard_text
from ...utils.open_browser import open_browser
from ...utils.shell import kill_tracked_detached_children
from ...utils.version_check import check_for_new_pi_version
from .components.assistant_message import AssistantMessageComponent
from .components.bash_execution import BashExecutionComponent
from .components.custom_editor import CustomEditor
from .components.custom_message import CustomEntryComponent, CustomMessageComponent
from .components.dynamic_border import DynamicBorder
from .components.footer import FooterComponent
from .components.keybinding_hints import key_hint, key_text, raw_key_hint
from .components.login_dialog import LoginCancelledError, LoginDialogComponent
from .components.model_selector import ModelSelectorComponent, ScopedModelItem
from .components.oauth_selector import (
    AuthSelectorProvider,
    AuthType,
    OAuthSelectorComponent,
    format_auth_selector_provider_type,
)
from .components.scoped_models_selector import (
    ModelsCallbacks,
    ModelsConfig,
    ScopedModelsSelectorComponent,
)
from .components.session_selector import SessionSelectorComponent
from .components.settings_selector import (
    SettingsCallbacks,
    SettingsConfig,
    SettingsSelectorComponent,
)
from .components.simple_selectors import (
    ConfirmSelectorComponent,
    ShowImagesSelectorComponent,
    ThemeSelectorComponent,
    ThinkingSelectorComponent,
)
from .components.status_indicator import (
    BranchSummaryStatusIndicator,
    CompactionStatusIndicator,
    IdleStatus,
    RetryStatusIndicator,
    StatusIndicator,
    WorkingStatusIndicator,
)
from .components.summary_messages import (
    BranchSummaryMessageComponent,
    CompactionSummaryMessageComponent,
)
from .components.tool_execution import (
    ToolExecutionComponent,
    ToolExecutionOptions,
    ToolResult,
)
from .components.tree_selector import TreeSelectorComponent
from .components.trust_selector import TrustSelectorComponent
from .components.user_message import UserMessageComponent
from .components.user_message_selector import UserMessageItem, UserMessageSelectorComponent
from .external_editor import ExternalEditorOptions, edit_in_external_editor
from .model_search import ModelSearchItem, get_model_search_text
from .theme.theme import (
    get_available_themes,
    get_editor_theme,
    get_markdown_theme,
    set_registered_themes,
    set_theme,
    theme,
)
from .theme.theme_controller import InteractiveThemeController

if TYPE_CHECKING:
    from ...core.agent_session import AgentSession
    from ...core.agent_session_runtime import AgentSessionRuntime
    from ...core.session_manager import SessionManager

DOUBLE_PRESS_WINDOW_S = 0.5

ANTHROPIC_SUBSCRIPTION_AUTH_WARNING = (
    "Anthropic subscription auth is active. Third-party harness usage draws from extra usage and is billed "
    "per token, not your Claude plan limits. Manage extra usage at https://claude.ai/settings/usage. "
    "Disable this warning in /settings."
)


def _is_anthropic_subscription_auth_key(api_key: str | None) -> bool:
    return isinstance(api_key, str) and api_key.startswith("sk-ant-oat")


_AUTH_TYPE_ORDER: dict[str, int] = {"oauth": 0, "api_key": 1}


@dataclass
class LoginProviderCompletionOption:
    """One provider in `/login`'s argument completions, with its auth methods merged."""

    id: str
    name: str
    auth_types: list[AuthType] = field(default_factory=list)


def get_login_provider_completion_options(
    provider_options: Sequence[AuthSelectorProvider],
) -> list[LoginProviderCompletionOption]:
    """Collapse the per-method selector options into one entry per provider."""
    by_id: dict[str, LoginProviderCompletionOption] = {}
    for provider in provider_options:
        existing = by_id.get(provider.id)
        if existing is not None:
            if provider.auth_type not in existing.auth_types:
                existing.auth_types.append(provider.auth_type)
                existing.auth_types.sort(key=lambda auth_type: _AUTH_TYPE_ORDER[auth_type])
            continue
        by_id[provider.id] = LoginProviderCompletionOption(
            id=provider.id,
            name=provider.name,
            auth_types=[provider.auth_type],
        )
    return sorted(by_id.values(), key=lambda option: option.name)


def get_login_provider_search_text(provider: LoginProviderCompletionOption) -> str:
    auth_types = " ".join(
        f"{auth_type} {format_auth_selector_provider_type(auth_type)}" for auth_type in provider.auth_types
    )
    return f"{provider.id} {provider.name} {auth_types}"


def format_login_provider_completion_description(provider: LoginProviderCompletionOption) -> str:
    auth_types = "/".join(format_auth_selector_provider_type(auth_type) for auth_type in provider.auth_types)
    return auth_types if provider.name == provider.id else f"{provider.name} · {auth_types}"


class _DialogAuthInteraction(AuthInteraction):
    """Bridges `pi_ai`'s auth flows to the login dialog."""

    def __init__(self, mode: InteractiveMode, dialog: LoginDialogComponent) -> None:
        self.signal = dialog.signal  # type: ignore[assignment]
        self._mode = mode
        self._dialog = dialog

    async def prompt(self, prompt: AuthPrompt) -> str:
        return await self._mode._show_auth_prompt(self._dialog, prompt)

    def notify(self, event: AuthEvent) -> None:
        self._mode._notify_auth_dialog(self._dialog, event)


def _debug_encode(message: object) -> object:
    """Best-effort JSON encoding of a transcript message for the debug log."""
    if is_dataclass(message) and not isinstance(message, type):
        return dataclass_asdict(message)
    return repr(message)


def _quote_if_needed(value: str) -> str:
    """Shell-quote a path for the resume hint unless it is already shell-safe."""
    if value and re.fullmatch(r"[a-zA-Z0-9_\-./~:@]*", value):
        return value
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"


def format_resume_command(session_manager: SessionManager) -> str | None:
    """Build the `pi --session ...` hint printed on exit, or `None` if not resumable."""
    if not sys.stdout.isatty():
        return None
    if not session_manager.is_persisted():
        return None
    session_file = session_manager.get_session_file()
    if not session_file or not os.path.exists(session_file):
        return None
    args = [APP_NAME]
    if not session_manager.uses_default_session_dir():
        args.extend(["--session-dir", _quote_if_needed(session_manager.get_session_dir())])
    args.extend(["--session", session_manager.get_session_id()])
    return " ".join(args)


@dataclass
class CompactionQueuedMessage:
    """Input typed while compaction is running, replayed once it finishes."""

    text: str
    mode: Literal["steer", "followUp"]


@dataclass
class InteractiveModeOptions:
    migrated_providers: list[str] = field(default_factory=list)
    model_fallback_message: str | None = None
    auto_trust_on_reload_cwd: str | None = None
    initial_message: str | None = None
    initial_images: list[Any] = field(default_factory=list)
    initial_messages: list[str] = field(default_factory=list)
    verbose: bool = False
    tui_mode: str | None = None
    initial_theme_setting: str | None = None
    """`--use-theme`: the theme for this run only, never written to settings."""


def create_interactive_tui(
    tui_mode: str = "regular",
    show_hardware_cursor: bool = False,
    log_directory: str | None = None,
    terminal: Terminal | None = None,
    on_right_click_paste: Callable[[], None] | None = None,
) -> TuiMainScreen | TuiAltScreen:
    """Composition root for the interactive renderer.

    ``fullscreen`` uses the alternate screen with an application-owned
    scrollable viewport; ``regular`` renders inline in the terminal scrollback.
    """
    resolved_terminal = terminal or ProcessTerminal()
    if tui_mode == "fullscreen":
        return TuiAltScreen(
            resolved_terminal,
            show_hardware_cursor,
            log_directory,
            TuiAltScreenOptions(open_url=open_browser, on_right_click_paste=on_right_click_paste),
        )
    return TuiMainScreen(resolved_terminal, show_hardware_cursor, log_directory)


class InteractiveTuiReference:
    """Stable handle to a renderer that may be swapped out underneath it.

    Python port of `createInteractiveTuiReference` in
    `packages/coding-agent/src/modes/interactive/interactive-mode.ts` (a
    `Proxy` there; `__getattr__`/`__setattr__` here).

    Every attribute read is resolved against the *current* renderer. Callables
    are additionally wrapped so that a method captured once
    (``request_render = ui.request_render``) keeps working after the renderer
    is replaced: the wrapper re-resolves the bound method whenever the renderer
    identity changed since capture. That is what makes it safe to hand
    ``ui.request_render`` to a component and later switch the whole renderer.

    The re-resolution is deliberately conditional on renderer *identity*. If it
    re-resolved unconditionally, the common monkey-patch idiom
    ``original = ui.render; ui.render = lambda w: original(w)`` would recurse
    forever, because ``original`` would find the patched attribute again.
    """

    def __init__(self, get_tui: Callable[[], Any]) -> None:
        object.__setattr__(self, "_get_tui", get_tui)

    def __getattr__(self, name: str) -> Any:
        get_tui: Callable[[], Any] = object.__getattribute__(self, "_get_tui")
        tui = get_tui()
        value = getattr(tui, name)
        if not callable(value):
            return value

        captured = {"tui": tui, "method": value}

        def call(*args: Any, **kwargs: Any) -> Any:
            current = get_tui()
            if current is not captured["tui"]:
                current_method = getattr(current, name)
                if not callable(current_method):
                    raise TypeError(f"TUI property {name} is not callable")
                captured["tui"] = current
                captured["method"] = current_method
            return captured["method"](*args, **kwargs)

        return call

    def __setattr__(self, name: str, value: Any) -> None:
        get_tui: Callable[[], Any] = object.__getattribute__(self, "_get_tui")
        setattr(get_tui(), name, value)

    def __delattr__(self, name: str) -> None:
        get_tui: Callable[[], Any] = object.__getattribute__(self, "_get_tui")
        delattr(get_tui(), name)


def create_interactive_tui_reference(get_tui: Callable[[], Any]) -> Any:
    """See :class:`InteractiveTuiReference`."""
    return InteractiveTuiReference(get_tui)


class InteractiveMode:
    def __init__(
        self,
        runtime_host: AgentSessionRuntime,
        options: InteractiveModeOptions | None = None,
        terminal: Terminal | None = None,
    ) -> None:
        self.runtime_host = runtime_host
        # Port of TS's constructor wiring: without this the mode keeps its event
        # subscription attached to the session that `/new`, `/import` and
        # `/clone` just disposed, and the replacement never gets its
        # `session_start`.
        self.runtime_host.set_rebind_session(self._rebind_current_session)
        self.options = options or InteractiveModeOptions()
        self.version = VERSION

        self.keybindings = KeybindingsManager.create()
        set_keybindings(self.keybindings)

        self.tui_mode = self.options.tui_mode or self.settings_manager.get_tui_mode()
        self.renderer = create_interactive_tui(
            tui_mode=self.tui_mode,
            show_hardware_cursor=self.settings_manager.get_show_hardware_cursor(),
            log_directory=get_agent_dir(),
            terminal=terminal,
            on_right_click_paste=lambda: spawn(self._handle_right_click_paste()),
        )
        self._main_screen_render_state: Any = None
        self.ui = create_interactive_tui_reference(lambda: self.renderer)
        self.ui.set_clear_on_shrink(self.settings_manager.get_clear_on_shrink())

        self.header_container = Container()
        self.loaded_resources_container = Container()
        self.chat_container = Container()
        self.document_container = Container()
        self.document_container.add_child(self.header_container)
        self.document_container.add_child(self.loaded_resources_container)
        self.document_container.add_child(self.chat_container)
        self.pending_messages_container = Container()
        self.status_container = Container()

        self.default_editor = CustomEditor(self.ui, get_editor_theme(), self.keybindings)
        self.editor: Any = self.default_editor
        self.editor_container = Container()
        self.editor_container.add_child(self.editor)

        self.footer_data_provider = FooterDataProvider(self.session_manager.get_cwd())
        self.footer = FooterComponent(self.session, self.footer_data_provider)
        self.footer.set_auto_compact_enabled(self.session.auto_compaction_enabled)
        self.footer_container = Container()
        self.footer_container.add_child(self.footer)

        self.hide_thinking_block = self.settings_manager.get_hide_thinking_block()
        self.output_pad = self.settings_manager.get_output_pad()
        self.hidden_thinking_label = "Thinking..."
        self.default_working_message = "Working..."

        self.is_initialized = False
        self.shutdown_requested = False
        self.is_shutting_down = False
        self.anthropic_subscription_warning_shown = False
        self.compaction_queued_messages: list[CompactionQueuedMessage] = []
        self.auto_compaction_escape_handler: Callable[[], None] | None = None
        self._on_input_future: asyncio.Future[str] | None = None
        self.pending_user_inputs: list[str] = []

        self.idle_status = IdleStatus()
        self.active_status_indicator: StatusIndicator | None = None
        self.working_visible = True

        self.last_sigint_time = 0.0
        self.last_escape_time = 0.0

        self.streaming_component: AssistantMessageComponent | None = None
        self.streaming_message: AssistantMessage | None = None
        self.pending_tools: dict[str, ToolExecutionComponent] = {}
        self.tool_output_expanded = False
        # Back-to-back status lines update the previous line instead of piling
        # up; these hold the pair `show_status` last appended.
        self.last_status_spacer: Spacer | None = None
        self.last_status_text: Text | None = None

        self.is_bash_mode = False
        self.bash_component: BashExecutionComponent | None = None
        self.pending_bash_components: list[BashExecutionComponent] = []

        self.fd_path: str | None = None
        self.transcript_scroll_view: ScrollView | None = None
        self.skill_commands: dict[str, str] = {}
        self.autocomplete_provider: CombinedAutocompleteProvider | None = None
        self._active_selector: Component | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._signal_cleanup: list[Callable[[], None]] = []

        # TypeScript registers themes discovered by the resource loader. This
        # port's `ResourceLoader` deliberately does not load themes (see its
        # module docstring), so only the built-in and custom-directory themes
        # that `init_theme` finds on its own are available.
        set_registered_themes([])
        self.theme_controller = InteractiveThemeController(
            self.ui,
            lambda: self.settings_manager,
            self.show_error,
            self._update_editor_border_color,
            self.options.initial_theme_setting,
        )

    # -- convenience accessors ---------------------------------------------

    @property
    def session(self) -> AgentSession:
        return self.runtime_host.session

    @property
    def session_manager(self) -> Any:
        return self.session.session_manager

    @property
    def settings_manager(self) -> Any:
        return self.session.settings_manager

    def _markdown_theme(self) -> MarkdownTheme:
        return replace(get_markdown_theme(), code_block_indent=self.settings_manager.get_code_block_indent())

    # -- lifecycle ----------------------------------------------------------

    async def init(self) -> None:
        if self.is_initialized:
            return

        self._register_signal_handlers()

        self._mount_layout()
        self.ui.set_focus(self.editor)

        self.fd_path = self._resolve_fd_path()
        self._setup_key_handlers()
        self._setup_editor_submit_handler()
        self._apply_runtime_settings()
        self._setup_autocomplete_provider()

        self.ui.start()
        self.is_initialized = True

        await self.theme_controller.apply_from_settings()

        if self.options.verbose or not self.settings_manager.get_quiet_startup():
            self.header_container.add_child(Spacer(1))
            self.header_container.add_child(Text(self._build_header_text(), 1, 0))
            self.header_container.add_child(Spacer(1))
        else:
            self.header_container.add_child(Text("", 0, 0))

        # Port of TS `rebindCurrentSession()`, which `init()` awaits before rendering the
        # initial messages: it binds the extension host and, in doing so, emits
        # `session_start`. The UI-host half of the binding is a documented omission, but
        # the event must still reach extensions or no `on_session_start` handler ever runs.
        startup_session = self.session
        await startup_session.bind_extensions()

        # TS's `if (this.session !== session) return;`: a replacement that
        # landed during that await already rebound and subscribed, so resuming
        # here would attach a second listener to the same session.
        if self.session is not startup_session:
            self.status_container.add_child(self.idle_status)
            self.footer_data_provider.on_branch_change(self.ui.request_render)
            self.footer_data_provider.start_watching()
            self.ui.request_render()
            return

        self._subscribe_to_agent()
        self._update_available_provider_count()
        self._render_initial_messages()
        self._update_terminal_title()
        self._update_editor_border_color()
        self.status_container.add_child(self.idle_status)

        self.footer_data_provider.on_branch_change(self.ui.request_render)
        self.footer_data_provider.start_watching()
        self.ui.request_render()

    def switch_tui_mode(self, mode: str, *, restore_progress: bool = True, start_renderer: bool = True) -> bool:
        """Swap the live renderer between `regular` and `fullscreen`.

        Port of `interactive-mode.ts:788`. Previously this port only persisted
        the setting and reported "applies on next start", because swapping the
        renderer means re-parenting every component onto a new one.

        Returns `False` without touching anything when an overlay is open --
        upstream refuses for the same reason: the overlay belongs to the
        renderer being torn down, so it would be orphaned.
        """
        previous = self.renderer
        if mode == self.tui_mode:
            return True
        if getattr(previous, "has_overlay_entries", False):
            return False

        components = list(previous.children)
        focus = previous.get_focused_component()
        terminal = previous.terminal
        show_hardware_cursor = previous.get_show_hardware_cursor()
        clear_on_shrink = previous.get_clear_on_shrink()
        if isinstance(previous, TuiMainScreen):
            self._main_screen_render_state = previous.capture_render_state()

        previous.stop(TuiStopOptions(preserve_screen=True))
        previous.set_focus(None)
        previous.clear()
        if isinstance(previous, TuiAltScreen):
            previous.set_layout_root(None)

        self.renderer = create_interactive_tui(
            tui_mode=mode,
            show_hardware_cursor=show_hardware_cursor,
            log_directory=get_agent_dir(),
            terminal=terminal,
            on_right_click_paste=lambda: spawn(self._handle_right_click_paste()),
        )
        self.renderer.set_clear_on_shrink(clear_on_shrink)
        if isinstance(self.renderer, TuiMainScreen) and self._main_screen_render_state is not None:
            self.renderer.restore_render_state(self._main_screen_render_state)

        self.tui_mode = mode
        self.options.tui_mode = mode
        # `_mount_layout` reads `self.renderer`, so the containers are re-parented
        # onto the new renderer and the fullscreen dock is rebuilt from scratch.
        self._mount_layout()
        for component in components:
            if component not in self.renderer.children:
                self.renderer.add_child(component)
        self.renderer.invalidate()
        self.renderer.set_focus(focus)
        if not start_renderer:
            return True
        self.renderer.start()
        self._apply_fullscreen_scrollbar_setting()
        if (
            restore_progress
            and self.settings_manager.get_show_terminal_progress()
            and (self.session.is_streaming or self.session.is_compacting)
        ):
            terminal.set_progress(True)
        self.ui.request_render()
        return True

    def _mount_layout(self) -> None:
        """Mount the component tree for the active renderer.

        Fullscreen builds a `ScrollView` transcript above a fixed dock so the
        editor and footer stay pinned; regular mode appends the same containers
        to the main screen, which grows the scrollback naturally.
        """
        for container in (
            self.document_container,
            self.pending_messages_container,
            self.status_container,
            self.editor_container,
            self.footer_container,
        ):
            self.ui.add_child(container)

        if isinstance(self.renderer, TuiAltScreen):
            self.transcript_scroll_view = ScrollView(
                self.document_container,
                ScrollViewOptions(
                    follow="end",
                    primary=True,
                    overscroll="chain",
                    scrollbar=self.settings_manager.get_fullscreen_scrollbar(),
                    scrollbar_style=lambda text: theme.bg("scrollbarThumb", text),
                ),
            )
            dock = VStack(
                [
                    StackEntry(component=self.pending_messages_container, shrink=1, min_size=0),
                    StackEntry(component=self.status_container, shrink=1, min_size=0),
                    StackEntry(component=self.editor_container, shrink=1, min_size=3),
                    StackEntry(component=self.footer_container, shrink=1, min_size=1),
                ]
            )
            self.renderer.set_layout_root(
                VStack(
                    [
                        StackEntry(
                            component=self.transcript_scroll_view,
                            basis=0,
                            grow=1,
                            shrink=1,
                            min_size=1,
                        ),
                        StackEntry(component=dock, basis="auto", grow=0, shrink=1, min_size=1),
                    ]
                )
            )

    def _build_header_text(self) -> str:
        logo = theme.bold(theme.fg("accent", APP_NAME)) + theme.fg("dim", f" v{self.version}")
        compact_instructions = theme.fg("muted", " · ").join(
            [
                key_hint("app.interrupt", "interrupt"),
                raw_key_hint(f"{key_text('app.clear')}/{key_text('app.exit')}", "clear/exit"),
                raw_key_hint("/", "commands"),
                raw_key_hint("!", "bash"),
                key_hint("app.tools.expand", "more"),
            ]
        )
        onboarding = theme.fg(
            "dim", "Pi can explain its own features and look up its docs. Ask it how to use or extend Pi."
        )
        return f"{logo}\n{compact_instructions}\n\n{onboarding}"

    def _update_terminal_title(self) -> None:
        cwd_basename = os.path.basename(self.session_manager.get_cwd())
        session_name = self.session_manager.get_session_name()
        title = f"{APP_TITLE} - {session_name} - {cwd_basename}" if session_name else f"{APP_TITLE} - {cwd_basename}"
        with contextlib.suppress(Exception):
            self.ui.terminal.set_title(title)

    async def run(self) -> None:
        await self.init()

        if self.options.migrated_providers:
            self.show_warning(f"Migrated credentials to auth.json: {', '.join(self.options.migrated_providers)}")
        if self.options.model_fallback_message:
            self.show_warning(self.options.model_fallback_message)

        spawn(self._maybe_warn_about_anthropic_subscription_auth())
        spawn(self._check_for_new_version())

        for message in [
            *([self.options.initial_message] if self.options.initial_message else []),
            *self.options.initial_messages,
        ]:
            await self._prompt_safely(message)

        while not self.shutdown_requested:
            user_input = await self.get_user_input()
            if self.shutdown_requested:
                break
            await self._prompt_safely(user_input)

    async def _check_for_new_version(self) -> None:
        """Announce a newer GitHub release, if any. Never raises."""
        release = await check_for_new_pi_version(self.version, settings_manager=self.settings_manager)
        if release is None:
            return
        message = f"A new version is available: {release.version} (current {self.version})"
        if release.url:
            message += f"\n{release.url}"
        self.show_status(message)

    async def _prompt_safely(self, text: str, **kwargs: Any) -> None:
        try:
            await self.session.prompt(text, **kwargs)
        except Exception as error:
            self.show_error(str(error) or "Unknown error occurred")

    async def get_user_input(self) -> str:
        if self.pending_user_inputs:
            return self.pending_user_inputs.pop(0)
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._on_input_future = future
        return await future

    def _deliver_input(self, text: str) -> None:
        future = self._on_input_future
        if future is not None and not future.done():
            self._on_input_future = None
            future.set_result(text)
        else:
            self.pending_user_inputs.append(text)

    # -- status / notices ---------------------------------------------------

    def _append_status(self, text: str) -> None:
        self.chat_container.add_child(Spacer(1))
        self.chat_container.add_child(Text(text, 1, 0))
        self.ui.request_render()

    def show_status(self, message: str) -> None:
        """Show a status line in the chat.

        Statuses emitted back to back, with nothing else added to the chat in
        between, replace the previous line instead of appending another one.
        """
        children = self.chat_container.children
        last = children[-1] if len(children) > 0 else None
        second_last = children[-2] if len(children) > 1 else None

        if (
            last is not None
            and second_last is not None
            and last is self.last_status_text
            and second_last is self.last_status_spacer
        ):
            self.last_status_text.set_text(theme.fg("dim", message))
            self.ui.request_render()
            return

        spacer = Spacer(1)
        text = Text(theme.fg("dim", message), 1, 0)
        self.chat_container.add_child(spacer)
        self.chat_container.add_child(text)
        self.last_status_spacer = spacer
        self.last_status_text = text
        self.ui.request_render()

    def show_warning(self, message: str) -> None:
        self._append_status(theme.fg("warning", message))

    def show_error(self, message: str) -> None:
        self._append_status(theme.fg("error", message))

    def _show_status_indicator(self, indicator: StatusIndicator) -> None:
        self._clear_status_indicator()
        self.active_status_indicator = indicator
        self.status_container.clear()
        self.status_container.add_child(indicator)
        self.ui.request_render()

    def _clear_status_indicator(self, kind: str | None = None) -> None:
        indicator = self.active_status_indicator
        if kind is not None and (indicator is None or indicator.kind != kind):
            return
        had_active_status_indicator = indicator is not None
        if indicator is not None:
            indicator.dispose()
        self.active_status_indicator = None
        self.status_container.clear()
        # The idle placeholder only reserves height on the main-screen renderer,
        # and only when `clearOnShrink` is on; the alt screen owns its viewport.
        if had_active_status_indicator and self.tui_mode == "regular" and self.ui.get_clear_on_shrink():
            self.status_container.add_child(self.idle_status)

    # -- key handlers -------------------------------------------------------

    def _setup_key_handlers(self) -> None:
        self.default_editor.on_escape = self._handle_escape
        self.default_editor.on_ctrl_d = self._handle_ctrl_d
        self.default_editor.on_action("app.clear", self._handle_ctrl_c)
        self.default_editor.on_action("app.suspend", self._handle_ctrl_z)
        self.default_editor.on_action("app.thinking.cycle", self._cycle_thinking_level)
        self.default_editor.on_action("app.model.cycleForward", lambda: spawn(self._cycle_model("forward")))
        self.default_editor.on_action("app.model.cycleBackward", lambda: spawn(self._cycle_model("backward")))
        self.default_editor.on_action("app.model.select", self.show_model_selector)
        self.default_editor.on_action("app.tools.expand", self._toggle_tool_output_expansion)
        self.default_editor.on_action("app.thinking.toggle", self._toggle_thinking_block_visibility)
        self.default_editor.on_action("app.session.new", lambda: spawn(self._handle_clear_command()))
        self.default_editor.on_action("app.session.tree", self.show_tree_selector)
        self.default_editor.on_action("app.session.fork", self.show_user_message_selector)
        self.default_editor.on_action("app.session.resume", self.show_session_selector)
        self.default_editor.on_action(
            "app.message.copy", lambda: spawn(self._handle_copy_command(flash_confirmation=True))
        )
        self.default_editor.on_action("app.message.followUp", lambda: spawn(self._handle_follow_up()))
        self.default_editor.on_action("app.message.dequeue", self._handle_dequeue)
        self.default_editor.on_action("app.editor.external", lambda: spawn(self._handle_open_external_editor()))
        self.default_editor.on_change = self._on_editor_change

    def _apply_runtime_settings(self) -> None:
        """Push settings into the pieces that cache them.

        Ported from ``applyRuntimeSettings``. The TypeScript version runs this
        on every session rebind; here it runs at startup and after ``/reload``.
        """
        configure_http_dispatcher(self.settings_manager.get_http_idle_timeout_ms())
        self._apply_fullscreen_scrollbar_setting()
        self.footer.set_session(self.session)
        self.footer.set_auto_compact_enabled(self.session.auto_compaction_enabled)
        self.footer_data_provider.set_cwd(self.session_manager.get_cwd())
        self.hide_thinking_block = self.settings_manager.get_hide_thinking_block()
        self.output_pad = self.settings_manager.get_output_pad()
        self.ui.set_show_hardware_cursor(self.settings_manager.get_show_hardware_cursor())
        clear_on_shrink = self.settings_manager.get_clear_on_shrink()
        self.ui.set_clear_on_shrink(clear_on_shrink)
        if not clear_on_shrink and self.active_status_indicator is None:
            self.status_container.clear()

        editor_padding_x = self.settings_manager.get_editor_padding_x()
        autocomplete_max_visible = self.settings_manager.get_autocomplete_max_visible()
        self.default_editor.set_padding_x(editor_padding_x)
        self.default_editor.set_autocomplete_max_visible(autocomplete_max_visible)
        if self.editor is not self.default_editor:
            set_padding = getattr(self.editor, "set_padding_x", None)
            if set_padding is not None:
                set_padding(editor_padding_x)
            set_max_visible = getattr(self.editor, "set_autocomplete_max_visible", None)
            if set_max_visible is not None:
                set_max_visible(autocomplete_max_visible)

    def _update_available_provider_count(self) -> None:
        scoped = self.session.scoped_models
        models = [item.model for item in scoped] if scoped else self.session.model_runtime.get_available_snapshot()
        self.footer_data_provider.set_available_provider_count(len({model.provider for model in models}))

    # -- autocomplete -------------------------------------------------------

    def _prefix_autocomplete_description(self, description: str | None, source_info: Any = None) -> str | None:
        tag = self._get_autocomplete_source_tag(source_info)
        if tag is None:
            return description
        return f"[{tag}] {description}" if description else f"[{tag}]"

    def _get_autocomplete_source_tag(self, source_info: Any) -> str | None:
        if source_info is None:
            return None
        scope = getattr(source_info, "scope", None)
        scope_prefix = "u" if scope == "user" else "p" if scope == "project" else "t"
        source = (getattr(source_info, "source", "") or "").strip()
        if source in ("auto", "local", "cli"):
            return scope_prefix
        return f"{scope_prefix}:{source}" if source else scope_prefix

    async def _model_argument_completions(self, prefix: str) -> list[AutocompleteItem] | None:
        scoped = self.session.scoped_models
        models = [item.model for item in scoped] if scoped else self.session.model_runtime.get_available_snapshot()
        if not models:
            return None
        matches = fuzzy_filter(
            models,
            prefix,
            lambda model: get_model_search_text(ModelSearchItem(id=model.id, provider=model.provider, name=model.name)),
        )
        if not matches:
            return None
        return [
            AutocompleteItem(value=f"{model.provider}/{model.id}", label=model.id, description=model.provider)
            for model in matches
        ]

    async def _login_argument_completions(self, prefix: str) -> list[AutocompleteItem] | None:
        providers = get_login_provider_completion_options(self.get_login_provider_options())
        matches = fuzzy_filter(providers, prefix, get_login_provider_search_text)
        if not matches:
            return None
        return [
            AutocompleteItem(
                value=provider.id,
                label=provider.id,
                description=format_login_provider_completion_description(provider),
            )
            for provider in matches
        ]

    def _create_base_autocomplete_provider(self) -> CombinedAutocompleteProvider:
        commands: list[SlashCommand] = []
        for builtin in BUILTIN_SLASH_COMMANDS:
            command = SlashCommand(
                name=builtin.name,
                description=builtin.description,
                argument_hint=builtin.argument_hint,
            )
            if builtin.name == "model":
                command.get_argument_completions = self._model_argument_completions
            if builtin.name == "login":
                command.get_argument_completions = self._login_argument_completions
            commands.append(command)

        builtin_names = {command.name for command in commands}

        for template in self.session.prompt_templates:
            commands.append(
                SlashCommand(
                    name=template.name,
                    description=self._prefix_autocomplete_description(template.description, template.source_info),
                    argument_hint=template.argument_hint,
                )
            )

        # Skills become `/skill:<name>` commands when enabled.
        self.skill_commands.clear()
        if self.settings_manager.get_enable_skill_commands():
            for skill in self.session.resource_loader.get_skills().skills:
                command_name = f"skill:{skill.name}"
                if command_name in builtin_names:
                    continue
                self.skill_commands[command_name] = skill.file_path
                commands.append(
                    SlashCommand(
                        name=command_name,
                        description=self._prefix_autocomplete_description(
                            skill.description, getattr(skill, "source_info", None)
                        ),
                    )
                )

        return CombinedAutocompleteProvider(list(commands), self.session_manager.get_cwd(), self.fd_path)

    def _resolve_fd_path(self) -> str | None:
        """Locate ``fd``, used for `@`-prefixed file completion.

        TypeScript's ``ensureTool("fd")`` (``utils/tools-manager.ts``) downloads
        a pinned binary into the agent's ``bin`` directory when it is missing.
        That downloader is not ported, so this only looks for an already
        installed binary: the agent bin directory first, then ``PATH``.
        Debian and Ubuntu ship the binary as ``fdfind``.
        """
        for candidate in ("fd", "fdfind"):
            in_bin_dir = Path(get_bin_dir()) / candidate
            if in_bin_dir.is_file() and os.access(in_bin_dir, os.X_OK):
                return str(in_bin_dir)
        for candidate in ("fd", "fdfind"):
            found = shutil.which(candidate)
            if found:
                return found
        return None

    def _setup_autocomplete_provider(self) -> None:
        self.autocomplete_provider = self._create_base_autocomplete_provider()
        self.default_editor.set_autocomplete_provider(self.autocomplete_provider)
        if self.editor is not self.default_editor:
            setter = getattr(self.editor, "set_autocomplete_provider", None)
            if setter is not None:
                setter(self.autocomplete_provider)

    def _on_editor_change(self, text: str) -> None:
        was_bash_mode = self.is_bash_mode
        self.is_bash_mode = text.lstrip().startswith("!")
        if was_bash_mode != self.is_bash_mode:
            self._update_editor_border_color()

    def _update_editor_border_color(self) -> None:
        # `Editor` copies `theme.border_color` into its own public
        # `border_color` at construction (`editor.py:421`) and renders from
        # that, so replacing `_theme` changed a field nothing reads and the
        # border never moved -- shift+tab cycled the thinking level with no
        # visible feedback. TypeScript assigns `this.editor.borderColor`
        # directly (`interactive-mode.ts:3996`).
        if self.is_bash_mode:
            border_color = theme.get_bash_mode_border_color()
        else:
            border_color = theme.get_thinking_border_color(self.session.thinking_level)
        self.default_editor.border_color = border_color
        self.default_editor.invalidate()
        self.ui.request_render()

    def _handle_escape(self) -> None:
        if self.session.is_streaming:
            self._restore_queued_messages_to_editor(abort=True)
            return
        if self.session.is_bash_running:
            self.session.abort_bash()
            return
        if self.is_bash_mode:
            self.editor.set_text("")
            self.is_bash_mode = False
            self._update_editor_border_color()
            return
        if self.editor.get_text().strip():
            return

        action = self.settings_manager.get_double_escape_action()
        if action == "none":
            return
        now = time.monotonic()
        if now - self.last_escape_time < DOUBLE_PRESS_WINDOW_S:
            if action == "tree":
                self.show_tree_selector()
            else:
                self.show_user_message_selector()
            self.last_escape_time = 0.0
        else:
            self.last_escape_time = now

    def _handle_ctrl_c(self) -> None:
        if self.editor.get_text():
            self.editor.set_text("")
            self.last_sigint_time = 0.0
            self.ui.request_render()
            return
        now = time.monotonic()
        if now - self.last_sigint_time < DOUBLE_PRESS_WINDOW_S:
            spawn(self.shutdown())
        else:
            self.last_sigint_time = now
            self.show_status(f"Press {key_text('app.clear')} again to exit")

    def _handle_ctrl_d(self) -> None:
        spawn(self.shutdown())

    def _handle_ctrl_z(self) -> None:
        if sys.platform == "win32":
            self.show_status("Suspend to background is not supported on Windows")
            return

        # Ignore SIGINT while suspended so Ctrl+C in the terminal does not kill
        # the backgrounded process. Restored on resume.
        #
        # TypeScript additionally holds a `setInterval` open across the suspend,
        # because `process.kill` returns immediately and Node exits once no
        # ref'd handles remain. `os.kill` blocks inside the stopped process
        # here, so there is nothing to keep alive.
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self.ui.stop()
            # pid=0 targets the whole process group, matching TypeScript's
            # `process.kill(0, "SIGTSTP")` (stops the foreground job, not just
            # this process).
            os.kill(0, signal.SIGTSTP)
        except BaseException:
            signal.signal(signal.SIGINT, previous_sigint)
            raise
        signal.signal(signal.SIGINT, previous_sigint)
        self.ui.start()
        self.ui.request_render(True)

    # -- editor submit / slash commands -------------------------------------

    # Easter eggs only; both are bundled ASCII-art animations upstream.
    _UNSUPPORTED_COMMANDS = (
        "/arminsayshi",
        "/dementedelves",
    )

    def _unsupported_command(self, command: str) -> None:
        self.show_warning(f"{command} is not available in this Python port yet.")

    def _setup_editor_submit_handler(self) -> None:
        self.default_editor.on_submit = lambda text: spawn(self._handle_submit(text))

    async def _handle_submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if await self._handle_slash_command(text):
            return

        if text.startswith("!"):
            is_excluded = text.startswith("!!")
            command = text[2:].strip() if is_excluded else text[1:].strip()
            if command:
                if self.session.is_bash_running:
                    self.show_warning("A bash command is already running. Press Esc to cancel it first.")
                    self.editor.set_text(text)
                    return
                self.editor.add_to_history(text)
                self.editor.set_text("")
                await self._handle_bash_command(command, is_excluded)
                self.is_bash_mode = False
                self._update_editor_border_color()
                return

        # Queue input during compaction (extension commands execute immediately).
        if self.session.is_compacting:
            if self._is_extension_command(text):
                self.editor.add_to_history(text)
                self.editor.set_text("")
                await self.session.prompt(text)
            else:
                self._queue_compaction_message(text, "steer")
            return

        if self.session.is_streaming:
            self.editor.add_to_history(text)
            self.editor.set_text("")
            await self._prompt_safely(text, streaming_behavior="steer")
            self.ui.request_render()
            return

        self._flush_pending_bash_components()
        self.editor.add_to_history(text)
        self.editor.set_text("")
        self._deliver_input(text)

    async def _handle_slash_command(self, text: str) -> bool:
        command = text.split(" ", 1)[0]
        argument = text[len(command) :].strip()

        if command == "/settings":
            self.editor.set_text("")
            self.show_settings_selector()
            return True
        if command == "/model":
            self.editor.set_text("")
            await self._handle_model_command(argument or None)
            return True
        if command == "/resume":
            self.editor.set_text("")
            self.show_session_selector()
            return True
        if command == "/fork":
            self.editor.set_text("")
            self.show_user_message_selector()
            return True
        if command == "/trust":
            self.editor.set_text("")
            self.show_trust_selector()
            return True
        if command == "/tree":
            self.editor.set_text("")
            self.show_tree_selector()
            return True
        if command == "/scoped-models":
            self.editor.set_text("")
            self.show_scoped_models_selector()
            return True
        if command == "/copy":
            self.editor.set_text("")
            await self._handle_copy_command()
            return True
        if command == "/changelog":
            self.editor.set_text("")
            self._handle_changelog_command()
            return True
        if command == "/debug":
            self.editor.set_text("")
            self._handle_debug_command()
            return True
        if command == "/reload":
            self.editor.set_text("")
            await self._handle_reload_command()
            return True
        if command == "/export":
            self.editor.set_text("")
            await self._handle_export_command(text)
            return True
        if command == "/import":
            self.editor.set_text("")
            await self._handle_import_command(text)
            return True
        if command == "/clone":
            self.editor.set_text("")
            await self._handle_clone_command()
            return True
        if command == "/share":
            self.editor.set_text("")
            await self._handle_share_command()
            return True
        if command == "/login":
            self.editor.set_text("")
            await self._handle_login_command(argument or None)
            return True
        if command == "/logout":
            self.editor.set_text("")
            self.show_oauth_selector("logout")
            return True
        if command == "/session":
            self.editor.set_text("")
            self._handle_session_command()
            return True
        if command == "/name":
            self.editor.set_text("")
            self._handle_name_command(argument)
            return True
        if command == "/hotkeys":
            self.editor.set_text("")
            self._handle_hotkeys_command()
            return True
        if command == "/new":
            self.editor.set_text("")
            await self._handle_clear_command()
            return True
        if command == "/compact":
            self.editor.set_text("")
            await self._handle_compact_command(argument or None)
            return True
        if command == "/quit":
            self.editor.set_text("")
            await self.shutdown()
            return True
        if command in self._UNSUPPORTED_COMMANDS:
            self.editor.set_text("")
            self._unsupported_command(command)
            return True
        return False

    def _handle_name_command(self, argument: str) -> None:
        if not argument:
            current = self.session_manager.get_session_name()
            self.show_status(f"Session name: {current}" if current else "Session has no name")
            return
        self.session.set_session_name(argument)
        normalized = self.session_manager.get_session_name()
        if normalized != argument:
            self.show_warning(f"Session name was normalized from {argument!r} to {normalized!r}")
        self._update_terminal_title()
        self.show_status(f"Session name set: {normalized or argument}")

    async def _handle_copy_command(self, flash_confirmation: bool = False) -> None:
        text = self.session.get_last_assistant_text()
        if not text:
            self.show_error("No agent messages to copy yet.")
            return
        try:
            await copy_to_clipboard(text)
        except Exception as error:
            self.show_error(str(error))
            return
        if flash_confirmation and isinstance(self.ui, TuiAltScreen):
            self.ui.flash("Copied!")
        else:
            self.show_status("Copied last agent message to clipboard")

    async def _handle_right_click_paste(self) -> None:
        target = self.renderer.get_focused_component()
        handle_input = getattr(target, "handle_input", None)
        if target is None or handle_input is None:
            return
        try:
            text = await read_clipboard_text()
            if not text or self.renderer.get_focused_component() is not target:
                return
            handle_input(f"\x1b[200~{text}\x1b[201~")
            self.ui.request_render()
        except Exception:
            # Silently ignore clipboard errors (may not have permission, etc.)
            return

    # -- auth ---------------------------------------------------------------

    def get_login_provider_options(self, auth_type: str | None = None) -> list[AuthSelectorProvider]:
        """Every provider/auth-method pair that can be logged into."""
        options: list[AuthSelectorProvider] = []
        runtime = self.session.model_runtime
        for provider in runtime.get_providers():
            check = runtime.get_provider_auth_status(provider.id)
            status = (
                AuthCheck(
                    configured=True,
                    type="oauth" if runtime.is_using_oauth(provider.id) else "api_key",
                    source=check.source,
                )
                if check.configured
                else None
            )
            oauth = provider.auth.oauth
            api_key = provider.auth.api_key
            if (auth_type is None or auth_type == "oauth") and oauth is not None:
                options.append(
                    AuthSelectorProvider(
                        id=provider.id,
                        name=provider.name,
                        auth_type="oauth",
                        method=oauth,
                        status=status,
                    )
                )
            if (auth_type is None or auth_type == "api_key") and api_key is not None:
                options.append(
                    AuthSelectorProvider(
                        id=provider.id,
                        name=provider.name,
                        auth_type="api_key",
                        method=api_key,
                        status=status,
                    )
                )
        return sorted(options, key=lambda option: option.name)

    def _find_login_provider_options(self, provider_ref: str) -> list[AuthSelectorProvider]:
        reference = provider_ref.strip().lower()
        return [
            option
            for option in self.get_login_provider_options()
            if option.id.lower() == reference or option.name.lower() == reference
        ]

    async def _handle_login_command(self, provider_ref: str | None) -> None:
        if not provider_ref:
            self.show_oauth_selector("login")
            return

        options = self._find_login_provider_options(provider_ref)
        if len(options) == 1:
            await self._start_provider_login(options[0])
            return
        self.show_oauth_selector("login", provider_ref)

    def show_oauth_selector(self, mode: str, initial_search: str | None = None) -> None:
        if mode == "logout":
            providers = [
                option
                for option in self.get_login_provider_options()
                if option.status is not None and option.status.configured
            ]
        else:
            providers = self.get_login_provider_options()

        def select(provider_id: str, auth_type: str) -> None:
            self._hide_selector()
            option = next(
                (
                    candidate
                    for candidate in providers
                    if candidate.id == provider_id and candidate.auth_type == auth_type
                ),
                None,
            )
            if option is None:
                return
            if mode == "logout":
                spawn(self._logout_provider(option))
            else:
                spawn(self._start_provider_login(option))

        self._show_selector(
            OAuthSelectorComponent(
                "logout" if mode == "logout" else "login",
                providers,
                select,
                self._hide_selector,
                initial_search,
            )
        )

    async def _logout_provider(self, option: AuthSelectorProvider) -> None:
        try:
            await self.session.model_runtime.logout(option.id)
        except Exception as error:
            self.show_error(f"Failed to log out of {option.name}: {error}")
            return
        self._update_available_provider_count()
        self.footer.invalidate()
        self.show_status(f"Logged out of {option.name}")
        self.ui.request_render()

    async def _start_provider_login(self, option: AuthSelectorProvider) -> None:
        dialog = LoginDialogComponent(self.ui, option.id, lambda *_args: None, option.name)
        self._show_selector(dialog)
        try:
            await self._login_provider(dialog, option.id, option.auth_type)
        except LoginCancelledError:
            self._hide_selector()
            self.show_status("Login cancelled")
            return
        except Exception as error:
            self._hide_selector()
            message = str(error)
            if message != "Login cancelled":
                self.show_error(f"Failed to login to {option.name}: {message}")
            return

        self._hide_selector()
        self._update_available_provider_count()
        self.footer.invalidate()
        self.show_status(f"Logged in to {option.name}")
        self.ui.request_render()

    async def _login_provider(self, dialog: LoginDialogComponent, provider_id: str, method: str) -> None:
        interaction = _DialogAuthInteraction(self, dialog)
        if method == "oauth":
            await self.session.model_runtime.login_oauth(provider_id, interaction)
            return
        api_key = await self._show_auth_prompt(
            dialog, AuthPrompt(type="secret", message=f"Enter your {provider_id} API key")
        )
        await self.session.model_runtime.login(provider_id, api_key)

    async def _show_auth_prompt(self, dialog: LoginDialogComponent, prompt: AuthPrompt) -> str:
        if prompt.type == "select":
            response = dialog.show_prompt(prompt.message, prompt.placeholder)
        elif prompt.type == "manual_code":
            response = dialog.show_manual_input(prompt.message)
        else:
            response = dialog.show_prompt(prompt.message, prompt.placeholder)

        signal = getattr(prompt, "signal", None)
        if signal is None:
            return await response
        if signal.aborted:
            raise LoginCancelledError("Login cancelled")
        return await race_with_abort_signal(response, signal)

    def _notify_auth_dialog(self, dialog: LoginDialogComponent, event: AuthEvent) -> None:
        if event.type == "auth_url":
            dialog.show_auth(event.url or "", event.instructions)
        elif event.type == "device_code":
            dialog.show_device_code(event)
            dialog.show_waiting("Waiting for authentication...")
        elif event.type == "info":
            dialog.show_info(event.message or "", event.links)
        else:
            dialog.show_progress(event.message or "")

    def _handle_session_command(self) -> None:
        stats = self.session.get_session_stats()
        name = self.session_manager.get_session_name()
        lines = [theme.bold("Session")]
        if name:
            lines.append(f"  {theme.fg('muted', 'Name:')} {name}")
        lines.append(f"  {theme.fg('muted', 'Id:')} {self.session_manager.get_session_id()}")
        lines.append(f"  {theme.fg('muted', 'Cwd:')} {self.session_manager.get_cwd()}")
        session_file = self.session_manager.get_session_file()
        if session_file:
            lines.append(f"  {theme.fg('muted', 'File:')} {session_file}")
        model = self.session.model
        if model is not None:
            lines.append(f"  {theme.fg('muted', 'Model:')} {model.provider}/{model.id}")
        for label, value in (
            ("Messages", getattr(stats, "message_count", None)),
            ("Input tokens", getattr(stats, "input_tokens", None)),
            ("Output tokens", getattr(stats, "output_tokens", None)),
            ("Cost", getattr(stats, "cost", None)),
        ):
            if value is not None:
                lines.append(f"  {theme.fg('muted', label + ':')} {value}")
        self._append_status("\n".join(lines))

    async def _handle_export_command(self, text: str) -> None:
        output_path = self._get_path_command_argument(text, "/export")
        try:
            if output_path and output_path.endswith(".jsonl"):
                file_path = self.session.export_to_jsonl(output_path)
            else:
                file_path = await self.session.export_to_html(output_path)
        except Exception as error:
            self.show_error(f"Failed to export session: {error or 'Unknown error'}")
            return
        self.show_status(f"Session exported to: {file_path}")

    def _get_path_command_argument(self, text: str, command: str) -> str | None:
        """Extract a single path argument from ``/export`` / ``/import``.

        Quoted arguments keep their spaces and lose their quotes; unquoted ones
        stop at the first whitespace, so an apostrophe inside a bare path (
        ``john's/session.jsonl``) survives.
        """
        if text == command or not text.startswith(f"{command} "):
            return None
        args_string = text[len(command) + 1 :].lstrip()
        if not args_string:
            return None
        first_char = args_string[0]
        if first_char in ('"', "'"):
            closing = args_string.find(first_char, 1)
            if closing < 0:
                return None
            return args_string[1:closing]
        match = re.search(r"\s", args_string)
        return args_string if match is None else args_string[: match.start()]

    async def _show_confirm(self, title: str, message: str) -> bool:
        """Yes/No dialog. Mirrors upstream ``showExtensionConfirm``."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()

        def answer(confirmed: bool) -> None:
            self._hide_selector()
            if not future.done():
                future.set_result(confirmed)

        self._show_selector(ConfirmSelectorComponent(title, message, answer, lambda: answer(False)))
        return await future

    async def _prompt_for_missing_session_cwd(self, error: MissingSessionCwdError) -> str | None:
        confirmed = await self._show_confirm("Session cwd not found", format_missing_session_cwd_prompt(error.issue))
        return error.issue.fallbackCwd if confirmed else None

    async def _handle_import_command(self, text: str) -> None:
        input_path = self._get_path_command_argument(text, "/import")
        if not input_path:
            self.show_error("Usage: /import <path.jsonl>")
            return
        confirmed = await self._show_confirm("Import session", f"Replace current session with {input_path}?")
        if not confirmed:
            self.show_status("Import cancelled")
            return
        try:
            self._clear_status_indicator()
            result = await self.runtime_host.import_from_jsonl(input_path)
        except MissingSessionCwdError as error:
            selected_cwd = await self._prompt_for_missing_session_cwd(error)
            if not selected_cwd:
                self.show_status("Import cancelled")
                return
            # TS does not guard this retry call either; a failure here is not
            # a recognized recoverable case and propagates like any other
            # unexpected error below.
            result = await self.runtime_host.import_from_jsonl(input_path, selected_cwd)
        except SessionImportFileNotFoundError as error:
            self.show_error(f"Failed to import session: {error}")
            return
        if result.get("cancelled"):
            self.show_status("Import cancelled")
            return
        self.footer.set_session(self.session)
        self._rebuild_chat_from_session()
        self._update_terminal_title()
        self.show_status(f"Session imported from: {input_path}")

    async def _handle_clone_command(self) -> None:
        leaf_id = self.session_manager.get_leaf_id()
        if not leaf_id:
            self.show_status("Nothing to clone yet")
            return
        try:
            result = await self.runtime_host.fork(leaf_id, "at")
        except Exception as error:
            self.show_error(str(error))
            return
        if result.get("cancelled"):
            self.ui.request_render()
            return
        self.editor.set_text("")
        self.show_status("Cloned to new session")

    async def _handle_share_command(self) -> None:
        """Export the transcript to HTML and upload it as a secret gist."""
        if shutil.which("gh") is None:
            self.show_error("GitHub CLI (gh) is not installed. Install it from https://cli.github.com/")
            return
        auth = await asyncio.to_thread(subprocess.run, ["gh", "auth", "status"], capture_output=True, check=False)
        if auth.returncode != 0:
            self.show_error("GitHub CLI is not logged in. Run 'gh auth login' first.")
            return

        temp_file = os.path.join(tempfile.gettempdir(), "session.html")
        try:
            await self.session.export_to_html(temp_file)
        except Exception as error:
            self.show_error(f"Failed to export session: {error or 'Unknown error'}")
            return

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                ["gh", "gist", "create", "--public=false", temp_file],
                capture_output=True,
                check=False,
            )
        except OSError as error:
            self.show_error(f"Failed to create gist: {error}")
            return
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp_file)

        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip() or "Unknown error"
            self.show_error(f"Failed to create gist: {message}")
            return

        gist_url = completed.stdout.decode("utf-8", errors="replace").strip()
        gist_id = gist_url.rsplit("/", 1)[-1] if gist_url else ""
        if not gist_id:
            self.show_error("Failed to parse gist ID from gh output")
            return
        self.show_status(f"Share URL: {get_share_viewer_url(gist_id)}\nGist: {gist_url}")

    async def _handle_reload_command(self) -> None:
        """Reload keybindings, settings and resources.

        TypeScript also reloads extensions and themes through
        ``session.reload``; neither the extension host nor resource-loaded
        themes exist in this port (see their module docstrings), so this
        reloads the parts that do.
        """
        if self.session.is_streaming:
            self.show_warning("Wait for the current response to finish before reloading.")
            return
        if self.session.is_compacting:
            self.show_warning("Wait for compaction to finish before reloading.")
            return

        self.keybindings.reload()
        set_keybindings(self.keybindings)
        try:
            await self.settings_manager.reload()
        except Exception as error:
            self.show_error(f"Failed to reload settings: {error}")
            return
        self.session.resource_loader.reload()

        self._apply_runtime_settings()
        self._setup_autocomplete_provider()
        self._update_available_provider_count()
        await self.theme_controller.apply_from_settings()
        self.ui.invalidate()
        self._update_editor_border_color()
        self._rebuild_chat_from_session()
        self.show_status("Reloaded keybindings, settings, skills, prompts, and context files")

    def _handle_changelog_command(self) -> None:
        entries = parse_changelog(get_changelog_path())
        if entries:
            markdown = "\n\n".join(normalize_changelog_links(entry.content, entry) for entry in reversed(entries))
        else:
            markdown = "No changelog entries found."

        self.chat_container.add_child(Spacer(1))
        self.chat_container.add_child(DynamicBorder())
        self.chat_container.add_child(Text(theme.bold(theme.fg("accent", "What's New")), 1, 0))
        self.chat_container.add_child(Spacer(1))
        self.chat_container.add_child(Markdown(markdown, 1, 1, self._markdown_theme()))
        self.chat_container.add_child(DynamicBorder())
        self.ui.request_render()

    def _handle_debug_command(self) -> None:
        width = self.ui.terminal.columns
        height = self.ui.terminal.rows
        all_lines = self.ui.render(width)

        debug_log_path = get_debug_log_path()
        parts = [
            f"Debug output at {datetime.now(tz=UTC).isoformat()}",
            f"Terminal: {width}x{height}",
            f"Total lines: {len(all_lines)}",
            "",
            "=== All rendered lines with visible widths ===",
            *(f"[{index}] (w={visible_width(line)}) {json.dumps(line)}" for index, line in enumerate(all_lines)),
            "",
            "=== Agent messages (JSONL) ===",
            *(json.dumps(_debug_encode(message)) for message in self.session.messages),
            "",
        ]

        Path(debug_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(debug_log_path).write_text("\n".join(parts), encoding="utf-8")

        self.chat_container.add_child(Spacer(1))
        self.chat_container.add_child(
            Text(
                theme.fg("accent", "✓ Debug log written") + "\n" + theme.fg("muted", debug_log_path),
                1,
                1,
            )
        )
        self.ui.request_render()

    def _handle_hotkeys_command(self) -> None:
        lines = [theme.bold("Keybindings"), ""]
        for name, keys in self.keybindings.get_effective_config().items():
            rendered = keys if isinstance(keys, str) else "/".join(keys)
            if rendered:
                lines.append(f"  {theme.fg('dim', rendered)}  {theme.fg('muted', name)}")
        self._append_status("\n".join(lines))

    async def _handle_clear_command(self) -> None:
        self.chat_container.clear()
        self.pending_tools.clear()
        self.streaming_component = None
        self.streaming_message = None
        try:
            await self.runtime_host.new_session()
        except Exception as error:
            self.show_error(f"Failed to start a new session: {error}")
            return
        self.footer.set_session(self.session)
        self._update_terminal_title()
        self.show_status("Started a new session")

    async def _handle_compact_command(self, custom_instructions: str | None) -> None:
        try:
            await self.session.compact(custom_instructions)
        except Exception as error:
            self.show_error(f"Compaction failed: {error}")

    # -- bash ---------------------------------------------------------------

    async def _handle_bash_command(self, command: str, exclude_from_context: bool) -> None:
        event_result = await self.session.extension_runner.emit_user_bash(
            UserBashEvent(
                command=command,
                exclude_from_context=exclude_from_context,
                cwd=self.session_manager.get_cwd(),
            )
        )

        # A handler returning a complete result replaces execution entirely.
        if event_result is not None and event_result.result is not None:
            result = event_result.result
            component = BashExecutionComponent(command, self.ui, exclude_from_context)
            self.bash_component = component
            self._add_bash_component(component)
            if result.output:
                component.append_output(result.output)
            component.set_complete(result.exit_code, result.cancelled, None, result.full_output_path)
            self.session.record_bash_result(command, result, exclude_from_context=exclude_from_context)
            self.bash_component = None
            self.ui.request_render()
            return

        component = BashExecutionComponent(command, self.ui, exclude_from_context)
        self.bash_component = component
        is_deferred = self._add_bash_component(component)
        self.ui.request_render()

        def on_chunk(chunk: str) -> None:
            component.append_output(chunk)
            self.ui.request_render()

        try:
            result = await self.session.execute_bash(
                command,
                on_chunk,
                exclude_from_context=exclude_from_context,
                operations=event_result.operations if event_result is not None else None,
            )
            # `BashResult.truncated` is a bool; `set_complete` takes the richer
            # `TruncationResult` the bash tool produces, so pass `None` and let
            # the component's own context truncation drive the warning.
            component.set_complete(
                result.exit_code,
                result.cancelled,
                None,
                result.full_output_path,
            )
        except Exception as error:
            component.set_complete(None, False)
            self.show_error(f"Bash execution failed: {error}")
        finally:
            self.bash_component = None
            if is_deferred:
                self.pending_bash_components.append(component)
            self.ui.request_render()

    def _add_bash_component(self, component: BashExecutionComponent) -> bool:
        """Place a bash component and report whether it went to the pending area.

        Mirrors `handleBashCommand`: while the assistant is streaming the output
        belongs above the editor and is migrated into the transcript later, but
        an idle `!command` is appended straight to the chat -- otherwise its
        output sits in the pending area until the next message is submitted.
        """
        if self.session.is_streaming:
            self.pending_messages_container.add_child(component)
            return True
        self.chat_container.add_child(component)
        return False

    def _queue_compaction_message(self, text: str, mode: Literal["steer", "followUp"]) -> None:
        self.compaction_queued_messages.append(CompactionQueuedMessage(text=text, mode=mode))
        self.editor.add_to_history(text)
        self.editor.set_text("")
        self._update_pending_messages_display()
        self.show_status("Queued message for after compaction")

    def _is_extension_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        return self.session.extension_runner.get_command(command_name) is not None

    async def _flush_compaction_queue(self, *, will_retry: bool = False) -> None:
        if not self.compaction_queued_messages:
            return

        queued_messages = list(self.compaction_queued_messages)
        self.compaction_queued_messages = []
        self._update_pending_messages_display()

        def restore_queue(error: BaseException) -> None:
            self.session.clear_queue()
            self.compaction_queued_messages = queued_messages
            self._update_pending_messages_display()
            plural = "s" if len(queued_messages) > 1 else ""
            self.show_error(f"Failed to send queued message{plural}: {error}")

        async def dispatch(message: CompactionQueuedMessage) -> None:
            if self._is_extension_command(message.text):
                await self.session.prompt(message.text)
            elif message.mode == "followUp":
                await self.session.follow_up(message.text)
            else:
                await self.session.steer(message.text)

        try:
            if will_retry:
                # When retry is pending, queue messages for the retry turn.
                for message in queued_messages:
                    await dispatch(message)
                self._update_pending_messages_display()
                return

            first_prompt_index = next(
                (
                    index
                    for index, message in enumerate(queued_messages)
                    if not self._is_extension_command(message.text)
                ),
                -1,
            )
            if first_prompt_index == -1:
                for message in queued_messages:
                    await self.session.prompt(message.text)
                return

            pre_commands = queued_messages[:first_prompt_index]
            first_prompt = queued_messages[first_prompt_index]
            rest = queued_messages[first_prompt_index + 1 :]

            for message in pre_commands:
                await self.session.prompt(message.text)

            # Start a prompt when idle, or queue it into a run still finishing compaction.
            async def run_first_prompt() -> None:
                try:
                    await self.session.prompt(first_prompt.text, streaming_behavior=first_prompt.mode)
                except Exception as error:
                    restore_queue(error)

            prompt_task = spawn(run_first_prompt())

            for message in rest:
                await dispatch(message)
            self._update_pending_messages_display()
            # `void promptPromise` in TypeScript: JS promises start running
            # synchronously, so yield once to give the task the same head start.
            if not prompt_task.done():
                await asyncio.sleep(0)
        except Exception as error:
            restore_queue(error)

    def _flush_pending_bash_components(self) -> None:
        for component in self.pending_bash_components:
            self.pending_messages_container.remove_child(component)
            self.chat_container.add_child(component)
        self.pending_bash_components = []

    # -- transcript rendering ----------------------------------------------

    def _render_initial_messages(self) -> None:
        # `renderSessionItems` starts from an empty pending-tool map: only tool
        # calls still unresolved in the replayed transcript may stay registered
        # for a later live completion event.
        self.pending_tools.clear()
        self._render_session_entries(self.session_manager.get_branch())

    def _render_session_entries(self, entries: list[Any]) -> None:
        for entry in entries:
            entry_type = getattr(entry, "type", None)
            if entry_type == "message":
                self._add_message_to_chat(entry.message)
            elif entry_type == "custom":
                self._add_custom_entry_to_chat(entry)

    def _rebuild_chat_from_messages(self) -> None:
        self.chat_container.clear()
        self._render_session_entries(self.session_manager.build_context_entries())

    def _add_custom_entry_to_chat(self, entry: Any) -> None:
        renderer = self._get_entry_renderer(entry.custom_type)
        if renderer is None:
            return
        component = CustomEntryComponent(entry, renderer)
        if component.has_content():
            self.chat_container.add_child(component)

    def _get_entry_renderer(self, _custom_type: str) -> Any:
        # Extension-registered entry renderers are part of the unported
        # extension UI host; without one there is nothing to draw.
        return None

    def _add_message_to_chat(self, message: Any) -> None:
        role = getattr(message, "role", None)
        if role == "user":
            text = self._get_user_message_text(message)
            if text:
                self.chat_container.add_child(UserMessageComponent(text, self._markdown_theme(), self.output_pad))
        elif role == "assistant":
            component = AssistantMessageComponent(
                message,
                self.hide_thinking_block,
                self._markdown_theme(),
                self.hidden_thinking_label,
                self.output_pad,
            )
            self.chat_container.add_child(component)
            self._render_tool_calls(message, live=False)
        elif role == "custom":
            # `message.display` gates rendering upstream (`interactive-mode.ts:3476`);
            # a custom message that opts out of display must not occupy a row.
            if getattr(message, "display", True):
                component = CustomMessageComponent(message, None, self._markdown_theme(), self.output_pad)
                component.set_expanded(self.tool_output_expanded)
                self.chat_container.add_child(component)
        elif role == "branchSummary":
            # Summaries are separated from the preceding turn by a blank row
            # (`interactive-mode.ts:3496`); without it they butt against the
            # message above.
            self.chat_container.add_child(Spacer(1))
            component = BranchSummaryMessageComponent(message, self._markdown_theme())
            component.set_expanded(self.tool_output_expanded)
            self.chat_container.add_child(component)
        elif role == "compactionSummary":
            self.chat_container.add_child(Spacer(1))
            component = CompactionSummaryMessageComponent(message, self._markdown_theme())
            component.set_expanded(self.tool_output_expanded)
            self.chat_container.add_child(component)
        elif role == "bashExecution":
            component = BashExecutionComponent(message.command, self.ui, message.exclude_from_context)
            component.append_output(message.output)
            component.set_complete(message.exit_code, message.cancelled)
            self.chat_container.add_child(component)
        elif role == "toolResult":
            self._apply_tool_result(message)

    def _get_user_message_text(self, message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return "\n".join(part.text for part in content if getattr(part, "type", None) == "text")

    def _tool_execution_options(self) -> ToolExecutionOptions:
        return ToolExecutionOptions(
            show_images=self.settings_manager.get_show_images(),
            image_width_cells=self.settings_manager.get_image_width_cells(),
        )

    def _render_tool_calls(self, message: AssistantMessage, live: bool) -> None:
        for content in message.content:
            if getattr(content, "type", None) != "toolCall":
                continue
            existing = self.pending_tools.get(content.id)
            if existing is None:
                component = ToolExecutionComponent(
                    content.name,
                    content.id,
                    content.arguments,
                    self._tool_execution_options(),
                    None,
                    self.ui,
                    self.session_manager.get_cwd(),
                )
                component.set_expanded(self.tool_output_expanded)
                self.chat_container.add_child(component)
                self.pending_tools[content.id] = component
            elif live:
                existing.update_args(content.arguments)

    def _apply_tool_result(self, message: Any) -> None:
        tool_call_id = getattr(message, "tool_call_id", "")
        component = self.pending_tools.get(tool_call_id)
        if component is None:
            return
        component.update_result(
            ToolResult(
                content=list(getattr(message, "content", []) or []),
                is_error=bool(getattr(message, "is_error", False)),
                details=getattr(message, "details", None),
            ),
            is_partial=False,
        )
        # A resolved call is no longer pending (TS: `renderedPendingTools.delete`).
        self.pending_tools.pop(tool_call_id, None)

    # -- agent events -------------------------------------------------------

    def _subscribe_to_agent(self) -> None:
        self._unsubscribe = self.session.subscribe(lambda event: spawn(self._handle_event(event)))

    async def _handle_event(self, event: Any) -> None:
        if not self.is_initialized:
            await self.init()
        self.footer.invalidate()
        event_type = getattr(event, "type", None)

        if event_type == "agent_start":
            self.pending_tools.clear()
            if self.working_visible:
                self._show_status_indicator(WorkingStatusIndicator(self.ui, self.default_working_message))
            else:
                self._clear_status_indicator()
        elif event_type == "queue_update":
            self._update_pending_messages_display()
        elif event_type == "entry_appended":
            if getattr(event.entry, "type", None) == "custom":
                self._add_custom_entry_to_chat(event.entry)
        elif event_type == "session_info_changed":
            self._update_terminal_title()
        elif event_type == "thinking_level_changed":
            self._update_editor_border_color()
        elif event_type == "message_start":
            self._handle_message_start(event.message)
        elif event_type == "message_update":
            self._handle_message_update(event.message)
        elif event_type == "message_end":
            self._handle_message_end(event.message)
        elif event_type == "tool_execution_update":
            component = self.pending_tools.get(getattr(event, "tool_call_id", ""))
            if component is not None:
                partial = getattr(event, "partial_result", None)
                component.update_result(
                    ToolResult(
                        content=list(getattr(partial, "content", []) or []),
                        is_error=False,
                        details=getattr(partial, "details", None),
                    ),
                    is_partial=True,
                )
        elif event_type == "tool_execution_end":
            tool_call_id = getattr(event, "tool_call_id", "")
            component = self.pending_tools.get(tool_call_id)
            if component is not None:
                result = getattr(event, "result", None)
                component.update_result(
                    ToolResult(
                        content=list(getattr(result, "content", []) or []),
                        is_error=bool(getattr(event, "is_error", False)),
                        details=getattr(result, "details", None),
                    ),
                    is_partial=False,
                )
                self.pending_tools.pop(tool_call_id, None)
        elif event_type == "agent_end":
            self._clear_status_indicator()
        elif event_type == "compaction_start":
            if self.settings_manager.get_show_terminal_progress():
                self.ui.terminal.set_progress(True)
            # Keep the editor active; submissions are queued during compaction.
            self.auto_compaction_escape_handler = self.default_editor.on_escape
            self.default_editor.on_escape = self.session.abort_compaction
            self._show_status_indicator(CompactionStatusIndicator(self.ui, getattr(event, "reason", "manual")))
        elif event_type == "compaction_end":
            if self.settings_manager.get_show_terminal_progress():
                self.ui.terminal.set_progress(False)
            if self.auto_compaction_escape_handler is not None:
                self.default_editor.on_escape = self.auto_compaction_escape_handler
                self.auto_compaction_escape_handler = None
            self._clear_status_indicator("compaction")
            reason = getattr(event, "reason", "manual")
            result = getattr(event, "result", None)
            error_message = getattr(event, "error_message", None)
            if getattr(event, "aborted", False):
                if reason == "manual":
                    self.show_error("Compaction cancelled")
                else:
                    self.show_status("Auto-compaction cancelled")
            elif result is not None:
                self.chat_container.clear()
                self._rebuild_chat_from_messages()
                self._add_message_to_chat(
                    create_compaction_summary_message(
                        result.summary,
                        result.tokens_before,
                        int(datetime.now(UTC).timestamp() * 1000),
                    )
                )
                self.footer.invalidate()
            elif error_message:
                if reason == "manual":
                    self.show_error(error_message)
                else:
                    self.chat_container.add_child(Spacer(1))
                    self.chat_container.add_child(Text(theme.fg("error", error_message), 1, 0))
            spawn(self._flush_compaction_queue(will_retry=bool(getattr(event, "will_retry", False))))
        elif event_type == "auto_retry_start":
            self._show_status_indicator(
                RetryStatusIndicator(
                    self.ui,
                    getattr(event, "attempt", 1),
                    getattr(event, "max_attempts", 1),
                    getattr(event, "delay_ms", 0),
                )
            )
        elif event_type == "auto_retry_end":
            self._clear_status_indicator("retry")
        elif event_type == "summarization_retry_attempt_start":
            self._show_status_indicator(BranchSummaryStatusIndicator(self.ui))
        elif event_type == "summarization_retry_finished":
            self._clear_status_indicator("branchSummary")
        elif event_type == "bash_execution_update":
            if self.bash_component is not None:
                self.bash_component.append_output(getattr(event, "chunk", ""))

        self.ui.request_render()

    def _handle_message_start(self, message: Any) -> None:
        role = getattr(message, "role", None)
        if role == "assistant":
            self.streaming_component = AssistantMessageComponent(
                None,
                self.hide_thinking_block,
                self._markdown_theme(),
                self.hidden_thinking_label,
                self.output_pad,
            )
            self.streaming_message = message
            self.chat_container.add_child(self.streaming_component)
            self.streaming_component.update_content(message, True)
        else:
            self._add_message_to_chat(message)
            if role == "user":
                self._update_pending_messages_display()

    def _handle_message_update(self, message: Any) -> None:
        if self.streaming_component is None or getattr(message, "role", None) != "assistant":
            return
        self.streaming_message = message
        self.streaming_component.update_content(message, True)
        self._render_tool_calls(message, live=True)

    def _handle_message_end(self, message: Any) -> None:
        role = getattr(message, "role", None)
        if role == "user":
            return
        if role == "toolResult":
            self._apply_tool_result(message)
            return
        if self.streaming_component is not None and role == "assistant":
            self.streaming_message = message
            self.streaming_component.update_content(message, False)
            stop_reason = getattr(message, "stop_reason", None)
            if stop_reason in ("aborted", "error"):
                error_message = getattr(message, "error_message", None) or "Error"
                for component in self.pending_tools.values():
                    component.update_result(
                        ToolResult(content=[{"type": "text", "text": error_message}], is_error=True)
                    )
            self.streaming_component = None
            self.streaming_message = None

    def _update_pending_messages_display(self) -> None:
        self.ui.request_render()

    def _clear_all_queues(self) -> tuple[list[str], list[str]]:
        """Drain both the session queues and the compaction hold queue.

        Port of `clearAllQueues`. The compaction-queued messages carry their own
        `steer`/`followUp` mode and must be merged into the matching list.
        """
        snapshot = self.session.clear_queue()
        compaction_steering = [msg.text for msg in self.compaction_queued_messages if msg.mode == "steer"]
        compaction_follow_up = [msg.text for msg in self.compaction_queued_messages if msg.mode == "followUp"]
        self.compaction_queued_messages = []
        return ([*snapshot.steering, *compaction_steering], [*snapshot.follow_up, *compaction_follow_up])

    def _restore_queued_messages_to_editor(self, *, abort: bool = False, current_text: str | None = None) -> int:
        """Move every queued message back into the editor, newest text last.

        Port of `restoreQueuedMessagesToEditor`. Aborting a stream drops the
        queues, so without this the user silently loses whatever they typed
        while the assistant was answering.
        """
        steering, follow_up = self._clear_all_queues()
        all_queued = [*steering, *follow_up]
        if not all_queued:
            self._update_pending_messages_display()
            if abort:
                self.session.agent.abort()
            return 0
        queued_text = "\n\n".join(all_queued)
        text = self.editor.get_text() if current_text is None else current_text
        combined = "\n\n".join(part for part in (queued_text, text) if part.strip())
        self.editor.set_text(combined)
        self._update_pending_messages_display()
        if abort:
            self.session.agent.abort()
        return len(all_queued)

    async def _handle_follow_up(self) -> None:
        """Port of `handleFollowUp` (the `app.message.followUp` action).

        Queues the editor text as a follow-up instead of a steering message;
        when nothing is streaming it behaves like a plain Enter.
        """
        text = self.editor.get_expanded_text().strip()
        if not text:
            return

        if self.session.is_compacting:
            if self._is_extension_command(text):
                self.editor.add_to_history(text)
                self.editor.set_text("")
                await self._prompt_safely(text)
            else:
                self._queue_compaction_message(text, "followUp")
            return

        if self.session.is_streaming:
            self.editor.add_to_history(text)
            self.editor.set_text("")
            await self._prompt_safely(text, streaming_behavior="followUp")
            self._update_pending_messages_display()
            self.ui.request_render()
        elif self.editor.on_submit is not None:
            self.editor.set_text("")
            self.editor.on_submit(text)

    def _handle_dequeue(self) -> None:
        """Port of `handleDequeue` (the `app.message.dequeue` action)."""
        restored = self._restore_queued_messages_to_editor()
        if restored == 0:
            self.show_status("No queued messages to restore")
        else:
            plural = "s" if restored > 1 else ""
            self.show_status(f"Restored {restored} queued message{plural} to editor")

    async def _handle_open_external_editor(self) -> None:
        """Port of `handleOpenExternalEditor` (the `app.editor.external` action)."""
        content = self.editor.get_expanded_text()
        self.ui.stop()
        try:
            result = await edit_in_external_editor(
                ExternalEditorOptions(
                    command=self.settings_manager.get_external_editor_command(),
                    content=content,
                )
            )
            if result.status == "complete":
                self.editor.set_text(result.content)
        finally:
            self.ui.start()
            self.ui.request_render(True)

    # -- toggles ------------------------------------------------------------

    def _toggle_tool_output_expansion(self) -> None:
        self.set_tools_expanded(not self.tool_output_expanded)

    def set_tools_expanded(self, expanded: bool) -> None:
        """Expand or collapse every expandable transcript entry.

        The loaded-resources container is walked too: its sections share the
        same expansion state as tool output, so toggling must reach them even
        though they live outside the chat.

        Only the direct children of each container are visited, matching
        TypeScript's `setToolsExpanded`, which iterates `container.children`
        without descending. Both ports build these two containers flat, so a
        nested expandable is not a transcript entry and must not be toggled.
        """
        if expanded == self.tool_output_expanded:
            return

        self.tool_output_expanded = expanded
        for container in (self.loaded_resources_container, self.chat_container):
            for component in container.children:
                setter = getattr(component, "set_expanded", None)
                if setter is not None:
                    setter(expanded)
        self.show_status(f"Tool output: {'expanded' if expanded else 'collapsed'}")
        self.ui.request_render()

    def _iter_components(self, container: Container) -> list[Component]:
        result: list[Component] = []
        for child in container.children:
            result.append(child)
            if isinstance(child, Container):
                result.extend(self._iter_components(child))
        return result

    def _toggle_thinking_block_visibility(self) -> None:
        self.hide_thinking_block = not self.hide_thinking_block
        self.settings_manager.set_hide_thinking_block(self.hide_thinking_block)
        for component in self._iter_components(self.chat_container):
            if isinstance(component, AssistantMessageComponent):
                component.set_hide_thinking_block(self.hide_thinking_block)
        self.ui.request_render()

    def _cycle_thinking_level(self) -> None:
        new_level = self.session.cycle_thinking_level()
        if new_level is None:
            self.show_status("Current model does not support thinking")
            return
        self.footer.invalidate()
        self._update_editor_border_color()
        self.show_status(f"Thinking level: {new_level}")

    async def _handle_model_command(self, search_term: str | None) -> None:
        """Port of `interactive-mode.ts`'s `handleModelCommand`.

        `/model <term>` switches directly when `term` names exactly one model
        that is already available, and only falls back to the selector when it
        does not.
        """
        if not search_term:
            self.show_model_selector()
            return

        model = self._find_exact_model_match(search_term)
        if model is not None:
            try:
                await self.session.set_model(model)
                self.footer.invalidate()
                self._update_editor_border_color()
                self.show_status(f"Model: {model.id}")
                spawn(self._maybe_warn_about_anthropic_subscription_auth(model))
            except Exception as error:
                self.show_error(str(error))
            return

        self.show_model_selector(search_term)

    def _find_exact_model_match(self, search_term: str) -> Model | None:
        """Port of `findExactModelMatch`, cached-snapshot half only.

        TypeScript falls back to `modelRuntime.refresh({ signal })` behind a
        15s deadline when the cached snapshot has no match, showing
        "Refreshing model catalogs…" first. This port has no remote catalog
        refresh (see `core/model_runtime.py`'s module docstring), so a cache
        miss goes straight to the selector; the cache-first behaviour that
        issue #7443 is about is preserved exactly.
        """
        scoped_models = self.session.scoped_models
        if scoped_models:
            cached_models = [scoped.model for scoped in scoped_models]
        else:
            cached_models = list(self.session.model_runtime.get_available_snapshot())
        return find_exact_model_reference_match(search_term, cached_models)

    async def _cycle_model(self, direction: str) -> None:
        try:
            result = await self.session.cycle_model(direction)
        except Exception as error:
            self.show_error(str(error))
            return
        if result is None:
            self.show_status("Only one model in scope" if self.session.scoped_models else "Only one model available")
            return
        self.footer.invalidate()
        self._update_editor_border_color()
        thinking = (
            f" (thinking: {result.thinking_level})" if result.model.reasoning and result.thinking_level != "off" else ""
        )
        self.show_status(f"Switched to {result.model.name or result.model.id}{thinking}")
        spawn(self._maybe_warn_about_anthropic_subscription_auth(result.model))

    async def _maybe_warn_about_anthropic_subscription_auth(self, model: Model | None = None) -> None:
        """Warn once that Anthropic subscription auth bills third-party harnesses as extra usage."""
        if self.settings_manager.get_warnings().get("anthropicExtraUsage") is False:
            return
        if self.anthropic_subscription_warning_shown:
            return
        target = model if model is not None else self.session.model
        if target is None or target.provider != "anthropic":
            return

        try:
            check = await self.session.model_runtime.check_auth("anthropic")
            if check is not None and check.type == "oauth":
                self.anthropic_subscription_warning_shown = True
                self.show_warning(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING)
                return
            auth = await self.session.model_runtime.get_auth(target.provider)
            api_key = auth.auth.api_key if auth is not None else None
            if not _is_anthropic_subscription_auth_key(api_key):
                return
            self.anthropic_subscription_warning_shown = True
            self.show_warning(ANTHROPIC_SUBSCRIPTION_AUTH_WARNING)
        except Exception:
            # Ignore auth lookup failures for warning-only checks.
            pass

    # -- selector overlays --------------------------------------------------

    def _show_selector(self, component: Component) -> None:
        self._hide_selector()
        self._active_selector = component
        self.editor_container.clear()
        self.editor_container.add_child(component)
        self.ui.set_focus(component)
        self.ui.request_render()

    def _hide_selector(self) -> None:
        if self._active_selector is None:
            return
        dispose = getattr(self._active_selector, "dispose", None)
        if dispose is not None:
            with contextlib.suppress(Exception):
                dispose()
        self._active_selector = None
        self.editor_container.clear()
        self.editor_container.add_child(self.editor)
        self.ui.set_focus(self.editor)
        self.ui.request_render()

    def show_model_selector(self, initial_search: str | None = None) -> None:
        def select(model: Any) -> None:
            self._hide_selector()
            spawn(self.session.set_model(model))
            self.show_status(f"Model: {model.id}")

        scoped = [
            ScopedModelItem(model=scoped_model.model, thinking_level=scoped_model.thinking_level)
            for scoped_model in self.session.scoped_models
        ]
        self._show_selector(
            ModelSelectorComponent(
                self.ui,
                self.session.model,
                self.settings_manager,
                self.session.model_runtime,
                scoped,
                select,
                self._hide_selector,
                initial_search,
            )
        )

    def show_thinking_selector(self) -> None:
        def select(level: str) -> None:
            self._hide_selector()
            self.session.set_thinking_level(level)
            self._update_editor_border_color()

        self._show_selector(
            ThinkingSelectorComponent(
                self.session.thinking_level or "off",
                self.session.get_available_thinking_levels(),
                select,
                self._hide_selector,
            )
        )

    def show_theme_selector(self) -> None:
        current = self.settings_manager.get_theme()

        def select(name: str) -> None:
            self._hide_selector()
            set_theme(name)
            self.settings_manager.set_theme(name)
            self.ui.invalidate()
            self._update_editor_border_color()

        def cancel() -> None:
            set_theme(current)
            self.ui.invalidate()
            self._hide_selector()

        def preview(name: str) -> None:
            set_theme(name)
            self.ui.invalidate()
            self.ui.request_render()

        self._show_selector(ThemeSelectorComponent(current, select, cancel, preview))

    def show_images_selector(self) -> None:
        def select(show: bool) -> None:
            self._hide_selector()
            self.settings_manager.set_show_images(show)

        self._show_selector(
            ShowImagesSelectorComponent(self.settings_manager.get_show_images(), select, self._hide_selector)
        )

    def show_settings_selector(self) -> None:
        settings = self.settings_manager
        config = SettingsConfig(
            auto_compact=self.session.auto_compaction_enabled,
            show_images=settings.get_show_images(),
            image_width_cells=settings.get_image_width_cells(),
            auto_resize_images=settings.get_image_auto_resize(),
            block_images=settings.get_block_images(),
            enable_skill_commands=settings.get_enable_skill_commands(),
            steering_mode=self.session.steering_mode,
            follow_up_mode=self.session.follow_up_mode,
            transport=settings.get_transport(),
            http_idle_timeout_ms=settings.get_http_idle_timeout_ms(),
            thinking_level=self.session.thinking_level or "off",
            available_thinking_levels=self.session.get_available_thinking_levels(),
            current_theme=self.theme_controller.get_theme_selection() or "dark",
            terminal_theme=self.theme_controller.get_terminal_theme(),
            available_themes=get_available_themes(),
            hide_thinking_block=self.hide_thinking_block,
            mermaid_rendering_mode=settings.get_mermaid_rendering_mode(),
            show_cache_miss_notices=settings.get_show_cache_miss_notices(),
            collapse_changelog=settings.get_collapse_changelog(),
            enable_install_telemetry=settings.get_enable_install_telemetry(),
            double_escape_action=settings.get_double_escape_action(),
            tree_filter_mode=settings.get_tree_filter_mode(),
            show_hardware_cursor=settings.get_show_hardware_cursor(),
            default_project_trust=settings.get_default_project_trust(),
            editor_padding_x=settings.get_editor_padding_x(),
            output_pad=self.output_pad,
            autocomplete_max_visible=settings.get_autocomplete_max_visible(),
            quiet_startup=settings.get_quiet_startup(),
            clear_on_shrink=settings.get_clear_on_shrink(),
            show_terminal_progress=settings.get_show_terminal_progress(),
            tui_mode=self.tui_mode,
            fullscreen_exit_output=settings.get_fullscreen_exit_output(),
            fullscreen_scrollbar=settings.get_fullscreen_scrollbar(),
            warnings=settings.get_warnings(),
        )
        callbacks = SettingsCallbacks(
            on_auto_compact_change=self._set_auto_compact,
            on_show_images_change=self._set_show_images,
            on_image_width_cells_change=self._set_image_width_cells,
            on_auto_resize_images_change=settings.set_image_auto_resize,
            on_block_images_change=settings.set_block_images,
            on_enable_skill_commands_change=self._set_enable_skill_commands,
            on_steering_mode_change=self.session.set_steering_mode,
            on_follow_up_mode_change=self.session.set_follow_up_mode,
            on_transport_change=self._set_transport,
            on_http_idle_timeout_ms_change=self._set_http_idle_timeout_ms,
            on_thinking_level_change=self._set_thinking_level_from_settings,
            on_theme_change=self._apply_theme,
            on_theme_preview=self._preview_theme,
            on_hide_thinking_block_change=self._set_hide_thinking_block,
            on_mermaid_rendering_mode_change=self._set_mermaid_rendering_mode,
            on_show_cache_miss_notices_change=self._set_show_cache_miss_notices,
            on_collapse_changelog_change=settings.set_collapse_changelog,
            on_enable_install_telemetry_change=settings.set_enable_install_telemetry,
            on_double_escape_action_change=settings.set_double_escape_action,
            on_tree_filter_mode_change=settings.set_tree_filter_mode,
            on_show_hardware_cursor_change=self._set_show_hardware_cursor,
            on_editor_padding_x_change=self._set_editor_padding_x,
            on_output_pad_change=self._set_output_pad,
            on_autocomplete_max_visible_change=self._set_autocomplete_max_visible,
            on_quiet_startup_change=settings.set_quiet_startup,
            on_default_project_trust_change=settings.set_default_project_trust,
            on_clear_on_shrink_change=self._set_clear_on_shrink,
            on_show_terminal_progress_change=settings.set_show_terminal_progress,
            # `switchTuiMode` (live renderer swap) is not ported; persist the
            # choice so it applies on the next start instead of dropping it.
            on_tui_mode_change=self._set_tui_mode,
            on_fullscreen_exit_output_change=settings.set_fullscreen_exit_output,
            on_fullscreen_scrollbar_change=self._set_fullscreen_scrollbar,
            on_warnings_change=settings.set_warnings,
            on_cancel=self._hide_selector,
        )
        self._show_selector(SettingsSelectorComponent(config, callbacks))

    def _set_auto_compact(self, enabled: bool) -> None:
        self.session.set_auto_compaction_enabled(enabled)
        self.footer.set_auto_compact_enabled(enabled)

    def _set_show_images(self, enabled: bool) -> None:
        self.settings_manager.set_show_images(enabled)
        for child in self.chat_container.children:
            if isinstance(child, ToolExecutionComponent):
                child.set_show_images(enabled)

    def _set_image_width_cells(self, width: int) -> None:
        self.settings_manager.set_image_width_cells(width)
        for child in self.chat_container.children:
            if isinstance(child, ToolExecutionComponent):
                child.set_image_width_cells(width)

    def _set_enable_skill_commands(self, enabled: bool) -> None:
        self.settings_manager.set_enable_skill_commands(enabled)
        self._setup_autocomplete_provider()

    def _set_transport(self, transport: str) -> None:
        self.settings_manager.set_transport(transport)
        self.session.agent.transport = transport

    def _set_http_idle_timeout_ms(self, timeout_ms: int) -> None:
        self.settings_manager.set_http_idle_timeout_ms(timeout_ms)
        configure_http_dispatcher(timeout_ms)
        self.show_status(f"HTTP idle timeout: {format_http_idle_timeout_ms(timeout_ms)}")

    def _set_thinking_level_from_settings(self, level: str) -> None:
        self.session.set_thinking_level(level)
        self.footer.invalidate()
        self._update_editor_border_color()

    def _set_mermaid_rendering_mode(self, mode: str) -> None:
        self.settings_manager.set_mermaid_rendering_mode(mode)
        self.chat_container.invalidate()
        self.ui.request_render()

    def _set_show_cache_miss_notices(self, shown: bool) -> None:
        self.settings_manager.set_show_cache_miss_notices(shown)
        self._rebuild_chat_from_messages()

    def _set_show_hardware_cursor(self, enabled: bool) -> None:
        self.settings_manager.set_show_hardware_cursor(enabled)
        self.ui.set_show_hardware_cursor(enabled)

    def _set_editor_padding_x(self, padding: int) -> None:
        self.settings_manager.set_editor_padding_x(padding)
        self.default_editor.set_padding_x(padding)
        if self.editor is not self.default_editor:
            set_padding = getattr(self.editor, "set_padding_x", None)
            if set_padding is not None:
                set_padding(padding)

    def _set_output_pad(self, padding: int) -> None:
        self.settings_manager.set_output_pad(padding)
        self.output_pad = padding
        if self.streaming_component is not None or self.session.is_streaming:
            for child in self.chat_container.children:
                if isinstance(child, AssistantMessageComponent | CustomMessageComponent | UserMessageComponent):
                    child.set_output_pad(padding)
            if self.streaming_component is not None:
                self.streaming_component.set_output_pad(padding)
            self.ui.request_render()
            return
        self._rebuild_chat_from_messages()

    def _set_autocomplete_max_visible(self, max_visible: int) -> None:
        self.settings_manager.set_autocomplete_max_visible(max_visible)
        self.default_editor.set_autocomplete_max_visible(max_visible)
        if self.editor is not self.default_editor:
            set_max_visible = getattr(self.editor, "set_autocomplete_max_visible", None)
            if set_max_visible is not None:
                set_max_visible(max_visible)

    def _set_clear_on_shrink(self, enabled: bool) -> None:
        self.settings_manager.set_clear_on_shrink(enabled)
        self.ui.set_clear_on_shrink(enabled)
        if not enabled and self.active_status_indicator is None:
            self.status_container.clear()

    def _set_tui_mode(self, mode: str) -> None:
        # Port of `onTuiModeChange` (`interactive-mode.ts:4557-4566`): the swap
        # is attempted first and the setting is only persisted if it succeeded.
        if not self.switch_tui_mode(mode):
            self.show_status("Close active overlays before changing TUI mode")
            return
        self.settings_manager.set_tui_mode(mode)

    def _set_fullscreen_scrollbar(self, mode: str) -> None:
        self.settings_manager.set_fullscreen_scrollbar(mode)
        self._apply_fullscreen_scrollbar_setting()

    def _apply_fullscreen_scrollbar_setting(self) -> None:
        if self.transcript_scroll_view is not None:
            self.transcript_scroll_view.set_scrollbar(self.settings_manager.get_fullscreen_scrollbar())

    def _set_hide_thinking_block(self, hidden: bool) -> None:
        self.hide_thinking_block = hidden
        self.settings_manager.set_hide_thinking_block(hidden)
        for child in self.chat_container.children:
            if isinstance(child, AssistantMessageComponent):
                child.set_hide_thinking_block(hidden)
        self.chat_container.clear()
        self._rebuild_chat_from_messages()

    def _apply_theme(self, name: str) -> None:
        self.settings_manager.set_theme(name)
        spawn(self.theme_controller.set_theme_setting(name))

    def _preview_theme(self, name: str) -> None:
        self.theme_controller.preview(name)

    def show_user_message_selector(self) -> None:
        messages: list[UserMessageItem] = []
        for entry in self.session_manager.get_branch():
            if getattr(entry, "type", None) != "message":
                continue
            if getattr(entry.message, "role", None) != "user":
                continue
            text = self._get_user_message_text(entry.message)
            if text:
                messages.append(UserMessageItem(id=entry.id, text=text, timestamp=entry.timestamp))

        if not messages:
            self.show_status("No user messages to fork from")
            return

        def select(entry_id: str) -> None:
            self._hide_selector()
            self.show_status(f"Forking from entry {entry_id} is not available in this port yet.")

        self._show_selector(UserMessageSelectorComponent(messages, select, self._hide_selector))

    def show_trust_selector(self) -> None:
        store = ProjectTrustStore(get_agent_dir())
        cwd = self.session_manager.get_cwd()

        def select(selection: Any) -> None:
            self._hide_selector()
            if selection.updates:
                store.set_many(selection.updates)
            self.show_status(f"Project trust: {'trusted' if selection.trusted else 'untrusted'}")

        self._show_selector(TrustSelectorComponent(cwd, store.get_entry(cwd), True, select, self._hide_selector))

    def show_tree_selector(self) -> None:
        tree = self.session_manager.get_tree()
        if not tree:
            self.show_status("Session tree is empty")
            return

        def select(entry_id: str) -> None:
            self._hide_selector()
            spawn(self._navigate_tree(entry_id))

        def label_change(entry_id: str, label: str | None) -> None:
            self.session_manager.append_label_change(entry_id, label)

        selector = TreeSelectorComponent(
            tree,
            self.session_manager.get_leaf_id(),
            self.ui.terminal.rows,
            select,
            self._hide_selector,
            label_change,
            initial_filter_mode=self.settings_manager.get_tree_filter_mode(),
        )
        selector.on_copy = self._copy_to_clipboard
        self._show_selector(selector)

    async def _navigate_tree(self, entry_id: str) -> None:
        # The user committed to navigating: stop the active response first, or
        # `navigate_tree` rejects with "Wait for the current response to finish".
        if self.session.is_streaming:
            self._restore_queued_messages_to_editor()
            await self.session.abort()
        try:
            result = await self.session.navigate_tree(entry_id)
        except Exception as error:
            self.show_error(str(error))
            return
        if result.aborted:
            self.show_status("Branch summarization cancelled")
            return
        if result.cancelled:
            self.show_status("Navigation cancelled")
            return
        self._rebuild_chat_from_session()
        if result.editor_text and not self.editor.get_text().strip():
            self.editor.set_text(result.editor_text)
        self.show_status("Navigated to selected point")
        spawn(self._flush_compaction_queue(will_retry=False))

    def _rebuild_chat_from_session(self) -> None:
        self.chat_container.clear()
        self.pending_tools.clear()
        self.streaming_component = None
        self.streaming_message = None
        self._render_initial_messages()
        self.ui.request_render()

    async def _rebind_current_session(self, session: AgentSession, *, render_before_bind: bool = True) -> None:
        """Re-attach the UI to a replacement session. Port of `rebindCurrentSession`.

        TypeScript registers this with `runtimeHost.setRebindSession` in the
        constructor and does *all* post-replacement work here, which is why its
        `/new`, `/import` and `/clone` handlers only print a status line.

        `render_before_bind` ports TS's option of the same name: the
        replacement path renders and subscribes *before* awaiting
        `bind_extensions()` so the new session's transcript is on screen while
        extensions load, and the startup path subscribes after. Either way the
        method bails if the session was replaced again during that await, so a
        stale rebind never subscribes a second listener to the current session.
        """
        if self._unsubscribe is not None:
            with contextlib.suppress(Exception):
                self._unsubscribe()
            self._unsubscribe = None
        self._apply_runtime_settings()

        if render_before_bind:
            self.compaction_queued_messages = []
            self._rebuild_chat_from_session()
            self._subscribe_to_agent()

        await session.bind_extensions()
        if self.session is not session:
            return

        if not render_before_bind:
            self._subscribe_to_agent()

        self.footer.set_session(session)
        self._update_available_provider_count()
        self._update_editor_border_color()
        self._update_terminal_title()
        self.ui.request_render()

    def _copy_to_clipboard(self, text: str | None) -> None:
        if not text:
            self.show_status("Nothing to copy")
            return
        spawn(self._copy_to_clipboard_async(text))

    async def _copy_to_clipboard_async(self, text: str) -> None:
        try:
            await copy_to_clipboard(text)
        except Exception as error:
            self.show_error(f"Copy failed: {error}")
            return
        self.show_status("Copied to clipboard")

    def show_scoped_models_selector(self) -> None:
        available_models = list(self.session.model_runtime.get_available_snapshot())
        available_model_ids = {f"{model.provider}/{model.id}" for model in available_models}
        configured_patterns = self.settings_manager.get_enabled_models()
        session_scoped_models = self.session.scoped_models

        def configured_enabled_ids(models: list[Model]) -> list[str] | None:
            if not configured_patterns:
                return None
            resolved = resolve_model_scope_from_models(list(configured_patterns), list(models))
            ids = [f"{scoped.model.provider}/{scoped.model.id}" for scoped in resolved.scoped_models]
            # Patterns that matched nothing stay in the list so the selector can
            # show them as unavailable instead of silently dropping them.
            for diagnostic in resolved.diagnostics:
                if diagnostic.code == "no-match" and diagnostic.pattern not in ids:
                    ids.append(diagnostic.pattern)
            return ids

        current_enabled_ids = (
            [f"{item.model.provider}/{item.model.id}" for item in session_scoped_models]
            if session_scoped_models
            else configured_enabled_ids(available_models)
        )

        def apply(enabled_model_ids: list[str] | None) -> None:
            has_enabled_available_model = any(model_id in available_model_ids for model_id in (enabled_model_ids or []))
            all_available_models_enabled = enabled_model_ids is not None and all(
                model_id in enabled_model_ids for model_id in available_model_ids
            )
            if enabled_model_ids and has_enabled_available_model and not all_available_models_enabled:
                new_scoped_models = resolve_model_scope_from_models(
                    list(enabled_model_ids), available_models
                ).scoped_models
                self.session.set_scoped_models(
                    [
                        ScopedModel(model=scoped.model, thinking_level=scoped.thinking_level)
                        for scoped in new_scoped_models
                    ]
                )
            else:
                self.session.set_scoped_models([])
            self.footer.invalidate()
            self.ui.request_render()

        def persist(enabled_model_ids: list[str] | None) -> None:
            all_enabled = (
                enabled_model_ids is not None
                and len(enabled_model_ids) == len(available_models)
                and all(model_id in available_model_ids for model_id in enabled_model_ids)
            )
            new_patterns = None if (enabled_model_ids is None or all_enabled) else list(enabled_model_ids)
            self.settings_manager.set_enabled_models(new_patterns)
            self.show_status("Model selection saved to settings")

        self._show_selector(
            ScopedModelsSelectorComponent(
                ModelsConfig(all_models=available_models, enabled_model_ids=current_enabled_ids),
                ModelsCallbacks(on_change=apply, on_persist=persist, on_cancel=self._hide_selector),
            )
        )

    def show_session_selector(self) -> None:
        session_manager_class = type(self.session_manager)
        session_dir = self.session_manager.get_session_dir()

        async def load_current(on_progress: Any = None) -> list[Any]:
            return await session_manager_class.list(self.session_manager.get_cwd(), session_dir, on_progress)

        async def load_all(on_progress: Any = None) -> list[Any]:
            return await session_manager_class.list_all(session_dir, on_progress)

        def select(session_path: str) -> None:
            self._hide_selector()
            self.show_status(f"Resuming {session_path} is not available in this port yet.")

        selector = SessionSelectorComponent(
            load_current,
            load_all,
            select,
            self._hide_selector,
            self._hide_selector,
            self.ui.request_render,
            self.keybindings,
            current_session_file_path=self.session_manager.get_session_file(),
        )
        self._show_selector(selector)

    # -- shutdown -----------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in self._shutdown_signals():
            try:
                loop.add_signal_handler(sig, self._handle_shutdown_signal)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            self._signal_cleanup.append(
                lambda sig=sig: contextlib.suppress(Exception).__enter__() or loop.remove_signal_handler(sig)
            )

    @staticmethod
    def _shutdown_signals() -> tuple[signal.Signals, ...]:
        signals: list[signal.Signals] = [signal.SIGINT, signal.SIGTERM]
        if sys.platform != "win32":
            signals.append(signal.SIGHUP)
        return tuple(signals)

    def _handle_shutdown_signal(self) -> None:
        # Bash children are spawned in their own session, so they never see the
        # terminal's SIGHUP: kill them explicitly before tearing the session down.
        with contextlib.suppress(Exception):
            kill_tracked_detached_children()
        spawn(self.shutdown(from_signal=True))

    def _unregister_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in self._shutdown_signals():
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(sig)
        self._signal_cleanup = []

    async def shutdown(self, from_signal: bool = False) -> None:
        if self.is_shutting_down:
            return
        self.is_shutting_down = True
        self.shutdown_requested = True

        self._clear_status_indicator()
        # TS's `stop()`/signal path both call `themeController.disableAutoSync()`
        # before touching the terminal, so the DEC 2031 color-scheme
        # notification mode is turned back off while the terminal is still ours.
        with contextlib.suppress(Exception):
            self.theme_controller.disable_auto_sync()
        self.footer_data_provider.dispose()
        if self._unsubscribe is not None:
            with contextlib.suppress(Exception):
                self._unsubscribe()
        # Signal handlers stay registered until cleanup has finished, so a second
        # SIGINT/SIGTERM arriving mid-teardown is absorbed by the `is_shutting_down`
        # guard instead of killing the process with the default disposition.

        with contextlib.suppress(Exception):
            await self.session.abort()

        if from_signal:
            # Signal-triggered shutdown (SIGTERM/SIGHUP): tear the session down
            # before touching the terminal, so session cleanup is not skipped when
            # a later terminal-restore write fails on a dead or stalled terminal.
            with contextlib.suppress(Exception):
                await self.runtime_host.dispose()
            with contextlib.suppress(Exception):
                await self.ui.terminal.drain_input(1000)
            with contextlib.suppress(Exception):
                self.ui.stop()
            self._unregister_signal_handlers()
            self._resolve_pending_input()
            return

        # Interactive quit (Ctrl+D, Ctrl+C, /quit): stop the TUI first so session
        # teardown cannot repaint over the final frame.
        with contextlib.suppress(Exception):
            await self.ui.terminal.drain_input(1000)
        with contextlib.suppress(Exception):
            self.ui.stop()
        with contextlib.suppress(Exception):
            await self.runtime_host.dispose()
        self._unregister_signal_handlers()
        self._resolve_pending_input()

        resume = self._format_resume_command()
        if resume:
            sys.stdout.write(f"To resume this session: {resume}\n")

    def _resolve_pending_input(self) -> None:
        future = self._on_input_future
        if future is not None and not future.done():
            self._on_input_future = None
            future.set_result("")

    def _format_resume_command(self) -> str | None:
        return format_resume_command(self.session_manager)


__all__ = [
    "BUILTIN_SLASH_COMMANDS",
    "DynamicBorder",
    "InteractiveMode",
    "InteractiveModeOptions",
    "create_interactive_tui",
    "find_builtin_slash_command",
    "format_resume_command",
]

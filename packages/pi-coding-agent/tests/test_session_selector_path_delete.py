"""Python port of `packages/coding-agent/test/session-selector-path-delete.test.ts`.

Two deliberate hardenings over a literal translation, both because the
TypeScript relies on ambient state that is not actually guaranteed here:

- TS's `flushPromises()` is `setImmediate`, which drains the *entire* pending
  microtask queue. `asyncio.sleep(0)` only advances one loop iteration, so a
  literal port becomes "yield N times and hope every spawned task settled" --
  an assumption that holds only while the component's internal await points
  stay under N. `_SpawnTracker` replaces it: it intercepts the `spawn()` the
  component uses and awaits exactly those tasks to completion, so the tests
  assert on a settled component rather than on a task-scheduling race.
- `KeybindingsManager.create()` with no agent dir reads the real
  `~/.pi/agent/keybindings.json`. Tab is what toggles the scope in two of
  these tests, so a user (or a parallel test) remapping Tab would silently
  change what they exercise. The keybindings here are built from an empty
  directory, pinning the defaults.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.session_manager import SessionInfo
from pi_coding_agent.modes.interactive.components import session_selector as session_selector_module
from pi_coding_agent.modes.interactive.components.session_selector import SessionSelectorComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi
from pi_tui.keybindings import set_keybindings

CTRL_D = "\x04"
CTRL_BACKSPACE = "\x1b[127;5u"

_EPOCH = datetime.fromtimestamp(0, tz=UTC)

_KEYBINDINGS: KeybindingsManager | None = None


@pytest.fixture(autouse=True)
def _theme_and_keybindings(tmp_path_factory: pytest.TempPathFactory) -> None:
    global _KEYBINDINGS

    # The session selector uses the global theme instance.
    init_theme("dark")
    # An empty agent dir means `load_from_file` finds nothing, so only the
    # built-in defaults apply -- see this module's docstring.
    isolated_agent_dir = tmp_path_factory.mktemp("keybindings-agent-dir")
    _KEYBINDINGS = KeybindingsManager.create(str(isolated_agent_dir))
    # Ensure test isolation: keybindings are a global singleton.
    set_keybindings(KeybindingsManager.create(str(isolated_agent_dir)))


class _SpawnTracker:
    """Deterministic stand-in for TypeScript's `flushPromises()`.

    Records every task `SessionSelectorComponent` starts through the module's
    `spawn()`, so a test can await precisely those tasks instead of guessing
    how many event-loop turns they need. No test in this file triggers the
    header's auto-hide status timer, so `settle()` never has a timer to wait
    out; anything it awaits is component work that is already in flight.
    """

    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[object]] = []

    async def settle(self) -> None:
        while True:
            pending = [task for task in self.tasks if not task.done()]
            if not pending:
                return
            await asyncio.wait(pending)


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> _SpawnTracker:
    real_spawn = session_selector_module.spawn
    tracker = _SpawnTracker()

    def tracking_spawn(coro: object) -> asyncio.Task[object]:
        task = real_spawn(coro)
        tracker.tasks.append(task)
        return task

    monkeypatch.setattr(session_selector_module, "spawn", tracking_spawn)
    return tracker


def make_session(
    session_id: str,
    *,
    path: str | None = None,
    name: str | None = None,
    parent_session_path: str | None = None,
    modified: datetime | None = None,
) -> SessionInfo:
    return SessionInfo(
        path=path if path is not None else f"/tmp/{session_id}.jsonl",
        id=session_id,
        cwd="",
        name=name,
        parent_session_path=parent_session_path,
        created=_EPOCH,
        modified=modified if modified is not None else _EPOCH,
        message_count=1,
        first_message="hello",
        all_messages_text="hello",
    )


class SymlinkedSessionPaths:
    def __init__(self, base_dir: Path) -> None:
        real_dir = base_dir / "real"
        alias_a_dir = base_dir / "alias-a"
        alias_b_dir = base_dir / "alias-b"
        real_dir.mkdir(parents=True, exist_ok=True)
        alias_a_dir.mkdir(parents=True, exist_ok=True)
        alias_b_dir.mkdir(parents=True, exist_ok=True)

        shared_dir = real_dir / "sessions"
        shared_dir.mkdir(parents=True, exist_ok=True)
        alias_a_sessions = alias_a_dir / "sessions"
        alias_b_sessions = alias_b_dir / "sessions"
        alias_a_sessions.symlink_to(shared_dir, target_is_directory=True)
        alias_b_sessions.symlink_to(shared_dir, target_is_directory=True)

        (shared_dir / "parent.jsonl").write_text("parent\n")
        (shared_dir / "child.jsonl").write_text("child\n")

        self.parent_alias_a = str(alias_a_sessions / "parent.jsonl")
        self.parent_alias_b = str(alias_b_sessions / "parent.jsonl")
        self.child_alias_b = str(alias_b_sessions / "child.jsonl")


def _build(
    current_loader: object,
    all_loader: object,
    *,
    current_session_file_path: str | None = None,
) -> SessionSelectorComponent:
    return SessionSelectorComponent(
        current_loader,
        all_loader,
        lambda _path: None,
        lambda: None,
        lambda: None,
        lambda: None,
        _KEYBINDINGS,
        current_session_file_path=current_session_file_path,
    )


def _static_loaders(sessions: list[SessionInfo]) -> tuple[object, object]:
    async def load_current(_on_progress: object = None) -> list[SessionInfo]:
        return sessions

    async def load_all(_on_progress: object = None) -> list[SessionInfo]:
        return []

    return load_current, load_all


class TestSessionSelectorPathDelete:
    @pytest.mark.asyncio
    async def test_does_not_treat_ctrl_backspace_as_delete_when_query_is_non_empty(
        self, spawned: _SpawnTracker
    ) -> None:
        sessions = [make_session("a"), make_session("b")]
        selector = _build(*_static_loaders(sessions))
        await spawned.settle()

        session_list = selector.get_session_list()
        confirmation_changes: list[str | None] = []
        session_list.on_delete_confirmation_change = confirmation_changes.append

        session_list.handle_input("a")
        session_list.handle_input(CTRL_BACKSPACE)

        assert confirmation_changes == []

    @pytest.mark.asyncio
    async def test_enters_confirmation_mode_on_ctrl_d_even_with_a_non_empty_query(self, spawned: _SpawnTracker) -> None:
        sessions = [make_session("a"), make_session("b")]
        selector = _build(*_static_loaders(sessions))
        await spawned.settle()

        session_list = selector.get_session_list()
        confirmation_changes: list[str | None] = []
        session_list.on_delete_confirmation_change = confirmation_changes.append

        session_list.handle_input("a")
        session_list.handle_input(CTRL_D)

        assert confirmation_changes == [sessions[0].path]

    @pytest.mark.asyncio
    async def test_enters_confirmation_mode_on_ctrl_backspace_when_query_is_empty(self, spawned: _SpawnTracker) -> None:
        sessions = [make_session("a"), make_session("b")]
        selector = _build(*_static_loaders(sessions))
        await spawned.settle()

        session_list = selector.get_session_list()
        confirmation_changes: list[str | None] = []
        session_list.on_delete_confirmation_change = confirmation_changes.append
        deleted: list[str] = []

        async def delete_session(session_path: str) -> None:
            deleted.append(session_path)

        session_list.on_delete_session = delete_session

        session_list.handle_input(CTRL_BACKSPACE)
        assert confirmation_changes == [sessions[0].path]

        session_list.handle_input("\r")
        await spawned.settle()
        assert confirmation_changes == [sessions[0].path, None]
        assert deleted == [sessions[0].path]

    @pytest.mark.asyncio
    async def test_does_not_switch_scope_back_to_all_when_the_all_load_resolves_late(
        self, spawned: _SpawnTracker
    ) -> None:
        current_sessions = [make_session("current")]
        all_future: asyncio.Future[list[SessionInfo]] = asyncio.get_running_loop().create_future()
        all_load_started = asyncio.Event()
        all_load_calls = 0

        async def load_current(_on_progress: object = None) -> list[SessionInfo]:
            return current_sessions

        async def load_all(_on_progress: object = None) -> list[SessionInfo]:
            nonlocal all_load_calls
            all_load_calls += 1
            all_load_started.set()
            return await all_future

        selector = _build(load_current, load_all)
        await spawned.settle()

        session_list = selector.get_session_list()
        session_list.handle_input("\t")  # current -> all (starts async load)
        # TS assumes the All load is in flight by this point; wait for the
        # loader to actually be entered so "resolves late" is a fact, not a
        # scheduling coincidence.
        await all_load_started.wait()
        session_list.handle_input("\t")  # all -> current, while All is pending

        all_future.set_result([make_session("all")])
        # The late result is fully applied before anything is asserted.
        await spawned.settle()

        assert all_load_calls == 1
        output = "\n".join(selector.render(120))
        assert "Resume Session (Current Folder)" in output
        assert "Resume Session (All)" not in output

    @pytest.mark.asyncio
    async def test_does_not_start_redundant_all_loads_while_all_is_already_loading(
        self, spawned: _SpawnTracker
    ) -> None:
        current_sessions = [make_session("current")]
        all_future: asyncio.Future[list[SessionInfo]] = asyncio.get_running_loop().create_future()
        all_load_started = asyncio.Event()
        all_load_calls = 0

        async def load_current(_on_progress: object = None) -> list[SessionInfo]:
            return current_sessions

        async def load_all(_on_progress: object = None) -> list[SessionInfo]:
            nonlocal all_load_calls
            all_load_calls += 1
            all_load_started.set()
            return await all_future

        selector = _build(load_current, load_all)
        await spawned.settle()

        session_list = selector.get_session_list()
        session_list.handle_input("\t")  # current -> all (starts async load)
        await all_load_started.wait()  # the All load is in flight and parked
        session_list.handle_input("\t")  # all -> current

        loads_in_flight = len(spawned.tasks)
        session_list.handle_input("\t")  # current -> all again while load pending

        # The third Tab must reuse the in-flight load rather than start another.
        # `spawn()` is synchronous, so a redundant load would already be a task
        # here -- no event-loop turn needed to observe it.
        assert len(spawned.tasks) == loads_in_flight
        assert all_load_calls == 1

        all_future.set_result([make_session("all")])
        await spawned.settle()

        # And nothing was queued behind it either: resolving the first load
        # would have let a second loader run.
        assert all_load_calls == 1

    @pytest.mark.asyncio
    async def test_threads_sessions_when_parent_and_child_use_different_symlink_aliases(
        self, spawned: _SpawnTracker, tmp_path: Path
    ) -> None:
        paths = SymlinkedSessionPaths(tmp_path)
        sessions = [
            make_session(
                "parent",
                path=paths.parent_alias_b,
                name="Parent",
                modified=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            make_session(
                "child",
                path=paths.child_alias_b,
                parent_session_path=paths.parent_alias_a,
                name="Child",
                modified=datetime(2025, 12, 31, tzinfo=UTC),
            ),
        ]

        selector = _build(*_static_loaders(sessions))
        await spawned.settle()

        output = strip_ansi("\n".join(selector.render(120)))
        assert "Parent" in output
        assert "└─ Child" in output

    @pytest.mark.asyncio
    async def test_sorts_threaded_sessions_by_latest_activity_in_their_subtree(self, spawned: _SpawnTracker) -> None:
        parent_one = make_session("parent-one", name="Parent one", modified=datetime(2026, 1, 2, tzinfo=UTC))
        parent_two = make_session("parent-two", name="Parent two", modified=datetime(2026, 1, 1, tzinfo=UTC))
        child_two = make_session(
            "child-two",
            name="Child two",
            parent_session_path=parent_two.path,
            modified=datetime(2026, 1, 3, tzinfo=UTC),
        )

        selector = _build(*_static_loaders([parent_one, parent_two, child_two]))
        await spawned.settle()

        output = strip_ansi("\n".join(selector.render(120)))
        parent_two_index = output.find("Parent two")
        child_two_index = output.find("└─ Child two")
        parent_one_index = output.find("Parent one")

        assert parent_two_index >= 0
        assert child_two_index > parent_two_index
        assert parent_one_index > child_two_index

    @pytest.mark.asyncio
    async def test_treats_the_current_session_as_active_across_symlink_aliases(
        self, spawned: _SpawnTracker, tmp_path: Path
    ) -> None:
        paths = SymlinkedSessionPaths(tmp_path)
        sessions = [make_session("parent", path=paths.parent_alias_b, name="Parent")]
        selector = _build(
            *_static_loaders(sessions),
            current_session_file_path=paths.parent_alias_a,
        )
        await spawned.settle()

        session_list = selector.get_session_list()
        confirmation_changes: list[str | None] = []
        errors: list[str] = []
        session_list.on_delete_confirmation_change = confirmation_changes.append
        session_list.on_error = errors.append

        session_list.handle_input(CTRL_D)

        assert confirmation_changes == []
        assert errors == ["Cannot delete the currently active session"]

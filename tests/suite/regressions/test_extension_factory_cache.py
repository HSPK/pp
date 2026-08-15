"""Python port of `packages/coding-agent/test/suite/regressions/extension-factory-cache.test.ts`.

The TypeScript test counts how many times an extension *module* executes versus
how many times its exported factory runs, using a global counter object the
extension file mutates. The Python equivalent writes an extension `.py` file
that increments counters on a module the test also imports, which is the same
observation: module top-level code runs once per cached cwd, the factory runs
once per load.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from pi_coding_agent.core.extensions.loader import (
    clear_extension_cache,
    load_extensions,
    load_extensions_cached,
)

COUNTER_MODULE = """
import json
import os

_STATE_PATH = os.environ["PI_EXTENSION_CACHE_TEST_STATE"]


def _bump(key):
    try:
        with open(_STATE_PATH, encoding="utf-8") as handle:
            state = json.load(handle)
    except FileNotFoundError:
        state = {}
    state[key] = state.get(key, 0) + 1
    with open(_STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(state, handle)


_bump("module_loads")


def pi_extension(pi):
    _bump("factory_runs")
"""


@pytest.fixture(autouse=True)
def _cache_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    # Short mkdtemp root rather than tmp_path: extension paths end up in
    # module names and error strings, and the fixtures here are throwaway.
    root = Path(tempfile.mkdtemp(prefix="pi-t-"))
    state_path = root / "state.json"
    monkeypatch.setenv("PI_EXTENSION_CACHE_TEST_STATE", str(state_path))
    clear_extension_cache()
    try:
        yield root
    finally:
        clear_extension_cache()
        shutil.rmtree(root, ignore_errors=True)


def _read_state(root: Path) -> dict[str, int]:
    import json

    state_path = root / "state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _write_counting_extension(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(COUNTER_MODULE, encoding="utf-8")


async def test_caches_extension_modules_for_cached_same_cwd_loads_but_reruns_factories(
    _cache_state: Path,
) -> None:
    root = _cache_state
    cwd = root / "project"
    cwd.mkdir()
    extension_path = root / "counting.py"
    _write_counting_extension(extension_path)

    first = await load_extensions_cached([str(extension_path)], str(cwd))
    second = await load_extensions_cached([str(extension_path)], str(cwd))

    assert first.errors == []
    assert second.errors == []
    state = _read_state(root)
    assert state.get("module_loads") == 1
    assert state.get("factory_runs") == 2
    assert first.extensions[0] is not second.extensions[0]
    # TypeScript also asserts `first.runtime !== second.runtime`. This port has
    # no per-load `ExtensionRuntime` object -- runtime actions are passed in by
    # the caller as `ExtensionRuntimeActions` and `LoadExtensionsResult` carries
    # only `extensions`/`errors` -- so there is nothing to compare.


async def test_does_not_cache_direct_load_extensions_calls(_cache_state: Path) -> None:
    root = _cache_state
    cwd = root / "project"
    cwd.mkdir()
    extension_path = root / "counting.py"
    _write_counting_extension(extension_path)

    await load_extensions([str(extension_path)], str(cwd))
    await load_extensions([str(extension_path)], str(cwd))

    state = _read_state(root)
    assert state.get("module_loads") == 2
    assert state.get("factory_runs") == 2


async def test_clearing_the_cache_reloads_the_module(_cache_state: Path) -> None:
    """Stands in for "clears the cache on resource loader reload".

    TypeScript's `DefaultResourceLoader.reload()` discovers and loads
    extensions itself and calls `clearExtensionCache()` on every reload after
    the first. This port's `ResourceLoader` does not own extensions at all --
    they are loaded via `discover_and_load_extensions()` by the caller and
    handed to `AgentSession` -- so there is no `loader.reload()` to drive.
    What the TypeScript test actually pins is that a cleared cache re-executes
    the module, which is asserted directly here.
    """
    root = _cache_state
    cwd = root / "project"
    cwd.mkdir()
    extension_path = root / "counting.py"
    _write_counting_extension(extension_path)

    await load_extensions_cached([str(extension_path)], str(cwd))
    clear_extension_cache()
    await load_extensions_cached([str(extension_path)], str(cwd))

    state = _read_state(root)
    assert state.get("module_loads") == 2
    assert state.get("factory_runs") == 2


async def test_keeps_the_cache_scoped_to_one_cwd(_cache_state: Path) -> None:
    root = _cache_state
    first_cwd = root / "first"
    second_cwd = root / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    extension_path = root / "counting.py"
    _write_counting_extension(extension_path)

    await load_extensions_cached([str(extension_path)], str(first_cwd))
    await load_extensions_cached([str(extension_path)], str(second_cwd))
    await load_extensions_cached([str(extension_path)], str(second_cwd))

    state = _read_state(root)
    assert state.get("module_loads") == 2
    assert state.get("factory_runs") == 3


async def test_discovery_never_uses_the_cache(_cache_state: Path) -> None:
    """`discoverAndLoadExtensions` calls the uncached `loadExtensions`."""
    from pi_coding_agent.core.extensions.loader import discover_and_load_extensions

    root = _cache_state
    cwd = root / "project"
    agent_dir = root / "agent"
    (agent_dir / "extensions").mkdir(parents=True)
    cwd.mkdir()
    _write_counting_extension(agent_dir / "extensions" / "counting.py")

    await discover_and_load_extensions([], str(cwd), agent_dir=str(agent_dir))
    await discover_and_load_extensions([], str(cwd), agent_dir=str(agent_dir))

    state = _read_state(root)
    assert state.get("module_loads") == 2
    assert state.get("factory_runs") == 2

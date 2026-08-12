"""Guard against test-module basename collisions across the workspace.

pytest collects this repo in its default `prepend` import mode, so two test
files sharing a basename — even in different packages — collide in
`sys.modules` and abort collection for the **entire** suite:

    import file mismatch: imported module 'test_unix' has this __file__ ...
    Interrupted: 1 error during collection

The failure is remote from its cause: a new file in `pi-server` breaks
`pi-client`'s run, and a per-package `pytest packages/pi-foo` still passes, so
whoever added the file sees nothing wrong. That happened four times while
several people were adding tests in parallel.

`--import-mode=importlib` would lift the restriction, but it breaks the
`from conftest import ...` / `from support import ...` sibling imports several
suites rely on. So the constraint stays, and this test enforces it directly.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

PACKAGES_ROOT = Path(__file__).resolve().parents[3]

_IGNORED_DIRS = frozenset({"__pycache__", "node_modules", "build", "dist"})


def collect_test_modules() -> dict[str, list[Path]]:
    """Map every collectable test module basename to the files that use it.

    Directories pytest never collects from are ignored, so a scratch copy
    parked in `.wip/` (or any dot-directory) does not trip a guard about a
    collision that cannot actually happen.
    """
    by_basename: dict[str, list[Path]] = defaultdict(list)
    for path in PACKAGES_ROOT.rglob("test_*.py"):
        if any(part.startswith(".") or part in _IGNORED_DIRS for part in path.parts):
            continue
        by_basename[path.name].append(path)
    return by_basename


def test_no_two_test_modules_share_a_basename() -> None:
    duplicates = {name: paths for name, paths in collect_test_modules().items() if len(paths) > 1}

    if duplicates:
        report = "\n".join(
            f"  {name}:\n" + "\n".join(f"    {path.relative_to(PACKAGES_ROOT)}" for path in sorted(paths))
            for name, paths in sorted(duplicates.items())
        )
        raise AssertionError(
            "Test module basenames must be unique across the whole repo, or pytest's\n"
            "default `prepend` import mode aborts collection for every package.\n"
            "Rename one of each pair, qualifying it with its package or subject\n"
            f"(e.g. `test_server_unix.py`, `test_ai_max_thinking.py`):\n{report}"
        )


def test_the_guard_actually_scans_the_workspace() -> None:
    """A typo in the glob would make the check above vacuously pass."""
    modules = collect_test_modules()

    assert len(modules) > 100, f"expected to find the workspace's test modules, found {len(modules)}"
    assert "test_console_scripts.py" in modules

"""The shipped examples must stay runnable.

The system prompt points the model at `examples/`, and `docs/sdk.md` links to
`examples/sdk/`, so a stale example is documentation that lies. TypeScript
gets this for free from `tsc`; Python needs a test, or an example keeps
importing a symbol that was renamed months ago.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SDK_EXAMPLES = sorted((EXAMPLES / "sdk").glob("*.py"))
EXTENSION_EXAMPLES = sorted((EXAMPLES / "extensions").glob("*.py"))
ALL_EXAMPLES = SDK_EXAMPLES + EXTENSION_EXAMPLES


def test_the_sdk_examples_directory_is_not_empty() -> None:
    """`docs/sdk.md` links here; an empty directory makes that link a lie."""
    assert SDK_EXAMPLES, "docs/sdk.md points at examples/sdk/"


@pytest.mark.parametrize("path", ALL_EXAMPLES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_an_example_compiles(path: Path) -> None:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("path", ALL_EXAMPLES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_symbol_an_example_imports_still_exists(path: Path) -> None:
    """Catches the rename that leaves an example importing a gone name.

    Only `pi_*` imports are checked: third-party ones are the environment's
    problem, not the port's.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("pi_"):
            continue
        module = importlib.import_module(node.module)
        missing.extend(f"{node.module}.{alias.name}" for alias in node.names if not hasattr(module, alias.name))

    assert missing == []

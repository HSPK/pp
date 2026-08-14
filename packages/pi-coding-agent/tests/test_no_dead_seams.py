"""No inert default may be supplied only by tests.

Five separate defects in this port shared one shape: a capability was ported
and worked, but production never supplied the collaborator that activates it.
Each time, the whole suite stayed green, because the *test harness* supplied
what production forgot -- so the tests proved the capability worked and never
that anything reached it.

The shape is mechanical, so this catches it instead of the next audit:

* `pi.send_user_message` and six sibling actions defaulted to no-op lambdas
  and were bound only in `tests/suite/harness.py`.
* `ctx.shutdown()` was hardcoded to `lambda: None` in `AgentSession`.
* `pi.events` had no bus unless a host passed one, and none did.

A seam that is genuinely optional stays allowed by naming it below, which
turns "nobody wired this" into a decision someone wrote down.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[2]


def _is_inert_default(value: ast.expr) -> bool:
    """A default that does nothing: `lambda ...: None`, `lambda ...: []`, `list`.

    Matching on the lambda *body* rather than the source text: a parameter
    list can itself contain `:` and `=` (`lambda custom_type, data=None: None`),
    which a regex over the whole expression gets wrong -- and silently, by
    finding nothing at all.
    """
    if isinstance(value, ast.Name) and value.id == "list":
        return True
    if not isinstance(value, ast.Lambda):
        return False
    body = value.body
    if isinstance(body, ast.Constant) and body.value is None:
        return True
    return isinstance(body, ast.List) and not body.elts


# Seams deliberately left for a host to fill, with the reason.
ALLOWED: dict[tuple[str, str], str] = {
    # `AgentHarness` takes its tool context through each tool's closure, the
    # way `create_harness` builds them; the field is for SDK embedders.
    ("AgentHarnessOptions", "tool_context"): "supplied per-tool by create_harness",
}


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for package in sorted(PACKAGES.glob("pi-*")):
        for path in (package / "src").rglob("*.py"):
            if "__pycache__" not in str(path):
                files.append(path)
    return files


def _iter_test_files() -> list[Path]:
    files: list[Path] = []
    for package in sorted(PACKAGES.glob("pi-*")):
        for path in (package / "tests").rglob("*.py"):
            if "__pycache__" not in str(path):
                files.append(path)
    return files


def _inert_dataclass_fields() -> list[tuple[Path, str, str]]:
    """Every dataclass field defaulting to a no-op callable."""
    found: list[tuple[Path, str, str]] = []
    for path in _iter_source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if not isinstance(statement, ast.AnnAssign) or statement.value is None:
                    continue
                if not isinstance(statement.target, ast.Name):
                    continue
                if _is_inert_default(statement.value):
                    found.append((path, node.name, statement.target.id))
    return found


def _keyword_arguments(path: Path) -> set[str]:
    """Field names this file passes as keyword arguments to any call.

    AST rather than a substring search: `field=` also matches an annotated
    declaration, a comparison and a docstring, and treating those as
    suppliers is what would make this test pass while the seam is dead.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            names.update(keyword.arg for keyword in node.keywords if keyword.arg)
    return names


def test_every_inert_default_is_supplied_outside_tests() -> None:
    source_keywords = {path: _keyword_arguments(path) for path in _iter_source_files()}
    test_keywords = {path: _keyword_arguments(path) for path in _iter_test_files()}

    dead: list[str] = []
    for path, class_name, field in _inert_dataclass_fields():
        if (class_name, field) in ALLOWED:
            continue
        if any(field in keywords for keywords in source_keywords.values()):
            continue
        suppliers = sorted(other.name for other, keywords in test_keywords.items() if field in keywords)
        dead.append(
            f"{class_name}.{field} ({path.name}) is inert by default and supplied only by {suppliers or ['nothing']}"
        )

    assert dead == [], (
        "These collaborators are no-ops in the shipped package. Wire them, or add them to ALLOWED "
        "with a reason:\n  " + "\n  ".join(dead)
    )

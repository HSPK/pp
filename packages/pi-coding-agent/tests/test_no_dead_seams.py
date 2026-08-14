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


def _source_roots() -> list[Path]:
    """Every `src/` tree this guard should scan, in either repository layout.

    In the monorepo that is all nine `packages/pi-*/src`. In this package's own
    repository the siblings are installed dependencies, not checkouts, so the
    scope is this repository alone.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "packages":
            return sorted(package / "src" for package in parent.glob("pi-*") if (package / "src").is_dir())
    return [here.parents[1] / "src"]


def _test_roots() -> list[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "packages":
            return sorted(package / "tests" for package in parent.glob("pi-*") if (package / "tests").is_dir())
    return [here.parents[1] / "tests"]


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


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
    for root in _source_roots():
        for path in root.rglob("*.py"):
            if "__pycache__" not in str(path):
                files.append(path)
    # A guard that scans nothing passes. That has already happened twice in
    # this file (a regex that matched no lambda, and a wrong `parents[]`
    # index), so an empty scan is a failure, not a pass.
    assert files, f"no source files found under {[str(root) for root in _source_roots()]}"
    return files


def _iter_test_files() -> list[Path]:
    files: list[Path] = []
    for root in _test_roots():
        for path in root.rglob("*.py"):
            if "__pycache__" not in str(path):
                files.append(path)
    assert files, f"no test files found under {[str(root) for root in _test_roots()]}"
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


# --------------------------------------------------------------------------
# The other two shapes the same defect takes.
# --------------------------------------------------------------------------

ARGS_MODULE = PACKAGE_ROOT / "src/pi_coding_agent/cli/args.py"
EXTENSION_TYPES = PACKAGE_ROOT / "src/pi_coding_agent/core/extensions/types.py"

# Args fields that exist to be reported, not consumed.
ALLOWED_UNREAD_ARGS: dict[str, str] = {
    "diagnostics": "collected by the parser and printed by the caller",
}


def _attribute_reads(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def test_every_parsed_argument_is_read_somewhere() -> None:
    """`--use-theme` and `--no-themes` were both parsed and never consumed.

    The flag is accepted, the field is set, and nothing downstream looks at
    it -- so the option does nothing and says nothing.
    """
    tree = ast.parse(ARGS_MODULE.read_text(encoding="utf-8"))
    fields = [
        statement.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Args"
        for statement in node.body
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
    ]
    assert fields, "Args has no annotated fields; this test is not looking at the right class"

    reads: set[str] = set()
    for path in _iter_source_files():
        if path == ARGS_MODULE:
            continue
        reads |= _attribute_reads(path)

    unread = sorted(f for f in fields if f not in reads and f not in ALLOWED_UNREAD_ARGS)
    assert unread == [], (
        "These CLI options are parsed and never read, so they are accepted and ignored:\n  " + "\n  ".join(unread)
    )


def test_every_extension_event_is_constructed_somewhere() -> None:
    """`ProjectTrustEvent` was defined, subscribable, and never emitted.

    An event an extension can register for, that nothing constructs, is a
    handler that never runs.
    """
    tree = ast.parse(EXTENSION_TYPES.read_text(encoding="utf-8"))
    events = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name.endswith("Event")
        and any(
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "type"
            for statement in node.body
        )
    ]
    assert events, "no event payloads found; this test is not looking at the right module"

    constructed: set[str] = set()
    for path in _iter_source_files():
        if path == EXTENSION_TYPES:
            continue
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        # Several of these names collide with `pi_agent`'s own events, so this
        # package imports them aliased (`AgentStartEvent as ExtAgentStartEvent`)
        # and constructs the alias. Matching on the written name alone would
        # miss every one -- and in a monorepo checkout it *passed anyway*,
        # because `pi_agent`'s unrelated same-named classes were in scope.
        aliases: dict[str, str] = {}
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom | ast.Import):
                for alias in node.names:
                    if alias.asname:
                        aliases[alias.asname] = alias.name.rsplit(".", 1)[-1]

        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                constructed.add(aliases.get(name, name))

    never = sorted(name for name in events if name not in constructed)
    assert never == [], (
        "These extension events are declared but never emitted, so their handlers never run:\n  " + "\n  ".join(never)
    )

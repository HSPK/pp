#!/usr/bin/env python3
"""Check that every cross-package import is a declared dependency.

`uv sync --all-packages` installs the whole workspace, so a package can import
a sibling it never declared and every test still passes. The failure only
appears once a wheel is installed on its own, where `Requires-Dist` is the only
thing that pulls siblings in -- an end user gets `ModuleNotFoundError` on
startup. That is exactly how `pp-coding-agent` shipped without `pp-rpc-server`.

This walks each package's `src/` with `ast`, collects the top-level modules it
imports, maps the ones belonging to this workspace back to their distribution
names, and requires each to appear in `[project].dependencies`.

    python scripts/check_dependencies.py

Exits non-zero on the first package with an undeclared internal import.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = REPO_ROOT / "packages"


def _import_name_to_distribution() -> dict[str, str]:
    """Map each workspace import package (`pi_ai`) to its distribution (`pp-ai`)."""
    mapping: dict[str, str] = {}
    for manifest_path in sorted(PACKAGES_DIR.glob("*/pyproject.toml")):
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        distribution = manifest["project"]["name"]
        for entry in manifest["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]:
            mapping[Path(entry).name] = distribution
    return mapping


def _top_level_imports(source_dir: Path) -> set[str]:
    modules: set[str] = set()
    for path in source_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            # `node.level > 0` is a relative import, which can only ever reach
            # the package's own modules.
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].strip()


def main() -> int:
    import_to_distribution = _import_name_to_distribution()
    problems: list[str] = []

    for manifest_path in sorted(PACKAGES_DIR.glob("*/pyproject.toml")):
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        project = manifest["project"]
        distribution = project["name"]

        source_dir = manifest_path.parent / "src"
        own_modules = {Path(entry).name for entry in manifest["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]}

        declared = {_requirement_name(requirement) for requirement in project.get("dependencies", [])}
        for extra in project.get("optional-dependencies", {}).values():
            declared.update(_requirement_name(requirement) for requirement in extra)

        imported = _top_level_imports(source_dir)
        needed = {
            import_to_distribution[module]
            for module in imported
            if module in import_to_distribution and module not in own_modules
        }

        for missing in sorted(needed - declared):
            problems.append(f"{distribution}: imports {missing} but does not declare it as a dependency")

        status = "ok" if needed <= declared else "FAIL"
        print(f"{distribution:18} internal imports: {sorted(needed) or '-'}  [{status}]")

    if problems:
        print("\ndependency check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print("\ndependency check passed: every cross-package import is declared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

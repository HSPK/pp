#!/usr/bin/env python3
"""Set or check the workspace-wide version.

The nine `packages/*` distributions are released in lockstep: they always carry
the same version, and every internal dependency is pinned to `==<version>`
rather than a range. That pin is not stylistic. `[tool.uv.sources]` entries do
not survive into a built wheel, so an unpinned `pp-ai` in `Requires-Dist` would
be resolved from PyPI at install time, and a cross-version combination of these
packages has never been tested together.

Keeping nine `pyproject.toml` files in agreement by hand does not work, so this
script owns the bump. It is the Python counterpart of the upstream monorepo's
`scripts/sync-versions.js`.

    python scripts/set_version.py 0.2.0    # rewrite every version and pin
    python scripts/set_version.py --check  # verify agreement, exit 1 if not

`--check` runs from `check.sh`, so a hand-edited version that only lands in one
file fails the normal development loop instead of a release.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = REPO_ROOT / "packages"

# PEP 440, restricted to the subset this project actually releases: a three
# part release number with an optional `aN`/`bN`/`rcN` pre-release suffix.
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")

_PROJECT_VERSION_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")$', re.MULTILINE)


def _package_manifests() -> list[Path]:
    manifests = sorted(PACKAGES_DIR.glob("*/pyproject.toml"))
    if not manifests:
        raise SystemExit(f"no packages found under {PACKAGES_DIR}")
    return manifests


def _distribution_name(text: str) -> str:
    match = re.search(r'^name\s*=\s*"([^"]+)"$', text, re.MULTILINE)
    if not match:
        raise SystemExit("pyproject.toml has no [project] name")
    return match.group(1)


def _declared_version(text: str) -> str:
    match = _PROJECT_VERSION_RE.search(text)
    if not match:
        raise SystemExit("pyproject.toml has no [project] version")
    return match.group(2)


def _internal_names(manifests: list[Path]) -> set[str]:
    return {_distribution_name(path.read_text(encoding="utf-8")) for path in manifests}


def _pin_re(names: set[str]) -> re.Pattern[str]:
    """Matches `"<internal-dist>==<version>"` for any workspace distribution.

    Only `==` pins are rewritten. A third-party requirement such as
    `"pytest>=8.3"` never matches, because the alternation is built from the
    workspace's own distribution names.
    """
    alternation = "|".join(sorted((re.escape(name) for name in names), key=len, reverse=True))
    return re.compile(rf'"({alternation})==([^"]+)"')


def _rewrite(text: str, new_version: str, pin_re: re.Pattern[str]) -> str:
    # Only the first `version = "..."` is the project version; later ones would
    # belong to other tables, so the replacement count is capped at one.
    text = _PROJECT_VERSION_RE.sub(rf"\g<1>{new_version}\g<3>", text, count=1)
    return pin_re.sub(rf'"\g<1>=={new_version}"', text)


def set_version(new_version: str) -> None:
    if not VERSION_RE.match(new_version):
        raise SystemExit(f"{new_version!r} is not an accepted version (expected e.g. 0.2.0 or 0.2.0rc1)")

    manifests = _package_manifests()
    pin_re = _pin_re(_internal_names(manifests))

    for path in manifests:
        original = path.read_text(encoding="utf-8")
        updated = _rewrite(original, new_version, pin_re)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
        print(f"{_distribution_name(updated)} -> {new_version}")

    print("\nversions rewritten. Now run:  uv lock && uv sync --all-packages")


def check() -> int:
    manifests = _package_manifests()
    names = _internal_names(manifests)
    pin_re = _pin_re(names)

    versions: dict[str, str] = {}
    problems: list[str] = []

    for path in manifests:
        text = path.read_text(encoding="utf-8")
        name = _distribution_name(text)
        versions[name] = _declared_version(text)

    distinct = set(versions.values())
    if len(distinct) > 1:
        problems.append(f"packages disagree on the version: {sorted(distinct)}")
        for name, version in sorted(versions.items()):
            problems.append(f"  {name} = {version}")

    expected = next(iter(distinct)) if len(distinct) == 1 else None

    for path in manifests:
        text = path.read_text(encoding="utf-8")
        name = _distribution_name(text)
        for dependency, pinned in pin_re.findall(text):
            if expected is not None and pinned != expected:
                problems.append(f"{name}: pins {dependency}=={pinned}, expected =={expected}")

        # An unpinned internal dependency is the dangerous case: `Requires-Dist`
        # leaves this workspace without `[tool.uv.sources]`, so anything not
        # pinned is resolved from PyPI at install time. Only `[project]
        # .dependencies` and `.optional-dependencies` are published, which is
        # why this reads the parsed tables rather than the raw text --
        # `[dependency-groups]` entries stay in the workspace and are exempt.
        manifest = tomllib.loads(text)
        project = manifest.get("project", {})
        published: list[str] = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            published.extend(extra)

        for requirement in published:
            requirement_name = re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].strip()
            if requirement_name in names and f"{requirement_name}==" not in requirement:
                problems.append(f"{name}: depends on {requirement_name} without an == pin")

    if problems:
        print("version check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"version check passed: {len(manifests)} packages at {expected}, all internal deps pinned")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", nargs="?", help="the new version, e.g. 0.2.0")
    parser.add_argument("--check", action="store_true", help="verify agreement instead of rewriting")
    args = parser.parse_args()

    if args.check:
        return check()
    if not args.version:
        parser.error("provide a version, or --check")
    set_version(args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

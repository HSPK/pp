#!/usr/bin/env python3
"""Set or check versions and the internal pins that follow them.

Every internal dependency is constrained to `~=<major>.<minor>.0`, which admits
any later patch in the same minor series and nothing else. Some constraint is
mandatory: `[tool.uv.sources]` entries do not survive into a built wheel, so an
unconstrained `pp-ai` in `Requires-Dist` would be resolved from PyPI at install
time and could pick up anything at all.

It used to be `==`, which guaranteed users got exactly the combination CI had
tested. That guarantee cost too much: a one-line fix in `pp-ai` could not reach
anyone until all nine packages were re-released, because every dependent pinned
the old version exactly. `~=` lets a patch flow downstream on its own.

The guarantee is replaced, not abandoned. It now rests on two things, and both
have to hold:

  * A change to an API another package uses is a *minor* bump, never a patch.
    `~=0.2.0` will not cross into 0.3.0, so a minor bump stops the flow and
    forces the dependent to opt in.
  * Each repository re-resolves its dependencies on a schedule and runs its
    suite, so a patch that breaks a dependent surfaces on its own rather than
    waiting for someone to touch that repository.

The nine distributions used to be released in lockstep, all carrying the same
version. Now that each one lives in its own repository they are released
independently, so the invariant this enforces is narrower and truer: **every
pin must name the version that package actually declares**. Whether two
packages happen to share a version number is not interesting; whether
`pp-coding-agent` pins a `pp-tui` that exists is.

    python scripts/set_version.py 0.2.0                  # bump everything
    python scripts/set_version.py 0.1.1 --package pp-tui # bump one, fix its pins
    python scripts/set_version.py --check                # verify, exit 1 if not

A patch bump (0.2.0 -> 0.2.1) leaves every dependent's constraint untouched,
because `~=0.2.0` already admits it. Only a minor bump rewrites them.

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
    """Matches `"<internal-dist>~=<version>"` for any workspace distribution.

    A third-party requirement such as `"pytest>=8.3"` never matches, because
    the alternation is built from the workspace's own distribution names.
    """
    alternation = "|".join(sorted((re.escape(name) for name in names), key=len, reverse=True))
    return re.compile(rf'"({alternation})~=([^"]+)"')


def _series_floor(version: str) -> str | None:
    """`0.2.7` -> `0.2.0`, the floor of the minor series it belongs to.

    Returns `None` for anything that is not three dotted parts. `~=0.2` is
    legal PEP 440 but means something quite different (it admits 0.3 and 0.9),
    so the caller has to be able to reject it rather than crash on it.
    """
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts[:2]):
        return None
    return f"{parts[0]}.{parts[1]}.0"


def _rewrite(text: str, new_version: str, pin_re: re.Pattern[str]) -> str:
    # Only the first `version = "..."` is the project version; later ones would
    # belong to other tables, so the replacement count is capped at one.
    text = _PROJECT_VERSION_RE.sub(rf"\g<1>{new_version}\g<3>", text, count=1)
    return pin_re.sub(rf'"\g<1>~={_series_floor(new_version)}"', text)


def set_version(new_version: str, package: str | None = None) -> None:
    if not VERSION_RE.match(new_version):
        raise SystemExit(f"{new_version!r} is not an accepted version (expected e.g. 0.2.0 or 0.2.0rc1)")

    manifests = _package_manifests()
    names = _internal_names(manifests)

    if package is not None:
        if package not in names:
            raise SystemExit(f"{package!r} is not one of {sorted(names)}")
        # Only this distribution's own version and the pins *on* it move;
        # everything else keeps whatever version it already declares.
        pin_re = _pin_re({package})
        for path in manifests:
            original = path.read_text(encoding="utf-8")
            updated = pin_re.sub(rf'"\g<1>~={_series_floor(new_version)}"', original)
            if _distribution_name(original) == package:
                updated = _PROJECT_VERSION_RE.sub(rf"\g<1>{new_version}\g<3>", updated, count=1)
            if updated != original:
                path.write_text(updated, encoding="utf-8")
                print(f"{_distribution_name(updated)}: updated")
        print(f"\n{package} -> {new_version}")
        return

    pin_re = _pin_re(names)
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


    for path in manifests:
        text = path.read_text(encoding="utf-8")
        name = _distribution_name(text)
        for dependency, constraint in pin_re.findall(text):
            declared = versions.get(dependency)
            if declared is None:
                continue
            # The constraint must be the floor of a minor series (`~=0.2.0`).
            # `~=0.2` would admit 0.3 and 0.9 too, which is exactly the
            # crossing this scheme relies on being blocked.
            floor = _series_floor(constraint)
            if floor is None or constraint != floor:
                problems.append(
                    f"{name}: constrains {dependency}~={constraint}; use a three-part series "
                    f"floor such as ~={_series_floor(declared)} -- ~=X.Y admits X.Y+1 too"
                )
            elif _series_floor(declared) != constraint:
                problems.append(
                    f"{name}: constrains {dependency}~={constraint}, but {dependency} declares "
                    f"{declared}, which that constraint does not admit"
                )

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
            if requirement_name in names and f"{requirement_name}~=" not in requirement:
                problems.append(
                    f"{name}: depends on {requirement_name} without a ~= constraint; "
                    "an unconstrained sibling is resolved from PyPI at install time"
                )

    if problems:
        print("version check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    summary = ", ".join(f"{name} {version}" for name, version in sorted(versions.items()))
    print(f"version check passed: every internal constraint admits its package ({summary})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", nargs="?", help="the new version, e.g. 0.2.0")
    parser.add_argument("--check", action="store_true", help="verify agreement instead of rewriting")
    parser.add_argument("--package", help="bump only this distribution, and every pin on it")
    args = parser.parse_args()

    if args.check:
        return check()
    if not args.version:
        parser.error("provide a version, or --check")
    set_version(args.version, args.package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate one standalone repository per package from the monorepo.

Each package already carries its own `pyproject.toml`, `src/`, `tests/`,
`README.md` and `LICENSE`, so splitting is mostly a copy. What this script adds
is everything the monorepo supplied centrally and a standalone repo has to own:
the ruff/pytest/coverage configuration from the root `pyproject.toml`, a CI
workflow, and a `release.yml` that publishes exactly one distribution.

Run:  python scripts/split_repos.py --out ~/projects/pp-mono
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

MONOREPO = Path(__file__).resolve().parents[1]

# package directory -> (GitHub repo name, PyPI distribution name)
REPOS: dict[str, tuple[str, str]] = {
    "pi-telemetry": ("pp_telemetry", "pp-telemetry"),
    "pi-ai": ("pp_ai", "pp-ai"),
    "pi-agent": ("pp_agent_core", "pp-agent-core"),
    "pi-tui": ("pp_tui", "pp-tui"),
    "pi-protocol": ("pp_rpc_protocol", "pp-rpc-protocol"),
    "pi-client": ("pp_rpc_client", "pp-rpc-client"),
    "pi-server": ("pp_rpc_server", "pp-rpc-server"),
    "pi-coding-agent": ("pp", "pp-coding-agent"),
    "pi-evals": ("pp_evals", "pp-evals"),
}

OWNER = "HSPK"

# The `dev` group, kept separate because a package may already declare
# `[dependency-groups]` for its own test-only requirements (pi-server does).
# Emitting a second table of the same name is invalid TOML, so this is merged
# into the existing one rather than appended.
DEV_GROUP = """dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "pytest-xdist>=3.6",
    # Pinned, not floored: `release.yml` gates publishing on `ruff check`, so a
    # new ruff release introducing a rule would block every package's release
    # for a reason unrelated to the release. Bump it deliberately instead.
    "ruff==0.16.3",
    "mypy>=1.13",
]
"""

# Copied verbatim into every repo; the monorepo kept these at the root.
TOOLING = """
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
# `-n auto` matters: the suite is dominated by tests that wait (real
# `asyncio.sleep` for OAuth device-code polling, retry backoff, subprocess
# timeouts), so wall time was ~2.5x CPU time when run serially. Override
# with `-n0` to debug a single test.
addopts = "-q -n auto"
filterwarnings = ["error::DeprecationWarning:pi_.*"]

[tool.coverage.run]
branch = true
source = ["src"]

[tool.coverage.report]
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "@overload",
    "\\\\.\\\\.\\\\.$",
]

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
ignore = ["E501", "B008", "SIM108", "RUF001", "RUF002", "RUF003"]

[tool.mypy]
python_version = "3.11"
warn_unused_ignores = true
ignore_missing_imports = true
"""

GITIGNORE = """__pycache__/
uv.lock
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
.coverage
.coverage-html/
*.egg-info/
dist/
build/
.scratch/
sessions/
*.log
.env
"""

CI_YML = """# Lint, format and test checks.
#
# Runs on pull requests only: `main` is protected and takes commits through
# PRs, so a push-triggered run would re-test the exact tree the PR already
# tested. `release.yml` calls this same workflow through `workflow_call`, so
# the release path and the merge path cannot drift apart.
name: ci

on:
  pull_request:
    branches: [main]
  workflow_call:
  workflow_dispatch:

env:
  UV_VERSION: "0.12.1"

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with:
          version: ${{ env.UV_VERSION }}
          enable-cache: true
      # `--all-groups`: a package may declare dependency groups beyond `dev`
      # (pp-rpc-server puts its client-side integration dependency in `test`),
      # and a plain `uv sync` installs only `dev`, so those tests error out.
      - run: uv sync --all-groups
      - name: ruff check
        run: uv run ruff check .
      - name: ruff format --check
        run: uv run ruff format --check .
      - name: pytest
        run: uv run pytest
"""

RELEASE_YML = """# Publishes {dist} to PyPI on a `v*` tag.
#
# Publishing uses Trusted Publishing (OIDC), so no long-lived API token is
# stored here. PyPI matches the publisher on repository *and* workflow
# filename, so this file must stay named `release.yml`.
#
# This repository releases one distribution. Its internal `pp-*` dependencies
# are pinned with `==`, so whichever of them changed has to be on PyPI before
# this tag is pushed, or the published wheel is uninstallable.
name: release

on:
  push:
    tags: ["v*"]
  workflow_dispatch:
    inputs:
      repository:
        description: Index to publish to
        type: choice
        options: [testpypi, pypi]
        default: testpypi

env:
  UV_VERSION: "0.12.1"

jobs:
  # The same checks a pull request runs, so the release path cannot pass with
  # a tree the merge path would have rejected.
  check:
    uses: ./.github/workflows/ci.yml

  build:
    needs: check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      # A step rather than its own job: a skipped job would skip `build` with
      # it, and `workflow_dispatch` runs without a tag.
      - name: Verify the tag matches the declared version
        if: startsWith(github.ref, 'refs/tags/v')
        run: |
          tag="${{{{ github.ref_name }}}}"
          tag="${{tag#v}}"
          declared=$(python3 -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")
          if [ "$tag" != "$declared" ]; then
            echo "tag $tag does not match declared version $declared" >&2
            exit 1
          fi
      - uses: astral-sh/setup-uv@v7
        with:
          version: ${{{{ env.UV_VERSION }}}}
      - run: uv build --out-dir dist
      - name: Check the built metadata
        run: uvx twine check dist/*
      - name: Install from the built distribution alone
        # An editable/workspace install hides a missing `Requires-Dist` entry,
        # because the sources are importable either way. Installing the way a
        # user does is the check that catches it.
        run: |
          version=$(python3 -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")
          uv venv --python 3.11 /tmp/isolated
          VIRTUAL_ENV=/tmp/isolated uv pip install --find-links dist "{dist}==${{version}}"
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: ${{{{ inputs.repository || 'pypi' }}}}
    permissions:
      # Required for Trusted Publishing.
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist
          repository-url: ${{{{ (inputs.repository == 'testpypi') && 'https://test.pypi.org/legacy/' || '' }}}}
          # Lets a re-run after a partial failure continue instead of aborting.
          skip-existing: true

  # A GitHub Release for the tag, carrying the same artifacts that went to
  # PyPI. Runs last so a release is only announced once the package is
  # actually installable.
  github-release:
    needs: publish
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v5
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist
      - name: Create the release
        env:
          GH_TOKEN: ${{{{ github.token }}}}
        run: |
          version="${{{{ github.ref_name }}}}"
          version="${{version#v}}"
          printf '%s\\n' \\
            "Published to PyPI: [\\`{dist}\\`](https://pypi.org/project/{dist}/${{version}}/)" \\
            "" \\
            '```bash' \\
            "uv pip install {dist}==${{version}}" \\
            '```' > notes.md
          gh release create "${{{{ github.ref_name }}}}" \\
            --title "${{{{ github.ref_name }}}}" \\
            --notes-file notes.md \\
            dist/*
"""


def build_pyproject(package_dir: Path, repo: str) -> str:
    """The package's own `pyproject.toml`, made standalone.

    Three edits: drop `[tool.uv.sources]` (it points every internal dependency
    at a workspace that no longer exists, which would make `uv sync` fail in a
    fresh clone), repoint `[project.urls]` at the package's own repository, and
    add the tooling configuration the monorepo root used to supply.
    """
    text = (package_dir / "pyproject.toml").read_text(encoding="utf-8")

    lines = text.split("\n")
    kept: list[str] = []
    in_uv_sources = False
    for line in lines:
        if line.strip().startswith("["):
            in_uv_sources = line.strip() == "[tool.uv.sources]"
        if not in_uv_sources:
            kept.append(line)
    text = "\n".join(kept).rstrip("\n")

    url = f"https://github.com/{OWNER}/{repo}"
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("Homepage =", "Repository =")):
            out.append(f"{stripped.split(' =')[0]} = \"{url}\"")
        elif stripped.startswith("Issues ="):
            out.append(f'Issues = "{url}/issues"')
        elif stripped.startswith("Documentation ="):
            out.append(f'Documentation = "{url}/tree/main/docs"')
        else:
            out.append(line)

    # Merge the `dev` group into an existing `[dependency-groups]` rather than
    # emitting a second table with the same name, which is invalid TOML.
    if any(line.strip() == "[dependency-groups]" for line in out):
        merged: list[str] = []
        for line in out:
            merged.append(line)
            if line.strip() == "[dependency-groups]":
                merged.extend(DEV_GROUP.rstrip("\n").split("\n"))
        out = merged
        tooling = TOOLING
    else:
        tooling = "\n[dependency-groups]\n" + DEV_GROUP + TOOLING

    return "\n".join(out).rstrip("\n") + "\n" + tooling


def build_readme(package_dir: Path, repo: str, dist: str) -> str:
    """The package README, with a note about where it came from."""
    text = (package_dir / "README.md").read_text(encoding="utf-8")
    note = (
        f"\n---\n\n"
        f"`{dist}` is developed in [{OWNER}/{repo}](https://github.com/{OWNER}/{repo}). "
        f"It was split out of the `pp` monorepo; sibling packages "
        f"(`pp-ai`, `pp-agent-core`, `pp-tui`, `pp-coding-agent`, ...) each live in their own\n"
        f"repository and are consumed from PyPI.\n"
    )
    return text.rstrip("\n") + "\n" + note


def split(out_root: Path, only: set[str] | None) -> list[tuple[str, str, Path]]:
    created: list[tuple[str, str, Path]] = []
    for package_name, (repo, dist) in REPOS.items():
        if only and repo not in only:
            continue
        package_dir = MONOREPO / "packages" / package_name
        if not package_dir.is_dir():
            raise SystemExit(f"missing package directory: {package_dir}")

        target = out_root / repo
        target.mkdir(parents=True, exist_ok=True)

        # Replace only what this script generates. A plain `rmtree(target)`
        # would take `.git` and `.venv` with it, which matters as soon as
        # these directories are real checkouts being regenerated in place.
        for entry in ("src", "tests", "docs", "examples", "scripts", ".github"):
            existing = target / entry
            if existing.is_dir():
                shutil.rmtree(existing)
        for entry in ("pyproject.toml", "README.md", "LICENSE", ".gitignore"):
            existing = target / entry
            if existing.exists():
                existing.unlink()

        for entry in ("src", "tests", "docs", "examples", "scripts"):
            source = package_dir / entry
            if source.is_dir():
                shutil.copytree(source, target / entry, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        shutil.copy2(package_dir / "LICENSE", target / "LICENSE")
        (target / "README.md").write_text(build_readme(package_dir, repo, dist), encoding="utf-8")
        (target / "pyproject.toml").write_text(build_pyproject(package_dir, repo), encoding="utf-8")
        (target / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

        workflows = target / ".github" / "workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        (workflows / "ci.yml").write_text(CI_YML, encoding="utf-8")
        (workflows / "release.yml").write_text(RELEASE_YML.format(dist=dist), encoding="utf-8")

        # Fail loudly rather than emit a repo whose pyproject cannot be read.
        parsed = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
        assert parsed["project"]["name"] == dist, f"{repo}: expected {dist}, got {parsed['project']['name']}"

        created.append((repo, dist, target))

    # Import *sections* depend on which packages are first-party, and that
    # changes with the layout: inside the monorepo every `pi_*` package is
    # first-party, while in a single-package repository only its own is and
    # the siblings become third-party. So the sorted order that is correct
    # here is not the one committed in the monorepo, and CI's
    # `ruff check`/`ruff format --check` would fail on generated code.
    for _repo, _dist, target in created:
        for command in (["ruff", "check", "--fix", "--quiet", "."], ["ruff", "format", "--quiet", "."]):
            result = subprocess.run(command, cwd=target, capture_output=True, text=True, check=False)
            # `ruff check --fix` exits non-zero when it leaves unfixable
            # findings; those are real and must not be hidden.
            if result.returncode != 0 and command[1] == "format":
                raise SystemExit(f"{target}: {' '.join(command)} failed\n{result.stderr}")
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="parent directory for the repositories")
    parser.add_argument("--only", nargs="*", help="repo names to generate (default: all)")
    args = parser.parse_args()

    out_root = args.out.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for repo, dist, target in split(out_root, set(args.only) if args.only else None):
        print(f"{repo:<18} {dist:<18} {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

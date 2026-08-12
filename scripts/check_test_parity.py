#!/usr/bin/env python3
"""Report which TypeScript tests still have no Python counterpart.

It also reports *partial* ports with `--depth`: a TypeScript test file can
be "ported" by a Python file that names it while covering only half its
cases, which the file-level count alone would hide.

This replaces the old `check_port.py`, which counted whether every `.ts`
*source* file was named by some Python module's docstring. That measured
"accounted for", not "behaves the same": a module could name its TypeScript
source in a docstring while implementing none of it, and the check would pass.

A port is only really verified when the original's *tests* run against it. So
this walks every `*.test.ts` in the TypeScript tree and reports whether a
Python test names it. Porting a TypeScript test is what forces the two
implementations to agree, because a ported test fails until they do.

Usage:
    uv run python scripts/check_test_parity.py            # summary per package
    uv run python scripts/check_test_parity.py --list     # every unported file
    uv run python scripts/check_test_parity.py --package ai --list
    uv run python scripts/check_test_parity.py --json
    uv run python scripts/check_test_parity.py --depth       # partial ports
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

PYPI_ROOT = Path(__file__).resolve().parent.parent
PP_ROOT = PYPI_ROOT.parent / "pp"

# TypeScript package -> the Python package that ports it.
PACKAGES: tuple[tuple[str, str], ...] = (
    ("protocol", "pi-protocol"),
    ("telemetry", "pi-telemetry"),
    ("ai", "pi-ai"),
    ("agent", "pi-agent"),
    ("tui", "pi-tui"),
    ("client", "pi-client"),
    ("server", "pi-server"),
    ("coding-agent", "pi-coding-agent"),
    ("evals", "pp-evals"),
)

# TypeScript tests that cannot run without provider credentials. They are
# skipped in the TypeScript run too, so they pin no verified behavior on
# either side and are reported separately rather than as outstanding work.
E2E_ONLY: frozenset[str] = frozenset(
    {
        "abort.test.ts",
        "anthropic-opus-4-8-smoke.test.ts",
        "anthropic-tool-name-normalization.test.ts",
        "context-overflow.test.ts",
        "cross-provider-handoff.test.ts",
        "empty.test.ts",
        "google-thinking-disable.test.ts",
        "image-tool-result.test.ts",
        "images.test.ts",
        "interleaved-thinking.test.ts",
        "openai-codex-cache-affinity-e2e.test.ts",
        "openai-responses-cache-affinity-e2e.test.ts",
        "openai-responses-reasoning-replay-e2e.test.ts",
        "openai-responses-tool-result-images.test.ts",
        "openrouter-cache-write-repro.test.ts",
        "responseid.test.ts",
        "stream.test.ts",
        "tokens.test.ts",
        "tool-call-id-normalization.test.ts",
        "tool-call-without-result.test.ts",
        "total-tokens.test.ts",
        "unicode-surrogate.test.ts",
        "xhigh.test.ts",
        "xiaomi-token-plan-ams-anthropic-empty-signature-smoke.test.ts",
        "zen.test.ts",
    }
)

_TEST_REFERENCE = re.compile(r"[A-Za-z0-9_/.-]+\.test\.ts")
_TS_CASE = re.compile(r"^\s*(?:it|test)(?:\.\w+)?\s*\(", re.MULTILINE)
_PY_CASE = re.compile(r"^\s*(?:async )?def test_", re.MULTILINE)


@dataclass
class PackageParity:
    ts_package: str
    py_package: str
    ported: list[str] = field(default_factory=list)
    unported: list[str] = field(default_factory=list)
    e2e_only: list[str] = field(default_factory=list)

    @property
    def portable_total(self) -> int:
        """Test files that could be ported: everything except the credential-gated ones."""
        return len(self.ported) + len(self.unported)

    @property
    def percent(self) -> float:
        return (len(self.ported) / self.portable_total * 100) if self.portable_total else 100.0


@dataclass
class DepthGap:
    """A TypeScript test file whose Python counterpart covers fewer cases."""

    ts_package: str
    test_file: str
    ts_cases: int
    py_cases: int


def collect_test_references() -> dict[str, list[Path]]:
    """Map each referenced `*.test.ts` basename to the Python files naming it."""
    references: dict[str, list[Path]] = {}
    for path in (PYPI_ROOT / "packages").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _TEST_REFERENCE.findall(text):
            references.setdefault(Path(match).name, []).append(path)
    return references


def measure_depth(ts_package: str, references: dict[str, list[Path]]) -> list[DepthGap]:
    """Find ported test files whose Python counterpart declares fewer cases.

    Counting `it(...)` against `def test_` is a heuristic -- `parametrize`
    collapses many cases into one function, so a smaller Python count is not
    proof of a gap. It is a reliable enough signal to point a reviewer at the
    files worth checking by hand.
    """
    gaps: list[DepthGap] = []
    package_root = PP_ROOT / "packages" / ts_package
    if not package_root.is_dir():
        return gaps

    for path in sorted(package_root.rglob("*.test.ts")):
        if "node_modules" in path.parts or path.name in E2E_ONLY:
            continue
        python_files = references.get(path.name)
        if not python_files:
            continue
        ts_cases = len(_TS_CASE.findall(path.read_text(encoding="utf-8", errors="replace")))
        py_cases = sum(
            len(_PY_CASE.findall(py.read_text(encoding="utf-8", errors="replace"))) for py in python_files
        )
        if py_cases < ts_cases:
            gaps.append(
                DepthGap(
                    ts_package=ts_package,
                    test_file=path.relative_to(package_root).as_posix(),
                    ts_cases=ts_cases,
                    py_cases=py_cases,
                )
            )
    return gaps


def collect_referenced_tests() -> set[str]:
    """Basenames of every `*.test.ts` named by any Python file in the workspace.

    Matched across the whole workspace rather than per package: a Python test
    may legitimately cover a TypeScript test from a neighbouring package.
    """
    referenced: set[str] = set()
    for path in (PYPI_ROOT / "packages").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        referenced.update(Path(match).name for match in _TEST_REFERENCE.findall(text))
    return referenced


def measure(ts_package: str, py_package: str, referenced: set[str]) -> PackageParity:
    parity = PackageParity(ts_package=ts_package, py_package=py_package)
    package_root = PP_ROOT / "packages" / ts_package
    if not package_root.is_dir():
        return parity

    for path in sorted(package_root.rglob("*.test.ts")):
        if "node_modules" in path.parts:
            continue
        relative = path.relative_to(package_root).as_posix()
        if path.name in E2E_ONLY:
            parity.e2e_only.append(relative)
        elif path.name in referenced:
            parity.ported.append(relative)
        else:
            parity.unported.append(relative)
    return parity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List every unported test file")
    parser.add_argument("--package", help="Restrict to one TypeScript package")
    parser.add_argument("--depth", action="store_true", help="Report ported files that cover fewer cases")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON")
    args = parser.parse_args(argv)

    if not PP_ROOT.is_dir():
        print(f"TypeScript tree not found at {PP_ROOT}", file=sys.stderr)
        return 2

    referenced = collect_referenced_tests()
    selected = [pair for pair in PACKAGES if args.package in (None, pair[0])]
    if not selected:
        print(f"Unknown package: {args.package}", file=sys.stderr)
        return 2

    results = [measure(ts, py, referenced) for ts, py in selected]

    if args.as_json:
        print(
            json.dumps(
                [
                    {
                        "ts_package": r.ts_package,
                        "py_package": r.py_package,
                        "ported": len(r.ported),
                        "unported": len(r.unported),
                        "e2e_only": len(r.e2e_only),
                        "percent": round(r.percent, 1),
                        "unported_files": r.unported,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        return 0

    print(f"{'package':16} {'ported':>18}  {'e2e-only':>9}")
    total_ported = total_portable = total_e2e = 0
    for result in results:
        total_ported += len(result.ported)
        total_portable += result.portable_total
        total_e2e += len(result.e2e_only)
        summary = f"{len(result.ported)}/{result.portable_total} ({result.percent:.0f}%)"
        print(f"{result.ts_package:16} {summary:>18}  {len(result.e2e_only):>9}")
        if args.list and result.unported:
            for name in result.unported:
                print(f"    - {name}")

    percent = (total_ported / total_portable * 100) if total_portable else 100.0
    print(f"{'TOTAL':16} {f'{total_ported}/{total_portable} ({percent:.0f}%)':>18}  {total_e2e:>9}")

    if args.depth:
        references = collect_test_references()
        gaps = [gap for ts, _ in selected for gap in measure_depth(ts, references)]
        print(f"\n{len(gaps)} ported test file(s) appear to cover fewer cases than the TypeScript:")
        for gap in sorted(gaps, key=lambda g: g.ts_cases - g.py_cases, reverse=True):
            print(f"  {gap.ts_package}/{gap.test_file}: {gap.ts_cases} TS cases vs {gap.py_cases} Python")
        if gaps:
            print("\n(Heuristic: `parametrize` collapses cases, so verify by hand before acting.)")
    if total_e2e:
        print(f"\n{total_e2e} test files need provider credentials; they are skipped in the TypeScript run too.")
    return 0 if total_ported == total_portable else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# Run the full check: lint, format check, tests and coverage.
# Mirrors `npm run check` + `npm test` in the TypeScript project.
set -euo pipefail

cd "$(dirname "$0")"

echo "== ruff check =="
uv run ruff check packages/

echo
echo "== version consistency =="
# The nine packages release in lockstep with `==` pinned internal deps; a
# hand-edited version that only lands in one file has to fail here rather than
# at publish time. See scripts/set_version.py.
uv run python scripts/set_version.py --check

echo
echo "== cross-package dependency declarations =="
# `uv sync --all-packages` hides an undeclared sibling import; a standalone
# wheel install does not. See scripts/check_dependencies.py.
uv run python scripts/check_dependencies.py

echo
echo "== ruff format --check =="
uv run ruff format --check packages/

echo
echo "== pytest + coverage =="
uv run pytest packages/ --cov --cov-report=term-missing --cov-report=html:.coverage-html

echo
echo "== test parity with the TypeScript suite =="
# Informational: a port is only verified once the original's tests run against
# it, so this reports what is still outstanding. It does not fail the check.
uv run python scripts/check_test_parity.py || true

echo
echo "Coverage HTML report: .coverage-html/index.html"

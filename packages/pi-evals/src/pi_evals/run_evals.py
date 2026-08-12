"""The `pp-evals` runner: resolve the default model, then run pytest.

Python port of `packages/evals/scripts/run-evals.mjs`.

Same contract as the Node runner: `--provider`/`--model` (or `--provider=`/
`--model=`) must be supplied together and take precedence over the
`PI_PROVIDER`/`PI_MODEL` environment variables, which must themselves be
supplied together. Any remaining arguments are forwarded to the test runner
(pytest here, Vitest there). Each invocation creates a fresh
`.eval/<timestamp>_<uuid>/` artifact directory, exported as
`PI_EVAL_ARTIFACT_DIR`, unless that variable is already set.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
"""`packages/pp-evals`, the equivalent of the Node runner's `packageRoot`."""

EVALS_PATH = Path(__file__).resolve().parent / "evals"


@dataclass
class RunnerArguments:
    provider: str | None = None
    model: str | None = None
    has_cli_model_selection: bool = False
    forwarded: list[str] = field(default_factory=list)


class RunnerError(Exception):
    """A usage error; `main` prints it to stderr and exits with status 1."""


def parse_arguments(argv: list[str]) -> RunnerArguments:
    """Port of the runner's argument scan."""
    parsed = RunnerArguments()
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--provider", "--model"):
            value = argv[index + 1] if index + 1 < len(argv) else None
            if not value or value.startswith("-"):
                raise RunnerError(f"Missing value for {argument}")
            if argument == "--provider":
                parsed.provider = value
            else:
                parsed.model = value
            parsed.has_cli_model_selection = True
            index += 2
            continue
        if argument.startswith("--provider="):
            parsed.provider = argument[len("--provider=") :]
            parsed.has_cli_model_selection = True
            index += 1
            continue
        if argument.startswith("--model="):
            parsed.model = argument[len("--model=") :]
            parsed.has_cli_model_selection = True
            index += 1
            continue
        parsed.forwarded.append(argument)
        index += 1
    return parsed


def resolve_default_model(parsed: RunnerArguments, environment: dict[str, str]) -> tuple[str | None, str | None]:
    """Port of the runner's provider/model resolution.

    A CLI selection needs both halves; otherwise both halves come from the
    environment, and supplying only one is an error. No default at all is
    allowed: every executed harness may configure its own model.
    """
    provider = (parsed.provider or "").strip() or None
    model = (parsed.model or "").strip() or None
    if parsed.has_cli_model_selection:
        if not provider or not model:
            raise RunnerError("CLI model selection requires both --provider and --model.")
        return provider, model
    provider = (environment.get("PI_PROVIDER") or "").strip() or None
    model = (environment.get("PI_MODEL") or "").strip() or None
    if bool(provider) != bool(model):
        raise RunnerError("Default model selection requires both PI_PROVIDER and PI_MODEL.")
    return provider, model


def default_artifact_directory(environment: dict[str, str]) -> Path:
    configured = environment.get("PI_EVAL_ARTIFACT_DIR")
    if configured:
        return (PACKAGE_ROOT / configured).resolve()
    stamp = datetime.now(UTC).isoformat().replace(":", "-")
    return (PACKAGE_ROOT / ".eval" / f"{stamp}_{uuid.uuid4()}").resolve()


EVAL_FILE_PATTERN = "*_eval.py"
"""Port of `vitest.config.ts`'s `include: ["src/**/*.eval.ts"]`.

Eval modules are named `*_eval.py`, which pytest's default `python_files`
would not collect from a directory, so the runner widens the pattern.
"""


def _is_test_target(argument: str) -> bool:
    """Whether a forwarded argument selects test files itself.

    `-k`-style filters must still run against the eval modules, exactly as
    `npm run eval -- -t "..."` filters within `vitest.config.ts`'s `include`.
    Only an explicit path (or `path::case` node id) replaces that default.
    """
    if argument.startswith("-"):
        return False
    path = argument.split("::", 1)[0]
    return Path(path).exists()


def build_command(forwarded: list[str]) -> list[str]:
    """Run pytest on the eval modules, plus whatever the caller forwarded."""
    targets = forwarded if any(_is_test_target(argument) for argument in forwarded) else [*forwarded, str(EVALS_PATH)]
    return [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        f"python_files={EVAL_FILE_PATTERN} test_*.py",
        *targets,
    ]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    environment = dict(os.environ)
    try:
        parsed = parse_arguments(arguments)
        provider, model = resolve_default_model(parsed, environment)
    except RunnerError as error:
        print(str(error), file=sys.stderr)
        return 1

    artifact_directory = default_artifact_directory(environment)
    artifact_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    default_model = f"{provider}/{model}" if provider and model else "none"
    print(f"[eval] default-model={default_model}", file=sys.stderr)
    print(f"[eval] artifacts={artifact_directory}", file=sys.stderr)

    child_environment = dict(environment)
    child_environment["PI_EVAL_ARTIFACT_DIR"] = str(artifact_directory)
    if provider and model:
        child_environment["PI_PROVIDER"] = provider
        child_environment["PI_MODEL"] = model
    else:
        child_environment.pop("PI_PROVIDER", None)
        child_environment.pop("PI_MODEL", None)

    completed = subprocess.run(
        build_command(parsed.forwarded),
        cwd=str(PACKAGE_ROOT),
        env=child_environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["RunnerArguments", "RunnerError", "build_command", "main", "parse_arguments", "resolve_default_model"]

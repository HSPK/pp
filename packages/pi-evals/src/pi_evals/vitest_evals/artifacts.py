"""Eval artifact registration and on-disk persistence.

Python port of `packages/evals/src/vitest-evals/artifacts.ts`.

Two artifact kinds are recorded against the running test:
`@earendil-works/pi-evals:session` (the native Pi session JSONL snapshot the Pi
harness takes before deleting its temporary workspace) and
`@earendil-works/pi-evals:source` (an eval-specific source file, such as an
extension the model wrote). `persist_eval_artifact_references` writes their
attachments under `<artifact dir>/{sessions,sources}/<sha256 of run id>/<name>`
with the same 0700/0600 modes and the same relative paths the TypeScript
reporter produces, so both implementations' `.eval/` directories are directly
comparable.

Vitest's `recordArtifact(task, artifact)` has no pytest equivalent; here the
artifact list lives on `pi_evals.harness.EvalTask`, which
`pi_evals.vitest_evals.reporter` reads back for the running test.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

from pi_evals.harness import EvalTask, HarnessRun, TestArtifact, TestArtifactAttachment

PI_SESSION_SNAPSHOT_ARTIFACT = "piSessionJsonl"
"""Artifact name. Kept in its TypeScript spelling: it is a JSON key on disk."""

SESSION_ARTIFACT_TYPE = "@earendil-works/pi-evals:session"
SOURCE_ARTIFACT_TYPE = "@earendil-works/pi-evals:source"


def record_eval_session_artifact(task: EvalTask, run: HarnessRun) -> None:
    """Port of `recordEvalSessionArtifact`.

    Does nothing when the run recorded no session snapshot (for example a
    harness that failed before its session existed).
    """
    run_id = run.artifacts.get("runId")
    session = run.artifacts.get(PI_SESSION_SNAPSHOT_ARTIFACT)
    if session is None:
        return
    if not isinstance(run_id, str) or not isinstance(session, str):
        raise TypeError("Pi eval session artifact metadata is invalid.")
    task.artifacts.append(
        TestArtifact(
            type=SESSION_ARTIFACT_TYPE,
            run_id=run_id,
            attachments=[
                TestArtifactAttachment(
                    name="session.jsonl",
                    content_type="application/jsonl",
                    body=session,
                )
            ],
        )
    )


def record_eval_source_artifact(task: EvalTask, run_id: str, attachment: TestArtifactAttachment) -> None:
    """Port of `recordEvalSourceArtifact`."""
    task.artifacts.append(TestArtifact(type=SOURCE_ARTIFACT_TYPE, run_id=run_id, attachments=[attachment]))


def persist_eval_artifact_references(
    artifacts: Sequence[TestArtifact],
    run_id: str,
    artifact_directory: str | os.PathLike[str],
) -> list[dict[str, str]]:
    """Port of `persistEvalArtifactReferences`.

    Returns `{"name", "path"}` records with `path` relative to
    `artifact_directory`, exactly as the TypeScript writes them into
    `runs.jsonl`.
    """
    root = Path(artifact_directory)
    references: list[dict[str, str]] = []
    for artifact in artifacts:
        if artifact.type not in (SESSION_ARTIFACT_TYPE, SOURCE_ARTIFACT_TYPE) or artifact.run_id != run_id:
            continue
        category = "sessions" if artifact.type == SESSION_ARTIFACT_TYPE else "sources"
        for attachment in artifact.attachments:
            name = os.path.basename(attachment.name)
            if name != attachment.name:
                raise TypeError(f"Invalid eval artifact name: {attachment.name}")
            directory = root / category / hashlib.sha256(run_id.encode("utf-8")).hexdigest()
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = directory / name
            path.write_text(attachment.body, encoding="utf-8")
            path.chmod(0o600)
            references.append({"name": name, "path": str(path.relative_to(root))})
    return references


__all__ = [
    "PI_SESSION_SNAPSHOT_ARTIFACT",
    "SESSION_ARTIFACT_TYPE",
    "SOURCE_ARTIFACT_TYPE",
    "persist_eval_artifact_references",
    "record_eval_session_artifact",
    "record_eval_source_artifact",
]

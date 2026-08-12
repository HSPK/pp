"""Tests for eval artifact registration and persistence.

Python port of `packages/evals/test/vitest-evals/artifacts.test.ts`.
"""

from __future__ import annotations

import re
from pathlib import Path

from pi_evals.harness import EvalTask, HarnessRun, TestArtifact, TestArtifactAttachment
from pi_evals.vitest_evals.artifacts import (
    persist_eval_artifact_references,
    record_eval_session_artifact,
    record_eval_source_artifact,
)


def test_records_session_and_source_artifacts_against_the_explicit_test_task() -> None:
    task = EvalTask(name="records artifacts")
    run_id = "run-1"
    record_eval_session_artifact(
        task,
        HarnessRun(artifacts={"runId": run_id, "piSessionJsonl": '{"type":"session"}\n'}),
    )
    record_eval_source_artifact(
        task,
        run_id,
        TestArtifactAttachment(
            name="hello.py",
            content_type="text/x-python",
            body="def pi_extension(pi):\n    return None\n",
        ),
    )

    assert (
        TestArtifact(
            type="@earendil-works/pi-evals:session",
            run_id=run_id,
            attachments=[
                TestArtifactAttachment(
                    name="session.jsonl",
                    content_type="application/jsonl",
                    body='{"type":"session"}\n',
                    body_encoding="utf-8",
                )
            ],
        )
        in task.artifacts
    )
    assert (
        TestArtifact(
            type="@earendil-works/pi-evals:source",
            run_id=run_id,
            attachments=[
                TestArtifactAttachment(
                    name="hello.py",
                    content_type="text/x-python",
                    body="def pi_extension(pi):\n    return None\n",
                    body_encoding="utf-8",
                )
            ],
        )
        in task.artifacts
    )


def test_ignores_a_run_without_a_session_snapshot() -> None:
    task = EvalTask()
    record_eval_session_artifact(task, HarnessRun(artifacts={"runId": "run-1"}))
    assert task.artifacts == []


def test_persists_and_selects_attachments_belonging_to_the_reported_run(tmp_path: Path) -> None:
    references = persist_eval_artifact_references(
        [
            TestArtifact(
                type="@earendil-works/pi-evals:session",
                run_id="run-1",
                attachments=[
                    TestArtifactAttachment(
                        name="session.jsonl",
                        content_type="application/jsonl",
                        body='{"type":"session"}\n',
                    )
                ],
            ),
            TestArtifact(type="@earendil-works/pi-evals:session", run_id="run-2", attachments=[]),
            TestArtifact(
                type="@earendil-works/pi-evals:source",
                run_id="run-1",
                attachments=[
                    TestArtifactAttachment(
                        name="hello.py",
                        content_type="text/x-python",
                        body="def pi_extension(pi):\n    return None\n",
                    )
                ],
            ),
            TestArtifact(type="internal:annotation", annotation={"message": "other", "type": "info"}),
        ],
        "run-1",
        tmp_path,
    )

    assert [reference["name"] for reference in references] == ["session.jsonl", "hello.py"]
    assert re.fullmatch(r"sessions/[a-f0-9]{64}/session\.jsonl", references[0]["path"])
    assert re.fullmatch(r"sources/[a-f0-9]{64}/hello\.py", references[1]["path"])
    for reference in references:
        expected = (
            '{"type":"session"}\n'
            if reference["name"] == "session.jsonl"
            else "def pi_extension(pi):\n    return None\n"
        )
        assert (tmp_path / reference["path"]).read_text(encoding="utf-8") == expected

"""Python port of `packages/coding-agent/test/suite/regressions/5661-uppercase-header-values.test.ts`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness import create_harness
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.migrations import run_migrations

MODELS_JSON = {
    "providers": {
        "my-provider": {
            "baseUrl": "https://example.com/v1",
            "apiKey": "CUSTOM_API_KEY",
            "api": "openai-completions",
            "headers": {"Authorization": "BEARER"},
            "models": [{"id": "my-model"}],
        }
    }
}


async def test_keeps_uppercase_header_strings_as_literals_during_startup_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = await create_harness(tmp_path, with_configured_auth=False)
    try:
        for key in ("CUSTOM_API_KEY", "BEARER"):
            monkeypatch.setenv(key, f"env-{key}")

        models_path = harness.temp_dir / "models.json"
        models_path.write_text(f"{json.dumps(MODELS_JSON, indent=2)}\n", encoding="utf-8")

        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(harness.temp_dir))
        run_migrations(str(harness.temp_dir), str(harness.temp_dir))

        migrated = json.loads(models_path.read_text(encoding="utf-8"))
        assert migrated["providers"]["my-provider"]["apiKey"] == "CUSTOM_API_KEY"
        assert migrated["providers"]["my-provider"]["headers"]["Authorization"] == "BEARER"

        runtime = await ModelRuntime.create(agent_dir=harness.temp_dir, models_path=models_path)
        model = runtime.get_model("my-provider", "my-model")
        assert model is not None

        auth = await runtime.get_auth(model)
        assert auth is not None
        assert auth.auth.api_key == "CUSTOM_API_KEY"
        assert auth.auth.headers == {"Authorization": "BEARER"}
    finally:
        harness.cleanup()

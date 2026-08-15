"""Python port of `packages/coding-agent/test/config-value-migration.test.ts`.

Pins that startup migrations never rewrite config *values*: bare uppercase
strings in `auth.json` and `models.json` are literals, not env var references,
so `run_migrations` must leave them byte-for-byte alone, print nothing, and
survive an unparseable `models.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.migrations import run_migrations


async def test_leaves_uppercase_auth_json_api_key_values_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "auth.json").write_text(
        json.dumps(
            {
                "anthropic": {"type": "api_key", "key": "ANTHROPIC_API_KEY"},
                "openai": {"type": "api_key", "key": "$OPENAI_API_KEY"},
                "opencode": {"type": "api_key", "key": "public"},
                "github": {"type": "oauth", "access": "ACCESS_TOKEN", "refresh": "REFRESH_TOKEN", "expires": 1},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    capsys.readouterr()

    run_migrations(str(tmp_path), str(agent_dir))

    migrated = json.loads((agent_dir / "auth.json").read_text(encoding="utf-8"))
    assert migrated["anthropic"]["key"] == "ANTHROPIC_API_KEY"
    assert migrated["openai"]["key"] == "$OPENAI_API_KEY"
    assert migrated["opencode"]["key"] == "public"
    assert migrated["github"]["access"] == "ACCESS_TOKEN"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("name", "content"),
    [("malformed", '{\n  "providers": {\n'), ("blank", "")],
)
async def test_does_not_throw_on_broken_models_json_during_migrations(tmp_path: Path, name: str, content: str) -> None:
    agent_dir = tmp_path / f"agent-{name}"
    agent_dir.mkdir()
    models_path = agent_dir / "models.json"
    models_path.write_text(content, encoding="utf-8")

    run_migrations(str(tmp_path), str(agent_dir))

    assert models_path.read_text(encoding="utf-8") == content
    runtime = await ModelRuntime.create(
        agent_dir=str(agent_dir),
        auth_path=str(agent_dir / "auth.json"),
        models_path=str(models_path),
    )
    load_error = runtime.get_error()
    assert load_error is not None
    assert "Failed to parse models.json" in load_error
    assert f"File: {models_path}" in load_error


async def test_leaves_uppercase_models_json_api_key_and_header_values_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    env = {key: f"env-{key}" for key in ("CUSTOM_API_KEY", "HEADER_API_KEY", "MODEL_API_KEY", "OVERRIDE_API_KEY")}
    models_path = agent_dir / "models.json"
    models_path.write_text(
        json.dumps(
            {
                "providers": {
                    "custom-provider": {
                        "baseUrl": "https://example.com/v1",
                        "apiKey": "CUSTOM_API_KEY",
                        "api": "openai-completions",
                        "headers": {"x-api-key": "HEADER_API_KEY", "x-literal": "literal"},
                        "models": [{"id": "model-a", "headers": {"x-model-key": "MODEL_API_KEY"}}],
                        "modelOverrides": {"model-b": {"headers": {"x-override-key": "OVERRIDE_API_KEY"}}},
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    capsys.readouterr()

    run_migrations(str(tmp_path), str(agent_dir))

    migrated = json.loads(models_path.read_text(encoding="utf-8"))
    provider = migrated["providers"]["custom-provider"]
    assert provider["apiKey"] == "CUSTOM_API_KEY"
    assert provider["headers"]["x-api-key"] == "HEADER_API_KEY"
    assert provider["headers"]["x-literal"] == "literal"
    assert provider["models"][0]["headers"]["x-model-key"] == "MODEL_API_KEY"
    assert provider["modelOverrides"]["model-b"]["headers"]["x-override-key"] == "OVERRIDE_API_KEY"
    assert capsys.readouterr().out == ""

    runtime = await ModelRuntime.create(
        agent_dir=str(agent_dir),
        auth_path=str(agent_dir / "auth.json"),
        models_path=str(models_path),
        env=env,
    )
    model = runtime.get_model("custom-provider", "model-a")
    assert model is not None

    provider_auth = await runtime.get_auth("custom-provider")
    assert provider_auth is not None
    assert provider_auth.auth.api_key == "CUSTOM_API_KEY"

    model_auth = await runtime.get_auth(model)
    assert model_auth is not None
    assert model_auth.auth.api_key == "CUSTOM_API_KEY"
    assert model_auth.auth.headers == {
        "x-api-key": "HEADER_API_KEY",
        "x-literal": "literal",
        "x-model-key": "MODEL_API_KEY",
    }

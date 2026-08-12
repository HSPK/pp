"""Python port of `packages/coding-agent/test/experimental.test.ts`."""

from __future__ import annotations

import pytest
from pi_coding_agent.core.experimental import are_experimental_features_enabled


def test_returns_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_EXPERIMENTAL", raising=False)

    assert are_experimental_features_enabled() is False


def test_returns_false_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_EXPERIMENTAL", "")

    assert are_experimental_features_enabled() is False


def test_returns_true_when_set_to_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")

    assert are_experimental_features_enabled() is True


def test_returns_false_when_set_to_0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_EXPERIMENTAL", "0")

    assert are_experimental_features_enabled() is False


def test_returns_false_for_non_1_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_EXPERIMENTAL", "true")

    assert are_experimental_features_enabled() is False

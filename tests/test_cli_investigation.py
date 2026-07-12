"""CLI benchmark-investigation: deterministic path and remote opt-in gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from inverse_agent.cli import benchmark_investigation_command


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "model": None,
        "model_base_url": None,
        "model_allow_remote": False,
        "output": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_deterministic_path_passes_and_writes_output(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    code = benchmark_investigation_command(_args(output=str(out)))
    assert code == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["planner"] == "deterministic"
    assert summary["gate_passed"] is True
    assert summary["model_provenance"] is None


def test_remote_endpoint_without_dual_optin_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Flag set but env not set -> allow_remote stays False -> a remote https
    # endpoint is refused by validate_model_endpoint before any model call.
    monkeypatch.delenv("INVERSE_AGENT_MODEL_ALLOW_REMOTE", raising=False)
    with pytest.raises(ValueError, match="dual opt-in"):
        benchmark_investigation_command(
            _args(
                model="some-model",
                model_base_url="https://api.example.test/v1",
                model_allow_remote=True,
            )
        )


def test_remote_endpoint_env_only_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Env set but flag not passed -> still refused.
    monkeypatch.setenv("INVERSE_AGENT_MODEL_ALLOW_REMOTE", "1")
    with pytest.raises(ValueError, match="dual opt-in"):
        benchmark_investigation_command(
            _args(
                model="some-model",
                model_base_url="https://api.example.test/v1",
                model_allow_remote=False,
            )
        )

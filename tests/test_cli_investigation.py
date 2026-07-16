"""CLI benchmark-investigation: deterministic path and remote opt-in gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from inverse_agent.cli import benchmark_investigation_command, build_parser


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "model": None,
        "model_base_url": None,
        "model_context_tokens": None,
        "model_estimator_bytes_per_token": None,
        "model_reasoning_effort": None,
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
    assert summary["integrity_failures"] == []
    assert summary["model_provenance"] is None
    first = summary["variants"][0]
    assert first["physical_requests_used"] >= 1
    assert first["completion_tokens_charged"] == 0
    assert first["model_calls"] == []
    assert first["integrity_failures"] == []
    git_variant = next(
        item for item in summary["variants"] if item["case"] == "git_approval_replanning"
    )
    assert [item["status"] for item in git_variant["command_audit"]] == [
        "failed",
        "succeeded",
    ]


def test_explicit_empty_model_cannot_fall_back_to_scripted_gate() -> None:
    with pytest.raises(ValueError, match="non-empty identifier"):
        benchmark_investigation_command(_args(model=""))


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


def test_model_context_environment_requires_calibration_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INVERSE_AGENT_MODEL_CONTEXT_TOKENS", "20000")

    with pytest.raises(ValueError, match="must be one of"):
        benchmark_investigation_command(
            _args(
                model="some-model",
                model_base_url="http://127.0.0.1:1234/v1",
            )
        )


def test_model_run_requires_calibrated_token_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN", raising=False)

    with pytest.raises(ValueError, match="requires a calibrated estimator"):
        benchmark_investigation_command(
            _args(
                model="some-model",
                model_base_url="http://127.0.0.1:1234/v1",
                model_context_tokens=24_576,
            )
        )


def test_model_run_requires_calibrated_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVERSE_AGENT_MODEL_CONTEXT_TOKENS", raising=False)

    with pytest.raises(ValueError, match="requires a calibrated model context"):
        benchmark_investigation_command(
            _args(
                model="some-model",
                model_base_url="http://127.0.0.1:1234/v1",
                model_estimator_bytes_per_token=2.0,
            )
        )


def test_model_run_requires_explicit_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVERSE_AGENT_MODEL_REASONING_EFFORT", raising=False)

    with pytest.raises(ValueError, match="requires an explicit reasoning-effort"):
        benchmark_investigation_command(
            _args(
                model="some-model",
                model_base_url="http://127.0.0.1:1234/v1",
                model_context_tokens=24_576,
                model_estimator_bytes_per_token=2.0,
            )
        )


def test_model_context_help_matches_required_live_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build_parser().parse_args(["benchmark-investigation", "--help"])

    help_text = " ".join(capsys.readouterr().out.split())
    assert "INVERSE_AGENT_MODEL_CONTEXT_TOKENS for model runs" in help_text
    assert "INVERSE_AGENT_MODEL_REASONING_EFFORT" in help_text
    assert "or 16384" not in help_text

import json
from pathlib import Path

import pytest

from inverse_agent.cli import main
from inverse_agent.dogfood import evaluate_workspace, save_evaluation

FIXTURES = Path(__file__).parent / "fixtures"


def test_django_dogfood_evaluation_builds_default_plan(tmp_path: Path) -> None:
    result = evaluate_workspace(FIXTURES / "django_project")
    assert result.passed
    assert result.domains[0].planned_tools == ("django.check", "django.test")
    output = tmp_path / "evaluation.json"
    save_evaluation(result, output)
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_ios_dogfood_evaluation_reports_missing_host_toolchain() -> None:
    result = evaluate_workspace(FIXTURES / "ios_project")
    if not result.passed:
        assert "xcodebuild" in " ".join(result.domains[0].unavailable)


def test_cli_evaluate_ignores_model_environment_without_opt_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INVERSE_AGENT_MODEL_NAME", "configured-but-incomplete")
    code = main(["evaluate", str(FIXTURES / "django_project")])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["passed"] is True


def test_cli_model_evaluation_redacts_errors_before_output_and_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SecretFailingPlanner:
        def plan(self, **_kwargs):
            raise ValueError("token=super-secret-evaluation-value")

    class Resolution:
        planner = SecretFailingPlanner()

    monkeypatch.setattr("inverse_agent.cli.resolve_planner", lambda **_kwargs: Resolution())
    output = tmp_path / "evaluation.json"
    code = main(
        [
            "evaluate",
            str(FIXTURES / "django_project"),
            "--use-model",
            "--output",
            str(output),
        ]
    )
    stdout = capsys.readouterr().out
    persisted = output.read_text(encoding="utf-8")

    assert code == 1
    assert "super-secret-evaluation-value" not in stdout
    assert "super-secret-evaluation-value" not in persisted
    assert "[REDACTED_SECRET]" in stdout
    assert "[REDACTED_SECRET]" in persisted

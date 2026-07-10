import json
from pathlib import Path

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

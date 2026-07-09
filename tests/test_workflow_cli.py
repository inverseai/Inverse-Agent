import json
from pathlib import Path

from inverse_agent.cli import main
from inverse_agent.eval import load_trace
from inverse_agent.models import AutonomyLevel, Domain, RunSpec, RunStatus
from inverse_agent.workflow import build_langgraph_workflow, run_django_replay


def test_django_replay_writes_trace(tmp_path: Path) -> None:
    workspace = Path(__file__).parent / "fixtures" / "django_project"
    spec = RunSpec(
        goal="Run fixture checks",
        workspace=workspace,
        domain=Domain.DJANGO,
        autonomy_level=AutonomyLevel.ASSISTED,
    )

    result = run_django_replay(spec, tmp_path)

    assert result.trace.status == RunStatus.SUCCEEDED
    trace_artifacts = [artifact for artifact in result.trace.artifacts if artifact.path]
    assert trace_artifacts
    assert load_trace(trace_artifacts[-1].path)["status"] == "succeeded"


def test_cli_profile_outputs_json(capsys) -> None:
    workspace = Path(__file__).parent / "fixtures" / "django_project"

    code = main(["profile", str(workspace)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "django" in payload["domains"]


def test_langgraph_workflow_optional() -> None:
    workflow = build_langgraph_workflow()
    assert workflow is None or hasattr(workflow, "invoke")


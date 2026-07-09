"""Workflow skeleton for approval-gated runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from inverse_agent.adapters.django import DjangoAdapter
from inverse_agent.eval import save_trace
from inverse_agent.models import Artifact, ArtifactKind, EvalTrace, RunSpec, RunStatus
from inverse_agent.policies import default_policy
from inverse_agent.runner import LocalRunner


@dataclass
class WorkflowResult:
    trace: EvalTrace
    artifacts: list[Artifact] = field(default_factory=list)


def run_django_replay(spec: RunSpec, output_dir: Path) -> WorkflowResult:
    """Run the first executable dogfood workflow against a Django workspace."""

    trace = EvalTrace(
        task_input=spec.goal,
        domain=spec.domain,
        baseline="fixture-django-replay",
        run_id=spec.run_id,
        status=RunStatus.RUNNING,
    )
    started = time.monotonic()
    policy = default_policy(spec.workspace)
    runner = LocalRunner(policy)
    adapter = DjangoAdapter()
    trace.record_action("profile", workspace=str(spec.workspace))
    check = adapter.run_checks(runner, spec.workspace)
    trace.record_action("django.check", ok=check.ok, reason=check.command.reason if check.command else "")
    trace.approvals.append(
        {
            "action": "django.test",
            "approved": True,
            "reason": "fixture replay permits executing workspace tests",
        }
    )
    test = adapter.run_tests(runner, spec.workspace, approved=True)
    trace.record_action("django.test", ok=test.ok, reason=test.command.reason if test.command else "")
    trace.status = RunStatus.SUCCEEDED if check.ok and test.ok else RunStatus.FAILED
    report = Artifact(
        kind=ArtifactKind.REPORT,
        summary=f"Django replay status: {trace.status.value}; check={check.ok}; test={test.ok}",
    )
    trace.record_artifact(report)
    trace_path = output_dir / f"{spec.run_id}.trace.json"
    file_artifact = Artifact(kind=ArtifactKind.TRACE, path=trace_path, summary="Eval trace JSON")
    trace.record_artifact(file_artifact)
    trace.duration_seconds = time.monotonic() - started
    save_trace(trace, trace_path)
    return WorkflowResult(trace=trace, artifacts=[report, file_artifact])


def build_langgraph_workflow() -> object | None:
    """Build a tiny LangGraph workflow when langgraph is installed.

    Tests do not require LangGraph to be importable; dependency installation gates
    verify the real package in development environments.
    """

    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return None

    graph = StateGraph(dict)

    def plan(state: dict) -> dict:
        return {**state, "planned": True}

    graph.add_node("plan", plan)
    graph.set_entry_point("plan")
    graph.add_edge("plan", END)
    return graph.compile()

"""Durable, approval-interrupting LangGraph workflows for every supported domain."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.approvals import ApprovalAuthority
from inverse_agent.eval import save_trace
from inverse_agent.models import (
    Artifact,
    ArtifactKind,
    AutonomyLevel,
    Domain,
    EvalTrace,
    RunSpec,
    RunStatus,
)
from inverse_agent.planner import DeterministicPlanner, Planner
from inverse_agent.policies import default_policy
from inverse_agent.redaction import redact_text
from inverse_agent.runner import ApprovalNotRequired, CommandRequest, LocalRunner, PolicyViolation


class AgentState(TypedDict, total=False):
    run_id: str
    goal: str
    workspace: str
    domain: str
    autonomy_level: int
    available_tools: list[str]
    tool_commands: dict[str, list[str]]
    plan: list[str]
    plan_rationale: str
    action_index: int
    actions: list[dict[str, Any]]
    status: str
    error: str
    started_at: float
    planner_fingerprint: str
    duration_seconds: float
    trace_path: str


@dataclass
class WorkflowResult:
    trace: EvalTrace
    artifacts: list[Artifact] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None


class DurableAgentWorkflow:
    """A restart-safe graph whose risky nodes pause for a signed capability."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        trace_dir: Path,
        approval_authority: ApprovalAuthority,
        planner: Planner | None = None,
    ) -> None:
        checkpoint_path = checkpoint_path.resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_dir = trace_dir.resolve()
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.approval_authority = approval_authority
        self.planner = planner or DeterministicPlanner()
        self._connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._connection)
        self.graph = self._build_graph().compile(checkpointer=self._checkpointer)

    def close(self) -> None:
        self._connection.close()

    def start(self, spec: RunSpec) -> WorkflowResult:
        initial: AgentState = {
            "run_id": spec.run_id,
            "goal": spec.goal,
            "workspace": str(spec.workspace.resolve()),
            "domain": spec.domain.value,
            "autonomy_level": spec.autonomy_level.value,
            "action_index": 0,
            "actions": [],
            "status": RunStatus.PLANNED.value,
            "started_at": time.time(),
            "planner_fingerprint": spec.planner_fingerprint,
        }
        state = cast(AgentState, self.graph.invoke(initial, self._config(spec.run_id)))
        return self._result_from_state(state)

    def resume(self, run_id: str, approval_token: str) -> WorkflowResult:
        state = cast(
            AgentState,
            self.graph.invoke(Command(resume=approval_token), self._config(run_id)),
        )
        return self._result_from_state(state)

    def current(self, run_id: str) -> WorkflowResult:
        snapshot = self.graph.get_state(self._config(run_id))
        state = cast(AgentState, snapshot.values)
        if not state:
            raise KeyError(f"no workflow checkpoint for run: {run_id}")
        materialized = dict(state)
        pending = snapshot.interrupts[0].value if snapshot.interrupts else None
        if pending:
            materialized["__interrupt__"] = list(snapshot.interrupts)
        return self._result_from_state(cast(AgentState, materialized))

    def _build_graph(self) -> StateGraph[AgentState]:
        graph = StateGraph(AgentState)
        graph.add_node("profile", self._profile_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_edge(START, "profile")
        graph.add_conditional_edges(
            "profile",
            lambda state: "plan" if state["status"] == RunStatus.RUNNING.value else "finalize",
        )
        graph.add_conditional_edges(
            "plan",
            lambda state: (
                "finalize"
                if state["status"] != RunStatus.RUNNING.value
                or state["autonomy_level"] == AutonomyLevel.ADVISORY.value
                else "execute"
            ),
        )
        graph.add_conditional_edges(
            "execute",
            lambda state: (
                "execute"
                if state["status"] == RunStatus.RUNNING.value
                and state["action_index"] < len(state["plan"])
                else "finalize"
            ),
        )
        graph.add_edge("finalize", END)
        return graph

    def _profile_node(self, state: AgentState) -> AgentState:
        workspace = Path(state["workspace"])
        domain = Domain(state["domain"])
        profile = detect_workspace(workspace)
        if domain not in profile.domains:
            return {
                "status": RunStatus.FAILED.value,
                "error": f"workspace does not contain the requested {domain.value} domain",
            }
        available = {
            name: command
            for name, command in profile.commands.items()
            if name.startswith(f"{domain.value}.")
        }
        if not available:
            reason = "; ".join(profile.unavailable_tools.values()) or "no executable tools detected"
            return {"status": RunStatus.FAILED.value, "error": reason}
        return {
            "available_tools": sorted(available),
            "tool_commands": available,
            "status": RunStatus.RUNNING.value,
        }

    def _plan_node(self, state: AgentState) -> AgentState:
        profile = detect_workspace(Path(state["workspace"]))
        try:
            plan = self.planner.plan(
                goal=state["goal"],
                domain=Domain(state["domain"]),
                profile=profile,
                available_tools=tuple(state["available_tools"]),
            )
        except (TypeError, ValueError) as exc:
            error = redact_text(str(exc)).text
            return {"status": RunStatus.FAILED.value, "error": f"planning failed: {error}"}
        return {
            "plan": [action.tool_name for action in plan.actions],
            "plan_rationale": plan.rationale,
            "action_index": 0,
            "status": (
                RunStatus.SUCCEEDED.value
                if state["autonomy_level"] == AutonomyLevel.ADVISORY.value
                else RunStatus.RUNNING.value
            ),
        }

    def _execute_node(self, state: AgentState) -> AgentState:
        index = state["action_index"]
        tool_name = state["plan"][index]
        argv = tuple(state["tool_commands"][tool_name])
        workspace = Path(state["workspace"])
        domain = Domain(state["domain"])
        runner = LocalRunner(default_policy(workspace), self.approval_authority)
        request = CommandRequest(argv=argv, cwd=workspace, domain=domain)
        try:
            challenge = runner.approval_challenge(request)
            token = interrupt({"kind": "command_approval", **asdict(challenge)})
            request = CommandRequest(
                argv=argv,
                cwd=workspace,
                domain=domain,
                approval_token=str(token),
            )
        except ApprovalNotRequired:
            pass
        except PolicyViolation as exc:
            return {"status": RunStatus.REFUSED.value, "error": str(exc)}

        result = runner.run(request)
        actions = [
            *state.get("actions", []),
            {
                "tool": tool_name,
                "status": result.status.value,
                "rule": result.rule,
                "reason": result.reason,
                "returncode": result.returncode,
                "approval_id": result.approval_id,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        ]
        if result.status != RunStatus.SUCCEEDED:
            return {"actions": actions, "status": result.status.value, "error": result.reason}
        return {
            "actions": actions,
            "action_index": index + 1,
            "status": RunStatus.RUNNING.value,
        }

    def _finalize_node(self, state: AgentState) -> AgentState:
        status = RunStatus(state["status"])
        if status == RunStatus.RUNNING:
            status = RunStatus.SUCCEEDED
        duration = max(0.0, time.time() - state["started_at"])
        trace = self._trace_from_state(
            {**state, "status": status.value, "duration_seconds": duration}
        )
        trace_path = self.trace_dir / f"{state['run_id']}.trace.json"
        trace.record_artifact(
            Artifact(kind=ArtifactKind.TRACE, path=trace_path, summary="Eval trace JSON")
        )
        save_trace(trace, trace_path)
        return {
            "status": status.value,
            "duration_seconds": duration,
            "trace_path": str(trace_path),
        }

    def _result_from_state(self, state: AgentState) -> WorkflowResult:
        raw_state = cast(dict[str, Any], state)
        interrupts = raw_state.get("__interrupt__", [])
        pending = interrupts[0].value if interrupts else None
        materialized = dict(state)
        if pending:
            materialized["status"] = RunStatus.WAITING_FOR_APPROVAL.value
        trace = self._trace_from_state(materialized)
        artifacts = list(trace.artifacts)
        return WorkflowResult(trace=trace, artifacts=artifacts, pending_approval=pending)

    @staticmethod
    def _trace_from_state(state: dict[str, Any]) -> EvalTrace:
        status = RunStatus(state.get("status", RunStatus.PLANNED.value))
        trace = EvalTrace(
            task_input=state["goal"],
            domain=Domain(state["domain"]),
            baseline="durable-agent-workflow-v1",
            run_id=state["run_id"],
            status=status,
            duration_seconds=float(state.get("duration_seconds", 0.0)),
            planner_fingerprint=str(state.get("planner_fingerprint", "deterministic")),
        )
        for action in state.get("actions", []):
            trace.record_action(action["tool"], **action)
            if action.get("approval_id"):
                trace.approvals.append(
                    {
                        "approval_id": action["approval_id"],
                        "action": action["tool"],
                        "consumed": True,
                    }
                )
        if state.get("error"):
            trace.record_action("workflow.error", reason=state["error"])
        trace_path = state.get("trace_path")
        if trace_path:
            trace.record_artifact(
                Artifact(kind=ArtifactKind.TRACE, path=Path(trace_path), summary="Eval trace JSON")
            )
        return trace

    @staticmethod
    def _config(run_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": run_id}}


def build_langgraph_workflow(
    *,
    checkpoint_path: Path,
    trace_dir: Path,
    approval_authority: ApprovalAuthority,
    planner: Planner | None = None,
) -> DurableAgentWorkflow:
    return DurableAgentWorkflow(
        checkpoint_path=checkpoint_path,
        trace_dir=trace_dir,
        approval_authority=approval_authority,
        planner=planner,
    )

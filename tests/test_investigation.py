"""Investigation loop: budgets, citation validation, verdicts, attestation gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from inverse_agent.attestations import AttestationScope, ScopedTrustStore
from inverse_agent.investigation import (
    AgentAnswer,
    AgentBudget,
    Decision,
    InvestigationLoop,
    InvestigationVerdict,
    ScriptedInvestigationPlanner,
    SourceCitation,
    StopReason,
    ToolCall,
    ToolObservation,
)


@pytest.fixture
def trusted_workspace(tmp_path: Path) -> tuple[Path, ScopedTrustStore]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "app.py").write_text("def f():\n    return 42\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")
    return workspace, trust


def _cite_first_line(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
    obs = catalog[0]
    return AgentAnswer(
        summary="found it",
        findings=("app.py defines f",),
        next_actions=("done",),
        citations=(
            SourceCitation(
                observation_id=obs.observation_id,
                path=obs.path,
                start_line=obs.start_line,
                end_line=obs.start_line,
            ),
        ),
    )


def test_loop_passes_with_valid_citation(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace
    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="app.py"),),
        build_answer=_cite_first_line,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="what does app.py do", workspace=workspace)
    assert report.verdict is InvestigationVerdict.PASS
    assert report.stop_reason is StopReason.FINISHED
    assert report.tool_calls_used == 1


def test_loop_blocks_without_attestation(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    planner = ScriptedInvestigationPlanner(steps=(), build_answer=_cite_first_line)
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.NOT_ATTESTED


def test_unsupported_citation_forces_incomplete(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def bad_answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="claim",
            findings=("made up",),
            next_actions=(),
            citations=(
                SourceCitation(
                    observation_id="obs_does_not_exist",
                    path="app.py",
                    start_line=1,
                    end_line=1,
                ),
            ),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="app.py"),),
        build_answer=bad_answer,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.UNSUPPORTED_CITATION


def test_citation_outside_returned_range_forces_incomplete(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def out_of_range(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        obs = catalog[0]
        return AgentAnswer(
            summary="claim",
            findings=("x",),
            next_actions=(),
            citations=(
                SourceCitation(
                    observation_id=obs.observation_id,
                    path=obs.path,
                    start_line=1,
                    end_line=9999,
                ),
            ),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="app.py"),),
        build_answer=out_of_range,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.UNSUPPORTED_CITATION


def test_budget_exhaustion_is_incomplete(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    class NeverAnswers:
        def decide(
            self, *, goal: str, catalog: tuple[ToolObservation, ...]
        ) -> Decision:
            del goal, catalog
            return ToolCall(tool="list_files", path=".")

    loop = InvestigationLoop(
        planner=NeverAnswers(),
        trust=trust,
        budget=AgentBudget(max_decisions=4, max_tool_calls=3, max_physical_requests=8),
    )
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason in {StopReason.BUDGET_EXHAUSTED, StopReason.NO_PROGRESS}


def test_repeated_no_progress_call_stops(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    class Repeats:
        def decide(
            self, *, goal: str, catalog: tuple[ToolObservation, ...]
        ) -> Decision:
            del goal, catalog
            return ToolCall(tool="read_file", path="app.py")

    loop = InvestigationLoop(planner=Repeats(), trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.stop_reason is StopReason.NO_PROGRESS
    assert report.verdict is InvestigationVerdict.INCOMPLETE


def test_planner_protocol_failure_is_failed(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    class Broken:
        def decide(
            self, *, goal: str, catalog: tuple[ToolObservation, ...]
        ) -> Decision:
            raise RuntimeError("model returned garbage")

    loop = InvestigationLoop(planner=Broken(), trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.FAILED
    assert report.stop_reason is StopReason.PROTOCOL_FAILURE


def test_strict_decode_refusal_is_incomplete(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "blob.bin").write_bytes(b"text\n\xff\xfe\x01more\n")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    # Force a strict-decode refusal by reading a file that is invalid UTF-8 but
    # not binary enough to be flagged (mostly text with a couple bad bytes).
    (workspace / "mixed.txt").write_bytes(b"line one\nline two \xff bad\nline three\n")
    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="mixed.txt"),),
        build_answer=_cite_first_line,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.STRICT_DECODE_REFUSAL


def test_source_read_revoked_mid_run_stops_reads(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "b.py").write_text("y = 2\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    class RevokeAfterFirst:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
            del goal, catalog
            self.calls += 1
            if self.calls == 1:
                return ToolCall(tool="read_file", path="a.py")
            trust.revoke(workspace, AttestationScope.SOURCE_READ)
            return ToolCall(tool="read_file", path="b.py")

    loop = InvestigationLoop(planner=RevokeAfterFirst(), trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.NOT_ATTESTED


def test_budget_validation_rejects_inconsistent() -> None:
    with pytest.raises(ValueError, match="tool-call budget"):
        AgentBudget(max_decisions=5, max_tool_calls=6).validate()


def test_physical_budget_counts_real_requests(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    # A planner that reports more real client requests than the physical budget
    # must stop with BUDGET_EXHAUSTED, and the report reflects the real count.
    workspace, trust = trusted_workspace

    class RequestHungryPlanner:
        max_total_requests = 999  # the loop overrides this to the budget
        requests_made = 0

        def decide(
            self, *, goal: str, catalog: tuple[ToolObservation, ...]
        ) -> Decision:
            # Simulate two client requests per decision (a retry). Vary the call
            # each turn so the no-progress guard does not fire before the budget.
            self.requests_made += 2
            if self.requests_made > self.max_total_requests:
                raise RuntimeError("model request budget exhausted")
            return ToolCall(tool="list_files", path=".", glob=f"*{self.requests_made}")

    planner = RequestHungryPlanner()
    loop = InvestigationLoop(
        planner=planner,
        trust=trust,
        budget=AgentBudget(max_decisions=12, max_tool_calls=10, max_physical_requests=6),
    )
    # The loop should have set the planner's cap to the physical budget.
    assert planner.max_total_requests == 6
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert report.physical_requests_used <= 8  # real count, not decision count


def test_budget_override_above_ceiling_rejected() -> None:
    with pytest.raises(ValueError, match="max_decisions"):
        AgentBudget(max_decisions=999, max_tool_calls=5).validate()


def test_read_file_past_eof_citation_is_rejected(tmp_path: Path) -> None:
    # A citation to a manufactured past-EOF line must not validate.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "small.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    def cite_phantom(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        obs = catalog[0]
        return AgentAnswer(
            summary="claim",
            findings=("x",),
            next_actions=(),
            citations=(
                SourceCitation(
                    observation_id=obs.observation_id,
                    path=obs.path,
                    start_line=9999,
                    end_line=9999,
                ),
            ),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="small.py", start_line=9999),),
        build_answer=cite_phantom,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.UNSUPPORTED_CITATION


def test_non_decision_type_is_protocol_failure(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    class BadType:
        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> object:
            del goal, catalog
            return {"tool": "read_file"}  # not a ToolCall/AgentAnswer

    loop = InvestigationLoop(planner=BadType(), trust=trust)  # type: ignore[arg-type]
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.FAILED
    assert report.stop_reason is StopReason.PROTOCOL_FAILURE


def test_sensitive_file_read_is_gate_fatal(tmp_path: Path) -> None:
    # A security-policy violation (reading a denied file) terminates the run.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".env").write_text("api_key=sk_live_0123456789abcdef\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path=".env"),),
        build_answer=_cite_first_line,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.POLICY_VIOLATION


def test_policy_violation_after_valid_read_still_fatal(tmp_path: Path) -> None:
    # A later traversal attempt must be gate-fatal even after a good read.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    planner = ScriptedInvestigationPlanner(
        steps=(
            ToolCall(tool="read_file", path="app.py"),
            ToolCall(tool="read_file", path="../escape.txt"),
        ),
        build_answer=_cite_first_line,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.stop_reason is StopReason.POLICY_VIOLATION


def test_citation_to_blank_line_is_rejected(tmp_path: Path) -> None:
    # A citation resolving only to a blank line is not real evidence.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "spaced.py").write_text("x = 1\n\n\ny = 2\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    def cite_blank(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        obs = catalog[0]
        return AgentAnswer(
            summary="claim",
            findings=("x",),
            next_actions=(),
            citations=(
                SourceCitation(
                    observation_id=obs.observation_id,
                    path=obs.path,
                    start_line=2,  # a blank line
                    end_line=2,
                ),
            ),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="spaced.py"),),
        build_answer=cite_blank,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.UNSUPPORTED_CITATION


def test_citation_to_empty_search_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    def cite_empty(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        empty = catalog[0]
        return AgentAnswer(
            summary="claim",
            findings=("x",),
            next_actions=(),
            citations=(
                SourceCitation(
                    observation_id=empty.observation_id,
                    path=empty.path,
                    start_line=1,
                    end_line=1,
                ),
            ),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="search_text", query="NO_SUCH_STRING_ANYWHERE"),),
        build_answer=cite_empty,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.UNSUPPORTED_CITATION

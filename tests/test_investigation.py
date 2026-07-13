"""Investigation loop: budgets, citation validation, verdicts, attestation gate."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import inverse_agent.investigation as investigation_module
from inverse_agent.attestations import AttestationScope, ScopedTrustStore
from inverse_agent.fs_tools import FsToolError, WorkspaceReader
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
    _has_unresolved_negative_uncertainty,
    _validate_citations,
    citation_intersects_redaction,
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


def test_negative_answer_cannot_pass_after_incomplete_search(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace
    (workspace / "large.txt").write_text("hidden issue\n" + "x" * (1024 * 1024))

    def negative_answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        read = next(observation for observation in catalog if observation.tool == "read_file")
        return AgentAnswer(
            summary="issue absent",
            findings=("app.py looks normal",),
            next_actions=("Review the omitted search scope.",),
            citations=(
                SourceCitation(
                    observation_id=read.observation_id,
                    path=read.path,
                    start_line=read.start_line,
                    end_line=read.start_line,
                ),
            ),
            complete=True,
            issue_present=False,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(
            ToolCall(tool="search_text", query="hidden issue"),
            ToolCall(tool="read_file", path="app.py"),
        ),
        build_answer=negative_answer,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-incomplete-negative",
        goal="is hidden issue present",
        workspace=workspace,
    )
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.INCOMPLETE_EVIDENCE


def test_unrelated_retryable_read_refusal_does_not_poison_grounded_negative(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def negative_answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        read = next(observation for observation in catalog if observation.content_hash)
        return AgentAnswer(
            summary="app.py does not return zero",
            findings=("app.py returns 42",),
            next_actions=("Keep the current return value.",),
            citations=(
                SourceCitation(
                    observation_id=read.observation_id,
                    path=read.path,
                    start_line=read.start_line,
                    end_line=read.start_line,
                ),
            ),
            issue_present=False,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(
            ToolCall(tool="read_file", path="missing.py"),
            ToolCall(tool="read_file", path="app.py"),
        ),
        build_answer=negative_answer,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-refused-negative",
        goal="does app.py return zero",
        workspace=workspace,
    )
    assert report.catalog[0].metadata["refused"] is True
    assert report.catalog[0].incomplete and report.catalog[0].truncated
    assert report.verdict is InvestigationVerdict.PASS
    assert report.stop_reason is StopReason.FINISHED


def test_bounded_read_window_can_support_localized_negative(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace
    (workspace / "long.py").write_text(
        "\n".join(f"value_{line} = {line}" for line in range(1, 401)) + "\n",
        encoding="utf-8",
    )

    def negative_answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        read = catalog[0]
        return AgentAnswer(
            summary="the cited assignment is not zero",
            findings=("value_250 is assigned 250",),
            next_actions=("Keep the assignment unchanged.",),
            citations=(
                SourceCitation(
                    observation_id=read.observation_id,
                    path=read.path,
                    start_line=250,
                    end_line=250,
                ),
            ),
            issue_present=False,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="long.py", start_line=250, max_lines=20),),
        build_answer=negative_answer,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-window-negative",
        goal="is value_250 assigned zero",
        workspace=workspace,
    )
    assert report.catalog[0].truncated
    assert not report.catalog[0].incomplete
    assert report.verdict is InvestigationVerdict.PASS


def test_positive_answer_can_pass_over_unrelated_incomplete_catalog(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace
    (workspace / "large.txt").write_text("x" * (1024 * 1024 + 1), encoding="utf-8")

    def positive_answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        read = next(observation for observation in catalog if observation.tool == "read_file")
        return AgentAnswer(
            summary="app.py returns 42",
            findings=("the return statement is present",),
            next_actions=("Keep the return statement.",),
            citations=(
                SourceCitation(
                    observation_id=read.observation_id,
                    path=read.path,
                    start_line=2,
                    end_line=2,
                ),
            ),
            issue_present=True,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(
            ToolCall(tool="search_text", query="never-present"),
            ToolCall(tool="read_file", path="app.py"),
        ),
        build_answer=positive_answer,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-positive-partial",
        goal="does app.py return 42",
        workspace=workspace,
    )
    assert report.catalog[0].incomplete and report.catalog[0].truncated
    assert report.verdict is InvestigationVerdict.PASS


def test_later_complete_pointer_result_supersedes_earlier_uncertainty(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace
    oversized = workspace / "large.txt"
    oversized.write_text("x" * (1024 * 1024 + 1), encoding="utf-8")

    class RecoversCoverage:
        turn = 0

        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
            del goal
            self.turn += 1
            if self.turn == 1:
                return ToolCall(tool="search_text", query="return 0")
            if self.turn == 2:
                oversized.unlink()
                return ToolCall(tool="search_text", query="return 0")
            if self.turn == 3:
                return ToolCall(tool="read_file", path="app.py")
            read = next(observation for observation in catalog if observation.tool == "read_file")
            return AgentAnswer(
                summary="app.py does not return zero",
                findings=("the function returns 42",),
                next_actions=("Keep the current return value.",),
                citations=(
                    SourceCitation(
                        observation_id=read.observation_id,
                        path=read.path,
                        start_line=2,
                        end_line=2,
                    ),
                ),
                issue_present=False,
            )

    report = InvestigationLoop(planner=RecoversCoverage(), trust=trust).run(
        run_id="r-recovered-pointer",
        goal="does app.py return zero",
        workspace=workspace,
    )
    searches = [observation for observation in report.catalog if observation.tool == "search_text"]
    assert searches[0].incomplete and searches[0].truncated
    assert not searches[1].incomplete and not searches[1].truncated
    assert report.verdict is InvestigationVerdict.PASS


def test_recursive_list_retry_uses_the_same_uncertainty_scope(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    class RecoversRecursiveList:
        turn = 0

        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
            del goal
            self.turn += 1
            if self.turn == 1:
                return ToolCall(tool="list_files", path="generated/.", glob="**/*.py")
            if self.turn == 2:
                (workspace / "generated").mkdir()
                return ToolCall(tool="list_files", path="generated/.", glob="**/*.py")
            if self.turn == 3:
                return ToolCall(tool="read_file", path="app.py")
            read = next(observation for observation in catalog if observation.tool == "read_file")
            return AgentAnswer(
                summary="app.py does not return zero",
                findings=("the function returns 42",),
                next_actions=("Keep the current return value.",),
                citations=(SourceCitation(read.observation_id, read.path, 2, 2),),
                issue_present=False,
            )

    report = InvestigationLoop(planner=RecoversRecursiveList(), trust=trust).run(
        run_id="r-recovered-recursive-list",
        goal="does app.py return zero",
        workspace=workspace,
    )
    listings = [observation for observation in report.catalog if observation.tool == "list_files"]
    assert listings[0].metadata["recursive"] is True
    assert listings[0].incomplete and listings[0].truncated
    assert listings[1].metadata["recursive"] is True
    assert not listings[1].incomplete and not listings[1].truncated
    assert report.verdict is InvestigationVerdict.PASS


@pytest.mark.parametrize(
    ("invalid_call", "corrected_call"),
    [
        (
            ToolCall(tool="search_text", query="x" * 300),
            ToolCall(tool="search_text", query="return 0"),
        ),
        (
            ToolCall(tool="list_files", path=".", glob="x" * 200),
            ToolCall(tool="list_files", path="."),
        ),
        (
            ToolCall(tool="list_files", path="x" * 513),
            ToolCall(tool="list_files", path="."),
        ),
    ],
)
def test_request_invalid_pointer_does_not_poison_corrected_negative_answer(
    trusted_workspace: tuple[Path, ScopedTrustStore],
    invalid_call: ToolCall,
    corrected_call: ToolCall,
) -> None:
    workspace, trust = trusted_workspace

    class CorrectsRequest:
        turn = 0

        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
            del goal
            self.turn += 1
            if self.turn == 1:
                return invalid_call
            if self.turn == 2:
                return corrected_call
            if self.turn == 3:
                return ToolCall(tool="read_file", path="app.py")
            read = next(observation for observation in catalog if observation.tool == "read_file")
            return AgentAnswer(
                summary="app.py does not return zero",
                findings=("the function returns 42",),
                next_actions=("Keep the current return value.",),
                citations=(SourceCitation(read.observation_id, read.path, 2, 2),),
                issue_present=False,
            )

    report = InvestigationLoop(planner=CorrectsRequest(), trust=trust).run(
        run_id="r-corrected-request",
        goal="does app.py return zero",
        workspace=workspace,
    )
    assert report.catalog[0].metadata["refused"] is True
    assert report.catalog[0].metadata["request_invalid"] is True
    assert report.verdict is InvestigationVerdict.PASS


def test_request_invalid_read_does_not_supersede_valid_cited_read(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    class InvalidReadAfterEvidence:
        turn = 0

        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
            del goal
            self.turn += 1
            if self.turn == 1:
                return ToolCall(tool="read_file", path="app.py", max_lines=2)
            if self.turn == 2:
                return ToolCall(tool="read_file", path="app.py", max_lines=999)
            valid = catalog[0]
            return AgentAnswer(
                summary="app.py does not return zero",
                findings=("the function returns 42",),
                next_actions=("Keep the current return value.",),
                citations=(SourceCitation(valid.observation_id, valid.path, 2, 2),),
                issue_present=False,
            )

    report = InvestigationLoop(planner=InvalidReadAfterEvidence(), trust=trust).run(
        run_id="r-invalid-read-after-evidence",
        goal="does app.py return zero",
        workspace=workspace,
    )
    assert report.catalog[1].metadata["request_invalid"] is True
    assert report.verdict is InvestigationVerdict.PASS


def test_literal_none_glob_cannot_supersede_unfiltered_scope(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace
    sensitive = workspace / ".env"
    sensitive.write_text("TOKEN=secret\n", encoding="utf-8")

    class DifferentScopes:
        turn = 0

        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
            del goal
            self.turn += 1
            if self.turn == 1:
                return ToolCall(tool="list_files", path=".")
            if self.turn == 2:
                sensitive.unlink()
                return ToolCall(tool="list_files", path=".", glob="None")
            if self.turn == 3:
                return ToolCall(tool="read_file", path="app.py")
            read = next(observation for observation in catalog if observation.tool == "read_file")
            return AgentAnswer(
                summary="app.py does not return zero",
                findings=("the function returns 42",),
                next_actions=("Keep the current return value.",),
                citations=(SourceCitation(read.observation_id, read.path, 2, 2),),
                issue_present=False,
            )

    report = InvestigationLoop(planner=DifferentScopes(), trust=trust).run(
        run_id="r-distinct-none-glob",
        goal="does app.py return zero",
        workspace=workspace,
    )
    listings = [observation for observation in report.catalog if observation.tool == "list_files"]
    assert listings[0].metadata["glob"] is None and listings[0].incomplete
    assert listings[1].metadata["glob"] == "None" and not listings[1].incomplete
    assert report.stop_reason is StopReason.INCOMPLETE_EVIDENCE


def test_answer_complete_false_forces_incomplete(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def incomplete_answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        answer = _cite_first_line(catalog)
        return AgentAnswer(
            summary=answer.summary,
            findings=answer.findings,
            next_actions=answer.next_actions,
            citations=answer.citations,
            complete=False,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="app.py"),),
        build_answer=incomplete_answer,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-self-incomplete",
        goal="inspect app",
        workspace=workspace,
    )
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.INCOMPLETE_EVIDENCE


def test_loop_blocks_without_attestation(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    planner = ScriptedInvestigationPlanner(steps=(), build_answer=_cite_first_line)
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.NOT_ATTESTED


def test_citation_to_search_pointer_is_rejected(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    # A citation to a search_text/list_files pointer observation is not grounded
    # evidence: only a read_file observation is citable.
    workspace, trust = trusted_workspace

    def cite_search(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        pointer = next(o for o in catalog if o.tool == "search_text")
        return AgentAnswer(
            summary="claim",
            findings=("f",),
            next_actions=("Read the source file before concluding.",),
            citations=(
                SourceCitation(
                    observation_id=pointer.observation_id,
                    path=pointer.path,
                    start_line=pointer.start_line,
                    end_line=pointer.start_line,
                ),
            ),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="search_text", query="return"),),
        build_answer=cite_search,
    )
    loop = InvestigationLoop(planner=planner, trust=trust)
    report = loop.run(run_id="r1", goal="x", workspace=workspace)
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.UNSUPPORTED_CITATION


def test_unsupported_citation_forces_incomplete(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def bad_answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="claim",
            findings=("made up",),
            next_actions=("Verify the claim against real evidence.",),
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


def test_unsupported_citation_precedes_answer_incompleteness(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def incomplete_fabrication(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        del catalog
        return AgentAnswer(
            summary="uncertain claim",
            findings=("fabricated",),
            next_actions=("Gather valid evidence before concluding.",),
            citations=(
                SourceCitation(
                    observation_id="obs_fabricated",
                    path="app.py",
                    start_line=1,
                    end_line=1,
                ),
            ),
            complete=False,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="app.py"),),
        build_answer=incomplete_fabrication,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-invalid-incomplete",
        goal="inspect app",
        workspace=workspace,
    )
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.UNSUPPORTED_CITATION


def test_each_finding_requires_a_corresponding_citation(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def floating_finding(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        grounded = _cite_first_line(catalog)
        return AgentAnswer(
            summary="two claims",
            findings=("grounded", "floating"),
            next_actions=("Investigate both claims.",),
            citations=grounded.citations,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="app.py"),),
        build_answer=floating_finding,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-floating-finding",
        goal="inspect app",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.MALFORMED_ANSWER
    assert "each finding" in report.error


def test_findings_cannot_reuse_one_citation_range(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def duplicate_citation(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        grounded = _cite_first_line(catalog)
        citation = grounded.citations[0]
        return AgentAnswer(
            summary="two claims",
            findings=("function exists", "function returns 42"),
            next_actions=("Keep the implementation covered by tests.",),
            citations=(citation, citation),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="app.py"),),
        build_answer=duplicate_citation,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-duplicate-citation",
        goal="inspect app",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.MALFORMED_ANSWER
    assert "distinct citation" in report.error


def test_overlapping_read_ids_cannot_reuse_one_physical_citation_range(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def duplicate_range(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        first, second = catalog
        assert first.observation_id != second.observation_id
        return AgentAnswer(
            summary="two claims",
            findings=("function exists", "function returns 42"),
            next_actions=("Keep the implementation covered by tests.",),
            citations=(
                SourceCitation(first.observation_id, first.path, 2, 2),
                SourceCitation(second.observation_id, second.path, 2, 2),
            ),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(
            ToolCall(tool="read_file", path="app.py", start_line=1, max_lines=2),
            ToolCall(tool="read_file", path="app.py", start_line=2, max_lines=1),
        ),
        build_answer=duplicate_range,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-overlapping-citation",
        goal="inspect app",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.MALFORMED_ANSWER
    assert "distinct citation" in report.error


@pytest.mark.skipif(os.name != "nt", reason="Windows case-alias regression")
def test_windows_case_alias_cannot_reuse_one_physical_citation_range(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "Report.txt").write_text("same physical evidence\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    def duplicate_range(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        first, second = catalog
        assert first.path != second.path
        assert "file_identity" not in first.metadata
        return AgentAnswer(
            summary="two claims",
            findings=("the report exists", "the report has evidence"),
            next_actions=("Keep the report under review.",),
            citations=(
                SourceCitation(first.observation_id, first.path, 1, 1),
                SourceCitation(second.observation_id, second.path, 1, 1),
            ),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(
            ToolCall(tool="read_file", path="Report.txt"),
            ToolCall(tool="read_file", path="REPORT.TXT"),
        ),
        build_answer=duplicate_range,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-windows-case-alias",
        goal="inspect report",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.UNSUPPORTED_CITATION
    assert "distinct physical citation" in report.error


@pytest.mark.skipif(os.name != "nt", reason="Windows case-alias regression")
def test_windows_case_alias_refusal_supersedes_cited_read(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "Report.txt"
    source.write_text("no issue here\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    class AliasThenRemovePlanner:
        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
            del goal
            if not catalog:
                return ToolCall(tool="read_file", path="Report.txt")
            if len(catalog) == 1:
                source.unlink()
                return ToolCall(tool="read_file", path="REPORT.TXT")
            first = catalog[0]
            return AgentAnswer(
                summary="the issue is absent",
                findings=("the report says no issue is present",),
                next_actions=("Restore the source and repeat the investigation.",),
                citations=(SourceCitation(first.observation_id, first.path, 1, 1),),
                issue_present=False,
            )

    report = InvestigationLoop(planner=AliasThenRemovePlanner(), trust=trust).run(
        run_id="r-windows-case-alias-refusal",
        goal="check whether the issue is absent",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.INCOMPLETE_EVIDENCE
    assert "unresolved evidence omissions" in report.error


@pytest.mark.skipif(os.name != "nt", reason="Windows case-alias regression")
def test_successful_case_alias_retry_supersedes_earlier_refusal(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "Report.txt"
    source.write_text("no issue here\n", encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    class RecoveringAliasPlanner:
        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
            del goal
            if not catalog:
                return ToolCall(tool="read_file", path="Report.txt")
            if len(catalog) == 1:
                source.unlink()
                return ToolCall(tool="read_file", path="REPORT.TXT")
            if len(catalog) == 2:
                source.write_text("no issue here\n", encoding="utf-8")
                return ToolCall(tool="read_file", path="REPORT.TXT")
            first = catalog[0]
            return AgentAnswer(
                summary="the issue is absent",
                findings=("the report says no issue is present",),
                next_actions=("Keep the report under observation.",),
                citations=(SourceCitation(first.observation_id, first.path, 1, 1),),
                issue_present=False,
            )

    report = InvestigationLoop(planner=RecoveringAliasPlanner(), trust=trust).run(
        run_id="r-windows-case-alias-recovery",
        goal="check whether the issue is absent",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.FINISHED
    assert report.verdict is InvestigationVerdict.PASS


def test_unknown_case_variant_refusal_can_be_cleared_by_exact_cited_retry() -> None:
    cited = ToolObservation(
        observation_id="obs-cited",
        tool="read_file",
        path="123/A.py",
        content_hash="hash-a",
        text="safe",
        lines=("1: safe",),
    )
    refusal = ToolObservation(
        observation_id="obs-refused",
        tool="read_file",
        path="123/a.py",
        content_hash="",
        text="[refused] unavailable",
        truncated=True,
        incomplete=True,
        metadata={"refused": True},
    )
    cited_retry = ToolObservation(
        observation_id="obs-cited-retry",
        tool="read_file",
        path="123/A.py",
        content_hash="hash-a",
        text="safe",
        lines=("1: safe",),
    )
    variant_retry = ToolObservation(
        observation_id="obs-variant-retry",
        tool="read_file",
        path="123/a.py",
        content_hash="hash-b",
        text="safe too",
        lines=("1: safe too",),
    )
    answer = AgentAnswer(
        summary="localized absence",
        findings=("the cited file is safe",),
        next_actions=("Keep both paths readable.",),
        citations=(SourceCitation("obs-cited", "123/A.py", 1, 1),),
        issue_present=False,
    )
    identities = {
        "obs-cited": "file-a",
        "obs-cited-retry": "file-a",
        "obs-variant-retry": "file-b",
    }
    identity_for = identities.get

    assert _has_unresolved_negative_uncertainty(
        answer,
        (cited, refusal),
        identity_for=identity_for,
    )
    assert not _has_unresolved_negative_uncertainty(
        answer,
        (cited, refusal, cited_retry),
        identity_for=identity_for,
    )
    assert not _has_unresolved_negative_uncertainty(
        answer,
        (cited, refusal, variant_retry),
        identity_for=identity_for,
    )


def test_loop_exact_cited_retry_clears_wrong_case_not_found(
    trusted_workspace: tuple[Path, ScopedTrustStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, trust = trusted_workspace
    real_dispatch = investigation_module._dispatch

    def case_sensitive_dispatch(reader: WorkspaceReader, call: ToolCall) -> ToolObservation:
        if call.tool == "read_file" and call.path == "APP.PY":
            raise FsToolError("path component could not be opened")
        return real_dispatch(reader, call)

    monkeypatch.setattr(investigation_module, "_dispatch", case_sensitive_dispatch)

    def negative_answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        first = catalog[0]
        return AgentAnswer(
            summary="the localized issue is absent",
            findings=("the cited function remains safe",),
            next_actions=("Keep the exact cited path readable.",),
            citations=(SourceCitation(first.observation_id, first.path, 1, 1),),
            issue_present=False,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(
            ToolCall(tool="read_file", path="app.py"),
            ToolCall(tool="read_file", path="APP.PY"),
            ToolCall(tool="read_file", path="app.py"),
        ),
        build_answer=negative_answer,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-case-variant-recovered",
        goal="check the localized issue",
        workspace=workspace,
    )
    assert report.verdict is InvestigationVerdict.PASS
    assert report.stop_reason is StopReason.FINISHED


def test_citation_validation_fails_closed_when_runtime_identity_is_lost() -> None:
    observation = ToolObservation(
        observation_id="obs-lost-identity",
        tool="read_file",
        path="README.md",
        content_hash="sanitized-hash",
        text="evidence",
        lines=("1: evidence",),
    )
    answer = AgentAnswer(
        summary="grounded claim",
        findings=("the evidence exists",),
        next_actions=("Rehydrate the private evidence identity before reconnect.",),
        citations=(SourceCitation(observation.observation_id, observation.path, 1, 1),),
    )

    error = _validate_citations(
        answer,
        (observation,),
        identity_for=lambda observation_id: None,
    )
    assert error == "citation evidence identity is unavailable"


def test_empty_substantive_answer_is_rejected(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def empty_summary(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        grounded = _cite_first_line(catalog)
        return AgentAnswer(
            summary="",
            findings=grounded.findings,
            next_actions=("Read the cited range.",),
            citations=grounded.citations,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="app.py"),),
        build_answer=empty_summary,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-empty-answer",
        goal="inspect app",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.MALFORMED_ANSWER
    assert "summary is empty" in report.error


def test_citation_intersecting_redacted_line_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "config.py").write_text(
        "safe = 1\napi_key=sk_live_0123456789abcdef\nafter = 2\n",
        encoding="utf-8",
    )
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    def cite_secret(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        read = catalog[0]
        return AgentAnswer(
            summary="secret claim",
            findings=("the source contains a credential",),
            next_actions=("Rotate the credential outside this tool.",),
            citations=(SourceCitation(read.observation_id, read.path, 2, 2),),
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="config.py"),),
        build_answer=cite_secret,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-redacted-citation",
        goal="inspect config",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.UNSUPPORTED_CITATION
    assert "redacted" in report.error


def test_negative_answer_is_blocked_by_redaction_outside_read_window(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = "safe = 1\n" + "\n".join(f"filler_{i} = {i}" for i in range(2, 220))
    source += "\napi_key=sk_live_0123456789abcdef\n"
    (workspace / "config.py").write_text(source, encoding="utf-8")
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")

    def negative_answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        read = catalog[0]
        assert read.incomplete and read.redacted
        assert read.metadata["redacted_lines"] == ()
        return AgentAnswer(
            summary="the visible setting is safe",
            findings=("line one sets safe to one",),
            next_actions=("Inspect the omitted redacted content separately.",),
            citations=(SourceCitation(read.observation_id, read.path, 1, 1),),
            issue_present=False,
        )

    planner = ScriptedInvestigationPlanner(
        steps=(ToolCall(tool="read_file", path="config.py", max_lines=1),),
        build_answer=negative_answer,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-redaction-outside-window",
        goal="is config.py free of credentials",
        workspace=workspace,
    )
    assert report.verdict is InvestigationVerdict.INCOMPLETE
    assert report.stop_reason is StopReason.INCOMPLETE_EVIDENCE


def test_huge_citation_range_redaction_check_is_bounded() -> None:
    observation = ToolObservation(
        observation_id="obs_mask",
        tool="read_file",
        path="config.py",
        content_hash="h",
        text="[REDACTED_SECRET]",
        lines=("1: [REDACTED_SECRET]",),
        metadata={"redacted_lines": (1,)},
    )
    citation = SourceCitation(
        observation_id=observation.observation_id,
        path=observation.path,
        start_line=1,
        end_line=10**12,
    )
    assert citation_intersects_redaction(observation, citation)


def test_citation_outside_returned_range_forces_incomplete(
    trusted_workspace: tuple[Path, ScopedTrustStore],
) -> None:
    workspace, trust = trusted_workspace

    def out_of_range(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        obs = catalog[0]
        return AgentAnswer(
            summary="claim",
            findings=("x",),
            next_actions=("Read an existing source line.",),
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
        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
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
        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
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
        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
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

        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
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
            next_actions=("Cite a substantive source line.",),
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


@pytest.mark.parametrize(
    "call",
    [
        ToolCall(tool="read_file", path="../escape.py", start_line=0),
        ToolCall(tool="list_files", path="../escape", glob="x" * 200),
        ToolCall(tool="read_file", path="x" * 508 + "/.env", start_line=0),
    ],
)
def test_path_policy_precedes_other_invalid_arguments(
    trusted_workspace: tuple[Path, ScopedTrustStore], call: ToolCall
) -> None:
    workspace, trust = trusted_workspace
    planner = ScriptedInvestigationPlanner(steps=(call,), build_answer=_cite_first_line)
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="r-mixed-invalid-policy",
        goal="inspect workspace",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.POLICY_VIOLATION
    assert report.catalog == ()


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
            next_actions=("Read a source file before citing it.",),
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
            next_actions=("Read a source file before citing it.",),
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

"""Semantic investigation benchmark, negative controls, and fatal integrity gate."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path

import pytest

import inverse_agent.adapters.generic as generic_adapter_module
import inverse_agent.investigation_benchmark as benchmark_module
import inverse_agent.policies as policies_module
from inverse_agent.attestations import AttestationScope, ScopedTrustStore
from inverse_agent.fs_tools import ToolObservation
from inverse_agent.investigation import (
    AgentAnswer,
    AgentBudget,
    InvestigationLoop,
    InvestigationReport,
    InvestigationVerdict,
    ModelCallRecord,
    ScriptedInvestigationPlanner,
    SourceCitation,
    StopReason,
    ToolCall,
)
from inverse_agent.investigation_benchmark import (
    BenchmarkDefinitionError,
    ModelEndpointAudit,
    _aggregate,
    _find_citation,
    _finish_model_endpoint_audit,
    _integrity_failures,
    _model_planner_is_trusted,
    _parent_probe_proves_missing_revision,
    _score_variant_model,
    _start_model_endpoint_audit,
    _suite_definition_is_valid,
    default_cases,
    materialize,
    run_benchmark,
    run_benchmark_with_planner,
    run_case_with_planner,
)
from inverse_agent.investigation_model import ModelInvestigationPlanner
from inverse_agent.planner import OpenAICompatibleClient


def test_control_plane_git_wait_uses_the_active_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]

    class Response:
        status_code = 200

        def json(self) -> dict[str, str]:
            status = "waiting_for_approval" if clock[0] >= 106.0 else "starting"
            return {"status": status}

    class Client:
        def get(self, _path: str, *, headers: dict[str, str]) -> Response:
            assert headers == {}
            return Response()

    monkeypatch.setattr(benchmark_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        benchmark_module.time,
        "sleep",
        lambda _seconds: clock.__setitem__(0, clock[0] + 1.0),
    )
    executor = object.__new__(benchmark_module._ControlPlaneGitExecutor)
    executor._client = Client()
    executor._operator_headers = {}

    payload = executor._wait_for_status(
        "run-1",
        expected=frozenset({"waiting_for_approval"}),
        active_deadline=110.0,
        stage="approval wait",
    )

    assert payload == {"status": "waiting_for_approval"}
    assert clock[0] == 106.0


def test_seven_cases_cover_priority_stacks_and_real_git() -> None:
    cases = default_cases()
    assert len(cases) == 7
    names = {case.name for case in cases}
    assert names == {
        "android_exported_activity",
        "ios_main_thread_ui",
        "cpp_dangling_view",
        "django_react_injection",
        "pytorch_eval_mode",
        "generic_architecture",
        "git_approval_replanning",
    }
    priority = names - {"generic_architecture", "git_approval_replanning"}
    for case in cases:
        assert len(case.goal_variants) == 3
        assert case.claims
        assert case.required_observations
        if case.name in priority:
            assert any(claim.negative_control for claim in case.claims)

    full_stack = next(case for case in cases if case.name == "django_react_injection")
    react_source = full_stack.files["projects/static/projects/SearchResults.jsx"]
    assert "dangerouslySetInnerHTML" in react_source
    assert "return <div>{term}</div>" in react_source
    assert any(item.path == "package.json" for item in full_stack.required_observations)

    git_case = next(case for case in cases if case.name == "git_approval_replanning")
    assert git_case.expected_issue is True
    assert all("root commit" not in goal.casefold() for goal in git_case.goal_variants)
    assert git_case.command_recovery_dependencies == (
        ("generic.head_commit", "generic.parent_commit"),
    )


def test_deterministic_gate_passes_semantics_integrity_and_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_benchmark(default_cases(), root=tmp_path)
    assert result.total_cases == 7
    assert result.total_variants == 21
    assert result.cases_passed == 7, [item for item in result.variants if not item.passed]
    assert result.variants_passed == 21, [item for item in result.variants if not item.passed]
    assert result.integrity_failures == ()
    assert result.gate_passed

    git_variants = [item for item in result.variants if item.case == "git_approval_replanning"]
    assert len(git_variants) == 3
    for variant in git_variants:
        assert [item.command for item in variant.command_audit] == [
            "generic.parent_commit",
            "generic.head_commit",
        ]
        assert [item.status for item in variant.command_audit] == ["failed", "succeeded"]
        assert all(item.approved_via_control_plane for item in variant.command_audit)
        assert len({item.control_run_id for item in variant.command_audit}) == 2
        assert len({item.action_digest for item in variant.command_audit}) == 2
        assert len({item.challenge_id for item in variant.command_audit}) == 2
        assert [item.rule for item in variant.command_audit] == [
            "git-parent-commit",
            "git-head-commit",
        ]
        assert all(
            item.argv and Path(item.argv[0]).name.casefold().startswith("git")
            for item in variant.command_audit
        )
        assert all(item.domain == "generic" for item in variant.command_audit)
        assert all(item.workspace for item in variant.command_audit)
        assert variant.command_calls_used == 2
        assert variant.command_audit[1].based_on_observation_id == (
            variant.command_audit[0].observation_id
        )

    hollow_cases = tuple(
        replace(
            case,
            goal_variants=("", " ", "  "),
            claims=tuple(claim for claim in case.claims if not claim.negative_control),
        )
        for case in default_cases()
    )
    hollow_result = _aggregate(hollow_cases, list(result.variants), result.budget)
    assert not hollow_result.definition_contract_valid
    assert not hollow_result.suite_contract_valid
    assert not hollow_result.gate_passed

    class EqualDict(dict[str, str]):
        def __eq__(self, other: object) -> bool:
            del other
            return True

    canonical = default_cases()
    equality_spoof = replace(
        canonical[0],
        files=EqualDict({**canonical[0].files, "ANSWER.txt": "planted answer"}),
    )
    assert not _suite_definition_is_valid((equality_spoof, *canonical[1:]))
    monkeypatch.setattr(
        benchmark_module,
        "default_cases",
        lambda: (equality_spoof, *canonical[1:]),
    )
    assert not _suite_definition_is_valid((equality_spoof, *canonical[1:]))


def test_git_gate_rejects_policy_and_adapter_argv_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compromised = (*policies_module.GIT_SAFE_PREFIX, "hash-object", "README.md")
    monkeypatch.setattr(policies_module, "GIT_HEAD_COMMIT_ARGV", compromised)
    monkeypatch.setattr(generic_adapter_module, "GIT_HEAD_COMMIT_ARGV", compromised)

    with pytest.raises(BenchmarkDefinitionError, match="frozen argv"):
        run_benchmark(default_cases(), root=tmp_path)


def test_noncanonical_budget_cannot_certify_public_gate(tmp_path: Path) -> None:
    def factory(case, _goal):  # type: ignore[no-untyped-def]
        planner_type = (
            benchmark_module._BenchmarkReplanningPlanner
            if case.command_tools
            else ScriptedInvestigationPlanner
        )
        return planner_type(
            steps=case.steps,
            build_answer=benchmark_module._make_answer_builder(case),
        )

    expanded = AgentBudget(
        max_decisions=24,
        max_tool_calls=20,
        max_physical_requests=36,
    )
    result = run_benchmark_with_planner(
        default_cases(),
        factory,
        root=tmp_path,
        budget=expanded,
    )

    assert result.variants_passed == 21
    assert not result.gate_passed
    assert "benchmark: noncanonical_budget" in result.integrity_failures


def _run_source_answer(case, tmp_path: Path, builder):  # type: ignore[no-untyped-def]
    workspace = materialize(case, tmp_path)
    trust = ScopedTrustStore(tmp_path / "trust.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="test")
    planner = ScriptedInvestigationPlanner(steps=case.steps, build_answer=builder)
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id="semantic-test",
        goal=case.goal_variants[0],
        workspace=workspace,
    )
    return report, workspace


def _case_citations(case, catalog):  # type: ignore[no-untyped-def]
    citations = tuple(_find_citation(catalog, claim.anchor) for claim in case.claims)
    assert all(item is not None for item in citations)
    return tuple(item for item in citations if item is not None)


def test_semantic_scorer_accepts_synonyms_not_exact_answer_text(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "pytorch_eval_mode")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The two helpers use different inference state.",
            findings=(
                "The bad evaluation helper leaves the module in training mode during inference.",
                "The safe evaluation helper switches to evaluation mode and disables gradients.",
            ),
            next_actions=("Replace the bad helper with the safe state handling.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert passed, reason


def test_semantic_scorer_rejects_anchor_text_without_meaning(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "pytorch_eval_mode")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="Copied source tokens.",
            findings=("model.train()", "model.eval()"),
            next_actions=("Review manually.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


def test_semantic_scorer_rejects_contrary_global_conclusion(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="Both controls were inspected.",
            findings=tuple(claim.answer_text for claim in case.claims),
            next_actions=("Keep the app private.",),
            citations=_case_citations(case, catalog),
            issue_present=False,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "issue_present" in reason


@pytest.mark.parametrize(
    "contrary",
    (
        "InternalSettingsActivity is not in any practical sense private.",
        "InternalSettingsActivity isn't private.",
        "InternalSettingsActivity shouldn't be private.",
        "InternalSettingsActivity private is false.",
        "InternalSettingsActivity private was false.",
        "The statement that InternalSettingsActivity was private was false.",
        "InternalSettingsActivity is private, but it is not private.",
        "InternalSettingsActivity exists. DeepLinkActivity is private.",
    ),
)
def test_semantic_scorer_rejects_negated_negative_control(tmp_path: Path, contrary: str) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="Both activities were inspected.",
            findings=(
                case.claims[0].answer_text,
                contrary,
            ),
            next_actions=("Keep the internal activity private.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


def test_semantic_scorer_rejects_subject_shift_and_discourse_correction(
    tmp_path: Path,
) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The findings switch subjects and then contradict the control.",
            findings=(
                (
                    "DeepLinkActivity is listed. The internal settings component is "
                    "exported and externally reachable."
                ),
                ("InternalSettingsActivity is private and not exported. No, it is exported."),
            ),
            next_actions=("Require explicit component binding.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


@pytest.mark.parametrize(
    ("unsafe_finding", "safe_finding"),
    (
        (
            (
                "DeepLinkActivity is exported and externally reachable. "
                "InternalSettingsActivity is exported."
            ),
            None,
        ),
        (
            None,
            ("InternalSettingsActivity is private and not exported. Actually, it is exported."),
        ),
        (
            (
                'DeepLinkActivity is listed." The internal settings component is '
                "exported and externally reachable."
            ),
            None,
        ),
        (
            None,
            "InternalSettingsActivity is private and not exported\nIt is exported.",
        ),
        (
            None,
            ("InternalSettingsActivity is private and not exported. Correction: it is exported."),
        ),
        (
            None,
            (
                "InternalSettingsActivity is private and not exported. "
                "It is exported because InternalSettingsActivity exists."
            ),
        ),
        (
            (
                "DeepLinkActivity is listed.\u201d The admin screen is exported and "
                "externally reachable."
            ),
            None,
        ),
    ),
)
def test_semantic_scorer_rejects_cross_sentence_subject_loopholes(
    tmp_path: Path,
    unsafe_finding: str | None,
    safe_finding: str | None,
) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="A later sentence changes or obscures the claim subject.",
            findings=(
                unsafe_finding or case.claims[0].answer_text,
                safe_finding or case.claims[1].answer_text,
            ),
            next_actions=("Use one explicit component per finding.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


def test_semantic_subject_fragment_exemption_requires_full_containment() -> None:
    case = next(item for item in default_cases() if item.name == "git_approval_replanning")
    claim = next(item for item in case.claims if item.claim_id == "git-head")

    for finding in (
        "The current commit ID is not valid, but HEAD was resolved.",
        "The current commit ID was not valid, but HEAD was resolved.",
        "The current commit ID wasn't valid, but HEAD was resolved.",
        "The current commit ID didn't resolve, but HEAD was identified.",
        "The current commit ID hadn't resolved, but HEAD was identified.",
        "The claim that HEAD was resolved must not be believed.",
        "The claim that HEAD was resolved mustn't be believed.",
        "The current HEAD commit ID is unavailable.",
        "The current HEAD commit hash is unknown.",
        "The current HEAD commit ID is unresolved.",
        "The current HEAD commit hash is missing.",
        "The current HEAD commit ID was not found.",
    ):
        assert not benchmark_module._semantic_match(claim, finding, case.claims)
    assert benchmark_module._semantic_match(
        claim,
        "The current commit ID was resolved after the failed parent probe.",
        case.claims,
    )
    assert benchmark_module._semantic_match(
        claim,
        (
            "The current commit ID was not only resolved but also identified after "
            "the failed parent probe."
        ),
        case.claims,
    )
    assert benchmark_module._semantic_match(
        claim,
        (
            "The current commit ID, not only resolved but also identified after the "
            "failed parent probe, was recorded."
        ),
        case.claims,
    )
    assert benchmark_module._semantic_match(claim, claim.answer_text, case.claims)


def test_git_identity_finding_requires_the_exact_cited_commit() -> None:
    commit = "1724a08da0b04ff81d5aa5cb661a21c3e1897754"
    observation = ToolObservation(
        observation_id="obs_head",
        tool="run_command",
        path="command/generic.head_commit",
        content_hash="hash-head",
        text=f"HEAD commit: {commit}",
        lines=(f"1: HEAD commit: {commit}",),
        metadata={"command_name": "generic.head_commit", "status": "succeeded"},
    )
    citation = SourceCitation(
        observation_id=observation.observation_id,
        path=observation.path,
        start_line=1,
        end_line=1,
    )
    assert benchmark_module._git_identity_finding_matches(
        f"The current HEAD commit ID is {commit}.",
        citation,
        (observation,),
    )
    assert benchmark_module._git_identity_finding_matches(
        f"HEAD commit hash: {commit}",
        citation,
        (observation,),
    )
    assert benchmark_module._git_identity_finding_matches(
        f"Current HEAD commit is {commit}.",
        citation,
        (observation,),
    )
    for finding in (
        "The current HEAD commit ID is unavailable.",
        "The current HEAD commit hash is unknown.",
        "The current HEAD commit ID failed to resolve.",
        "The current HEAD commit ID resolution failed.",
        f"The current HEAD commit ID is {'0' * 40}.",
        f"The current HEAD commit ID is {commit}, but it is unavailable.",
    ):
        assert not benchmark_module._git_identity_finding_matches(
            finding,
            citation,
            (observation,),
        )


def test_git_identity_rejects_mixed_unresolved_language() -> None:
    for finding in (
        "The current HEAD commit ID was resolved as unavailable.",
        "The current HEAD commit hash was identified as unknown.",
        "The current HEAD commit ID was resolved, but failed to resolve.",
        "The current HEAD commit ID was resolved but is unresolved.",
        "The current HEAD commit hash was identified but is missing.",
        "The current HEAD commit ID resolution failed.",
        "The current HEAD commit ID could not be resolved.",
        "The current HEAD commit hash was not identified.",
        "The current HEAD commit ID was resolved, but is not available.",
        "The current HEAD commit hash was identified, but is not known.",
        "The current HEAD commit ID was resolved, but identification failed.",
        "The current HEAD commit ID was resolved, but failed to identify it.",
    ):
        assert benchmark_module._git_identity_has_unresolved_language(finding)
    assert not benchmark_module._git_identity_has_unresolved_language(
        "The current HEAD commit was resolved after the failed parent probe."
    )


def test_git_head_claim_rejects_wrong_hashes_and_mixed_contradictions() -> None:
    case = next(item for item in default_cases() if item.name == "git_approval_replanning")
    claim = next(item for item in case.claims if item.claim_id == "git-head")
    commit = "1724a08da0b04ff81d5aa5cb661a21c3e1897754"
    observation = ToolObservation(
        observation_id="obs_head",
        tool="run_command",
        path="command/generic.head_commit",
        content_hash="hash-head",
        text=f"HEAD commit: {commit}",
        lines=(f"1: HEAD commit: {commit}",),
        metadata={"command_name": "generic.head_commit", "status": "succeeded"},
    )
    citation = SourceCitation(
        observation_id=observation.observation_id,
        path=observation.path,
        start_line=1,
        end_line=1,
    )
    report = InvestigationReport(
        run_id="git-claim",
        verdict=InvestigationVerdict.PASS,
        stop_reason=StopReason.FINISHED,
        answer=None,
        catalog=(observation,),
        decisions_used=1,
        tool_calls_used=1,
        physical_requests_used=1,
    )

    def matches(finding: str) -> bool:
        return benchmark_module._claim_pair_matches(
            claim,
            finding,
            citation,
            report,
            Path("."),
            case.claims,
        )

    assert matches("The current HEAD commit was resolved after the failed parent probe.")
    assert matches(f"The current HEAD commit was resolved as {commit}.")
    assert matches(f"Current HEAD commit is {commit}.")
    assert not matches(f"The current HEAD commit ID was resolved as {'0' * 40}.")
    assert not matches(f"The current HEAD commit ID was resolved as hash_{'0' * 40}.")
    for finding in (
        "The current HEAD commit ID was resolved as unavailable.",
        "The current HEAD commit hash was identified as unknown.",
        "The current HEAD commit ID was resolved, but failed to resolve.",
        "The current HEAD commit ID was resolved but is unresolved.",
        "The current HEAD commit hash was identified but is missing.",
        "The current HEAD commit ID was resolved, but is not available.",
        "The current HEAD commit hash was identified, but is not known.",
        "The current HEAD commit ID was resolved, but identification failed.",
        "The current HEAD commit ID was resolved, but failed to identify it.",
    ):
        assert not matches(finding)


def test_git_full_scorer_rejects_wrong_hash_next_to_word_character(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "git_approval_replanning")

    def factory(selected, _goal):  # type: ignore[no-untyped-def]
        def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
            return AgentAnswer(
                summary="The probes completed.",
                findings=(
                    selected.claims[0].answer_text,
                    f"The current HEAD commit ID was resolved as hash_{'0' * 40}.",
                ),
                next_actions=("Keep the observed identity.",),
                citations=_case_citations(selected, catalog),
                issue_present=True,
            )

        return benchmark_module._BenchmarkReplanningPlanner(
            steps=selected.steps,
            build_answer=answer,
        )

    results = run_case_with_planner(
        case,
        tmp_path,
        ScopedTrustStore(tmp_path / "wrong-hash-trust.sqlite"),
        factory,
    )
    assert len(results) == 3
    assert all(not result.passed for result in results)
    assert all("semantic claims" in result.reason for result in results)


def test_semantic_predicates_bind_to_the_nearest_subject(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="Both activity names occur, but the predicates are reversed.",
            findings=(
                (
                    "DeepLinkActivity is compared with InternalSettingsActivity, "
                    "which is exported and externally reachable."
                ),
                (
                    "InternalSettingsActivity is compared with DeepLinkActivity, "
                    "which is private and not exported."
                ),
            ),
            next_actions=("Do not accept cross-subject predicate binding.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


def test_semantic_inverted_predicates_bind_to_the_following_subject(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The predicates are inverted onto the opposite activities.",
            findings=(
                (
                    "DeepLinkActivity is under review. Exported and externally reachable "
                    "is InternalSettingsActivity."
                ),
                (
                    "InternalSettingsActivity is under review. Private and not exported "
                    "is DeepLinkActivity."
                ),
            ),
            next_actions=("Do not accept inverted subject assignment.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


@pytest.mark.parametrize("connector", ("because", "as", "given that"))
def test_semantic_subordinate_predicates_bind_to_the_following_subject(
    tmp_path: Path,
    connector: str,
) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The subordinate predicates are assigned to the opposite activities.",
            findings=(
                (
                    f"DeepLinkActivity is different, {connector} the exported and "
                    "externally reachable one is InternalSettingsActivity."
                ),
                (
                    f"InternalSettingsActivity is different, {connector} the private and "
                    "not exported one is DeepLinkActivity."
                ),
            ),
            next_actions=("Do not accept cross-clause subject inversion.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


def test_semantic_scorer_rejects_ambiguous_noncopular_following_subject(
    tmp_path: Path,
) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="Both predicates are positioned ambiguously between two subjects.",
            findings=(
                (
                    "DeepLinkActivity is different, with exported and externally reachable "
                    "applying to InternalSettingsActivity."
                ),
                (
                    "InternalSettingsActivity is different, with private and not exported "
                    "applying to DeepLinkActivity."
                ),
            ),
            next_actions=("Use one explicit subject per finding.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


@pytest.mark.parametrize(
    "unsafe_finding",
    (
        "DeepLinkActivity is not only exported but also externally reachable.",
        ("DeepLinkActivity is listed, and it is not only exported but also externally reachable."),
        ("DeepLinkActivity is listed, and it isn't only exported but also externally reachable."),
    ),
)
def test_semantic_scorer_accepts_not_only_emphasis(
    tmp_path: Path,
    unsafe_finding: str,
) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The public and private activities have different exposure.",
            findings=(
                unsafe_finding,
                case.claims[1].answer_text,
            ),
            next_actions=("Keep internal settings private.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert passed, reason


@pytest.mark.parametrize(
    ("unsafe_finding", "safe_finding"),
    (
        (
            (
                "DeepLinkActivity is not only exported but also externally reachable, "
                "though it is not."
            ),
            None,
        ),
        (
            None,
            "InternalSettingsActivity is private and not exported, though it is not.",
        ),
    ),
)
def test_semantic_scorer_rejects_unresolved_anaphoric_negation(
    tmp_path: Path,
    unsafe_finding: str | None,
    safe_finding: str | None,
) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="A trailing pronoun contradicts one of the findings.",
            findings=(
                unsafe_finding or case.claims[0].answer_text,
                safe_finding or case.claims[1].answer_text,
            ),
            next_actions=("Require an explicit subject for every negation.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


@pytest.mark.parametrize(
    "unsafe_finding",
    (
        "DeepLinkActivity is neither exported nor externally reachable.",
        (
            "DeepLinkActivity is exported and externally reachable, though it never "
            "remains exported or externally reachable."
        ),
    ),
)
def test_semantic_scorer_rejects_neither_and_bare_anaphoric_negation(
    tmp_path: Path,
    unsafe_finding: str,
) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The unsafe claim is explicitly contradicted.",
            findings=(unsafe_finding, case.claims[1].answer_text),
            next_actions=("Reject the contradiction.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


@pytest.mark.parametrize(
    "unsafe_finding",
    (
        "ProfileViewController updates nameLabel off the main thread.",
        (
            "ProfileViewController updates nameLabel on the global background queue "
            "instead of the main queue."
        ),
    ),
)
def test_semantic_scorer_accepts_off_main_thread_wording(
    tmp_path: Path, unsafe_finding: str
) -> None:
    case = next(item for item in default_cases() if item.name == "ios_main_thread_ui")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The unsafe and safe paths differ by queue confinement.",
            findings=(
                unsafe_finding,
                case.claims[1].answer_text,
            ),
            next_actions=("Dispatch the label update to the main queue.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert passed, reason


def test_semantic_scorer_rejects_terms_negated_for_the_target(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "ios_main_thread_ui")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="Queue vocabulary alone does not prove a UI mutation.",
            findings=(
                (
                    "ProfileViewController runs unrelated work on a global background "
                    "queue and does not update nameLabel."
                ),
                case.claims[1].answer_text,
            ),
            next_actions=("Keep predicates bound to the named controller.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


def test_semantic_scorer_preserves_outer_subject_across_parenthetical_contrast(
    tmp_path: Path,
) -> None:
    case = next(item for item in default_cases() if item.name == "ios_main_thread_ui")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The two controllers differ in queue confinement.",
            findings=(
                (
                    "ProfileViewController, unlike AvatarViewController which uses the "
                    "main queue, updates nameLabel on a global background queue."
                ),
                case.claims[1].answer_text,
            ),
            next_actions=("Move the profile mutation to the main queue.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert passed, reason


def test_semantic_scorer_requires_the_negative_control(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "cpp_dangling_view")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        citations = _case_citations(case, catalog)
        return AgentAnswer(
            summary="The defect is present.",
            findings=(case.claims[0].answer_text,),
            next_actions=("Return owning storage.",),
            citations=(citations[0],),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    assert report.verdict is InvestigationVerdict.PASS
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


def test_semantic_scorer_rejects_right_words_bound_to_wrong_anchor(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "ios_main_thread_ui")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        citations = _case_citations(case, catalog)
        return AgentAnswer(
            summary="Both UI paths were inspected.",
            findings=tuple(claim.answer_text for claim in case.claims),
            next_actions=("Move the unsafe mutation to the main queue.",),
            citations=tuple(reversed(citations)),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "validated evidence anchors" in reason


def test_semantic_scorer_rejects_broad_overlapping_anchor_ranges(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        read = next(item for item in catalog if item.tool == "read_file")
        return AgentAnswer(
            summary="Both activities were inspected.",
            findings=tuple(claim.answer_text for claim in case.claims),
            next_actions=("Keep the internal activity private.",),
            citations=(
                SourceCitation(read.observation_id, read.path, 1, 8),
                SourceCitation(read.observation_id, read.path, 1, 7),
            ),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    assert report.verdict is InvestigationVerdict.PASS
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "validated evidence anchors" in reason


def test_tampered_anchor_prevents_case_and_suite_pass(tmp_path: Path) -> None:
    cases = list(default_cases())
    original = cases[0]
    first = original.claims[0]
    bad_anchor = replace(first.anchor, content_sha256="0" * 64)
    cases[0] = replace(original, claims=(replace(first, anchor=bad_anchor), *original.claims[1:]))
    result = run_benchmark(tuple(cases), root=tmp_path)
    assert result.cases_passed == 6
    assert not result.gate_passed


def test_claims_use_hashed_minimal_evidence_ranges() -> None:
    cases = {case.name: case for case in default_cases()}
    ios_anchor = cases["ios_main_thread_ui"].claims[0].anchor
    pytorch_anchor = cases["pytorch_eval_mode"].claims[1].anchor
    assert ios_anchor.end_line - ios_anchor.start_line + 1 == 4
    assert pytorch_anchor.end_line - pytorch_anchor.start_line + 1 == 3
    assert ios_anchor.content_sha256
    assert pytorch_anchor.content_sha256
    for case in cases.values():
        for claim in case.claims:
            if claim.anchor.command is not None:
                continue
            lines = case.files[claim.anchor.path].splitlines()
            anchor_text = (
                claim.anchor.path
                + "\n"
                + "\n".join(lines[claim.anchor.start_line - 1 : claim.anchor.end_line])
            )
            normalized_anchor = re.sub(r"[^a-z0-9]", "", anchor_text.casefold())
            normalized_subject = re.sub(r"[^a-z0-9]", "", claim.term_groups[0][0].casefold())
            assert normalized_subject in normalized_anchor, claim.claim_id


def test_negated_architecture_relation_is_rejected(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "generic_architecture")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The path was traced.",
            findings=(
                "gateway.handle does not forward the request to billing_worker.enqueue.",
                case.claims[1].answer_text,
            ),
            next_actions=("Keep the queue contract documented.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


def test_disconnected_architecture_relationships_are_rejected(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "generic_architecture")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The named components are explicitly disconnected.",
            findings=(
                "gateway.handle is unrelated to billing_worker.enqueue.",
                "billing_worker says enqueue is disconnected from BILLING_QUEUE.",
            ),
            next_actions=("Do not infer a relationship from endpoint names.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "semantic claims" in reason


def test_empty_required_read_does_not_satisfy_full_stack_case(tmp_path: Path) -> None:
    original = next(item for item in default_cases() if item.name == "django_react_injection")
    case = replace(
        original,
        steps=(
            *original.steps[:2],
            ToolCall(tool="read_file", path="package.json", start_line=999),
        ),
    )

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The full-stack paths were inspected.",
            findings=tuple(claim.answer_text for claim in case.claims),
            next_actions=("Keep the safe controls.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    assert report.verdict is InvestigationVerdict.PASS
    passed, reason = _score_variant_model(case, report, workspace)
    assert not passed
    assert "required observation" in reason


def test_truncated_reads_do_not_satisfy_required_observations(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="Both activities were inspected.",
            findings=tuple(claim.answer_text for claim in case.claims),
            next_actions=("Keep the private control.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    required = {(item.tool, item.path) for item in case.required_observations}
    truncated_catalog = tuple(
        replace(item, truncated=True) if (item.tool, item.path) in required else item
        for item in report.catalog
    )
    passed, reason = _score_variant_model(
        case,
        replace(report, catalog=truncated_catalog),
        workspace,
    )
    assert not passed
    assert "required observation" in reason


def test_wholly_redacted_reads_do_not_satisfy_required_observations(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="Both activities were inspected.",
            findings=tuple(claim.answer_text for claim in case.claims),
            next_actions=("Keep the private control.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    required = {(item.tool, item.path) for item in case.required_observations}
    redacted_catalog = []
    for observation in report.catalog:
        if (observation.tool, observation.path) not in required:
            redacted_catalog.append(observation)
            continue
        redacted_lines = tuple(
            range(observation.start_line, observation.start_line + len(observation.lines))
        )
        redacted_catalog.append(
            replace(
                observation,
                text="[REDACTED]",
                lines=tuple(f"{line}: [REDACTED]" for line in redacted_lines),
                incomplete=True,
                redacted=True,
                metadata={**observation.metadata, "redacted_lines": redacted_lines},
            )
        )
    passed, reason = _score_variant_model(
        case,
        replace(report, catalog=tuple(redacted_catalog)),
        workspace,
    )
    assert not passed
    assert "required observation" in reason


def test_redaction_outside_returned_window_preserves_substantive_evidence(
    tmp_path: Path,
) -> None:
    case = next(item for item in default_cases() if item.name == "generic_architecture")

    def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        return AgentAnswer(
            summary="The gateway-to-queue path was traced.",
            findings=tuple(claim.answer_text for claim in case.claims),
            next_actions=("Keep the queue boundary observable.",),
            citations=_case_citations(case, catalog),
            issue_present=True,
        )

    report, workspace = _run_source_answer(case, tmp_path, answer)
    catalog: list[ToolObservation] = []
    for observation in report.catalog:
        if observation.path != "src/billing_worker.py":
            catalog.append(observation)
            continue
        visible_lines = observation.lines[2:]
        catalog.append(
            replace(
                observation,
                text="\n".join(visible_lines),
                lines=visible_lines,
                start_line=3,
                truncated=False,
                incomplete=True,
                redacted=True,
                metadata={**observation.metadata, "redacted_lines": ()},
            )
        )

    passed, reason = _score_variant_model(
        case,
        replace(report, catalog=tuple(catalog)),
        workspace,
    )
    assert passed, reason


def test_gate_rejects_empty_and_partial_suite_definitions(tmp_path: Path) -> None:
    empty = run_benchmark((), root=tmp_path / "empty")
    partial = run_benchmark((default_cases()[0],), root=tmp_path / "partial")
    assert not empty.gate_passed
    assert not partial.gate_passed
    assert "benchmark: invalid_suite_contract" in empty.integrity_failures
    assert "benchmark: invalid_suite_contract" in partial.integrity_failures


def _report(**changes: object) -> InvestigationReport:
    base = InvestigationReport(
        run_id="integrity",
        verdict=InvestigationVerdict.PASS,
        stop_reason=StopReason.FINISHED,
        answer=None,
        catalog=(),
        decisions_used=1,
        tool_calls_used=0,
        physical_requests_used=0,
    )
    return replace(base, **changes)


def test_integrity_failures_cover_policy_protocol_secret_model_and_budgets() -> None:
    case = default_cases()[0]
    budget = AgentBudget()
    assert "unsupported_citation" in _integrity_failures(
        case,
        _report(stop_reason=StopReason.UNSUPPORTED_CITATION),
        budget,
        expected_model=None,
    )
    assert "path_or_command_policy_violation" in _integrity_failures(
        case,
        _report(stop_reason=StopReason.POLICY_VIOLATION),
        budget,
        expected_model=None,
    )
    assert "unrecovered_model_protocol_failure" in _integrity_failures(
        case,
        _report(stop_reason=StopReason.PROTOCOL_FAILURE),
        budget,
        expected_model=None,
    )
    secret_case = replace(case, forbidden_secrets=("secret-canary",))
    leaking = ToolObservation("obs", "read_file", "x.py", "h", "secret-canary")
    assert "unredacted_secret" in _integrity_failures(
        secret_case,
        _report(catalog=(leaking,)),
        budget,
        expected_model=None,
    )
    leaking_citation_note = AgentAnswer(
        summary="Evidence was inspected.",
        findings=("A supported finding.",),
        next_actions=(),
        citations=(SourceCitation("obs", "x.py", 1, 1, note="secret-canary"),),
    )
    assert "unredacted_secret" in _integrity_failures(
        secret_case,
        _report(answer=leaking_citation_note),
        budget,
        expected_model=None,
    )
    call = ModelCallRecord(
        request_index=1,
        logical_decision=1,
        requested_completion_tokens=1024,
        charged_completion_tokens=10,
        reported_prompt_tokens=100,
        reported_completion_tokens=10,
        reported_model="substituted-model",
        latency_seconds=0.1,
        outcome="success",
    )
    assert "endpoint_model_mismatch" in _integrity_failures(
        case,
        _report(model_calls=(call,), physical_requests_used=1),
        budget,
        expected_model="requested-model",
    )
    assert "decision_budget_breach" in _integrity_failures(
        case,
        _report(decisions_used=budget.max_decisions + 1),
        budget,
        expected_model=None,
    )
    assert "command_budget_breach" in _integrity_failures(
        case,
        _report(
            decisions_used=budget.max_command_calls + 1,
            tool_calls_used=budget.max_command_calls + 1,
            command_calls_used=budget.max_command_calls + 1,
        ),
        budget,
        expected_model=None,
    )
    assert "command_tool_accounting_invalid" in _integrity_failures(
        case,
        _report(command_calls_used=1),
        budget,
        expected_model=None,
    )


def test_recovered_protocol_call_still_enforces_model_provenance() -> None:
    case = default_cases()[0]
    calls = (
        ModelCallRecord(1, 1, 100, 10, 20, 10, "substituted-model", 0.1, "protocol_error"),
        ModelCallRecord(2, 1, 100, 10, 20, 10, "requested-model", 0.1, "success"),
    )
    report = _report(
        model_calls=calls,
        physical_requests_used=2,
        completion_tokens_requested=200,
        completion_tokens_charged=20,
        completion_tokens_used=20,
        schema_retries=1,
    )
    assert "endpoint_model_mismatch" in _integrity_failures(
        case,
        report,
        AgentBudget(),
        expected_model="requested-model",
    )


def test_recovered_protocol_envelope_requires_transport_attribution() -> None:
    case = default_cases()[0]
    calls = (
        ModelCallRecord(1, 1, 100, 10, 20, 10, None, 0.1, "protocol_error"),
        ModelCallRecord(2, 1, 100, 10, 20, 10, "requested-model", 0.1, "success"),
    )
    report = _report(
        model_calls=calls,
        physical_requests_used=2,
        completion_tokens_requested=200,
        completion_tokens_charged=20,
        completion_tokens_used=20,
        schema_retries=1,
    )
    audit = ModelEndpointAudit(
        trusted_planner=True,
        configured_model="requested-model",
        successful_responses=2,
        attributed_responses=1,
        reported_models=(None, "requested-model"),
    )

    failures = _integrity_failures(
        case,
        report,
        AgentBudget(),
        expected_model="requested-model",
        model_endpoint_audit=audit,
    )
    assert "endpoint_model_mismatch" in failures
    assert "model_transport_accounting_invalid" in failures


def test_transport_audit_reconciles_schema_error_response_envelopes() -> None:
    case = default_cases()[0]
    calls = (
        ModelCallRecord(1, 1, 100, 10, 20, 10, "requested-model", 0.1, "schema_error"),
        ModelCallRecord(2, 1, 100, 10, 20, 10, "requested-model", 0.1, "success"),
    )
    report = _report(
        model_calls=calls,
        physical_requests_used=2,
        completion_tokens_requested=200,
        completion_tokens_charged=20,
        completion_tokens_used=20,
        schema_retries=1,
    )
    audit = ModelEndpointAudit(
        trusted_planner=True,
        configured_model="requested-model",
        successful_responses=2,
        attributed_responses=2,
        reported_models=("requested-model", "requested-model"),
    )
    failures = _integrity_failures(
        case,
        report,
        AgentBudget(),
        expected_model="requested-model",
        model_endpoint_audit=audit,
    )
    assert "model_transport_accounting_invalid" not in failures
    assert "endpoint_model_mismatch" not in failures
    assert "model_decision_coverage_invalid" not in failures


def test_compaction_calls_are_accounted_without_faking_extra_logical_decisions() -> None:
    case = default_cases()[0]
    calls = (
        ModelCallRecord(
            1,
            1,
            100,
            10,
            20,
            10,
            "requested-model",
            0.1,
            "success",
            "compaction",
        ),
        ModelCallRecord(
            2,
            1,
            100,
            10,
            20,
            10,
            "requested-model",
            0.1,
            "success",
            "decision",
        ),
    )
    report = _report(
        model_calls=calls,
        physical_requests_used=2,
        completion_tokens_requested=200,
        completion_tokens_charged=20,
        completion_tokens_used=20,
    )
    audit = ModelEndpointAudit(
        trusted_planner=True,
        configured_model="requested-model",
        successful_responses=2,
        attributed_responses=2,
        reported_models=("requested-model", "requested-model"),
    )

    failures = _integrity_failures(
        case,
        report,
        AgentBudget(),
        expected_model="requested-model",
        model_endpoint_audit=audit,
    )

    assert "model_call_sequence_invalid" not in failures
    assert "model_decision_coverage_invalid" not in failures
    assert "retry_accounting_invalid" not in failures


def test_interrupted_model_call_is_a_valid_recovery_ledger_transition() -> None:
    case = default_cases()[0]
    calls = (
        ModelCallRecord(
            1,
            1,
            100,
            100,
            None,
            None,
            None,
            0.1,
            "process_interrupted",
        ),
        ModelCallRecord(
            2,
            1,
            100,
            10,
            20,
            10,
            "requested-model",
            0.1,
            "success",
        ),
    )
    report = _report(
        model_calls=calls,
        physical_requests_used=2,
        completion_tokens_requested=200,
        completion_tokens_charged=110,
        completion_tokens_used=10,
    )
    audit = ModelEndpointAudit(
        trusted_planner=True,
        configured_model="requested-model",
        successful_responses=1,
        attributed_responses=1,
        reported_models=("requested-model",),
    )

    failures = _integrity_failures(
        case,
        report,
        AgentBudget(),
        expected_model="requested-model",
        model_endpoint_audit=audit,
    )

    assert "model_call_accounting_invalid" not in failures
    assert "model_call_sequence_invalid" not in failures
    assert "retry_accounting_invalid" not in failures


def test_endpoint_audit_uses_per_variant_response_model_delta() -> None:
    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:1234/v1",
        model="requested-model",
    )
    planner = ModelInvestigationPlanner(client=client)
    client._record_successful_response("stale-model")
    baseline = _start_model_endpoint_audit(planner, client)

    client._record_successful_response("requested-model")
    audit = _finish_model_endpoint_audit(planner, client, baseline)

    assert audit.trusted_planner
    assert audit.successful_responses == 1
    assert audit.attributed_responses == 1
    assert audit.reported_models == ("requested-model",)


def test_endpoint_audit_rejects_descriptor_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:1234/v1",
        model="requested-model",
    )
    planner = ModelInvestigationPlanner(client=client)
    assert _model_planner_is_trusted(planner, client)

    with monkeypatch.context() as patch:
        patch.setattr(
            OpenAICompatibleClient,
            "_read_response",
            classmethod(lambda cls, response, connection, deadline: b"{}"),
        )
        assert not _model_planner_is_trusted(planner, client)

    with monkeypatch.context() as patch:
        patch.setattr(
            OpenAICompatibleClient,
            "successful_response_count",
            property(lambda self: 0),
        )
        assert not _model_planner_is_trusted(planner, client)


def test_expected_model_requires_nonempty_call_ledger_in_core_gate(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")

    def factory(selected, goal):  # type: ignore[no-untyped-def]
        del goal

        def answer(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
            return AgentAnswer(
                summary="Android evidence inspected.",
                findings=tuple(claim.answer_text for claim in selected.claims),
                next_actions=("Keep the control private.",),
                citations=_case_citations(selected, catalog),
                issue_present=selected.expected_issue,
            )

        return ScriptedInvestigationPlanner(steps=selected.steps, build_answer=answer)

    result = run_benchmark_with_planner(
        (case,),
        factory,
        root=tmp_path,
        expected_model="never-observed",
    )
    assert not result.gate_passed
    assert all("model_call_ledger_missing" in item.integrity_failures for item in result.variants)


def test_forged_planner_ledger_cannot_replace_transport_owned_provenance(
    tmp_path: Path,
) -> None:
    case = next(item for item in default_cases() if item.name == "android_exported_activity")
    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:1234/v1",
        model="forged-model",
    )

    class ForgedPlanner:
        def __init__(self, selected) -> None:  # type: ignore[no-untyped-def]
            self.inner = ScriptedInvestigationPlanner(
                steps=selected.steps,
                build_answer=lambda catalog: AgentAnswer(
                    summary="Android evidence inspected.",
                    findings=tuple(claim.answer_text for claim in selected.claims),
                    next_actions=("Keep the private control.",),
                    citations=_case_citations(selected, catalog),
                    issue_present=selected.expected_issue,
                ),
            )
            self.requests_made = 0
            self.completion_tokens_requested = 0
            self.completion_tokens_charged = 0
            self.completion_tokens_reported = 0
            self.transport_retries = 0
            self.schema_retries = 0
            self.model_calls: list[ModelCallRecord] = []

        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]):  # type: ignore[no-untyped-def]
            decision = self.inner.decide(goal=goal, catalog=catalog)
            self.requests_made += 1
            self.completion_tokens_requested += 1024
            self.completion_tokens_charged += 1
            self.completion_tokens_reported += 1
            self.model_calls.append(
                ModelCallRecord(
                    self.requests_made,
                    self.requests_made,
                    1024,
                    1,
                    1,
                    1,
                    "forged-model",
                    0.001,
                    "success",
                )
            )
            return decision

    results = run_case_with_planner(
        case,
        tmp_path,
        ScopedTrustStore(tmp_path / "forged-trust.sqlite"),
        lambda selected, _goal: ForgedPlanner(selected),
        expected_model="forged-model",
        model_client=client,
    )
    assert all(not item.passed for item in results)
    assert all("untrusted_model_planner" in item.integrity_failures for item in results)
    assert all("model_transport_accounting_invalid" in item.integrity_failures for item in results)

    def shadowed_factory(selected, goal):  # type: ignore[no-untyped-def]
        del goal
        delegate = ForgedPlanner(selected)
        planner = ModelInvestigationPlanner(client=client)

        def scripted_decide(*, goal: str, catalog: tuple[ToolObservation, ...]):  # type: ignore[no-untyped-def]
            decision = delegate.decide(goal=goal, catalog=catalog)
            planner.requests_made = delegate.requests_made
            planner.completion_tokens_requested = delegate.completion_tokens_requested
            planner.completion_tokens_charged = delegate.completion_tokens_charged
            planner.completion_tokens_reported = delegate.completion_tokens_reported
            planner.model_calls = list(delegate.model_calls)
            return decision

        planner.decide = scripted_decide  # type: ignore[method-assign]
        return planner

    shadowed_results = run_case_with_planner(
        case,
        tmp_path / "shadowed",
        ScopedTrustStore(tmp_path / "shadowed-trust.sqlite"),
        shadowed_factory,
        expected_model="forged-model",
        model_client=client,
    )
    assert all(not item.passed for item in shadowed_results)
    assert all("untrusted_model_planner" in item.integrity_failures for item in shadowed_results)


def test_expected_model_requires_successful_call_for_every_decision() -> None:
    case = default_cases()[0]
    one_success = ModelCallRecord(1, 1, 100, 10, 20, 10, "requested", 0.1, "success")
    mostly_scripted = _report(
        decisions_used=3,
        physical_requests_used=1,
        model_calls=(one_success,),
        completion_tokens_requested=100,
        completion_tokens_charged=10,
        completion_tokens_used=10,
    )
    assert "model_decision_coverage_invalid" in _integrity_failures(
        case,
        mostly_scripted,
        AgentBudget(),
        expected_model="requested",
    )

    transport_only = replace(
        one_success,
        charged_completion_tokens=100,
        reported_completion_tokens=None,
        outcome="transport_error",
    )
    failures = _integrity_failures(
        case,
        _report(
            physical_requests_used=1,
            model_calls=(transport_only,),
            completion_tokens_requested=100,
            completion_tokens_charged=100,
        ),
        AgentBudget(),
        expected_model="requested",
    )
    assert "model_decision_coverage_invalid" in failures
    assert "expected_model_invalid" in _integrity_failures(
        case,
        _report(),
        AgentBudget(),
        expected_model="",
    )


def test_retry_accounting_requires_failure_typed_transitions() -> None:
    case = default_cases()[0]
    calls = tuple(
        ModelCallRecord(index, 1, 100, 10, 20, 10, "requested", 0.1, "success")
        for index in range(1, 4)
    )
    failures = _integrity_failures(
        case,
        _report(
            physical_requests_used=3,
            model_calls=calls,
            completion_tokens_requested=300,
            completion_tokens_charged=30,
            completion_tokens_used=30,
            transport_retries=2,
        ),
        AgentBudget(),
        expected_model=None,
    )
    assert "retry_accounting_invalid" in failures

    too_many_transport_retries = (
        ModelCallRecord(1, 1, 100, 100, None, None, None, 0.1, "transport_error"),
        ModelCallRecord(2, 1, 100, 100, None, None, None, 0.1, "transport_error"),
        ModelCallRecord(3, 1, 100, 100, None, None, None, 0.1, "transport_error"),
        ModelCallRecord(4, 1, 100, 10, 20, 10, "requested", 0.1, "success"),
    )
    failures = _integrity_failures(
        case,
        _report(
            physical_requests_used=4,
            model_calls=too_many_transport_retries,
            completion_tokens_requested=400,
            completion_tokens_charged=310,
            completion_tokens_used=10,
            transport_retries=3,
        ),
        AgentBudget(),
        expected_model=None,
    )
    assert "retry_accounting_invalid" in failures

    noncontiguous_successes = (
        ModelCallRecord(1, 1, 100, 10, 20, 10, "requested", 0.1, "success"),
        ModelCallRecord(2, 2, 100, 10, 20, 10, "requested", 0.1, "success"),
        ModelCallRecord(3, 1, 100, 10, 20, 10, "requested", 0.1, "success"),
    )
    failures = _integrity_failures(
        case,
        _report(
            decisions_used=2,
            physical_requests_used=3,
            model_calls=noncontiguous_successes,
            completion_tokens_requested=300,
            completion_tokens_charged=30,
            completion_tokens_used=30,
        ),
        AgentBudget(),
        expected_model=None,
    )
    assert "model_call_sequence_invalid" in failures


def test_integrity_reconciles_per_call_completion_ledger_and_cap() -> None:
    case = default_cases()[0]
    calls = tuple(
        ModelCallRecord(index, index, 4096, 4096, None, None, None, 0.1, "transport_error")
        for index in range(1, 8)
    )
    failures = _integrity_failures(
        case,
        _report(decisions_used=7, physical_requests_used=7, model_calls=calls),
        AgentBudget(),
        expected_model=None,
    )
    assert "completion_charge_accounting_invalid" in failures
    assert "completion_request_accounting_invalid" in failures
    assert "completion_token_budget_breach" in failures


def test_parent_probe_requires_exact_non_timeout_git_failure() -> None:
    valid: dict[str, object] = {
        "status": "failed",
        "returncode": 128,
        "rule": "git-parent-commit",
        "stdout": "",
        "stderr": "fatal: Needed a single revision\n",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "reason": "completed in 0.012s",
    }
    assert _parent_probe_proves_missing_revision(valid)
    assert not _parent_probe_proves_missing_revision(
        {**valid, "returncode": None, "reason": "command exceeded compute budget after 1 seconds"}
    )
    assert not _parent_probe_proves_missing_revision(
        {**valid, "stderr": "fatal: repository unavailable"}
    )


def test_fixed_git_command_counter_fails_without_observation_dependency(tmp_path: Path) -> None:
    case = next(item for item in default_cases() if item.name == "git_approval_replanning")
    trust = ScopedTrustStore(tmp_path / "counter-trust.sqlite")

    def factory(selected, goal):  # type: ignore[no-untyped-def]
        del goal
        return ScriptedInvestigationPlanner(
            steps=selected.steps,
            build_answer=lambda _catalog: AgentAnswer(
                summary="counter",
                findings=("counter",),
                next_actions=("stop",),
                citations=(),
                complete=False,
            ),
        )

    results = run_case_with_planner(case, tmp_path, trust, factory)
    assert all(not item.passed for item in results)
    assert all("path_or_command_policy_violation" in item.integrity_failures for item in results)


def test_precomputed_git_observation_id_cannot_fake_replanning(tmp_path: Path) -> None:
    original = next(item for item in default_cases() if item.name == "git_approval_replanning")
    case = replace(
        original,
        steps=(
            original.steps[0],
            replace(
                original.steps[1],
                based_on_observation_id="obs_git_1_b6549f3c83a8",
            ),
        ),
    )
    trust = ScopedTrustStore(tmp_path / "precomputed-trust.sqlite")

    def factory(selected, goal):  # type: ignore[no-untyped-def]
        del goal
        return ScriptedInvestigationPlanner(
            steps=selected.steps,
            build_answer=lambda _catalog: AgentAnswer(
                summary="precomputed dependency",
                findings=("precomputed dependency",),
                next_actions=("stop",),
                citations=(),
                complete=False,
            ),
        )

    results = run_case_with_planner(case, tmp_path, trust, factory)
    assert all(not item.passed for item in results)
    assert all("path_or_command_policy_violation" in item.integrity_failures for item in results)


def test_compliant_budget_exhaustion_is_not_an_integrity_failure() -> None:
    case = default_cases()[0]
    budget = AgentBudget()
    report = _report(
        verdict=InvestigationVerdict.INCOMPLETE,
        stop_reason=StopReason.BUDGET_EXHAUSTED,
        decisions_used=budget.max_decisions,
    )
    assert _integrity_failures(case, report, budget, expected_model=None) == ()


def test_deadline_after_successful_request_counts_the_logical_decision(tmp_path: Path) -> None:
    case = default_cases()[0]
    workspace = materialize(case, tmp_path)
    trust = ScopedTrustStore(tmp_path / "deadline-trust.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="test")

    class SlowPlanner:
        requests_made = 0
        completion_tokens_requested = 0
        completion_tokens_charged = 0
        completion_tokens_reported = 0
        transport_retries = 0
        schema_retries = 0
        model_calls: list[ModelCallRecord] = []
        active_deadline: float | None = None

        def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> ToolCall:
            del goal, catalog
            assert self.active_deadline is not None
            time.sleep(max(0.0, self.active_deadline - time.monotonic()) + 0.01)
            self.requests_made = 1
            self.completion_tokens_requested = 1024
            self.completion_tokens_charged = 1
            self.completion_tokens_reported = 1
            self.model_calls = [ModelCallRecord(1, 1, 1024, 1, 1, 1, "model", 0.01, "success")]
            return ToolCall(tool="list_files", path=".")

    budget = AgentBudget(
        max_decisions=1,
        max_tool_calls=1,
        max_physical_requests=1,
        max_completion_tokens=1024,
        max_active_seconds=0.2,
    )
    report = InvestigationLoop(planner=SlowPlanner(), trust=trust, budget=budget).run(
        run_id="deadline-after-model",
        goal="inspect",
        workspace=workspace,
    )
    assert report.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert report.decisions_used == 1
    assert _integrity_failures(case, report, budget, expected_model=None) == ()


def test_any_integrity_failure_is_global_even_with_21_capability_passes(tmp_path: Path) -> None:
    result = run_benchmark(default_cases(), root=tmp_path)
    poisoned = replace(
        result.variants[0],
        passed=True,
        integrity_failures=("unsupported_citation",),
    )
    result = replace(result, variants=(poisoned, *result.variants[1:]))
    assert result.variants_passed == 21
    assert result.cases_passed == 7
    assert result.integrity_failures
    assert not result.gate_passed

    all_failed = replace(
        result,
        variants=tuple(replace(variant, passed=False) for variant in result.variants),
    )
    assert all_failed.variants_passed == 0
    assert all_failed.cases_passed == 0
    assert not all_failed.gate_passed

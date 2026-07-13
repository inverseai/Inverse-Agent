"""The seven-domain investigation benchmark gate."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.attestations import AttestationScope, ScopedTrustStore
from inverse_agent.investigation import (
    AgentAnswer,
    Decision,
    InvestigationLoop,
    SourceCitation,
    ToolCall,
    ToolObservation,
)
from inverse_agent.investigation_benchmark import (
    _score_variant_model,
    default_cases,
    materialize,
    run_benchmark,
)


def test_seven_domains_present() -> None:
    cases = default_cases()
    assert len(cases) == 7
    domains = {case.domain for case in cases}
    names = {case.name for case in cases}
    # The five priority stacks plus generic research and git observation.
    assert "android_exported_activity" in names
    assert "ios_main_thread_ui" in names
    assert "cpp_dangling_view" in names
    assert "django_react_injection" in names
    assert "pytorch_eval_mode" in names
    assert "generic_architecture" in names
    assert "git_observation" in names
    assert "android" in domains and "ios" in domains and "pytorch" in domains


def test_every_case_has_three_variants() -> None:
    for case in default_cases():
        assert len(case.goal_variants) == 3


def test_benchmark_gate_passes(tmp_path: Path) -> None:
    result = run_benchmark(default_cases(), root=tmp_path)
    # Deterministic solver: every variant should pass -> 7/7 cases, 21/21 variants.
    assert result.total_cases == 7
    assert result.total_variants == 21
    assert result.cases_passed == 7, [v for v in result.variants if not v.passed]
    assert result.variants_passed == 21, [v for v in result.variants if not v.passed]
    assert result.gate_passed


class _FixedPlanner:
    """Drives the real loop: reads the marker file, then emits a chosen answer."""

    def __init__(self, read_path: str, answer_factory) -> None:  # type: ignore[no-untyped-def]
        self._read_path = read_path
        self._answer_factory = answer_factory
        self._read = False

    def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
        del goal
        if not self._read:
            self._read = True
            return ToolCall(tool="read_file", path=self._read_path)
        return self._answer_factory(catalog)


def _run_with_answer(case, tmp_path, answer_factory):  # type: ignore[no-untyped-def]
    workspace = materialize(case, tmp_path)
    trust = ScopedTrustStore(tmp_path / "t.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="t")
    read_path = next(iter(case.files))
    loop = InvestigationLoop(planner=_FixedPlanner(read_path, answer_factory), trust=trust)
    report = loop.run(run_id="s", goal=case.goal_variants[0], workspace=workspace)
    return _score_variant_model(case, report, workspace)


def _cite_marker(case, catalog):  # type: ignore[no-untyped-def]
    for obs in catalog:
        for numbered in obs.lines:
            head, _, body = numbered.partition(": ")
            if case.marker in body:
                return int(head), obs
    return None, None


def test_scorer_rejects_contrary_conclusion(tmp_path: Path) -> None:
    # Cites the real evidence line but asserts issue_present=False -> must fail.
    case = next(c for c in default_cases() if c.name == "android_exported_activity")

    def contrary(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        line, obs = _cite_marker(case, catalog)
        assert obs is not None
        return AgentAnswer(
            summary="Looks fine.",
            findings=("No activity is exported.",),
            next_actions=("Keep the manifest unchanged.",),
            citations=(SourceCitation(obs.observation_id, obs.path, line, line),),
            issue_present=False,
        )

    passed, reason = _run_with_answer(case, tmp_path, contrary)
    assert not passed
    assert "issue_present" in reason


def test_scorer_rejects_empty_findings(tmp_path: Path) -> None:
    case = next(c for c in default_cases() if c.name == "cpp_dangling_view")

    def empty(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        line, obs = _cite_marker(case, catalog)
        assert obs is not None
        return AgentAnswer(
            summary="",
            findings=(),
            next_actions=(),
            citations=(SourceCitation(obs.observation_id, obs.path, line, line),),
            issue_present=True,
        )

    passed, reason = _run_with_answer(case, tmp_path, empty)
    assert not passed
    assert "unsupported_citation" in reason


def test_scorer_accepts_correct_grounded_answer(tmp_path: Path) -> None:
    case = next(c for c in default_cases() if c.name == "pytorch_eval_mode")

    def correct(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        line, obs = _cite_marker(case, catalog)
        assert obs is not None
        return AgentAnswer(
            summary="evaluate uses train mode.",
            findings=("evaluate() calls model.train() instead of eval().",),
            next_actions=("call model.eval()",),
            citations=(SourceCitation(obs.observation_id, obs.path, line, line),),
            issue_present=True,
        )

    passed, reason = _run_with_answer(case, tmp_path, correct)
    assert passed, reason


def test_benchmark_gate_requires_all_cases(tmp_path: Path) -> None:
    # If we drop the evidence marker for one case, that case must fail the gate,
    # proving the scorer is grounded in real evidence, not a constant pass.
    cases = list(default_cases())
    broken = cases[0]
    tampered = type(broken)(
        name=broken.name,
        domain=broken.domain,
        files=broken.files,
        goal_variants=broken.goal_variants,
        steps=broken.steps,
        required_concept=broken.required_concept,
        marker="THIS_MARKER_DOES_NOT_EXIST_IN_ANY_FILE",
        concept_phrase=broken.concept_phrase,
    )
    cases[0] = tampered
    result = run_benchmark(tuple(cases), root=tmp_path)
    assert result.cases_passed == 6
    assert not result.gate_passed

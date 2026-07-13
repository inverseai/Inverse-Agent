from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from inverse_agent.cli import (
    BUILTIN_BENCHMARK_SUITE,
    _benchmark_suite_path,
    benchmark_review_command,
    review_commit_command,
)
from inverse_agent.commit_review import (
    CommitReviewReport,
    ReviewConfidence,
    ReviewDomain,
    ReviewFinding,
    ReviewSeverity,
)
from inverse_agent.model_config import PlannerConfig
from inverse_agent.review_benchmark import (
    BenchmarkDefinitionError,
    BenchmarkSuiteResult,
    ExpectedAnchor,
    ExpectedFinding,
    _select_consistent_assignment,
    load_benchmark_suite,
    run_benchmark_suite,
    score_review,
)

SUITE = Path(__file__).parents[1] / "benchmarks" / "commit_review" / "suite.json"
PACKAGED_BENCHMARK = (
    Path(__file__).parents[1] / "src" / "inverse_agent" / "benchmark_assets" / "commit_review"
)


class FakeReviewClient:
    observed_response_models = ("benchmark-model",)
    observed_response_models_overflowed = False
    response_model_mismatch_observed = False
    successful_response_count = 1
    attributed_response_count = 1

    def __init__(self, response: dict[str, Any] | list[dict[str, Any]]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.calls = 0

    def complete_structured_json(
        self,
        *,
        system: str,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        del system, prompt, schema_name, schema, max_tokens
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def test_packaged_benchmark_assets_match_checkout() -> None:
    checkout_root = SUITE.parent
    checkout_files = {
        path.relative_to(checkout_root): path.read_bytes()
        for path in checkout_root.rglob("*")
        if path.is_file()
    }
    packaged_files = {
        path.relative_to(PACKAGED_BENCHMARK): path.read_bytes()
        for path in PACKAGED_BENCHMARK.rglob("*")
        if path.is_file()
    }

    assert packaged_files == checkout_files
    assert (Path(__file__).parents[1] / "docs" / "commit-review-benchmark.md").read_bytes() == (
        Path(__file__).parents[1]
        / "src"
        / "inverse_agent"
        / "benchmark_assets"
        / "commit-review-benchmark.md"
    ).read_bytes()


def test_builtin_benchmark_suite_resolves_from_package_assets() -> None:
    with _benchmark_suite_path(BUILTIN_BENCHMARK_SUITE) as suite_path:
        assert suite_path.is_file()
        assert suite_path.read_bytes() == SUITE.read_bytes()
        suite = load_benchmark_suite(
            suite_path,
            repository_root=Path(__file__).parents[1],
        )

    assert len(suite) == 6
    assert suite[-1].workspace == Path(__file__).parents[1]


def test_benchmark_cli_prepares_output_directory_before_model_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "nested" / "results" / "benchmark.json"

    class Resolution:
        client = FakeReviewClient({"summary": "", "findings": []})
        config = PlannerConfig(
            kind="openai-compatible",
            base_url="http://127.0.0.1:1234/v1",
            model="benchmark-model",
        )

    def resolve(**_kwargs: object) -> Resolution:
        assert output.parent.is_dir()
        return Resolution()

    result = BenchmarkSuiteResult(
        suite="suite.json",
        passed=True,
        cases=(),
        passed_cases=0,
        total_cases=0,
        duration_seconds=0.0,
    )
    monkeypatch.setattr("inverse_agent.cli.resolve_planner", resolve)
    monkeypatch.setattr("inverse_agent.cli.run_benchmark_suite", lambda *_args, **_kwargs: result)
    args = type(
        "Args",
        (),
        {
            "output": str(output),
            "repository_root": None,
            "suite": str(tmp_path / "suite.json"),
        },
    )()

    assert benchmark_review_command(args) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    displayed = json.loads(capsys.readouterr().out)
    assert written["passed"] is True
    assert displayed["passed"] is True
    assert written["model_provenance"] == {
        "kind": "openai-compatible",
        "requested_model": "benchmark-model",
        "base_url": "http://127.0.0.1:1234/v1",
        "config_fingerprint": Resolution.config.fingerprint,
        "endpoint_reported_models": ["benchmark-model"],
        "successful_responses": 1,
        "attributed_responses": 1,
        "endpoint_model_consistent": True,
    }
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_benchmark_cli_fails_when_endpoint_reported_model_does_not_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Client:
        observed_response_models = ("different-model",)
        observed_response_models_overflowed = False
        response_model_mismatch_observed = True
        successful_response_count = 1
        attributed_response_count = 1

    class Resolution:
        client = Client()
        config = PlannerConfig(
            kind="openai-compatible",
            base_url="http://127.0.0.1:1234/v1",
            model="benchmark-model",
        )

    result = BenchmarkSuiteResult(
        suite="suite.json",
        passed=True,
        cases=(),
        passed_cases=0,
        total_cases=0,
        duration_seconds=0.0,
    )
    monkeypatch.setattr("inverse_agent.cli.resolve_planner", lambda **_kwargs: Resolution())
    monkeypatch.setattr(
        "inverse_agent.cli.run_benchmark_suite",
        lambda *_args, **_kwargs: result,
    )
    args = type(
        "Args",
        (),
        {
            "output": None,
            "repository_root": None,
            "suite": str(tmp_path / "suite.json"),
        },
    )()

    assert benchmark_review_command(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["model_provenance"]["endpoint_model_consistent"] is False
    assert payload["model_provenance"]["endpoint_reported_models"] == ["different-model"]


def test_benchmark_cli_fails_when_any_successful_response_lacks_model_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Client:
        observed_response_models = ("benchmark-model",)
        observed_response_models_overflowed = False
        response_model_mismatch_observed = False
        successful_response_count = 3
        attributed_response_count = 2

    class Resolution:
        client = Client()
        config = PlannerConfig(
            kind="openai-compatible",
            base_url="http://127.0.0.1:1234/v1",
            model="benchmark-model",
        )

    result = BenchmarkSuiteResult(
        suite="suite.json",
        passed=True,
        cases=(),
        passed_cases=0,
        total_cases=0,
        duration_seconds=0.0,
    )
    monkeypatch.setattr("inverse_agent.cli.resolve_planner", lambda **_kwargs: Resolution())
    monkeypatch.setattr(
        "inverse_agent.cli.run_benchmark_suite",
        lambda *_args, **_kwargs: result,
    )
    args = type(
        "Args",
        (),
        {
            "output": None,
            "repository_root": None,
            "suite": str(tmp_path / "suite.json"),
        },
    )()

    assert benchmark_review_command(args) == 1
    provenance = json.loads(capsys.readouterr().out)["model_provenance"]
    assert provenance["successful_responses"] == 3
    assert provenance["attributed_responses"] == 2
    assert provenance["endpoint_model_consistent"] is False


@pytest.mark.parametrize(
    ("verdict", "findings", "expected_exit"),
    [
        ("PASS", (), 0),
        (
            "FINDINGS",
            (
                ReviewFinding(
                    ReviewSeverity.P1,
                    "Authentication bypass",
                    "The changed condition permits unauthenticated access.",
                    "module.py",
                    1,
                    ReviewConfidence.HIGH,
                    "allow = True",
                ),
            ),
            1,
        ),
        ("INCOMPLETE", (), 3),
    ],
)
def test_review_commit_cli_exit_status_reflects_verdict(
    verdict: str,
    findings: tuple[ReviewFinding, ...],
    expected_exit: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = CommitReviewReport(
        commit="a" * 40,
        domain=ReviewDomain.GENERIC,
        verdict=verdict,
        summary="Review result",
        findings=findings,
        changed_files=("module.py",),
        input_truncated=verdict == "INCOMPLETE",
        input_sanitized=False,
        context_truncated=False,
        review_passes=3,
        discarded_model_findings=0,
        static_signals=0,
    )

    class Resolution:
        client = object()

    monkeypatch.setattr("inverse_agent.cli.resolve_planner", lambda **_kwargs: Resolution())
    monkeypatch.setattr("inverse_agent.cli.review_commit", lambda *_args, **_kwargs: report)
    args = type(
        "Args",
        (),
        {
            "workspace": ".",
            "commit": "a" * 40,
            "domain": "generic",
            "goal": "Review",
        },
    )()

    assert review_commit_command(args) == expected_exit
    assert json.loads(capsys.readouterr().out)["verdict"] == verdict


def test_benchmark_cli_rejects_read_only_output_before_model_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "benchmark.json"
    output.write_text("unchanged", encoding="utf-8")
    output.chmod(0o444)
    resolved = False

    def resolve(**_kwargs: object) -> object:
        nonlocal resolved
        resolved = True
        return object()

    monkeypatch.setattr("inverse_agent.cli.resolve_planner", resolve)
    args = type(
        "Args",
        (),
        {
            "output": str(output),
            "repository_root": None,
            "suite": str(tmp_path / "suite.json"),
        },
    )()

    try:
        with pytest.raises(PermissionError, match="not writable"):
            benchmark_review_command(args)
    finally:
        output.chmod(0o666)

    assert resolved is False
    assert output.read_text(encoding="utf-8") == "unchanged"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def _report(
    *findings: ReviewFinding,
    model_supported_findings: int = 0,
    model_findings: tuple[ReviewFinding, ...] = (),
) -> CommitReviewReport:
    return CommitReviewReport(
        commit="a" * 40,
        domain=ReviewDomain.IOS,
        verdict="FINDINGS" if findings else "PASS",
        summary="Review complete",
        findings=findings,
        changed_files=("App/ViewController.swift",),
        input_truncated=False,
        input_sanitized=False,
        context_truncated=False,
        review_passes=3,
        discarded_model_findings=0,
        static_signals=0,
        model_supported_findings=model_supported_findings,
        model_findings=model_findings,
    )


def test_repository_suite_requires_explicit_confined_root() -> None:
    with pytest.raises(BenchmarkDefinitionError, match="explicit repository root"):
        load_benchmark_suite(SUITE)

    cases = load_benchmark_suite(SUITE, repository_root=Path(__file__).parents[1])

    assert len(cases) == 6
    assert {case.domain for case in cases} == {
        ReviewDomain.ANDROID,
        ReviewDomain.IOS,
        ReviewDomain.CPP,
        ReviewDomain.DJANGO,
        ReviewDomain.PYTORCH,
        ReviewDomain.GENERIC,
    }
    assert all(
        expected.max_match_findings == 1 for case in cases for expected in case.expected_findings
    )
    assert all(expected.anchors for case in cases for expected in case.expected_findings)
    assert all(case.min_model_supported_findings == 1 for case in cases[:5])
    assert cases[5].min_model_supported_findings == 0


def test_android_bridge_claim_cannot_stand_in_for_navigation_finding() -> None:
    android = load_benchmark_suite(
        SUITE,
        repository_root=Path(__file__).parents[1],
    )[0]
    findings = (
        ReviewFinding(
            ReviewSeverity.P2,
            "Exported deep-link activity",
            "Any external app can launch this activity component.",
            "app/src/main/AndroidManifest.xml",
            5,
            ReviewConfidence.HIGH,
            'android:exported="true"',
        ),
        ReviewFinding(
            ReviewSeverity.P1,
            "JavaScript bridge exposure",
            "The WebView JavaScript interface exposes data to external web content.",
            "app/src/main/java/com/inverse/DeepLinkActivity.kt",
            15,
            ReviewConfidence.HIGH,
            'webView.addJavascriptInterface(AccountBridge(), "Account")',
        ),
    )

    passed, missing, unmatched = score_review(
        _report(*findings),
        android.expected_findings,
        max_unmatched_findings=android.max_unmatched_findings,
        max_alternative_findings=android.max_alternative_findings,
    )

    assert passed is False
    assert missing == ("untrusted-webview-navigation",)
    assert unmatched == ()


def test_repository_case_cannot_escape_operator_root(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    (fixture / "base").mkdir(parents=True)
    (fixture / "after").mkdir()
    payload = {
        "version": 2,
        "cases": [
            {
                "id": "fixture",
                "domain": "generic",
                "fixture": "fixture",
                "goal": "Review",
                "expected_findings": [],
            }
        ],
        "repository_cases": [
            {
                "id": "escape",
                "domain": "generic",
                "workspace": "..",
                "commit": "a" * 40,
                "goal": "Review",
                "expected_findings": [],
            }
        ],
    }
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps(payload), encoding="utf-8")
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    with pytest.raises(BenchmarkDefinitionError, match="escapes"):
        load_benchmark_suite(suite, repository_root=allowed)


@pytest.mark.parametrize("value", [0, 9, True])
def test_expected_finding_rejects_invalid_match_width(tmp_path: Path, value: object) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "bounded",
                        "domain": "generic",
                        "fixture": "bounded",
                        "goal": "Review",
                        "expected_findings": [
                            {
                                "id": "finding",
                                "paths": ["module.py"],
                                "severities": ["P2"],
                                "keyword_groups": [["behavior"]],
                                "anchors": [
                                    {
                                        "path": "module.py",
                                        "change": "added",
                                        "lines": [1],
                                        "evidence": ["behavior"],
                                    }
                                ],
                                "max_match_findings": value,
                            }
                        ],
                    }
                ],
                "repository_cases": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkDefinitionError, match="max_match_findings"):
        load_benchmark_suite(suite)


def test_suite_rejects_misspelled_expected_findings_field(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "typo",
                        "domain": "generic",
                        "fixture": "typo",
                        "goal": "Review",
                        "expected_finding": [],
                    }
                ],
                "repository_cases": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkDefinitionError, match="unsupported field.*expected_finding"):
        load_benchmark_suite(suite)


def test_suite_rejects_unknown_expected_finding_field(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "unknown",
                        "domain": "generic",
                        "fixture": "unknown",
                        "goal": "Review",
                        "expected_findings": [
                            {
                                "id": "finding",
                                "paths": ["module.py"],
                                "severities": ["P2"],
                                "keyword_groups": [["behavior"]],
                                "keyword_group": ["typo"],
                            }
                        ],
                    }
                ],
                "repository_cases": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkDefinitionError, match="unsupported field.*keyword_group"):
        load_benchmark_suite(suite)


def test_score_normalizes_unicode_punctuation() -> None:
    finding = ReviewFinding(
        severity=ReviewSeverity.P1,
        title="UIKit main‑thread violation",
        body="statusLabel is updated by a URLSession callback.",
        file="App/ViewController.swift",
        line=10,
        confidence=ReviewConfidence.HIGH,
    )
    expected = ExpectedFinding(
        finding_id="main-thread",
        paths=("App/ViewController.swift",),
        severities=(ReviewSeverity.P1,),
        keyword_groups=(("main thread",), ("statuslabel",)),
    )

    passed, missing, unmatched = score_review(
        _report(finding),
        (expected,),
        max_unmatched_findings=0,
    )

    assert passed is True
    assert missing == ()
    assert unmatched == ()


def test_score_requires_expected_line_side_and_evidence_anchor() -> None:
    expected = ExpectedFinding(
        "removed-guard",
        ("module.py",),
        (ReviewSeverity.P1,),
        (("guard",),),
        anchors=(
            ExpectedAnchor(
                path="module.py",
                change="removed",
                lines=(4,),
                evidence=("dangerous_guard",),
            ),
        ),
        max_match_findings=1,
    )
    invalid = (
        ReviewFinding(
            ReviewSeverity.P1,
            "Guard removed",
            "The guard was removed.",
            "module.py",
            5,
            ReviewConfidence.HIGH,
            "dangerous_guard()",
            "removed",
        ),
        ReviewFinding(
            ReviewSeverity.P1,
            "Guard removed",
            "The guard was removed.",
            "module.py",
            4,
            ReviewConfidence.HIGH,
            "dangerous_guard()",
            "added",
        ),
        ReviewFinding(
            ReviewSeverity.P1,
            "Guard removed",
            "The guard was removed.",
            "module.py",
            4,
            ReviewConfidence.HIGH,
            "unrelated_line()",
            "removed",
        ),
    )

    for finding in invalid:
        passed, missing, _unmatched = score_review(
            _report(finding),
            (expected,),
            max_unmatched_findings=0,
        )
        assert passed is False
        assert missing == ("removed-guard",)

    valid = replace(invalid[0], line=4)
    passed, missing, unmatched = score_review(
        _report(valid),
        (expected,),
        max_unmatched_findings=0,
    )
    assert passed is True
    assert missing == ()
    assert unmatched == ()


def test_score_rejects_one_omnibus_finding_for_independent_expectations() -> None:
    finding = ReviewFinding(
        ReviewSeverity.P1,
        "Evaluation mode and gradient regression",
        "Evaluation uses training mode and tracks gradients during inference.",
        "experiment.py",
        15,
        ReviewConfidence.HIGH,
        "model.train()",
        "added",
    )
    anchor = ExpectedAnchor("experiment.py", "added", (15,), ("model.train",))
    mode = ExpectedFinding(
        "mode",
        ("experiment.py",),
        (ReviewSeverity.P1,),
        (("training mode",), ("evaluation",)),
        anchors=(anchor,),
        max_match_findings=1,
    )
    gradients = ExpectedFinding(
        "gradients",
        ("experiment.py",),
        (ReviewSeverity.P1,),
        (("gradient",), ("inference",)),
        anchors=(anchor,),
        max_match_findings=1,
    )

    passed, missing, unmatched = score_review(
        _report(finding),
        (mode, gradients),
        max_unmatched_findings=0,
    )

    assert passed is False
    assert missing == ("distinct-finding-assignment",)
    assert unmatched == ()


def test_score_rejects_any_unmatched_hallucination() -> None:
    finding = ReviewFinding(
        severity=ReviewSeverity.P1,
        title="Unsupported security claim",
        body="This claim has no expected evidence.",
        file="App/ViewController.swift",
        line=1,
        confidence=ReviewConfidence.HIGH,
    )

    passed, _, unmatched = score_review(
        _report(finding),
        (),
        max_unmatched_findings=0,
        max_alternative_findings=16,
    )

    assert passed is False
    assert unmatched and "Unsupported security claim" in unmatched[0]


def test_score_requires_adjudicated_model_origin_when_configured() -> None:
    finding = ReviewFinding(
        ReviewSeverity.P1,
        "SQL injection",
        "Request interpolation creates a SQL injection vulnerability.",
        "views.py",
        10,
        ReviewConfidence.HIGH,
    )
    expected = ExpectedFinding(
        finding_id="sql-injection",
        paths=("views.py",),
        severities=(ReviewSeverity.P1,),
        keyword_groups=(("sql injection",),),
        max_match_findings=1,
    )

    static_only, missing, _unmatched = score_review(
        _report(finding),
        (expected,),
        max_unmatched_findings=0,
        min_model_supported_findings=1,
    )
    model_backed, model_missing, _model_unmatched = score_review(
        _report(
            finding,
            model_supported_findings=1,
            model_findings=(finding,),
        ),
        (expected,),
        max_unmatched_findings=0,
        min_model_supported_findings=1,
    )

    assert static_only is False
    assert missing == (
        "model-supported-findings:0/1",
        "model-supported:sql-injection",
    )
    assert model_backed is True
    assert model_missing == ()


def test_score_requires_model_support_for_every_expected_defect() -> None:
    sql = ReviewFinding(
        ReviewSeverity.P1,
        "SQL injection",
        "Request interpolation creates SQL injection.",
        "views.py",
        10,
        ReviewConfidence.HIGH,
    )
    xss = ReviewFinding(
        ReviewSeverity.P1,
        "DOM XSS",
        "Dynamic data reaches innerHTML and enables XSS.",
        "search.js",
        2,
        ReviewConfidence.HIGH,
    )
    expected = (
        ExpectedFinding(
            "sql-injection",
            ("views.py",),
            (ReviewSeverity.P1,),
            (("sql injection",),),
            1,
        ),
        ExpectedFinding(
            "dom-xss",
            ("search.js",),
            (ReviewSeverity.P1,),
            (("xss",), ("innerhtml",)),
            1,
        ),
    )
    report = _report(
        sql,
        xss,
        model_supported_findings=1,
        model_findings=(sql,),
    )

    passed, missing, unmatched = score_review(
        report,
        expected,
        max_unmatched_findings=0,
        min_model_supported_findings=1,
    )

    assert passed is False
    assert missing == ("model-supported:dom-xss",)
    assert unmatched == ()


def test_score_rejects_negated_keyword_shaped_non_finding() -> None:
    finding = ReviewFinding(
        severity=ReviewSeverity.P1,
        title="No SQL injection",
        body="The f-string is parameterized and safe from SQL injection.",
        file="projects/views.py",
        line=7,
        confidence=ReviewConfidence.HIGH,
    )
    expected = ExpectedFinding(
        finding_id="sql-injection",
        paths=("projects/views.py",),
        severities=(ReviewSeverity.P1,),
        keyword_groups=(("sql injection",), ("f-string", "parameter")),
    )

    passed, missing, _ = score_review(
        CommitReviewReport(
            commit="a" * 40,
            domain=ReviewDomain.DJANGO,
            verdict="FINDINGS",
            summary="Review",
            findings=(finding,),
            changed_files=("projects/views.py",),
            input_truncated=False,
            input_sanitized=False,
            context_truncated=False,
            review_passes=3,
            discarded_model_findings=0,
            static_signals=0,
        ),
        (expected,),
        max_unmatched_findings=0,
    )

    assert passed is False
    assert missing == ("sql-injection",)


def test_score_rejects_postfix_negation() -> None:
    finding = ReviewFinding(
        severity=ReviewSeverity.P1,
        title="SQL injection is not possible",
        body="The raw query uses an f-string but remains safe.",
        file="projects/views.py",
        line=7,
        confidence=ReviewConfidence.HIGH,
    )
    expected = ExpectedFinding(
        "sql-injection",
        ("projects/views.py",),
        (ReviewSeverity.P1,),
        (("sql injection",), ("f-string",)),
    )
    report = CommitReviewReport(
        commit="a" * 40,
        domain=ReviewDomain.DJANGO,
        verdict="FINDINGS",
        summary="Review",
        findings=(finding,),
        changed_files=("projects/views.py",),
        input_truncated=False,
        input_sanitized=False,
        context_truncated=False,
        review_passes=3,
        discarded_model_findings=0,
        static_signals=0,
    )

    passed, missing, _ = score_review(report, (expected,), max_unmatched_findings=0)

    assert passed is False
    assert missing == ("sql-injection",)


def test_score_rejects_cannot_and_impossible_denials() -> None:
    finding = ReviewFinding(
        severity=ReviewSeverity.P1,
        title="SQL injection cannot occur",
        body="The f-string is impossible to exploit.",
        file="projects/views.py",
        line=7,
        confidence=ReviewConfidence.HIGH,
    )
    expected = ExpectedFinding(
        "sql-injection",
        ("projects/views.py",),
        (ReviewSeverity.P1,),
        (("sql injection",), ("f-string",)),
    )

    passed, missing, _unmatched = score_review(
        _report(finding),
        (expected,),
        max_unmatched_findings=0,
    )

    assert passed is False
    assert missing == ("sql-injection",)


def test_score_rejects_safety_inflections_but_accepts_failed_prevention() -> None:
    expected = ExpectedFinding(
        "sql-injection",
        ("projects/views.py",),
        (ReviewSeverity.P1,),
        (("sql injection",), ("f-string",)),
    )
    safe_claim = ReviewFinding(
        ReviewSeverity.P1,
        "The f-string prevents SQL injection",
        "This parameterization safely protects the query.",
        "projects/views.py",
        7,
        ReviewConfidence.HIGH,
    )
    unsafe_claim = ReviewFinding(
        ReviewSeverity.P1,
        "The f-string cannot prevent SQL injection",
        "User input remains exploitable.",
        "projects/views.py",
        7,
        ReviewConfidence.HIGH,
    )

    safe_passed, safe_missing, _unmatched = score_review(
        _report(safe_claim),
        (expected,),
        max_unmatched_findings=0,
    )
    unsafe_passed, unsafe_missing, unsafe_unmatched = score_review(
        _report(unsafe_claim),
        (expected,),
        max_unmatched_findings=0,
    )

    assert safe_passed is False
    assert safe_missing == ("sql-injection",)
    assert unsafe_passed is True
    assert unsafe_missing == ()
    assert unsafe_unmatched == ()


@pytest.mark.parametrize(
    ("title", "keyword"),
    [
        ("User input is not sanitized", "sanitized"),
        ("The renderer fails to escape HTML", "escape"),
        ("The query executes without parameterization", "parameterization"),
    ],
)
def test_score_accepts_missing_safety_control_language(title: str, keyword: str) -> None:
    finding = ReviewFinding(
        ReviewSeverity.P1,
        title,
        "The missing control leaves an exploitable input path.",
        "App/ViewController.swift",
        10,
        ReviewConfidence.HIGH,
    )
    expected = ExpectedFinding(
        "missing-control",
        ("App/ViewController.swift",),
        (ReviewSeverity.P1,),
        ((keyword,),),
    )

    passed, missing, unmatched = score_review(
        _report(finding),
        (expected,),
        max_unmatched_findings=0,
    )

    assert passed is True
    assert missing == ()
    assert unmatched == ()


def test_score_rejects_positive_safety_control_language() -> None:
    finding = ReviewFinding(
        ReviewSeverity.P1,
        "User input is sanitized",
        "The safety control is present and effective.",
        "App/ViewController.swift",
        10,
        ReviewConfidence.HIGH,
    )
    expected = ExpectedFinding(
        "missing-sanitization",
        ("App/ViewController.swift",),
        (ReviewSeverity.P1,),
        (("sanitized",),),
    )

    passed, missing, _unmatched = score_review(
        _report(finding),
        (expected,),
        max_unmatched_findings=0,
    )

    assert passed is False
    assert missing == ("missing-sanitization",)


@pytest.mark.parametrize(
    "section",
    [
        "Recommendation: Remove the JavaScript bridge.",
        "**Remediation:** Remove the JavaScript bridge.",
        "Suggested fix:\nRemove the JavaScript bridge.",
    ],
)
def test_score_does_not_count_remediation_text_as_a_defect_claim(section: str) -> None:
    finding = ReviewFinding(
        ReviewSeverity.P1,
        "Exported activity accepts external navigation",
        f"An untrusted intent can load an arbitrary WebView URL.\n\n{section}",
        "Activity.kt",
        8,
        ReviewConfidence.HIGH,
    )
    expected = ExpectedFinding(
        "bridge-exposure",
        ("Activity.kt",),
        (ReviewSeverity.P1,),
        (("javascript bridge", "bridge"), ("untrusted", "arbitrary")),
    )

    passed, missing, unmatched = score_review(
        _report(finding),
        (expected,),
        max_unmatched_findings=0,
    )

    assert passed is False
    assert missing == ("bridge-exposure",)
    assert len(unmatched) == 1


def test_score_can_combine_separate_findings_for_one_expected_defect() -> None:
    findings = (
        ReviewFinding(
            ReviewSeverity.P1,
            "Exported activity trusts an external intent",
            "Any external caller can supply the untrusted intent URL.",
            "Activity.kt",
            4,
            ReviewConfidence.HIGH,
        ),
        ReviewFinding(
            ReviewSeverity.P1,
            "JavaScript WebView bridge exposed",
            "The WebView enables JavaScript and loadUrl with a privileged bridge.",
            "Activity.kt",
            8,
            ReviewConfidence.HIGH,
        ),
    )
    expected = ExpectedFinding(
        "webview-boundary",
        ("Activity.kt",),
        (ReviewSeverity.P1,),
        (("webview", "loadurl"), ("intent", "external")),
    )
    report = CommitReviewReport(
        commit="a" * 40,
        domain=ReviewDomain.ANDROID,
        verdict="FINDINGS",
        summary="Review",
        findings=findings,
        changed_files=("Activity.kt",),
        input_truncated=False,
        input_sanitized=False,
        context_truncated=False,
        review_passes=3,
        discarded_model_findings=0,
        static_signals=0,
    )

    passed, missing, unmatched = score_review(report, (expected,), max_unmatched_findings=0)

    assert passed is True
    assert missing == ()
    assert unmatched == ()


def test_score_can_require_one_coherent_finding_per_expected_defect() -> None:
    findings = (
        ReviewFinding(
            ReviewSeverity.P1,
            "External intent accepted",
            "An external caller controls the intent.",
            "Activity.kt",
            4,
            ReviewConfidence.HIGH,
        ),
        ReviewFinding(
            ReviewSeverity.P1,
            "WebView URL loaded",
            "The WebView calls loadUrl.",
            "Activity.kt",
            8,
            ReviewConfidence.HIGH,
        ),
    )
    expected = ExpectedFinding(
        "webview-boundary",
        ("Activity.kt",),
        (ReviewSeverity.P1,),
        (("webview", "loadurl"), ("external", "untrusted")),
        max_match_findings=1,
    )

    passed, missing, unmatched = score_review(
        _report(*findings),
        (expected,),
        max_unmatched_findings=0,
    )

    assert passed is False
    assert missing == ("webview-boundary",)
    assert len(unmatched) == 2


def test_score_uses_one_consistent_cover_and_leaves_alternatives_unmatched() -> None:
    findings = (
        ReviewFinding(
            ReviewSeverity.P1,
            "External intent accepted",
            "An external caller controls the intent.",
            "Activity.kt",
            4,
            ReviewConfidence.HIGH,
        ),
        ReviewFinding(
            ReviewSeverity.P1,
            "WebView URL loaded",
            "The WebView calls loadUrl.",
            "Activity.kt",
            8,
            ReviewConfidence.HIGH,
        ),
        ReviewFinding(
            ReviewSeverity.P1,
            "Duplicate caller claim",
            "Another external intent claim.",
            "Activity.kt",
            5,
            ReviewConfidence.MEDIUM,
        ),
    )
    expected = ExpectedFinding(
        "webview-boundary",
        ("Activity.kt",),
        (ReviewSeverity.P1,),
        (("webview", "loadurl"), ("intent", "external")),
    )

    passed, missing, unmatched = score_review(
        _report(*findings),
        (expected,),
        max_unmatched_findings=0,
    )

    assert passed is False
    assert missing == ()
    assert len(unmatched) == 1
    assert "Duplicate caller claim" in unmatched[0]

    allowed, _missing, allowed_unmatched = score_review(
        _report(*findings),
        (expected,),
        max_unmatched_findings=0,
        max_alternative_findings=1,
    )
    assert allowed is True
    assert allowed_unmatched == unmatched


def test_score_reuses_broad_finding_but_prefers_novel_specific_cover() -> None:
    broad = ReviewFinding(
        ReviewSeverity.P1,
        "External WebView bridge",
        "An external intent loads an arbitrary WebView URL through a JavaScript bridge.",
        "Activity.kt",
        8,
        ReviewConfidence.HIGH,
    )
    bridge = ReviewFinding(
        ReviewSeverity.P2,
        "JavaScript bridge exposure",
        "The bridge is callable by untrusted loaded pages.",
        "Activity.kt",
        12,
        ReviewConfidence.HIGH,
    )
    boundary = ExpectedFinding(
        "external-url",
        ("Activity.kt",),
        (ReviewSeverity.P1,),
        (("external",), ("webview",)),
    )
    exposure = ExpectedFinding(
        "bridge",
        ("Activity.kt",),
        (ReviewSeverity.P1, ReviewSeverity.P2),
        (("bridge",), ("untrusted", "arbitrary")),
    )

    passed, missing, unmatched = score_review(
        _report(broad, bridge),
        (boundary, exposure),
        max_unmatched_findings=0,
    )

    assert passed is True
    assert missing == ()
    assert unmatched == ()


def test_score_supports_five_nonredundant_split_findings() -> None:
    findings = tuple(
        ReviewFinding(
            ReviewSeverity.P2,
            f"Facet keyword{index}",
            f"Evidence for keyword{index}.",
            "module.py",
            index + 1,
            ReviewConfidence.HIGH,
        )
        for index in range(5)
    )
    expected = ExpectedFinding(
        "five-facets",
        ("module.py",),
        (ReviewSeverity.P2,),
        tuple((f"keyword{index}",) for index in range(5)),
    )

    passed, missing, unmatched = score_review(
        _report(*findings),
        (expected,),
        max_unmatched_findings=0,
    )

    assert passed is True
    assert missing == ()
    assert unmatched == ()


def test_score_can_allow_low_severity_control_observations_only() -> None:
    p2 = ReviewFinding(
        ReviewSeverity.P2,
        "Bounded residual",
        "A low-impact issue remains.",
        "module.py",
        1,
        ReviewConfidence.MEDIUM,
    )
    p1 = ReviewFinding(
        ReviewSeverity.P1,
        "Serious regression",
        "A serious correctness defect remains.",
        "module.py",
        1,
        ReviewConfidence.HIGH,
    )

    allowed, _missing, unmatched = score_review(
        _report(p2),
        (),
        max_unmatched_findings=0,
        allowed_unmatched_severities=(ReviewSeverity.P2, ReviewSeverity.P3),
    )
    blocked, _missing, _unmatched = score_review(
        _report(p1),
        (),
        max_unmatched_findings=0,
        allowed_unmatched_severities=(ReviewSeverity.P2, ReviewSeverity.P3),
    )

    assert allowed is True
    assert unmatched and "Bounded residual" in unmatched[0]
    assert blocked is False


def test_run_suite_materializes_fixture_and_scores_review(tmp_path: Path) -> None:
    fixture = tmp_path / "sample"
    (fixture / "base").mkdir(parents=True)
    (fixture / "after").mkdir()
    (fixture / "base" / "module.py").write_text("value = 1\n", encoding="utf-8")
    (fixture / "after" / "module.py").write_text("value = 0\n", encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "sample",
                        "domain": "generic",
                        "fixture": "sample",
                        "goal": "Review the commit",
                        "expected_findings": [
                            {
                                "id": "zero-value",
                                "paths": ["module.py"],
                                "severities": ["P2"],
                                "keyword_groups": [["zero"], ["behavior"]],
                                "anchors": [
                                    {
                                        "path": "module.py",
                                        "change": "added",
                                        "lines": [1],
                                        "evidence": ["value = 0"],
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "repository_cases": [],
            }
        ),
        encoding="utf-8",
    )
    finding = {
        "severity": "P2",
        "title": "Zero changes behavior",
        "body": "The new zero value changes the supported behavior.",
        "file": "F001",
        "evidence": "value = 0",
        "change": "added",
        "confidence": "high",
    }
    client = FakeReviewClient(
        [
            {"summary": "Supported", "findings": [finding]},
            {"summary": "Supported", "findings": [finding]},
            {
                "summary": "Supported",
                "decisions": [{"candidate": "K001", "accepted": True, "severity": "P2"}],
            },
        ]
    )

    result = run_benchmark_suite(suite, client=client)

    assert result.passed is True
    assert result.passed_cases == 1
    assert result.cases[0].commit and len(result.cases[0].commit) == 40
    assert client.calls == 3


def test_fixture_commit_identity_is_reproducible(tmp_path: Path) -> None:
    fixture = tmp_path / "stable"
    (fixture / "base").mkdir(parents=True)
    (fixture / "after").mkdir()
    (fixture / "base" / "module.py").write_bytes(b"value = 1\n")
    (fixture / "after" / "module.py").write_bytes(b"value = 2\n")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "stable",
                        "domain": "generic",
                        "fixture": "stable",
                        "goal": "Review",
                        "expected_findings": [],
                    }
                ],
                "repository_cases": [],
            }
        ),
        encoding="utf-8",
    )
    client = FakeReviewClient({"summary": "No defect", "findings": []})

    first = run_benchmark_suite(suite, client=client)
    second = run_benchmark_suite(suite, client=client)

    assert first.cases[0].commit == second.cases[0].commit
    assert first.cases[0].commit == "2744ebf2bc0a3a99e036b6cb46c428f14f996c63"


def test_consistent_assignment_search_does_not_prune_valid_complete_cover() -> None:
    broad = tuple(frozenset({index}) for index in range(20))
    options = [broad] * 5 + [(frozenset({index}),) for index in range(11)]

    matched, complete = _select_consistent_assignment(options)

    assert complete is True
    assert set(range(11)).issubset(matched)
    assert len(matched) == 16


@pytest.mark.parametrize("git_name", [".git", ".GIT"])
def test_run_suite_refuses_fixture_git_state(tmp_path: Path, git_name: str) -> None:
    fixture = tmp_path / "unsafe"
    (fixture / "base" / git_name).mkdir(parents=True)
    (fixture / "base" / git_name / "config").write_text("[core]\n", encoding="utf-8")
    (fixture / "after").mkdir()
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "unsafe",
                        "domain": "generic",
                        "fixture": "unsafe",
                        "goal": "Review",
                        "expected_findings": [],
                    }
                ],
                "repository_cases": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_suite(
        suite,
        client=FakeReviewClient({"verdict": "PASS", "summary": "", "findings": []}),
    )

    assert result.passed is False
    assert result.cases[0].error and "Git state" in result.cases[0].error


def test_run_suite_refuses_linked_fixture_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    (outside / "base").mkdir(parents=True)
    (outside / "after").mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "linked",
                        "domain": "generic",
                        "fixture": "linked",
                        "goal": "Review",
                        "expected_findings": [],
                    }
                ],
                "repository_cases": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_suite(
        suite,
        client=FakeReviewClient({"summary": "", "findings": []}),
    )

    assert result.passed is False
    assert result.cases[0].error and "links or junctions" in result.cases[0].error


def test_fixture_after_tree_can_delete_baseline_files(tmp_path: Path) -> None:
    fixture = tmp_path / "deletion"
    (fixture / "base").mkdir(parents=True)
    (fixture / "after").mkdir()
    (fixture / "base" / "removed.py").write_text("obsolete = True\n", encoding="utf-8")
    (fixture / "after" / "kept.py").write_text("active = True\n", encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": "deletion",
                        "domain": "generic",
                        "fixture": "deletion",
                        "goal": "Review",
                        "expected_findings": [],
                    }
                ],
                "repository_cases": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_suite(
        suite,
        client=FakeReviewClient({"summary": "No defect", "findings": []}),
    )

    assert result.passed is True
    report = result.cases[0].report
    assert report is not None
    assert set(report.changed_files) == {"kept.py", "removed.py"}


@pytest.mark.parametrize("mode", ["add-only", "no-op"])
def test_fixture_materialization_supports_empty_and_unchanged_trees(
    tmp_path: Path,
    mode: str,
) -> None:
    fixture = tmp_path / mode
    (fixture / "base").mkdir(parents=True)
    (fixture / "after").mkdir()
    if mode == "add-only":
        (fixture / "after" / "module.py").write_text("value = 1\n", encoding="utf-8")
    else:
        (fixture / "base" / "module.py").write_text("value = 1\n", encoding="utf-8")
        (fixture / "after" / "module.py").write_text("value = 1\n", encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "version": 2,
                "cases": [
                    {
                        "id": mode,
                        "domain": "generic",
                        "fixture": mode,
                        "goal": "Review",
                        "expected_findings": [],
                    }
                ],
                "repository_cases": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_suite(
        suite,
        client=FakeReviewClient({"summary": "No defect", "findings": []}),
    )

    assert result.passed is True
    report = result.cases[0].report
    assert report is not None
    assert report.changed_files == (("module.py",) if mode == "add-only" else ())

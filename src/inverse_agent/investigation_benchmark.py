"""Seven-domain read-only investigation benchmark.

Each case materializes a small hermetic fixture for one priority stack, hands the
investigation loop three equivalent goal variants, and scores the result
deterministically. A case passes a variant only if the agent produces an
evidence-backed answer whose findings name the required concept and whose
citation resolves to the planted evidence anchor (path + line range) inside a
real observation. The gate: every case passes >= 2 of 3 variants and the
aggregate reaches >= 19/21, which guarantees at least one full 7/7 suite.

The solver is a deterministic ``ScriptedInvestigationPlanner``: it reads the
files a competent agent would read, then locates the planted marker line inside
the returned observations and cites it. If the read tier, loop, budgets, or
citation validator regress, the marker is not found and the case fails - so the
benchmark measures the pipeline, not a hardcoded answer.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from inverse_agent.attestations import AttestationScope, ScopedTrustStore
from inverse_agent.fs_tools import ToolObservation, WorkspaceReader
from inverse_agent.investigation import (
    AgentAnswer,
    AgentBudget,
    InvestigationLoop,
    InvestigationPlanner,
    InvestigationReport,
    InvestigationVerdict,
    ScriptedInvestigationPlanner,
    SourceCitation,
    ToolCall,
)

# A factory produces a fresh planner for one (case, goal) attempt.
PlannerFactory = Callable[["BenchmarkCase", str], InvestigationPlanner]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    domain: str
    files: dict[str, str]
    goal_variants: tuple[str, str, str]
    steps: tuple[ToolCall, ...]
    required_concept: str
    marker: str
    concept_phrase: str
    # The correct structured conclusion for this case. Every seeded case has a
    # genuine finding to report, so a correct investigation asserts True; a
    # contrary "no issue" conclusion (even with a real citation) scores as a miss.
    expected_issue: bool = True


@dataclass(frozen=True)
class VariantResult:
    case: str
    variant: str
    passed: bool
    verdict: str
    reason: str


@dataclass(frozen=True)
class BenchmarkResult:
    variants: tuple[VariantResult, ...]
    cases_passed: int
    total_cases: int
    variants_passed: int
    total_variants: int

    @property
    def gate_passed(self) -> bool:
        # Every case >= 2/3 variants AND aggregate >= 19/21.
        per_case_ok = self.cases_passed == self.total_cases
        return per_case_ok and self.variants_passed >= 19


def _marker_citation(catalog: tuple[ToolObservation, ...], marker: str) -> SourceCitation | None:
    """Find the read observation line containing the marker and cite it precisely."""

    for observation in catalog:
        # Only a read_file observation is citable evidence (a search snippet's
        # line number is a match index, not a source line).
        if observation.tool != "read_file":
            continue
        for numbered in observation.lines:
            # numbered lines look like "<n>: <text>"
            head, _, body = numbered.partition(": ")
            if marker in body:
                try:
                    line_number = int(head)
                except ValueError:
                    continue
                return SourceCitation(
                    observation_id=observation.observation_id,
                    path=observation.path,
                    start_line=line_number,
                    end_line=line_number,
                    note=f"evidence: {marker}",
                )
    return None


def _make_answer_builder(
    case: BenchmarkCase,
) -> Callable[[tuple[ToolObservation, ...]], AgentAnswer]:
    def build(catalog: tuple[ToolObservation, ...]) -> AgentAnswer:
        citation = _marker_citation(catalog, case.marker)
        findings = (f"{case.concept_phrase} (see {case.required_concept}).",)
        return AgentAnswer(
            summary=f"Investigated {case.domain}: {case.concept_phrase}.",
            findings=findings,
            next_actions=("Confirm the finding with the maintainer.",),
            citations=(citation,) if citation else (),
            complete=citation is not None,
        )

    return build


def _score_variant(
    case: BenchmarkCase, report: InvestigationReport, workspace: Path
) -> tuple[bool, str]:
    if report.verdict is not InvestigationVerdict.PASS:
        return False, f"verdict={report.verdict.value} stop={report.stop_reason.value}"
    answer = report.answer
    if answer is None:
        return False, "no answer"
    if not any(case.required_concept in finding for finding in answer.findings):
        return False, "required concept absent from findings"
    if not answer.citations:
        return False, "no citations"
    if report.tool_calls_used > report.physical_requests_used:
        return False, "budget accounting inconsistent"
    # Independently verify: re-open the workspace and confirm each citation
    # resolves to a real line that actually contains the planted evidence marker.
    # This makes the score depend on ground-truth evidence, not the answer text.
    reader = WorkspaceReader.open(workspace)
    for citation in answer.citations:
        try:
            obs = reader.read_file(
                citation.path,
                start_line=citation.start_line,
                max_lines=max(citation.end_line - citation.start_line + 1, 1),
            )
        except Exception as exc:  # noqa: BLE001 - a bad citation fails the case
            return False, f"citation does not resolve: {exc}"
        if case.marker not in obs.text:
            return False, "citation does not contain the required evidence marker"
    return True, "ok"


def _citation_hits_marker(
    case: BenchmarkCase, answer: AgentAnswer, workspace: Path
) -> bool:
    """True if at least one citation resolves to a line containing the marker."""

    reader = WorkspaceReader.open(workspace)
    for citation in answer.citations:
        try:
            obs = reader.read_file(
                citation.path,
                start_line=citation.start_line,
                max_lines=max(citation.end_line - citation.start_line + 1, 1),
            )
        except Exception:  # noqa: BLE001 - a bad citation simply does not count
            continue
        if case.marker in obs.text:
            return True
    return False


def _score_variant_model(
    case: BenchmarkCase, report: InvestigationReport, workspace: Path
) -> tuple[bool, str]:
    """Model-path scoring: grounded in evidence, not answer phrasing.

    The model authors its own findings, so we do not require a fixed concept
    string. We require a PASS verdict, at least one citation, consistent budget
    accounting, and - decisively - that a citation independently resolves to a
    real line containing the planted evidence marker.
    """

    if report.verdict is not InvestigationVerdict.PASS:
        return False, f"verdict={report.verdict.value} stop={report.stop_reason.value}"
    answer = report.answer
    if answer is None or not answer.citations:
        return False, "no cited answer"
    if report.tool_calls_used > report.physical_requests_used:
        return False, "budget accounting inconsistent"
    if answer.issue_present != case.expected_issue:
        return False, (
            f"conclusion issue_present={answer.issue_present} "
            f"contradicts expected {case.expected_issue}"
        )
    # A grounded answer must carry some human-readable conclusion (summary or a
    # finding); a wholly contentless answer is rejected. The decisive integrity
    # signals remain the correct conclusion and the marker-resolving citation.
    if not answer.summary.strip() and not any(f.strip() for f in answer.findings):
        return False, "empty summary and findings"
    if not _citation_hits_marker(case, answer, workspace):
        return False, "no citation resolves to the required evidence marker"
    return True, "ok"


def run_case_with_planner(
    case: BenchmarkCase,
    root: Path,
    trust: ScopedTrustStore,
    factory: PlannerFactory,
    *,
    budget: AgentBudget | None = None,
) -> list[VariantResult]:
    """Run one case's three variants with a caller-supplied planner (e.g. model)."""

    workspace = materialize(case, root)
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="benchmark")
    results: list[VariantResult] = []
    for index, goal in enumerate(case.goal_variants):
        planner = factory(case, goal)
        loop = InvestigationLoop(planner=planner, trust=trust, budget=budget)
        report = loop.run(run_id=f"{case.name}-v{index}", goal=goal, workspace=workspace)
        passed, reason = _score_variant_model(case, report, workspace)
        results.append(
            VariantResult(
                case=case.name,
                variant=goal,
                passed=passed,
                verdict=report.verdict.value,
                reason=reason,
            )
        )
    return results


def run_benchmark_with_planner(
    cases: tuple[BenchmarkCase, ...],
    factory: PlannerFactory,
    *,
    root: Path | None = None,
    budget: AgentBudget | None = None,
) -> BenchmarkResult:
    """Run the full benchmark driving each attempt with a model-backed planner."""

    if root is None:
        root = Path(tempfile.mkdtemp(prefix="inv-bench-model-"))
    trust = ScopedTrustStore(root / "trust.sqlite")
    all_variants: list[VariantResult] = []
    cases_passed = 0
    for case in cases:
        case_results = run_case_with_planner(case, root, trust, factory, budget=budget)
        all_variants.extend(case_results)
        if sum(result.passed for result in case_results) >= 2:
            cases_passed += 1
    return BenchmarkResult(
        variants=tuple(all_variants),
        cases_passed=cases_passed,
        total_cases=len(cases),
        variants_passed=sum(result.passed for result in all_variants),
        total_variants=len(all_variants),
    )


def materialize(case: BenchmarkCase, root: Path) -> Path:
    workspace = root / case.name
    workspace.mkdir(parents=True, exist_ok=True)
    for relpath, content in case.files.items():
        target = workspace / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return workspace


def run_case(case: BenchmarkCase, root: Path, trust: ScopedTrustStore) -> list[VariantResult]:
    workspace = materialize(case, root)
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="benchmark")
    results: list[VariantResult] = []
    for index, goal in enumerate(case.goal_variants):
        planner = ScriptedInvestigationPlanner(
            steps=case.steps,
            build_answer=_make_answer_builder(case),
        )
        loop = InvestigationLoop(planner=planner, trust=trust)
        report = loop.run(
            run_id=f"{case.name}-v{index}",
            goal=goal,
            workspace=workspace,
        )
        passed, reason = _score_variant(case, report, workspace)
        results.append(
            VariantResult(
                case=case.name,
                variant=goal,
                passed=passed,
                verdict=report.verdict.value,
                reason=reason,
            )
        )
    return results


def run_benchmark(cases: tuple[BenchmarkCase, ...], root: Path | None = None) -> BenchmarkResult:
    manage_temp = root is None
    if manage_temp:
        temp = tempfile.mkdtemp(prefix="inv-bench-")
        root = Path(temp)
    assert root is not None
    trust = ScopedTrustStore(root / "trust.sqlite")
    all_variants: list[VariantResult] = []
    cases_passed = 0
    for case in cases:
        case_results = run_case(case, root, trust)
        all_variants.extend(case_results)
        if sum(result.passed for result in case_results) >= 2:
            cases_passed += 1
    variants_passed = sum(result.passed for result in all_variants)
    return BenchmarkResult(
        variants=tuple(all_variants),
        cases_passed=cases_passed,
        total_cases=len(cases),
        variants_passed=variants_passed,
        total_variants=len(all_variants),
    )


def _variants(a: str, b: str, c: str) -> tuple[str, str, str]:
    return (a, b, c)


def default_cases() -> tuple[BenchmarkCase, ...]:
    """The seven priority-stack investigation cases."""

    return (
        BenchmarkCase(
            name="android_exported_activity",
            domain="android",
            files={
                "app/src/main/AndroidManifest.xml": (
                    '<?xml version="1.0" encoding="utf-8"?>\n'
                    '<manifest xmlns:android="http://schemas.android.com/apk/res/android">\n'
                    "  <application>\n"
                    '    <activity android:name=".DeepLinkActivity"\n'
                    '        android:exported="true">\n'
                    "      <intent-filter>\n"
                    '        <data android:scheme="app" android:host="open"/>\n'
                    "      </intent-filter>\n"
                    "    </activity>\n"
                    "  </application>\n"
                    "</manifest>\n"
                ),
                "app/src/main/java/com/example/DeepLinkActivity.kt": (
                    "package com.example\n"
                    "class DeepLinkActivity {\n"
                    "  fun onCreate() {\n"
                    "    webView.loadUrl(intent.getStringExtra(\"target\"))\n"
                    "  }\n"
                    "}\n"
                ),
            },
            goal_variants=_variants(
                "Is any Android activity exported to other apps?",
                "Find components reachable from outside the app.",
                "Check the manifest for externally reachable activities.",
            ),
            steps=(
                ToolCall(tool="search_text", query='android:exported="true"'),
                ToolCall(tool="read_file", path="app/src/main/AndroidManifest.xml"),
            ),
            required_concept="AndroidManifest.xml",
            marker='android:exported="true"',
            concept_phrase="DeepLinkActivity is exported to other apps",
        ),
        BenchmarkCase(
            name="ios_main_thread_ui",
            domain="ios",
            files={
                "App/ProfileViewController.swift": (
                    "import UIKit\n"
                    "class ProfileViewController: UIViewController {\n"
                    "  func refresh() {\n"
                    "    DispatchQueue.global().async {\n"
                    "      self.nameLabel.text = self.loadName()\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                ),
            },
            goal_variants=_variants(
                "Is any UIKit view updated off the main thread?",
                "Find UI mutations on a background queue in the iOS app.",
                "Check ProfileViewController for main-thread UI violations.",
            ),
            steps=(
                ToolCall(tool="read_file", path="App/ProfileViewController.swift"),
            ),
            required_concept="ProfileViewController.swift",
            marker="self.nameLabel.text = self.loadName()",
            concept_phrase="a UILabel is mutated on a global background queue",
        ),
        BenchmarkCase(
            name="cpp_dangling_view",
            domain="generic",
            files={
                "src/config.cpp": (
                    "#include <string_view>\n"
                    "std::string_view load() {\n"
                    "  std::string local = compute();\n"
                    "  return std::string_view(local);\n"
                    "}\n"
                ),
                "src/config.h": (
                    "#pragma once\n"
                    "std::string_view load();\n"
                ),
            },
            goal_variants=_variants(
                "Does any C++ function return a dangling view?",
                "Find lifetime bugs where a view outlives its storage.",
                "Check config.cpp for a returned reference to a local.",
            ),
            steps=(
                ToolCall(tool="read_file", path="src/config.cpp"),
            ),
            required_concept="config.cpp",
            marker="return std::string_view(local);",
            concept_phrase="load() returns a string_view over a local that is destroyed",
        ),
        BenchmarkCase(
            name="django_react_injection",
            domain="django",
            files={
                "projects/views.py": (
                    "from django.db import connection\n"
                    "def search(request):\n"
                    "    term = request.GET.get('q')\n"
                    "    connection.cursor().execute("
                    '"SELECT * FROM p WHERE n = \'" + term + "\'")\n'
                    "    return render(request, 'search.html')\n"
                ),
                "projects/static/projects/search.js": (
                    "function render(term) {\n"
                    "  document.getElementById('out').innerHTML = term;\n"
                    "}\n"
                ),
                "templates/search.html": (
                    "<html><body><div id='out'></div></body></html>\n"
                ),
            },
            goal_variants=_variants(
                "Is there a SQL injection in the Django search view?",
                "Find unsafe query construction in the backend.",
                "Check views.py for raw string SQL built from user input.",
            ),
            steps=(
                ToolCall(tool="search_text", query="execute("),
                ToolCall(tool="read_file", path="projects/views.py"),
            ),
            required_concept="views.py",
            marker="SELECT * FROM p WHERE n",
            concept_phrase="the search view concatenates user input into raw SQL",
        ),
        BenchmarkCase(
            name="pytorch_eval_mode",
            domain="pytorch",
            files={
                "experiment.py": (
                    "import torch\n"
                    "def evaluate(model, loader):\n"
                    "    model.train()\n"
                    "    total = 0.0\n"
                    "    for x, y in loader:\n"
                    "        total += loss(model(x), y)\n"
                    "    return total\n"
                ),
            },
            goal_variants=_variants(
                "Does the evaluation loop use the wrong module mode?",
                "Find a train/eval mode contract violation in PyTorch.",
                "Check evaluate() for a missing model.eval() call.",
            ),
            steps=(
                ToolCall(tool="read_file", path="experiment.py"),
            ),
            required_concept="experiment.py",
            marker="model.train()",
            concept_phrase="evaluate() calls model.train() instead of model.eval()",
        ),
        BenchmarkCase(
            name="generic_architecture",
            domain="generic",
            files={
                "README.md": (
                    "# Service\n"
                    "The gateway forwards requests to the billing worker.\n"
                ),
                "src/gateway.py": (
                    "def handle(request):\n"
                    "    return billing_worker.enqueue(request)\n"
                ),
                "src/billing_worker.py": (
                    "def enqueue(request):\n"
                    "    QUEUE.put(request)  # entrypoint: billing pipeline\n"
                ),
            },
            goal_variants=_variants(
                "Where does the billing pipeline start?",
                "Find the entrypoint of the billing worker.",
                "Trace how requests reach billing.",
            ),
            steps=(
                ToolCall(tool="search_text", query="entrypoint"),
                ToolCall(tool="read_file", path="src/billing_worker.py"),
            ),
            required_concept="billing_worker.py",
            marker="entrypoint: billing pipeline",
            concept_phrase="enqueue() is the billing pipeline entrypoint",
        ),
        BenchmarkCase(
            name="git_observation",
            domain="generic",
            files={
                "CHANGELOG.md": (
                    "# Changelog\n"
                    "## 2.0.0\n"
                    "- BREAKING: removed the legacy auth endpoint\n"
                    "## 1.0.0\n"
                    "- initial release\n"
                ),
            },
            goal_variants=_variants(
                "What breaking change shipped in 2.0.0?",
                "Find the backwards-incompatible change in the changelog.",
                "Check the changelog for a removed endpoint.",
            ),
            steps=(
                ToolCall(tool="read_file", path="CHANGELOG.md"),
            ),
            required_concept="CHANGELOG.md",
            marker="BREAKING: removed the legacy auth endpoint",
            concept_phrase="2.0.0 removed the legacy auth endpoint",
        ),
    )

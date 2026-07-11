"""Hermetic multi-domain acceptance benchmark for commit review."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from unicodedata import category

from inverse_agent.commit_review import (
    MAX_FINDINGS,
    CommitReviewReport,
    ReviewDomain,
    ReviewFinding,
    ReviewSeverity,
    StructuredReviewClient,
    review_commit,
)
from inverse_agent.environments import discover_trusted_git
from inverse_agent.models import RunnerPolicy
from inverse_agent.redaction import redact_text
from inverse_agent.runner import build_safe_subprocess_env

MAX_BENCHMARK_CASES = 32
MAX_EXPECTED_FINDINGS = 16
MAX_FIXTURE_FILES = 128
MAX_FIXTURE_BYTES = 1024 * 1024
MAX_SUITE_BYTES = 1024 * 1024
BENCHMARK_GIT_TIMEOUT_SECONDS = 20
FIXTURE_GIT_DATE = "2000-01-01T00:00:00+00:00"
MAX_MATCH_OPTIONS = 256
MAX_MATCH_FINDINGS_PER_EXPECTED = 8
REMEDIATION_SECTION_HEADINGS = frozenset(
    {"fix", "recommendation", "remediation", "suggested fix", "how to fix"}
)
SUITE_FIELDS = frozenset({"version", "cases", "repository_cases"})
COMMON_CASE_FIELDS = frozenset(
    {
        "id",
        "domain",
        "goal",
        "expected_findings",
        "max_unmatched_findings",
        "allowed_unmatched_severities",
        "max_alternative_findings",
        "min_model_supported_findings",
    }
)
EXPECTED_FINDING_FIELDS = frozenset(
    {"id", "paths", "severities", "keyword_groups", "anchors", "max_match_findings"}
)
EXPECTED_ANCHOR_FIELDS = frozenset({"path", "change", "lines", "evidence"})


class BenchmarkDefinitionError(ValueError):
    """Raised when a benchmark suite is malformed or unsafe."""


@dataclass(frozen=True)
class ExpectedAnchor:
    path: str
    change: str
    lines: tuple[int, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ExpectedFinding:
    finding_id: str
    paths: tuple[str, ...]
    severities: tuple[ReviewSeverity, ...]
    keyword_groups: tuple[tuple[str, ...], ...]
    max_match_findings: int = MAX_MATCH_FINDINGS_PER_EXPECTED
    anchors: tuple[ExpectedAnchor, ...] = ()


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    domain: ReviewDomain
    goal: str
    expected_findings: tuple[ExpectedFinding, ...]
    max_unmatched_findings: int
    allowed_unmatched_severities: tuple[ReviewSeverity, ...] = ()
    max_alternative_findings: int = 0
    min_model_supported_findings: int = 0
    fixture: Path | None = None
    workspace: Path | None = None
    commit: str | None = None


@dataclass(frozen=True)
class BenchmarkCaseResult:
    case_id: str
    passed: bool
    commit: str | None
    report: CommitReviewReport | None
    missing_findings: tuple[str, ...]
    unmatched_findings: tuple[str, ...]
    error: str | None
    duration_seconds: float


@dataclass(frozen=True)
class BenchmarkModelProvenance:
    kind: str
    requested_model: str
    base_url: str
    config_fingerprint: str
    endpoint_reported_models: tuple[str, ...]
    successful_responses: int
    attributed_responses: int
    endpoint_model_consistent: bool


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    suite: str
    passed: bool
    cases: tuple[BenchmarkCaseResult, ...]
    passed_cases: int
    total_cases: int
    duration_seconds: float
    model_provenance: BenchmarkModelProvenance | None = None


def load_benchmark_suite(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> tuple[BenchmarkCase, ...]:
    suite_path = path.resolve()
    try:
        with suite_path.open("rb") as suite_file:
            raw_suite = suite_file.read(MAX_SUITE_BYTES + 1)
        if len(raw_suite) > MAX_SUITE_BYTES:
            raise BenchmarkDefinitionError("benchmark suite exceeds its size limit")
        payload = json.loads(raw_suite.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkDefinitionError("benchmark suite could not be read") from exc
    if not isinstance(payload, dict):
        raise BenchmarkDefinitionError("benchmark suite must be an object")
    _validate_object_fields(
        payload,
        allowed=SUITE_FIELDS,
        required=SUITE_FIELDS,
        label="benchmark suite",
    )
    if payload["version"] != 2:
        raise BenchmarkDefinitionError("benchmark suite version must be 2")
    fixture_cases = payload["cases"]
    repository_cases = payload["repository_cases"]
    if not isinstance(fixture_cases, list) or not isinstance(repository_cases, list):
        raise BenchmarkDefinitionError("benchmark cases must be arrays")
    if not fixture_cases:
        raise BenchmarkDefinitionError("benchmark suite must contain fixture cases")
    if len(fixture_cases) + len(repository_cases) > MAX_BENCHMARK_CASES:
        raise BenchmarkDefinitionError("benchmark suite exceeds the case limit")

    root = suite_path.parent
    cases: list[BenchmarkCase] = []
    identifiers: set[str] = set()
    for raw in fixture_cases:
        case = _parse_case(raw, root=root, repository_case=False)
        _add_case(case, cases, identifiers)
    for raw in repository_cases:
        case = _parse_case(
            raw,
            root=root,
            repository_case=True,
            repository_root=repository_root,
        )
        _add_case(case, cases, identifiers)
    return tuple(cases)


def run_benchmark_suite(
    suite_path: Path,
    *,
    client: StructuredReviewClient,
    repository_root: Path | None = None,
) -> BenchmarkSuiteResult:
    started = time.monotonic()
    cases = load_benchmark_suite(suite_path, repository_root=repository_root)
    results: list[BenchmarkCaseResult] = []
    with tempfile.TemporaryDirectory(prefix="inverse-agent-review-benchmark-") as temporary:
        temporary_root = Path(temporary)
        for case in cases:
            results.append(_run_case(case, temporary_root=temporary_root, client=client))
    passed_cases = sum(item.passed for item in results)
    return BenchmarkSuiteResult(
        suite=str(suite_path.resolve()),
        passed=passed_cases == len(results),
        cases=tuple(results),
        passed_cases=passed_cases,
        total_cases=len(results),
        duration_seconds=time.monotonic() - started,
    )


def score_review(
    report: CommitReviewReport,
    expected_findings: tuple[ExpectedFinding, ...],
    *,
    max_unmatched_findings: int,
    allowed_unmatched_severities: tuple[ReviewSeverity, ...] = (),
    max_alternative_findings: int = 0,
    min_model_supported_findings: int = 0,
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    options = [_expected_options(report, expected) for expected in expected_findings]
    missing = [
        expected.finding_id
        for expected, expected_options in zip(expected_findings, options, strict=True)
        if not expected_options
    ]
    if report.model_supported_findings < min_model_supported_findings:
        missing.append(
            "model-supported-findings:"
            f"{report.model_supported_findings}/{min_model_supported_findings}"
        )
    if min_model_supported_findings:
        model_options = [
            _finding_options(report.model_findings, expected) for expected in expected_findings
        ]
        model_missing = [
            f"model-supported:{expected.finding_id}"
            for expected, expected_options in zip(
                expected_findings,
                model_options,
                strict=True,
            )
            if not expected_options
        ]
        missing.extend(model_missing)
        _model_indexes, model_assignment_complete = _select_consistent_assignment(model_options)
        if not model_missing and not model_assignment_complete:
            missing.append("model-supported:distinct-finding-assignment")
    matched_indexes, assignment_complete = _select_consistent_assignment(options)
    if not missing and not assignment_complete:
        missing.append("distinct-finding-assignment")
    alternative_indexes = {
        index for expected_options in options for option in expected_options for index in option
    }

    unmatched_items = tuple(
        (
            index,
            finding.severity,
            f"{finding.severity.value} {finding.file}:{finding.line} {finding.title}",
        )
        for index, finding in enumerate(report.findings)
        if index not in matched_indexes
    )
    unmatched = tuple(description for _index, _severity, description in unmatched_items)
    alternative_unmatched = tuple(
        description
        for index, _severity, description in unmatched_items
        if index in alternative_indexes
    )
    blocking_unmatched = tuple(
        description
        for index, severity, description in unmatched_items
        if index not in alternative_indexes and severity not in allowed_unmatched_severities
    )
    passed = (
        not missing
        and report.verdict != "INCOMPLETE"
        and len(blocking_unmatched) <= max_unmatched_findings
        and len(alternative_unmatched) <= max_alternative_findings
    )
    return passed, tuple(missing), unmatched


def _select_consistent_assignment(
    options: list[tuple[frozenset[int], ...]],
) -> tuple[set[int], bool]:
    states = {0: 0}
    for expected_options in options:
        option_masks = {
            sum(1 << index for index in option) for option in expected_options if option
        }
        next_states = dict(states)
        for state, assigned in states.items():
            for option in option_masks:
                if state & option:
                    continue
                merged = state | option
                next_states[merged] = max(next_states.get(merged, -1), assigned + 1)
        states = next_states
    selected, assigned = max(
        states.items(),
        key=lambda item: (item[1], item[0].bit_count(), -item[0]),
    )
    return (
        {index for index in range(MAX_FINDINGS) if selected & (1 << index)},
        assigned == len(options),
    )


def _expected_options(
    report: CommitReviewReport,
    expected: ExpectedFinding,
) -> tuple[frozenset[int], ...]:
    return _finding_options(report.findings, expected)


def _finding_options(
    findings: Sequence[ReviewFinding],
    expected: ExpectedFinding,
) -> tuple[frozenset[int], ...]:
    coverages = {
        index: _finding_coverage(finding, expected)
        for index, finding in enumerate(findings)
        if finding.file in expected.paths
        and finding.severity in expected.severities
        and _finding_is_grounded(finding, expected)
    }
    coverages = {index: coverage for index, coverage in coverages.items() if coverage}
    required = set(range(len(expected.keyword_groups)))
    options: list[frozenset[int]] = []
    maximum = min(len(coverages), len(required), expected.max_match_findings)
    for size in range(1, maximum + 1):
        for raw_option in combinations(coverages, size):
            option = frozenset(raw_option)
            if set().union(*(coverages[index] for index in option)) != required:
                continue
            if any(
                coverages[index]
                <= set().union(
                    *(coverages[other] for other in option if other != index),
                    set(),
                )
                for index in option
            ):
                continue
            options.append(option)
            if len(options) >= MAX_MATCH_OPTIONS:
                return tuple(options)
    return tuple(options)


def _run_case(
    case: BenchmarkCase,
    *,
    temporary_root: Path,
    client: StructuredReviewClient,
) -> BenchmarkCaseResult:
    started = time.monotonic()
    commit: str | None = None
    try:
        if case.fixture is not None:
            workspace, commit = _materialize_fixture(case, temporary_root)
        else:
            if case.workspace is None or case.commit is None:
                raise BenchmarkDefinitionError("repository case is incomplete")
            workspace, commit = case.workspace, case.commit
        report = review_commit(
            workspace,
            commit,
            domain=case.domain,
            goal=case.goal,
            client=client,
        )
        passed, missing, unmatched = score_review(
            report,
            case.expected_findings,
            max_unmatched_findings=case.max_unmatched_findings,
            allowed_unmatched_severities=case.allowed_unmatched_severities,
            max_alternative_findings=case.max_alternative_findings,
            min_model_supported_findings=case.min_model_supported_findings,
        )
        return BenchmarkCaseResult(
            case_id=case.case_id,
            passed=passed,
            commit=report.commit,
            report=report,
            missing_findings=missing,
            unmatched_findings=unmatched,
            error=None,
            duration_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        return BenchmarkCaseResult(
            case_id=case.case_id,
            passed=False,
            commit=commit,
            report=None,
            missing_findings=tuple(item.finding_id for item in case.expected_findings),
            unmatched_findings=(),
            error=redact_text(str(exc)).text[:2000],
            duration_seconds=time.monotonic() - started,
        )


def _materialize_fixture(case: BenchmarkCase, temporary_root: Path) -> tuple[Path, str]:
    if case.fixture is None:
        raise BenchmarkDefinitionError("fixture case has no fixture path")
    base = case.fixture / "base"
    after = case.fixture / "after"
    _validate_fixture_tree(case.fixture)
    if not base.is_dir() or not after.is_dir():
        raise BenchmarkDefinitionError("benchmark fixture requires base and after directories")
    workspace = temporary_root / case.case_id
    shutil.copytree(base, workspace)
    if _fixture_contains_git_state(workspace):
        raise BenchmarkDefinitionError("copied benchmark fixture contains Git state")
    git = discover_trusted_git()
    if git is None:
        raise BenchmarkDefinitionError("Git is required to materialize benchmark fixtures")
    allowed_env = RunnerPolicy(workspace_root=workspace, allowed_commands=[]).allowed_env_names
    env = build_safe_subprocess_env(allowed_env)
    env.update(
        {
            "GIT_AUTHOR_NAME": "Inverse-Agent Benchmark",
            "GIT_AUTHOR_EMAIL": "benchmark@inverse.local",
            "GIT_AUTHOR_DATE": FIXTURE_GIT_DATE,
            "GIT_COMMITTER_NAME": "Inverse-Agent Benchmark",
            "GIT_COMMITTER_EMAIL": "benchmark@inverse.local",
            "GIT_COMMITTER_DATE": FIXTURE_GIT_DATE,
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
        }
    )
    _fixture_git(
        git,
        workspace,
        env,
        "init",
        "--quiet",
        "--initial-branch=main",
        "--object-format=sha1",
    )
    _fixture_git(git, workspace, env, "config", "user.name", "Inverse-Agent Benchmark")
    _fixture_git(git, workspace, env, "config", "user.email", "benchmark@inverse.local")
    _fixture_git(git, workspace, env, "config", "commit.gpgSign", "false")
    _fixture_git(git, workspace, env, "config", "core.autocrlf", "false")
    _fixture_git(git, workspace, env, "config", "core.filemode", "false")
    _fixture_git(git, workspace, env, "add", "--all")
    _fixture_git(
        git,
        workspace,
        env,
        "commit",
        "--quiet",
        "--allow-empty",
        "-m",
        "benchmark baseline",
    )
    if workspace.resolve().parent != temporary_root.resolve():
        raise BenchmarkDefinitionError("benchmark workspace escaped its temporary root")
    for item in workspace.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    shutil.copytree(after, workspace, dirs_exist_ok=True, copy_function=shutil.copy)
    # Rebuild the index so same-size fixture files with colliding timestamps cannot
    # be mistaken for unchanged content by Git's stat cache.
    _fixture_git(
        git,
        workspace,
        env,
        "rm",
        "-r",
        "--cached",
        "--ignore-unmatch",
        "--quiet",
        "--",
        ".",
    )
    _fixture_git(git, workspace, env, "add", "--all")
    _fixture_git(
        git,
        workspace,
        env,
        "commit",
        "--quiet",
        "--allow-empty",
        "-m",
        "benchmark change",
    )
    commit = _fixture_git(git, workspace, env, "rev-parse", "HEAD").strip()
    return workspace, commit


def _fixture_git(git: Path, workspace: Path, env: dict[str, str], *arguments: str) -> str:
    try:
        completed = subprocess.run(
            [str(git), "--no-optional-locks", *arguments],
            cwd=workspace,
            env=env,
            shell=False,
            capture_output=True,
            check=False,
            timeout=BENCHMARK_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        stage = arguments[0] if arguments else "unknown"
        raise BenchmarkDefinitionError(f"benchmark Git setup failed during {stage}") from exc
    if completed.returncode != 0:
        raw_error = completed.stderr or completed.stdout
        error = redact_text(raw_error.decode("utf-8", errors="replace")).text.strip()
        stage = arguments[0] if arguments else "unknown"
        raise BenchmarkDefinitionError(
            f"benchmark Git setup failed during {stage}: {error or 'unknown error'}"
        )
    if len(completed.stdout) > 64 * 1024:
        raise BenchmarkDefinitionError("benchmark Git setup output exceeded its limit")
    return completed.stdout.decode("utf-8", errors="replace")


def _validate_fixture_tree(root: Path) -> None:
    is_junction = bool(getattr(root, "is_junction", lambda: False)())
    if root.is_symlink() or is_junction:
        raise BenchmarkDefinitionError("benchmark fixture roots must not be links or junctions")
    if not root.is_dir():
        raise BenchmarkDefinitionError(f"benchmark fixture directory is missing: {root}")
    files = 0
    total_bytes = 0
    for item in root.rglob("*"):
        is_junction = bool(getattr(item, "is_junction", lambda: False)())
        if item.is_symlink() or is_junction:
            raise BenchmarkDefinitionError("benchmark fixtures must not contain links or junctions")
        relative = item.relative_to(root)
        if any(_is_git_state_component(part) for part in relative.parts):
            raise BenchmarkDefinitionError("benchmark fixtures must not contain Git state")
        if not item.is_file():
            continue
        files += 1
        total_bytes += item.stat().st_size
        if files > MAX_FIXTURE_FILES or total_bytes > MAX_FIXTURE_BYTES:
            raise BenchmarkDefinitionError("benchmark fixture exceeds its size limit")


def _fixture_contains_git_state(root: Path) -> bool:
    return any(
        any(_is_git_state_component(part) for part in item.relative_to(root).parts)
        for item in root.rglob("*")
    )


def _is_git_state_component(name: str) -> bool:
    normalized = name.rstrip(" .").casefold()
    return normalized == ".git" or normalized.startswith(".git:")


def _parse_case(
    raw: object,
    *,
    root: Path,
    repository_case: bool,
    repository_root: Path | None = None,
) -> BenchmarkCase:
    if not isinstance(raw, dict):
        raise BenchmarkDefinitionError("benchmark case must be an object")
    case_fields = COMMON_CASE_FIELDS | ({"workspace", "commit"} if repository_case else {"fixture"})
    _validate_object_fields(
        raw,
        allowed=case_fields,
        required={"id", "domain", "goal", "expected_findings"}
        | ({"workspace", "commit"} if repository_case else {"fixture"}),
        label="benchmark case",
    )
    case_id = _required_text(raw, "id", maximum=100)
    if not all(character.isalnum() or character in "-_" for character in case_id):
        raise BenchmarkDefinitionError("benchmark case ID contains unsupported characters")
    try:
        domain = ReviewDomain(_required_text(raw, "domain", maximum=30))
    except ValueError as exc:
        raise BenchmarkDefinitionError("benchmark case has an unsupported domain") from exc
    goal = _required_text(raw, "goal", maximum=2000)
    expected = _parse_expected(raw["expected_findings"])
    max_unmatched_findings = raw.get("max_unmatched_findings", 0)
    if not isinstance(max_unmatched_findings, int) or isinstance(max_unmatched_findings, bool):
        raise BenchmarkDefinitionError("max_unmatched_findings must be an integer")
    if not 0 <= max_unmatched_findings <= MAX_EXPECTED_FINDINGS:
        raise BenchmarkDefinitionError("max_unmatched_findings is outside its limit")
    max_alternative_findings = raw.get("max_alternative_findings", 0)
    if not isinstance(max_alternative_findings, int) or isinstance(max_alternative_findings, bool):
        raise BenchmarkDefinitionError("max_alternative_findings must be an integer")
    if not 0 <= max_alternative_findings <= MAX_EXPECTED_FINDINGS:
        raise BenchmarkDefinitionError("max_alternative_findings is outside its limit")
    min_model_supported_findings = raw.get("min_model_supported_findings", 0)
    if not isinstance(min_model_supported_findings, int) or isinstance(
        min_model_supported_findings, bool
    ):
        raise BenchmarkDefinitionError("min_model_supported_findings must be an integer")
    if not 0 <= min_model_supported_findings <= MAX_EXPECTED_FINDINGS:
        raise BenchmarkDefinitionError("min_model_supported_findings is outside its limit")
    allowed_unmatched_values = _text_array(
        raw.get("allowed_unmatched_severities", []),
        "allowed unmatched severities",
    )
    try:
        allowed_unmatched_severities = tuple(
            ReviewSeverity(value) for value in allowed_unmatched_values
        )
    except ValueError as exc:
        raise BenchmarkDefinitionError("allowed unmatched severity is invalid") from exc

    if repository_case:
        if repository_root is None:
            raise BenchmarkDefinitionError("repository cases require an explicit repository root")
        allowed_root = repository_root.resolve()
        workspace_value = _required_text(raw, "workspace", maximum=500)
        workspace = (allowed_root / workspace_value).resolve()
        if not workspace.is_relative_to(allowed_root):
            raise BenchmarkDefinitionError("repository case escapes the allowed repository root")
        commit = _required_text(raw, "commit", maximum=64)
        return BenchmarkCase(
            case_id=case_id,
            domain=domain,
            goal=goal,
            expected_findings=expected,
            max_unmatched_findings=max_unmatched_findings,
            allowed_unmatched_severities=allowed_unmatched_severities,
            max_alternative_findings=max_alternative_findings,
            min_model_supported_findings=min_model_supported_findings,
            workspace=workspace,
            commit=commit,
        )
    fixture_name = _required_text(raw, "fixture", maximum=100)
    if not all(character.isalnum() or character in "-_" for character in fixture_name):
        raise BenchmarkDefinitionError("fixture name contains unsupported characters")
    return BenchmarkCase(
        case_id=case_id,
        domain=domain,
        goal=goal,
        expected_findings=expected,
        max_unmatched_findings=max_unmatched_findings,
        allowed_unmatched_severities=allowed_unmatched_severities,
        max_alternative_findings=max_alternative_findings,
        min_model_supported_findings=min_model_supported_findings,
        fixture=root / fixture_name,
    )


def _parse_expected(raw_items: object) -> tuple[ExpectedFinding, ...]:
    if not isinstance(raw_items, list) or len(raw_items) > MAX_EXPECTED_FINDINGS:
        raise BenchmarkDefinitionError("expected findings are invalid")
    expected: list[ExpectedFinding] = []
    identifiers: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise BenchmarkDefinitionError("expected finding must be an object")
        _validate_object_fields(
            raw,
            allowed=EXPECTED_FINDING_FIELDS,
            required={"id", "paths", "severities", "keyword_groups", "anchors"},
            label="expected finding",
        )
        finding_id = _required_text(raw, "id", maximum=100)
        if finding_id in identifiers:
            raise BenchmarkDefinitionError(f"duplicate expected finding ID: {finding_id}")
        identifiers.add(finding_id)
        paths = _text_array(raw.get("paths"), "expected finding paths")
        if not paths:
            raise BenchmarkDefinitionError("expected finding must include a path")
        severity_values = _text_array(raw.get("severities"), "expected severities")
        try:
            severities = tuple(ReviewSeverity(value) for value in severity_values)
        except ValueError as exc:
            raise BenchmarkDefinitionError("expected finding severity is invalid") from exc
        groups_raw = raw.get("keyword_groups")
        if (
            not isinstance(groups_raw, list)
            or not groups_raw
            or len(groups_raw) > MAX_MATCH_FINDINGS_PER_EXPECTED
        ):
            raise BenchmarkDefinitionError("expected finding must include keyword groups")
        groups = tuple(
            tuple(value.casefold() for value in _text_array(group, "keyword group"))
            for group in groups_raw
        )
        if any(not group for group in groups):
            raise BenchmarkDefinitionError("keyword groups must not be empty")
        anchors_raw = raw.get("anchors")
        if (
            not isinstance(anchors_raw, list)
            or not anchors_raw
            or len(anchors_raw) > MAX_EXPECTED_FINDINGS
        ):
            raise BenchmarkDefinitionError("expected finding must include bounded anchors")
        anchors: list[ExpectedAnchor] = []
        for anchor_raw in anchors_raw:
            if not isinstance(anchor_raw, dict):
                raise BenchmarkDefinitionError("expected finding anchor must be an object")
            _validate_object_fields(
                anchor_raw,
                allowed=EXPECTED_ANCHOR_FIELDS,
                required=EXPECTED_ANCHOR_FIELDS,
                label="expected finding anchor",
            )
            anchor_path = _required_text(anchor_raw, "path", maximum=500)
            if anchor_path not in paths:
                raise BenchmarkDefinitionError("expected anchor path must be an expected path")
            change = _required_text(anchor_raw, "change", maximum=10)
            if change not in {"added", "removed"}:
                raise BenchmarkDefinitionError("expected anchor change side is invalid")
            raw_lines = anchor_raw.get("lines")
            if (
                not isinstance(raw_lines, list)
                or not raw_lines
                or len(raw_lines) > MAX_EXPECTED_FINDINGS
                or any(
                    not isinstance(line, int)
                    or isinstance(line, bool)
                    or not 1 <= line <= 10_000_000
                    for line in raw_lines
                )
            ):
                raise BenchmarkDefinitionError("expected anchor lines are invalid")
            evidence = _text_array(anchor_raw.get("evidence"), "expected anchor evidence")
            if not evidence:
                raise BenchmarkDefinitionError("expected anchor evidence must not be empty")
            anchors.append(
                ExpectedAnchor(
                    path=anchor_path,
                    change=change,
                    lines=tuple(raw_lines),
                    evidence=evidence,
                )
            )
        max_match_findings = raw.get("max_match_findings", MAX_MATCH_FINDINGS_PER_EXPECTED)
        if (
            not isinstance(max_match_findings, int)
            or isinstance(max_match_findings, bool)
            or not 1 <= max_match_findings <= MAX_MATCH_FINDINGS_PER_EXPECTED
        ):
            raise BenchmarkDefinitionError("max_match_findings is outside its limit")
        expected.append(
            ExpectedFinding(
                finding_id,
                paths,
                severities,
                groups,
                anchors=tuple(anchors),
                max_match_findings=max_match_findings,
            )
        )
    return tuple(expected)


def _finding_is_grounded(finding: ReviewFinding, expected: ExpectedFinding) -> bool:
    if not expected.anchors:
        return True
    normalized_evidence = _normalize_match_text(finding.evidence)
    return any(
        finding.file == anchor.path
        and finding.change == anchor.change
        and finding.line in anchor.lines
        and any(_normalize_match_text(value) in normalized_evidence for value in anchor.evidence)
        for anchor in expected.anchors
    )


def _finding_coverage(finding: ReviewFinding, expected: ExpectedFinding) -> set[int]:
    claim_lines: list[str] = []
    for line in finding.body.splitlines():
        heading = re.sub(r"[*_#`]", "", line).strip().casefold()
        section_label = heading.split(":", maxsplit=1)[0].strip()
        if section_label in REMEDIATION_SECTION_HEADINGS:
            break
        claim_lines.append(line)
    claim_body = "\n".join(claim_lines)
    haystack = _normalize_match_text(f"{finding.title}\n{claim_body}")
    return {
        index
        for index, group in enumerate(expected.keyword_groups)
        if any(_contains_positive_keyword(haystack, keyword) for keyword in group)
    }


def _contains_positive_keyword(haystack: str, keyword: str) -> bool:
    tokens = haystack.split()
    needle = _normalize_match_text(keyword).split()
    if not needle:
        return False
    negations = {"no", "not", "never", "without", "neither", "cannot", "cant"}
    safety_failures = {"fail", "fails", "failed", "failure", "lacks", "missing", *negations}
    width = len(needle)
    for index in range(0, len(tokens) - width + 1):
        actual = tokens[index : index + width]
        if not all(
            value == expected or (len(expected) >= 6 and value.startswith(expected))
            for value, expected in zip(actual, needle, strict=True)
        ):
            continue
        before = set(tokens[max(0, index - 4) : index])
        after_tokens = tokens[index + width : index + width + 5]
        after = set(after_tokens)
        needle_is_safety_control = _has_safety_language(set(actual))
        safety_denied_before = bool(before & safety_failures) and (
            _has_safety_language(before) or needle_is_safety_control
        )
        safety_denied_after = bool(after & safety_failures) and (
            _has_safety_language(after) or needle_is_safety_control
        )
        postfix_negated = bool(
            after_tokens
            and (
                after_tokens[0] in {"no", "not", "never", "cannot", "cant"}
                or (
                    after_tokens[0] in {"is", "are", "was", "were", "seems", "appears"}
                    and bool(
                        set(after_tokens[1:3])
                        & {
                            "no",
                            "not",
                            "never",
                            "impossible",
                            "unlikely",
                            "missing",
                            "absent",
                            "disabled",
                        }
                    )
                )
            )
        )
        if needle_is_safety_control:
            if safety_denied_before or safety_denied_after or postfix_negated:
                return True
            continue
        if (
            (before & negations and not safety_denied_before)
            or (postfix_negated and not safety_denied_after)
            or (_has_safety_language(before) and not safety_denied_before)
            or (_has_safety_language(after) and not safety_denied_after)
        ):
            continue
        return True
    return False


def _has_safety_language(tokens: set[str]) -> bool:
    exact = {"safe", "safely", "secure", "impossible", "unlikely"}
    stems = ("prevent", "mitigat", "avoid", "escap", "parameteriz", "sanitiz", "protect")
    return bool(tokens & exact) or any(token.startswith(stems) for token in tokens)


def _normalize_match_text(value: str) -> str:
    normalized = "".join(
        character.casefold() if character.isalnum() or category(character).startswith("L") else " "
        for character in value
    )
    return " ".join(normalized.split())


def _validate_object_fields(
    raw: dict[str, object],
    *,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str],
    label: str,
) -> None:
    unknown = set(raw) - set(allowed)
    if unknown:
        raise BenchmarkDefinitionError(
            f"{label} contains unsupported field(s): {', '.join(sorted(unknown))}"
        )
    missing = set(required) - set(raw)
    if missing:
        raise BenchmarkDefinitionError(
            f"{label} is missing required field(s): {', '.join(sorted(missing))}"
        )


def _required_text(raw: dict[str, object], name: str, *, maximum: int) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise BenchmarkDefinitionError(f"{name} must be non-empty bounded text")
    return value.strip()


def _text_array(raw: object, name: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) > 32:
        raise BenchmarkDefinitionError(f"{name} must be a bounded array")
    values: list[str] = []
    for value in raw:
        if not isinstance(value, str) or not value or len(value) > 200:
            raise BenchmarkDefinitionError(f"{name} contains invalid text")
        values.append(value)
    return tuple(values)


def _add_case(
    case: BenchmarkCase,
    cases: list[BenchmarkCase],
    identifiers: set[str],
) -> None:
    if case.case_id in identifiers:
        raise BenchmarkDefinitionError(f"duplicate benchmark case ID: {case.case_id}")
    identifiers.add(case.case_id)
    cases.append(case)

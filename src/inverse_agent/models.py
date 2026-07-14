"""Core typed contracts for Inverse-Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4


class Domain(StrEnum):
    DJANGO = "django"
    PYTORCH = "pytorch"
    ANDROID = "android"
    ANDROID_NDK = "android_ndk"
    IOS = "ios"
    GENERIC = "generic"


class AutonomyLevel(int, Enum):
    ADVISORY = 0
    ASSISTED = 1
    BOUNDED_AUTO = 2
    WORKFLOW_AUTO = 3


class InferenceMode(StrEnum):
    CLOUD_NO_RETENTION = "cloud_no_retention"
    LOCAL_SELF_HOSTED = "local_self_hosted"
    HYBRID = "hybrid"


class ArtifactKind(StrEnum):
    DIFF = "diff"
    BUILD_LOG = "build_log"
    TEST_LOG = "test_log"
    EXPERIMENT_RESULT = "experiment_result"
    CHART = "chart"
    NOTEBOOK = "notebook"
    REPORT = "report"
    PR_DRAFT = "pr_draft"
    TRACE = "trace"


class RunKind(StrEnum):
    """The durable workflow family executed for a run."""

    VERIFICATION = "verification"
    INVESTIGATION = "investigation"


class RunStatus(StrEnum):
    PLANNED = "planned"
    QUEUED = "queued"
    STARTING = "starting"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    APPROVING = "approving"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REFUSED = "refused"


@dataclass(frozen=True)
class ApprovalRule:
    """A policy statement for actions that need human approval."""

    name: str
    reason: str
    categories: tuple[str, ...] = ()


@dataclass
class WorkspaceProfile:
    """Detected repository/toolchain profile."""

    root: Path
    domains: set[Domain]
    commands: dict[str, list[str]] = field(default_factory=dict)
    test_targets: list[str] = field(default_factory=list)
    toolchain: dict[str, str] = field(default_factory=dict)
    unavailable_tools: dict[str, str] = field(default_factory=dict)
    secrets_required: list[str] = field(default_factory=list)
    risk_rules: list[ApprovalRule] = field(default_factory=list)
    inference_mode: InferenceMode = InferenceMode.CLOUD_NO_RETENTION
    autonomy: dict[Domain, AutonomyLevel] = field(default_factory=dict)

    def autonomy_for(self, domain: Domain) -> AutonomyLevel:
        return self.autonomy.get(domain, AutonomyLevel.ASSISTED)


@dataclass
class AgentSpec:
    name: str
    role: str
    allowed_tools: set[str]
    model_policy: dict[str, Any] = field(default_factory=dict)
    memory_policy: dict[str, Any] = field(default_factory=dict)
    approval_policy: list[ApprovalRule] = field(default_factory=list)
    fallback_behavior: str = "stop_and_report"


@dataclass
class RunSpec:
    goal: str
    workspace: Path
    domain: Domain
    kind: RunKind = RunKind.VERIFICATION
    autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED
    budget: dict[str, int | float] = field(default_factory=dict)
    expected_artifacts: set[ArtifactKind] = field(default_factory=set)
    stop_conditions: list[str] = field(default_factory=list)
    planner_fingerprint: str = "deterministic"
    run_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class CommandRule:
    """Structured argv allowlist rule.

    Rules match exact argv by default after executable normalization. A rule may
    opt into trailing arguments only when a caller supplies a separate validator.
    """

    name: str
    argv_prefix: tuple[str, ...]
    domain: Domain
    requires_approval: bool = False
    network_required: bool = False
    reason: str = ""
    workspace_path_args: tuple[int, ...] = ()


@dataclass
class RunnerPolicy:
    workspace_root: Path
    allowed_commands: list[CommandRule]
    network_default: str = "deny"
    secrets_default: str = "deny"
    compute_budget_seconds: int = 300
    artifact_upload_default: str = "metadata_only"
    trusted_executables: dict[str, tuple[Path, ...]] = field(default_factory=dict)
    allowed_workspace_executables: tuple[Path, ...] = ()
    output_limit_bytes: int = 2_000_000
    allowed_env_names: tuple[str, ...] = (
        "ANDROID_HOME",
        "ANDROID_SDK_ROOT",
        "COMSPEC",
        "CUDA_VISIBLE_DEVICES",
        "GRADLE_USER_HOME",
        "HOME",
        "JAVA_HOME",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
    )

    def rules_for(self, domain: Domain) -> list[CommandRule]:
        return [rule for rule in self.allowed_commands if rule.domain in {domain, Domain.GENERIC}]


@dataclass
class Artifact:
    kind: ArtifactKind
    path: Path | None = None
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class EvalTrace:
    task_input: str
    domain: Domain
    baseline: str
    run_id: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    cost: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    status: RunStatus = RunStatus.PLANNED
    human_edits_after_output: int = 0
    planner_fingerprint: str = "deterministic"
    plan: list[str] = field(default_factory=list)
    plan_rationale: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def record_action(self, name: str, **metadata: Any) -> None:
        self.actions.append(
            {"name": name, "metadata": metadata, "at": datetime.now(UTC).isoformat()}
        )

    def record_artifact(self, artifact: Artifact) -> None:
        self.artifacts.append(artifact)

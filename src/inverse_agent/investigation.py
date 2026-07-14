"""Read-only investigation loop.

A durable ``decide -> dispatch -> observe -> decide`` loop drives the safe read
tools and returns an evidence-backed answer. A planner (a scripted deterministic
planner here; an OpenAI-compatible client in production) selects exactly one tool
call per decision or emits a final answer. Every observation is retained in a
durable catalog, and the final answer's citations are validated against that
catalog, not against the model's prose. Budgets bound the loop, and a small set
of mechanically decidable conditions force an ``INCOMPLETE`` verdict.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol

from inverse_agent.attestations import AttestationScope
from inverse_agent.fs_tools import (
    FsToolError,
    PolicyViolationError,
    RequestValidationError,
    StrictDecodeError,
    ToolObservation,
    WorkspaceReader,
    canonical_glob_scope,
    glob_uses_recursive_listing,
)
from inverse_agent.planner import PlannerAttestationError, PlannerBudgetError

__all__ = [
    "AgentAnswer",
    "AgentBudget",
    "CommandExecutor",
    "CommandExecution",
    "Decision",
    "InvestigationLoop",
    "ModelCallRecord",
    "InvestigationPlanner",
    "InvestigationReport",
    "InvestigationVerdict",
    "ScriptedInvestigationPlanner",
    "SourceCitation",
    "StopReason",
    "ToolCall",
    "ToolObservation",
    "citation_intersects_redaction",
    "line_body",
]


class InvestigationVerdict(StrEnum):
    PASS = "pass"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    FAILED = "failed"


class StopReason(StrEnum):
    FINISHED = "finished"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MALFORMED_ANSWER = "malformed_answer"
    UNSUPPORTED_CITATION = "unsupported_citation"
    STRICT_DECODE_REFUSAL = "strict_decode_refusal"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    NO_PROGRESS = "no_progress"
    PROTOCOL_FAILURE = "protocol_failure"
    NOT_ATTESTED = "not_attested"
    POLICY_VIOLATION = "policy_violation"


# Hard ceilings that a per-run budget override may never exceed.
MAX_DECISIONS_CEILING = 24
MAX_TOOL_CALLS_CEILING = 20
MAX_COMMAND_CALLS_CEILING = 4
MAX_PHYSICAL_CEILING = 36
MAX_COMPLETION_TOKENS_CEILING = 49_152
MAX_OBSERVATION_BYTES_CEILING = 2 * 1024 * 1024
MAX_ACTIVE_SECONDS_CEILING = 3_600.0
MIN_COMPLETION_TOKENS_PER_DECISION = 1_024


@dataclass(frozen=True)
class ToolCall:
    """A model decision to invoke one read tool with validated arguments."""

    tool: str
    path: str = "."
    query: str | None = None
    glob: str | None = None
    start_line: int = 1
    max_lines: int = 200
    command: str | None = None
    based_on_observation_id: str | None = None


@dataclass(frozen=True)
class SourceCitation:
    """A claim anchored to a real earlier observation and line range."""

    observation_id: str
    path: str
    start_line: int
    end_line: int
    note: str = ""


@dataclass(frozen=True)
class AgentAnswer:
    """A structured final answer the planner emits to conclude a run."""

    summary: str
    findings: tuple[str, ...]
    next_actions: tuple[str, ...]
    citations: tuple[SourceCitation, ...]
    complete: bool = True
    # The model's explicit conclusion: does the investigated concern hold? A
    # benchmark case scores this against its expected value, so a citation to the
    # right evidence line paired with a contrary conclusion cannot pass.
    issue_present: bool = True


@dataclass(frozen=True)
class ModelCallRecord:
    """One physical model request, including failures and conservative charging."""

    request_index: int
    logical_decision: int
    requested_completion_tokens: int
    charged_completion_tokens: int
    reported_prompt_tokens: int | None
    reported_completion_tokens: int | None
    reported_model: str | None
    latency_seconds: float
    outcome: str
    request_kind: str = "decision"


@dataclass(frozen=True)
class AgentBudget:
    """Loop budgets. Decisions decompose as tool decisions plus answer/recovery."""

    max_decisions: int = 20
    max_tool_calls: int = 16
    max_command_calls: int = 4
    max_physical_requests: int = 30
    max_completion_tokens: int = 24_576
    max_observation_bytes: int = 512 * 1024
    max_active_seconds: float = 600.0

    def validate(self) -> None:
        if self.max_tool_calls > self.max_decisions:
            raise ValueError("tool-call budget cannot exceed the decision budget")
        if not 1 <= self.max_decisions <= MAX_DECISIONS_CEILING:
            raise ValueError(f"max_decisions must be between 1 and {MAX_DECISIONS_CEILING}")
        if not 1 <= self.max_tool_calls <= MAX_TOOL_CALLS_CEILING:
            raise ValueError(f"max_tool_calls must be between 1 and {MAX_TOOL_CALLS_CEILING}")
        if not 0 <= self.max_command_calls <= MAX_COMMAND_CALLS_CEILING:
            raise ValueError(f"max_command_calls must be between 0 and {MAX_COMMAND_CALLS_CEILING}")
        if not 1 <= self.max_physical_requests <= MAX_PHYSICAL_CEILING:
            raise ValueError(f"max_physical_requests must be between 1 and {MAX_PHYSICAL_CEILING}")
        minimum_completion_tokens = self.max_decisions * MIN_COMPLETION_TOKENS_PER_DECISION
        if (
            not minimum_completion_tokens
            <= self.max_completion_tokens
            <= (MAX_COMPLETION_TOKENS_CEILING)
        ):
            raise ValueError(
                "max_completion_tokens must preserve at least 1024 tokens per decision "
                f"and not exceed {MAX_COMPLETION_TOKENS_CEILING}"
            )
        if not 1 <= self.max_observation_bytes <= MAX_OBSERVATION_BYTES_CEILING:
            raise ValueError(
                f"max_observation_bytes must be between 1 and {MAX_OBSERVATION_BYTES_CEILING}"
            )
        if not 0 < self.max_active_seconds <= MAX_ACTIVE_SECONDS_CEILING:
            raise ValueError(
                f"max_active_seconds must be greater than zero and at most "
                f"{MAX_ACTIVE_SECONDS_CEILING:g}"
            )


def _call_signature(call: ToolCall) -> tuple[object, ...]:
    """Signature over only the arguments the tool actually consumes.

    Prevents evading no-progress detection by varying an ignored field (e.g. a
    ``query`` on ``list_files``, which the dispatcher ignores).
    """

    if call.tool == "read_file":
        return ("read_file", call.path, call.start_line, call.max_lines)
    if call.tool == "list_files":
        return ("list_files", call.path, call.glob)
    if call.tool == "search_text":
        return ("search_text", call.query, call.glob)
    if call.tool == "run_command":
        return ("run_command", call.command)
    # Unknown tools carry no dispatched arguments, so vary only by tool name;
    # otherwise a changing (ignored) path would evade no-progress detection.
    return (call.tool,)


# A decision is either a tool call or a final answer.
Decision = ToolCall | AgentAnswer


class InvestigationPlanner(Protocol):
    def decide(
        self,
        *,
        goal: str,
        catalog: tuple[ToolObservation, ...],
    ) -> Decision: ...


class ScopeGuard(Protocol):
    def has_scope(self, workspace: Path, scope: AttestationScope) -> bool: ...


@dataclass(frozen=True)
class CommandExecution:
    """A command observation plus human-approval time excluded from compute."""

    observation: ToolObservation
    approval_wait_seconds: float = 0.0


class CommandExecutor(Protocol):
    """Dispatch one frozen command tool and return a bounded observation."""

    def execute(
        self,
        call: ToolCall,
        *,
        run_id: str,
        active_deadline: float,
    ) -> CommandExecution: ...


@dataclass(frozen=True)
class InvestigationReport:
    run_id: str
    verdict: InvestigationVerdict
    stop_reason: StopReason
    answer: AgentAnswer | None
    catalog: tuple[ToolObservation, ...]
    decisions_used: int
    tool_calls_used: int
    physical_requests_used: int
    command_calls_used: int = 0
    completion_tokens_used: int = 0
    completion_tokens_charged: int = 0
    completion_tokens_requested: int = 0
    observation_bytes_used: int = 0
    active_seconds: float = 0.0
    transport_retries: int = 0
    schema_retries: int = 0
    model_calls: tuple[ModelCallRecord, ...] = ()
    error: str = ""


def _dispatch(reader: WorkspaceReader, call: ToolCall) -> ToolObservation:
    if call.tool == "read_file":
        return reader.read_file(call.path, start_line=call.start_line, max_lines=call.max_lines)
    if call.tool == "list_files":
        return reader.list_files(call.path, glob=call.glob)
    if call.tool == "search_text":
        if call.query is None:
            raise RequestValidationError("search_text requires a query")
        return reader.search_text(call.query, glob=call.glob)
    raise RequestValidationError(f"unknown tool: {call.tool}")


def line_body(numbered: str) -> str:
    """Strip the ``N: `` prefix a read/list observation prepends to each line."""

    _, _, body = numbered.partition(": ")
    return body


def _redacted_lines(observation: ToolObservation) -> frozenset[int]:
    raw = observation.metadata.get("redacted_lines", ())
    if not isinstance(raw, list | tuple):
        return frozenset()
    return frozenset(item for item in raw if isinstance(item, int) and item >= 1)


def citation_intersects_redaction(observation: ToolObservation, citation: SourceCitation) -> bool:
    redacted = _redacted_lines(observation)
    return any(citation.start_line <= line <= citation.end_line for line in redacted)


def _canonical_request_path(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/")
    parts = tuple(part for part in PurePosixPath(normalized).parts if part not in ("", "."))
    return "/".join(parts) or "."


def _optional_scope_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _is_evidence(observation: ToolObservation) -> bool:
    """Accept retained source reads and explicitly typed command evidence.

    Pointer results (``list_files`` / ``search_text``) locate files but are not
    citable evidence: their "lines" are directory entries or ``path:line: match``
    snippets whose numbering is a result index, not a source line, so a citation
    against them would resolve to nothing meaningful. Requiring a ``read_file``
    observation keeps the grounding invariant honest — an answer must cite content
    from a file it actually read.
    """

    citable_command = observation.tool == "run_command" and bool(
        observation.metadata.get("citable_command")
    )
    if observation.tool != "read_file" and not citable_command:
        return False
    if observation.metadata.get("refused"):
        return False
    if not observation.content_hash:
        return False
    if citable_command and (observation.incomplete or observation.truncated):
        return False
    # A blank or entirely-redacted window carries no citable evidence. A whole
    # line touched by a redaction span is non-citable even when safe text remains
    # beside the marker, because the omitted bytes may change its meaning.
    redacted = _redacted_lines(observation)
    for offset, numbered in enumerate(observation.lines):
        line_number = observation.start_line + offset
        if line_number not in redacted and line_body(numbered).strip():
            return True
    return False


def _validate_answer_structure(answer: AgentAnswer) -> str | None:
    if not answer.summary.strip():
        return "final answer summary is empty"
    if not answer.findings or any(not finding.strip() for finding in answer.findings):
        return "final answer must contain non-empty findings"
    if not answer.next_actions or any(not action.strip() for action in answer.next_actions):
        return "final answer must contain non-empty recommended next actions"
    if len(answer.findings) != len(answer.citations):
        return "each finding must have one positionally corresponding citation"
    if not answer.citations:
        return "final answer contains no citations"
    citation_keys = {
        (_canonical_request_path(citation.path), citation.start_line, citation.end_line)
        for citation in answer.citations
    }
    if len(citation_keys) != len(answer.citations):
        return "each finding must use a distinct citation range"
    return None


def _validate_citations(
    answer: AgentAnswer,
    catalog: tuple[ToolObservation, ...],
    *,
    identity_for: Callable[[str], str | None],
    enforce_distinct: bool = True,
) -> str | None:
    """Return an error string if any citation is unsupported, else None."""

    # Refusals, empty results, and binary observations are not citable evidence.
    by_id = {obs.observation_id: obs for obs in catalog if _is_evidence(obs)}
    physical_ranges: set[tuple[tuple[object, ...], int, int]] = set()
    for citation in answer.citations:
        observation = by_id.get(citation.observation_id)
        if observation is None:
            return f"citation references an unknown or non-evidence observation: {citation.observation_id}"
        if observation.path != citation.path:
            return f"citation path does not match observation: {citation.path}"
        if citation.start_line < 1 or citation.end_line < citation.start_line:
            return "citation line range is invalid"
        # The cited range must fall inside the non-empty line set actually returned.
        max_line = observation.start_line + len(observation.lines) - 1
        if citation.start_line < observation.start_line or citation.end_line > max_line:
            return "citation line range is outside the returned observation"
        if citation_intersects_redaction(observation, citation):
            return "citation intersects a redacted source line"
        # The cited lines themselves must contain real (non-blank) content.
        lo = citation.start_line - observation.start_line
        hi = citation.end_line - observation.start_line + 1
        if not any(line_body(line).strip() for line in observation.lines[lo:hi]):
            return "citation resolves only to blank or redacted content"
        identity = identity_for(observation.observation_id)
        if identity is None:
            return "citation evidence identity is unavailable"
        source_key: tuple[object, ...] = ("file", identity)
        range_key = (source_key, citation.start_line, citation.end_line)
        if enforce_distinct and range_key in physical_ranges:
            return "each finding must use a distinct physical citation range"
        physical_ranges.add(range_key)
    return None


def _has_unresolved_negative_uncertainty(
    answer: AgentAnswer,
    catalog: tuple[ToolObservation, ...],
    *,
    identity_for: Callable[[str], str | None],
) -> bool:
    """Whether current evidence can support the answer's negative conclusion.

    Catalog tools establish the explored scope, so only their latest result for
    the same request matters: a complete retry supersedes an earlier omission.
    Read windows are localized evidence rather than whole-repository coverage;
    truncation alone is acceptable when the cited lines are visible, but the
    latest read of a cited path may not be refused, redacted, or incomplete.
    """

    latest_pointers: dict[tuple[object, ...], ToolObservation] = {}
    for observation in catalog:
        if observation.metadata.get("request_invalid"):
            continue
        if observation.tool == "list_files":
            scope: tuple[object, ...] = (
                observation.tool,
                observation.path,
                _optional_scope_text(observation.metadata.get("glob")),
                bool(observation.metadata.get("recursive")),
            )
            latest_pointers[scope] = observation
        elif observation.tool == "search_text":
            scope = (
                observation.tool,
                _optional_scope_text(observation.metadata.get("query")),
                _optional_scope_text(observation.metadata.get("glob")),
            )
            latest_pointers[scope] = observation
    if any(obs.incomplete or obs.truncated for obs in latest_pointers.values()):
        return True

    by_id = {observation.observation_id: observation for observation in catalog}
    cited_observations: list[ToolObservation] = []
    for citation in answer.citations:
        cited_observation = by_id.get(citation.observation_id)
        if cited_observation is not None:
            cited_observations.append(cited_observation)
    for cited in cited_observations:
        cited_identity = identity_for(cited.observation_id)
        cited_path = _canonical_request_path(cited.path)
        folded_path = cited_path.casefold()
        latest_variants: dict[str, tuple[int, ToolObservation]] = {}
        last_authoritative_success = -1
        for index, observation in enumerate(catalog):
            if observation.tool != "read_file" or observation.metadata.get("request_invalid"):
                continue
            observed_path = _canonical_request_path(observation.path)
            if observed_path.casefold() != folded_path:
                continue
            latest_variants[observed_path] = (index, observation)
            observed_identity = identity_for(observation.observation_id)
            same_identity = (
                observed_identity is not None
                and cited_identity is not None
                and observed_identity == cited_identity
            )
            unresolved = observation.incomplete or bool(observation.metadata.get("refused"))
            if not unresolved and (same_identity or observed_path == cited_path):
                last_authoritative_success = index
        if any(
            index > last_authoritative_success
            and (observation.incomplete or bool(observation.metadata.get("refused")))
            for index, observation in latest_variants.values()
        ):
            return True
    return False


class InvestigationLoop:
    """Runs one durable investigation to a verdict."""

    def __init__(
        self,
        *,
        planner: InvestigationPlanner,
        trust: ScopeGuard,
        budget: AgentBudget | None = None,
        command_executor: CommandExecutor | None = None,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        evidence_identity_key: bytes | None = None,
    ) -> None:
        self.planner = planner
        self.trust = trust
        self.budget = budget or AgentBudget()
        self.command_executor = command_executor
        self.event_sink = event_sink
        self.cancel_requested = cancel_requested
        self.evidence_identity_key = evidence_identity_key
        self.budget.validate()
        self._started_at: float | None = None
        self._paused_seconds = 0.0
        self._command_calls_used = 0
        self._observation_bytes_used = 0
        self._prior_usage: dict[str, int | float] = {}
        self._prior_model_calls: tuple[ModelCallRecord, ...] = ()
        # If the planner makes real client requests, bound its total to the
        # physical-request budget so retries cannot exceed it, and count actual
        # requests rather than one-per-decision.
        if hasattr(planner, "max_total_requests"):
            planner.max_total_requests = self.budget.max_physical_requests
        if hasattr(planner, "max_logical_decisions"):
            planner.max_logical_decisions = self.budget.max_decisions
        if hasattr(planner, "max_completion_tokens"):
            planner.max_completion_tokens = self.budget.max_completion_tokens

    def _physical_count(self, fallback: int) -> int:
        """Actual client requests so far, from the planner when it tracks them."""

        made = getattr(self.planner, "requests_made", None)
        prior = int(self._prior_usage.get("physical_requests_used", 0))
        return prior + int(made) if isinstance(made, int) else max(prior, fallback)

    def _completion_charged(self) -> int:
        charged = getattr(self.planner, "completion_tokens_charged", None)
        return int(self._prior_usage.get("completion_tokens_charged", 0)) + (
            int(charged) if isinstance(charged, int) else 0
        )

    def _completion_reported(self) -> int:
        reported = getattr(self.planner, "completion_tokens_reported", None)
        return int(self._prior_usage.get("completion_tokens_used", 0)) + (
            int(reported) if isinstance(reported, int) else 0
        )

    def _completion_requested(self) -> int:
        requested = getattr(self.planner, "completion_tokens_requested", None)
        return int(self._prior_usage.get("completion_tokens_requested", 0)) + (
            int(requested) if isinstance(requested, int) else 0
        )

    def _model_calls(self) -> tuple[ModelCallRecord, ...]:
        calls = getattr(self.planner, "model_calls", ())
        if not isinstance(calls, list | tuple) or not all(
            isinstance(call, ModelCallRecord) for call in calls
        ):
            return self._prior_model_calls
        prior_requests = int(self._prior_usage.get("physical_requests_used", 0))
        prior_decisions = int(self._prior_usage.get("decisions_used", 0))
        rebased = tuple(
            replace(
                call,
                request_index=prior_requests + call.request_index,
                logical_decision=prior_decisions + call.logical_decision,
            )
            for call in calls
        )
        return self._prior_model_calls + rebased

    def _active_seconds(self) -> float:
        if self._started_at is None:
            return float(self._prior_usage.get("active_seconds", 0.0))
        return float(self._prior_usage.get("active_seconds", 0.0)) + max(
            0.0, time.monotonic() - self._started_at - self._paused_seconds
        )

    def _active_budget_exhausted(self) -> bool:
        return self._active_seconds() >= self.budget.max_active_seconds

    def _progress_snapshot(
        self,
        *,
        decisions: int,
        tool_calls: int,
        physical: int,
        command_calls: int,
    ) -> dict[str, object]:
        return {
            "decisions_used": decisions,
            "tool_calls_used": tool_calls,
            "physical_requests_used": physical,
            "command_calls_used": command_calls,
            "completion_tokens_used": self._completion_reported(),
            "completion_tokens_charged": self._completion_charged(),
            "completion_tokens_requested": self._completion_requested(),
            "observation_bytes_used": self._observation_bytes_used,
            "active_seconds": self._active_seconds(),
            "transport_retries": int(self._prior_usage.get("transport_retries", 0))
            + int(getattr(self.planner, "transport_retries", 0)),
            "schema_retries": int(self._prior_usage.get("schema_retries", 0))
            + int(getattr(self.planner, "schema_retries", 0)),
            "model_calls": [asdict(call) for call in self._model_calls()],
        }

    @staticmethod
    def _validated_prior_usage(
        prior_usage: dict[str, int | float] | None,
    ) -> dict[str, int | float]:
        usage = dict(prior_usage or {})
        integer_fields = (
            "decisions_used",
            "tool_calls_used",
            "physical_requests_used",
            "command_calls_used",
            "completion_tokens_used",
            "completion_tokens_charged",
            "completion_tokens_requested",
            "observation_bytes_used",
            "transport_retries",
            "schema_retries",
        )
        for name in integer_fields:
            value = usage.get(name, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"persisted investigation usage {name} is invalid")
        active_seconds = usage.get("active_seconds", 0.0)
        if (
            not isinstance(active_seconds, int | float)
            or isinstance(active_seconds, bool)
            or not math.isfinite(float(active_seconds))
            or active_seconds < 0
        ):
            raise ValueError("persisted investigation usage active_seconds is invalid")
        return usage

    def run(
        self,
        *,
        run_id: str,
        goal: str,
        workspace: Path,
        initial_catalog: tuple[ToolObservation, ...] = (),
        prior_usage: dict[str, int | float] | None = None,
        initial_evidence_identities: dict[str, str] | None = None,
        prior_tool_calls: tuple[ToolCall, ...] = (),
        resume_decision: Decision | None = None,
        initial_model_calls: tuple[ModelCallRecord, ...] = (),
        initial_transport_retries_used: int = 0,
        initial_schema_retries_used: int = 0,
        initial_physical_attempts_used: int = 0,
        initial_resume_request_kind: str = "decision",
        initial_compaction_notes: str = "",
        initial_compacted_observation_ids: tuple[str, ...] = (),
    ) -> InvestigationReport:
        if not 0 <= initial_transport_retries_used <= 1:
            raise ValueError("persisted logical transport-retry state is invalid")
        if not 0 <= initial_schema_retries_used <= 1:
            raise ValueError("persisted logical schema-retry state is invalid")
        if not 0 <= initial_physical_attempts_used <= 3:
            raise ValueError("persisted logical physical-attempt state is invalid")
        if initial_resume_request_kind not in {"decision", "compaction"}:
            raise ValueError("persisted model request kind is invalid")
        if not isinstance(initial_compaction_notes, str):
            raise ValueError("persisted compaction notes are invalid")
        if len(initial_compaction_notes) > 4096:
            raise ValueError("persisted compaction notes exceed the schema limit")
        if len(set(initial_compacted_observation_ids)) != len(
            initial_compacted_observation_ids
        ) or any(not item for item in initial_compacted_observation_ids):
            raise ValueError("persisted compacted observation IDs are invalid")
        self._prior_usage = self._validated_prior_usage(prior_usage)
        self._prior_model_calls = tuple(initial_model_calls)
        self._started_at = time.monotonic()
        self._paused_seconds = 0.0
        self._command_calls_used = int(self._prior_usage.get("command_calls_used", 0))
        catalog_bytes = sum(len(item.text.encode("utf-8")) for item in initial_catalog)
        self._observation_bytes_used = max(
            catalog_bytes,
            int(self._prior_usage.get("observation_bytes_used", 0)),
        )
        remaining_active = max(0.0, self.budget.max_active_seconds - self._active_seconds())
        active_deadline = self._started_at + remaining_active
        prior_physical = int(self._prior_usage.get("physical_requests_used", 0))
        prior_decisions = int(self._prior_usage.get("decisions_used", 0))
        prior_completion = int(self._prior_usage.get("completion_tokens_charged", 0))
        if hasattr(self.planner, "max_total_requests"):
            self.planner.max_total_requests = max(
                0, self.budget.max_physical_requests - prior_physical
            )
        if hasattr(self.planner, "max_logical_decisions"):
            self.planner.max_logical_decisions = max(0, self.budget.max_decisions - prior_decisions)
        if hasattr(self.planner, "max_completion_tokens"):
            self.planner.max_completion_tokens = max(
                0, self.budget.max_completion_tokens - prior_completion
            )
        if hasattr(self.planner, "active_deadline"):
            self.planner.active_deadline = active_deadline
        resolved = workspace.resolve()
        if not self.trust.has_scope(resolved, AttestationScope.SOURCE_READ):
            return self._finish(
                run_id,
                InvestigationVerdict.INCOMPLETE,
                StopReason.NOT_ATTESTED,
                None,
                list(initial_catalog),
                prior_decisions,
                int(self._prior_usage.get("tool_calls_used", 0)),
                prior_physical,
                error="workspace is not attested for source_read",
            )
        if hasattr(self.planner, "source_read_guard"):
            self.planner.source_read_guard = lambda: self.trust.has_scope(
                resolved, AttestationScope.SOURCE_READ
            )
        reader = WorkspaceReader.open(
            resolved,
            active_deadline=active_deadline,
            identity_key=self.evidence_identity_key,
        )
        catalog: list[ToolObservation] = list(initial_catalog)
        if not set(initial_compacted_observation_ids).issubset(
            {observation.observation_id for observation in catalog}
        ) or (initial_compacted_observation_ids and not initial_compaction_notes.strip()):
            raise ValueError("persisted compaction state is inconsistent")
        durable_identities = dict(initial_evidence_identities or {})

        def evidence_identity(observation_id: str) -> str | None:
            source_identity = reader.evidence_identity(observation_id)
            if source_identity is not None:
                return source_identity
            durable_identity = durable_identities.get(observation_id)
            if durable_identity is not None:
                return durable_identity
            for observed in catalog:
                if observed.observation_id != observation_id:
                    continue
                identity = observed.metadata.get("evidence_identity")
                return identity if isinstance(identity, str) and identity else None
            return None

        decisions = prior_decisions
        tool_calls = int(self._prior_usage.get("tool_calls_used", 0))
        command_calls = self._command_calls_used
        physical = prior_physical
        seen_calls: set[tuple[object, ...]] = set()
        no_progress_repeats = 0
        for prior_call in prior_tool_calls:
            signature = _call_signature(prior_call)
            if signature in seen_calls:
                no_progress_repeats += 1
            else:
                no_progress_repeats = 0
                seen_calls.add(signature)
        pending_decision = resume_decision

        def model_request_started(
            request: dict[str, int | float | str | None],
        ) -> None:
            if self.event_sink is None:
                return
            request_index = request.get("request_index")
            logical_decision = request.get("logical_decision")
            if (
                not isinstance(request_index, int)
                or isinstance(request_index, bool)
                or not isinstance(logical_decision, int)
                or isinstance(logical_decision, bool)
            ):
                raise ValueError("model request-start accounting is invalid")
            durable_request = dict(request)
            durable_request["request_index"] = (
                int(self._prior_usage.get("physical_requests_used", 0)) + request_index
            )
            durable_request["logical_decision"] = (
                int(self._prior_usage.get("decisions_used", 0)) + logical_decision
            )
            snapshot = self._progress_snapshot(
                decisions=decisions,
                tool_calls=tool_calls,
                physical=self._physical_count(decisions + 1),
                command_calls=command_calls,
            )
            if request["retry_kind"] == "transport":
                snapshot["transport_retries"] = max(
                    int(self._prior_usage.get("transport_retries", 0))
                    + int(getattr(self.planner, "transport_retries", 0)),
                    int(self._prior_usage.get("transport_retries", 0)) + 1,
                )
            if request["retry_kind"] == "schema":
                snapshot["schema_retries"] = max(
                    int(self._prior_usage.get("schema_retries", 0))
                    + int(getattr(self.planner, "schema_retries", 0)),
                    int(self._prior_usage.get("schema_retries", 0)) + 1,
                )
            self.event_sink(
                "investigation.model_request_started",
                {
                    **snapshot,
                    "request": durable_request,
                },
            )

        def model_compaction_completed(compaction: dict[str, object]) -> None:
            if self.event_sink is None:
                return
            notes = compaction.get("pinned_notes")
            compacted_ids = compaction.get("compacted_observation_ids")
            if (
                not isinstance(notes, str)
                or not isinstance(compacted_ids, list)
                or any(not isinstance(item, str) or not item for item in compacted_ids)
            ):
                raise ValueError("model compaction accounting is invalid")
            self.event_sink(
                "investigation.compaction",
                {
                    **self._progress_snapshot(
                        decisions=decisions,
                        tool_calls=tool_calls,
                        physical=self._physical_count(decisions + 1),
                        command_calls=command_calls,
                    ),
                    "pinned_notes": notes,
                    "compacted_observation_ids": compacted_ids,
                },
            )

        if self.event_sink is not None and hasattr(self.planner, "request_event_sink"):
            self.planner.request_event_sink = model_request_started
        if self.event_sink is not None and hasattr(self.planner, "compaction_event_sink"):
            self.planner.compaction_event_sink = model_compaction_completed
        if hasattr(self.planner, "pinned_notes"):
            self.planner.pinned_notes = initial_compaction_notes
        if hasattr(self.planner, "compacted_observation_ids"):
            self.planner.compacted_observation_ids = set(initial_compacted_observation_ids)
        if hasattr(self.planner, "resume_request_kind"):
            self.planner.resume_request_kind = initial_resume_request_kind
        if hasattr(self.planner, "resume_transport_retries_used"):
            self.planner.resume_transport_retries_used = initial_transport_retries_used
        if hasattr(self.planner, "resume_schema_retries_used"):
            self.planner.resume_schema_retries_used = initial_schema_retries_used
        if hasattr(self.planner, "resume_physical_attempts_used"):
            self.planner.resume_physical_attempts_used = initial_physical_attempts_used

        while True:
            if self.cancel_requested is not None and self.cancel_requested():
                return self._finish(
                    run_id,
                    InvestigationVerdict.CANCELLED,
                    StopReason.CANCELLED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                )
            if self._active_budget_exhausted():
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.BUDGET_EXHAUSTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error="active-time budget exhausted",
                )
            if pending_decision is None and decisions >= self.budget.max_decisions:
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.BUDGET_EXHAUSTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                )
            if not self.trust.has_scope(resolved, AttestationScope.SOURCE_READ):
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.NOT_ATTESTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error="source_read was revoked before the next model decision",
                )
            # Stop before another decision if the planner has already made as many
            # real client requests as the physical budget allows. ``physical``
            # tracks actual requests (from the planner when it exposes a counter),
            # so retries inside a single decision are charged, not hidden.
            if pending_decision is None:
                physical = self._physical_count(decisions + 1)
                if physical >= self.budget.max_physical_requests:
                    return self._finish(
                        run_id,
                        InvestigationVerdict.INCOMPLETE,
                        StopReason.BUDGET_EXHAUSTED,
                        None,
                        catalog,
                        decisions,
                        tool_calls,
                        physical,
                    )
            if pending_decision is not None:
                decision = pending_decision
                pending_decision = None
            else:
                try:
                    decision = self.planner.decide(goal=goal, catalog=tuple(catalog))
                except Exception as exc:  # planner/model protocol failure
                    physical = self._physical_count(decisions + 1)
                    attestation_stop = isinstance(exc, PlannerAttestationError)
                    budget_stop = isinstance(exc, PlannerBudgetError)
                    return self._finish(
                        run_id,
                        (
                            InvestigationVerdict.INCOMPLETE
                            if attestation_stop or budget_stop
                            else InvestigationVerdict.FAILED
                        ),
                        (
                            StopReason.NOT_ATTESTED
                            if attestation_stop
                            else (
                                StopReason.BUDGET_EXHAUSTED
                                if budget_stop
                                else StopReason.PROTOCOL_FAILURE
                            )
                        ),
                        None,
                        catalog,
                        decisions,
                        tool_calls,
                        physical,
                        error=str(exc),
                    )
                physical = self._physical_count(decisions + 1)
                # A planner request that returned a decision consumed one logical
                # decision even when the active deadline expired during that request.
                # Count it before the deadline/type exits so the model call ledger and
                # report cannot disagree on an otherwise compliant budget stop.
                decisions += 1
                if self.event_sink is not None:
                    self.event_sink(
                        "investigation.decision",
                        {
                            **self._progress_snapshot(
                                decisions=decisions,
                                tool_calls=tool_calls,
                                physical=physical,
                                command_calls=command_calls,
                            ),
                            "decision_kind": (
                                "answer" if isinstance(decision, AgentAnswer) else "tool"
                            ),
                            "decision": (
                                asdict(decision)
                                if isinstance(decision, ToolCall | AgentAnswer)
                                else None
                            ),
                        },
                    )
            if self._active_budget_exhausted():
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.BUDGET_EXHAUSTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error="active-time budget exhausted",
                )
            if not isinstance(decision, ToolCall | AgentAnswer):
                return self._finish(
                    run_id,
                    InvestigationVerdict.FAILED,
                    StopReason.PROTOCOL_FAILURE,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error=f"planner returned an unsupported decision type: {type(decision)!r}",
                )
            if isinstance(decision, AgentAnswer):
                # Invalid references remain security/integrity failures even when
                # another answer field is malformed. Delay only the one-to-one
                # distinct-range rule until after structural validation so an
                # ordinary duplicate/mismatched finding remains a malformed answer.
                citation_error = _validate_citations(
                    decision,
                    tuple(catalog),
                    identity_for=evidence_identity,
                    enforce_distinct=False,
                )
                if citation_error is not None:
                    return self._finish(
                        run_id,
                        InvestigationVerdict.INCOMPLETE,
                        StopReason.UNSUPPORTED_CITATION,
                        decision,
                        catalog,
                        decisions,
                        tool_calls,
                        physical,
                        error=citation_error,
                    )
                structure_error = _validate_answer_structure(decision)
                if structure_error is not None:
                    return self._finish(
                        run_id,
                        InvestigationVerdict.INCOMPLETE,
                        StopReason.MALFORMED_ANSWER,
                        decision,
                        catalog,
                        decisions,
                        tool_calls,
                        physical,
                        error=structure_error,
                    )
                citation_error = _validate_citations(
                    decision,
                    tuple(catalog),
                    identity_for=evidence_identity,
                )
                if citation_error is not None:
                    return self._finish(
                        run_id,
                        InvestigationVerdict.INCOMPLETE,
                        StopReason.UNSUPPORTED_CITATION,
                        decision,
                        catalog,
                        decisions,
                        tool_calls,
                        physical,
                        error=citation_error,
                    )
                unresolved_uncertainty = (
                    not decision.issue_present
                    and _has_unresolved_negative_uncertainty(
                        decision,
                        tuple(catalog),
                        identity_for=evidence_identity,
                    )
                )
                if not decision.complete or unresolved_uncertainty:
                    return self._finish(
                        run_id,
                        InvestigationVerdict.INCOMPLETE,
                        StopReason.INCOMPLETE_EVIDENCE,
                        decision,
                        catalog,
                        decisions,
                        tool_calls,
                        physical,
                        error=(
                            "final answer declared itself incomplete"
                            if not decision.complete
                            else "unresolved evidence omissions cannot support a negative conclusion"
                        ),
                    )
                return self._finish(
                    run_id,
                    InvestigationVerdict.PASS,
                    StopReason.FINISHED,
                    decision,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                )

            # A tool call.
            if tool_calls >= self.budget.max_tool_calls:
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.BUDGET_EXHAUSTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                )
            if decision.tool == "run_command" and command_calls >= self.budget.max_command_calls:
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.BUDGET_EXHAUSTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error="command-call budget exhausted",
                )
            signature = _call_signature(decision)
            if signature in seen_calls:
                no_progress_repeats += 1
                if no_progress_repeats >= 2:
                    return self._finish(
                        run_id,
                        InvestigationVerdict.INCOMPLETE,
                        StopReason.NO_PROGRESS,
                        None,
                        catalog,
                        decisions,
                        tool_calls,
                        physical,
                    )
            else:
                no_progress_repeats = 0
                seen_calls.add(signature)
            tool_calls += 1
            # Re-check the attestation before every source-bearing read, so a
            # revocation mid-run stops further disclosure.
            if not self.trust.has_scope(resolved, AttestationScope.SOURCE_READ):
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.NOT_ATTESTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error="source_read was revoked during the investigation",
                )
            if decision.tool == "run_command" and not self.trust.has_scope(
                resolved, AttestationScope.CODE_EXECUTION
            ):
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.NOT_ATTESTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error="code_execution is not attested for this workspace",
                )
            try:
                if decision.tool == "run_command":
                    if self.command_executor is None:
                        raise PolicyViolationError("command tools are unavailable in this run")
                    if decision.based_on_observation_id is not None:
                        dependency = next(
                            (
                                item
                                for item in catalog
                                if item.observation_id == decision.based_on_observation_id
                            ),
                            None,
                        )
                        if (
                            dependency is None
                            or dependency.tool != "run_command"
                            or dependency.metadata.get("status") != "failed"
                        ):
                            raise PolicyViolationError(
                                "command recovery must reference an earlier failed command observation"
                            )
                    # Charge an attempted dispatch before crossing into the executor.
                    # Transport and policy failures still consume command capacity.
                    command_calls += 1
                    self._command_calls_used = command_calls
                    command_started = time.monotonic()
                    execution = self.command_executor.execute(
                        decision,
                        run_id=run_id,
                        active_deadline=active_deadline,
                    )
                    command_elapsed = max(0.0, time.monotonic() - command_started)
                    if not isinstance(execution, CommandExecution):
                        raise PolicyViolationError(
                            "command executor returned an unsupported execution result"
                        )
                    approval_wait = execution.approval_wait_seconds
                    if (
                        not math.isfinite(approval_wait)
                        or approval_wait < 0
                        or approval_wait > command_elapsed + 0.001
                    ):
                        raise PolicyViolationError(
                            "command executor returned invalid approval-wait accounting"
                        )
                    self._paused_seconds += approval_wait
                    active_deadline += approval_wait
                    reader = reader.with_active_deadline(active_deadline)
                    if hasattr(self.planner, "active_deadline"):
                        self.planner.active_deadline = active_deadline
                    observation = execution.observation
                    expected_command_path = f"command/{decision.command or ''}"
                    if (
                        observation.tool != "run_command"
                        or observation.path != expected_command_path
                        or observation.metadata.get("command_name") != decision.command
                    ):
                        raise PolicyViolationError(
                            "command executor returned an observation outside the requested tool"
                        )
                else:
                    observation = _dispatch(reader, decision)
            except StrictDecodeError as exc:
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.STRICT_DECODE_REFUSAL,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error=str(exc),
                )
            except PolicyViolationError as exc:
                # A security-policy violation is an immediate, gate-fatal refusal.
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.POLICY_VIOLATION,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error=str(exc),
                )
            except FsToolError as exc:
                message = str(exc)
                # Benign, retryable errors become observations so the planner adapts.
                observation = ToolObservation(
                    observation_id=f"obs_error_{tool_calls}",
                    tool=decision.tool,
                    path=(
                        "."
                        if decision.tool == "search_text"
                        else _canonical_request_path(decision.path)
                    ),
                    content_hash="",
                    text=f"[refused] {message}",
                    truncated=True,
                    incomplete=True,
                    metadata={
                        "refused": True,
                        "request_invalid": isinstance(exc, RequestValidationError),
                        "query": decision.query,
                        "glob": canonical_glob_scope(decision.glob),
                        "recursive": (
                            decision.tool == "list_files"
                            and glob_uses_recursive_listing(decision.glob)
                        ),
                    },
                )
            # A strict-decode refusal surfaced during a search also forces INCOMPLETE.
            if observation.metadata.get("decode_refused"):
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.STRICT_DECODE_REFUSAL,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error="search encountered a file that failed strict UTF-8 decoding",
                )
            observation_bytes = len(observation.text.encode("utf-8"))
            if self._observation_bytes_used + observation_bytes > self.budget.max_observation_bytes:
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.BUDGET_EXHAUSTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error="observation-byte budget exhausted",
                )
            self._observation_bytes_used += observation_bytes
            catalog.append(observation)
            source_identity = reader.evidence_identity(observation.observation_id)
            if source_identity is None:
                metadata_identity = observation.metadata.get("evidence_identity")
                source_identity = (
                    metadata_identity
                    if isinstance(metadata_identity, str) and metadata_identity
                    else None
                )
            if source_identity is not None:
                durable_identities[observation.observation_id] = source_identity
            if self.event_sink is not None:
                self.event_sink(
                    "investigation.observation",
                    {
                        "observation": asdict(observation),
                        "evidence_identity": source_identity,
                        **self._progress_snapshot(
                            decisions=decisions,
                            tool_calls=tool_calls,
                            physical=physical,
                            command_calls=command_calls,
                        ),
                    },
                )
            if self.cancel_requested is not None and self.cancel_requested():
                return self._finish(
                    run_id,
                    InvestigationVerdict.CANCELLED,
                    StopReason.CANCELLED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                )
            if self._active_budget_exhausted():
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE,
                    StopReason.BUDGET_EXHAUSTED,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error="active-time budget exhausted",
                )

    def _finish(
        self,
        run_id: str,
        verdict: InvestigationVerdict,
        stop_reason: StopReason,
        answer: AgentAnswer | None,
        catalog: list[ToolObservation],
        decisions: int,
        tool_calls: int,
        physical: int,
        *,
        error: str = "",
    ) -> InvestigationReport:
        report = InvestigationReport(
            run_id=run_id,
            verdict=verdict,
            stop_reason=stop_reason,
            answer=answer,
            catalog=tuple(catalog),
            decisions_used=decisions,
            tool_calls_used=tool_calls,
            physical_requests_used=physical,
            command_calls_used=getattr(self, "_command_calls_used", 0),
            completion_tokens_used=self._completion_reported(),
            completion_tokens_charged=self._completion_charged(),
            completion_tokens_requested=self._completion_requested(),
            observation_bytes_used=self._observation_bytes_used,
            active_seconds=self._active_seconds(),
            transport_retries=int(self._prior_usage.get("transport_retries", 0))
            + int(getattr(self.planner, "transport_retries", 0)),
            schema_retries=int(self._prior_usage.get("schema_retries", 0))
            + int(getattr(self.planner, "schema_retries", 0)),
            model_calls=self._model_calls(),
            error=error,
        )
        if self.event_sink is not None:
            self.event_sink(
                "investigation.result",
                {
                    "verdict": report.verdict.value,
                    "stop_reason": report.stop_reason.value,
                    "decisions_used": report.decisions_used,
                    "tool_calls_used": report.tool_calls_used,
                    "physical_requests_used": report.physical_requests_used,
                    "command_calls_used": report.command_calls_used,
                    "completion_tokens_charged": report.completion_tokens_charged,
                    "completion_tokens_used": report.completion_tokens_used,
                    "completion_tokens_requested": report.completion_tokens_requested,
                    "observation_bytes_used": report.observation_bytes_used,
                    "active_seconds": report.active_seconds,
                    "transport_retries": report.transport_retries,
                    "schema_retries": report.schema_retries,
                    "model_calls": [asdict(call) for call in report.model_calls],
                    "answer": asdict(report.answer) if report.answer is not None else None,
                    "error": report.error,
                },
            )
        return report


AnswerBuilder = Callable[[tuple[ToolObservation, ...]], AgentAnswer]


@dataclass
class ScriptedInvestigationPlanner:
    """A deterministic planner: a fixed script of tool calls, then an answer.

    This is the offline solver used by CI, fixtures, and the investigation
    benchmark. It walks its ``steps`` in order, then calls ``build_answer`` with
    the accumulated catalog to build a cited answer, so the same loop, budgets,
    and citation validation exercise the deterministic path exactly as a model
    would.
    """

    steps: tuple[ToolCall, ...]
    build_answer: AnswerBuilder
    _index: int = field(default=0, init=False)

    def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
        del goal
        if self._index < len(self.steps):
            call = self.steps[self._index]
            self._index += 1
            return call
        return self.build_answer(catalog)

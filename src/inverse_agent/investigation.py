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

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol

from inverse_agent.attestations import AttestationScope, ScopedTrustStore
from inverse_agent.fs_tools import (
    FsToolError,
    PolicyViolationError,
    RequestValidationError,
    StrictDecodeError,
    ToolObservation,
    WorkspaceReader,
)

__all__ = [
    "AgentAnswer",
    "AgentBudget",
    "Decision",
    "InvestigationLoop",
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
    FAILED = "failed"


class StopReason(StrEnum):
    FINISHED = "finished"
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
MAX_PHYSICAL_CEILING = 36


@dataclass(frozen=True)
class ToolCall:
    """A model decision to invoke one read tool with validated arguments."""

    tool: str
    path: str = "."
    query: str | None = None
    glob: str | None = None
    start_line: int = 1
    max_lines: int = 200


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
class AgentBudget:
    """Loop budgets. Decisions decompose as tool decisions plus answer/recovery."""

    max_decisions: int = 12
    max_tool_calls: int = 10
    max_physical_requests: int = 18

    def validate(self) -> None:
        if self.max_tool_calls > self.max_decisions:
            raise ValueError("tool-call budget cannot exceed the decision budget")
        if not 1 <= self.max_decisions <= MAX_DECISIONS_CEILING:
            raise ValueError(f"max_decisions must be between 1 and {MAX_DECISIONS_CEILING}")
        if not 1 <= self.max_tool_calls <= MAX_TOOL_CALLS_CEILING:
            raise ValueError(f"max_tool_calls must be between 1 and {MAX_TOOL_CALLS_CEILING}")
        if not 1 <= self.max_physical_requests <= MAX_PHYSICAL_CEILING:
            raise ValueError(f"max_physical_requests must be between 1 and {MAX_PHYSICAL_CEILING}")


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
    """Only a real, non-refusal ``read_file`` observation with content is citable.

    Pointer results (``list_files`` / ``search_text``) locate files but are not
    citable evidence: their "lines" are directory entries or ``path:line: match``
    snippets whose numbering is a result index, not a source line, so a citation
    against them would resolve to nothing meaningful. Requiring a ``read_file``
    observation keeps the grounding invariant honest — an answer must cite content
    from a file it actually read.
    """

    if observation.tool != "read_file":
        return False
    if observation.metadata.get("refused"):
        return False
    if not observation.content_hash:
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
        source_key: tuple[object, ...]
        if identity is not None:
            source_key = ("file", identity)
        else:
            source_key = ("path", _canonical_request_path(observation.path))
        range_key = (source_key, citation.start_line, citation.end_line)
        if range_key in physical_ranges:
            return "each finding must use a distinct physical citation range"
        physical_ranges.add(range_key)
    return None


def _has_unresolved_negative_uncertainty(
    answer: AgentAnswer,
    catalog: tuple[ToolObservation, ...],
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
        cited_path = _canonical_request_path(cited.path)
        folded_path = cited_path.casefold()
        latest_variants: dict[str, ToolObservation] = {}
        for observation in catalog:
            if observation.tool != "read_file" or observation.metadata.get("request_invalid"):
                continue
            observed_path = _canonical_request_path(observation.path)
            if observed_path.casefold() != folded_path:
                continue
            latest_variants[observed_path] = observation
        if any(
            observation.incomplete or bool(observation.metadata.get("refused"))
            for observation in latest_variants.values()
        ):
            return True
    return False


class InvestigationLoop:
    """Runs one durable investigation to a verdict."""

    def __init__(
        self,
        *,
        planner: InvestigationPlanner,
        trust: ScopedTrustStore,
        budget: AgentBudget | None = None,
    ) -> None:
        self.planner = planner
        self.trust = trust
        self.budget = budget or AgentBudget()
        self.budget.validate()
        # If the planner makes real client requests, bound its total to the
        # physical-request budget so retries cannot exceed it, and count actual
        # requests rather than one-per-decision.
        if hasattr(planner, "max_total_requests"):
            planner.max_total_requests = self.budget.max_physical_requests

    def _physical_count(self, fallback: int) -> int:
        """Actual client requests so far, from the planner when it tracks them."""

        made = getattr(self.planner, "requests_made", None)
        return int(made) if isinstance(made, int) else fallback

    def run(self, *, run_id: str, goal: str, workspace: Path) -> InvestigationReport:
        resolved = workspace.resolve()
        if not self.trust.has_scope(resolved, AttestationScope.SOURCE_READ):
            return InvestigationReport(
                run_id=run_id,
                verdict=InvestigationVerdict.INCOMPLETE,
                stop_reason=StopReason.NOT_ATTESTED,
                answer=None,
                catalog=(),
                decisions_used=0,
                tool_calls_used=0,
                physical_requests_used=0,
                error="workspace is not attested for source_read",
            )
        reader = WorkspaceReader.open(resolved)
        catalog: list[ToolObservation] = []
        decisions = 0
        tool_calls = 0
        physical = 0
        seen_calls: set[tuple[object, ...]] = set()
        no_progress_repeats = 0

        while True:
            if decisions >= self.budget.max_decisions:
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
            # Stop before another decision if the planner has already made as many
            # real client requests as the physical budget allows. ``physical``
            # tracks actual requests (from the planner when it exposes a counter),
            # so retries inside a single decision are charged, not hidden.
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
            try:
                decision = self.planner.decide(goal=goal, catalog=tuple(catalog))
            except Exception as exc:  # planner/model protocol failure
                physical = self._physical_count(decisions + 1)
                budget_stop = "request budget exhausted" in str(exc)
                return self._finish(
                    run_id,
                    InvestigationVerdict.INCOMPLETE if budget_stop else InvestigationVerdict.FAILED,
                    StopReason.BUDGET_EXHAUSTED if budget_stop else StopReason.PROTOCOL_FAILURE,
                    None,
                    catalog,
                    decisions,
                    tool_calls,
                    physical,
                    error=str(exc),
                )
            physical = self._physical_count(decisions + 1)
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
            decisions += 1

            if isinstance(decision, AgentAnswer):
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
                    identity_for=reader.evidence_identity,
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
            try:
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
                        "glob": decision.glob,
                        "recursive": (
                            decision.tool == "list_files"
                            and decision.glob is not None
                            and "**" in decision.glob
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
            catalog.append(observation)

    @staticmethod
    def _finish(
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
        return InvestigationReport(
            run_id=run_id,
            verdict=verdict,
            stop_reason=stop_reason,
            answer=answer,
            catalog=tuple(catalog),
            decisions_used=decisions,
            tool_calls_used=tool_calls,
            physical_requests_used=physical,
            error=error,
        )


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

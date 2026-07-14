"""Model-backed investigation planner.

Drives the read-only investigation loop with an OpenAI-compatible model endpoint
(e.g. a local GPT-OSS-20B served by LM Studio). Each decision is a single strict
JSON object: either one read-tool call or a final, cited answer. The catalog of
prior observations is rendered compactly into the prompt so citations resolve
against real returned content. A bounded per-decision retry (one transport, one
schema) is applied; anything beyond that raises and the loop records a terminal
protocol failure.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from inverse_agent.fs_tools import (
    CHARS_PER_TOKEN,
    FILE_MAX_BYTES,
    PATH_MAX_CHARS,
    READ_MAX_LINES,
    READ_MAX_TOKENS,
)
from inverse_agent.investigation import (
    AgentAnswer,
    Decision,
    ModelCallRecord,
    SourceCitation,
    ToolCall,
    ToolObservation,
    citation_intersects_redaction,
    line_body,
)
from inverse_agent.planner import (
    MAX_MODEL_COMPLETION_TOKENS,
    ModelResponseMetadata,
    PlannerAttestationError,
    PlannerBudgetError,
    PlannerError,
    PlannerProtocolError,
    PlannerTransportError,
)

__all__ = ["ModelInvestigationPlanner", "SupportsStructuredJson", "parse_decision"]

# Direct unit callers get the legacy generous cap. Production planning derives
# the actual catalog budget from the endpoint's calibrated context capacity.
CATALOG_TOKEN_BUDGET = 20_000
CATALOG_LINES_PER_OBS = 60
CONTEXT_CALIBRATION_POINTS = (16_384, 24_576, 32_768, 49_152)
MIN_COMPLETION_ALLOWANCE = 1_024
MAX_COMPLETION_BUDGET = 49_152
MAX_LOGICAL_DECISIONS = 24
MAX_PHYSICAL_REQUESTS = 36
PROMPT_TRANSPORT_OVERHEAD_TOKENS = 512
DEFAULT_ESTIMATOR_BYTES_PER_TOKEN = 2.0

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["read_file", "list_files", "search_text", "final_answer"],
        },
        "path": {"type": "string"},
        "query": {"type": "string"},
        "glob": {"type": "string"},
        "based_on_observation_id": {"type": "string"},
        "start_line": {"type": "integer", "minimum": 1},
        "max_lines": {"type": "integer", "minimum": 1, "maximum": 200},
        "summary": {"type": "string"},
        "condition_holds": {"type": "boolean"},
        "complete": {"type": "boolean"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "next_actions": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "observation_id": {"type": "string"},
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["observation_id", "path", "start_line", "end_line"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "action",
        "path",
        "query",
        "glob",
        "based_on_observation_id",
        "start_line",
        "max_lines",
        "summary",
        "condition_holds",
        "complete",
        "findings",
        "next_actions",
        "citations",
    ],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You are a read-only code investigator. Return exactly ONE JSON action per "
    "turn. Actions: read_file (set path), list_files (set path, default '.'), "
    "search_text (set query), final_answer.\n"
    "Procedure: 1) list_files or search_text to find the file. 2) read_file it. "
    "3) final_answer. Always read the relevant file before concluding - never "
    "answer without having read evidence, and never conclude the condition is "
    "absent without inspecting the code.\n"
    "Never invent a path: only use a path that appears in an observation or that "
    "you have already read. If a read_file observation already shows the answer, "
    "send final_answer now - do not repeat a call.\n"
    "Citations: cite a read_file or explicitly CITABLE command observation only. "
    "Copy its observation_id exactly "
    "from the id= field, use its path, and set start_line/end_line to the numbers "
    "shown before the colon (a line '12: foo' is line 12). Every finding needs a "
    "distinct citation range to a line you actually saw; combine findings when "
    "the same range would otherwise be repeated.\n"
    "In final_answer set condition_holds=true when the code confirms the "
    "condition or fact the goal asks about (e.g. the component IS exported, the "
    "entrypoint DOES exist, the bug IS present) and false only if the code shows "
    "it genuinely does not hold; give a non-empty summary, at least one finding, "
    "and at least one recommended next action. Provide exactly one citation for "
    "each finding in the same order.\n"
    "Observation completeness: headers explicitly show truncated/incomplete flags. "
    "A bounded read_file window may support a localized claim when every cited line "
    "is visible. Never infer broad absence from an incomplete or truncated list_files "
    "or search_text result, or from an incomplete read of a cited path. Retry the same "
    "catalog request successfully to replace earlier uncertainty. If the final answer "
    "still depends on missing content, set complete=false. Set complete=true on tool "
    "actions.\n"
    'Fill unused fields with "" or [].'
)

_COMMAND_PROMPT_APPENDIX = (
    "\nThis run also permits run_command. Set path to one exact name from "
    "available_commands. Every command requires a fresh human approval. A failed "
    "command is an observation: diagnose it and replan instead of repeating it. "
    "When selecting a different command to recover from a failed command, set "
    "based_on_observation_id to that failed command's exact observation ID; "
    "otherwise set it to an empty string."
)


def _schema_for_commands(allowed_commands: tuple[str, ...]) -> dict[str, Any]:
    if not allowed_commands:
        return DECISION_SCHEMA
    action = dict(DECISION_SCHEMA["properties"]["action"])
    action["enum"] = [
        "read_file",
        "list_files",
        "search_text",
        "run_command",
        "final_answer",
    ]
    properties = dict(DECISION_SCHEMA["properties"])
    properties["action"] = action
    return {**DECISION_SCHEMA, "properties": properties}


class SupportsStructuredJson(Protocol):
    def complete_structured_json(
        self,
        *,
        system: str,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        max_tokens: int = ...,
        timeout_seconds: float | None = ...,
    ) -> dict[str, Any]: ...


def _render_block(obs: ToolObservation) -> tuple[str, bool]:
    """Render one observation to a prompt block, returning (block, is_citable_read)."""

    raw_redacted = obs.metadata.get("redacted_lines", ())
    redacted_lines = (
        {item for item in raw_redacted if isinstance(item, int)}
        if isinstance(raw_redacted, list | tuple)
        else set()
    )
    has_citable_line = any(
        obs.start_line + offset not in redacted_lines and line_body(line).strip()
        for offset, line in enumerate(obs.lines)
    )
    if obs.metadata.get("refused"):
        citable = "not citable"
    elif obs.metadata.get("binary"):
        citable = "not citable (binary)"
    elif obs.tool == "read_file" and not has_citable_line:
        citable = "not citable (blank or redacted)"
    elif obs.tool == "read_file" or obs.metadata.get("citable_command"):
        citable = "CITABLE - cite this observation_id with its N: line numbers"
    else:
        citable = "pointer only - read_file a listed path to cite it"
    status = (
        f"truncated={str(obs.truncated).lower()} "
        f"incomplete={str(obs.incomplete).lower()} "
        f"redacted={str(obs.redacted).lower()}"
    )
    header = f"id={obs.observation_id} tool={obs.tool} path={obs.path} status[{status}] ({citable})"
    if redacted_lines:
        header += f" non_citable_redacted_lines={_render_redacted_lines(redacted_lines)}"
    if obs.metadata.get("refused"):
        return f"{header}\n  REFUSED: {obs.text}", False
    if obs.metadata.get("binary"):
        return f"{header}\n  (binary file)", False
    # A read_file observation is already bounded (<=200 lines / ~3k tokens), so
    # show ALL of its lines: the model may only cite content it was actually shown.
    # Pointer results (list/search) stay capped.
    limit = (
        len(obs.lines)
        if obs.tool == "read_file" or obs.metadata.get("citable_command")
        else CATALOG_LINES_PER_OBS
    )
    shown = obs.lines[:limit]
    fully_rendered = limit >= len(obs.lines)
    rendered_lines = []
    for line in shown:
        rendered_lines.append(f"  {line}")
    body = "\n".join(rendered_lines) or "  (no matching content)"
    if obs.incomplete or obs.truncated:
        body = (
            "  WARNING: this result is incomplete; omitted content may change a "
            f"negative conclusion.\n{body}"
        )
    is_citable_read = (
        (obs.tool == "read_file" or bool(obs.metadata.get("citable_command")))
        and bool(obs.content_hash)
        and fully_rendered
        and has_citable_line
        and (obs.tool == "read_file" or (not obs.incomplete and not obs.truncated))
    )
    return f"{header}\n{body}", is_citable_read


def _encoded_string_tokens(value: str, *, bytes_per_token: float) -> int:
    """Estimate tokens from the exact JSON-encoded observation representation."""

    encoded_bytes = len(json.dumps(value, ensure_ascii=True).encode("utf-8"))
    return math.ceil(encoded_bytes / bytes_per_token)


def _render_redacted_lines(lines: set[int]) -> str:
    """Render at most 200 non-citable line numbers with a simple length bound."""

    return ",".join(str(line) for line in sorted(lines))


def _maximum_read_probe() -> ToolObservation:
    """Build a conservative maximum legal read observation for calibration."""

    serialized_budget = READ_MAX_TOKENS * CHARS_PER_TOKEN
    line_break_bytes = 2 * (READ_MAX_LINES - 1)
    # BEL is accepted as text by the read tier and expands to six ASCII bytes in
    # JSON. Filling with it models the worst per-source-byte prompt expansion.
    content_budget = serialized_budget - 2 - line_break_bytes
    bel_count, ascii_remainder = divmod(content_budget, 6)
    payload = "\a" * bel_count + "x" * ascii_remainder
    width, remainder = divmod(len(payload), READ_MAX_LINES)
    start_line = FILE_MAX_BYTES - READ_MAX_LINES + 2
    line_contents: list[str] = []
    cursor = 0
    for offset in range(READ_MAX_LINES):
        size = width + (1 if offset < remainder else 0)
        line_contents.append(payload[cursor : cursor + size])
        cursor += size
    source_text = "\n".join(line_contents)
    lines = tuple(
        f"{start_line + offset}: {content}" for offset, content in enumerate(line_contents)
    )
    return ToolObservation(
        observation_id="obs_0123456789abcdef",
        tool="read_file",
        # One non-BMP code point is four UTF-8 bytes and twelve bytes under
        # ensure_ascii JSON escaping, the maximum expansion of accepted path
        # text. Component limits can only make a real path smaller.
        path="\U00010000" * PATH_MAX_CHARS,
        content_hash="h" * 64,
        text=source_text,
        lines=lines,
        start_line=start_line,
        truncated=True,
        incomplete=True,
        redacted=True,
        # Leave one visible line so the maximum observation remains citable.
        # Explicit line-number rendering makes all 199 redacted lines the exact
        # maximum metadata overhead, independent of their grouping pattern.
        metadata={"redacted_lines": tuple(range(start_line, start_line + READ_MAX_LINES - 1))},
    )


def _maximum_read_probe_tokens(*, bytes_per_token: float) -> int:
    """Token estimate for the largest JSON-bounded read the tool can emit."""

    probe = _maximum_read_probe()
    block, _citable = _render_block(probe)
    return _encoded_string_tokens(block, bytes_per_token=bytes_per_token)


def _render_catalog(
    catalog: tuple[ToolObservation, ...],
    *,
    token_budget: int = CATALOG_TOKEN_BUDGET,
    estimator_bytes_per_token: float = DEFAULT_ESTIMATOR_BYTES_PER_TOKEN,
) -> tuple[str, frozenset[str]]:
    """Render the catalog and return (prompt text, ids of fully-rendered reads).

    A read_file observation only becomes citable when all of its lines were
    actually placed in the prompt; an observation omitted for space is excluded,
    so the model can never be led to cite content it was not shown. Selection
    guarantees the most-recent citable read is always included (reads are the only
    citable evidence, so a burst of large pointer results must never crowd out the
    latest read), then fills the remaining budget with other observations
    newest-first; dropped context is always the oldest.
    """

    if token_budget < 0:
        raise ValueError("catalog token budget cannot be negative")
    if not math.isfinite(estimator_bytes_per_token) or estimator_bytes_per_token <= 0:
        raise ValueError("estimator bytes per token must be positive and finite")
    if not catalog:
        return "(no observations yet)", frozenset()

    blocks = [_render_block(obs) for obs in catalog]
    newest_read_index = next(
        (i for i in range(len(catalog) - 1, -1, -1) if blocks[i][1]),
        None,
    )

    marker = "(earlier observations omitted for space)"

    def render(indices: set[int]) -> str:
        text_blocks = [blocks[i][0] for i in sorted(indices)]
        if len(indices) < len(catalog):
            text_blocks.insert(0, marker)
        return "\n".join(text_blocks)

    selected: set[int] = set()
    if newest_read_index is not None:
        # Preserve the latest citable read only when it fits the calibrated
        # history allowance. Oversized evidence is omitted and therefore cannot
        # become a repair/validation target.
        trial = {newest_read_index}
        if (
            _encoded_string_tokens(render(trial), bytes_per_token=estimator_bytes_per_token)
            <= token_budget
        ):
            selected = trial
    for index in range(len(catalog) - 1, -1, -1):
        if index in selected:
            continue
        trial = {*selected, index}
        if (
            _encoded_string_tokens(render(trial), bytes_per_token=estimator_bytes_per_token)
            <= token_budget
        ):
            selected = trial

    omitted = len(selected) < len(catalog)
    rendered_read_ids = {catalog[i].observation_id for i in selected if blocks[i][1]}
    if selected:
        rendered = render(selected)
    elif omitted and (
        _encoded_string_tokens(marker, bytes_per_token=estimator_bytes_per_token) <= token_budget
    ):
        rendered = marker
    else:
        rendered = ""
    return rendered, frozenset(rendered_read_ids)


def _repair_citations(
    answer: AgentAnswer,
    catalog: tuple[ToolObservation, ...],
    rendered_ids: frozenset[str],
) -> AgentAnswer:
    """Remap each citation to a read observation that was actually shown.

    Small models frequently mis-copy the opaque observation_id or cite a
    search/list pointer using a file line number. This repair rebinds a citation
    only to a read_file observation whose id was rendered in full to the model
    (``rendered_ids``), of the SAME path, whose returned window contains the
    cited line - content the model genuinely saw. It never invents evidence,
    never widens beyond the returned window, and never binds to an observation
    the model was not shown; the loop validator and the benchmark scorer still
    independently confirm the cited line resolves to real content.
    """

    reads = [
        obs
        for obs in catalog
        if (obs.tool == "read_file" or obs.metadata.get("citable_command"))
        and obs.content_hash
        and obs.observation_id in rendered_ids
    ]
    by_id = {obs.observation_id: obs for obs in reads}

    def rebind(citation: SourceCitation) -> SourceCitation:
        if citation.start_line < 1 or citation.end_line < citation.start_line:
            return citation
        existing = by_id.get(citation.observation_id)
        if (
            existing is not None
            and existing.path == citation.path
            and not citation_intersects_redaction(existing, citation)
        ):
            return citation
        for obs in reads:
            if obs.path != citation.path:
                continue
            last = obs.start_line + len(obs.lines) - 1
            repaired = SourceCitation(
                observation_id=obs.observation_id,
                path=obs.path,
                start_line=citation.start_line,
                end_line=min(citation.end_line, last),
                note=citation.note,
            )
            if obs.start_line <= citation.start_line <= last and not citation_intersects_redaction(
                obs, repaired
            ):
                return repaired
        return citation

    return AgentAnswer(
        summary=answer.summary,
        findings=answer.findings,
        next_actions=answer.next_actions,
        citations=tuple(rebind(citation) for citation in answer.citations),
        complete=answer.complete,
        issue_present=answer.issue_present,
    )


def _coerce_optional(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _normalize_id(value: object) -> str:
    """Strip stray brackets/whitespace a model may copy around an observation id."""

    return str(value).strip().strip("[]").strip()


def _normalize_path(value: object) -> str:
    """Coerce a model path to workspace-relative form.

    Strips surrounding whitespace and a leading slash/backslash (a model that
    writes '/src/x' means the workspace-root-relative 'src/x'). Traversal and
    escape are still rejected by the read tier.
    """

    text = str(value or ".").strip().replace("\\", "/")
    text = text.lstrip("/")
    return text or "."


def parse_decision(payload: Mapping[str, Any]) -> Decision:
    action = payload.get("action")
    if action == "final_answer":
        citations = tuple(
            SourceCitation(
                observation_id=_normalize_id(item["observation_id"]),
                path=_normalize_path(item["path"]),
                start_line=int(item["start_line"]),
                end_line=int(item["end_line"]),
            )
            for item in payload.get("citations", [])
            if isinstance(item, Mapping)
        )
        complete = payload["complete"]
        condition_holds = payload["condition_holds"]
        if type(complete) is not bool or type(condition_holds) is not bool:
            raise TypeError("complete and condition_holds must be JSON booleans")
        return AgentAnswer(
            summary=str(payload.get("summary", "")),
            findings=tuple(str(f) for f in payload.get("findings", [])),
            next_actions=tuple(str(a) for a in payload.get("next_actions", [])),
            citations=citations,
            complete=complete,
            issue_present=condition_holds,
        )
    if action in {"read_file", "list_files", "search_text"}:
        return ToolCall(
            tool=str(action),
            path=_normalize_path(payload.get("path")),
            query=_coerce_optional(payload.get("query")),
            glob=_coerce_optional(payload.get("glob")),
            start_line=max(1, int(payload.get("start_line") or 1)),
            max_lines=min(200, max(1, int(payload.get("max_lines") or 200))),
        )
    if action == "run_command":
        command = str(payload.get("path") or "").strip()
        if not command:
            raise ValueError("run_command requires an available command name in path")
        raw_dependency = _coerce_optional(payload.get("based_on_observation_id"))
        return ToolCall(
            tool="run_command",
            command=command,
            based_on_observation_id=(
                _normalize_id(raw_dependency) if raw_dependency is not None else None
            ),
        )
    raise ValueError(f"model returned an unsupported action: {action!r}")


@dataclass
class ModelInvestigationPlanner:
    """An investigation planner backed by an OpenAI-compatible model client.

    Each ``decide`` makes one primary request plus at most one transport retry and
    at most one schema retry (a repeated failure of the same class is not
    retried). Every client request is counted in ``requests_made`` and bounded by
    ``max_total_requests`` across the whole run, so retries cannot inflate the
    real request count past the budget.
    """

    client: SupportsStructuredJson
    goal_hint: str = ""
    allowed_commands: tuple[str, ...] = ()
    max_transport_retries: int = 1
    max_schema_retries: int = 1
    max_auto_reads: int = 3
    max_nudges: int = 3
    max_total_requests: int = 18
    max_logical_decisions: int = 12
    max_completion_tokens: int = 24_576
    context_tokens: int = 24_576
    estimator_bytes_per_token: float = DEFAULT_ESTIMATOR_BYTES_PER_TOKEN
    max_estimator_error_tokens: int = 0
    requests_made: int = field(default=0, init=False)
    completion_tokens_requested: int = field(default=0, init=False)
    completion_tokens_charged: int = field(default=0, init=False)
    completion_tokens_reported: int = field(default=0, init=False)
    completion_allowances: list[int] = field(default_factory=list, init=False)
    model_calls: list[ModelCallRecord] = field(default_factory=list, init=False)
    transport_retries: int = field(default=0, init=False)
    schema_retries: int = field(default=0, init=False)
    active_deadline: float | None = field(default=None, init=False)
    source_read_guard: Callable[[], bool] | None = field(default=None, init=False, repr=False)
    request_event_sink: Callable[[dict[str, int | float | str | None]], None] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    resume_transport_retries_used: int = field(default=0, init=False, repr=False)
    resume_schema_retries_used: int = field(default=0, init=False, repr=False)
    resume_physical_attempts_used: int = field(default=0, init=False, repr=False)
    _turn: int = field(default=0, init=False)
    _auto_reads: int = field(default=0, init=False)
    _nudges: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if len(set(self.allowed_commands)) != len(self.allowed_commands) or any(
            not command or len(command) > 120 for command in self.allowed_commands
        ):
            raise ValueError("allowed_commands must contain unique non-empty names")
        if not 0 <= self.max_transport_retries <= 1:
            raise ValueError("max_transport_retries must be 0 or 1")
        if not 0 <= self.max_schema_retries <= 1:
            raise ValueError("max_schema_retries must be 0 or 1")
        if not 0 <= self.max_auto_reads <= 3:
            raise ValueError("max_auto_reads must be between 0 and 3")
        if not 0 <= self.max_nudges <= 3:
            raise ValueError("max_nudges must be between 0 and 3")
        if not 1 <= self.max_total_requests <= MAX_PHYSICAL_REQUESTS:
            raise ValueError(f"max_total_requests must be between 1 and {MAX_PHYSICAL_REQUESTS}")
        if not 1 <= self.max_logical_decisions <= MAX_LOGICAL_DECISIONS:
            raise ValueError(f"max_logical_decisions must be between 1 and {MAX_LOGICAL_DECISIONS}")
        minimum_completion = self.max_logical_decisions * MIN_COMPLETION_ALLOWANCE
        if not minimum_completion <= self.max_completion_tokens <= MAX_COMPLETION_BUDGET:
            raise ValueError(
                "max_completion_tokens must preserve at least 1024 tokens per decision "
                f"and not exceed {MAX_COMPLETION_BUDGET}"
            )
        if self.context_tokens not in CONTEXT_CALIBRATION_POINTS:
            raise ValueError(
                "context_tokens must be one of the measured calibration points: "
                "16384, 24576, 32768, or 49152"
            )
        if not 1.0 <= self.estimator_bytes_per_token <= 4.0:
            raise ValueError("estimator_bytes_per_token must be between 1.0 and 4.0")
        if (
            _maximum_read_probe_tokens(bytes_per_token=self.estimator_bytes_per_token)
            > self.context_tokens // 2
        ):
            raise ValueError(
                "context/estimator pair cannot render one maximum legal read; "
                "select a larger calibrated context"
            )
        if not 0 <= self.max_estimator_error_tokens <= self.context_tokens:
            raise ValueError("max_estimator_error_tokens is outside the context range")

    def _nudge_if_ungrounded(
        self, answer: AgentAnswer, catalog: tuple[ToolObservation, ...]
    ) -> ToolCall | None:
        """Redirect an ungrounded conclusion back into investigation.

        A small model sometimes concludes without reading the relevant file, or
        cites a list/search pointer instead of a read. Rather than accept a
        conclusion no read observation supports, nudge it to keep investigating
        (list the workspace root) so it can find and read the evidence. Bounded by
        ``max_nudges``; fires only while no citation resolves to a real, in-range
        read observation.
        """

        if self._nudges >= self.max_nudges:
            return None
        reads = [
            obs
            for obs in catalog
            if (obs.tool == "read_file" or obs.metadata.get("citable_command")) and obs.content_hash
        ]
        for citation in answer.citations:
            for obs in reads:
                if obs.path != citation.path:
                    continue
                last = obs.start_line + len(obs.lines) - 1
                if obs.start_line <= citation.start_line <= last:
                    # A read of this path covers the cited line; even if the id was
                    # mis-copied, repair will bind it. Do not nudge.
                    return None
        self._nudges += 1
        return ToolCall(tool="list_files", path=".")

    def _auto_read(
        self, answer: AgentAnswer, catalog: tuple[ToolObservation, ...]
    ) -> ToolCall | None:
        """If the answer cites a range not yet read, fetch it so it can be validated.

        The model often knows which file holds the evidence but answers before
        reading the cited line. We read (bounded by ``max_auto_reads``) starting at
        the first cited line no existing observation covers, and let the model cite
        it next turn. This never fabricates evidence: the file is genuinely read,
        and a nonexistent path simply refuses.
        """

        if self._auto_reads >= self.max_auto_reads:
            return None
        covered: dict[str, list[tuple[int, int]]] = {}
        for obs in catalog:
            if (
                obs.tool == "read_file" or obs.metadata.get("citable_command")
            ) and obs.content_hash:
                last = obs.start_line + len(obs.lines) - 1
                covered.setdefault(obs.path, []).append((obs.start_line, last))
        for citation in answer.citations:
            if not citation.path:
                continue
            spans = covered.get(citation.path, [])
            if any(lo <= citation.start_line <= hi for lo, hi in spans):
                continue
            self._auto_reads += 1
            return ToolCall(tool="read_file", path=citation.path, start_line=citation.start_line)
        return None

    def _completion_allowance(self) -> int:
        remaining_budget = self.max_completion_tokens - self.completion_tokens_charged
        remaining_decisions = max(1, self.max_logical_decisions - self._turn + 1)
        allowance = min(MAX_MODEL_COMPLETION_TOKENS, remaining_budget // remaining_decisions)
        if allowance < MIN_COMPLETION_ALLOWANCE:
            raise PlannerBudgetError("model completion-token budget exhausted")
        return allowance

    def _system_prompt(self) -> str:
        return _SYSTEM_PROMPT + (_COMMAND_PROMPT_APPENDIX if self.allowed_commands else "")

    def _decision_schema(self) -> dict[str, Any]:
        return _schema_for_commands(self.allowed_commands)

    def _prompt_token_bound(self, prompt: str) -> int:
        encoded_bytes = (
            len(self._system_prompt().encode("utf-8"))
            + len(json.dumps(self._decision_schema(), ensure_ascii=True).encode("utf-8"))
            + len(prompt.encode("utf-8"))
        )
        estimated = math.ceil(encoded_bytes / self.estimator_bytes_per_token)
        return estimated + PROMPT_TRANSPORT_OVERHEAD_TOKENS

    def _history_token_budget(self, *, goal: str, completion_reserve: int) -> int:
        empty_prompt = self._build_prompt(goal=goal, observations="")
        non_observation_tokens = self._prompt_token_bound(empty_prompt)
        safety_margin = max(
            (self.context_tokens + 9) // 10,
            2 * self.max_estimator_error_tokens,
        )
        return max(
            0,
            min(
                self.context_tokens // 2,
                self.context_tokens - completion_reserve - non_observation_tokens - safety_margin,
            ),
        )

    def _build_prompt(self, *, goal: str, observations: str) -> str:
        return json.dumps(
            {
                "goal": goal,
                "hint": self.goal_hint,
                "available_commands": list(self.allowed_commands),
                "turn": self._turn,
                "observations": observations,
                "instructions": (
                    "Return one action. If you have enough evidence, return "
                    "final_answer with citations; otherwise read, search, or select one "
                    "available command."
                ),
            },
            ensure_ascii=True,
        )

    def _reconcile_response_metadata(
        self,
        *,
        allowance: int,
        prompt: str,
    ) -> tuple[int, int | None, int | None, str | None]:
        metadata = getattr(self.client, "last_response_metadata", None)
        if not isinstance(metadata, ModelResponseMetadata):
            return allowance, None, None, None
        reported_completion = metadata.completion_tokens
        charged = allowance
        if reported_completion is not None:
            charged = reported_completion
            self.completion_tokens_charged -= allowance - charged
            self.completion_tokens_reported += reported_completion
        if metadata.prompt_tokens is not None:
            estimated_prompt = self._prompt_token_bound(prompt)
            self.max_estimator_error_tokens = max(
                self.max_estimator_error_tokens,
                metadata.prompt_tokens - estimated_prompt,
            )
        return charged, metadata.prompt_tokens, reported_completion, metadata.model

    def _request(
        self,
        prompt: str,
        *,
        retry_kind: str | None,
        transport_retries_used: int,
        schema_retries_used: int,
        physical_attempts_used: int,
    ) -> dict[str, Any]:
        if self.source_read_guard is not None and not self.source_read_guard():
            raise PlannerAttestationError("source_read was revoked before model request")
        if self.requests_made >= self.max_total_requests:
            raise PlannerBudgetError("model request budget exhausted")
        timeout_seconds: float | None = None
        if self.active_deadline is not None:
            timeout_seconds = self.active_deadline - time.monotonic()
            if timeout_seconds <= 0:
                raise PlannerBudgetError("active-time budget exhausted")
        allowance = self._completion_allowance()
        self.requests_made += 1
        # Charge before transport so failed and retried requests cannot receive
        # free completion capacity when an endpoint omits usage.
        self.completion_tokens_charged += allowance
        self.completion_tokens_requested += allowance
        self.completion_allowances.append(allowance)
        if self.request_event_sink is not None:
            self.request_event_sink(
                {
                    "request_index": self.requests_made,
                    "logical_decision": self._turn,
                    "requested_completion_tokens": allowance,
                    "charged_completion_tokens": allowance,
                    "started_at": time.time(),
                    "retry_kind": retry_kind,
                    "transport_retries_used": transport_retries_used,
                    "schema_retries_used": schema_retries_used,
                    "physical_attempts_used": physical_attempts_used,
                }
            )
        started_at = time.monotonic()
        try:
            payload = self.client.complete_structured_json(
                system=self._system_prompt(),
                prompt=prompt,
                schema_name="investigation_decision",
                schema=self._decision_schema(),
                max_tokens=allowance,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            if isinstance(exc, PlannerTransportError):
                outcome = "transport_error"
            elif isinstance(exc, PlannerProtocolError):
                outcome = "protocol_error"
            elif isinstance(exc, PlannerError):
                outcome = "planner_error"
            else:
                outcome = "client_error"
            charged, reported_prompt, reported_completion, reported_model = (
                self._reconcile_response_metadata(allowance=allowance, prompt=prompt)
            )
            self.model_calls.append(
                ModelCallRecord(
                    request_index=self.requests_made,
                    logical_decision=self._turn,
                    requested_completion_tokens=allowance,
                    charged_completion_tokens=charged,
                    reported_prompt_tokens=reported_prompt,
                    reported_completion_tokens=reported_completion,
                    reported_model=reported_model,
                    latency_seconds=max(0.0, time.monotonic() - started_at),
                    outcome=outcome,
                )
            )
            raise
        charged, reported_prompt, reported_completion, reported_model = (
            self._reconcile_response_metadata(allowance=allowance, prompt=prompt)
        )
        self.model_calls.append(
            ModelCallRecord(
                request_index=self.requests_made,
                logical_decision=self._turn,
                requested_completion_tokens=allowance,
                charged_completion_tokens=charged,
                reported_prompt_tokens=reported_prompt,
                reported_completion_tokens=reported_completion,
                reported_model=reported_model,
                latency_seconds=max(0.0, time.monotonic() - started_at),
                outcome="success",
            )
        )
        return payload

    def _mark_last_call_schema_error(self) -> None:
        if self.model_calls and self.model_calls[-1].outcome == "success":
            self.model_calls[-1] = replace(self.model_calls[-1], outcome="schema_error")

    def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
        self._turn += 1
        completion_reserve = self._completion_allowance()
        history_budget = self._history_token_budget(
            goal=goal,
            completion_reserve=completion_reserve,
        )
        observations, rendered_ids = _render_catalog(
            catalog,
            token_budget=history_budget,
            estimator_bytes_per_token=self.estimator_bytes_per_token,
        )
        prompt = self._build_prompt(goal=goal, observations=observations)
        safety_margin = max(
            (self.context_tokens + 9) // 10,
            2 * self.max_estimator_error_tokens,
        )
        if (
            self._prompt_token_bound(prompt) + completion_reserve + safety_margin
            > self.context_tokens
        ):
            raise PlannerBudgetError("model context budget exhausted")
        transport_used = self.resume_transport_retries_used if self._turn == 1 else 0
        schema_used = self.resume_schema_retries_used if self._turn == 1 else 0
        physical_attempts_used = self.resume_physical_attempts_used if self._turn == 1 else 0
        if self._turn == 1:
            self.resume_transport_retries_used = 0
            self.resume_schema_retries_used = 0
            self.resume_physical_attempts_used = 0
        pending_retry: str | None = None
        pending_failure: Exception | None = None

        def record_executed_retry(kind: str | None) -> None:
            if kind == "transport":
                self.transport_retries += 1
            elif kind == "schema":
                self.schema_retries += 1

        def account_started_attempt(
            *,
            requests_before: int,
            attempts_before: int,
            retry_kind: str | None,
            retry_recorded: bool,
        ) -> tuple[int, bool, bool]:
            started = self.requests_made > requests_before
            if not started:
                return attempts_before, retry_recorded, False
            if not retry_recorded:
                record_executed_retry(retry_kind)
            return attempts_before + 1, True, True

        while True:
            if physical_attempts_used >= 3:
                if pending_failure is not None:
                    raise pending_failure
                raise PlannerBudgetError("per-decision physical request budget exhausted")
            payload_received = False
            requests_before = self.requests_made
            attempts_before = physical_attempts_used
            request_retry_kind = pending_retry
            retry_recorded = False

            try:
                payload = self._request(
                    prompt,
                    retry_kind=request_retry_kind,
                    transport_retries_used=transport_used,
                    schema_retries_used=schema_used,
                    physical_attempts_used=physical_attempts_used + 1,
                )
                physical_attempts_used, retry_recorded, _ = account_started_attempt(
                    requests_before=requests_before,
                    attempts_before=attempts_before,
                    retry_kind=request_retry_kind,
                    retry_recorded=retry_recorded,
                )
                payload_received = True
                decision = parse_decision(payload)
                if (
                    isinstance(decision, ToolCall)
                    and decision.tool == "run_command"
                    and decision.command not in self.allowed_commands
                ):
                    raise ValueError("model selected a command that is unavailable in this run")
                pending_retry = None
                pending_failure = None
            except PlannerAttestationError:
                raise
            except PlannerTransportError as exc:
                physical_attempts_used, retry_recorded, _ = account_started_attempt(
                    requests_before=requests_before,
                    attempts_before=attempts_before,
                    retry_kind=request_retry_kind,
                    retry_recorded=retry_recorded,
                )
                pending_retry = None
                pending_failure = None
                if transport_used >= self.max_transport_retries:
                    raise
                transport_used += 1
                pending_retry = "transport"
                pending_failure = exc
                continue
            except PlannerBudgetError as budget_error:
                if pending_retry is not None:
                    physical_attempts_used, retry_recorded, started = account_started_attempt(
                        requests_before=requests_before,
                        attempts_before=attempts_before,
                        retry_kind=request_retry_kind,
                        retry_recorded=retry_recorded,
                    )
                    if not started and pending_failure is not None:
                        raise pending_failure from budget_error
                raise
            except (PlannerError, ValueError, KeyError, TypeError) as exc:
                physical_attempts_used, retry_recorded, _ = account_started_attempt(
                    requests_before=requests_before,
                    attempts_before=attempts_before,
                    retry_kind=request_retry_kind,
                    retry_recorded=retry_recorded,
                )
                pending_retry = None
                pending_failure = None
                if payload_received:
                    self._mark_last_call_schema_error()
                if schema_used >= self.max_schema_retries:
                    raise
                schema_used += 1
                pending_retry = "schema"
                pending_failure = exc
                continue
            if isinstance(decision, AgentAnswer):
                # Prefer a targeted read of a cited-but-unread file; fall back to a
                # general "keep investigating" nudge only if nothing grounds it.
                pending_read = self._auto_read(decision, catalog)
                if pending_read is not None:
                    return pending_read
                nudge = self._nudge_if_ungrounded(decision, catalog)
                if nudge is not None:
                    return nudge
                return _repair_citations(decision, catalog, rendered_ids)
            return decision

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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from inverse_agent.investigation import (
    AgentAnswer,
    Decision,
    SourceCitation,
    ToolCall,
    ToolObservation,
)
from inverse_agent.planner import (
    MAX_MODEL_COMPLETION_TOKENS,
    PlannerError,
    PlannerTransportError,
)

# Must comfortably exceed one full read (READ_MAX_TOKENS*CHARS_PER_TOKEN plus
# per-line "N: " prefixes and the header) so a legal 200-line read is never
# dropped for lack of room; well within a 32K-token context.
CATALOG_CHAR_BUDGET = 40_000
CATALOG_LINES_PER_OBS = 60

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
    "Citations: cite a read_file observation only. Copy its observation_id exactly "
    "from the id= field, use its path, and set start_line/end_line to the numbers "
    "shown before the colon (a line '12: foo' is line 12). Every finding needs a "
    "citation to a line you actually saw.\n"
    "In final_answer set condition_holds=true when the code confirms the "
    "condition or fact the goal asks about (e.g. the component IS exported, the "
    "entrypoint DOES exist, the bug IS present) and false only if the code shows "
    "it genuinely does not hold; give a non-empty summary and at least one "
    "finding.\n"
    "Observation completeness: headers explicitly show truncated/incomplete flags. "
    "Never infer absence from a truncated or incomplete observation. If the final "
    "answer depends on missing content, set complete=false. Set complete=true on "
    "tool actions.\n"
    'Fill unused fields with "" or [].'
)


class SupportsStructuredJson(Protocol):
    def complete_structured_json(
        self,
        *,
        system: str,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        max_tokens: int = ...,
    ) -> dict[str, Any]: ...


def _render_block(obs: ToolObservation) -> tuple[str, bool]:
    """Render one observation to a prompt block, returning (block, is_citable_read)."""

    if obs.metadata.get("refused"):
        citable = "not citable"
    elif obs.metadata.get("binary"):
        citable = "not citable (binary)"
    elif obs.tool == "read_file":
        citable = "CITABLE - cite this observation_id with its N: line numbers"
    else:
        citable = "pointer only - read_file a listed path to cite it"
    status = (
        f"truncated={str(obs.truncated).lower()} "
        f"incomplete={str(obs.incomplete).lower()} "
        f"redacted={str(obs.redacted).lower()}"
    )
    header = f"id={obs.observation_id} tool={obs.tool} path={obs.path} status[{status}] ({citable})"
    if obs.metadata.get("refused"):
        return f"{header}\n  REFUSED: {obs.text}", False
    if obs.metadata.get("binary"):
        return f"{header}\n  (binary file)", False
    # A read_file observation is already bounded (<=200 lines / ~3k tokens), so
    # show ALL of its lines: the model may only cite content it was actually shown.
    # Pointer results (list/search) stay capped.
    limit = len(obs.lines) if obs.tool == "read_file" else CATALOG_LINES_PER_OBS
    shown = obs.lines[:limit]
    fully_rendered = limit >= len(obs.lines)
    body = "\n".join(f"  {line}" for line in shown) or "  (no matching content)"
    if obs.incomplete or obs.truncated:
        body = (
            "  WARNING: this result is incomplete; omitted content may change a "
            f"negative conclusion.\n{body}"
        )
    is_citable_read = obs.tool == "read_file" and bool(obs.content_hash) and fully_rendered
    return f"{header}\n{body}", is_citable_read


def _render_catalog(catalog: tuple[ToolObservation, ...]) -> tuple[str, frozenset[str]]:
    """Render the catalog and return (prompt text, ids of fully-rendered reads).

    A read_file observation only becomes citable when all of its lines were
    actually placed in the prompt; an observation omitted for space is excluded,
    so the model can never be led to cite content it was not shown. Selection
    guarantees the most-recent citable read is always included (reads are the only
    citable evidence, so a burst of large pointer results must never crowd out the
    latest read), then fills the remaining budget with other observations
    newest-first; dropped context is always the oldest.
    """

    if not catalog:
        return "(no observations yet)", frozenset()

    blocks = [_render_block(obs) for obs in catalog]
    newest_read_index = next(
        (i for i in range(len(catalog) - 1, -1, -1) if blocks[i][1]),
        None,
    )

    selected: set[int] = set()
    used = 0
    if newest_read_index is not None:
        # Force-include the latest read (bounded well under the budget).
        selected.add(newest_read_index)
        used += len(blocks[newest_read_index][0])
    for index in range(len(catalog) - 1, -1, -1):
        if index in selected:
            continue
        block_len = len(blocks[index][0])
        if used + block_len > CATALOG_CHAR_BUDGET and selected:
            continue
        selected.add(index)
        used += block_len

    omitted = len(selected) < len(catalog)
    rendered_read_ids = {catalog[i].observation_id for i in selected if blocks[i][1]}
    text_blocks = [blocks[i][0] for i in sorted(selected)]
    if omitted:
        text_blocks.insert(0, "(earlier observations omitted for space)")
    return "\n".join(text_blocks), frozenset(rendered_read_ids)


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
        if obs.tool == "read_file" and obs.content_hash and obs.observation_id in rendered_ids
    ]
    by_id = {obs.observation_id: obs for obs in reads}

    def rebind(citation: SourceCitation) -> SourceCitation:
        existing = by_id.get(citation.observation_id)
        if existing is not None and existing.path == citation.path:
            return citation
        for obs in reads:
            if obs.path != citation.path:
                continue
            last = obs.start_line + len(obs.lines) - 1
            if obs.start_line <= citation.start_line <= last:
                return SourceCitation(
                    observation_id=obs.observation_id,
                    path=obs.path,
                    start_line=citation.start_line,
                    end_line=min(citation.end_line, last),
                    note=citation.note,
                )
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


def _parse_decision(payload: Mapping[str, Any]) -> Decision:
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
        return AgentAnswer(
            summary=str(payload.get("summary", "")),
            findings=tuple(str(f) for f in payload.get("findings", [])),
            next_actions=tuple(str(a) for a in payload.get("next_actions", [])),
            citations=citations,
            complete=bool(payload.get("complete", True)),
            issue_present=bool(payload.get("condition_holds", True)),
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
    max_transport_retries: int = 1
    max_schema_retries: int = 1
    max_auto_reads: int = 3
    max_nudges: int = 3
    max_total_requests: int = 54
    requests_made: int = field(default=0, init=False)
    transport_retries: int = field(default=0, init=False)
    schema_retries: int = field(default=0, init=False)
    _turn: int = field(default=0, init=False)
    _auto_reads: int = field(default=0, init=False)
    _nudges: int = field(default=0, init=False)

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
        reads = [obs for obs in catalog if obs.tool == "read_file" and obs.content_hash]
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
            if obs.tool == "read_file" and obs.content_hash:
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

    def _request(self, prompt: str) -> dict[str, Any]:
        if self.requests_made >= self.max_total_requests:
            raise PlannerError("model request budget exhausted")
        self.requests_made += 1
        return self.client.complete_structured_json(
            system=_SYSTEM_PROMPT,
            prompt=prompt,
            schema_name="investigation_decision",
            schema=DECISION_SCHEMA,
            max_tokens=MAX_MODEL_COMPLETION_TOKENS,
        )

    def decide(self, *, goal: str, catalog: tuple[ToolObservation, ...]) -> Decision:
        self._turn += 1
        observations, rendered_ids = _render_catalog(catalog)
        prompt = json.dumps(
            {
                "goal": goal,
                "hint": self.goal_hint,
                "turn": self._turn,
                "observations": observations,
                "instructions": (
                    "Return one action. If you have enough evidence, return "
                    "final_answer with citations; otherwise read or search."
                ),
            }
        )
        transport_used = 0
        schema_used = 0
        while True:
            try:
                payload = self._request(prompt)
                decision = _parse_decision(payload)
            except PlannerTransportError:
                if transport_used >= self.max_transport_retries:
                    raise
                transport_used += 1
                self.transport_retries += 1
                continue
            except (PlannerError, ValueError, KeyError, TypeError):
                if schema_used >= self.max_schema_retries:
                    raise
                schema_used += 1
                self.schema_retries += 1
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

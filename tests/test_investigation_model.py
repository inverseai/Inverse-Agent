"""Unit tests for the model-backed investigation planner (fake client)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from inverse_agent.attestations import AttestationScope, ScopedTrustStore
from inverse_agent.fs_tools import ToolObservation, WorkspaceReader
from inverse_agent.investigation import (
    AgentAnswer,
    InvestigationLoop,
    InvestigationVerdict,
    SourceCitation,
    StopReason,
    ToolCall,
)
from inverse_agent.investigation_model import (
    _SOURCE_LEXICAL_CONTEXT_ERROR,
    _SOURCE_SYMBOL_CITATION_ERROR,
    ModelInvestigationPlanner,
    _encoded_string_tokens,
    _expand_immediate_symbol_declaration_citations,
    _explicit_declaration_symbols,
    _grounded_answer_structure_error,
    _has_direct_untrusted_html_flow,
    _maximum_read_probe,
    _maximum_read_probe_tokens,
    _merge_duplicate_citation_findings,
    _recover_inline_citations,
    _render_catalog,
    _repair_citation_label_findings,
    _repair_citations,
    _repair_non_evidentiary_answer_fields,
    _repair_unique_listed_read_path,
    _requires_named_source_symbol,
    _trim_blank_citation_edges,
    _visible_declaration_symbols,
    parse_decision,
)
from inverse_agent.planner import (
    ModelResponseMetadata,
    PlannerAttestationError,
    PlannerBudgetError,
    PlannerProtocolError,
    PlannerTransportError,
)


class FakeClient:
    """Returns queued payloads; raises queued exceptions to exercise retry."""

    def __init__(
        self,
        responses: list[dict[str, Any] | Exception],
        *,
        metadata: list[ModelResponseMetadata | None] | None = None,
    ) -> None:
        self._responses = responses
        self._metadata = metadata or []
        self.calls = 0
        self.max_tokens_requested: list[int] = []
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []
        self.schemas: list[Mapping[str, Any]] = []
        self.timeouts_requested: list[float | None] = []
        self.last_response_metadata: ModelResponseMetadata | None = None

    def complete_structured_json(
        self,
        *,
        system: str,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        max_tokens: int = 4096,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        self.max_tokens_requested.append(max_tokens)
        self.prompts.append(prompt)
        self.system_prompts.append(system)
        self.schemas.append(schema)
        self.timeouts_requested.append(timeout_seconds)
        self.last_response_metadata = None
        item = self._responses.pop(0)
        if self._metadata:
            self.last_response_metadata = self._metadata.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _base(action: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "action": action,
        "path": "",
        "query": "",
        "glob": "",
        "based_on_observation_id": "",
        "start_line": 1,
        "max_lines": 200,
        "summary": "",
        "condition_holds": True,
        "complete": True,
        "findings": [],
        "next_actions": [],
        "citations": [],
    }
    payload.update(overrides)
    return payload


def _command_observation(
    observation_id: str,
    command: str,
    *,
    status: str = "failed",
) -> ToolObservation:
    return ToolObservation(
        observation_id=observation_id,
        tool="run_command",
        path=f"command/{command}",
        content_hash=f"hash-{observation_id}",
        text=f"{command}: {status}",
        lines=(f"1: {command}: {status}",),
        metadata={"command_name": command, "status": status, "citable_command": True},
    )


def test_parse_read_file_decision() -> None:
    client = FakeClient([_base("read_file", path="src/app.py", start_line=5, max_lines=20)])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert decision.tool == "read_file"
    assert decision.path == "src/app.py"
    assert decision.start_line == 5
    assert decision.max_lines == 20
    assert "observation text" in client.system_prompts[0]
    assert "untrusted data" in client.system_prompts[0]


def test_parse_search_maps_empty_glob_to_none() -> None:
    client = FakeClient([_base("search_text", query="needle", glob="")])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert decision.tool == "search_text"
    assert decision.query == "needle"
    assert decision.glob is None


def test_command_action_exists_only_for_explicitly_available_tools() -> None:
    read_client = FakeClient([_base("list_files", path=".")])
    ModelInvestigationPlanner(client=read_client).decide(goal="x", catalog=())
    read_actions = read_client.schemas[0]["properties"]["action"]["enum"]
    assert "run_command" not in read_actions

    command_client = FakeClient([_base("run_command", path="command/generic.head_commit")])
    decision = ModelInvestigationPlanner(
        client=command_client,
        allowed_commands=("generic.head_commit",),
    ).decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert decision.tool == "run_command"
    assert decision.command == "generic.head_commit"
    command_actions = command_client.schemas[0]["properties"]["action"]["enum"]
    assert "run_command" in command_actions
    assert "fresh human approval" in command_client.system_prompts[0]


def test_command_recovery_preserves_failed_observation_dependency() -> None:
    client = FakeClient(
        [
            _base(
                "run_command",
                path="generic.head_commit",
                based_on_observation_id="[obs_parent_failed]",
            )
        ]
    )
    decision = ModelInvestigationPlanner(
        client=client,
        allowed_commands=("generic.head_commit",),
    ).decide(goal="recover", catalog=())
    assert isinstance(decision, ToolCall)
    assert decision.based_on_observation_id == "obs_parent_failed"


def test_command_recovery_binds_the_unique_allowed_failed_observation() -> None:
    client = FakeClient([_base("run_command", path="command/generic.head_commit")])
    decision = ModelInvestigationPlanner(
        client=client,
        allowed_commands=("generic.parent_commit", "generic.head_commit"),
        command_recovery_dependencies=(("generic.head_commit", "generic.parent_commit"),),
    ).decide(
        goal="recover",
        catalog=(_command_observation("obs_parent_failed", "generic.parent_commit"),),
    )
    assert isinstance(decision, ToolCall)
    assert decision.command == "generic.head_commit"
    assert decision.based_on_observation_id == "obs_parent_failed"


def test_command_recovery_does_not_guess_between_failed_observations() -> None:
    client = FakeClient([_base("run_command", path="generic.head_commit")])
    decision = ModelInvestigationPlanner(
        client=client,
        allowed_commands=("generic.parent_commit", "generic.head_commit", "generic.status"),
        command_recovery_dependencies=(("generic.head_commit", "generic.parent_commit"),),
    ).decide(
        goal="recover",
        catalog=(
            _command_observation("obs_parent_failed_1", "generic.parent_commit"),
            _command_observation("obs_parent_failed_2", "generic.parent_commit"),
        ),
    )
    assert isinstance(decision, ToolCall)
    assert decision.based_on_observation_id is None


def test_command_recovery_cannot_run_before_its_declared_failure() -> None:
    client = FakeClient(
        [
            _base("run_command", path="generic.head_commit"),
            _base("run_command", path="generic.parent_commit"),
        ]
    )
    planner = ModelInvestigationPlanner(
        client=client,
        allowed_commands=("generic.parent_commit", "generic.head_commit"),
        command_recovery_dependencies=(("generic.head_commit", "generic.parent_commit"),),
    )
    decision = planner.decide(goal="recover", catalog=())
    assert isinstance(decision, ToolCall)
    assert decision.command == "generic.parent_commit"
    assert planner.schema_retries == 1


def test_completed_command_recovery_requires_final_answer_schema() -> None:
    parent = _command_observation("obs_parent_failed", "generic.parent_commit")
    head = ToolObservation(
        observation_id="obs_head_succeeded",
        tool="run_command",
        path="command/generic.head_commit",
        content_hash="hash-head",
        text="HEAD commit: abc",
        lines=("1: HEAD commit: abc",),
        metadata={
            "command_name": "generic.head_commit",
            "status": "succeeded",
            "citable_command": True,
            "based_on_observation_id": parent.observation_id,
        },
    )
    answer = _base(
        "final_answer",
        summary="done",
        findings=["HEAD is the root commit.", "HEAD commit ID is abc."],
        next_actions=["Keep the evidence."],
        citations=[
            {
                "observation_id": parent.observation_id,
                "path": parent.path,
                "start_line": 1,
                "end_line": 1,
            },
            {
                "observation_id": head.observation_id,
                "path": head.path,
                "start_line": 1,
                "end_line": 1,
            },
        ],
    )
    client = FakeClient([answer])
    decision = ModelInvestigationPlanner(
        client=client,
        allowed_commands=("generic.parent_commit", "generic.head_commit"),
        command_recovery_dependencies=(("generic.head_commit", "generic.parent_commit"),),
    ).decide(goal="recover", catalog=(parent, head))
    assert isinstance(decision, AgentAnswer)
    assert client.schemas[0]["properties"]["action"]["enum"] == ["final_answer"]
    assert "recovery sequence is complete" in client.system_prompts[0]
    assert client.max_tokens_requested == [4096]


def test_three_citable_observations_reserve_full_complex_answer_allowance() -> None:
    reads = tuple(
        ToolObservation(
            observation_id=f"obs_read_{index}",
            tool="read_file",
            path=f"src/file_{index}.py",
            content_hash=f"hash-{index}",
            text="def inspect_evidence(): pass",
            lines=("1: def inspect_evidence(): pass",),
        )
        for index in range(3)
    )
    answer = _base(
        "final_answer",
        summary="done",
        findings=["inspect_evidence was inspected."],
        next_actions=["Keep the evidence."],
        citations=[
            {
                "observation_id": reads[0].observation_id,
                "path": reads[0].path,
                "start_line": 1,
                "end_line": 1,
            }
        ],
    )
    client = FakeClient([answer])

    decision = ModelInvestigationPlanner(client=client).decide(
        goal="compare the inspected paths",
        catalog=reads,
    )

    assert isinstance(decision, AgentAnswer)
    assert client.max_tokens_requested == [4096]


def test_model_findings_are_not_rewritten_with_goal_subjects_or_hidden_oracles() -> None:
    read = ToolObservation(
        observation_id="obs_unmodified",
        tool="read_file",
        path="src/config.cpp",
        content_hash="hash-unmodified",
        text=("std::string_view load_bad();\n\n\n\n\n\n\nstd::string_view load_safe();"),
        lines=(
            "1: std::string_view load_bad();",
            "2: ",
            "3: ",
            "4: ",
            "5: ",
            "6: ",
            "7: ",
            "8: std::string_view load_safe();",
        ),
        start_line=1,
    )
    answer = _base(
        "final_answer",
        summary="Two paths were inspected.",
        findings=[
            "load_bad returns a dangling view.",
            "load_safe returns member storage.",
        ],
        next_actions=["Review both paths."],
        citations=[
            {
                "observation_id": read.observation_id,
                "path": read.path,
                "start_line": 1,
                "end_line": 1,
            },
            {
                "observation_id": read.observation_id,
                "path": read.path,
                "start_line": 8,
                "end_line": 8,
            },
        ],
    )
    client = FakeClient([answer])
    decision = ModelInvestigationPlanner(client=client).decide(
        goal="compare load_bad and load_safe", catalog=(read,)
    )

    assert isinstance(decision, AgentAnswer)
    assert decision.findings == (
        "load_bad returns a dangling view.",
        "load_safe returns member storage.",
    )
    prompt = json.loads(client.prompts[0])
    assert "required_evidence" not in prompt
    assert "required_finding_subjects" not in prompt
    assert "outstanding_required_evidence" not in prompt
    assert "dependency manifest" in client.system_prompts[0]
    assert "lexical_context_preserved=true" in client.system_prompts[0]
    assert "beginning at line 1" in client.system_prompts[0]
    assert "workspace-relative list entry is already complete" in client.system_prompts[0]
    assert "join the header path only for a header-relative" in client.system_prompts[0]
    assert "Never repeat an identical complete list" in client.system_prompts[0]
    assert "glob='**/*'" in client.system_prompts[0]
    assert "a caller proves only the handoff" in client.system_prompts[0]
    assert "protection mechanism" in client.system_prompts[0]
    assert "source-defined function, class, component, or symbol" in client.system_prompts[0]
    assert "name every observed mechanism" in client.system_prompts[0]
    assert "Do not combine distinct unsafe and safe subjects" in client.system_prompts[0]
    assert "positive-flow clause in this order" in client.system_prompts[0]
    assert "HTML sink alone is not data provenance" in client.system_prompts[0]
    assert "generic phrases such as 'the same file'" in client.system_prompts[0]
    assert "exactly one sentence per finding" in prompt["instructions"]
    assert "state the protection effect" in prompt["instructions"]


def test_basename_read_repairs_to_one_completed_listed_path() -> None:
    listing = ToolObservation(
        observation_id="obs_unique_listing",
        tool="list_files",
        path=".",
        content_hash="listing-hash",
        text="package.json\nprojects/views.py",
        lines=("package.json", "projects/views.py"),
        metadata={"recursive": True},
    )

    repaired = _repair_unique_listed_read_path(
        ToolCall(tool="read_file", path="views.py"),
        (listing,),
    )

    assert repaired == ToolCall(tool="read_file", path="projects/views.py")


@pytest.mark.parametrize("incomplete", (False, True))
def test_basename_read_does_not_guess_ambiguous_or_incomplete_listing(
    incomplete: bool,
) -> None:
    lines = ("api/views.py", "projects/views.py") if not incomplete else ("projects/views.py",)
    listing = ToolObservation(
        observation_id="obs_unsafe_listing",
        tool="list_files",
        path=".",
        content_hash="listing-hash",
        text="\n".join(lines),
        lines=lines,
        incomplete=incomplete,
        metadata={"recursive": True},
    )
    decision = ToolCall(tool="read_file", path="views.py")

    assert _repair_unique_listed_read_path(decision, (listing,)) == decision


def test_explicit_dependency_metadata_goal_auto_reads_only_visible_manifest() -> None:
    listing = ToolObservation(
        observation_id="obs_listing",
        tool="list_files",
        path=".",
        content_hash="hash-listing",
        text="package.json\nsrc/",
        lines=("package.json", "src/"),
    )
    client = FakeClient(
        [
            _base(
                "final_answer",
                summary="The source uses a named framework.",
                findings=["The source imports the framework."],
                next_actions=["Verify dependency metadata."],
                citations=[],
            )
        ]
    )
    planner = ModelInvestigationPlanner(client=client)

    decision = planner.decide(
        goal="Confirm the declared stack from dependency metadata.",
        catalog=(listing,),
    )

    assert decision == ToolCall(tool="read_file", path="package.json")


def test_recursive_nonroot_manifest_path_is_not_double_prefixed() -> None:
    listing = ToolObservation(
        observation_id="obs_listing",
        tool="list_files",
        path="web",
        content_hash="hash-listing",
        text="web/package.json",
        lines=("web/package.json",),
        metadata={"glob": "**/*", "recursive": True},
    )
    client = FakeClient(
        [
            _base(
                "final_answer",
                summary="The framework source was inspected.",
                findings=["The source uses a framework."],
                next_actions=["Verify its declaration."],
                citations=[],
            )
        ]
    )

    decision = ModelInvestigationPlanner(client=client).decide(
        goal="Confirm the stack from dependency metadata.",
        catalog=(listing,),
    )

    assert decision == ToolCall(tool="read_file", path="web/package.json")


def test_dependency_manifest_auto_read_does_not_invent_unlisted_path() -> None:
    listing = ToolObservation(
        observation_id="obs_listing",
        tool="list_files",
        path=".",
        content_hash="hash-listing",
        text="src/",
        lines=("src/",),
    )
    client = FakeClient([_base("list_files", path="src")])

    decision = ModelInvestigationPlanner(client=client).decide(
        goal="Confirm the declared stack from dependency metadata.",
        catalog=(listing,),
    )

    assert decision == ToolCall(tool="list_files", path="src")


def test_repeated_complete_recursive_listing_advances_only_to_emitted_file() -> None:
    listing = ToolObservation(
        observation_id="obs_listing",
        tool="list_files",
        path=".",
        content_hash="hash-listing",
        text="App/View.swift",
        lines=("App/View.swift",),
        metadata={"glob": "**/*", "recursive": True},
    )
    client = FakeClient([_base("list_files", path=".", glob="**/*")])

    decision = ModelInvestigationPlanner(client=client).decide(
        goal="inspect UIKit paths",
        catalog=(listing,),
    )

    assert decision == ToolCall(tool="read_file", path="App/View.swift")


def test_repeated_nonroot_recursive_listing_does_not_double_prefix_file() -> None:
    listing = ToolObservation(
        observation_id="obs_listing",
        tool="list_files",
        path="App",
        content_hash="hash-listing",
        text="App/View.swift",
        lines=("App/View.swift",),
        metadata={"glob": "**/*", "recursive": True},
    )
    client = FakeClient([_base("list_files", path="App", glob="**/*")])

    decision = ModelInvestigationPlanner(client=client).decide(
        goal="inspect UIKit paths",
        catalog=(listing,),
    )

    assert decision == ToolCall(tool="read_file", path="App/View.swift")


def test_repeated_complete_listing_advances_only_to_emitted_child_directory() -> None:
    listing = ToolObservation(
        observation_id="obs_listing",
        tool="list_files",
        path=".",
        content_hash="hash-listing",
        text="App/",
        lines=("App/",),
        metadata={"glob": "*"},
    )
    client = FakeClient([_base("list_files", path=".")])

    decision = ModelInvestigationPlanner(client=client).decide(
        goal="inspect UIKit paths",
        catalog=(listing,),
    )

    assert decision == ToolCall(tool="list_files", path="App", glob="**/*")


def test_repeated_truncated_listing_is_not_used_for_automatic_path_selection() -> None:
    listing = ToolObservation(
        observation_id="obs_listing",
        tool="list_files",
        path=".",
        content_hash="hash-listing",
        text="App/View.swift",
        lines=("App/View.swift",),
        truncated=True,
        metadata={"glob": "**/*"},
    )
    client = FakeClient([_base("list_files", path=".", glob="**/*")])

    decision = ModelInvestigationPlanner(client=client).decide(
        goal="inspect UIKit paths",
        catalog=(listing,),
    )

    assert decision == ToolCall(tool="list_files", path=".", glob="**/*")


def test_repeated_complete_search_advances_only_to_emitted_unread_path() -> None:
    search = ToolObservation(
        observation_id="obs_search",
        tool="search_text",
        path=".",
        content_hash="hash-search",
        text="src/app.py:7: target()",
        lines=("src/app.py:7: target()",),
        metadata={"query": "target", "glob": "*.py"},
    )
    client = FakeClient([_base("search_text", query="target", glob="*.py")])

    decision = ModelInvestigationPlanner(client=client).decide(
        goal="find target",
        catalog=(search,),
    )

    assert decision == ToolCall(tool="read_file", path="src/app.py")


@pytest.mark.parametrize(
    "dependencies",
    [
        (("generic.head_commit", "generic.head_commit"),),
        (("generic.missing", "generic.parent_commit"),),
        (("generic.head_commit", "generic.missing"),),
        (
            ("generic.head_commit", "generic.parent_commit"),
            ("generic.head_commit", "generic.status"),
        ),
    ],
)
def test_command_recovery_dependencies_must_be_unique_allowed_edges(
    dependencies: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(ValueError, match="recovery dependencies"):
        ModelInvestigationPlanner(
            client=FakeClient([]),
            allowed_commands=("generic.parent_commit", "generic.head_commit", "generic.status"),
            command_recovery_dependencies=dependencies,
        )


def test_command_recovery_does_not_overwrite_an_explicit_dependency() -> None:
    client = FakeClient(
        [
            _base(
                "run_command",
                path="generic.head_commit",
                based_on_observation_id="obs_explicit",
            )
        ]
    )
    decision = ModelInvestigationPlanner(
        client=client,
        allowed_commands=("generic.parent_commit", "generic.head_commit"),
    ).decide(
        goal="recover",
        catalog=(_command_observation("obs_parent_failed", "generic.parent_commit"),),
    )
    assert isinstance(decision, ToolCall)
    assert decision.based_on_observation_id == "obs_explicit"


def test_model_cannot_select_an_unavailable_command() -> None:
    client = FakeClient(
        [
            _base("run_command", path="command/evil"),
            _base("run_command", path="command/evil"),
        ]
    )
    planner = ModelInvestigationPlanner(client=client)
    with pytest.raises(ValueError, match="unavailable"):
        planner.decide(goal="x", catalog=())
    assert planner.schema_retries == 1


def test_parse_final_answer_with_citation() -> None:
    payload = _base(
        "final_answer",
        summary="done",
        findings=["found the bug"],
        next_actions=["fix it"],
        citations=[{"observation_id": "obs_1", "path": "a.py", "start_line": 2, "end_line": 3}],
    )
    client = FakeClient([payload])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, AgentAnswer)
    assert decision.findings == ("found the bug",)
    assert decision.citations[0].observation_id == "obs_1"
    assert decision.citations[0].end_line == 3


def test_schema_retry_then_success() -> None:
    good = _base("list_files", path=".")
    client = FakeClient([PlannerProtocolError("bad json"), good])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert client.calls == 2
    assert planner.schema_retries == 1
    assert planner.requests_made == 2
    assert client.max_tokens_requested == [2048, 1877]
    assert planner.completion_tokens_charged == sum(client.max_tokens_requested)
    first_prompt = json.loads(client.prompts[0])
    retry_prompt = json.loads(client.prompts[1])
    assert first_prompt["retry_correction"] == ""
    assert "previous response violated" in retry_prompt["retry_correction"]
    assert "bad json" not in client.prompts[1]


def test_grounded_malformed_answer_gets_one_accounted_schema_retry() -> None:
    evidence = _command_observation(
        "obs_parent_failed",
        "generic.parent_commit",
    )
    invalid = _base(
        "final_answer",
        summary="grounded but empty",
        findings=[],
        next_actions=[],
        citations=[
            {
                "observation_id": evidence.observation_id,
                "path": evidence.path,
                "start_line": 1,
                "end_line": 1,
            }
        ],
    )
    valid = _base(
        "final_answer",
        summary="grounded",
        findings=["HEAD has no first parent."],
        next_actions=["Inspect HEAD."],
        citations=[
            {
                "observation_id": evidence.observation_id,
                "path": evidence.path,
                "start_line": 1,
                "end_line": 1,
            }
        ],
    )
    client = FakeClient([invalid, valid])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)
    decision = planner.decide(goal="x", catalog=(evidence,))
    assert isinstance(decision, AgentAnswer)
    assert decision.findings == ("HEAD has no first parent.",)
    assert planner.schema_retries == 1
    assert [call.outcome for call in planner.model_calls] == ["schema_error", "success"]
    correction = json.loads(client.prompts[1])["retry_correction"]
    assert "one or more non-empty findings" in correction


def test_body_only_source_citation_expands_to_immediate_named_declaration() -> None:
    evidence = ToolObservation(
        observation_id="obs_search_safe",
        tool="read_file",
        path="views.py",
        content_hash="source-hash",
        text=(
            "def search_safe(request):\n"
            "    term = request.GET.get('q')\n"
            "    execute('SELECT ...', [term])"
        ),
        lines=(
            "1: def search_safe(request):",
            "2:     term = request.GET.get('q')",
            "3:     execute('SELECT ...', [term])",
        ),
        start_line=1,
    )
    body_only = _base(
        "final_answer",
        summary="The safe path was inspected.",
        findings=["search_safe uses a parameterized query."],
        next_actions=["Keep the control."],
        citations=[
            {
                "observation_id": evidence.observation_id,
                "path": evidence.path,
                "start_line": 2,
                "end_line": 3,
            }
        ],
    )
    client = FakeClient([body_only])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)

    decision = planner.decide(goal="verify search_safe", catalog=(evidence,))

    assert isinstance(decision, AgentAnswer)
    assert decision.citations[0].start_line == 1
    assert planner.schema_retries == 0
    assert len(client.prompts) == 1


def test_imported_mechanism_does_not_expand_a_complete_local_subject_anchor() -> None:
    source = (
        'import React from "react";\n'
        "export function UnsafeResult({ term }) {\n"
        "  return <div dangerouslySetInnerHTML={{ __html: term }} />;\n"
        "}\n"
    )
    evidence = ToolObservation(
        observation_id="obs_react_import_mechanism",
        tool="read_file",
        path="SearchResults.jsx",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The React component was inspected.",
        findings=(
            "React component UnsafeResult renders untrusted content via dangerouslySetInnerHTML.",
        ),
        next_actions=("Remove the unsafe sink.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 2, 3),),
        issue_present=True,
    )

    repaired = _expand_immediate_symbol_declaration_citations(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired == answer
    assert _grounded_answer_structure_error(repaired, (evidence,)) is None


def test_imported_mechanism_does_not_block_local_subject_declaration_expansion() -> None:
    source = (
        "import torch\n"
        "def evaluate_safe(model, loader):\n"
        "    model.eval()\n"
        "    with torch.inference_mode():\n"
        "        return list(loader)\n"
    )
    evidence = ToolObservation(
        observation_id="obs_pytorch_body_only",
        tool="read_file",
        path="experiment.py",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The safe evaluation helper was inspected.",
        findings=("evaluate_safe wraps evaluation in torch.inference_mode, disabling gradients.",),
        next_actions=("Keep the inference control.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 3, 5),),
        issue_present=False,
    )

    repaired = _expand_immediate_symbol_declaration_citations(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired.citations == (SourceCitation(evidence.observation_id, evidence.path, 2, 5),)
    assert _grounded_answer_structure_error(repaired, (evidence,)) is None


def test_python_import_module_is_a_grounded_import_only_subject() -> None:
    source = "from django.db import connection\ndef search_unsafe(request):\n    return request\n"
    evidence = ToolObservation(
        observation_id="obs_django_import",
        tool="read_file",
        path="views.py",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The backend import was inspected.",
        findings=("The code imports `django.db`, confirming the Django backend.",),
        next_actions=("Keep the dependency manifest explicit.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 1),),
        issue_present=False,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) is None


def test_import_only_citation_trims_unrelated_following_declaration() -> None:
    source = "from django.db import connection\ndef search_unsafe(request):\n    return request\n"
    evidence = ToolObservation(
        observation_id="obs_django_import_trim",
        tool="read_file",
        path="views.py",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The backend import was inspected.",
        findings=("The code imports `django.db`, confirming the Django backend.",),
        next_actions=("Keep the dependency manifest explicit.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 2),),
        issue_present=False,
    )

    repaired = _trim_blank_citation_edges(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired.citations == (SourceCitation(evidence.observation_id, evidence.path, 1, 1),)
    assert _grounded_answer_structure_error(repaired, (evidence,)) is None


def test_trailing_unmentioned_declaration_is_trimmed_from_subject_anchor() -> None:
    source = (
        "import UIKit\n"
        "class ProfileViewController: UIViewController { func refreshProfile() { "
        "DispatchQueue.global().async { self.nameLabel.text = self.loadName() } } }\n"
        "class AvatarViewController: UIViewController { func refreshAvatar() { "
        "DispatchQueue.main.async { self.avatarView.image = image } } }\n"
    )
    evidence = ToolObservation(
        observation_id="obs_swift_trailing_subject",
        tool="read_file",
        path="ProfileViewController.swift",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The unsafe refresh was inspected.",
        findings=(
            "ProfileViewController refreshes nameLabel on DispatchQueue.global, outside the "
            "main thread.",
        ),
        next_actions=("Move the UI write to the main queue.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 2, 3),),
        issue_present=True,
    )

    repaired = _trim_blank_citation_edges(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired.citations == (SourceCitation(evidence.observation_id, evidence.path, 2, 2),)
    assert _grounded_answer_structure_error(repaired, (evidence,)) is None


def test_trailing_standalone_brace_is_trimmed_from_subject_anchor() -> None:
    source = (
        "export function UnsafeResult({ userSuppliedTerm }) {\n"
        "  return <div dangerouslySetInnerHTML={{ __html: userSuppliedTerm }} />;\n"
        "}\n"
    )
    evidence = ToolObservation(
        observation_id="obs_jsx_trailing_brace",
        tool="read_file",
        path="SearchResults.jsx",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The unsafe component was inspected.",
        findings=("UnsafeResult renders userSuppliedTerm via dangerouslySetInnerHTML.",),
        next_actions=("Remove the unsafe sink.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 3),),
        issue_present=True,
    )

    repaired = _trim_blank_citation_edges(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired.citations == (SourceCitation(evidence.observation_id, evidence.path, 1, 2),)
    assert _grounded_answer_structure_error(repaired, (evidence,)) is None


def test_leading_unmentioned_import_is_trimmed_from_local_subject_anchor() -> None:
    source = (
        "from django.db import connection\n"
        "def search_unsafe(request):\n"
        "    term = request.GET.get('q')\n"
        '    connection.cursor().execute("SELECT * FROM p WHERE n = \'" + term + "\'")\n'
    )
    evidence = ToolObservation(
        observation_id="obs_python_leading_import",
        tool="read_file",
        path="views.py",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The unsafe query was inspected.",
        findings=(
            "The Django view search_unsafe concatenates user input into raw SQL, enabling "
            "SQL injection.",
        ),
        next_actions=("Replace concatenation with parameter binding.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 4),),
        issue_present=True,
    )

    repaired = _trim_blank_citation_edges(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired.citations == (SourceCitation(evidence.observation_id, evidence.path, 2, 4),)
    assert _grounded_answer_structure_error(repaired, (evidence,)) is None


def test_comment_cannot_masquerade_as_declaration_for_citation_expansion() -> None:
    evidence = ToolObservation(
        observation_id="obs_comment",
        tool="read_file",
        path="views.py",
        content_hash="source-hash",
        text="# search_safe declaration\nexecute('SELECT ...', [term])",
        lines=(
            "5: # search_safe declaration",
            "6: execute('SELECT ...', [term])",
        ),
        start_line=5,
    )
    answer = AgentAnswer(
        summary="The query path was inspected.",
        findings=("search_safe uses a parameterized query.",),
        next_actions=("Keep parameter binding.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 6, 6),),
        issue_present=False,
    )

    repaired = _expand_immediate_symbol_declaration_citations(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired == answer


def test_multiline_string_cannot_trigger_declaration_citation_expansion() -> None:
    source = (
        "class search_safe:\n"
        "    pass\n"
        "blob = '''\n"
        "def search_safe(request):\n"
        "    execute('SELECT ...', [term])\n"
        "'''"
    )
    evidence = ToolObservation(
        observation_id="obs_contextual_expansion",
        tool="read_file",
        path="views.py",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The query path was inspected.",
        findings=("`search_safe` uses a parameterized query.",),
        next_actions=("Keep parameter binding.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 5, 5),),
        issue_present=False,
    )

    repaired = _expand_immediate_symbol_declaration_citations(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired == answer


def test_multiline_markup_attribute_citation_expands_to_subject_line() -> None:
    source = (
        "<manifest>\n"
        '  <activity android:name=".InternalSettingsActivity"\n'
        '      android:exported="false"/>\n'
        "</manifest>"
    )
    evidence = ToolObservation(
        observation_id="obs_markup_expansion",
        tool="read_file",
        path="AndroidManifest.xml",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The private activity was inspected.",
        findings=("`InternalSettingsActivity` is not exported.",),
        next_actions=("Keep the activity private.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 3, 3),),
        issue_present=False,
    )

    repaired = _expand_immediate_symbol_declaration_citations(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired.citations == (SourceCitation(evidence.observation_id, evidence.path, 2, 3),)
    assert _grounded_answer_structure_error(repaired, (evidence,)) is None


@pytest.mark.parametrize(
    "preceding_line",
    (
        "marker = 1  # def search_safe(request):",
        'marker = "def search_safe(request):"',
        "marker = /def search_safe(request):/",
    ),
)
def test_inline_comment_or_string_cannot_masquerade_as_declaration(
    preceding_line: str,
) -> None:
    evidence = ToolObservation(
        observation_id="obs_inline_fake",
        tool="read_file",
        path="views.py",
        content_hash="source-hash",
        text=f"{preceding_line}\nexecute('SELECT ...', [term])",
        lines=(f"5: {preceding_line}", "6: execute('SELECT ...', [term])"),
        start_line=5,
    )
    answer = AgentAnswer(
        summary="The query path was inspected.",
        findings=("search_safe uses a parameterized query.",),
        next_actions=("Keep parameter binding.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 6, 6),),
        issue_present=False,
    )

    repaired = _expand_immediate_symbol_declaration_citations(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired == answer


@pytest.mark.parametrize(
    "preceding_line",
    (
        "return search_safe();",
        "throw search_safe();",
        "await search_safe();",
        "if search_safe()",
        "while search_safe()",
    ),
)
def test_call_statement_cannot_masquerade_as_declaration(preceding_line: str) -> None:
    evidence = ToolObservation(
        observation_id="obs_call_statement",
        tool="read_file",
        path="views.cpp",
        content_hash="source-hash",
        text=f"{preceding_line}\nexecute_query(term);",
        lines=(f"5: {preceding_line}", "6: execute_query(term);"),
        start_line=5,
    )
    answer = AgentAnswer(
        summary="The query path was inspected.",
        findings=("search_safe uses a parameterized query.",),
        next_actions=("Keep parameter binding.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 6, 6),),
        issue_present=False,
    )

    repaired = _expand_immediate_symbol_declaration_citations(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert "search_safe" not in _explicit_declaration_symbols(preceding_line)
    assert repaired == answer


@pytest.mark.parametrize(
    "source",
    (
        "// search_safe declaration\nexecute(query);",
        'marker = "search_safe declaration";\nexecute(query);',
        "marker = /search_safe declaration/;\nexecute(query);",
        "search_safe();\nexecute(query);",
    ),
)
def test_non_declaration_cannot_satisfy_named_symbol_validation(source: str) -> None:
    source_lines = source.splitlines()
    evidence = ToolObservation(
        observation_id="obs_fake_symbol",
        tool="read_file",
        path="x.js",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source_lines, 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The query path was inspected.",
        findings=("search_safe uses parameter binding.",),
        next_actions=("Keep parameter binding.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 2),),
        issue_present=False,
    )

    assert "search_safe" not in _visible_declaration_symbols(source, "x.js")
    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each code-source finding must name a concrete source-defined identifier supported "
        "by the cited declaration"
    )


def test_visible_blank_citation_padding_is_trimmed() -> None:
    evidence = ToolObservation(
        observation_id="obs_gateway",
        tool="read_file",
        path="gateway.py",
        content_hash="source-hash",
        text="def handle(request):\n    return worker.enqueue(request)\n",
        lines=(
            "1: def handle(request):",
            "2:     return worker.enqueue(request)",
            "3: ",
        ),
        start_line=1,
    )
    answer = _base(
        "final_answer",
        summary="The gateway path was inspected.",
        findings=["handle returns worker.enqueue(request)."],
        next_actions=["Keep the handoff observable."],
        citations=[
            {
                "observation_id": evidence.observation_id,
                "path": evidence.path,
                "start_line": 1,
                "end_line": 3,
            }
        ],
    )
    client = FakeClient([answer])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)

    decision = planner.decide(goal="trace the gateway handoff", catalog=(evidence,))

    assert isinstance(decision, AgentAnswer)
    assert decision.citations[0].start_line == 1
    assert decision.citations[0].end_line == 2
    assert planner.schema_retries == 0


def test_html_sink_without_untrusted_flow_gets_one_accounted_schema_retry() -> None:
    evidence = ToolObservation(
        observation_id="obs_unsafe_result",
        tool="read_file",
        path="SearchResults.jsx",
        content_hash="source-hash",
        text=(
            "export function UnsafeResult({ term }) {\n"
            "  return <div dangerouslySetInnerHTML={{ __html: term }} />;\n"
            "}"
        ),
        lines=(
            "1: export function UnsafeResult({ term }) {",
            "2:   return <div dangerouslySetInnerHTML={{ __html: term }} />;",
            "3: }",
        ),
        start_line=1,
    )
    weak = _base(
        "final_answer",
        summary="The browser path was inspected.",
        findings=["UnsafeResult renders raw HTML via dangerouslySetInnerHTML."],
        next_actions=["Remove the unsafe sink."],
        citations=[
            {
                "observation_id": evidence.observation_id,
                "path": evidence.path,
                "start_line": 1,
                "end_line": 2,
            }
        ],
    )
    corrected = _base(
        "final_answer",
        summary="The browser path was inspected.",
        findings=["UnsafeResult renders untrusted user-supplied data via dangerouslySetInnerHTML."],
        next_actions=["Remove the unsafe sink."],
        citations=weak["citations"],
    )
    client = FakeClient([weak, corrected])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)

    decision = planner.decide(goal="audit browser injection", catalog=(evidence,))

    assert isinstance(decision, AgentAnswer)
    assert "untrusted" in decision.findings[0]
    assert planner.schema_retries == 1
    correction = json.loads(client.prompts[1])["retry_correction"]
    assert "explicitly untrusted" in correction
    assert "do not substitute only 'raw HTML' or 'term prop'" in correction


@pytest.mark.parametrize(
    "finding",
    (
        "UnsafeResult refuses to pass untrusted input to dangerouslySetInnerHTML.",
        (
            "UnsafeResult passes untrusted input to an audit logger before using "
            "dangerouslySetInnerHTML."
        ),
        "UnsafeResult passes sanitized untrusted input to dangerouslySetInnerHTML.",
        (
            "UnsafeResult passes untrusted input to dangerouslySetInnerHTML, but it "
            "does not actually do so."
        ),
        (
            "UnsafeResult passes untrusted input to dangerouslySetInnerHTML only after "
            "sanitizing it."
        ),
        (
            "UnsafeResult passes untrusted input to dangerouslySetInnerHTML only in the "
            "audit logger."
        ),
        "It is false that UnsafeResult passes untrusted input to dangerouslySetInnerHTML.",
        "After sanitizing it, UnsafeResult passes untrusted input to dangerouslySetInnerHTML.",
        (
            "The audit logger reports that UnsafeResult passes untrusted input to "
            "dangerouslySetInnerHTML."
        ),
        (
            "UnsafeResult passes untrusted input to dangerouslySetInnerHTML, which is "
            "prevented by the wrapper."
        ),
        "It is false; UnsafeResult passes untrusted input to dangerouslySetInnerHTML.",
        (
            "UnsafeResult passes untrusted input to dangerouslySetInnerHTML; but it is "
            "prevented by a wrapper."
        ),
        (
            "UnsafeResult passes untrusted input to dangerouslySetInnerHTML. It is "
            "prevented by a wrapper."
        ),
        (
            "UnsafeResult passes untrusted input to dangerouslySetInnerHTML, but it "
            "doesn't actually do so."
        ),
        "It isn't true that UnsafeResult passes untrusted input to dangerouslySetInnerHTML.",
        (
            "UnsafeResult passes untrusted input to dangerouslySetInnerHTML, but cannot "
            "actually do so."
        ),
        "After escaping it, UnsafeResult passes untrusted input to dangerouslySetInnerHTML.",
        "After filtering it, UnsafeResult passes untrusted input to dangerouslySetInnerHTML.",
    ),
)
def test_production_react_flow_rejects_negated_disconnected_or_protected_edges(
    finding: str,
) -> None:
    assert not _has_direct_untrusted_html_flow(finding)


def test_production_react_flow_accepts_direct_explicit_edge() -> None:
    assert _has_direct_untrusted_html_flow(
        "UnsafeResult renders user-supplied term via dangerouslySetInnerHTML."
    )


def test_production_react_flow_accepts_explicit_untrusted_prop_edge() -> None:
    assert _has_direct_untrusted_html_flow(
        "UnsafeResult renders untrusted term prop directly into dangerouslySetInnerHTML."
    )


def test_production_react_flow_accepts_user_provided_content_edge() -> None:
    assert _has_direct_untrusted_html_flow(
        "UnsafeResult renders user-provided content via dangerouslySetInnerHTML."
    )


@pytest.mark.parametrize(
    "identifier", ("userControlledTerm", "userProvidedHtml", "userSuppliedTerm")
)
def test_production_react_flow_accepts_explicit_provenance_identifier(
    identifier: str,
) -> None:
    assert _has_direct_untrusted_html_flow(
        f"UnsafeResult renders {identifier} via dangerouslySetInnerHTML."
    )


def test_production_react_flow_accepts_unsafe_adverb_after_provenance() -> None:
    assert _has_direct_untrusted_html_flow(
        "UnsafeResult component renders userSuppliedTerm unsafely via dangerouslySetInnerHTML."
    )


@pytest.mark.parametrize(
    "path",
    (
        "projects/static/projects/SearchResults.jsx",
        "/projects/static/projects/SearchResults.jsx",
        "C:\\projects\\static\\projects\\SearchResults.jsx",
        "D:\\Office Repos\\LLM AGENT\\projects\\static\\projects\\SearchResults.jsx",
        "projects/static files/projects/SearchResults.jsx",
    ),
)
def test_production_react_flow_ignores_static_directory_in_code_path(
    path: str,
) -> None:
    assert _has_direct_untrusted_html_flow(
        f"`{path}` contains component "
        "`UnsafeResult` that renders user-supplied `term` via "
        "`dangerouslySetInnerHTML`, exposing XSS risk."
    )


@pytest.mark.parametrize(
    "path",
    (
        "projects/UnsafeRESULT/static.jsx",
        "projects/DangerouslySetInnerHTML/static.jsx",
    ),
)
def test_production_react_flow_ignores_wrong_case_candidates_in_code_path(
    path: str,
) -> None:
    assert _has_direct_untrusted_html_flow(
        f"`{path}` contains component `UnsafeResult` that renders user-supplied "
        "`term` via `dangerouslySetInnerHTML`."
    )


@pytest.mark.parametrize(
    "protection",
    (
        "The content is `trusted/sanitized`.",
        "The content passes through `sanitize(term)/encode(term)`.",
    ),
)
def test_production_react_flow_does_not_erase_slash_bearing_protection(
    protection: str,
) -> None:
    assert not _has_direct_untrusted_html_flow(
        "UnsafeResult renders user-supplied term via dangerouslySetInnerHTML, "
        f"exposing XSS risk. {protection}"
    )


def test_code_finding_without_named_source_symbol_gets_schema_retry() -> None:
    evidence = ToolObservation(
        observation_id="obs_load_safe",
        tool="read_file",
        path="config.cpp",
        content_hash="source-hash",
        text="std::string_view load_safe() { return storage_; }",
        lines=("1: std::string_view load_safe() { return storage_; }",),
        start_line=1,
    )
    citation = {
        "observation_id": evidence.observation_id,
        "path": evidence.path,
        "start_line": 1,
        "end_line": 1,
    }
    weak = _base(
        "final_answer",
        summary="The safe path was inspected.",
        findings=["Citation: obs_load_safe line 1."],
        next_actions=["Keep owning storage."],
        citations=[citation],
    )
    corrected = _base(
        "final_answer",
        summary="The safe path was inspected.",
        findings=["load_safe returns a string_view backed by member storage_."],
        next_actions=["Keep owning storage."],
        citations=[citation],
    )
    client = FakeClient([weak, corrected])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)

    decision = planner.decide(goal="compare lifetime paths", catalog=(evidence,))

    assert isinstance(decision, AgentAnswer)
    assert decision.findings == (corrected["findings"][0],)
    assert planner.schema_retries == 1
    correction = json.loads(client.prompts[1])["retry_correction"]
    assert "do not use a citation label as a finding" in correction


def test_generic_code_finding_without_declared_symbol_gets_schema_retry() -> None:
    evidence = ToolObservation(
        observation_id="obs_config_function",
        tool="read_file",
        path="config.cpp",
        content_hash="source-hash",
        text="std::string_view load_safe() { return storage_; }",
        lines=("1: std::string_view load_safe() { return storage_; }",),
        start_line=1,
    )
    citation = {
        "observation_id": evidence.observation_id,
        "path": evidence.path,
        "start_line": 1,
        "end_line": 1,
    }
    weak = _base(
        "final_answer",
        summary="The safe path was inspected.",
        findings=["The safe path returns a durable view."],
        next_actions=["Keep owning storage."],
        citations=[citation],
    )
    corrected = _base(
        "final_answer",
        summary="The safe path was inspected.",
        findings=["load_safe returns a durable view."],
        next_actions=["Keep owning storage."],
        citations=[citation],
    )
    client = FakeClient([weak, corrected])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)

    decision = planner.decide(goal="compare lifetime paths", catalog=(evidence,))

    assert isinstance(decision, AgentAnswer)
    assert decision.findings == (corrected["findings"][0],)
    assert planner.schema_retries == 1
    correction = json.loads(client.prompts[1])["retry_correction"]
    assert "identifier visible in the cited source" in correction


@pytest.mark.parametrize(
    "finding",
    (
        "The operation is safe.",
        "The cited code remains safe.",
        "Function Safe returns dangerous behavior.",
        "safe paths remain dangerous.",
        "safe operations remain dangerous.",
        "safe handling remains dangerous.",
        "safe methods are vulnerable.",
    ),
)
def test_plain_lowercase_symbol_cannot_be_satisfied_as_incidental_prose(
    finding: str,
) -> None:
    evidence = ToolObservation(
        observation_id="obs_safe_prose",
        tool="read_file",
        path="ops.py",
        content_hash="source-hash",
        text="def safe():\n    return destroy_everything()",
        lines=("1: def safe():", "2:     return destroy_everything()"),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The operation was inspected.",
        findings=(finding,),
        next_actions=("Review the cited behavior.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 2),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each code-source finding must name a concrete source-defined identifier supported "
        "by the cited declaration"
    )


@pytest.mark.parametrize(
    "finding",
    (
        "enqueue places requests on the billing queue.",
        "The enqueue function places requests on the billing queue.",
        "Function enqueue places requests on the billing queue.",
        "Calling enqueue places requests on the billing queue.",
        "enqueue submits requests to the billing queue.",
        "enqueue puts requests on the billing queue.",
        "enqueue placed requests on the billing queue.",
        "enqueue sends requests to the billing queue.",
    ),
)
def test_plain_lowercase_symbol_is_valid_as_a_concrete_subject(finding: str) -> None:
    evidence = ToolObservation(
        observation_id="obs_enqueue_subject",
        tool="read_file",
        path="worker.py",
        content_hash="source-hash",
        text="def enqueue(request):\n    BILLING_QUEUE.put(request)",
        lines=(
            "1: def enqueue(request):",
            "2:     BILLING_QUEUE.put(request)",
        ),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The worker was inspected.",
        findings=(finding,),
        next_actions=("Keep the queue behavior intentional.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 2),),
        issue_present=False,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) is None


@pytest.mark.parametrize(
    "finding",
    (
        "The file `src/safe.py` contains dangerous behavior.",
        "The `safe.py` file contains dangerous behavior.",
        "The path src/safe.py was inspected.",
    ),
)
def test_source_path_cannot_launder_a_plain_symbol_mention(finding: str) -> None:
    evidence = ToolObservation(
        observation_id="obs_safe_path",
        tool="read_file",
        path="safe.py",
        content_hash="source-hash",
        text="def safe():\n    return destroy_everything()",
        lines=("1: def safe():", "2:     return destroy_everything()"),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The path was inspected.",
        findings=(finding,),
        next_actions=("Review the cited behavior.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 2),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each code-source finding must name a concrete source-defined identifier supported "
        "by the cited declaration"
    )


@pytest.mark.parametrize("symbol", ("Safe", "Default", "Open"))
def test_ambiguous_pascal_symbol_cannot_be_used_only_as_an_adjective(
    symbol: str,
) -> None:
    evidence = ToolObservation(
        observation_id="obs_pascal_adjective",
        tool="read_file",
        path="Ops.java",
        content_hash="source-hash",
        text=f"class {symbol} {{}}",
        lines=(f"1: class {symbol} {{}}",),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The operation was inspected.",
        findings=(f"The operation is {symbol}.",),
        next_actions=("Review the cited behavior.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 1),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each code-source finding must name a concrete source-defined identifier supported "
        "by the cited declaration"
    )


@pytest.mark.parametrize(
    ("symbol", "finding"),
    (
        ("The", "The implementation fails."),
        ("This", "This code crashes."),
        ("That", "That path fails."),
        ("These", "These calls fail."),
        ("Those", "Those calls fail."),
        ("A", "A bug exists."),
        ("An", "An error occurs."),
        ("The", "The component fails."),
        ("This", "This function crashes."),
        ("A", "A class is insecure."),
        ("An", "An interface fails."),
        ("That", "That method throws."),
        ("I", "I found a bug."),
        ("It", "It fails while loading."),
        ("We", "We found a bug."),
        ("You", "You found a bug."),
        ("He", "He found a bug."),
        ("She", "She found a bug."),
        ("They", "They fail while loading."),
        ("There", "There is a bug."),
        ("Here", "Here is a bug."),
        ("When", "When loading, a bug occurs."),
        ("Where", "Where loading occurs, a bug follows."),
        ("Who", "Who found the bug?"),
        ("Which", "Which path fails?"),
        ("What", "What fails while loading?"),
        ("My", "My path fails."),
        ("Your", "Your path fails."),
        ("His", "His path fails."),
        ("Her", "Her path fails."),
        ("Its", "Its path fails."),
        ("Our", "Our path fails."),
        ("Their", "Their path fails."),
        ("Whose", "Whose path fails?"),
        ("Safe", "Safe component fails."),
        ("Default", "Default function crashes."),
        ("Open", "Open method leaks."),
        ("Raw", "Raw component fails."),
        ("Static", "Static class breaks."),
    ),
)
def test_grammatical_article_or_pronoun_cannot_launder_a_class_symbol(
    symbol: str,
    finding: str,
) -> None:
    evidence = ToolObservation(
        observation_id="obs_grammar_symbol",
        tool="read_file",
        path="Ops.java",
        content_hash="source-hash",
        text=f"class {symbol} {{}}",
        lines=(f"1: class {symbol} {{}}",),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The class was inspected.",
        findings=(finding,),
        next_actions=("Review the cited behavior.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 1),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each code-source finding must name a concrete source-defined identifier supported "
        "by the cited declaration"
    )


@pytest.mark.parametrize(
    "symbol",
    (
        "In",
        "On",
        "At",
        "For",
        "From",
        "To",
        "By",
        "With",
        "Without",
        "As",
        "And",
        "Or",
        "But",
        "Because",
        "Although",
        "While",
        "If",
    ),
)
def test_sentence_connector_cannot_launder_a_class_symbol(symbol: str) -> None:
    evidence = ToolObservation(
        observation_id="obs_connector_symbol",
        tool="read_file",
        path="Ops.java",
        content_hash="source-hash",
        text=f"class {symbol} {{}}",
        lines=(f"1: class {symbol} {{}}",),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The class was inspected.",
        findings=(f"{symbol} this path, a bug exists.",),
        next_actions=("Review the cited behavior.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 1),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each code-source finding must name a concrete source-defined identifier supported "
        "by the cited declaration"
    )


def test_ambiguous_symbol_is_valid_with_explicit_code_formatting() -> None:
    evidence = ToolObservation(
        observation_id="obs_safe_identifier",
        tool="read_file",
        path="ops.py",
        content_hash="source-hash",
        text="def safe():\n    return False",
        lines=("1: def safe():", "2:     return False"),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The function was inspected.",
        findings=("`safe` returns False.",),
        next_actions=("Review the cited behavior.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 2),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) is None


def test_unique_grounded_summary_sentence_repairs_citation_label_finding() -> None:
    evidence = ToolObservation(
        observation_id="obs_config_source",
        tool="read_file",
        path="config.cpp",
        content_hash="source-hash",
        text=(
            "std::string_view load_bad() { return local; }\n"
            "class ConfigCache { std::string storage_; public: "
            "std::string_view load_safe() { return storage_; } };"
        ),
        lines=(
            "1: std::string_view load_bad() { return local; }",
            "2: class ConfigCache { std::string storage_; public: "
            "std::string_view load_safe() { return storage_; } };",
        ),
        start_line=1,
    )
    safe_sentence = "ConfigCache::load_safe returns a string_view backed by member storage_."
    answer = _base(
        "final_answer",
        summary=(f"load_bad returns a view over local storage. {safe_sentence}"),
        findings=[
            "load_bad returns a view over local storage.",
            "Citation: obs_config_source line 2.",
        ],
        next_actions=["Keep returned views backed by durable storage."],
        citations=[
            {
                "observation_id": evidence.observation_id,
                "path": evidence.path,
                "start_line": 1,
                "end_line": 1,
            },
            {
                "observation_id": evidence.observation_id,
                "path": evidence.path,
                "start_line": 2,
                "end_line": 2,
            },
        ],
    )
    client = FakeClient([answer])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)

    decision = planner.decide(goal="compare lifetime paths", catalog=(evidence,))

    assert isinstance(decision, AgentAnswer)
    assert decision.findings == (answer["findings"][0], safe_sentence)
    assert planner.schema_retries == 0


def test_citation_label_repair_ignores_declaration_inside_multiline_string() -> None:
    source = "class Fake:\n    pass\nblob = '''\nclass Fake:\n    pass\n'''"
    evidence = ToolObservation(
        observation_id="obs_contextual_label",
        tool="read_file",
        path="config.py",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="Fake fails while loading configuration.",
        findings=("Citation: obs_contextual_label line 4.",),
        next_actions=("Review the failure.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 4, 4),),
        issue_present=True,
    )

    repaired = _repair_citation_label_findings(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired == answer


def test_citation_label_repair_cannot_reuse_one_summary_sentence() -> None:
    evidence = ToolObservation(
        observation_id="obs_duplicate_class",
        tool="read_file",
        path="config.cpp",
        content_hash="source-hash",
        text="class ConfigCache {};\nclass ConfigCache {};",
        lines=("1: class ConfigCache {};", "2: class ConfigCache {};"),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="ConfigCache is safe.",
        findings=(
            "Citation: obs_duplicate_class line 1.",
            "Citation: obs_duplicate_class line 2.",
        ),
        next_actions=("Keep durable storage.",),
        citations=(
            SourceCitation(evidence.observation_id, evidence.path, 1, 1),
            SourceCitation(evidence.observation_id, evidence.path, 2, 2),
        ),
        issue_present=False,
    )

    repaired = _repair_citation_label_findings(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired == answer


def test_lowercase_summary_sentence_boundary_prevents_claim_laundering() -> None:
    evidence = ToolObservation(
        observation_id="obs_lowercase",
        tool="read_file",
        path="config.py",
        content_hash="source-hash",
        text="def load_safe(): return storage",
        lines=("1: def load_safe(): return storage",),
        start_line=1,
    )
    grounded_sentence = "load_safe returns durable storage."
    answer = AgentAnswer(
        summary=f"An unrelated uncited vulnerability exists. {grounded_sentence}",
        findings=("Citation: obs_lowercase line 1.",),
        next_actions=("Keep durable storage.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 1),),
        issue_present=False,
    )

    repaired = _repair_citation_label_findings(
        answer,
        (evidence,),
        frozenset({evidence.observation_id}),
    )

    assert repaired.findings == (grounded_sentence,)


def test_citation_label_repair_ignores_unrendered_observation() -> None:
    evidence = ToolObservation(
        observation_id="obs_hidden_config",
        tool="read_file",
        path="config.cpp",
        content_hash="source-hash",
        text="std::string_view load_safe() { return storage_; }",
        lines=("7: std::string_view load_safe() { return storage_; }",),
        start_line=7,
    )
    answer = AgentAnswer(
        summary="load_safe returns a view backed by storage_.",
        findings=("Citation: obs_hidden_config line 7.",),
        next_actions=("Keep durable storage.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 7, 7),),
        issue_present=False,
    )

    repaired = _repair_citation_label_findings(answer, (evidence,), frozenset())

    assert repaired == answer


def test_non_code_finding_does_not_require_a_source_symbol() -> None:
    evidence = ToolObservation(
        observation_id="obs_readme",
        tool="read_file",
        path="README.md",
        content_hash="readme-hash",
        text="The investigation mode is read-only by design.",
        lines=("4: The investigation mode is read-only by design.",),
        start_line=4,
    )
    answer = _base(
        "final_answer",
        summary="The documented constraint was inspected.",
        findings=["The documentation describes investigation mode as read-only."],
        next_actions=["Keep the documented constraint."],
        citations=[
            {
                "observation_id": evidence.observation_id,
                "path": evidence.path,
                "start_line": 4,
                "end_line": 4,
            }
        ],
    )
    client = FakeClient([answer])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)

    decision = planner.decide(goal="inspect the documented constraint", catalog=(evidence,))

    assert isinstance(decision, AgentAnswer)
    assert planner.schema_retries == 0


@pytest.mark.parametrize("basename", ("build.sh", "schema.sql", "index.html"))
def test_common_executable_source_formats_require_named_symbols(basename: str) -> None:
    assert _requires_named_source_symbol(basename)


@pytest.mark.parametrize(
    ("path", "source", "symbol"),
    (
        (
            "config.cpp",
            "std::string_view load_safe()\n{\n    return storage_;\n}",
            "load_safe",
        ),
        ("config.hpp", "std::string_view load_safe() const;", "load_safe"),
        ("build.sh", "build_safe()\n{\n    echo ok\n}", "build_safe"),
        ("Config.java", "String loadSafe()\n{\n    return value;\n}", "loadSafe"),
        ("config.cpp", "constexpr int config_limit = 5;", "config_limit"),
        (
            "config.cpp",
            "auto load_safe() -> std::string_view { return storage_; }",
            "load_safe",
        ),
        (
            "config.cpp",
            "Config::Config(std::string value) { storage_ = value; }",
            "Config",
        ),
        ("Config.java", "Config(String value) { this.value = value; }", "Config"),
        ("config.cpp", "std::string storage_ = value;", "storage_"),
        ("config.cpp", "std::string* storage_ = value;", "storage_"),
        ("config.cpp", "std::string& storage_ = value;", "storage_"),
        ("search.js", "searchSafe(term) { return term; }", "searchSafe"),
        ("search.js", "import { searchSafe } from './search.js';", "searchSafe"),
        (
            "search.js",
            "import DefaultSafe, { searchSafe } from './search.js';",
            "searchSafe",
        ),
        (
            "search.js",
            "import {\n  searchSafe,\n} from './search.js';",
            "searchSafe",
        ),
        ("experiment.py", "import torch as th", "th"),
        (
            "experiment.py",
            "from torch import inference_mode as infer",
            "infer",
        ),
        ("Search.kt", "fun searchSafe(term: String) = term", "searchSafe"),
        ("Search.kt", "val searchSafe = { term: String -> term }", "searchSafe"),
        ("lib.rs", "let mut search_safe = value;", "search_safe"),
        ("schema.sql", 'CREATE TABLE "search_results" (id INTEGER);', "search_results"),
        ("index.html", '<div id="search_results"></div>', "search_results"),
    ),
)
def test_common_visible_declarations_require_symbol_named_finding(
    path: str,
    source: str,
    symbol: str,
) -> None:
    source_lines = source.splitlines()
    evidence = ToolObservation(
        observation_id="obs_visible_declaration",
        tool="read_file",
        path=path,
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source_lines, 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The safe path was inspected.",
        findings=("The safe path returns durable storage.",),
        next_actions=("Keep the safe path.",),
        citations=(SourceCitation(evidence.observation_id, path, 1, len(source_lines)),),
        issue_present=False,
    )

    assert symbol in _visible_declaration_symbols(source, path.casefold())
    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each code-source finding must name a concrete source-defined identifier supported "
        "by the cited declaration"
    )


def test_commented_markup_declaration_cannot_ground_a_finding() -> None:
    source = '<!-- <activity android:name=".FakeActivity" /> -->\n<manifest />'
    evidence = ToolObservation(
        observation_id="obs_commented_markup",
        tool="read_file",
        path="AndroidManifest.xml",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The manifest was inspected.",
        findings=("FakeActivity is exported.",),
        next_actions=("Keep exported components intentional.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 2),),
        issue_present=True,
    )

    assert "FakeActivity" not in _visible_declaration_symbols(source, "androidmanifest.xml")
    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each code-source finding must name a concrete source-defined identifier supported "
        "by the cited declaration"
    )


@pytest.mark.parametrize(
    "source",
    (
        '<!-- <activity android:name=".FakeActivity" />\n<manifest />',
        "<script>const text = 'id=\"FakeActivity\"';</script>",
        '<p>The example uses id="FakeActivity" in prose.</p>',
        '<![CDATA[<activity android:name=".FakeActivity" />]]>',
    ),
)
def test_non_tag_markup_text_cannot_declare_a_symbol(source: str) -> None:
    assert "FakeActivity" not in _visible_declaration_symbols(source, "androidmanifest.xml")


def test_truncated_sql_comment_cannot_declare_a_symbol() -> None:
    source = '/*\nCREATE TABLE "fake_results" (id INT);'

    assert "fake_results" not in _visible_declaration_symbols(source, "schema.sql")


@pytest.mark.parametrize("quoted_comment", ("'/*'", "'--'"))
def test_sql_comment_marker_in_string_does_not_hide_later_declaration(
    quoted_comment: str,
) -> None:
    source = f"SELECT {quoted_comment};\nCREATE TABLE real_results (id INT);"

    assert "real_results" in _visible_declaration_symbols(source, "schema.sql")


def test_nested_sql_comment_cannot_declare_a_symbol() -> None:
    source = "/* outer\n/* inner */\nCREATE TABLE fake_results (id INT);\n*/"

    assert "fake_results" not in _visible_declaration_symbols(source, "schema.sql")


def test_sql_dollar_string_masks_fake_declarations_but_not_later_source() -> None:
    source = (
        "SELECT $tag$/*\n"
        "CREATE TABLE fake_results (id INT);\n"
        "$tag$;\n"
        "CREATE TABLE real_results (id INT);"
    )
    symbols = _visible_declaration_symbols(source, "schema.sql")

    assert "fake_results" not in symbols
    assert "real_results" in symbols


@pytest.mark.parametrize(
    "literal",
    (
        "$tag$class FakeResult {}$tag$",
        "'class FakeResult {}'",
        "'first line\nclass FakeResult {}\nlast line'",
    ),
)
def test_sql_strings_do_not_create_generic_declaration_symbols(literal: str) -> None:
    source = f"SELECT {literal};\nCREATE TABLE real_results (id INT);"
    symbols = _visible_declaration_symbols(source, "schema.sql")

    assert "FakeResult" not in symbols
    assert "real_results" in symbols


@pytest.mark.parametrize("prefix", ("", "E"))
def test_sql_backslash_escaped_quote_keeps_fake_create_masked(prefix: str) -> None:
    source = (
        f"SELECT {prefix}'hello \\'\n"
        "CREATE TABLE fake_results (id INT);\n"
        "';\n"
        "CREATE TABLE real_results (id INT);"
    )
    symbols = _visible_declaration_symbols(source, "schema.sql")

    assert "fake_results" not in symbols
    assert "real_results" in symbols


@pytest.mark.parametrize(
    ("opening", "closing"),
    (("[", "]"), ('"', '"'), ("`", "`")),
)
def test_multiline_sql_quoted_span_cannot_create_a_fake_declaration(
    opening: str,
    closing: str,
) -> None:
    source = (
        f"SELECT {opening}hello\n"
        "CREATE TABLE fake_results (id INT);\n"
        f"{closing};\n"
        "CREATE TABLE real_results (id INT);"
    )
    symbols = _visible_declaration_symbols(source, "schema.sql")

    assert "fake_results" not in symbols
    assert "real_results" in symbols


def test_mysql_multiline_double_quote_honors_backslash_escaped_quote() -> None:
    source = (
        'SELECT "hello \\"\n'
        "CREATE TABLE fake_results (id INT);\n"
        '";\n'
        "CREATE TABLE real_results (id INT);"
    )
    symbols = _visible_declaration_symbols(source, "schema.sql")

    assert "fake_results" not in symbols
    assert "real_results" in symbols


def test_oracle_alternative_quote_keeps_fake_create_masked() -> None:
    source = (
        "DECLARE\n"
        " value VARCHAR2(100) := q'[it's text\n"
        "CREATE TABLE FakeResults(id int);\n"
        "]';\n"
        "CREATE TABLE RealResults(id int);\n"
    )
    symbols = _visible_declaration_symbols(source, "schema.sql")

    assert "FakeResults" not in symbols
    assert "RealResults" in symbols


@pytest.mark.parametrize(
    "source",
    (
        "CREATE TABLE IF NOT EXISTS RealResults(id int);",
        "CREATE VIEW IF NOT EXISTS RealResults AS SELECT 1;",
        "CREATE TEMP TABLE IF NOT EXISTS RealResults(id int);",
        "CREATE TEMPORARY VIEW RealResults AS SELECT 1;",
    ),
)
def test_sql_create_options_do_not_become_the_declaration_symbol(source: str) -> None:
    symbols = _visible_declaration_symbols(source, "schema.sql")

    assert "RealResults" in symbols
    assert "IF" not in symbols


@pytest.mark.parametrize("quoted_name", ('"IF"', "`IF`", "[IF]"))
def test_quoted_sql_reserved_word_remains_a_valid_identifier(
    quoted_name: str,
) -> None:
    assert "IF" in _visible_declaration_symbols(
        f"CREATE TABLE {quoted_name}(id int);", "schema.sql"
    )


@pytest.mark.parametrize("basename", ("result.html", "result.xml"))
def test_markup_body_text_does_not_create_generic_declaration_symbols(
    basename: str,
) -> None:
    source = '<div id="real_result">\nclass FakeResult {}\n</div>'
    symbols = _visible_declaration_symbols(source, basename)

    assert "FakeResult" not in symbols
    assert "real_result" in symbols


@pytest.mark.parametrize("basename", ("result.htm", "result.html"))
def test_html_script_body_is_code_but_page_text_is_not(basename: str) -> None:
    source = "<p>function fakeResult() {}</p>\n<script>function realResult() {}</script>"
    symbols = _visible_declaration_symbols(source, basename)

    assert "fakeResult" not in symbols
    assert "realResult" in symbols


@pytest.mark.parametrize(
    "script_type",
    ("text/plain", "application/ld+json", "importmap"),
)
def test_non_executable_html_script_data_is_not_source_code(script_type: str) -> None:
    source = (
        f'<script type="{script_type}">\nclass FakeResult {{}}\n</script>'
        '<div id="real_result"></div>'
    )
    symbols = _visible_declaration_symbols(source, "result.html")

    assert "FakeResult" not in symbols
    assert "real_result" in symbols


@pytest.mark.parametrize(
    "script_type",
    (" text/javascript ", " module ", "text/javascript ; charset=utf-8"),
)
def test_executable_html_script_type_ignores_surrounding_whitespace(
    script_type: str,
) -> None:
    source = f'<script type="{script_type}">\nfunction RealResult() {{}}\n</script>'

    assert "RealResult" in _visible_declaration_symbols(source, "result.html")


@pytest.mark.parametrize(
    "attribute",
    ('id="delete-account"', 'name="user[email]"'),
)
def test_fragmented_html_attribute_is_not_a_standalone_symbol(attribute: str) -> None:
    source = f"<div {attribute}></div>"

    assert not _visible_declaration_symbols(source, "result.html")


@pytest.mark.parametrize("basename", ("Result.vue", "Result.svelte"))
def test_component_template_text_is_not_code_but_script_body_is(
    basename: str,
) -> None:
    source = (
        "<template><div>\nclass FakeResult {}\n</div></template>\n"
        "<script>\nclass RealResult {}\n</script>\n"
    )
    symbols = _visible_declaration_symbols(source, basename)

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize(
    "literal",
    (
        "/hello\nclass FakeResult {}\n/",
        "$/hello\nclass FakeResult {}\n/$",
    ),
)
def test_gradle_multiline_literal_keeps_fake_class_masked(literal: str) -> None:
    source = f"def text = {literal}\nclass RealResult {{}}\n"
    symbols = _visible_declaration_symbols(source, "build.gradle")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


def test_gradle_dollar_slashy_escape_does_not_close_the_literal() -> None:
    source = "def text = $/\n$/$$\nclass FakeResult {}\n/$\nclass RealResult {}\n"
    symbols = _visible_declaration_symbols(source, "build.gradle")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize("basename", ("Result.jsx", "Result.tsx"))
def test_jsx_literal_text_cannot_create_a_fake_declaration(basename: str) -> None:
    source = "export function RealResult() { return <div>\nclass FakeResult {}\n</div>; }"
    symbols = _visible_declaration_symbols(source, basename)

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


def test_jsx_fragment_text_cannot_create_a_fake_declaration() -> None:
    source = "export function RealResult() { return <>\nclass FakeResult {}\n</>; }"
    symbols = _visible_declaration_symbols(source, "Result.jsx")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize("basename", ("Result.jsx", "Result.tsx"))
def test_unmatched_jsx_tag_masks_literal_text_through_observation_end(
    basename: str,
) -> None:
    source = "class Fake {}\nexport function Real() { return <div>\nclass Fake {}\n"
    symbols = _visible_declaration_symbols(source, basename)

    assert {"Fake", "Real"} <= symbols
    evidence = ToolObservation(
        observation_id="obs_truncated_jsx",
        tool="read_file",
        path=basename,
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
        truncated=True,
    )
    answer = AgentAnswer(
        summary="The fake class was inspected.",
        findings=("The `Fake` class fails.",),
        next_actions=("Fix Fake.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 3, 3),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (_SOURCE_SYMBOL_CITATION_ERROR)


@pytest.mark.parametrize(
    "generic_expression",
    (
        "const identity = <T extends object>(value: T) => value;",
        "type Box = Wrapper<Item>;",
        "const identity = <T,>(value: T) => value;",
        "const compared = a < b > c;",
    ),
)
def test_tsx_generic_or_comparison_does_not_hide_later_declaration(
    generic_expression: str,
) -> None:
    source = f"{generic_expression}\nclass RealResult {{}}"

    assert "RealResult" in _visible_declaration_symbols(source, "Result.tsx")


def test_literal_zero_preprocessor_branch_is_not_source_code() -> None:
    source = (
        "#if 0\n"
        "class FakeResult {};\n"
        "#if 1\n"
        "class NestedFakeResult {};\n"
        "#endif\n"
        "#elif 0\n"
        "class OtherFakeResult {};\n"
        "#else\n"
        "class RealResult {};\n"
        "#endif"
    )
    symbols = _visible_declaration_symbols(source, "result.cpp")

    assert "RealResult" in symbols
    assert not {"FakeResult", "NestedFakeResult", "OtherFakeResult"} & symbols


@pytest.mark.parametrize(
    "zero_literal",
    ("00", "0'0", "0x0", "0X00UL", "0b0", "0B00u"),
)
def test_all_zero_integer_literal_preprocessor_branch_is_not_source_code(
    zero_literal: str,
) -> None:
    source = f"#if {zero_literal}\nclass FakeResult {{}};\n#else\nclass RealResult {{}};\n#endif"
    symbols = _visible_declaration_symbols(source, "result.cpp")

    assert "RealResult" in symbols
    assert "FakeResult" not in symbols


def test_literal_zero_nested_branch_inside_unknown_condition_is_masked() -> None:
    source = (
        "#if FEATURE_FLAG\n"
        "#if (0L)\n"
        "class FakeResult {};\n"
        "#else\n"
        "class RealResult {};\n"
        "#endif\n"
        "#endif"
    )
    symbols = _visible_declaration_symbols(source, "result.h")

    assert "RealResult" in symbols
    assert "FakeResult" not in symbols


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("splice", ("\\", "??/"))
@pytest.mark.parametrize(
    ("condition_start", "condition_end"),
    (("#if ", "0"), ("#if (0 ", ")")),
)
def test_spliced_literal_zero_preprocessor_branch_is_not_source_code(
    newline: str,
    splice: str,
    condition_start: str,
    condition_end: str,
) -> None:
    source = newline.join(
        (
            f"{condition_start}{splice}",
            condition_end,
            "class FakeResult {};",
            "#else",
            "class RealResult {};",
            "#endif",
        )
    )
    symbols = _visible_declaration_symbols(source, "result.cpp")

    assert "RealResult" in symbols
    assert "FakeResult" not in symbols


def test_exact_nonzero_preprocessor_branch_masks_later_branches() -> None:
    source = (
        "#if 1\n"
        "class RealResult {};\n"
        "#elif UNKNOWN_FLAG\n"
        "class FakeElifResult {};\n"
        "#else\n"
        "class FakeElseResult {};\n"
        "#endif"
    )
    symbols = _visible_declaration_symbols(source, "result.cpp")

    assert "RealResult" in symbols
    assert not {"FakeElifResult", "FakeElseResult"} & symbols


@pytest.mark.parametrize(
    ("condition", "expected", "excluded"),
    (("false", "RealResult", "FakeResult"), ("true", "FakeResult", "RealResult")),
)
def test_csharp_boolean_preprocessor_condition_masks_inactive_branch(
    condition: str,
    expected: str,
    excluded: str,
) -> None:
    source = f"#if {condition}\nclass FakeResult {{}}\n#else\nclass RealResult {{}}\n#endif"
    symbols = _visible_declaration_symbols(source, "Result.cs")

    assert expected in symbols
    assert excluded not in symbols


def test_c_family_operator_sequence_is_not_an_html_block_comment() -> None:
    source = "bool value = a <!--b;\nclass RealResult {};"

    assert "RealResult" in _visible_declaration_symbols(source, "result.cpp")


def test_ruby_end_marker_makes_remaining_bytes_data() -> None:
    source = "class RealResult\nend\n__END__\nclass FakeResult\nend"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "RealResult" in symbols
    assert "FakeResult" not in symbols


def test_php_halt_compiler_makes_remaining_bytes_data() -> None:
    source = "function RealResult() {}\n__halt_compiler();\nfunction FakeResult() {}"
    symbols = _visible_declaration_symbols(source, "result.php")

    assert "RealResult" in symbols
    assert "FakeResult" not in symbols


@pytest.mark.parametrize(
    ("path", "source", "start_line", "end_line"),
    (
        (
            "fake.cpp",
            "class Fake {};\n#if 0\nclass Fake {};\n#endif",
            3,
            3,
        ),
        (
            "fake.cs",
            "class Fake {}\n#if false\nclass Fake {}\n#endif",
            3,
            3,
        ),
        (
            "fake.rb",
            "class Fake\nend\n__END__\nclass Fake\nend",
            4,
            5,
        ),
        (
            "fake.php",
            "function Fake() {}\n__halt_compiler();\nfunction Fake() {}",
            3,
            3,
        ),
    ),
)
def test_non_code_tail_cannot_ground_a_real_symbol_finding(
    path: str,
    source: str,
    start_line: int,
    end_line: int,
) -> None:
    evidence = ToolObservation(
        observation_id="obs_non_code_tail",
        tool="read_file",
        path=path,
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The fake path was inspected.",
        findings=("`Fake` fails.",),
        next_actions=("Fix Fake.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, start_line, end_line),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (_SOURCE_SYMBOL_CITATION_ERROR)


@pytest.mark.parametrize(
    ("opening", "closing"),
    (("<span>", "</span>"), ("<>", "</>")),
)
def test_jsx_nested_in_braced_expression_masks_literal_text(
    opening: str,
    closing: str,
) -> None:
    source = (
        f"export function RealResult() {{ return <div>{{ok && {opening}\n"
        "class FakeResult {}\n"
        f"{closing}}}</div>; }}"
    )
    symbols = _visible_declaration_symbols(source, "Result.jsx")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize("path", ("result.html", "Result.vue", "Result.svelte"))
def test_markup_script_body_citation_is_reparsed_in_script_context(path: str) -> None:
    source = "<script>\nfunction realResult() {\n  return true;\n}\n</script>"
    evidence = ToolObservation(
        observation_id="obs_script_body",
        tool="read_file",
        path=path,
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The script was inspected.",
        findings=("realResult returns true.",),
        next_actions=("Keep the behavior intentional.",),
        citations=(SourceCitation(evidence.observation_id, path, 2, 3),),
        issue_present=False,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) is None


@pytest.mark.parametrize(
    ("path", "source", "start_line", "end_line"),
    (
        (
            "fake.py",
            "class Fake:\n    pass\nblob = '''\nclass Fake:\n    pass\n'''",
            4,
            5,
        ),
        (
            "fake.cpp",
            "class Fake {};\n/*\nclass Fake {};\n*/",
            3,
            3,
        ),
        (
            "Fake.vue",
            (
                '<script lang="tsx">\n'
                "export function Fake() { return <div>\n"
                "class Fake {}\n"
                "</div>; }\n"
                "</script>"
            ),
            3,
            3,
        ),
    ),
)
def test_cited_declaration_is_masked_in_full_lexical_context(
    path: str,
    source: str,
    start_line: int,
    end_line: int,
) -> None:
    evidence = ToolObservation(
        observation_id="obs_contextual_grounding",
        tool="read_file",
        path=path,
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The fake path was inspected.",
        findings=("`Fake` fails.",),
        next_actions=("Fix Fake.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, start_line, end_line),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (_SOURCE_SYMBOL_CITATION_ERROR)


@pytest.mark.parametrize(
    ("start_line", "incomplete", "redacted"),
    (
        (20, False, False),
        (1, True, False),
        (1, False, True),
    ),
)
def test_code_source_anchor_requires_complete_line_one_lexical_context(
    start_line: int,
    incomplete: bool,
    redacted: bool,
) -> None:
    source = "class Fake:\n    pass"
    evidence = ToolObservation(
        observation_id="obs_missing_lexical_context",
        tool="read_file",
        path="fake.py",
        content_hash="source-hash",
        text=source,
        lines=(f"{start_line}: class Fake:", f"{start_line + 1}:     pass"),
        start_line=start_line,
        incomplete=incomplete,
        redacted=redacted,
    )
    answer = AgentAnswer(
        summary="The class was inspected.",
        findings=("`Fake` has an empty body.",),
        next_actions=("Implement Fake.",),
        citations=(
            SourceCitation(
                evidence.observation_id,
                evidence.path,
                start_line,
                start_line + 1,
            ),
        ),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (_SOURCE_LEXICAL_CONTEXT_ERROR)


def test_delimiter_safe_secret_redaction_preserves_code_source_context(
    tmp_path: Path,
) -> None:
    source = (
        'API_TOKEN = "ivbench-secret-7D4F91A2B6C8E0"\n'
        'WORKSPACE_NOTE = "ordinary note"\n'
        "def enqueue(request):\n"
        "    BILLING_QUEUE.put(request)\n"
    )
    (tmp_path / "billing_worker.py").write_text(source, encoding="utf-8")

    evidence = WorkspaceReader.open(tmp_path).read_file("billing_worker.py")

    assert evidence.incomplete and evidence.redacted
    assert evidence.metadata["lexical_context_preserved"] is True
    assert evidence.lines[0] == '1: API_TOKEN = "[REDACTED_SECRET]"'
    assert "ivbench-secret-7D4F91A2B6C8E0" not in evidence.text
    answer = AgentAnswer(
        summary="The worker queue path was inspected.",
        findings=("`enqueue` puts requests on `BILLING_QUEUE`.",),
        next_actions=("Keep the queue boundary explicit.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 3, 4),),
        issue_present=False,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) is None


def test_secret_redaction_cannot_turn_self_closing_script_into_code_context(
    tmp_path: Path,
) -> None:
    source = '<div id="Fake"></div>\n<script token=abcdefgh/>\nclass Fake {}\n</script>\n'
    (tmp_path / "hostile.html").write_text(source, encoding="utf-8")

    evidence = WorkspaceReader.open(tmp_path).read_file("hostile.html")

    assert evidence.lines[1] == "2: <script token=[REDACTED_SECRET]>"
    assert evidence.metadata["lexical_context_preserved"] is False
    answer = AgentAnswer(
        summary="The alleged class was inspected.",
        findings=("The Fake class fails.",),
        next_actions=("Verify executable script context.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 3, 3),),
        issue_present=True,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (_SOURCE_LEXICAL_CONTEXT_ERROR)


def test_javascript_regex_literal_does_not_hide_later_declaration() -> None:
    source = "/[/*]/.test(token);\nfunction SafeResult(term) { return term; }"

    assert "SafeResult" in _visible_declaration_symbols(source, "result.js")


@pytest.mark.parametrize("quote", ("'", '"'))
def test_unterminated_single_line_string_does_not_hide_later_declaration(
    quote: str,
) -> None:
    source = f"API_[REDACTED_SECRET]{quote}\ndef enqueue(request):\n    BILLING_QUEUE.put(request)"

    assert "enqueue" in _visible_declaration_symbols(source, "billing_worker.py")


@pytest.mark.parametrize("quote", ("'", '"'))
def test_crlf_escaped_string_continuation_keeps_fake_declaration_masked(
    quote: str,
) -> None:
    source = (
        f"marker = {quote}continued\\\r\n"
        "def hidden():\r\n"
        f"{quote}\r\n"
        "def visible():\r\n"
        "    return True\r\n"
    )
    symbols = _explicit_declaration_symbols(source)

    assert "hidden" not in symbols
    assert "visible" in symbols


@pytest.mark.parametrize("quote", ("'", '"'))
def test_cr_only_unterminated_string_resets_before_later_declaration(
    quote: str,
) -> None:
    source = f"marker = {quote}redacted\rdef visible():\r    return True\r"

    assert "visible" in _explicit_declaration_symbols(source)


@pytest.mark.parametrize("quote", ("'", '"'))
def test_cr_only_escaped_string_continuation_keeps_fake_declaration_masked(
    quote: str,
) -> None:
    source = (
        f"marker = {quote}continued\\\rdef hidden():\r{quote}\rdef visible():\r    return True\r"
    )
    symbols = _explicit_declaration_symbols(source)

    assert "hidden" not in symbols
    assert "visible" in symbols


@pytest.mark.parametrize("quote", ("'", '"'))
def test_shell_multiline_quote_keeps_fake_function_masked(quote: str) -> None:
    source = f"payload={quote}hello\nfake() {{ :; }}\n{quote}\nreal() {{ :; }}\n"
    symbols = _visible_declaration_symbols(source, "script.sh")

    assert "fake" not in symbols
    assert "real" in symbols


def test_csharp_verbatim_string_keeps_fake_class_masked() -> None:
    source = 'var text = @"hello\nclass FakeResult {}\n";\nclass RealResult {}\n'
    symbols = _visible_declaration_symbols(source, "Result.cs")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize("opening", ('@$"', '$@"'))
def test_csharp_interpolated_verbatim_string_keeps_fake_class_masked(
    opening: str,
) -> None:
    source = f'var text = {opening}hello\nclass FakeResult {{}}\n";\nclass RealResult {{}}\n'
    symbols = _visible_declaration_symbols(source, "Result.cs")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize("interpolation", ("", "$", "$$"))
def test_csharp_four_quote_raw_string_does_not_close_on_three_quotes(
    interpolation: str,
) -> None:
    source = (
        f'var text = {interpolation}""""\n'
        "class FakeResult {}\n"
        '"""\n'
        '"""";\n'
        "class RealResult {}\n"
    )
    symbols = _visible_declaration_symbols(source, "Result.cs")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize("basename", ("Result.cs", "Result.java"))
def test_cpp_raw_prefix_is_not_applied_to_other_c_like_languages(
    basename: str,
) -> None:
    source = 'var text = R"(redacted\nclass VisibleResult {}\n'

    assert "VisibleResult" in _visible_declaration_symbols(source, basename)


def test_cpp_raw_prefix_is_not_applied_to_c() -> None:
    source = 'const char *text = R"(redacted\nint visible_result(void) { return 1; }\n'

    assert "visible_result" in _visible_declaration_symbols(source, "result.c")


def test_cpp_raw_string_keeps_fake_class_masked() -> None:
    source = 'auto text = R"tag(hello\nclass FakeResult {}\n)tag";\nclass RealResult {};\n'
    symbols = _visible_declaration_symbols(source, "result.cpp")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize(
    "literal",
    (
        '"hello\nfn fake() {}\n"',
        'r#"hello\nfn fake() {}\n"#',
    ),
)
def test_rust_multiline_string_keeps_fake_function_masked(literal: str) -> None:
    source = f"let text = {literal};\nfn real() {{}}\n"
    symbols = _visible_declaration_symbols(source, "result.rs")

    assert "fake" not in symbols
    assert "real" in symbols


@pytest.mark.parametrize(
    ("basename", "source"),
    (
        ("result.rb", 'text = "hello\ndef fake\nend\n"\ndef real\nend\n'),
        (
            "result.php",
            '$text = "hello\nfunction fake() {}\n";\nfunction real() {}\n',
        ),
    ),
)
def test_dynamic_language_multiline_string_keeps_fake_declaration_masked(
    basename: str,
    source: str,
) -> None:
    symbols = _visible_declaration_symbols(source, basename)

    assert "fake" not in symbols
    assert "real" in symbols


def test_shell_single_quote_does_not_treat_backslash_as_an_escape() -> None:
    source = r"payload='trailing\'" + "\nreal() { :; }\n"

    assert "real" in _visible_declaration_symbols(source, "script.sh")


def test_shell_heredoc_keeps_fake_function_masked() -> None:
    source = "cat <<'EOF'\nfake() { :; }\nEOF\nreal() { :; }\n"
    symbols = _visible_declaration_symbols(source, "script.sh")

    assert "fake" not in symbols
    assert "real" in symbols


@pytest.mark.parametrize("opening", ("<<'END-MARKER'", r"<<\END-MARKER"))
def test_shell_quoted_heredoc_supports_non_identifier_delimiters(
    opening: str,
) -> None:
    source = f"cat {opening}\nfake() {{ :; }}\nEND-MARKER\nreal() {{ :; }}\n"
    symbols = _visible_declaration_symbols(source, "script.sh")

    assert "fake" not in symbols
    assert "real" in symbols


@pytest.mark.parametrize("basename", ("script.sh", "script.zsh"))
def test_shell_arithmetic_shift_is_not_treated_as_a_heredoc(basename: str) -> None:
    source = "value=$((1 << COUNT))\nreal() { :; }\n"

    assert "real" in _visible_declaration_symbols(source, basename)


@pytest.mark.parametrize("basename", ("script.sh", "script.zsh"))
def test_shell_arithmetic_command_shift_is_not_treated_as_a_heredoc(
    basename: str,
) -> None:
    source = "((value = 1 << COUNT))\nreal() { :; }\n"

    assert "real" in _visible_declaration_symbols(source, basename)


@pytest.mark.parametrize("quoted_prefix", ('echo "((" ', "printf '(( ' "))
def test_quoted_shell_parentheses_do_not_disable_a_real_heredoc(
    quoted_prefix: str,
) -> None:
    source = f"{quoted_prefix}<<EOF\nfake() {{ :; }}\nEOF\nreal() {{ :; }}\n"
    symbols = _visible_declaration_symbols(source, "script.sh")

    assert "fake" not in symbols
    assert "real" in symbols


def test_shell_heredoc_terminator_rejects_trailing_blanks() -> None:
    source = "cat <<EOF\nEOF \nfake() { :; }\nEOF\nreal() { :; }\n"
    symbols = _visible_declaration_symbols(source, "script.sh")

    assert "fake" not in symbols
    assert "real" in symbols


def test_multiple_shell_heredocs_are_consumed_in_declaration_order() -> None:
    source = "cat <<A <<B\nfirst\nA\nfake() { :; }\nB\nreal() { :; }\n"
    symbols = _visible_declaration_symbols(source, "script.sh")

    assert "fake" not in symbols
    assert "real" in symbols


def test_ruby_heredoc_keeps_fake_method_masked() -> None:
    source = "text = <<~RUBY\n  def fake\n  end\nRUBY\ndef real\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


def test_ruby_quoted_heredoc_supports_non_identifier_delimiter() -> None:
    source = "text = <<'END-MARKER'\ndef fake\nend\nEND-MARKER\ndef real\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


def test_ruby_shift_operator_is_not_treated_as_a_heredoc() -> None:
    source = "items << VALUE\ndef real\nend\n"

    assert "real" in _visible_declaration_symbols(source, "result.rb")


@pytest.mark.parametrize(
    "operator_line",
    ("items = []\nitems <<Widget", "value = 2 <<CONST"),
)
def test_ruby_no_space_shift_operator_is_not_treated_as_a_heredoc(
    operator_line: str,
) -> None:
    source = f"{operator_line}\nclass RealResult\nend\n"

    assert "RealResult" in _visible_declaration_symbols(source, "result.rb")


def test_ruby_heredoc_terminator_rejects_trailing_blanks() -> None:
    source = "text = <<DOC\nDOC \ndef fake\nend\nDOC\ndef real\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


def test_multiple_ruby_heredocs_are_consumed_in_declaration_order() -> None:
    source = "puts <<A, <<B\nfirst\nA\ndef fake\nend\nB\ndef real\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


@pytest.mark.parametrize(
    "literal",
    (
        "%q{hello\ndef fake\nend\n}",
        "%Q(hello\ndef fake\nend\n)",
        "%q{outer { nested }\ndef fake\nend\n}",
        "%r{hello\ndef fake\nend\n}",
        "%{hello\ndef fake\nend\n}",
        "%w[hello\ndef fake\nend\n]",
        "%x{hello\ndef fake\nend\n}",
    ),
)
def test_ruby_percent_literal_keeps_fake_method_masked(literal: str) -> None:
    source = f"text = {literal}\ndef real\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


def test_php_nowdoc_keeps_fake_function_masked() -> None:
    source = "$text = <<<'TXT'\nfunction fake() {}\nTXT;\nfunction real() {}\n"
    symbols = _visible_declaration_symbols(source, "result.php")

    assert "fake" not in symbols
    assert "real" in symbols


@pytest.mark.parametrize(
    "closing_line",
    ('END, "tail"];', 'END , "tail"];', 'END /*comment*/ , "tail"];'),
)
def test_php_flexible_heredoc_closing_marker_preserves_following_tokens(
    closing_line: str,
) -> None:
    source = f"$values = [<<<END\ntext\n{closing_line}\nfunction RealResult() {{}}\n"

    assert "RealResult" in _visible_declaration_symbols(source, "result.php")


def test_ruby_multiline_regex_keeps_fake_method_masked() -> None:
    source = "pattern = /hello\ndef fake\nend\n/\ndef real\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


@pytest.mark.parametrize("operator", ("=~", "!~"))
def test_ruby_match_operator_can_introduce_a_multiline_regex(operator: str) -> None:
    source = f"if value {operator} /hello\ndef fake\nend\n/\nend\ndef real\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


@pytest.mark.parametrize("prefix", ("puts ", "when "))
def test_ruby_command_and_when_can_introduce_a_multiline_regex(prefix: str) -> None:
    source = f"{prefix}/hello\ndef fake\nend\n/\ndef real\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


@pytest.mark.parametrize("prefix", ("pattern = ", "if value =~ "))
def test_unterminated_ruby_multiline_regex_masks_through_eof(prefix: str) -> None:
    source = f"{prefix}/hello\ndef fake\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols


def test_ruby_begin_comment_keeps_fake_method_masked() -> None:
    source = "=begin\ndef fake\nend\n=end\ndef real\nend\n"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


def test_ruby_begin_comment_supports_cr_only_line_endings() -> None:
    source = "=begin\rdef fake\rend\r=end\rdef real\rend\r"
    symbols = _visible_declaration_symbols(source, "result.rb")

    assert "fake" not in symbols
    assert "real" in symbols


def test_swift_extended_multiline_string_keeps_fake_class_masked() -> None:
    source = '#"""\nclass FakeResult {}\n"""#\nclass RealResult {}\n'
    symbols = _visible_declaration_symbols(source, "Result.swift")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize(
    ("basename", "declaration"),
    (("result.rs", "struct"), ("Result.swift", "class")),
)
def test_nested_block_comment_keeps_fake_declaration_masked(
    basename: str,
    declaration: str,
) -> None:
    source = (
        f"/* outer\n/* inner */\n{declaration} FakeResult {{}}\n*/\n{declaration} RealResult {{}}\n"
    )
    symbols = _visible_declaration_symbols(source, basename)

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("kind", ("comment", "macro"))
def test_c_line_splice_keeps_fake_class_masked(newline: str, kind: str) -> None:
    prefix = "// comment \\" if kind == "comment" else "#define DECLARE_FAKE \\"
    source = newline.join(
        (
            prefix,
            "class FakeResult {};",
            "class RealResult {};",
            "",
        )
    )
    symbols = _visible_declaration_symbols(source, "result.cpp")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


def test_jsx_dollar_prefixed_component_masks_literal_child_text() -> None:
    source = (
        "const $Widget = () => null;\n"
        "export function RealResult() { return <$Widget>\n"
        "class FakeResult {}\n"
        "</$Widget>; }"
    )
    symbols = _visible_declaration_symbols(source, "Result.jsx")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize(
    "source",
    (
        "/\\\n*\nclass FakeResult {};\n*/\nclass RealResult {};\n",
        "/*\nclass FakeResult {};\n*\\\n/\nclass RealResult {};\n",
        "/\\\n*\nclass FakeResult {};\n*\\\n/\nclass RealResult {};\n",
    ),
)
def test_cpp_block_comment_delimiters_honor_line_splices(source: str) -> None:
    symbols = _visible_declaration_symbols(source, "result.cpp")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize("basename", ("result.c", "result.h", "result.m"))
@pytest.mark.parametrize(
    "source",
    (
        "/??/\n*\nint FakeResult(void) {}\n*/\nint RealResult(void) {}\n",
        "/*\nint FakeResult(void) {}\n*??/\n/\nint RealResult(void) {}\n",
    ),
)
def test_c_family_trigraph_splices_form_block_comment_delimiters(
    basename: str,
    source: str,
) -> None:
    symbols = _visible_declaration_symbols(source, basename)

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


def test_c_trigraph_splice_continues_a_string_literal() -> None:
    source = (
        'const char *text = "value??/\nint FakeResult(void) {}??/\n";\nint RealResult(void) {}\n'
    )
    symbols = _visible_declaration_symbols(source, "result.c")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


def test_java_unicode_escapes_can_form_block_comment_delimiters() -> None:
    source = "\\u002f\\u002a\nclass FakeResult {}\n\\u002a\\u002f\nclass RealResult {}\n"
    symbols = _visible_declaration_symbols(source, "Result.java")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


def test_java_unicode_escapes_can_form_text_block_delimiters() -> None:
    source = (
        "String value = \\u0022\\u0022\\u0022\n"
        "class FakeResult {}\n"
        "\\u0022\\u0022\\u0022;\n"
        "class RealResult {}\n"
    )
    symbols = _visible_declaration_symbols(source, "Result.java")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


def test_jsx_unicode_component_masks_literal_child_text() -> None:
    source = (
        "const \u00c9vil = () => null;\n"
        "export function RealResult() { return <\u00c9vil>\n"
        "class FakeResult {}\n"
        "</\u00c9vil>; }"
    )
    symbols = _visible_declaration_symbols(source, "Result.jsx")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


def test_jsx_combining_mark_component_masks_literal_child_text() -> None:
    source = (
        "const E\u0301vil = () => null;\n"
        "export function RealResult() { return <E\u0301vil>\n"
        "class FakeResult {}\n"
        "</E\u0301vil>; }"
    )
    symbols = _visible_declaration_symbols(source, "Result.jsx")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize(
    ("source", "expected", "rejected"),
    (
        ("class Fake\\u0045vil {}", "FakeEvil", "Fake"),
        ("class R\\u0065al {}", "Real", "R"),
        ("\\u0063lass Real {}", "Real", "lass"),
        ("// comment \\u000a class Real {}", "Real", "comment"),
    ),
)
def test_java_unicode_escapes_are_translated_before_declaration_extraction(
    source: str,
    expected: str,
    rejected: str,
) -> None:
    symbols = _visible_declaration_symbols(source, "Result.java")

    assert expected in symbols
    assert rejected not in symbols


def test_java_unicode_escape_eligibility_uses_translated_backslash_parity() -> None:
    source = "// \\u005c\\u000a class FakeResult {}\nclass RealResult {}\n"
    symbols = _visible_declaration_symbols(source, "Result.java")

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize("basename", ("Result.vue", "Result.svelte"))
def test_component_tsx_script_masks_jsx_literal_text(basename: str) -> None:
    source = (
        '<script lang="tsx">\n'
        "export function RealResult() { return <div>\n"
        "class FakeResult {}\n"
        "</div>; }\n"
        "</script>"
    )
    symbols = _visible_declaration_symbols(source, basename)

    assert "FakeResult" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize(
    ("basename", "source"),
    (
        ("result.py", "class Fake\u00c9vil:\n    pass\nclass RealResult:\n    pass"),
        ("Result.java", "class Fake\u00c9vil {}\nclass RealResult {}"),
        ("Result.kt", "class Fake\u00c9vil {}\nclass RealResult {}"),
        ("result.js", "class Fake\u00c9vil {}\nclass RealResult {}"),
        ("result.cpp", "class Fake\u00c9vil {};\nclass RealResult {};"),
        ("result.rs", "struct Fake\u00c9vil {}\nstruct RealResult {}"),
        ("result.js", "class Fake$Evil {}\nclass RealResult {}"),
        ("result.py", "class Fake\u0301vil:\n    pass\nclass RealResult:\n    pass"),
        ("result.js", "class Fake\u0301vil {}\nclass RealResult {}"),
        ("Result.java", "class Fake\u0301vil {}\nclass RealResult {}"),
    ),
)
def test_unsupported_identifier_continuation_never_emits_ascii_prefix(
    basename: str,
    source: str,
) -> None:
    symbols = _visible_declaration_symbols(source, basename)

    assert "Fake" not in symbols
    assert "RealResult" in symbols


@pytest.mark.parametrize(
    ("basename", "source", "expected"),
    (
        ("result.rs", "pub fn RealResult() {}", "RealResult"),
        ("result.rs", "pub struct Thing {}", "Thing"),
        ("Result.cs", "internal class RealResult {}", "RealResult"),
        ("result.js", "function* Generate() {}", "Generate"),
        ("result.ts", "declare class RealResult {}", "RealResult"),
        ("Result.kt", "data class RealResult(val id: Int)", "RealResult"),
        ("Result.kt", "object Single {}", "Single"),
        ("Result.scala", "object RealResult {}", "RealResult"),
        ("Result.swift", "actor RealResult {}", "RealResult"),
        ("Result.kt", "enum class RealResult {}", "RealResult"),
        ("result.cpp", "enum struct RealResult {};", "RealResult"),
        ("Result.kt", "annotation class RealResult", "RealResult"),
        ("Result.kt", "value class RealResult(val id: Int)", "RealResult"),
        ("Result.scala", "case class RealResult(id: Int)", "RealResult"),
        ("Result.cs", "record class RealResult {}", "RealResult"),
        ("Result.cs", "record struct RealResult {}", "RealResult"),
        ("Result.cs", "readonly record struct RealResult {}", "RealResult"),
        ("Result.cs", "ref struct RealResult {}", "RealResult"),
    ),
)
def test_common_modified_declaration_forms_are_visible(
    basename: str,
    source: str,
    expected: str,
) -> None:
    assert expected in _visible_declaration_symbols(source, basename)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("class func RealResult() {}", "RealResult"),
        ("class var value: Int { 1 }", "value"),
    ),
)
def test_swift_class_member_modifier_extracts_the_member_name(
    source: str,
    expected: str,
) -> None:
    symbols = _visible_declaration_symbols(source, "Result.swift")

    assert expected in symbols
    assert not ({"func", "var", "subscript"} & symbols)


def test_swift_class_subscript_does_not_emit_a_keyword_symbol() -> None:
    source = "class subscript(index: Int) -> Int { index }"

    assert "subscript" not in _visible_declaration_symbols(source, "Result.swift")


@pytest.mark.parametrize("basename", ("result.py", "result.js"))
@pytest.mark.parametrize("symbol", ("Record", "Object", "Actor", "Module", "Type"))
def test_pascal_identifier_matching_a_keyword_remains_visible(
    basename: str,
    symbol: str,
) -> None:
    assert symbol in _visible_declaration_symbols(f"class {symbol} {{}}", basename)


def test_go_type_and_receiver_method_are_visible_declarations() -> None:
    source = (
        "type Server struct {}\n"
        "func (s *Server) Handle() {}\n"
        "// type Fake struct {}\n"
        "var text = `func (s *Server) Fake() {}`\n"
    )
    symbols = _visible_declaration_symbols(source, "server.go")

    assert {"Server", "Handle"} <= symbols
    assert "Fake" not in symbols


@pytest.mark.parametrize("basename", ("Result.m", "Result.mm"))
def test_objective_c_types_methods_and_c_functions_are_visible(
    basename: str,
) -> None:
    source = "@interface Widget : NSObject\n- (void)performAction;\n@end\nvoid Helper(void) {}\n"
    symbols = _visible_declaration_symbols(source, basename)

    assert {"Widget", "performAction", "Helper"} <= symbols


def test_javascript_arrow_regex_does_not_hide_later_declaration() -> None:
    source = "const check = () => /[/*]/.test(token);\nfunction SafeResult(term) { return term; }"

    assert "SafeResult" in _visible_declaration_symbols(source, "result.js")


@pytest.mark.parametrize(
    "prefix",
    (
        "if (ok) ",
        "do ",
        "if (ok) action(); else ",
        "for (const item of items) ",
    ),
)
def test_javascript_control_flow_regex_does_not_hide_later_declaration(
    prefix: str,
) -> None:
    source = f"{prefix}/[/*]/.test(token);\nfunction SafeResult(term) {{ return term; }}"

    assert "SafeResult" in _visible_declaration_symbols(source, "result.js")


def test_javascript_for_of_expression_regex_does_not_hide_later_declaration() -> None:
    source = "for (const token of /[/*]/.source) {}\nfunction SafeResult(term) { return term; }"

    assert "SafeResult" in _visible_declaration_symbols(source, "result.js")


@pytest.mark.parametrize(
    ("import_line", "local_name"),
    (
        ("import torch as th", "th"),
        ("from torch import inference_mode as infer", "infer"),
        ("from .torch import inference_mode as infer", "infer"),
        ("from . import inference_mode as infer", "infer"),
        (
            "from torch import (\n    inference_mode as infer,\n    no_grad,\n)",
            "infer",
        ),
    ),
)
def test_python_alias_fallback_survives_truncated_rendered_window(
    import_line: str,
    local_name: str,
) -> None:
    source = f"{import_line}\nif ("

    assert local_name in _visible_declaration_symbols(source, "experiment.py")


def test_generic_code_finding_fails_closed_without_an_extracted_symbol() -> None:
    source = "@[/*"
    evidence = ToolObservation(
        observation_id="obs_no_symbol",
        tool="read_file",
        path="result.js",
        content_hash="source-hash",
        text=source,
        lines=(f"1: {source}",),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The source was inspected.",
        findings=("The safe component returns the term.",),
        next_actions=("Keep the safe behavior.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 1, 1),),
        issue_present=False,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each code-source finding must name a concrete source-defined identifier supported "
        "by the cited declaration"
    )


def test_imported_mechanism_does_not_require_a_second_subject_citation() -> None:
    source = (
        "import torch\n"
        "def evaluate_safe(model, loader):\n"
        "    model.eval()\n"
        "    with torch.inference_mode():\n"
        "        return list(loader)\n"
    )
    evidence = ToolObservation(
        observation_id="obs_imported_mechanism",
        tool="read_file",
        path="experiment.py",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The evaluation helper was inspected.",
        findings=("evaluate_safe uses model.eval and torch.inference_mode for inference.",),
        next_actions=("Keep the inference control.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 2, 5),),
        issue_present=False,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) is None


def test_import_only_finding_must_cite_the_import_declaration() -> None:
    source = (
        "import torch\n"
        "def evaluate_safe(model, loader):\n"
        "    model.eval()\n"
        "    with torch.inference_mode():\n"
        "        return list(loader)\n"
    )
    evidence = ToolObservation(
        observation_id="obs_import_only",
        tool="read_file",
        path="experiment.py",
        content_hash="source-hash",
        text=source,
        lines=tuple(f"{index}: {line}" for index, line in enumerate(source.splitlines(), 1)),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The inference module was inspected.",
        findings=("torch provides inference_mode.",),
        next_actions=("Keep the dependency explicit.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 3, 5),),
        issue_present=False,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each source citation must include the source-defined symbol named in its finding "
        "plus the decisive behavior; expand body-only ranges to the declaration"
    )


def test_one_import_cannot_stand_in_for_another_missing_import() -> None:
    source = "import torch\nimport numpy\n"
    evidence = ToolObservation(
        observation_id="obs_two_imports",
        tool="read_file",
        path="experiment.py",
        content_hash="source-hash",
        text=source,
        lines=("1: import torch", "2: import numpy"),
        start_line=1,
    )
    answer = AgentAnswer(
        summary="The inference imports were inspected.",
        findings=("torch and numpy provide inference mechanisms.",),
        next_actions=("Keep both dependencies explicit.",),
        citations=(SourceCitation(evidence.observation_id, evidence.path, 2, 2),),
        issue_present=False,
    )

    assert _grounded_answer_structure_error(answer, (evidence,)) == (
        "each source citation must include the source-defined symbol named in its finding "
        "plus the decisive behavior; expand body-only ranges to the declaration"
    )


def test_empty_recommendations_get_a_fixed_non_evidentiary_default() -> None:
    evidence = _command_observation("obs_parent_failed", "generic.parent_commit")
    answer = AgentAnswer(
        summary="HEAD has no parent.",
        findings=("HEAD is the root commit.",),
        next_actions=(),
        citations=(
            SourceCitation(
                evidence.observation_id,
                evidence.path,
                1,
                1,
            ),
        ),
        issue_present=True,
    )

    repaired = _repair_non_evidentiary_answer_fields(answer)

    assert repaired.findings == answer.findings
    assert repaired.citations == answer.citations
    assert repaired.issue_present is True
    assert repaired.next_actions == ("Review and address the cited findings.",)


def test_duplicate_citation_findings_are_combined_without_reassigning_evidence() -> None:
    citation = SourceCitation("obs_parent", "command/generic.parent_commit", 1, 1)
    head = SourceCitation("obs_head", "command/generic.head_commit", 1, 1)
    answer = AgentAnswer(
        summary="The root and HEAD were checked.",
        findings=(
            "HEAD has no first parent.",
            "HEAD is therefore the root commit.",
            "The current HEAD commit was identified.",
        ),
        next_actions=("Record the result.",),
        citations=(citation, citation, head),
        issue_present=True,
    )

    repaired = _merge_duplicate_citation_findings(answer)

    assert repaired.findings == (
        "HEAD has no first parent. HEAD is therefore the root commit.",
        "The current HEAD commit was identified.",
    )
    assert repaired.citations == (citation, head)
    assert repaired.summary == answer.summary
    assert repaired.next_actions == answer.next_actions
    assert repaired.issue_present is True


def test_duplicate_citation_repair_fails_closed_on_misaligned_arrays() -> None:
    answer = AgentAnswer(
        summary="Malformed answer.",
        findings=("One.", "Two."),
        next_actions=("Review it.",),
        citations=(SourceCitation("obs_1", "a.py", 1, 1),),
        issue_present=True,
    )

    assert _merge_duplicate_citation_findings(answer) is answer


def test_duplicate_citation_repair_does_not_merge_different_observations() -> None:
    first = SourceCitation("obs_first", "shared.py", 4, 4)
    second = SourceCitation("obs_second", "shared.py", 4, 4)
    answer = AgentAnswer(
        summary="Two snapshots used the same source range.",
        findings=("The first snapshot has one state.", "The second has another."),
        next_actions=("Review both snapshots.",),
        citations=(first, second),
        issue_present=True,
    )

    assert _merge_duplicate_citation_findings(answer) is answer


def test_inline_citation_recovery_binds_only_exact_rendered_ranges() -> None:
    first = ToolObservation(
        observation_id="obs_0123456789abcdef",
        tool="read_file",
        path="projects/views.py",
        content_hash="hash-first",
        text="unsafe\nsafe",
        lines=("2: unsafe", "3: safe"),
        start_line=2,
    )
    second = ToolObservation(
        observation_id="obs_fedcba9876543210",
        tool="read_file",
        path="SearchResults.jsx",
        content_hash="hash-second",
        text="unsafe\nsafe",
        lines=("5: unsafe", "6: safe"),
        start_line=5,
    )
    answer = AgentAnswer(
        summary="Both files were inspected.",
        findings=(
            "search_unsafe is vulnerable (obs_0123456789abcdef 2-3).",
            "SafeResult is safe (obs_fedcba9876543210 lines 5-6).",
        ),
        next_actions=("Fix the unsafe path.",),
        citations=(),
        issue_present=True,
    )

    recovered = _recover_inline_citations(
        answer,
        (first, second),
        frozenset({first.observation_id, second.observation_id}),
    )

    assert recovered.citations == (
        SourceCitation(first.observation_id, first.path, 2, 3),
        SourceCitation(second.observation_id, second.path, 5, 6),
    )


def test_inline_citation_recovery_replaces_partial_array_only_when_every_finding_binds() -> None:
    first = ToolObservation(
        observation_id="obs_0123456789abcdef",
        tool="read_file",
        path="a.txt",
        content_hash="hash-first",
        text="unsafe",
        lines=("2: unsafe",),
        start_line=2,
    )
    second = ToolObservation(
        observation_id="obs_fedcba9876543210",
        tool="read_file",
        path="b.py",
        content_hash="hash-second",
        text="safe",
        lines=("5: safe",),
        start_line=5,
    )
    answer = AgentAnswer(
        summary="Both paths were inspected.",
        findings=(
            "Unsafe path (obs_0123456789abcdef 2-2).",
            "Safe path (obs_fedcba9876543210 5-5).",
        ),
        next_actions=("Fix the unsafe path.",),
        citations=(SourceCitation(first.observation_id, first.path, 2, 2),),
        issue_present=True,
    )

    recovered = _recover_inline_citations(
        answer,
        (first, second),
        frozenset({first.observation_id, second.observation_id}),
    )

    assert recovered.citations == (
        SourceCitation(first.observation_id, first.path, 2, 2),
        SourceCitation(second.observation_id, second.path, 5, 5),
    )


@pytest.mark.parametrize(
    "finding",
    (
        "unknown (obs_aaaaaaaaaaaaaaaa 2-3)",
        "outside range (obs_0123456789abcdef 2-99)",
        ("ambiguous (obs_0123456789abcdef 2-3 and obs_0123456789abcdef 2-3)"),
    ),
)
def test_inline_citation_recovery_fails_closed(finding: str) -> None:
    observation = ToolObservation(
        observation_id="obs_0123456789abcdef",
        tool="read_file",
        path="a.py",
        content_hash="hash",
        text="x\ny",
        lines=("2: x", "3: y"),
        start_line=2,
    )
    answer = AgentAnswer(
        summary="Evidence.",
        findings=(finding,),
        next_actions=("Review.",),
        citations=(),
    )

    recovered = _recover_inline_citations(
        answer,
        (observation,),
        frozenset({observation.observation_id}),
    )

    assert recovered.citations == ()


def test_transport_retry_after_schema_failure_keeps_corrective_message() -> None:
    good = _base("list_files", path=".")
    client = FakeClient([PlannerProtocolError("bad json"), PlannerTransportError("net"), good])
    planner = ModelInvestigationPlanner(client=client)

    decision = planner.decide(goal="x", catalog=())

    assert isinstance(decision, ToolCall)
    assert planner.schema_retries == 1
    assert planner.transport_retries == 1
    assert client.calls == 3
    assert json.loads(client.prompts[1])["retry_correction"]
    assert json.loads(client.prompts[2])["retry_correction"]


def test_second_same_class_failure_is_not_retried() -> None:
    # One schema retry only: a second schema-class failure raises (not a third call).
    client = FakeClient([PlannerProtocolError("bad"), PlannerProtocolError("bad")])
    planner = ModelInvestigationPlanner(client=client)
    with pytest.raises(PlannerProtocolError):
        planner.decide(goal="x", catalog=())
    assert client.calls == 2


def test_protocol_failure_retains_validated_usage_metadata() -> None:
    metadata = ModelResponseMetadata(
        model="inverse-gpt-oss-20b",
        prompt_tokens=100,
        completion_tokens=7,
        total_tokens=107,
    )
    client = FakeClient(
        [PlannerProtocolError("bad json")],
        metadata=[metadata],
    )
    planner = ModelInvestigationPlanner(client=client, max_schema_retries=0)

    with pytest.raises(PlannerProtocolError):
        planner.decide(goal="x", catalog=())

    assert planner.completion_tokens_requested == 2048
    assert planner.completion_tokens_charged == 7
    assert planner.completion_tokens_reported == 7
    assert planner.model_calls[0].reported_prompt_tokens == 100
    assert planner.model_calls[0].reported_model == "inverse-gpt-oss-20b"
    assert planner.model_calls[0].outcome == "protocol_error"


def test_unadmitted_schema_retry_preserves_original_failure() -> None:
    client = FakeClient([PlannerProtocolError("malformed decision")])
    planner = ModelInvestigationPlanner(
        client=client,
        max_logical_decisions=1,
        max_completion_tokens=1024,
    )

    with pytest.raises(PlannerProtocolError, match="malformed decision"):
        planner.decide(goal="x", catalog=())

    assert client.calls == 1
    assert planner.requests_made == 1
    assert planner.schema_retries == 0
    assert [call.outcome for call in planner.model_calls] == ["protocol_error"]


def test_transport_and_schema_retries_are_separate() -> None:
    good = _base("list_files", path=".")
    client = FakeClient([PlannerTransportError("net"), PlannerProtocolError("bad"), good])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert planner.transport_retries == 1
    assert planner.schema_retries == 1
    assert client.calls == 3
    assert [call.outcome for call in planner.model_calls] == [
        "transport_error",
        "protocol_error",
        "success",
    ]
    assert all(call.logical_decision == 1 for call in planner.model_calls)


def test_source_revocation_stops_transport_retry_before_second_request() -> None:
    client = FakeClient([PlannerTransportError("net"), _base("list_files", path=".")])
    planner = ModelInvestigationPlanner(client=client)
    guard_results = iter((True, False))
    planner.source_read_guard = lambda: next(guard_results)

    with pytest.raises(PlannerAttestationError, match="source_read"):
        planner.decide(goal="x", catalog=())

    assert client.calls == 1
    assert planner.requests_made == 1
    assert planner.transport_retries == 0


def test_total_request_budget_bounds_calls() -> None:
    client = FakeClient([_base("list_files", path=".")] * 3)
    planner = ModelInvestigationPlanner(client=client, max_total_requests=3)
    for _ in range(3):
        planner.decide(goal="x", catalog=())
    with pytest.raises(PlannerBudgetError, match="request budget"):
        planner.decide(goal="x", catalog=())
    assert planner.requests_made == 3


def test_completion_budget_exhaustion_is_not_schema_retried() -> None:
    client = FakeClient([_base("list_files", path=".")])
    planner = ModelInvestigationPlanner(
        client=client,
        max_logical_decisions=1,
        max_completion_tokens=1024,
    )
    planner.decide(goal="x", catalog=())

    with pytest.raises(PlannerBudgetError, match="completion-token budget"):
        planner.decide(goal="x", catalog=())

    assert client.calls == 1
    assert planner.schema_retries == 0


def test_retry_limits_cannot_exceed_protocol_contract() -> None:
    with pytest.raises(ValueError, match="max_transport_retries"):
        ModelInvestigationPlanner(client=FakeClient([]), max_transport_retries=2)
    with pytest.raises(ValueError, match="max_schema_retries"):
        ModelInvestigationPlanner(client=FakeClient([]), max_schema_retries=2)


def test_maximum_read_probe_defines_context_estimator_boundary() -> None:
    context_tokens = 16_384
    catalog_budget = context_tokens // 2
    encoded_probe_bytes = _maximum_read_probe_tokens(bytes_per_token=1.0)
    boundary = encoded_probe_bytes / catalog_budget

    with pytest.raises(ValueError, match="context/estimator pair"):
        ModelInvestigationPlanner(
            client=FakeClient([]),
            context_tokens=context_tokens,
            estimator_bytes_per_token=boundary - 1e-6,
        )
    planner = ModelInvestigationPlanner(
        client=FakeClient([]),
        context_tokens=context_tokens,
        estimator_bytes_per_token=boundary + 1e-6,
    )
    probe = _maximum_read_probe()
    planner._turn = 1
    allowance = planner._completion_allowance()
    history_budget = planner._history_token_budget(goal="inspect", completion_reserve=allowance)

    _text, ids = _render_catalog(
        (probe,),
        token_budget=history_budget,
        estimator_bytes_per_token=planner.estimator_bytes_per_token,
    )

    assert probe.start_line > 1_000_000
    assert _encoded_string_tokens(probe.text, bytes_per_token=1.0) == 12_000
    redacted_lines = probe.metadata["redacted_lines"]
    assert isinstance(redacted_lines, tuple)
    assert len(redacted_lines) == 199
    assert ids == frozenset({probe.observation_id})


def test_unsupported_action_raises() -> None:
    client = FakeClient([_base("delete_everything")])
    planner = ModelInvestigationPlanner(client=client, max_schema_retries=0)
    with pytest.raises(ValueError, match="unsupported action"):
        planner.decide(goal="x", catalog=())


def test_render_catalog_marks_refusal_and_binary() -> None:
    refused = ToolObservation(
        observation_id="obs_r",
        tool="read_file",
        path="x",
        content_hash="",
        text="[refused] denied",
        metadata={"refused": True},
    )
    binary = ToolObservation(
        observation_id="obs_b",
        tool="read_file",
        path="y",
        content_hash="abc",
        text="",
        metadata={"binary": True},
    )
    rendered, rendered_ids = _render_catalog((refused, binary))
    assert "REFUSED" in rendered
    assert "binary" in rendered
    assert "obs_r" in rendered and "obs_b" in rendered
    # Neither a refusal nor a binary observation is citable.
    assert rendered_ids == frozenset()


def test_render_empty_catalog() -> None:
    text, ids = _render_catalog(())
    assert "no observations" in text
    assert ids == frozenset()


def test_render_catalog_exposes_incomplete_pointer_state() -> None:
    observation = ToolObservation(
        observation_id="obs_partial",
        tool="search_text",
        path=".",
        content_hash="h",
        text="",
        lines=(),
        truncated=True,
        incomplete=True,
        metadata={"oversized_skipped": 1},
    )
    text, ids = _render_catalog((observation,))
    assert "truncated=true" in text
    assert "incomplete=true" in text
    assert "WARNING: this result is incomplete" in text
    assert ids == frozenset()


def test_parse_final_answer_preserves_explicit_incomplete() -> None:
    client = FakeClient(
        [
            _base(
                "final_answer",
                summary="partial",
                findings=["uncertain"],
                complete=False,
            )
        ]
    )
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, AgentAnswer)
    assert decision.complete is False


def test_parse_final_answer_requires_complete_field() -> None:
    payload = _base("final_answer", summary="partial", findings=["uncertain"])
    del payload["complete"]
    with pytest.raises(KeyError, match="complete"):
        parse_decision(payload)


@pytest.mark.parametrize("field", ["complete", "condition_holds"])
def test_parse_final_answer_requires_exact_boolean_fields(field: str) -> None:
    missing = _base("final_answer", summary="done", findings=["grounded"])
    del missing[field]
    with pytest.raises(KeyError, match=field):
        parse_decision(missing)

    wrong_type = _base("final_answer", summary="done", findings=["grounded"])
    wrong_type[field] = "false"
    with pytest.raises(TypeError, match="JSON booleans"):
        parse_decision(wrong_type)


@pytest.mark.parametrize("failure", ["missing", "wrong_type"])
def test_loop_maps_repeated_final_boolean_schema_failure_to_protocol_failure(
    tmp_path: Path, failure: str
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    trust = ScopedTrustStore(tmp_path / "att.sqlite")
    trust.grant(workspace, AttestationScope.SOURCE_READ, granted_by="tester")
    invalid = _base(
        "final_answer",
        summary="done",
        findings=["grounded"],
        next_actions=["verify"],
    )
    if failure == "missing":
        del invalid["complete"]
    else:
        invalid["condition_holds"] = "false"
    client = FakeClient([invalid.copy(), invalid.copy()])
    planner = ModelInvestigationPlanner(
        client=client,
        max_auto_reads=0,
        max_nudges=0,
    )
    report = InvestigationLoop(planner=planner, trust=trust).run(
        run_id=f"r-schema-{failure}",
        goal="inspect workspace",
        workspace=workspace,
    )
    assert client.calls == 2
    assert planner.schema_retries == 1
    assert report.verdict is InvestigationVerdict.FAILED
    assert report.stop_reason is StopReason.PROTOCOL_FAILURE
    assert report.completion_tokens_used == 0
    assert report.completion_tokens_charged == planner.completion_tokens_charged
    assert report.completion_tokens_requested == planner.completion_tokens_requested
    assert report.decisions_used == 1
    assert [call.outcome for call in report.model_calls] == ["schema_error", "schema_error"]


def test_large_read_is_fully_rendered_not_dropped() -> None:
    # A 200-line read (~ up to 12k chars) must be rendered in full and citable,
    # even alongside other observations - never dropped for lack of catalog room.
    big_lines = tuple(f"{i}: " + "x" * 50 for i in range(1, 201))
    big = ToolObservation(
        observation_id="obs_big",
        tool="read_file",
        path="big.py",
        content_hash="h",
        text="x",
        lines=big_lines,
        start_line=1,
    )
    pointer = ToolObservation(
        observation_id="obs_list",
        tool="list_files",
        path=".",
        content_hash="h2",
        text="big.py",
        lines=("big.py",),
    )
    text, ids = _render_catalog((pointer, big))
    assert "obs_big" in ids
    assert "200: " in text  # the last line reached the model


def test_newest_read_survives_large_pointer_bursts() -> None:
    # Codex counterexample: a read followed by two large pointer observations
    # must not crowd out the read - the only citable evidence must survive.
    read = ToolObservation(
        observation_id="obs_read",
        tool="read_file",
        path="evidence.py",
        content_hash="h",
        text="x",
        lines=tuple(f"{i}: line-{i}" for i in range(1, 201)),
        start_line=1,
    )
    big_pointer_lines = tuple(f"path/to/file_{i}.py" for i in range(400))
    pointers = [
        ToolObservation(
            observation_id=f"obs_ptr_{k}",
            tool="list_files",
            path=".",
            content_hash=f"hp{k}",
            text="x",
            lines=big_pointer_lines,
        )
        for k in range(2)
    ]
    # Read is OLDEST; the two big pointers are newest.
    text, ids = _render_catalog((read, *pointers))
    assert "obs_read" in ids  # the citable read is preserved
    assert "200: line-200" in text


def test_recent_observation_survives_when_budget_exceeded() -> None:
    # With many observations, the MOST RECENT read is kept, older context dropped.
    obs = []
    for i in range(40):
        lines = tuple(f"{j}: filler-{i}-{j}" for j in range(1, 60))
        obs.append(
            ToolObservation(
                observation_id=f"obs_{i}",
                tool="read_file",
                path=f"f{i}.py",
                content_hash=f"h{i}",
                lines=lines,
                text="x",
                start_line=1,
            )
        )
    text, ids = _render_catalog(tuple(obs))
    # The last observation must be present and citable; the earliest is dropped.
    assert "obs_39" in ids
    assert "obs_0" not in ids
    assert "omitted for space" in text


def test_catalog_honors_json_encoded_token_budget() -> None:
    observations = tuple(
        ToolObservation(
            observation_id=f"obs_{index}",
            tool="list_files",
            path=".",
            content_hash=f"h{index}",
            text='quoted "\\ value',
            lines=(f'quoted-{index}-"\\-' + "x" * 100,),
        )
        for index in range(20)
    )

    text, ids = _render_catalog(observations, token_budget=512)

    assert _encoded_string_tokens(text, bytes_per_token=2.0) <= 512
    assert ids == frozenset()
    assert "obs_19" in text


def test_calibrated_context_bounds_full_prompt() -> None:
    observations = tuple(
        ToolObservation(
            observation_id=f"obs_{index}",
            tool="list_files",
            path=".",
            content_hash=f"h{index}",
            text="x" * 500,
            lines=(f"path/{index}/" + "x" * 300,),
        )
        for index in range(40)
    )
    client = FakeClient([_base("list_files", path=".")])
    planner = ModelInvestigationPlanner(client=client, context_tokens=24_576)

    planner.decide(goal="inspect the workspace", catalog=observations)

    safety_margin = (planner.context_tokens + 9) // 10
    assert (
        planner._prompt_token_bound(client.prompts[0])
        + client.max_tokens_requested[0]
        + safety_margin
        <= planner.context_tokens
    )


def _compaction_history(count: int = 3) -> tuple[ToolObservation, ...]:
    return tuple(
        ToolObservation(
            observation_id=f"obs_history_{index}",
            tool="read_file",
            path=f"src/history_{index}.py",
            content_hash=str(index) * 64,
            text="bounded source",
            lines=tuple(f"{line}: " + "x" * 80 for line in range(1, 101)),
            start_line=1,
        )
        for index in range(count)
    )


def test_history_compaction_keeps_runtime_catalog_and_non_citable_notes() -> None:
    catalog = _compaction_history()
    client = FakeClient(
        [
            {"notes": "Older source windows were inspected; continue from the recent read."},
            _base("list_files", path="."),
        ]
    )
    planner = ModelInvestigationPlanner(client=client, context_tokens=24_576)
    compacted_events: list[dict[str, object]] = []
    request_events: list[dict[str, int | float | str | None]] = []
    planner.compaction_event_sink = compacted_events.append
    planner.request_event_sink = request_events.append

    decision = planner.decide(goal="inspect", catalog=catalog)

    assert isinstance(decision, ToolCall)
    assert [call.request_kind for call in planner.model_calls] == ["compaction", "decision"]
    assert [event["request_kind"] for event in request_events] == ["compaction", "decision"]
    assert compacted_events == [
        {
            "pinned_notes": planner.pinned_notes,
            "compacted_observation_ids": [
                "obs_history_0",
                "obs_history_1",
            ],
        }
    ]
    decision_prompt = json.loads(client.prompts[-1])
    assert [item["id"] for item in decision_prompt["observation_catalog"]] == [
        observation.observation_id for observation in catalog
    ]
    assert decision_prompt["pinned_investigation_notes"] == planner.pinned_notes
    assert "non-authoritative and never citable" in decision_prompt["notes_authority"]
    assert "obs_history_0" not in decision_prompt["observations"]
    assert "obs_history_2" in decision_prompt["observations"]


def test_compaction_schema_retry_is_corrective_and_fully_accounted() -> None:
    metadata = [
        ModelResponseMetadata("local-model", 100, 8, 108),
        ModelResponseMetadata("local-model", 100, 8, 108),
        ModelResponseMetadata("local-model", 100, 8, 108),
    ]
    client = FakeClient(
        [
            {"notes": ""},
            {"notes": "Concise replacement notes."},
            _base("list_files", path="."),
        ],
        metadata=metadata,
    )
    planner = ModelInvestigationPlanner(client=client, context_tokens=24_576)

    decision = planner.decide(goal="inspect", catalog=_compaction_history())

    assert isinstance(decision, ToolCall)
    assert planner.schema_retries == 1
    assert [call.request_kind for call in planner.model_calls] == [
        "compaction",
        "compaction",
        "decision",
    ]
    assert [call.outcome for call in planner.model_calls] == [
        "schema_error",
        "success",
        "success",
    ]
    assert json.loads(client.prompts[0])["retry_correction"] == ""
    assert "previous response violated" in json.loads(client.prompts[1])["retry_correction"]
    assert "non-empty notes" in json.loads(client.prompts[1])["retry_correction"]


def test_compaction_retry_admission_preserves_original_protocol_failure() -> None:
    client = FakeClient([{"notes": ""}])
    planner = ModelInvestigationPlanner(
        client=client,
        context_tokens=24_576,
        max_logical_decisions=20,
        max_completion_tokens=24_576,
    )

    with pytest.raises(ValueError, match="non-empty string"):
        planner.decide(goal="inspect", catalog=_compaction_history())

    assert client.calls == 1
    assert planner.schema_retries == 0
    assert planner.model_calls[0].request_kind == "compaction"
    assert planner.model_calls[0].outcome == "schema_error"


def test_restored_compaction_state_skips_old_detail_but_keeps_every_index_entry() -> None:
    catalog = _compaction_history()
    client = FakeClient([_base("list_files", path=".")])
    planner = ModelInvestigationPlanner(client=client, context_tokens=24_576)
    planner.pinned_notes = "Durably restored non-authoritative notes."
    planner.compacted_observation_ids = {"obs_history_0", "obs_history_1"}

    planner.decide(goal="inspect", catalog=catalog)

    prompt = json.loads(client.prompts[0])
    assert client.calls == 1
    assert len(prompt["observation_catalog"]) == 3
    assert prompt["pinned_investigation_notes"] == planner.pinned_notes
    assert "obs_history_0" not in prompt["observations"]
    assert "obs_history_2" in prompt["observations"]


def test_twenty_plus_step_loop_preserves_citation_across_repeated_compaction(
    tmp_path: Path,
) -> None:
    file_count = 16
    for index in range(file_count):
        (tmp_path / f"history_{index}.py").write_text(
            "\n".join([f"evidence_{index} = True", *("x" * 80 for _ in range(99))]),
            encoding="utf-8",
        )

    class AdaptiveCompactionClient:
        def __init__(self) -> None:
            self.decisions = 0
            self.last_response_metadata: ModelResponseMetadata | None = None

        def complete_structured_json(
            self,
            *,
            system: str,
            prompt: str,
            schema_name: str,
            schema: Mapping[str, Any],
            max_tokens: int = 4096,
            timeout_seconds: float | None = None,
        ) -> dict[str, Any]:
            del system, schema, max_tokens, timeout_seconds
            self.last_response_metadata = ModelResponseMetadata(
                model="local-model",
                prompt_tokens=1_000,
                completion_tokens=8,
                total_tokens=1_008,
            )
            request = json.loads(prompt)
            if schema_name == "investigation_compaction":
                return {"notes": "Older reads completed; continue with the remaining files."}
            if self.decisions < file_count:
                path = f"history_{self.decisions}.py"
                self.decisions += 1
                return _base("read_file", path=path)
            first = request["observation_catalog"][0]
            return _base(
                "final_answer",
                summary="The first history file contains the expected evidence.",
                findings=["history_0.py sets evidence_0 to true."],
                next_actions=["Keep the evidence assignment."],
                citations=[
                    {
                        "observation_id": first["id"],
                        "path": first["path"],
                        "start_line": 1,
                        "end_line": 1,
                    }
                ],
            )

    trust = ScopedTrustStore(tmp_path / "trust.sqlite")
    trust.grant(tmp_path, AttestationScope.SOURCE_READ, granted_by="test")
    client = AdaptiveCompactionClient()
    planner = ModelInvestigationPlanner(client=client, context_tokens=24_576)
    events: list[tuple[str, dict[str, object]]] = []
    report = InvestigationLoop(
        planner=planner,
        trust=trust,
        event_sink=lambda kind, payload: events.append((kind, payload)),
    ).run(run_id="compaction-loop", goal="inspect history", workspace=tmp_path)

    assert report.verdict is InvestigationVerdict.PASS
    assert report.decisions_used == file_count + 1
    assert report.tool_calls_used == file_count
    assert len(report.catalog) == file_count
    assert report.answer is not None
    assert report.answer.citations[0].observation_id == report.catalog[0].observation_id
    assert report.catalog[0].observation_id in planner.compacted_observation_ids
    assert sum(kind == "investigation.compaction" for kind, _payload in events) >= 2
    assert report.physical_requests_used > report.decisions_used
    assert sum(call.request_kind == "compaction" for call in report.model_calls) >= 2


def test_json_escape_heavy_maximum_read_remains_citable_at_safe_context(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.py"
    source.write_text("\n".join("\\" * 60 for _ in range(200)), encoding="utf-8")
    read = WorkspaceReader.open(tmp_path).read_file("large.py")
    planner = ModelInvestigationPlanner(
        client=FakeClient([]),
        context_tokens=24_576,
        estimator_bytes_per_token=2.0,
    )
    planner._turn = 1
    allowance = planner._completion_allowance()
    history_budget = planner._history_token_budget(goal="inspect", completion_reserve=allowance)

    text, ids = _render_catalog(
        (read,),
        token_budget=history_budget,
        estimator_bytes_per_token=planner.estimator_bytes_per_token,
    )

    assert read.truncated
    assert "\\\\" in text
    assert ids == frozenset({read.observation_id})


def test_redacted_line_metadata_does_not_crowd_out_citable_read_at_16k() -> None:
    read = ToolObservation(
        observation_id="obs_redacted_max",
        tool="read_file",
        path="config.py",
        content_hash="h",
        text="",
        lines=tuple(f"{line}: [REDACTED_SECRET]" for line in range(1, 200))
        + ("200: " + "\a" * 1_300,),
        start_line=1,
        truncated=True,
        incomplete=True,
        redacted=True,
        metadata={"redacted_lines": tuple(range(1, 200, 2))},
    )
    planner = ModelInvestigationPlanner(
        client=FakeClient([]),
        context_tokens=24_576,
        estimator_bytes_per_token=2.0,
    )
    planner._turn = 1
    allowance = planner._completion_allowance()
    history_budget = planner._history_token_budget(goal="inspect", completion_reserve=allowance)

    text, ids = _render_catalog(
        (read,),
        token_budget=history_budget,
        estimator_bytes_per_token=planner.estimator_bytes_per_token,
    )

    assert "non_citable_redacted_lines=1,3,5" in text
    assert ",197,199" in text
    assert ids == frozenset({read.observation_id})


def test_reported_usage_reconciles_conservative_completion_charge() -> None:
    metadata = ModelResponseMetadata(
        model="inverse-gpt-oss-20b",
        prompt_tokens=900,
        completion_tokens=37,
        total_tokens=937,
    )
    client = FakeClient([_base("list_files", path=".")], metadata=[metadata])
    planner = ModelInvestigationPlanner(client=client)

    planner.decide(goal="inspect", catalog=())

    assert planner.completion_tokens_requested == 2048
    assert planner.completion_tokens_charged == 37
    assert planner.completion_tokens_reported == 37
    assert planner.model_calls[0].reported_prompt_tokens == 900
    assert planner.model_calls[0].reported_model == "inverse-gpt-oss-20b"
    assert planner.model_calls[0].outcome == "success"


def test_active_deadline_caps_model_transport_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient([_base("list_files", path=".")])
    planner = ModelInvestigationPlanner(client=client)
    planner.active_deadline = 10.5
    monkeypatch.setattr("inverse_agent.investigation_model.time.monotonic", lambda: 10.0)

    planner.decide(goal="inspect", catalog=())

    assert client.timeouts_requested == [0.5]


def test_render_marks_fully_shown_read_as_citable() -> None:
    obs = ToolObservation(
        observation_id="obs_full",
        tool="read_file",
        path="a.py",
        content_hash="h",
        text="x",
        lines=("1: x", "2: y"),
        start_line=1,
    )
    _text, ids = _render_catalog((obs,))
    assert ids == frozenset({"obs_full"})


def test_render_marks_redacted_lines_non_citable() -> None:
    obs = ToolObservation(
        observation_id="obs_redacted",
        tool="read_file",
        path="config.py",
        content_hash="h",
        text="safe\n[REDACTED_SECRET]",
        lines=("1: safe", "2: [REDACTED_SECRET]"),
        start_line=1,
        incomplete=True,
        redacted=True,
        metadata={"redacted_lines": (2,)},
    )
    text, ids = _render_catalog((obs,))
    assert "non_citable_redacted_lines=2" in text
    assert "2: [REDACTED_SECRET]" in text
    assert ids == frozenset({"obs_redacted"})


def test_repair_only_binds_to_rendered_observation() -> None:
    # A read observation exists but was NOT rendered (not in rendered_ids); a
    # citation that mis-ids it must not be repaired to it.
    from inverse_agent.investigation import AgentAnswer as _Answer
    from inverse_agent.investigation import SourceCitation as _Cite
    from inverse_agent.investigation_model import _repair_citations

    obs = ToolObservation(
        observation_id="obs_hidden",
        tool="read_file",
        path="a.py",
        content_hash="h",
        text="x",
        lines=("1: x", "2: y"),
        start_line=1,
    )
    answer = _Answer(
        summary="s",
        findings=("f",),
        next_actions=(),
        citations=(_Cite("wrong-id", "a.py", 2, 2),),
    )
    repaired = _repair_citations(answer, (obs,), frozenset())
    # Not rebound, because obs_hidden was not rendered to the model.
    assert repaired.citations[0].observation_id == "wrong-id"


def test_repair_does_not_bind_to_redacted_line() -> None:
    from inverse_agent.investigation import AgentAnswer as _Answer
    from inverse_agent.investigation import SourceCitation as _Cite
    from inverse_agent.investigation_model import _repair_citations

    obs = ToolObservation(
        observation_id="obs_redacted",
        tool="read_file",
        path="config.py",
        content_hash="h",
        text="safe\n[REDACTED_SECRET]",
        lines=("1: safe", "2: [REDACTED_SECRET]"),
        metadata={"redacted_lines": (2,)},
    )
    answer = _Answer(
        summary="s",
        findings=("f",),
        next_actions=(),
        citations=(_Cite("wrong-id", "config.py", 2, 2),),
    )
    repaired = _repair_citations(answer, (obs,), frozenset({"obs_redacted"}))
    assert repaired.citations[0].observation_id == "wrong-id"


@pytest.mark.parametrize("observation_id", ["obs_visible", "wrong-id"])
def test_repair_does_not_clamp_oversized_citation_range(observation_id: str) -> None:
    observation = ToolObservation(
        observation_id="obs_visible",
        tool="read_file",
        path="a.py",
        content_hash="hash-visible",
        text="alpha\nbeta",
        lines=("1: alpha", "2: beta"),
        start_line=1,
    )
    original = SourceCitation(observation_id, "a.py", 2, 999)
    answer = AgentAnswer(
        summary="Oversized range.",
        findings=("A claim was made.",),
        next_actions=("Read the missing range.",),
        citations=(original,),
    )

    repaired = _repair_citations(answer, (observation,), frozenset({"obs_visible"}))

    assert repaired.citations == (original,)


def test_ungrounded_answer_is_nudged_to_investigate() -> None:
    # An answer with no grounding (here, no citations and nothing read) should be
    # redirected into further investigation (a list_files), not accepted.
    answer = _base("final_answer", summary="done", findings=["exported"], citations=[])
    client = FakeClient([answer])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert decision.tool == "list_files"


def test_grounded_answer_is_not_nudged() -> None:
    # A citation covered by a real read (even with a mis-copied id) is grounded;
    # the answer proceeds and the citation is repaired, not nudged.
    read = ToolObservation(
        observation_id="obs_real",
        tool="read_file",
        path="a.txt",
        content_hash="h",
        text="x",
        lines=("1: alpha", "2: beta"),
        start_line=1,
    )
    answer = _base(
        "final_answer",
        summary="done",
        findings=["found"],
        next_actions=["verify"],
        citations=[
            {"observation_id": "mis-copied", "path": "a.txt", "start_line": 2, "end_line": 2}
        ],
    )
    client = FakeClient([answer])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=(read,))
    assert isinstance(decision, AgentAnswer)
    assert decision.citations[0].observation_id == "obs_real"


def test_citation_id_brackets_are_stripped() -> None:
    payload = _base(
        "final_answer",
        summary="s",
        findings=["f"],
        citations=[{"observation_id": "[obs_abc]", "path": "a.py", "start_line": 1, "end_line": 1}],
    )
    client = FakeClient([payload])
    planner = ModelInvestigationPlanner(client=client, max_auto_reads=0, max_nudges=0)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, AgentAnswer)
    assert decision.citations[0].observation_id == "obs_abc"


def test_leading_slash_path_is_normalized() -> None:
    client = FakeClient([_base("read_file", path="/src/config.cpp")])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert decision.path == "src/config.cpp"


def test_auto_read_fetches_cited_unread_path() -> None:
    # The model cites a file emitted by completed discovery; the planner reads it.
    listing = ToolObservation(
        observation_id="obs_list",
        tool="list_files",
        path=".",
        content_hash="list-hash",
        text="experiment.py",
        lines=("experiment.py",),
    )
    answer = _base(
        "final_answer",
        summary="s",
        findings=["f"],
        next_actions=["verify"],
        citations=[
            {"observation_id": "obs_x", "path": "experiment.py", "start_line": 2, "end_line": 2}
        ],
    )
    client = FakeClient([answer])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=(listing,))
    assert isinstance(decision, ToolCall)
    assert decision.tool == "read_file"
    assert decision.path == "experiment.py"


def test_auto_read_uses_nonroot_recursive_path_without_double_prefix() -> None:
    listing = ToolObservation(
        observation_id="obs_list",
        tool="list_files",
        path="src",
        content_hash="list-hash",
        text="src/experiment.py",
        lines=("src/experiment.py",),
        metadata={"glob": "**/*", "recursive": True},
    )
    answer = _base(
        "final_answer",
        summary="A candidate was found.",
        findings=["The candidate needs inspection."],
        next_actions=["Verify it."],
        citations=[
            {
                "observation_id": "obs_unread",
                "path": "src/experiment.py",
                "start_line": 2,
                "end_line": 2,
            }
        ],
    )

    decision = ModelInvestigationPlanner(client=FakeClient([answer])).decide(
        goal="inspect the experiment",
        catalog=(listing,),
    )

    assert decision == ToolCall(tool="read_file", path="src/experiment.py", start_line=2)


def test_auto_read_never_dispatches_an_unlisted_or_virtual_citation_path() -> None:
    answer = _base(
        "final_answer",
        summary="s",
        findings=["f"],
        next_actions=["verify"],
        citations=[
            {
                "observation_id": "obs_unknown",
                "path": "command/generic.parent_commit/",
                "start_line": 1,
                "end_line": 1,
            }
        ],
    )
    client = FakeClient([answer])

    decision = ModelInvestigationPlanner(client=client).decide(goal="x", catalog=())

    assert decision == ToolCall(tool="list_files", path=".")


def test_exact_rendered_command_id_repairs_a_miscopied_virtual_path() -> None:
    command = _command_observation("obs_head", "generic.head_commit", status="succeeded")
    answer = _base(
        "final_answer",
        summary="HEAD was identified.",
        findings=["The current HEAD commit was identified."],
        next_actions=["Record it."],
        citations=[
            {
                "observation_id": command.observation_id,
                "path": "generic.head_commit",
                "start_line": 1,
                "end_line": 1,
            }
        ],
    )
    client = FakeClient([answer])

    decision = ModelInvestigationPlanner(client=client).decide(goal="x", catalog=(command,))

    assert isinstance(decision, AgentAnswer)
    assert decision.citations == (SourceCitation(command.observation_id, command.path, 1, 1),)


def test_auto_read_skipped_when_path_already_read() -> None:
    already = ToolObservation(
        observation_id="obs_r",
        tool="read_file",
        path="experiment.txt",
        content_hash="abc",
        text="model.train()",
        lines=("2: model.train()",),
        start_line=2,
    )
    answer = _base(
        "final_answer",
        summary="s",
        findings=["f"],
        next_actions=["verify"],
        citations=[
            {"observation_id": "wrong", "path": "experiment.txt", "start_line": 2, "end_line": 2}
        ],
    )
    client = FakeClient([answer])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=(already,))
    # No auto-read: the answer is returned, with the citation repaired to obs_r.
    assert isinstance(decision, AgentAnswer)
    assert decision.citations[0].observation_id == "obs_r"

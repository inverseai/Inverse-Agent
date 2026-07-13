"""Unit tests for the model-backed investigation planner (fake client)."""

from __future__ import annotations

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
    StopReason,
    ToolCall,
)
from inverse_agent.investigation_model import (
    ModelInvestigationPlanner,
    _encoded_string_tokens,
    _maximum_read_probe,
    _maximum_read_probe_tokens,
    _render_catalog,
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


def test_parse_read_file_decision() -> None:
    client = FakeClient([_base("read_file", path="src/app.py", start_line=5, max_lines=20)])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert decision.tool == "read_file"
    assert decision.path == "src/app.py"
    assert decision.start_line == 5
    assert decision.max_lines == 20


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

    command_client = FakeClient([_base("run_command", path="generic.head_commit")])
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


def test_model_cannot_select_an_unavailable_command() -> None:
    client = FakeClient(
        [
            _base("run_command", path="generic.head_commit"),
            _base("run_command", path="generic.head_commit"),
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
        path="a.py",
        content_hash="h",
        text="x",
        lines=("1: alpha", "2: beta"),
        start_line=1,
    )
    answer = _base(
        "final_answer",
        summary="done",
        findings=["found"],
        citations=[
            {"observation_id": "mis-copied", "path": "a.py", "start_line": 2, "end_line": 2}
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
    # The model answers citing a file it never read; the planner injects a read.
    answer = _base(
        "final_answer",
        summary="s",
        findings=["f"],
        citations=[
            {"observation_id": "obs_x", "path": "experiment.py", "start_line": 2, "end_line": 2}
        ],
    )
    client = FakeClient([answer])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert decision.tool == "read_file"
    assert decision.path == "experiment.py"


def test_auto_read_skipped_when_path_already_read() -> None:
    already = ToolObservation(
        observation_id="obs_r",
        tool="read_file",
        path="experiment.py",
        content_hash="abc",
        text="model.train()",
        lines=("2: model.train()",),
        start_line=2,
    )
    answer = _base(
        "final_answer",
        summary="s",
        findings=["f"],
        citations=[
            {"observation_id": "wrong", "path": "experiment.py", "start_line": 2, "end_line": 2}
        ],
    )
    client = FakeClient([answer])
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=(already,))
    # No auto-read: the answer is returned, with the citation repaired to obs_r.
    assert isinstance(decision, AgentAnswer)
    assert decision.citations[0].observation_id == "obs_r"

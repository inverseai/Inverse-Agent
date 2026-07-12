"""Unit tests for the model-backed investigation planner (fake client)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from inverse_agent.fs_tools import ToolObservation
from inverse_agent.investigation import AgentAnswer, ToolCall
from inverse_agent.investigation_model import (
    ModelInvestigationPlanner,
    _render_catalog,
)
from inverse_agent.planner import (
    PlannerError,
    PlannerProtocolError,
    PlannerTransportError,
)


class FakeClient:
    """Returns queued payloads; raises queued exceptions to exercise retry."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = responses
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
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _base(action: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "action": action,
        "path": "",
        "query": "",
        "glob": "",
        "start_line": 1,
        "max_lines": 200,
        "summary": "",
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


def test_parse_final_answer_with_citation() -> None:
    payload = _base(
        "final_answer",
        summary="done",
        findings=["found the bug"],
        next_actions=["fix it"],
        citations=[
            {"observation_id": "obs_1", "path": "a.py", "start_line": 2, "end_line": 3}
        ],
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


def test_second_same_class_failure_is_not_retried() -> None:
    # One schema retry only: a second schema-class failure raises (not a third call).
    client = FakeClient([PlannerProtocolError("bad"), PlannerProtocolError("bad")])
    planner = ModelInvestigationPlanner(client=client)
    with pytest.raises(PlannerProtocolError):
        planner.decide(goal="x", catalog=())
    assert client.calls == 2


def test_transport_and_schema_retries_are_separate() -> None:
    good = _base("list_files", path=".")
    client = FakeClient(
        [PlannerTransportError("net"), PlannerProtocolError("bad"), good]
    )
    planner = ModelInvestigationPlanner(client=client)
    decision = planner.decide(goal="x", catalog=())
    assert isinstance(decision, ToolCall)
    assert planner.transport_retries == 1
    assert planner.schema_retries == 1
    assert client.calls == 3


def test_total_request_budget_bounds_calls() -> None:
    client = FakeClient([PlannerTransportError("net")] * 10)
    planner = ModelInvestigationPlanner(
        client=client, max_transport_retries=100, max_total_requests=3
    )
    with pytest.raises(PlannerError):
        planner.decide(goal="x", catalog=())
    assert planner.requests_made == 3


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
        citations=[
            {"observation_id": "[obs_abc]", "path": "a.py", "start_line": 1, "end_line": 1}
        ],
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

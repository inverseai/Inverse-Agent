import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import inverse_agent.planner as planner_module
from inverse_agent.cli import _service, main
from inverse_agent.investigation_model import ModelInvestigationPlanner
from inverse_agent.model_config import PlannerConfig, resolve_planner
from inverse_agent.models import AutonomyLevel, Domain, RunKind
from inverse_agent.planner import (
    DeterministicPlanner,
    OpenAICompatibleClient,
    PlannerProtocolError,
    PlannerTransportError,
    StructuredPlanner,
    validate_model_endpoint,
)

FIXTURES = Path(__file__).parent / "fixtures"
APPROVAL_SECRET = "test-model-secret-that-is-at-least-32-bytes"


class _ModelServer(HTTPServer):
    response_body = b""
    response_status = 200
    response_location: str | None = None
    authorization: str | None = None
    request_payload: dict[str, object] | None = None
    chunk_delay_seconds = 0.0
    content_length: int | None = None


class _ModelHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server = self.server
        assert isinstance(server, _ModelServer)
        size = int(self.headers.get("Content-Length", "0"))
        server.request_payload = json.loads(self.rfile.read(size))
        server.authorization = self.headers.get("Authorization")
        self.send_response(server.response_status)
        if server.response_location:
            self.send_header("Location", server.response_location)
        self.send_header("Content-Type", "application/json")
        if server.content_length is not None:
            self.send_header("Content-Length", str(server.content_length))
        self.end_headers()
        if server.chunk_delay_seconds:
            try:
                for value in server.response_body:
                    self.wfile.write(bytes((value,)))
                    self.wfile.flush()
                    time.sleep(server.chunk_delay_seconds)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.wfile.write(server.response_body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _response(content: str, *, model: str = "endpoint-reported-model") -> bytes:
    return json.dumps({"model": model, "choices": [{"message": {"content": content}}]}).encode()


def _serve(
    body: bytes,
    *,
    status: int = 200,
    location: str | None = None,
    chunk_delay_seconds: float = 0.0,
    content_length: int | None = None,
):
    server = _ModelServer(("127.0.0.1", 0), _ModelHandler)
    server.response_body = body
    server.response_status = status
    server.response_location = location
    server.chunk_delay_seconds = chunk_delay_seconds
    server.content_length = content_length
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop(server: _ModelServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_resolver_defaults_to_deterministic() -> None:
    resolution = resolve_planner(env={})
    assert isinstance(resolution.planner, DeterministicPlanner)
    assert resolution.config.fingerprint == "deterministic"


def test_resolver_requires_complete_model_configuration() -> None:
    with pytest.raises(ValueError, match="configured together"):
        resolve_planner(env={"INVERSE_AGENT_MODEL_NAME": "openai/gpt-oss-20b"})
    with pytest.raises(ValueError, match="configured together"):
        resolve_planner(env={"INVERSE_AGENT_MODEL_BASE_URL": "http://127.0.0.1:1234/v1"})


def test_cli_values_override_environment() -> None:
    args = argparse.Namespace(
        model="cli/model",
        model_base_url="http://127.0.0.1:1234/v1",
        model_timeout_seconds=10,
        model_max_actions=4,
        model_allow_remote=False,
    )
    resolution = resolve_planner(
        args=args,
        env={
            "INVERSE_AGENT_MODEL_NAME": "env/model",
            "INVERSE_AGENT_MODEL_BASE_URL": "http://localhost:9999/v1",
            "INVERSE_AGENT_MODEL_TIMEOUT_SECONDS": "20",
            "INVERSE_AGENT_MODEL_MAX_ACTIONS": "7",
        },
    )
    assert isinstance(resolution.planner, StructuredPlanner)
    assert resolution.config.model == "cli/model"
    assert resolution.config.timeout_seconds == 10
    assert resolution.config.max_actions == 4


def test_api_key_is_excluded_from_config_output() -> None:
    resolution = resolve_planner(
        env={
            "INVERSE_AGENT_MODEL_NAME": "openai/gpt-oss-20b",
            "INVERSE_AGENT_MODEL_BASE_URL": "http://127.0.0.1:1234/v1",
            "INVERSE_AGENT_MODEL_API_KEY": "super-secret-model-key",
        }
    )
    assert "super-secret-model-key" not in repr(resolution.config)
    assert "super-secret-model-key" not in json.dumps(resolution.config.safe_summary())
    assert "super-secret-model-key" not in resolution.config.fingerprint


def test_investigation_calibration_is_paired_validated_and_fingerprinted() -> None:
    base = {
        "INVERSE_AGENT_MODEL_NAME": "openai/gpt-oss-20b",
        "INVERSE_AGENT_MODEL_BASE_URL": "http://127.0.0.1:1234/v1",
    }
    with pytest.raises(ValueError, match="configured together"):
        resolve_planner(env={**base, "INVERSE_AGENT_MODEL_CONTEXT_TOKENS": "32768"})
    with pytest.raises(ValueError, match="must be one of"):
        resolve_planner(
            env={
                **base,
                "INVERSE_AGENT_MODEL_CONTEXT_TOKENS": "20000",
                "INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN": "2.0",
            }
        )

    calibrated = resolve_planner(
        env={
            **base,
            "INVERSE_AGENT_MODEL_CONTEXT_TOKENS": "32768",
            "INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN": "2.0",
        }
    )
    uncalibrated = resolve_planner(env=base)
    assert calibrated.config.investigation_available
    assert calibrated.config.safe_summary()["context_tokens"] == 32768
    assert calibrated.config.fingerprint != uncalibrated.config.fingerprint


def test_remote_calibrated_model_is_not_available_for_investigation() -> None:
    resolution = resolve_planner(
        args=argparse.Namespace(model_allow_remote=True),
        env={
            "INVERSE_AGENT_MODEL_NAME": "provider/model",
            "INVERSE_AGENT_MODEL_BASE_URL": "https://models.example.test/v1",
            "INVERSE_AGENT_MODEL_ALLOW_REMOTE": "1",
            "INVERSE_AGENT_MODEL_CONTEXT_TOKENS": "32768",
            "INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN": "2.0",
        },
    )
    assert not resolution.config.investigation_available


def test_cli_service_wires_calibrated_model_investigation_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "manage.py").write_text("print('ok')\n", encoding="utf-8")
    resolution = resolve_planner(
        env={
            "INVERSE_AGENT_MODEL_NAME": "openai/gpt-oss-20b",
            "INVERSE_AGENT_MODEL_BASE_URL": "http://127.0.0.1:1234/v1",
            "INVERSE_AGENT_MODEL_CONTEXT_TOKENS": "32768",
            "INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN": "2.0",
        }
    )
    monkeypatch.setenv("INVERSE_AGENT_APPROVAL_SECRET", APPROVAL_SECRET)
    service = _service(
        argparse.Namespace(state_dir=str(tmp_path / "state")),
        workspace,
        resolution=resolution,
    )
    try:
        created = service.create_run(
            goal="Inspect the workspace",
            workspace=workspace,
            domain=Domain.DJANGO,
            kind=RunKind.INVESTIGATION,
            autonomy_level=AutonomyLevel.ASSISTED,
        )
        assert service.investigation_planner_factory is not None
        planner = service.investigation_planner_factory(created)
    finally:
        service.close()

    assert isinstance(planner, ModelInvestigationPlanner)
    assert planner.context_tokens == 32768
    assert "django.check" in planner.allowed_commands


def test_planner_fingerprint_covers_non_secret_runtime_configuration() -> None:
    baseline = PlannerConfig(
        kind="openai-compatible",
        model="model",
        base_url="http://127.0.0.1:1234/v1",
        timeout_seconds=60,
        max_actions=8,
    )
    changed_timeout = PlannerConfig(
        kind="openai-compatible",
        model="model",
        base_url="http://127.0.0.1:1234/v1",
        timeout_seconds=61,
        max_actions=8,
    )
    assert baseline.fingerprint != changed_timeout.fingerprint


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/model",
        "ftp://127.0.0.1/model",
        "http://user:password@127.0.0.1:1234/v1",
        "http://127.0.0.1:1234/v1?token=value",
        "http://example.test/v1",
        "http://localhost:1234/v1",
        "https://LOCALHOST:1234/v1",
        "http://127.0.0.1:1234/v1\nignored",
        "http://127.0.0.1:1234/v1\x7fignored",
        "http://127.0.0.1:99999/v1",
    ],
)
def test_endpoint_validation_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(ValueError):
        validate_model_endpoint(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.2:1234/v1",
        "http://[::1]:1234/v1",
        "https://127.0.0.1:1234/v1",
    ],
)
def test_endpoint_validation_accepts_loopback_variants(url: str) -> None:
    assert validate_model_endpoint(url) == url


def test_remote_endpoint_requires_dual_opt_in_and_https() -> None:
    env = {
        "INVERSE_AGENT_MODEL_NAME": "provider/model",
        "INVERSE_AGENT_MODEL_BASE_URL": "https://models.example.test/v1",
        "INVERSE_AGENT_MODEL_ALLOW_REMOTE": "1",
    }
    with pytest.raises(ValueError, match="dual opt-in"):
        resolve_planner(env=env)
    with pytest.raises(ValueError, match="dual opt-in"):
        resolve_planner(
            args=argparse.Namespace(model_allow_remote=True),
            env={
                key: value
                for key, value in env.items()
                if key != "INVERSE_AGENT_MODEL_ALLOW_REMOTE"
            },
        )
    args = argparse.Namespace(model_allow_remote=True)
    resolution = resolve_planner(args=args, env=env)
    assert resolution.config.allow_remote
    with pytest.raises(ValueError, match="must use https"):
        resolve_planner(
            args=args,
            env={**env, "INVERSE_AGENT_MODEL_BASE_URL": "http://models.example.test/v1"},
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("INVERSE_AGENT_MODEL_TIMEOUT_SECONDS", "0"),
        ("INVERSE_AGENT_MODEL_TIMEOUT_SECONDS", "601"),
        ("INVERSE_AGENT_MODEL_MAX_ACTIONS", "0"),
        ("INVERSE_AGENT_MODEL_MAX_ACTIONS", "33"),
    ],
)
def test_model_configuration_rejects_out_of_range_values(name: str, value: str) -> None:
    env = {
        "INVERSE_AGENT_MODEL_NAME": "model",
        "INVERSE_AGENT_MODEL_BASE_URL": "http://127.0.0.1:1234/v1",
        name: value,
    }
    with pytest.raises(ValueError, match="must be between"):
        resolve_planner(env=env)


def test_openai_compatible_client_round_trip_and_authorization() -> None:
    response_payload = {"actions": ["generic.inspect"], "rationale": "Inspect the workspace"}
    server, thread = _serve(_response(json.dumps(response_payload)))
    try:
        host, port = server.server_address
        client = OpenAICompatibleClient(
            base_url=f"http://{host}:{port}/v1",
            model="openai/gpt-oss-20b",
            api_key="secret-model-key",
        )
        result = client.complete_json(system="system", prompt="prompt")
    finally:
        _stop(server, thread)
    assert result == response_payload
    assert client.observed_response_models == ("endpoint-reported-model",)
    assert client.successful_response_count == 1
    assert client.attributed_response_count == 1
    assert server.authorization == "Bearer secret-model-key"
    assert server.request_payload and server.request_payload["temperature"] == 0
    assert server.request_payload["max_tokens"] == 4096
    assert server.request_payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "inverse_agent_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                },
                "required": ["actions", "rationale"],
                "additionalProperties": False,
            },
        },
    }


def test_openai_compatible_client_accepts_a_bounded_custom_schema() -> None:
    server, thread = _serve(_response(json.dumps({"verdict": "PASS"})))
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
        "additionalProperties": False,
    }
    try:
        host, port = server.server_address
        client = OpenAICompatibleClient(base_url=f"http://{host}:{port}/v1", model="model")
        result = client.complete_structured_json(
            system="system",
            prompt="prompt",
            schema_name="commit_review",
            schema=schema,
            max_tokens=512,
        )
    finally:
        _stop(server, thread)

    assert result == {"verdict": "PASS"}
    assert server.request_payload and server.request_payload["max_tokens"] == 512
    response_format = server.request_payload["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["json_schema"] == {
        "name": "commit_review",
        "strict": True,
        "schema": schema,
    }


def test_openai_compatible_client_retains_validated_usage_metadata() -> None:
    content = json.dumps({"actions": ["generic.inspect"], "rationale": "Inspect"})
    body = json.dumps(
        {
            "model": "endpoint-model",
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 17,
                "total_tokens": 140,
            },
            "choices": [{"message": {"content": content}}],
        }
    ).encode()
    server, thread = _serve(body)
    try:
        host, port = server.server_address
        client = OpenAICompatibleClient(base_url=f"http://{host}:{port}/v1", model="model")
        client.complete_json(system="system", prompt="prompt")
    finally:
        _stop(server, thread)

    metadata = client.last_response_metadata
    assert metadata is not None
    assert metadata.model == "endpoint-model"
    assert metadata.prompt_tokens == 123
    assert metadata.completion_tokens == 17
    assert metadata.total_tokens == 140


def test_openai_compatible_client_retains_usage_on_protocol_failure() -> None:
    body = json.dumps(
        {
            "model": "endpoint-model",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 7,
                "total_tokens": 107,
            },
            "choices": [{"message": {"content": "not-json"}}],
        }
    ).encode()
    server, thread = _serve(body)
    try:
        host, port = server.server_address
        client = OpenAICompatibleClient(base_url=f"http://{host}:{port}/v1", model="model")
        with pytest.raises(PlannerProtocolError, match="invalid response"):
            client.complete_json(system="system", prompt="prompt")
    finally:
        _stop(server, thread)

    metadata = client.last_response_metadata
    assert metadata is not None
    assert metadata.model == "endpoint-model"
    assert metadata.prompt_tokens == 100
    assert metadata.completion_tokens == 7
    assert client.successful_response_count == 1
    assert client.attributed_response_count == 1
    assert client.successful_response_models == ("endpoint-model",)


@pytest.mark.parametrize(
    "usage",
    (
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 99},
        {"prompt_tokens": 1, "completion_tokens": -1, "total_tokens": 0},
        {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
    ),
)
def test_openai_compatible_client_rejects_invalid_usage(usage: dict[str, object]) -> None:
    content = json.dumps({"actions": ["generic.inspect"], "rationale": "Inspect"})
    body = json.dumps(
        {
            "model": "endpoint-model",
            "usage": usage,
            "choices": [{"message": {"content": content}}],
        }
    ).encode()
    server, thread = _serve(body)
    try:
        host, port = server.server_address
        client = OpenAICompatibleClient(base_url=f"http://{host}:{port}/v1", model="model")
        with pytest.raises(PlannerProtocolError, match="invalid response"):
            client.complete_json(system="system", prompt="prompt")
    finally:
        _stop(server, thread)


def test_openai_compatible_client_counts_success_without_reported_model() -> None:
    response_payload = {"actions": ["generic.inspect"], "rationale": "Inspect"}
    body = json.dumps(
        {"choices": [{"message": {"content": json.dumps(response_payload)}}]}
    ).encode()
    server, thread = _serve(body)
    try:
        host, port = server.server_address
        client = OpenAICompatibleClient(
            base_url=f"http://{host}:{port}/v1",
            model="requested-model",
        )
        assert client.complete_json(system="system", prompt="prompt") == response_payload
    finally:
        _stop(server, thread)

    assert client.observed_response_models == ()
    assert client.successful_response_count == 1
    assert client.attributed_response_count == 0


def test_openai_compatible_client_bounds_response_model_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(planner_module, "MAX_SUCCESSFUL_RESPONSE_MODEL_HISTORY", 3)
    monkeypatch.setattr(planner_module, "MAX_OBSERVED_RESPONSE_MODELS", 2)
    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:1234/v1",
        model="requested-model",
    )

    for index in range(5):
        client._record_successful_response(f"reported-model-{index}")

    assert client.successful_response_count == 5
    assert client.successful_response_models == (
        "reported-model-2",
        "reported-model-3",
        "reported-model-4",
    )
    assert client.successful_response_models_since(0) is None
    assert client.successful_response_models_since(2) == client.successful_response_models
    assert client.successful_response_models_since(4) == ("reported-model-4",)
    assert client.observed_response_models == ("reported-model-0", "reported-model-1")
    assert client.observed_response_models_overflowed
    assert client.response_model_mismatch_observed


@pytest.mark.parametrize(
    "content",
    [
        {"verdict": 1},
        {},
        {"verdict": "PASS", "unexpected": True},
    ],
)
def test_openai_compatible_client_validates_structured_output_locally(
    content: dict[str, object],
) -> None:
    server, thread = _serve(_response(json.dumps(content)))
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
        "additionalProperties": False,
    }
    try:
        host, port = server.server_address
        client = OpenAICompatibleClient(base_url=f"http://{host}:{port}/v1", model="model")
        with pytest.raises(PlannerProtocolError, match="requested JSON schema"):
            client.complete_structured_json(
                system="system",
                prompt="prompt",
                schema_name="commit_review",
                schema=schema,
            )
    finally:
        _stop(server, thread)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("[" * 1200 + "0" + "]" * 1200, id="deeply-nested"),
        pytest.param('{"verdict": ' + "9" * 10_000 + "}", id="oversized-integer"),
    ],
)
def test_openai_compatible_client_contains_pathological_json_errors(content: str) -> None:
    server, thread = _serve(_response(content))
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
        "additionalProperties": False,
    }
    try:
        host, port = server.server_address
        client = OpenAICompatibleClient(base_url=f"http://{host}:{port}/v1", model="model")
        with pytest.raises(PlannerProtocolError):
            client.complete_structured_json(
                system="system",
                prompt="prompt",
                schema_name="commit_review",
                schema=schema,
            )
    finally:
        _stop(server, thread)


def test_openai_compatible_client_rejects_invalid_custom_schema_controls() -> None:
    client = OpenAICompatibleClient(base_url="http://127.0.0.1:1234/v1", model="model")

    with pytest.raises(ValueError, match="schema name"):
        client.complete_structured_json(
            system="system",
            prompt="prompt",
            schema_name="invalid-name",
            schema={},
        )
    with pytest.raises(ValueError, match="max tokens"):
        client.complete_structured_json(
            system="system",
            prompt="prompt",
            schema_name="valid_name",
            schema={},
            max_tokens=4097,
        )
    with pytest.raises(ValueError, match="schema is invalid"):
        client.complete_structured_json(
            system="system",
            prompt="prompt",
            schema_name="valid_name",
            schema={"type": "not-a-json-schema-type"},
        )
    with pytest.raises(ValueError, match="request timeout"):
        client.complete_structured_json(
            system="system",
            prompt="prompt",
            schema_name="valid_name",
            schema={},
            timeout_seconds=0,
        )


def test_openai_compatible_client_rejects_redirects_and_oversized_output() -> None:
    redirect, redirect_thread = _serve(b"", status=302, location="http://127.0.0.1/")
    try:
        host, port = redirect.server_address
        client = OpenAICompatibleClient(base_url=f"http://{host}:{port}/v1", model="model")
        with pytest.raises(PlannerTransportError, match="HTTP 302"):
            client.complete_json(system="system", prompt="prompt")
    finally:
        _stop(redirect, redirect_thread)

    oversized, oversized_thread = _serve(b"x" * (1024 * 1024 + 1))
    try:
        host, port = oversized.server_address
        client = OpenAICompatibleClient(base_url=f"http://{host}:{port}/v1", model="model")
        with pytest.raises(PlannerProtocolError, match="size limit"):
            client.complete_json(system="system", prompt="prompt")
    finally:
        _stop(oversized, oversized_thread)

    declared, declared_thread = _serve(b"", content_length=1024 * 1024 + 1)
    try:
        host, port = declared.server_address
        client = OpenAICompatibleClient(base_url=f"http://{host}:{port}/v1", model="model")
        with pytest.raises(PlannerProtocolError, match="size limit"):
            client.complete_json(system="system", prompt="prompt")
    finally:
        _stop(declared, declared_thread)


def test_openai_compatible_client_enforces_total_response_deadline() -> None:
    slow, slow_thread = _serve(
        b"123456",
        chunk_delay_seconds=0.25,
    )
    started = time.monotonic()
    try:
        host, port = slow.server_address
        client = OpenAICompatibleClient(
            base_url=f"http://{host}:{port}/v1",
            model="model",
            timeout_seconds=1,
        )
        with pytest.raises(PlannerTransportError, match="timed out"):
            client.complete_json(system="system", prompt="prompt")
    finally:
        _stop(slow, slow_thread)
    assert time.monotonic() - started < 2.5


def test_model_check_uses_structured_planner_without_agent_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server, thread = _serve(
        _response(
            json.dumps({"actions": ["generic.inspect"], "rationale": "local model is available"})
        )
    )
    try:
        host, port = server.server_address
        monkeypatch.setenv("INVERSE_AGENT_MODEL_NAME", "openai/gpt-oss-20b")
        monkeypatch.setenv("INVERSE_AGENT_MODEL_BASE_URL", f"http://{host}:{port}/v1")
        code = main(["model-check"])
    finally:
        _stop(server, thread)
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["ok"] is True
    assert output["actions"] == ["generic.inspect"]


def test_model_check_redacts_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_resolution(**_kwargs):
        raise ValueError("api_key=supersecretvalue")

    monkeypatch.setattr("inverse_agent.cli.resolve_planner", fail_resolution)
    assert main(["model-check"]) == 1
    stderr = capsys.readouterr().err
    assert "supersecretvalue" not in stderr
    assert "[REDACTED_SECRET]" in stderr


def test_cli_start_returns_failure_when_model_plan_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server, thread = _serve(
        _response(json.dumps({"actions": ["unknown.tool"], "rationale": "Unknown"}))
    )
    state_dir = tmp_path / "state"
    try:
        host, port = server.server_address
        monkeypatch.setenv("INVERSE_AGENT_APPROVAL_SECRET", APPROVAL_SECRET)
        monkeypatch.setenv("INVERSE_AGENT_MODEL_NAME", "openai/gpt-oss-20b")
        monkeypatch.setenv("INVERSE_AGENT_MODEL_BASE_URL", f"http://{host}:{port}/v1")
        trust_code = main(
            [
                "trust-workspace",
                str(FIXTURES / "django_project"),
                "--trusted-by",
                "tester",
                "--workspace-root",
                str(FIXTURES),
                "--state-dir",
                str(state_dir),
            ]
        )
        capsys.readouterr()
        start_code = main(
            [
                "start",
                str(FIXTURES / "django_project"),
                "--domain",
                "django",
                "--state-dir",
                str(state_dir),
            ]
        )
    finally:
        _stop(server, thread)
    payload = json.loads(capsys.readouterr().out)
    assert trust_code == 0
    assert start_code == 1
    assert payload["status"] == "failed"
    assert "unknown tool" in payload["error"]

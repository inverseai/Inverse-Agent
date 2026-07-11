"""Provider-neutral planning that lets models select typed tools, never raw commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException, HTTPResponse, HTTPSConnection
from ipaddress import ip_address
from time import monotonic
from typing import Any, Protocol
from unicodedata import category
from urllib.parse import urlsplit, urlunsplit

from inverse_agent.models import Domain, WorkspaceProfile
from inverse_agent.redaction import redact_text

MAX_MODEL_RESPONSE_BYTES = 1024 * 1024
MAX_MODEL_COMPLETION_TOKENS = 4096


class PlannerError(ValueError):
    """Base error for model planning failures."""


class PlannerTransportError(PlannerError):
    """Raised when the model endpoint cannot complete a request."""


class PlannerProtocolError(PlannerError):
    """Raised when the model endpoint returns an invalid response."""


def validate_model_endpoint(base_url: str, *, allow_remote: bool = False) -> str:
    """Validate and normalize a trusted-operator model endpoint."""

    if any(character.isspace() or category(character) == "Cc" for character in base_url):
        raise ValueError("model base URL must not contain whitespace or control characters")
    value = base_url.rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("model base URL scheme must be http or https")
    if not parsed.hostname:
        raise ValueError("model base URL must include a host")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("model base URL contains an invalid port") from exc
    if parsed.username or parsed.password:
        raise ValueError("model base URL must not include user information")
    if parsed.query or parsed.fragment:
        raise ValueError("model base URL must not include a query or fragment")

    hostname = parsed.hostname.lower()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        if not allow_remote:
            raise ValueError("remote model endpoints require explicit dual opt-in")
        if parsed.scheme != "https":
            raise ValueError("remote model endpoints must use https")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


@dataclass(frozen=True)
class PlannedAction:
    tool_name: str


@dataclass(frozen=True)
class ExecutionPlan:
    actions: tuple[PlannedAction, ...]
    rationale: str = ""


class Planner(Protocol):
    def plan(
        self,
        *,
        goal: str,
        domain: Domain,
        profile: WorkspaceProfile,
        available_tools: tuple[str, ...],
    ) -> ExecutionPlan: ...


class JsonModelClient(Protocol):
    def complete_json(self, *, system: str, prompt: str) -> dict[str, Any]: ...


class OpenAICompatibleClient:
    """Small client for OpenAI-compatible cloud or local model endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 60,
        allow_remote: bool = False,
    ) -> None:
        self.base_url = validate_model_endpoint(base_url, allow_remote=allow_remote)
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def complete_json(self, *, system: str, prompt: str) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "inverse_agent_plan",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "actions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "rationale": {"type": "string"},
                            },
                            "required": ["actions", "rationale"],
                            "additionalProperties": False,
                        },
                    },
                },
                "temperature": 0,
                "max_tokens": MAX_MODEL_COMPLETION_TOKENS,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        parsed_url = urlsplit(self.base_url)
        hostname = parsed_url.hostname
        if hostname is None:
            raise PlannerProtocolError("model endpoint host is unavailable")
        connection_type = HTTPSConnection if parsed_url.scheme == "https" else HTTPConnection
        connection = connection_type(
            hostname,
            port=parsed_url.port,
            timeout=self.timeout_seconds,
        )
        request_path = f"{parsed_url.path.rstrip('/')}/chat/completions"
        if not request_path.startswith("/"):
            request_path = f"/{request_path}"
        deadline = monotonic() + self.timeout_seconds
        try:
            connection.request("POST", request_path, body=body, headers=headers)
            self._apply_deadline(connection, deadline)
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                raise PlannerTransportError(f"model endpoint returned HTTP {response.status}")
            raw = self._read_response(response, connection, deadline)
        except PlannerError:
            raise
        except (HTTPException, TimeoutError, OSError) as exc:
            raise PlannerTransportError("model endpoint request failed") from exc
        finally:
            connection.close()
        try:
            payload = json.loads(raw)
            choices = payload["choices"]
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content is not text")
            parsed = json.loads(content)
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PlannerProtocolError("model endpoint returned an invalid response") from exc
        if not isinstance(parsed, dict):
            raise PlannerProtocolError("model plan must be a JSON object")
        return parsed

    @staticmethod
    def _apply_deadline(
        connection: HTTPConnection | HTTPSConnection,
        deadline: float,
    ) -> None:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise PlannerTransportError("model endpoint request timed out")
        if connection.sock is not None:
            connection.sock.settimeout(remaining)

    @classmethod
    def _read_response(
        cls,
        response: HTTPResponse,
        connection: HTTPConnection | HTTPSConnection,
        deadline: float,
    ) -> bytes:
        content_length = response.getheader("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise PlannerProtocolError("model response has invalid content length") from exc
            if declared_size < 0:
                raise PlannerProtocolError("model response has invalid content length")
            if declared_size > MAX_MODEL_RESPONSE_BYTES:
                raise PlannerProtocolError("model response exceeds size limit")

        chunks: list[bytes] = []
        size = 0
        while True:
            cls._apply_deadline(connection, deadline)
            chunk = response.read1(min(65536, MAX_MODEL_RESPONSE_BYTES + 1 - size))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_MODEL_RESPONSE_BYTES:
                raise PlannerProtocolError("model response exceeds size limit")


class StructuredPlanner:
    """Constrains model output to registered tool identifiers."""

    def __init__(self, client: JsonModelClient, max_actions: int = 8):
        self.client = client
        self.max_actions = max_actions

    def plan(
        self,
        *,
        goal: str,
        domain: Domain,
        profile: WorkspaceProfile,
        available_tools: tuple[str, ...],
    ) -> ExecutionPlan:
        del profile
        payload = self.client.complete_json(
            system=(
                "You plan verification work. Select only supplied tool names. "
                "Never emit commands, arguments, source, or secrets."
            ),
            prompt=json.dumps(
                {
                    "goal": redact_text(goal).text,
                    "domain": domain.value,
                    "available_tools": list(available_tools),
                    "schema": {"actions": ["tool.name"], "rationale": "short text"},
                }
            ),
        )
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError("model plan must contain at least one action")
        if len(raw_actions) > self.max_actions:
            raise ValueError("model plan exceeds action budget")
        allowed = set(available_tools)
        actions: list[PlannedAction] = []
        selected: set[str] = set()
        for value in raw_actions:
            if not isinstance(value, str) or value not in allowed:
                raise ValueError("model selected an unknown tool")
            if value in selected:
                raise ValueError(f"model selected duplicate tool: {value!r}")
            selected.add(value)
            actions.append(PlannedAction(value))
        rationale = payload.get("rationale", "")
        if not isinstance(rationale, str):
            raise ValueError("model rationale must be text")
        return ExecutionPlan(tuple(actions), redact_text(rationale).text)


class DeterministicPlanner:
    """Offline planner used for CI, fixtures, and reproducible dogfood baselines."""

    DEFAULTS = {
        Domain.DJANGO: ("django.check", "django.test"),
        Domain.PYTORCH: ("pytorch.smoke_train", "pytorch.eval"),
        Domain.ANDROID: ("android.tasks", "android.test", "android.lint"),
        Domain.ANDROID_NDK: ("android_ndk.cmake_build",),
        Domain.IOS: ("ios.list", "ios.test"),
        Domain.GENERIC: ("generic.status", "generic.tracked_files"),
    }

    def plan(
        self,
        *,
        goal: str,
        domain: Domain,
        profile: WorkspaceProfile,
        available_tools: tuple[str, ...],
    ) -> ExecutionPlan:
        del goal, profile
        available = set(available_tools)
        actions = tuple(PlannedAction(name) for name in self.DEFAULTS[domain] if name in available)
        if not actions:
            raise ValueError(f"no executable tools available for {domain.value}")
        return ExecutionPlan(actions, "deterministic verification baseline")

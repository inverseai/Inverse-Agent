"""Provider-neutral planning that lets models select typed tools, never raw commands."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException, HTTPResponse, HTTPSConnection
from ipaddress import ip_address
from threading import Lock
from time import monotonic
from typing import Any, Protocol
from unicodedata import category
from urllib.parse import urlsplit, urlunsplit

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from inverse_agent.models import Domain, WorkspaceProfile
from inverse_agent.redaction import redact_text

MAX_MODEL_RESPONSE_BYTES = 1024 * 1024
MAX_MODEL_COMPLETION_TOKENS = 4096
MAX_SUCCESSFUL_RESPONSE_MODEL_HISTORY = 1024
MAX_OBSERVED_RESPONSE_MODELS = 64
MODEL_RESPONSE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,255}")
PLAN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actions": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["actions", "rationale"],
    "additionalProperties": False,
}


class PlannerError(ValueError):
    """Base error for model planning failures."""


class PlannerTransportError(PlannerError):
    """Raised when the model endpoint cannot complete a request."""


class PlannerProtocolError(PlannerError):
    """Raised when the model endpoint returns an invalid response."""


class PlannerBudgetError(PlannerError):
    """Raised before a model call would exceed a configured run budget."""


class PlannerAttestationError(PlannerError):
    """Raised before source is sent after its workspace attestation is revoked."""


@dataclass(frozen=True)
class ModelResponseMetadata:
    """Validated provider metadata for the most recent successful response."""

    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


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
        self._observed_response_models: set[str] = set()
        self._observed_response_models_overflowed = False
        self._response_model_mismatch_observed = False
        self._successful_response_count = 0
        self._attributed_response_count = 0
        self._successful_response_models: list[str | None] = []
        self._successful_response_history_start = 0
        self._last_response_metadata: ModelResponseMetadata | None = None
        self._observation_lock = Lock()

    @property
    def observed_response_models(self) -> tuple[str, ...]:
        with self._observation_lock:
            return tuple(sorted(self._observed_response_models))

    @property
    def observed_response_models_overflowed(self) -> bool:
        with self._observation_lock:
            return self._observed_response_models_overflowed

    @property
    def response_model_mismatch_observed(self) -> bool:
        with self._observation_lock:
            return self._response_model_mismatch_observed

    @property
    def successful_response_count(self) -> int:
        with self._observation_lock:
            return self._successful_response_count

    @property
    def attributed_response_count(self) -> int:
        with self._observation_lock:
            return self._attributed_response_count

    @property
    def successful_response_models(self) -> tuple[str | None, ...]:
        """Retained tail of reported models for successful response envelopes."""

        with self._observation_lock:
            return tuple(self._successful_response_models)

    def successful_response_models_since(
        self, successful_response_count: int
    ) -> tuple[str | None, ...] | None:
        """Return provenance after an absolute cursor, or None if it was evicted."""

        if (
            isinstance(successful_response_count, bool)
            or not isinstance(successful_response_count, int)
            or successful_response_count < 0
        ):
            raise ValueError("successful response cursor must be a non-negative integer")
        with self._observation_lock:
            if not (
                self._successful_response_history_start
                <= successful_response_count
                <= self._successful_response_count
            ):
                return None
            offset = successful_response_count - self._successful_response_history_start
            return tuple(self._successful_response_models[offset:])

    @property
    def last_response_metadata(self) -> ModelResponseMetadata | None:
        with self._observation_lock:
            return self._last_response_metadata

    def complete_json(self, *, system: str, prompt: str) -> dict[str, Any]:
        return self.complete_structured_json(
            system=system,
            prompt=prompt,
            schema_name="inverse_agent_plan",
            schema=PLAN_RESPONSE_SCHEMA,
        )

    def complete_structured_json(
        self,
        *,
        system: str,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        max_tokens: int = MAX_MODEL_COMPLETION_TOKENS,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Complete a caller-supplied strict JSON schema over the hardened transport."""

        if not schema_name or not schema_name.replace("_", "").isalnum():
            raise ValueError("schema name contains unsupported characters")
        if not 1 <= max_tokens <= MAX_MODEL_COMPLETION_TOKENS:
            raise ValueError(f"max tokens must be between 1 and {MAX_MODEL_COMPLETION_TOKENS}")
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("request timeout must be a positive finite number")
        effective_timeout = float(self.timeout_seconds)
        if timeout_seconds is not None:
            effective_timeout = min(effective_timeout, timeout_seconds)
        with self._observation_lock:
            self._last_response_metadata = None
        schema_payload = dict(schema)
        try:
            Draft202012Validator.check_schema(schema_payload)
            validator = Draft202012Validator(schema_payload)
        except SchemaError as exc:
            raise ValueError("model response schema is invalid") from exc
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
                        "name": schema_name,
                        "strict": True,
                        "schema": schema_payload,
                    },
                },
                "temperature": 0,
                "max_tokens": max_tokens,
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
            timeout=effective_timeout,
        )
        request_path = f"{parsed_url.path.rstrip('/')}/chat/completions"
        if not request_path.startswith("/"):
            request_path = f"/{request_path}"
        deadline = monotonic() + effective_timeout
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
            if not isinstance(payload, dict):
                raise TypeError("response is not an object")
            reported_model = payload.get("model")
            if reported_model is not None and (
                not isinstance(reported_model, str)
                or MODEL_RESPONSE_ID_PATTERN.fullmatch(reported_model) is None
            ):
                raise TypeError("response model identity is invalid")
            usage = payload.get("usage")
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None
            if usage is not None:
                if not isinstance(usage, dict):
                    raise TypeError("response usage is not an object")
                prompt_tokens = self._usage_count(usage, "prompt_tokens")
                completion_tokens = self._usage_count(usage, "completion_tokens")
                total_tokens = self._usage_count(usage, "total_tokens")
                if total_tokens != prompt_tokens + completion_tokens:
                    raise TypeError("response usage totals are inconsistent")
                if completion_tokens > max_tokens:
                    raise TypeError("response completion usage exceeds the request allowance")
            # Retain independently validated envelope metadata even when the
            # decision content below is malformed. The caller can then charge
            # reported usage and attribute a protocol failure accurately.
            with self._observation_lock:
                self._last_response_metadata = ModelResponseMetadata(
                    model=reported_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            # Record every validated response envelope before parsing its
            # decision content. A malformed decision can be retried, but its
            # endpoint attribution must remain visible to the capability gate.
            self._record_successful_response(reported_model)
            choices = payload["choices"]
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content is not text")
            parsed = json.loads(content)
        except (IndexError, KeyError, TypeError, ValueError, RecursionError) as exc:
            raise PlannerProtocolError("model endpoint returned an invalid response") from exc
        if not isinstance(parsed, dict):
            raise PlannerProtocolError("model plan must be a JSON object")
        try:
            validator.validate(parsed)
        except (ValidationError, ValueError, RecursionError) as exc:
            raise PlannerProtocolError(
                "model response does not match the requested JSON schema"
            ) from exc
        return parsed

    def _record_successful_response(self, reported_model: str | None) -> None:
        with self._observation_lock:
            self._successful_response_count += 1
            self._successful_response_models.append(reported_model)
            overflow = len(self._successful_response_models) - MAX_SUCCESSFUL_RESPONSE_MODEL_HISTORY
            if overflow > 0:
                del self._successful_response_models[:overflow]
            self._successful_response_history_start = self._successful_response_count - len(
                self._successful_response_models
            )
            if reported_model is not None:
                if reported_model != self.model:
                    self._response_model_mismatch_observed = True
                if (
                    reported_model in self._observed_response_models
                    or len(self._observed_response_models) < MAX_OBSERVED_RESPONSE_MODELS
                ):
                    self._observed_response_models.add(reported_model)
                else:
                    self._observed_response_models_overflowed = True
                self._attributed_response_count += 1

    @staticmethod
    def _usage_count(usage: dict[str, Any], name: str) -> int:
        value = usage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"response usage {name} is invalid")
        return value

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

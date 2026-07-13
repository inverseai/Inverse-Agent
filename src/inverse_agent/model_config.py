"""Validated model-planner configuration for CLI and service entry points."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from inverse_agent.planner import (
    DeterministicPlanner,
    OpenAICompatibleClient,
    Planner,
    StructuredPlanner,
    validate_model_endpoint,
)

MODEL_ENV_NAMES = (
    "INVERSE_AGENT_MODEL_BASE_URL",
    "INVERSE_AGENT_MODEL_NAME",
    "INVERSE_AGENT_MODEL_API_KEY",
    "INVERSE_AGENT_MODEL_TIMEOUT_SECONDS",
    "INVERSE_AGENT_MODEL_MAX_ACTIONS",
    "INVERSE_AGENT_MODEL_CONTEXT_TOKENS",
    "INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN",
    "INVERSE_AGENT_MODEL_ALLOW_REMOTE",
)
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,255}")


@dataclass(frozen=True)
class PlannerConfig:
    kind: str
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: int = 60
    max_actions: int = 8
    allow_remote: bool = False

    @property
    def fingerprint(self) -> str:
        if self.kind == "deterministic":
            return "deterministic"
        payload = {
            "allow_remote": self.allow_remote,
            "base_url": self.base_url,
            "kind": self.kind,
            "max_actions": self.max_actions,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"{self.kind}:sha256:{sha256(canonical.encode()).hexdigest()}"

    def safe_summary(self) -> dict[str, str | int | bool | None]:
        return {
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "max_actions": self.max_actions,
            "allow_remote": self.allow_remote,
            "api_key_set": bool(self.api_key),
        }


@dataclass(frozen=True)
class PlannerResolution:
    planner: Planner
    config: PlannerConfig
    client: OpenAICompatibleClient | None = None


def resolve_planner(
    *,
    args: Any | None = None,
    env: Mapping[str, str] | None = None,
    require_model: bool = False,
) -> PlannerResolution:
    values = os.environ if env is None else env
    model = _string_override(args, "model", values.get("INVERSE_AGENT_MODEL_NAME"))
    base_url = _string_override(
        args,
        "model_base_url",
        values.get("INVERSE_AGENT_MODEL_BASE_URL"),
    )
    if model is None and base_url is None:
        if require_model:
            raise ValueError("model planner configuration is required")
        config = PlannerConfig(kind="deterministic")
        return PlannerResolution(DeterministicPlanner(), config)
    if not model or not base_url:
        raise ValueError("model name and base URL must be configured together")
    if not _MODEL_NAME.fullmatch(model):
        raise ValueError("model name contains unsupported characters")

    timeout_seconds = _bounded_int(
        _number_override(
            args,
            "model_timeout_seconds",
            values.get("INVERSE_AGENT_MODEL_TIMEOUT_SECONDS", "60"),
        ),
        name="model timeout",
        minimum=1,
        maximum=600,
    )
    max_actions = _bounded_int(
        _number_override(
            args,
            "model_max_actions",
            values.get("INVERSE_AGENT_MODEL_MAX_ACTIONS", "8"),
        ),
        name="model max actions",
        minimum=1,
        maximum=32,
    )
    env_allows_remote = _parse_bool(values.get("INVERSE_AGENT_MODEL_ALLOW_REMOTE", "0"))
    flag_allows_remote = bool(getattr(args, "model_allow_remote", False)) if args else False
    allow_remote = env_allows_remote and flag_allows_remote
    normalized_url = validate_model_endpoint(base_url, allow_remote=allow_remote)
    api_key = values.get("INVERSE_AGENT_MODEL_API_KEY") or None
    config = PlannerConfig(
        kind="openai-compatible",
        base_url=normalized_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_actions=max_actions,
        allow_remote=allow_remote,
    )
    client = OpenAICompatibleClient(
        base_url=normalized_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        allow_remote=allow_remote,
    )
    return PlannerResolution(StructuredPlanner(client, max_actions=max_actions), config, client)


def _string_override(args: Any | None, name: str, fallback: str | None) -> str | None:
    if args is not None:
        value: object = getattr(args, name, None)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{name.replace('_', ' ')} must be text")
            return value
    return fallback


def _number_override(args: Any | None, name: str, fallback: str | None) -> str | int | None:
    if args is not None:
        value: object = getattr(args, name, None)
        if value is not None:
            if not isinstance(value, (str, int)):
                raise ValueError(f"{name.replace('_', ' ')} must be an integer")
            return value
    return fallback


def _bounded_int(value: str | int | None, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else 0
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("INVERSE_AGENT_MODEL_ALLOW_REMOTE must be a boolean")

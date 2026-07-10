"""Provider-neutral planning that lets models select typed tools, never raw commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen

from inverse_agent.models import Domain, WorkspaceProfile


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
    ) -> ExecutionPlan:
        ...


class JsonModelClient(Protocol):
    def complete_json(self, *, system: str, prompt: str) -> dict[str, Any]:
        ...


class OpenAICompatibleClient:
    """Small client for OpenAI-compatible cloud or local model endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
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
                "response_format": {"type": "json_object"},
                "temperature": 0,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read())
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("model plan must be a JSON object")
        return parsed


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
                    "goal": goal,
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
        for value in raw_actions:
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"model selected unknown tool: {value!r}")
            actions.append(PlannedAction(value))
        return ExecutionPlan(tuple(actions), str(payload.get("rationale", "")))


class DeterministicPlanner:
    """Offline planner used for CI, fixtures, and reproducible dogfood baselines."""

    DEFAULTS = {
        Domain.DJANGO: ("django.check", "django.test"),
        Domain.PYTORCH: ("pytorch.smoke_train", "pytorch.eval"),
        Domain.ANDROID: ("android.tasks", "android.test", "android.lint"),
        Domain.ANDROID_NDK: ("android_ndk.cmake_build",),
        Domain.IOS: ("ios.list", "ios.test"),
        Domain.GENERIC: (),
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
        actions = tuple(
            PlannedAction(name) for name in self.DEFAULTS[domain] if name in available
        )
        if not actions:
            raise ValueError(f"no executable tools available for {domain.value}")
        return ExecutionPlan(actions, "deterministic verification baseline")

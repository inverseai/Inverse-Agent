"""Reproducible advisory evaluations for in-house workspace onboarding."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.eval import json_default
from inverse_agent.models import Domain
from inverse_agent.planner import DeterministicPlanner, Planner
from inverse_agent.redaction import redact_text

MAX_EVALUATION_ERROR_CHARS = 4096


@dataclass(frozen=True)
class DomainEvaluation:
    domain: Domain
    passed: bool
    planned_tools: tuple[str, ...]
    unavailable: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class WorkspaceEvaluation:
    workspace: Path
    domains: tuple[DomainEvaluation, ...]

    @property
    def passed(self) -> bool:
        return bool(self.domains) and all(result.passed for result in self.domains)


def evaluate_workspace(root: Path, planner: Planner | None = None) -> WorkspaceEvaluation:
    profile = detect_workspace(root)
    selected_planner = planner or DeterministicPlanner()
    results: list[DomainEvaluation] = []
    for domain in sorted(profile.domains, key=lambda item: item.value):
        available = tuple(
            sorted(name for name in profile.commands if name.startswith(f"{domain.value}."))
        )
        unavailable = tuple(
            sorted(
                reason
                for name, reason in profile.unavailable_tools.items()
                if name.startswith(f"{domain.value}.")
            )
        )
        try:
            plan = selected_planner.plan(
                goal="Create the default verification plan",
                domain=domain,
                profile=profile,
                available_tools=available,
            )
            planned = tuple(action.tool_name for action in plan.actions)
            results.append(DomainEvaluation(domain, True, planned, unavailable))
        except (TypeError, ValueError) as exc:
            error = redact_text(str(exc)).text[:MAX_EVALUATION_ERROR_CHARS]
            results.append(DomainEvaluation(domain, False, (), unavailable, error))
    return WorkspaceEvaluation(root.resolve(), tuple(results))


def save_evaluation(result: WorkspaceEvaluation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(result), "passed": result.passed}
    path.write_text(json.dumps(payload, default=json_default, indent=2), encoding="utf-8")

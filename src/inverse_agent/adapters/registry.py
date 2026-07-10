"""Adapter registry and workspace detection."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.adapters.android import AndroidAdapter, AndroidNdkAdapter
from inverse_agent.adapters.base import ToolchainAdapter
from inverse_agent.adapters.django import DjangoAdapter
from inverse_agent.adapters.ios import IosAdapter
from inverse_agent.adapters.pytorch import PyTorchAdapter
from inverse_agent.models import Domain, WorkspaceProfile


def default_adapters() -> list[ToolchainAdapter]:
    return [
        DjangoAdapter(),
        PyTorchAdapter(),
        AndroidAdapter(),
        AndroidNdkAdapter(),
        IosAdapter(),
    ]


def detect_workspace(root: Path) -> WorkspaceProfile:
    root = root.resolve()
    domains: set[Domain] = set()
    commands: dict[str, list[str]] = {}
    tests: list[str] = []
    toolchain: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    for adapter in default_adapters():
        if adapter.detect(root):
            profile = adapter.profile(root)
            domains.update(profile.domains)
            commands.update({f"{adapter.domain.value}.{k}": v for k, v in profile.commands.items()})
            tests.extend(profile.test_targets)
            toolchain.update(profile.toolchain)
            unavailable.update(
                {f"{adapter.domain.value}.{key}": value for key, value in profile.unavailable_tools.items()}
            )
    if not domains:
        domains.add(Domain.GENERIC)
    return WorkspaceProfile(
        root=root,
        domains=domains,
        commands=commands,
        test_targets=tests,
        toolchain=toolchain,
        unavailable_tools=unavailable,
    )

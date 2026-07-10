"""iOS/Xcode adapter."""

from __future__ import annotations

import platform
from pathlib import Path

from inverse_agent.adapters.base import CommandAdapter, Tool
from inverse_agent.environments import discover_system_executable
from inverse_agent.models import Domain, WorkspaceProfile


class IosAdapter(CommandAdapter):
    domain = Domain.IOS

    def detect(self, root: Path) -> bool:
        return bool(list(root.glob("*.xcodeproj")) or list(root.glob("*.xcworkspace")))

    def profile(self, root: Path) -> WorkspaceProfile:
        xcodebuild = discover_system_executable("xcodebuild")
        commands: dict[str, list[str]] = {}
        unavailable: dict[str, str] = {}
        if platform.system() == "Darwin" and xcodebuild:
            commands.update(
                {
                    "list": [str(xcodebuild), "-list"],
                    "build": [str(xcodebuild), "build"],
                    "test": [str(xcodebuild), "test"],
                }
            )
        else:
            unavailable["xcodebuild"] = "iOS workflows require xcodebuild on macOS"
        return WorkspaceProfile(
            root=root,
            domains={Domain.IOS},
            commands=commands,
            test_targets=["xcodebuild test"],
            toolchain={
                "host": "macos-required",
                "build": str(xcodebuild) if xcodebuild else "unavailable",
            },
            unavailable_tools=unavailable,
        )

    def tools(self) -> list[Tool]:
        return [
            Tool("ios.list", "Inspect Xcode project configuration", "approval-required", self.domain),
            Tool("ios.build", "Run xcodebuild build on macOS", "approval-required", self.domain),
            Tool("ios.test", "Run xcodebuild test on macOS", "approval-required", self.domain),
        ]

    @staticmethod
    def host_supported() -> bool:
        return platform.system() == "Darwin"

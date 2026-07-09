"""iOS/Xcode adapter."""

from __future__ import annotations

import platform
from pathlib import Path

from inverse_agent.adapters.base import CommandAdapter, Tool
from inverse_agent.models import Domain, WorkspaceProfile


class IosAdapter(CommandAdapter):
    domain = Domain.IOS

    def detect(self, root: Path) -> bool:
        return bool(list(root.glob("*.xcodeproj")) or list(root.glob("*.xcworkspace")))

    def profile(self, root: Path) -> WorkspaceProfile:
        commands = {"list": ["xcodebuild", "-list"]}
        if platform.system() == "Darwin":
            commands.update(
                {
                    "build": ["xcodebuild", "build"],
                    "test": ["xcodebuild", "test"],
                }
            )
        return WorkspaceProfile(
            root=root,
            domains={Domain.IOS},
            commands=commands,
            test_targets=["xcodebuild test"],
            toolchain={"host": "macos-required", "build": "xcodebuild"},
        )

    def tools(self) -> list[Tool]:
        return [
            Tool("ios.list", "Inspect Xcode project configuration", "safe-read", self.domain),
            Tool("ios.build", "Run xcodebuild build on macOS", "safe-build", self.domain),
            Tool("ios.test", "Run xcodebuild test on macOS", "safe-read", self.domain),
        ]

    @staticmethod
    def host_supported() -> bool:
        return platform.system() == "Darwin"


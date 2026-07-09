"""Android and Android NDK adapters."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.adapters.base import CommandAdapter, Tool
from inverse_agent.models import Domain, WorkspaceProfile


class AndroidAdapter(CommandAdapter):
    domain = Domain.ANDROID

    def detect(self, root: Path) -> bool:
        return (root / "gradlew").exists() or (root / "gradlew.bat").exists() or (root / "build.gradle").exists()

    def profile(self, root: Path) -> WorkspaceProfile:
        gradle = "gradlew.bat" if (root / "gradlew.bat").exists() else "gradlew"
        return WorkspaceProfile(
            root=root,
            domains={Domain.ANDROID},
            commands={
                "tasks": [gradle, "tasks"],
                "test": [gradle, "test"],
                "lint": [gradle, "lint"],
                "assemble_debug": [gradle, "assembleDebug"],
            },
            test_targets=["gradle test", "gradle lint"],
            toolchain={"build": "gradle", "gradle": gradle},
        )

    def tools(self) -> list[Tool]:
        return [
            Tool("android.tasks", "List Gradle tasks", "safe-read", self.domain),
            Tool("android.test", "Run Android unit tests", "safe-read", self.domain),
            Tool("android.lint", "Run Android lint", "safe-read", self.domain),
            Tool("android.assemble_debug", "Build a debug APK", "safe-build", self.domain),
        ]


class AndroidNdkAdapter(CommandAdapter):
    domain = Domain.ANDROID_NDK

    def detect(self, root: Path) -> bool:
        return any(
            (root / marker).exists()
            for marker in ("CMakeLists.txt", "jni", "src/main/cpp", "Android.mk")
        )

    def profile(self, root: Path) -> WorkspaceProfile:
        return WorkspaceProfile(
            root=root,
            domains={Domain.ANDROID_NDK},
            commands={"cmake_build": ["cmake", "--build", "build"]},
            test_targets=["cmake --build build"],
            toolchain={"native": "ndk/cmake"},
        )

    def tools(self) -> list[Tool]:
        return [
            Tool("android_ndk.cmake_build", "Run configured CMake build", "safe-build", self.domain),
            Tool("android_ndk.inspect_jni", "Inspect JNI/native layout", "safe-read", self.domain),
        ]


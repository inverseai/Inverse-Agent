"""Android and Android NDK adapters."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.adapters.base import CommandAdapter, Tool
from inverse_agent.environments import discover_gradle_wrapper, discover_system_executable
from inverse_agent.models import Domain, WorkspaceProfile


class AndroidAdapter(CommandAdapter):
    domain = Domain.ANDROID

    def detect(self, root: Path) -> bool:
        return (root / "gradlew").exists() or (root / "gradlew.bat").exists() or (root / "build.gradle").exists()

    def profile(self, root: Path) -> WorkspaceProfile:
        gradle = discover_gradle_wrapper(root)
        commands: dict[str, list[str]] = {}
        unavailable: dict[str, str] = {}
        if gradle:
            executable = str(gradle)
            commands = {
                "tasks": [executable, "--offline", "tasks"],
                "test": [executable, "--offline", "test"],
                "lint": [executable, "--offline", "lint"],
                "assemble_debug": [executable, "--offline", "assembleDebug"],
            }
        else:
            unavailable["gradle"] = "Gradle wrapper not found in workspace root"
        return WorkspaceProfile(
            root=root,
            domains={Domain.ANDROID},
            commands=commands,
            test_targets=["gradle test", "gradle lint"],
            toolchain={"build": "gradle", "gradle": str(gradle) if gradle else "unavailable"},
            unavailable_tools=unavailable,
        )

    def tools(self) -> list[Tool]:
        return [
            Tool("android.tasks", "List Gradle tasks", "approval-required", self.domain),
            Tool("android.test", "Run Android unit tests", "approval-required", self.domain),
            Tool("android.lint", "Run Android lint", "approval-required", self.domain),
            Tool(
                "android.assemble_debug",
                "Build a debug APK",
                "approval-required",
                self.domain,
            ),
        ]


class AndroidNdkAdapter(CommandAdapter):
    domain = Domain.ANDROID_NDK

    def detect(self, root: Path) -> bool:
        return any(
            (root / marker).exists()
            for marker in ("CMakeLists.txt", "jni", "src/main/cpp", "Android.mk")
        )

    def profile(self, root: Path) -> WorkspaceProfile:
        cmake = discover_system_executable("cmake")
        commands = {"cmake_build": [str(cmake), "--build", "build"]} if cmake else {}
        return WorkspaceProfile(
            root=root,
            domains={Domain.ANDROID_NDK},
            commands=commands,
            test_targets=["cmake --build build"],
            toolchain={"native": "ndk/cmake", "cmake": str(cmake) if cmake else "unavailable"},
            unavailable_tools={} if cmake else {"cmake": "cmake executable not found"},
        )

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "android_ndk.cmake_build",
                "Run configured CMake build",
                "approval-required",
                self.domain,
            ),
            Tool("android_ndk.inspect_jni", "Inspect JNI/native layout", "safe-read", self.domain),
        ]

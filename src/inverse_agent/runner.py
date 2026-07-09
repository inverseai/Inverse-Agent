"""Local command runner with default-deny policy enforcement."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from inverse_agent.models import CommandRule, Domain, RunnerPolicy, RunStatus
from inverse_agent.policies import has_shell_metachar
from inverse_agent.redaction import redact_text


@dataclass(frozen=True)
class CommandRequest:
    argv: tuple[str, ...]
    cwd: Path
    domain: Domain
    approved: bool = False
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class CommandResult:
    status: RunStatus
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    rule: str | None = None
    reason: str = ""


def normalize_token(token: str) -> str:
    value = Path(token).name.lower()
    if value.endswith(".exe"):
        value = value[:-4]
    if value.endswith(".cmd"):
        value = value[:-4]
    if value in {"python3", "py"}:
        return "python"
    if value in {"gradlew.bat"}:
        return "gradlew"
    return value


def normalize_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    if not argv:
        return argv
    return (normalize_token(argv[0]), *argv[1:])


class PolicyViolation(ValueError):
    """Raised when a command violates runner policy."""


class LocalRunner:
    def __init__(self, policy: RunnerPolicy):
        self.policy = policy
        self.workspace_root = policy.workspace_root.resolve()

    def validate(self, request: CommandRequest) -> CommandRule:
        if not request.argv:
            raise PolicyViolation("empty argv refused")
        for arg in request.argv:
            if has_shell_metachar(arg):
                raise PolicyViolation(f"shell metacharacter refused in argument: {arg}")
        cwd = request.cwd.resolve()
        if not self._is_relative_to(cwd, self.workspace_root):
            raise PolicyViolation(f"cwd outside workspace refused: {cwd}")
        normalized = normalize_argv(request.argv)
        for rule in self.policy.rules_for(request.domain):
            if normalized[: len(rule.argv_prefix)] == rule.argv_prefix:
                if (
                    rule.network_required
                    and self.policy.network_default == "deny"
                    and not request.approved
                ):
                    raise PolicyViolation(f"network approval required for {rule.name}: {rule.reason}")
                if rule.requires_approval and not request.approved:
                    raise PolicyViolation(f"approval required for {rule.name}: {rule.reason}")
                return rule
        raise PolicyViolation(f"command is not allowlisted: {' '.join(request.argv)}")

    def run(self, request: CommandRequest) -> CommandResult:
        try:
            rule = self.validate(request)
        except PolicyViolation as exc:
            return CommandResult(
                status=RunStatus.REFUSED,
                argv=request.argv,
                returncode=None,
                stdout="",
                stderr="",
                reason=str(exc),
            )
        timeout = request.timeout_seconds or self.policy.compute_budget_seconds
        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(request.argv),
                cwd=request.cwd,
                shell=False,
                text=True,
                capture_output=True,
                timeout=timeout,
                env=self._safe_env(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = redact_text(exc.stdout or "").text
            stderr = redact_text(exc.stderr or "").text
            return CommandResult(
                status=RunStatus.FAILED,
                argv=request.argv,
                returncode=None,
                stdout=stdout,
                stderr=stderr,
                rule=rule.name,
                reason=f"command exceeded compute budget after {timeout} seconds",
            )
        except FileNotFoundError as exc:
            return CommandResult(
                status=RunStatus.FAILED,
                argv=request.argv,
                returncode=None,
                stdout="",
                stderr="",
                rule=rule.name,
                reason=f"executable not found: {exc.filename}",
            )
        except PermissionError as exc:
            return CommandResult(
                status=RunStatus.FAILED,
                argv=request.argv,
                returncode=None,
                stdout="",
                stderr="",
                rule=rule.name,
                reason=f"permission denied while executing command: {exc}",
            )
        except OSError as exc:
            return CommandResult(
                status=RunStatus.FAILED,
                argv=request.argv,
                returncode=None,
                stdout="",
                stderr="",
                rule=rule.name,
                reason=f"os error while executing command: {exc}",
            )
        stdout_redaction = redact_text(completed.stdout)
        stderr_redaction = redact_text(completed.stderr)
        status = RunStatus.SUCCEEDED if completed.returncode == 0 else RunStatus.FAILED
        redacted_streams = []
        if stdout_redaction.blocked:
            redacted_streams.append("stdout")
        if stderr_redaction.blocked:
            redacted_streams.append("stderr")
        elapsed = time.monotonic() - started
        reason = (
            f"redacted secret-like content from {', '.join(redacted_streams)}"
            if redacted_streams
            else f"completed in {elapsed:.3f}s"
        )
        return CommandResult(
            status=status,
            argv=request.argv,
            returncode=completed.returncode,
            stdout=stdout_redaction.text,
            stderr=stderr_redaction.text,
            rule=rule.name,
            reason=reason,
        )

    def _safe_env(self) -> dict[str, str]:
        env: dict[str, str] = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        allowed = {name.upper() for name in self.policy.allowed_env_names}
        for key, value in os.environ.items():
            if key.upper() not in allowed:
                continue
            if redact_text(f"{key}={value}").blocked:
                continue
            env[key] = value
        return env

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

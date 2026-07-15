"""Local command runner with exact policy and approval-capability enforcement."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from inverse_agent.approvals import ApprovalAuthority, ApprovalClaims, ApprovalError, action_digest
from inverse_agent.models import CommandRule, Domain, RunnerPolicy, RunStatus
from inverse_agent.redaction import redact_text

OUTPUT_REDACTION_OVERLAP_BYTES = 64 * 1024
PROCESS_TERMINATION_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class CommandRequest:
    argv: tuple[str, ...]
    cwd: Path
    domain: Domain
    approval_token: str | None = None
    approval_challenge_id: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class CommandResult:
    status: RunStatus
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    rule: str | None = None
    reason: str = ""
    approval_id: str | None = None
    stdout_redacted: bool = False
    stderr_redacted: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass(frozen=True)
class ApprovalChallenge:
    action_digest: str
    rule: str
    argv: tuple[str, ...]
    workspace: str
    domain: str
    reason: str


@dataclass(frozen=True)
class _PreparedCommand:
    rule: CommandRule
    resolved_argv: tuple[str, ...]
    approval: ApprovalClaims | None


class _BinaryStream(Protocol):
    def seek(self, offset: int, whence: int = 0) -> int: ...

    def read(self, size: int = -1) -> bytes: ...


def normalize_token(token: str) -> str:
    value = Path(token).name.lower()
    for suffix in (".exe", ".cmd", ".bat"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if value in {"python", "py"}:
        return "python"
    if value.startswith("python"):
        version = value.removeprefix("python")
        if version and all(part.isdigit() for part in version.split(".")):
            return "python"
    return value


def normalize_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    if not argv:
        return argv
    return (normalize_token(argv[0]), *argv[1:])


class PolicyViolation(ValueError):
    """Raised when a command violates runner policy."""


class ApprovalNotRequired(PolicyViolation):
    """Raised when a valid command can run without an approval interrupt."""


def build_safe_subprocess_env(allowed_env_names: tuple[str, ...]) -> dict[str, str]:
    """Build the minimal child environment shared by hardened local readers."""

    env: dict[str, str] = {
        "GCM_INTERACTIVE": "Never",
        "GIT_CONFIG_GLOBAL": "NUL" if os.name == "nt" else "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "NO_COLOR": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    allowed = {name.upper() for name in allowed_env_names}
    for key, value in os.environ.items():
        if key.upper() not in allowed:
            continue
        if redact_text(f"{key}={value}").blocked:
            continue
        env[key] = value
    return env


class LocalRunner:
    def __init__(self, policy: RunnerPolicy, approval_authority: ApprovalAuthority | None = None):
        self.policy = policy
        self.workspace_root = policy.workspace_root.resolve()
        self.approval_authority = approval_authority

    def validate(self, request: CommandRequest) -> CommandRule:
        """Validate without consuming an approval capability."""

        return self._prepare(request, consume_approval=False).rule

    def approval_challenge(self, request: CommandRequest) -> ApprovalChallenge:
        prepared = self._prepare(request, require_approval=False, consume_approval=False)
        if not prepared.rule.requires_approval and not prepared.rule.network_required:
            raise ApprovalNotRequired(f"command does not require approval: {prepared.rule.name}")
        return ApprovalChallenge(
            action_digest=action_digest(
                workspace=self.workspace_root,
                domain=request.domain,
                rule=prepared.rule,
                argv=prepared.resolved_argv,
            ),
            rule=prepared.rule.name,
            argv=prepared.resolved_argv,
            workspace=str(self.workspace_root),
            domain=request.domain.value,
            reason=prepared.rule.reason,
        )

    def run(self, request: CommandRequest) -> CommandResult:
        try:
            prepared = self._prepare(request, consume_approval=True)
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
        if timeout <= 0:
            return self._failed(request, prepared, "timeout must be positive")

        started = time.monotonic()
        try:
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                process = subprocess.Popen(
                    list(prepared.resolved_argv),
                    cwd=self.workspace_root,
                    shell=False,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=self._safe_env(),
                    start_new_session=os.name != "nt",
                    creationflags=(
                        int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
                        if os.name == "nt"
                        else 0
                    ),
                )
                timed_out = False
                termination_failed = False
                returncode: int | None
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._terminate_process_tree(process)
                    try:
                        returncode = process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        with suppress(OSError):
                            process.kill()
                        try:
                            returncode = process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
                        except subprocess.TimeoutExpired:
                            termination_failed = True
                            returncode = process.poll()
                stdout, stdout_truncated = self._read_output(stdout_file)
                stderr, stderr_truncated = self._read_output(stderr_file)
        except FileNotFoundError as exc:
            return self._failed(request, prepared, f"executable not found: {exc.filename}")
        except PermissionError as exc:
            return self._failed(
                request, prepared, f"permission denied while executing command: {exc}"
            )
        except OSError as exc:
            return self._failed(request, prepared, f"os error while executing command: {exc}")

        stdout_redaction = redact_text(stdout)
        stderr_redaction = redact_text(stderr)
        stdout_text = self._limit_output(stdout_redaction.text, stdout_truncated)
        stderr_text = self._limit_output(stderr_redaction.text, stderr_truncated)
        approval_id = prepared.approval.approval_id if prepared.approval else None
        if timed_out:
            reason = f"command exceeded compute budget after {timeout} seconds"
            if termination_failed:
                reason += "; process survived forced termination"
            return CommandResult(
                status=RunStatus.FAILED,
                argv=prepared.resolved_argv,
                returncode=returncode,
                stdout=stdout_text,
                stderr=stderr_text,
                rule=prepared.rule.name,
                reason=reason,
                approval_id=approval_id,
                stdout_redacted=stdout_redaction.blocked,
                stderr_redacted=stderr_redaction.blocked,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )

        status = RunStatus.SUCCEEDED if returncode == 0 else RunStatus.FAILED
        notes: list[str] = []
        if stdout_redaction.blocked:
            notes.append("redacted stdout")
        if stderr_redaction.blocked:
            notes.append("redacted stderr")
        if stdout_truncated:
            notes.append("truncated stdout")
        if stderr_truncated:
            notes.append("truncated stderr")
        elapsed = time.monotonic() - started
        reason = ", ".join(notes) if notes else f"completed in {elapsed:.3f}s"
        return CommandResult(
            status=status,
            argv=prepared.resolved_argv,
            returncode=returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            rule=prepared.rule.name,
            reason=reason,
            approval_id=approval_id,
            stdout_redacted=stdout_redaction.blocked,
            stderr_redacted=stderr_redaction.blocked,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def _prepare(
        self,
        request: CommandRequest,
        *,
        require_approval: bool = True,
        consume_approval: bool,
    ) -> _PreparedCommand:
        if not request.argv:
            raise PolicyViolation("empty argv refused")
        cwd = request.cwd.resolve()
        if cwd != self.workspace_root:
            raise PolicyViolation(f"command cwd must equal workspace root: {cwd}")

        normalized = normalize_argv(request.argv)
        rule = next(
            (
                candidate
                for candidate in self.policy.rules_for(request.domain)
                if normalized == candidate.argv_prefix
            ),
            None,
        )
        if rule is None:
            raise PolicyViolation(f"command is not exactly allowlisted: {' '.join(request.argv)}")

        for index in rule.workspace_path_args:
            try:
                raw_path = Path(request.argv[index])
            except IndexError as exc:
                raise PolicyViolation(
                    f"rule {rule.name} has an invalid path-argument index"
                ) from exc
            resolved_path = (
                raw_path.resolve()
                if raw_path.is_absolute()
                else (self.workspace_root / raw_path).resolve()
            )
            if not resolved_path.is_relative_to(self.workspace_root):
                raise PolicyViolation(f"path argument escapes workspace: {raw_path}")

        resolved_argv = (str(self._resolve_executable(request.argv[0], rule)), *request.argv[1:])
        approval: ApprovalClaims | None = None
        approval_needed = rule.requires_approval or (
            rule.network_required and self.policy.network_default == "deny"
        )
        if approval_needed and require_approval:
            if not request.approval_token:
                raise PolicyViolation(
                    f"approval capability required for {rule.name}: {rule.reason}"
                )
            if not request.approval_challenge_id:
                raise PolicyViolation(
                    f"approval challenge identity required for {rule.name}: {rule.reason}"
                )
            if self.approval_authority is None:
                raise PolicyViolation("runner has no approval authority configured")
            try:
                approval = self.approval_authority.verify(
                    request.approval_token,
                    workspace=self.workspace_root,
                    domain=request.domain,
                    rule=rule,
                    argv=resolved_argv,
                    expected_challenge_id=request.approval_challenge_id,
                    consume=consume_approval,
                )
            except ApprovalError as exc:
                raise PolicyViolation(str(exc)) from exc
        return _PreparedCommand(rule=rule, resolved_argv=resolved_argv, approval=approval)

    def _resolve_executable(self, requested: str, rule: CommandRule) -> Path:
        alias = normalize_token(requested)
        trusted = tuple(path.resolve() for path in self.policy.trusted_executables.get(alias, ()))
        workspace = tuple(
            path.resolve()
            for path in self.policy.allowed_workspace_executables
            if normalize_token(str(path)) == alias
        )
        candidates = (*trusted, *workspace)
        if not candidates:
            raise PolicyViolation(f"no trusted executable registered for {alias}")

        requested_path = Path(requested)
        has_path = requested_path.is_absolute() or requested_path.parent != Path(".")
        if has_path:
            resolved = requested_path.resolve()
            if resolved not in candidates:
                raise PolicyViolation(f"untrusted executable path refused: {resolved}")
        else:
            resolved = candidates[0]
        if not resolved.is_file():
            raise PolicyViolation(f"registered executable is missing: {resolved}")
        if resolved in workspace and not rule.requires_approval:
            raise PolicyViolation("workspace executable requires an approval-gated rule")
        return resolved

    def _read_output(self, stream: _BinaryStream) -> tuple[str, bool]:
        stream.seek(0)
        raw = stream.read(self.policy.output_limit_bytes + OUTPUT_REDACTION_OVERLAP_BYTES + 1)
        truncated = len(raw) > self.policy.output_limit_bytes
        return raw.decode("utf-8", errors="replace"), truncated

    def _limit_output(self, text: str, truncated: bool) -> str:
        if not truncated:
            return text
        raw = text.encode("utf-8")[: self.policy.output_limit_bytes]
        return raw.decode("utf-8", errors="replace") + "\n[OUTPUT_TRUNCATED]"

    def _safe_env(self) -> dict[str, str]:
        return build_safe_subprocess_env(self.policy.allowed_env_names)

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
        if os.name == "nt":
            taskkill = (
                Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "taskkill.exe"
            )
            with suppress(OSError, subprocess.TimeoutExpired):
                subprocess.run(
                    [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    timeout=PROCESS_TERMINATION_GRACE_SECONDS,
                )
            if process.poll() is None:
                with suppress(OSError):
                    process.kill()
            return
        killpg = cast(Callable[[int, int], None] | None, getattr(os, "killpg", None))
        sigkill = cast(int, getattr(signal, "SIGKILL", signal.SIGTERM))
        if killpg:
            with suppress(ProcessLookupError):
                killpg(process.pid, sigkill)
        elif process.poll() is None:
            with suppress(OSError):
                process.kill()

    @staticmethod
    def _failed(
        request: CommandRequest,
        prepared: _PreparedCommand,
        reason: str,
    ) -> CommandResult:
        return CommandResult(
            status=RunStatus.FAILED,
            argv=request.argv,
            returncode=None,
            stdout="",
            stderr="",
            rule=prepared.rule.name,
            reason=reason,
            approval_id=prepared.approval.approval_id if prepared.approval else None,
        )

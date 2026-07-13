import os
import subprocess
import sys
from pathlib import Path

import pytest

from inverse_agent.approvals import ApprovalAuthority, ApprovalError, SqliteApprovalReplayStore
from inverse_agent.models import CommandRule, Domain, RunnerPolicy, RunStatus
from inverse_agent.policies import default_policy
from inverse_agent.runner import ApprovalNotRequired, CommandRequest, LocalRunner, normalize_token

SECRET = b"test-approval-secret-that-is-at-least-32-bytes"


@pytest.mark.parametrize("token", ["python", "python3", "python3.12", "/usr/bin/python3.12"])
def test_normalize_token_accepts_versioned_python_executables(token: str) -> None:
    assert normalize_token(token) == "python"


def test_normalize_token_does_not_overmatch_python_tools() -> None:
    assert normalize_token("python-config") == "python-config"


def _python_runner(
    root: Path,
    argv: tuple[str, ...],
    *,
    approval: bool = False,
    output_limit: int = 2_000_000,
) -> tuple[LocalRunner, CommandRule, ApprovalAuthority]:
    rule = CommandRule(
        "probe",
        ("python", *argv[1:]),
        Domain.GENERIC,
        requires_approval=approval,
        reason="test probe",
    )
    policy = RunnerPolicy(
        workspace_root=root,
        allowed_commands=[rule],
        trusted_executables={"python": (Path(sys.executable).resolve(),)},
        output_limit_bytes=output_limit,
    )
    authority = ApprovalAuthority(SECRET)
    return LocalRunner(policy, authority), rule, authority


def _approved_request(
    runner: LocalRunner,
    authority: ApprovalAuthority,
    rule: CommandRule,
    request: CommandRequest,
    *,
    now: int | None = None,
    ttl_seconds: int = 300,
) -> CommandRequest:
    challenge = runner.approval_challenge(request)
    token, _claims = authority.issue(
        workspace=request.cwd,
        domain=request.domain,
        rule=rule,
        argv=challenge.argv,
        approved_by="tester",
        challenge_id="1" * 32,
        now=now,
        ttl_seconds=ttl_seconds,
    )
    return CommandRequest(
        argv=request.argv,
        cwd=request.cwd,
        domain=request.domain,
        approval_token=token,
        approval_challenge_id="1" * 32,
        timeout_seconds=request.timeout_seconds,
    )


def test_runner_refuses_unallowlisted_command(tmp_path: Path) -> None:
    result = LocalRunner(default_policy(tmp_path)).run(
        CommandRequest(("curl", "https://example.test"), tmp_path, Domain.GENERIC)
    )
    assert result.status == RunStatus.REFUSED
    assert "not exactly allowlisted" in result.reason


def test_runner_uses_typed_signal_when_approval_is_not_required(tmp_path: Path) -> None:
    argv = (sys.executable, "-c", "print('ok')")
    runner, _rule, _authority = _python_runner(tmp_path, argv)
    with pytest.raises(ApprovalNotRequired):
        runner.approval_challenge(CommandRequest(argv, tmp_path, Domain.GENERIC))


@pytest.mark.parametrize(
    "argv",
    [
        ("git", "diff", "--no-index", "C:/Windows/win.ini", "README.md"),
        ("git", "diff", "--output=C:/outside.txt"),
        ("ruff", "check", ".", "--fix"),
    ],
)
def test_runner_refuses_trailing_flag_bypasses(tmp_path: Path, argv: tuple[str, ...]) -> None:
    result = LocalRunner(default_policy(tmp_path)).run(
        CommandRequest(argv, tmp_path, Domain.GENERIC)
    )
    assert result.status == RunStatus.REFUSED
    assert "not exactly allowlisted" in result.reason


def test_runner_refuses_spoofed_executable_path(tmp_path: Path) -> None:
    policy = default_policy(tmp_path)
    policy.trusted_executables["git"] = (Path(sys.executable).resolve(),)
    result = LocalRunner(policy).run(
        CommandRequest(
            (str(tmp_path / "attacker" / "git.exe"), *policy.allowed_commands[0].argv_prefix[1:]),
            tmp_path,
            Domain.GENERIC,
        )
    )
    assert result.status == RunStatus.REFUSED
    assert "untrusted executable path" in result.reason


def test_runner_requires_exact_workspace_root(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    result = LocalRunner(default_policy(tmp_path)).run(
        CommandRequest(("git", "status"), child, Domain.GENERIC)
    )
    assert result.status == RunStatus.REFUSED
    assert "cwd must equal workspace root" in result.reason


def test_runner_refuses_workspace_path_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    link = tmp_path / "script.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    rule = CommandRule(
        "script",
        ("python", "script.py"),
        Domain.GENERIC,
        workspace_path_args=(1,),
    )
    policy = RunnerPolicy(
        tmp_path,
        [rule],
        trusted_executables={"python": (Path(sys.executable).resolve(),)},
    )
    result = LocalRunner(policy).run(
        CommandRequest((sys.executable, "script.py"), tmp_path, Domain.GENERIC)
    )
    assert result.status == RunStatus.REFUSED
    assert "path argument escapes workspace" in result.reason


def test_signed_approval_is_required_and_consumed_once(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    argv = (sys.executable, "probe.py")
    runner, rule, authority = _python_runner(tmp_path, argv, approval=True)
    request = CommandRequest(argv, tmp_path, Domain.GENERIC)

    denied = runner.run(request)
    approved = _approved_request(runner, authority, rule, request)
    first = runner.run(approved)
    replay = runner.run(approved)

    assert denied.status == RunStatus.REFUSED
    assert "approval capability required" in denied.reason
    assert first.status == RunStatus.SUCCEEDED
    assert first.approval_id
    assert replay.status == RunStatus.REFUSED
    assert "already consumed" in replay.reason


def test_approval_cannot_cross_command_scope(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "two.py").write_text("print(2)\n", encoding="utf-8")
    rule_one = CommandRule("one", ("python", "one.py"), Domain.GENERIC, requires_approval=True)
    rule_two = CommandRule("two", ("python", "two.py"), Domain.GENERIC, requires_approval=True)
    policy = RunnerPolicy(
        tmp_path,
        [rule_one, rule_two],
        trusted_executables={"python": (Path(sys.executable).resolve(),)},
    )
    authority = ApprovalAuthority(SECRET)
    runner = LocalRunner(policy, authority)
    one = CommandRequest((sys.executable, "one.py"), tmp_path, Domain.GENERIC)
    approved_one = _approved_request(runner, authority, rule_one, one)
    two = CommandRequest(
        (sys.executable, "two.py"),
        tmp_path,
        Domain.GENERIC,
        approval_token=approved_one.approval_token,
        approval_challenge_id=approved_one.approval_challenge_id,
    )
    result = runner.run(two)
    assert result.status == RunStatus.REFUSED
    assert "does not match this action" in result.reason


def test_approval_replay_is_refused_after_authority_restart(tmp_path: Path) -> None:
    rule = CommandRule("probe", ("python", "probe.py"), Domain.GENERIC, requires_approval=True)
    argv = (str(Path(sys.executable).resolve()), "probe.py")
    replay_path = tmp_path / "replay.sqlite"
    first = ApprovalAuthority(SECRET, SqliteApprovalReplayStore(replay_path))
    token, _claims = first.issue(
        workspace=tmp_path,
        domain=Domain.GENERIC,
        rule=rule,
        argv=argv,
        approved_by="tester",
        challenge_id="2" * 32,
    )
    first.verify(
        token,
        workspace=tmp_path,
        domain=Domain.GENERIC,
        rule=rule,
        argv=argv,
        expected_challenge_id="2" * 32,
    )
    restarted = ApprovalAuthority(SECRET, SqliteApprovalReplayStore(replay_path))
    with pytest.raises(ApprovalError, match="already consumed"):
        restarted.verify(
            token,
            workspace=tmp_path,
            domain=Domain.GENERIC,
            rule=rule,
            argv=argv,
            expected_challenge_id="2" * 32,
        )


def test_approval_capability_is_bound_to_current_challenge(tmp_path: Path) -> None:
    rule = CommandRule("probe", ("python", "probe.py"), Domain.GENERIC, requires_approval=True)
    argv = (str(Path(sys.executable).resolve()), "probe.py")
    authority = ApprovalAuthority(SECRET)
    token, _claims = authority.issue(
        workspace=tmp_path,
        domain=Domain.GENERIC,
        rule=rule,
        argv=argv,
        approved_by="tester",
        challenge_id="a" * 32,
    )

    with pytest.raises(ApprovalError, match="current challenge"):
        authority.verify(
            token,
            workspace=tmp_path,
            domain=Domain.GENERIC,
            rule=rule,
            argv=argv,
            expected_challenge_id="b" * 32,
            consume=False,
        )

    claims = authority.verify(
        token,
        workspace=tmp_path,
        domain=Domain.GENERIC,
        rule=rule,
        argv=argv,
        expected_challenge_id="a" * 32,
    )
    assert claims.challenge_id == "a" * 32


def test_expired_approval_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "probe.py").write_text("print('ok')\n", encoding="utf-8")
    argv = (sys.executable, "probe.py")
    runner, rule, authority = _python_runner(tmp_path, argv, approval=True)
    request = CommandRequest(argv, tmp_path, Domain.GENERIC)
    approved = _approved_request(runner, authority, rule, request, now=100, ttl_seconds=1)
    monkeypatch.setattr("inverse_agent.approvals.time.time", lambda: 102)
    result = runner.run(approved)
    assert result.status == RunStatus.REFUSED
    assert "expired" in result.reason


def test_runner_returns_failed_for_timeout_with_partial_output(tmp_path: Path) -> None:
    script = tmp_path / "sleep_probe.py"
    script.write_text(
        "import time\nprint('partial', flush=True)\ntime.sleep(3)\n", encoding="utf-8"
    )
    argv = (sys.executable, "sleep_probe.py")
    runner, _rule, _authority = _python_runner(tmp_path, argv)
    result = runner.run(CommandRequest(argv, tmp_path, Domain.GENERIC, timeout_seconds=1))
    assert result.status == RunStatus.FAILED
    assert "compute budget" in result.reason
    assert "partial" in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows taskkill timeout regression")
def test_windows_taskkill_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, float] = {}

    def stalled_taskkill(*_args, **kwargs) -> None:
        observed["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired("taskkill", kwargs["timeout"])

    class ProcessProbe:
        pid = 42
        killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

    process = ProcessProbe()
    monkeypatch.setattr(subprocess, "run", stalled_taskkill)
    LocalRunner._terminate_process_tree(process)  # type: ignore[arg-type]
    assert observed["timeout"] > 0
    assert process.killed


def test_runner_caps_output(tmp_path: Path) -> None:
    script = tmp_path / "output_probe.py"
    script.write_text("print('x' * 1000)\n", encoding="utf-8")
    argv = (sys.executable, "output_probe.py")
    runner, _rule, _authority = _python_runner(tmp_path, argv, output_limit=100)
    result = runner.run(CommandRequest(argv, tmp_path, Domain.GENERIC))
    assert result.status == RunStatus.SUCCEEDED
    assert "[OUTPUT_TRUNCATED]" in result.stdout
    assert "truncated stdout" in result.reason


def test_runner_redacts_secret_that_crosses_output_limit(tmp_path: Path) -> None:
    secret = "_".join(("ghp", "abcdefghijklmnopqrstuvwxyz123456"))
    script = tmp_path / "boundary_secret.py"
    script.write_text(f"print({'x' * 74 + ' ' + secret!r})\n", encoding="utf-8")
    argv = (sys.executable, "boundary_secret.py")
    runner, _rule, _authority = _python_runner(tmp_path, argv, output_limit=100)
    result = runner.run(CommandRequest(argv, tmp_path, Domain.GENERIC))
    assert result.status == RunStatus.SUCCEEDED
    assert secret[:20] not in result.stdout
    assert "[REDACTED_SECRET]" in result.stdout
    assert "[OUTPUT_TRUNCATED]" in result.stdout


def test_runner_does_not_inherit_secret_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INVERSE_AGENT_API_TOKEN", "token_should_not_be_visible")
    monkeypatch.setenv("INVERSE_AGENT_MODEL_API_KEY", "model_key_should_not_be_visible")
    script = tmp_path / "env_probe.py"
    script.write_text(
        "import os\n"
        "print(os.environ.get('INVERSE_AGENT_API_TOKEN', 'missing'))\n"
        "print(os.environ.get('INVERSE_AGENT_MODEL_API_KEY', 'missing'))\n"
        "print(os.environ.get('GIT_OPTIONAL_LOCKS', 'missing'))\n"
        "print(os.environ.get('GIT_CONFIG_NOSYSTEM', 'missing'))\n"
        "print(os.environ.get('GIT_NO_LAZY_FETCH', 'missing'))\n"
        "print(os.environ.get('GIT_NO_REPLACE_OBJECTS', 'missing'))\n",
        encoding="utf-8",
    )
    argv = (sys.executable, "env_probe.py")
    runner, _rule, _authority = _python_runner(tmp_path, argv)
    result = runner.run(CommandRequest(argv, tmp_path, Domain.GENERIC))
    assert result.status == RunStatus.SUCCEEDED
    assert result.stdout.splitlines() == ["missing", "missing", "0", "1", "1", "1"]


@pytest.mark.skipif(os.name != "nt", reason="Windows path-spoof regression")
def test_explicit_system_python_path_is_trusted_on_windows(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    argv = (sys.executable, "probe.py")
    runner, _rule, _authority = _python_runner(tmp_path, argv)
    result = runner.run(CommandRequest(argv, tmp_path, Domain.GENERIC))
    assert result.status == RunStatus.SUCCEEDED

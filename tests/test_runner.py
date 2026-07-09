import sys
from pathlib import Path

from inverse_agent.models import CommandRule, Domain, RunnerPolicy, RunStatus
from inverse_agent.policies import default_policy
from inverse_agent.runner import CommandRequest, LocalRunner


def test_runner_refuses_shell_metacharacters(tmp_path: Path) -> None:
    runner = LocalRunner(default_policy(tmp_path))
    result = runner.run(CommandRequest(argv=("python", "-c", "print(1); rm -rf /"), cwd=tmp_path, domain=Domain.DJANGO))

    assert result.status == RunStatus.REFUSED
    assert "shell metacharacter" in result.reason


def test_runner_refuses_outside_workspace(tmp_path: Path) -> None:
    runner = LocalRunner(default_policy(tmp_path))
    result = runner.run(CommandRequest(argv=("git", "status"), cwd=Path.cwd(), domain=Domain.GENERIC))

    assert result.status == RunStatus.REFUSED


def test_runner_refuses_unallowlisted_command_inside_workspace(tmp_path: Path) -> None:
    runner = LocalRunner(default_policy(tmp_path))
    result = runner.run(CommandRequest(argv=("curl", "https://example.test"), cwd=tmp_path, domain=Domain.GENERIC))

    assert result.status == RunStatus.REFUSED
    assert "not allowlisted" in result.reason


def test_runner_enforces_approval_required_branch() -> None:
    fixture = Path(__file__).parent / "fixtures" / "django_project"
    runner = LocalRunner(default_policy(fixture))

    denied = runner.run(
        CommandRequest(argv=(sys.executable, "manage.py", "test"), cwd=fixture, domain=Domain.DJANGO)
    )
    allowed = runner.run(
        CommandRequest(
            argv=(sys.executable, "manage.py", "test"),
            cwd=fixture,
            domain=Domain.DJANGO,
            approved=True,
        )
    )

    assert denied.status == RunStatus.REFUSED
    assert "approval required" in denied.reason
    assert allowed.status == RunStatus.SUCCEEDED


def test_runner_enforces_network_required_branch(tmp_path: Path) -> None:
    policy = RunnerPolicy(
        workspace_root=tmp_path,
        allowed_commands=[
            CommandRule(
                "network-tool",
                ("network-tool",),
                Domain.GENERIC,
                network_required=True,
                reason="network smoke",
            ),
        ],
    )
    result = LocalRunner(policy).run(
        CommandRequest(argv=("network-tool",), cwd=tmp_path, domain=Domain.GENERIC)
    )

    assert result.status == RunStatus.REFUSED
    assert "network approval required" in result.reason


def test_runner_returns_failed_for_missing_binary(tmp_path: Path) -> None:
    policy = RunnerPolicy(
        workspace_root=tmp_path,
        allowed_commands=[
            CommandRule("missing", ("missing-inverse-agent-tool",), Domain.GENERIC),
        ],
    )
    result = LocalRunner(policy).run(
        CommandRequest(argv=("missing-inverse-agent-tool",), cwd=tmp_path, domain=Domain.GENERIC)
    )

    assert result.status == RunStatus.FAILED
    assert "executable not found" in result.reason


def test_runner_returns_failed_for_timeout(tmp_path: Path) -> None:
    script = tmp_path / "sleep_probe.py"
    script.write_text("import time\ntime.sleep(2)\n", encoding="utf-8")
    policy = RunnerPolicy(
        workspace_root=tmp_path,
        allowed_commands=[
            CommandRule("sleep-probe", ("python", "sleep_probe.py"), Domain.GENERIC),
        ],
    )
    result = LocalRunner(policy).run(
        CommandRequest(
            argv=(sys.executable, "sleep_probe.py"),
            cwd=tmp_path,
            domain=Domain.GENERIC,
            timeout_seconds=1,
        )
    )

    assert result.status == RunStatus.FAILED
    assert "compute budget" in result.reason


def test_runner_does_not_inherit_secret_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INVERSE_AGENT_API_TOKEN", "token_should_not_be_visible")
    script = tmp_path / "env_probe.py"
    script.write_text(
        "import os\nprint(os.environ.get('INVERSE_AGENT_API_TOKEN', 'missing'))\n",
        encoding="utf-8",
    )
    policy = RunnerPolicy(
        workspace_root=tmp_path,
        allowed_commands=[
            CommandRule("env-probe", ("python", "env_probe.py"), Domain.GENERIC),
        ],
    )
    result = LocalRunner(policy).run(
        CommandRequest(argv=(sys.executable, "env_probe.py"), cwd=tmp_path, domain=Domain.GENERIC)
    )

    assert result.status == RunStatus.SUCCEEDED
    assert result.stdout.strip() == "missing"


def test_runner_allows_django_check_fixture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "django_project"
    runner = LocalRunner(default_policy(fixture))
    result = runner.run(
        CommandRequest(argv=(sys.executable, "manage.py", "check"), cwd=fixture, domain=Domain.DJANGO)
    )

    assert result.status == RunStatus.SUCCEEDED
    assert "System check identified no issues" in result.stdout

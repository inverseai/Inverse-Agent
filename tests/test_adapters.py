import platform
from pathlib import Path

from inverse_agent.adapters.android import AndroidAdapter, AndroidNdkAdapter
from inverse_agent.adapters.django import DjangoAdapter
from inverse_agent.adapters.generic import GenericGitAdapter
from inverse_agent.adapters.ios import IosAdapter
from inverse_agent.adapters.pytorch import PyTorchAdapter
from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.models import Domain
from inverse_agent.policies import GIT_STATUS_ARGV, default_policy

FIXTURES = Path(__file__).parent / "fixtures"


def test_django_adapter_detects_fixture_and_records_python_source() -> None:
    root = FIXTURES / "django_project"
    profile = DjangoAdapter().profile(root)
    assert DjangoAdapter().detect(root)
    assert Domain.DJANGO in profile.domains
    assert Path(profile.commands["check"][0]).is_absolute()
    assert profile.toolchain["python_source"]


def test_pytorch_adapter_detects_fixture() -> None:
    root = FIXTURES / "pytorch_project"
    profile = PyTorchAdapter().profile(root)
    assert PyTorchAdapter().detect(root)
    assert "smoke_train" in profile.commands
    assert Path(profile.commands["smoke_train"][0]).is_absolute()


def test_pytorch_detection_reads_only_bounded_prefix(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_bytes(
        b"\xff" * (1024 * 1024) + b"\ntorch\n",
    )
    assert not PyTorchAdapter().detect(tmp_path)


def test_android_uses_absolute_offline_wrapper() -> None:
    root = FIXTURES / "android_project"
    profile = AndroidAdapter().profile(root)
    assert AndroidAdapter().detect(root)
    assert AndroidNdkAdapter().detect(root)
    command = profile.commands["tasks"]
    assert Path(command[0]).is_absolute()
    assert command[1:] == ["--offline", "tasks"]


def test_ios_detection_is_explicitly_unavailable_off_macos() -> None:
    root = FIXTURES / "ios_project"
    profile = IosAdapter().profile(root)
    assert IosAdapter().detect(root)
    if platform.system() != "Darwin":
        assert not profile.commands
        assert "xcodebuild" in profile.unavailable_tools


def test_registry_merges_domains_with_deterministic_labels() -> None:
    profile = detect_workspace(FIXTURES / "android_project")
    assert Domain.ANDROID in profile.domains
    assert Domain.ANDROID_NDK in profile.domains
    assert all(name.startswith(("android.", "android_ndk.")) for name in profile.commands)


def test_generic_git_adapter_registers_hardened_read_tools(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    adapter = GenericGitAdapter()
    profile = adapter.profile(tmp_path)

    assert adapter.detect(tmp_path)
    assert Domain.GENERIC in profile.domains
    assert set(profile.commands) == {"status", "tracked_files"}
    assert Path(profile.commands["status"][0]).is_absolute()
    assert profile.commands["status"][1:] == list(GIT_STATUS_ARGV[1:])
    assert "--no-optional-locks" in profile.commands["status"]
    assert {tool.safety for tool in adapter.tools()} == {"approval-required"}
    git_rules = [
        rule for rule in default_policy(tmp_path).allowed_commands if rule.name.startswith("git-")
    ]
    assert git_rules and all(rule.requires_approval for rule in git_rules)
    status_rule = next(rule for rule in git_rules if rule.name == "git-status")
    assert "repository-configured clean/filter helpers" in status_rule.reason

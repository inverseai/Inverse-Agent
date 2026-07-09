from pathlib import Path

from inverse_agent.adapters.android import AndroidAdapter, AndroidNdkAdapter
from inverse_agent.adapters.django import DjangoAdapter
from inverse_agent.adapters.ios import IosAdapter
from inverse_agent.adapters.pytorch import PyTorchAdapter
from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.models import Domain

FIXTURES = Path(__file__).parent / "fixtures"


def test_django_adapter_detects_fixture() -> None:
    root = FIXTURES / "django_project"
    adapter = DjangoAdapter()

    assert adapter.detect(root)
    assert Domain.DJANGO in adapter.profile(root).domains


def test_pytorch_adapter_detects_fixture() -> None:
    root = FIXTURES / "pytorch_project"
    adapter = PyTorchAdapter()

    assert adapter.detect(root)
    assert "smoke_train" in adapter.profile(root).commands


def test_android_and_ndk_detection() -> None:
    root = FIXTURES / "android_project"

    assert AndroidAdapter().detect(root)
    assert AndroidNdkAdapter().detect(root)


def test_ios_detection() -> None:
    root = FIXTURES / "ios_project"

    assert IosAdapter().detect(root)


def test_registry_merges_domains() -> None:
    profile = detect_workspace(FIXTURES / "android_project")

    assert Domain.ANDROID in profile.domains
    assert Domain.ANDROID_NDK in profile.domains


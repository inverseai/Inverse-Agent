import pytest

from inverse_agent.model_config import MODEL_ENV_NAMES


@pytest.fixture(autouse=True)
def clear_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MODEL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

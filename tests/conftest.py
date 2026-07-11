from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import pytest

from inverse_agent.model_config import MODEL_ENV_NAMES


@pytest.fixture(scope="session", autouse=True)
def application_temp_directory(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    root = tmp_path_factory.getbasetemp() / "application-temp"
    root.mkdir()
    previous_tempdir = tempfile.tempdir
    previous_model_environment = {
        name: os.environ[name] for name in MODEL_ENV_NAMES if name in os.environ
    }
    for name in MODEL_ENV_NAMES:
        os.environ.pop(name, None)
    tempfile.tempdir = str(root)
    try:
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        for name in MODEL_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(previous_model_environment)

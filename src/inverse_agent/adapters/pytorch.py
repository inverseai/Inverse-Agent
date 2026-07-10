"""PyTorch research-engineering adapter."""

from __future__ import annotations

from pathlib import Path

from inverse_agent.adapters.base import CommandAdapter, Tool
from inverse_agent.environments import discover_python
from inverse_agent.models import Domain, WorkspaceProfile


class PyTorchAdapter(CommandAdapter):
    domain = Domain.PYTORCH

    def detect(self, root: Path) -> bool:
        markers = ["train.py", "eval.py", "requirements.txt", "pyproject.toml"]
        return any((root / marker).exists() for marker in markers) and self._mentions_torch(root)

    def profile(self, root: Path) -> WorkspaceProfile:
        environment = discover_python(root)
        python = str(environment.path)
        commands: dict[str, list[str]] = {}
        if (root / "train.py").exists():
            commands["smoke_train"] = [python, "train.py", "--smoke"]
        if (root / "eval.py").exists():
            commands["eval"] = [python, "eval.py"]
        return WorkspaceProfile(
            root=root,
            domains={Domain.PYTORCH},
            commands=commands,
            test_targets=list(commands),
            toolchain={
                "python": python,
                "python_source": environment.source,
                "framework": "pytorch",
            },
        )

    def tools(self) -> list[Tool]:
        return [
            Tool("pytorch.smoke_train", "Run a configured smoke training job", "budgeted", self.domain),
            Tool("pytorch.eval", "Run a configured evaluation command", "budgeted", self.domain),
            Tool("pytorch.report", "Create a reproducibility report", "safe-read", self.domain),
        ]

    @staticmethod
    def _mentions_torch(root: Path) -> bool:
        for name in ("requirements.txt", "pyproject.toml", "train.py", "eval.py"):
            path = root / name
            if path.exists() and "torch" in path.read_text(encoding="utf-8", errors="ignore").lower():
                return True
        return False

# Inverse-Agent

Inverse-Agent is an OSS-composed agentic workbench for engineering and research workflows. The v0.1 implementation focuses on an in-house dogfood loop across Django, PyTorch research engineering, Android/NDK, and iOS/Xcode while keeping the agent core small, auditable, and policy-driven.

## Principles

- Codex writes code; Claude Code is review-only.
- Compose existing ecosystems instead of rebuilding agent infrastructure.
- Keep local execution local, with explicit inference-plane policy.
- Default-deny command execution through structured argv allowlists.
- Scope autonomy per workspace and toolchain domain, never globally.
- Capture artifacts, approvals, and eval traces for every meaningful run.

## Quickstart

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m inverse_agent.cli profile tests\fixtures\django_project
.\.venv\Scripts\python.exe -m inverse_agent.cli run-django tests\fixtures\django_project
```

The repository also supports the Codex bundled Python runtime used during development:

```powershell
& "C:\Users\calic\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## v0.1 Scope

- Core contracts for workspace profiles, agents, runs, policies, artifacts, and eval traces.
- Local runner with default-deny policy enforcement and redaction hooks.
- Workflow skeleton with approval checkpoints and resumable trace records.
- MCP-style adapter contract with first-pass domain adapters.
- Django issue-to-PR style replay against a fixture project.
- PyTorch, Android/NDK, and iOS adapters with safe command generation and host guards.
- Minimal FastAPI control-plane API for runs, approvals, artifacts, and eval traces.

See [docs/architecture.md](docs/architecture.md), [docs/safety.md](docs/safety.md), and [docs/review-log.md](docs/review-log.md).


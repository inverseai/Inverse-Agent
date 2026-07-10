# Inverse-Agent

Inverse-Agent is a local-first agentic workbench for engineering and ML research. It gives a model a registry of typed tools rather than a raw shell, pauses durable workflows before workspace code executes, and resumes only after a human-authorized capability is verified.

The first dogfood domains match the in-house team:

- Native Android and Gradle
- Android NDK and CMake
- Native iOS and Xcode
- Django full-stack projects
- PyTorch research engineering

## Runtime Shape

1. The workspace profiler discovers domains, concrete executables, and unavailable toolchains.
2. A deterministic or model-backed planner selects registered tool identifiers. Models cannot emit raw commands.
3. LangGraph persists the run to SQLite and interrupts before an executable action.
4. The authenticated control plane records human approval and issues a signed, expiring, single-use capability bound to the exact action.
5. The runner verifies the capability, resolves a trusted executable, enforces exact argv, executes with a curated environment and output limits, then records a redacted trace.

Workspaces must be explicitly trusted before code execution. Untrusted repositories should be inspected in advisory mode or moved to an external VM/container sandbox.

## Quickstart

Install the universal lock exactly:

```powershell
uv sync --locked --extra dev --python 3.12
uv run pytest --cov=inverse_agent --cov-report=term-missing
uv run ruff check .
uv run mypy
```

Configure local secrets. The approval secret must be stable across CLI invocations because it signs resume capabilities; these credentials are not inherited by workspace subprocesses.

```powershell
$env:INVERSE_AGENT_APPROVAL_SECRET = "replace-with-at-least-32-random-bytes"
$env:INVERSE_AGENT_API_TOKEN = "replace-with-a-random-control-plane-token"
$env:INVERSE_AGENT_APPROVER_TOKEN = "replace-with-a-separate-human-approver-token"
$env:INVERSE_AGENT_APPROVER_ID = "engineer@example.com"
```

Profile, trust, and start a Django run:

```powershell
uv run inverse-agent profile D:\work\django-app
uv run inverse-agent trust-workspace D:\work\django-app `
  --workspace-root D:\work `
  --trusted-by engineer@example.com
uv run inverse-agent start D:\work\django-app --domain django
```

A waiting run exits with code `2` and prints its approval challenge. Approve the displayed run ID; each risky action receives its own approval.

```powershell
uv run inverse-agent approve RUN_ID `
  --workspace-root D:\work `
  --approved-by engineer@example.com `
  --action-digest ACTION_DIGEST_FROM_CHALLENGE
```

## Control Plane And MCP

The FastAPI control plane binds to `127.0.0.1`, refuses to start without both required secrets, and authenticates every endpoint except `/health`.

```powershell
uv run inverse-agent serve --workspace-root D:\work
```

The MCP stdio server exposes profiling, run creation, run start, and run status. It intentionally does not expose workspace trust or approval issuance, so a model cannot approve its own actions.

```powershell
uv run inverse-agent mcp --workspace-root D:\work
```

## Model Providers

`StructuredPlanner` accepts any `JsonModelClient`. `OpenAICompatibleClient` supports cloud or local OpenAI-compatible endpoints such as vLLM and LM Studio. Planning prompts contain the goal, domain, and registered tool names; raw source is not sent by the planner.

`DeterministicPlanner` is the offline baseline used by CI and reproducible evaluations.

## Safety Boundary

- Exact argv matching; unknown flags and trailing arguments are refused.
- Caller-provided executable paths are never trusted or executed verbatim.
- Workspace wrappers and virtual-environment executables require trusted-workspace attestation and approval-gated rules.
- Approval capabilities are HMAC-signed, expire quickly, are action-bound, and have SQLite-backed replay protection.
- Gradle verification runs offline. Git inspection disables global/system config, prompts, pagers, and fsmonitor hooks.
- Subprocess environments omit API tokens and other undeclared variables.
- Output is size-capped and secret-like material, including complete PEM blocks, is redacted.
- Timeouts terminate the spawned process group or Windows process tree.

This is an application-layer boundary, not an OS sandbox. See [docs/safety.md](docs/safety.md) for the threat model and residual risks.

## Development Gates

CI installs `uv.lock` and runs Ruff, strict Mypy, coverage-gated Pytest, and domain tests on Linux, Windows, and macOS. Claude Code is review-only; Codex writes the implementation. Review outcomes are recorded in [docs/review-log.md](docs/review-log.md).

No release, tag, or package publication is performed by the development workflow.

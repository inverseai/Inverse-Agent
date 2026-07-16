# Inverse-Agent

Inverse-Agent is a local-first agentic workbench for engineering and ML research. It gives a model a registry of typed tools rather than a raw shell, pauses durable workflows before workspace code executes, and resumes only after a human-authorized capability is verified.

The first dogfood domains match the in-house team:

- Native Android and Gradle
- Android NDK and CMake
- Native iOS and Xcode
- Django full-stack projects
- PyTorch research engineering
- Approval-gated generic Git repository inspection for monorepo roots
- Structured commit review for Android, iOS, C/C++, Django, PyTorch, and generic changes

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

### Engineering Workbench

Start the local GPT-OSS-20B model and Codex-style workbench together. The workspace root is the boundary within which engineers can select Android, iOS, Django, C/C++, or PyTorch projects. Investigation mode is enabled only when the loopback model's measured context and conservative bytes-per-token estimator are supplied together.

```powershell
.\scripts\start-workbench.ps1 -WorkspaceRoot "D:\Office Repos" `
  -ModelContextTokens 32768 `
  -ModelEstimatorBytesPerToken 2.0 `
  -ModelReasoningEffort high
```

The numbers above are examples, not defaults: use the pair produced by calibration for the exact local model, LM Studio build, and GPU configuration. Omitting both keeps verification available and leaves investigation disabled. Reasoning effort is a separate endpoint capability: the launcher sends it only when explicitly selected, and `default` omits the non-universal request field.

Open the printed loopback URL and enter the two session credentials shown in the same terminal. The operator token can profile workspaces and create tasks. The separate approver token is required to trust a workspace, approve a command, or decline a pending command. Operator access lasts only for the browser tab; approval access is memory-only and must be re-entered after a reload.

The workbench exposes advisory and approval-gated verification plus bounded read-only investigation. Investigation requires revocable `source_read` consent; assisted investigation also requires `code_execution`, and every frozen command still receives a fresh action-bound approval. The UI reconstructs durable events, budgets, evidence citations, final answers, cancellation, and bounded redacted traces after reconnect. It does not edit source code or provide an unrestricted shell.

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
  --action-digest ACTION_DIGEST_FROM_CHALLENGE `
  --challenge-id CHALLENGE_ID_FROM_CHALLENGE
```

## Control Plane And MCP

The FastAPI control plane serves the workbench and API from the same `127.0.0.1` origin, refuses to start without both required secrets, and authenticates every endpoint except the UI assets and `/health`.

```powershell
uv run inverse-agent serve --workspace-root D:\work
```

The MCP stdio server exposes safe projections for profiling, run creation/start/status/listing, plans, and bounded traces. It omits absolute paths, approval challenges, command output, workspace trust, and approval issuance, so a model cannot approve its own actions or recover source-bearing browser data through MCP.

```powershell
uv run inverse-agent mcp --workspace-root D:\work
```

## Model Providers

`StructuredPlanner` accepts any `JsonModelClient`. `OpenAICompatibleClient` supports cloud or local OpenAI-compatible endpoints such as vLLM and LM Studio. Planning prompts contain the goal, domain, and registered tool names; raw source is not sent by the planner.

`DeterministicPlanner` is the offline baseline used by CI and reproducible evaluations.

### Local GPT-OSS-20B

Load GPT-OSS-20B in LM Studio with the context length selected by the calibration gate and a stable API identifier:

```powershell
.\scripts\start-local-model.ps1 -ContextLength 32768
```

Configure Inverse-Agent in the same shell and verify structured planning before starting a run:

```powershell
$env:INVERSE_AGENT_MODEL_NAME = "inverse-gpt-oss-20b"
$env:INVERSE_AGENT_MODEL_BASE_URL = "http://127.0.0.1:1234/v1"
$env:INVERSE_AGENT_MODEL_CONTEXT_TOKENS = "32768"
$env:INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN = "2.0"
$env:INVERSE_AGENT_MODEL_REASONING_EFFORT = "high"
uv run inverse-agent model-check
uv run inverse-agent start D:\work\django-app --domain django
```

When neither model variable is present, Inverse-Agent remains deterministic. Model name and base URL must be configured together; investigation additionally requires the calibrated context/estimator pair and a numeric loopback endpoint. Reasoning effort is optional for production but, when configured, is validated, fingerprinted, and sent only as an explicit endpoint capability. Model failures stop planning and never silently fall back. `evaluate` remains deterministic unless `--use-model` is supplied explicitly.

Remote model endpoints are denied by default. They require HTTPS, `INVERSE_AGENT_MODEL_ALLOW_REMOTE=1`, and the `--model-allow-remote` flag together. API keys are accepted only through `INVERSE_AGENT_MODEL_API_KEY`, never through a command-line flag.

## Commit Review

Review a hexadecimal commit object ID with the local model:

```powershell
uv run inverse-agent review-commit D:\work\project COMMIT_SHA `
  --domain android `
  --goal "Review this change for introduced correctness and security defects" `
  --model inverse-gpt-oss-20b `
  --model-base-url http://127.0.0.1:1234/v1
```

The reader resolves the immutable commit with replacement objects and lazy fetch disabled, reads bounded Git blobs and raw tree modes, and constructs the unified diff in Python. It requires a normal in-workspace `.git` directory and refuses linked, common, alternate, grafted, or shallow object histories. It does not invoke repository hooks, external diff drivers, text conversion commands, or workspace code. Two independent model scouts propose findings under a fair candidate budget; PyTorch adds three contract-focused scouts for evaluation mode and normalization leakage so independent experiment defects are not lost to attention competition. A final model pass must accept or reject every labeled candidate. Findings supply verbatim source or Git-metadata evidence and its added/removed side, which Inverse-Agent maps to one unique changed line. Accepted scout findings are retained as provenance so the benchmark can require model support for every expected defect. High-signal Android, iOS, C++, Django, and PyTorch checks provide narrowly scoped redacted evidence, and the OpenAI-compatible client validates returned JSON locally against the requested schema. Omitted changed bytes, mixed line-ending ambiguity, unresolved or partial imported context, bounded dependency/context overflow, and candidate-budget overflow force an `INCOMPLETE` verdict. `review-commit` exits `0` for `PASS`, `1` for `FINDINGS`, and `3` for `INCOMPLETE`.

Run the packaged six-case acceptance benchmark, including the real `482fa05` Inverse-Agent control:

```powershell
uv run inverse-agent benchmark-review builtin `
  --repository-root . `
  --model inverse-gpt-oss-20b `
  --model-base-url http://127.0.0.1:1234/v1
```

The source checkout also retains `benchmarks\commit_review\suite.json` for fixture development. See the [commit-review benchmark specification](https://github.com/inverseai/Inverse-Agent/blob/main/docs/commit-review-benchmark.md) for task definitions and scoring.

## Safety Boundary

- Exact argv matching; unknown flags and trailing arguments are refused.
- Caller-provided executable paths are never trusted or executed verbatim.
- Workspace wrappers and virtual-environment executables require trusted-workspace attestation and approval-gated rules.
- Approval capabilities are HMAC-signed, expire quickly, are action-bound, and have SQLite-backed replay protection.
- Gradle verification runs offline. Git inspection disables global/system config, prompts, pagers, and fsmonitor hooks.
- Subprocess environments omit API tokens and other undeclared variables.
- Output is size-capped and secret-like material, including complete PEM blocks, is redacted.
- Source reads and directory walks are handle-relative, refuse links/reparse points and hard links, and never reopen a validated pathname to fetch bytes.
- Timeouts terminate the spawned process group or Windows process tree.
- Commit review accepts only bounded hexadecimal object IDs, uses fixed-location system Git discovery, neutralizes likely source-directed model instructions, hides filenames behind opaque IDs, and validates every model-reported line against an immutable changed hunk.

This is an application-layer boundary, not an OS sandbox. See [docs/safety.md](docs/safety.md) for the threat model and residual risks.

## Development Gates

CI installs `uv.lock` and runs Ruff, strict Mypy, coverage-gated Pytest, and domain tests on Linux, Windows, and macOS. Claude Code is review-only; Codex writes the implementation. Review outcomes are recorded in [docs/review-log.md](docs/review-log.md).

No release, tag, or package publication is performed by the development workflow.

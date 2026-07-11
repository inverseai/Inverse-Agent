# Architecture

Inverse-Agent separates decisions, execution, and product state so no model-controlled component can grant itself authority.

## Components

- `planner.py`: provider-neutral planning. Models select registered tool IDs under a strict action budget; raw commands are rejected and structured responses are validated locally against JSON Schema.
- `model_config.py`: validated model configuration, endpoint policy, planner construction, and non-secret provenance.
- `adapters/`: domain detection and command generation for Django, PyTorch, Android/NDK, and iOS.
- `policies.py`: exact command rules and the concrete trusted executable map.
- `approvals.py`: signed action capabilities plus durable replay prevention.
- `runner.py`: executable resolution, command enforcement, process isolation controls, capped capture, and redaction.
- `workflow.py`: LangGraph planning/execution graph with SQLite checkpoints and approval interrupts.
- `service.py`: persisted runs, explicit workspace trust, approval issuance, and workflow resume.
- `control_plane.py`: authenticated HTTP API, safe run projections, and exact same-origin UI assets.
- `ui/`: dependency-free engineering workbench with task history, plan rationale, approval gates, run output, and workspace inspection.
- `mcp_server.py`: executable MCP tools. Approval and trust mutation are intentionally absent.
- `eval.py`: portable trace serialization.
- `commit_review.py`: immutable Git blob extraction, bounded diff construction, source-instruction neutralization, multi-scout review, evidence adjudication, and structured finding validation.
- `review_benchmark.py`: hermetic multi-domain repository materialization and acceptance scoring.

## Trust Flow

The model sees goals, domain metadata, and registered tool names. It can create and start a run, but execution halts at a LangGraph interrupt. Operator credentials cannot approve. A separate human-approver credential maps to a server-configured identity and claims the exact pending action digest. The service signs the resolved argv, rule, domain, and workspace; the runner consumes that capability once. A different command, expired token, stale request, replay, or service restart cannot reuse it.

## Persistence

SQLite stores LangGraph checkpoints, run records, workspace trust attestations, approval replay state, and trace handles. A process-lifetime OS file lease permits one service writer per state directory, preventing recovery from racing an active start, approval, or decline. A process can close after an interrupt and reconstruct the service against the same state directory before approval arrives. Startup reconciliation restores terminal checkpoints and genuine approval interrupts. A nonterminal checkpoint without an interrupt is marked failed with an unknown outcome and is never replayed automatically.

Run records and traces include the planner fingerprint that created the plan. Configuration changes before a planned run starts are refused. Once a plan reaches an approval interrupt, resume uses the checkpointed plan and does not call the model again.

The browser reads only bounded projections. Trace paths and server state paths are never returned. Trace previews are loaded from the server-derived run path, allowlist fields, reapply redaction, cap metadata fields at 4 KiB, cap each stream at 16 KiB, and enforce a shared output budget. Concurrent start requests use a SQLite compare-and-swap transition through `starting`, so one run can invoke the workflow only once.

## Workbench Interaction

The task sidebar is a durable run index rather than a transcript database. A goal creates a typed run; the timeline then renders the planner rationale and tool sequence. Advisory mode stops at a completed plan. Assisted mode advances until the next command is resolved and pauses with its exact argv, rule, workspace, and reason visible to the human. Approval or decline requires a separate in-memory approver credential.

The workspace inspector is a read model over profiling, trust, runtime provenance, and the registered tool catalog. The UI deliberately has no raw-command input and makes no source-editing claim because v0.1 has no patch protocol or editor backend.

## Domain Behavior

- Django: system checks, tests, migration dry-run, and migration plan use the target `.venv` when present.
- PyTorch: smoke training and evaluation use the target `.venv` and remain budgeted approval actions.
- Android: the absolute project wrapper runs with `--offline`; all Gradle configuration is treated as workspace-code execution.
- Android NDK: CMake builds require a discovered system CMake and approval.
- iOS: tools are explicitly unavailable off macOS or without `xcodebuild`; Xcode actions require approval.

## Commit Review Flow

Commit review is a typed read-and-reason path rather than a raw-command workflow. The operator supplies a 7-64 character hexadecimal object ID. Inverse-Agent resolves it to a full commit SHA with fixed-location system Git while replacement objects and lazy fetching are disabled, enumerates changed tree entries, reads bounded blobs with `git cat-file`, and builds its own unified diff. On Windows, trusted installation roots come from the machine registry rather than process environment variables, and linked or junction-backed executable paths are refused. The strict v0.1 reader accepts only an in-workspace `.git` directory and rejects worktree indirection, links, common directories, and alternate object stores. Repository diff configuration, hooks, text conversion, and external diff programs are never invoked.

Likely reviewer-directed instructions in pure comments and commit metadata are replaced while surrounding line structure is preserved. Inline comment handling retains the executable prefix but conservatively marks the review incomplete; an instruction-like source line without a recognizable comment boundary is omitted and also forces `INCOMPLETE`. Nonmatching source is never Unicode-normalized or rewritten, and strict decode failures cannot produce a complete verdict. Filenames are not sent to the model; each file uses an opaque review ID and a constrained extension hint. For Python changes, bounded AST extraction follows imported symbols, namespace-package submodules, full-module imports, and every ancestor package initializer from both repository-root and `src/` layouts. A partially resolved symbol request retains the available definitions but marks the context incomplete. Missing or unsupported in-repository context is never silently omitted.

Two independent general model passes inspect the redacted diff. PyTorch adds one focused evaluation-mode scout plus independent normalization-leakage and evaluation-mode confirmation scouts, isolating those contracts from data, gradient, and state-restoration analysis at the cost of three extra model calls for that domain. Deterministic signals consume the bounded candidate budget first; remaining slots are distributed round-robin across all active scouts, so model noise cannot evict narrowly grounded local candidates. If any unique candidate cannot fit, the report is `INCOMPLETE` even when the retained candidates are rejected, preventing budget pressure from producing a false `PASS`. Scout and deterministic findings are untrusted hypotheses; a final pass verifies every retained candidate against the same diff, and rejected candidates are not presented. Its `accepted` decision means evidence-supported rather than display-unique, allowing Inverse-Agent to retain the accepted scout-origin set before presentation deduplication. Evidence is matched exactly first; a single declared-side diff marker is tolerated only as a fallback, preserving real `++` and `--` source lines. Model findings outside opaque file IDs or changed hunk lines are discarded. High-signal Android, iOS, C++, Django, and PyTorch checks produce mechanically anchored candidates for narrowly provable defect patterns. Android bridge registrations must target the navigated WebView, C++ lifetime scopes include parameterless lambdas, and iOS callback plus Django taint analysis remain inside lexical function/closure boundaries. PyTorch contradiction checks use ordered assignments and kill overwritten provenance before rejecting a model claim. Detector-supplied root lines and multi-label defect categories consolidate only true accepted restatements, so independent nearby findings remain separate. The benchmark can require every expected defect to match an adjudicated scout-origin finding and therefore cannot pass on deterministic rules alone.

The commit-review CLI is deliberately separate from durable command runs in v0.1. It reads source and sends the bounded review prompt to the operator-configured model endpoint, but it neither executes workspace code nor grants approval authority.

## OSS Composition

LangGraph provides durable orchestration and interrupts. MCP provides the tool protocol. The design borrows constrained action and patch-review patterns from OpenHands, SWE-agent, and Aider without maintaining a hard fork.

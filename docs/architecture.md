# Architecture

Inverse-Agent separates decisions, execution, and product state so no model-controlled component can grant itself authority.

## Components

- `planner.py`: provider-neutral planning. Models select registered tool IDs under a strict action budget; raw commands are rejected.
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

## OSS Composition

LangGraph provides durable orchestration and interrupts. MCP provides the tool protocol. The design borrows constrained action and patch-review patterns from OpenHands, SWE-agent, and Aider without maintaining a hard fork.

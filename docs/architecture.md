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
- `control_plane.py`: authenticated HTTP API for the human-facing product surface.
- `mcp_server.py`: executable MCP tools. Approval and trust mutation are intentionally absent.
- `eval.py`: portable trace serialization.

## Trust Flow

The model sees goals, domain metadata, and registered tool names. It can create and start a run, but execution halts at a LangGraph interrupt. Operator credentials cannot approve. A separate human-approver credential maps to a server-configured identity and claims the exact pending action digest. The service signs the resolved argv, rule, domain, and workspace; the runner consumes that capability once. A different command, expired token, stale request, replay, or service restart cannot reuse it.

## Persistence

SQLite stores LangGraph checkpoints, run records, workspace trust attestations, approval replay state, and trace handles. A process can close after an interrupt and reconstruct the service against the same state directory before approval arrives. Startup reconciliation treats the LangGraph checkpoint as authoritative when a crash leaves the run projection in an intermediate state.

Run records and traces include the planner fingerprint that created the plan. Configuration changes before a planned run starts are refused. Once a plan reaches an approval interrupt, resume uses the checkpointed plan and does not call the model again.

## Domain Behavior

- Django: system checks, tests, migration dry-run, and migration plan use the target `.venv` when present.
- PyTorch: smoke training and evaluation use the target `.venv` and remain budgeted approval actions.
- Android: the absolute project wrapper runs with `--offline`; all Gradle configuration is treated as workspace-code execution.
- Android NDK: CMake builds require a discovered system CMake and approval.
- iOS: tools are explicitly unavailable off macOS or without `xcodebuild`; Xcode actions require approval.

## OSS Composition

LangGraph provides durable orchestration and interrupts. MCP provides the tool protocol. The design borrows constrained action and patch-review patterns from OpenHands, SWE-agent, and Aider without maintaining a hard fork.

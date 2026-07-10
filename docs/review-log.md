# Review Log

## 2026-07-10 - Codex Independent Review

- P0: Git `--no-index` and output flags escaped workspace confinement. Fixed with exact argv and hardened Git shapes.
- P0: Workflows fabricated `approved=True`. Fixed by removing boolean approvals and adding signed, expiring, action-bound capabilities with replay protection.
- P1: Workspace code ran under commands labeled safe-read. Fixed by approval-gating every project-code tool and requiring explicit workspace trust.
- P2: Android wrapper resolution failed on POSIX. Fixed with absolute platform-aware wrapper discovery.
- P2: Private-key redaction leaked the body. Fixed by redacting complete PEM blocks and adding adversarial token cases.
- P2: The lockfile referenced the parent commit and CI ignored it. Replaced with a universal `uv.lock` consumed by a three-OS CI matrix.

## 2026-07-10 - Claude Fable 5 Max-Effort Design Gate

Review target: public commit `e565bb8`. Exact model usage confirmed `claude-fable-5`; the review ran for 617 seconds. Verdict: BLOCK pending three P0 closures.

Accepted requirements:

- Resolve and execute only trusted absolute binaries.
- Validate flags and arguments, not command prefixes.
- Land signed approvals and durable LangGraph pause/resume together.
- Refuse untrusted workspaces unless an external OS sandbox is used.
- Fail closed when API secrets or target toolchains are missing.
- Keep approval and trust mutation out of MCP.
- Add adversarial runner, approval, restart, redaction, multi-domain, type, coverage, and multi-OS gates.

Implementation status: requirements incorporated. A post-implementation Fable review is required before v1 completion.

## 2026-07-10 - Independent Codex Milestone Gate

Verdict: BLOCK with no P0 findings.

- P1: one bearer credential could both operate and approve. Fixed with distinct operator and approver credentials; approver identity is server-derived.
- P2: concurrent approvals could race. Fixed with a SQLite `BEGIN IMMEDIATE` compare-and-swap claim and a client-supplied expected action digest, preventing stale approval from advancing the next action.
- P2: a crash could leave the run projection behind the LangGraph checkpoint. Fixed with startup reconciliation from the durable graph state; commands retain at-least-once semantics after a mid-execution process crash.
- P2: state could default inside a model-writable workspace. Fixed by refusing state directories under the workspace root and defaulting to the operating system's per-user state directory.

Verification after fixes: 52 passed, 1 platform skip, 85% coverage, strict Mypy clean, Ruff clean, and universal lock check clean.

## 2026-07-10 - Independent Codex Final Staged Gate

Verdict: PASS. No P0, P1, P2, or P3 findings. The reviewer independently reran 52 tests, coverage, Ruff, Mypy, and `git diff --cached --check`, including the final removal of unauthenticated FastAPI documentation routes.

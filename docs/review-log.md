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

## 2026-07-10 - Claude Fable 5 Post-Implementation Gate

Review target: public commit `705d75d`. Runtime `modelUsage` confirmed exact model `claude-fable-5` at max effort. Verdict: PASS with no P0, P1, or P2 findings.

The review reported bounded P3 hardening opportunities. The subsequent batch added bounded timeout cleanup, redaction across output truncation, byte-safe token comparison, immutable CI action pins, explicit checkpoint recovery failures, typed approval control flow, bounded PyTorch profiling, CLI state-layout coverage, and missing control-plane authorization tests.

## 2026-07-10 - Independent Codex Hardening Gate

Initial verdict: BLOCK on three incomplete P3 remediations: unbounded `taskkill`, character-counted PyTorch sniffing, and incomplete containment of malformed checkpoints. All three were fixed and covered by targeted regressions.

Final verdict: PASS with no P0, P1, P2, or P3 findings. Verification after fixes: 66 passed, 1 platform skip, 86.45% coverage, Ruff clean, strict Mypy clean for Windows/Linux/macOS, universal lock check clean, and `git diff --check` clean.

## 2026-07-10 - Local Model Integration Gates

Claude CLI JSON smoke testing confirmed exact `claude-fable-5` model usage at max effort. The first implementation review returned BLOCK on an unredacted model-backed evaluation error path. The independent Codex reviewer found the same gap plus planner provenance, CLI exit-status, persisted-goal redaction, and URL control-character issues. All findings were fixed with regressions.

The final independent Codex review returned PASS with no P0, P1, P2, or P3 findings. The final Claude Fable 5 max-effort review returned PASS with no blocking findings. Its concrete P4 `Content-Length` diagnostic issue was also fixed and covered; the remaining slow-header availability note is documented in `docs/safety.md`.

Final verification: Ruff clean, strict Mypy clean for Windows/Linux/macOS, 103 tests passed, 1 platform skip, 88.54% coverage, and `git diff --check` clean.

LM Studio 0.4.19 loaded OpenAI GPT-OSS-20B MXFP4 with full GPU offload and a 16K context on an RTX 3090 Ti. Inverse-Agent's real `model-check` selected only `generic.inspect`, completed model inference in 0.797 seconds, and confirmed the loopback endpoint with no API key or fallback. The loaded setup used approximately 15.1 GiB of 24 GiB GPU memory.

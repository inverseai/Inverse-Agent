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
- P2: a crash could leave the run projection behind the LangGraph checkpoint. Startup reconciliation restores terminal states and approval interrupts; later workbench hardening made nonterminal, non-interrupt checkpoints fail with an unknown outcome instead of replaying a command.
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

## 2026-07-11 - Engineering Workbench Gates

The exact Claude Fable 5 max-effort implementation review ran for 1,022 seconds and `modelUsage` confirmed `claude-fable-5`. It returned BLOCK on three product defects: non-interrupt checkpoints could become permanent `running` tasks after a crash, persisted `planned` tasks had no UI resume action, and stale long-running responses could replace a newer task selection.

The remediation added conservative unknown-outcome recovery, an idempotent Start run control, navigation-epoch response fencing, expected-status CAS writes, and a cross-platform process-lifetime state-directory lease. Follow-up independent Codex reviews reproduced and closed two deeper interleavings: a stale worker overwriting recovery and recovery racing active approval/decline. The final independent Codex verdict was PASS with no P0-P3 findings.

Live GPT-OSS browser QA covered advisory planning, trust, exact-command approval, decline, approver re-entry after reload, persisted-plan resume, stale-response selection fencing, and 375x812 layout without horizontal overflow or console warnings. Final local verification: 112 passed, 1 platform skip, 87.93% coverage, Ruff clean, strict Mypy clean for Windows/Linux/macOS, lock check clean, PowerShell 5.1 launcher validation, and verified UI assets in both wheel and source archive.

The execution platform denied a final review of the private worktree under its external-disclosure policy. After the code was pushed, the user directed Claude to review the public GitHub commit instead. The exact Fable 5 max-effort public re-review ran for 941 seconds against commit `00f0ea0`; `modelUsage` confirmed `claude-fable-5`, and the verdict was PASS with no P0-P3 findings.

The reviewer recorded non-blocking P4 opportunities around defensive recovery redaction, profile-fetch rendering, synchronous trace loading in the UI, cross-tab waiting-state polling, explicit service cleanup, malformed trace-duration handling, automated browser fencing tests, and an empty assisted-plan guard. These remain hardening and product-polish work; this milestone passes its required review gates but is not declared v1.

## 2026-07-11 - Generic Git Workspace Gate

The workbench initially classified Git monorepo roots without a root-level domain marker as generic but exposed no executable tools. Commit `2f75cb1` added approval-gated `generic.status` and `generic.tracked_files` tools, exact shared Git argv policy constants, optional-lock suppression, deterministic generic planning, and end-to-end durable approval coverage.

The independent Codex review initially identified optional Git index writes, broad system-Git provenance, and incomplete disclosure of repository-configured clean/filter helper execution. The implementation added `--no-optional-locks`, `GIT_OPTIONAL_LOCKS=0`, mandatory per-command approval, exact executable/digest binding, and an explicit filter-helper warning. The final Codex verdict was PASS with no P0-P3 findings.

The public Claude Fable 5 max-effort review ran for 784 seconds against commit `2f75cb1`; runtime `modelUsage` included exact model `claude-fable-5`, and the verdict was PASS. It found no P0 or P1 issues and treated the disclosed, trust- and approval-gated Git filter-helper risk as a non-blocking P2 residual. Its P3 observations covered Windows script-wrapper discovery breadth, redundant optional-lock hardening, and minor advisory/filter test gaps.

Final local verification: 114 passed, 1 platform skip, 87.97% coverage, Ruff clean, strict Mypy clean for Windows/Linux/macOS, universal lock check clean, and `git diff --check` clean.

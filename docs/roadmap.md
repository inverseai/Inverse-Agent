# Roadmap: Frontier-Parity Agent for the Priority Stacks

This plan takes Inverse-Agent from the v0.1 approval-gated verification runner to a fully fledged agent with Claude Code / Codex-class capability for four priority stacks, in this order of importance:

1. Native Android and Native iOS
2. C/C++ (general, beyond Android NDK)
3. Django + React/HTML full-stack
4. ML research (PyTorch)

All other languages and platforms are explicitly deferred until these four reach parity.

## Parity, Defined Honestly

The milestones below deliver **workflow parity**: an iterative plan→act→observe→replan loop, unattended file reading and search, patch-based source editing, parameterized builds/tests/lints, streaming output, cancellation, and budgets — all inside the existing security model (typed registered tools, signed single-use human approvals, redaction, loopback-first inference).

**Intelligence parity is a model-selection variable, not a tooling deliverable.** GPT-OSS-20B cannot match frontier models on multi-file reasoning and patch synthesis regardless of tool quality. The plan therefore:

- Selects the local serving context by **measured calibration** across 16K, 24K, 32K, and 48K (no crash/OOM, ≥1.5 GiB VRAM headroom, 20/20 strict-schema probes) rather than asserting a number — 16K is an LM Studio configuration choice, not a model limit. The working hypothesis (unmeasured — no observe loop exists yet to measure) is that 16K starves an iterative loop within a few turns, so a larger passing point is expected but not presumed. All prompt and observation budgets derive from the measured capacity.
- Adds native Anthropic and OpenAI provider clients behind the existing `JsonModelClient` protocol, reusing the remote-endpoint dual opt-in and egress policy. Cloud use is per-workspace, default-off, and visible in every trace — sending file contents to a cloud provider is a posture change from today's redacted-goal-only egress and is treated as such.
- Adds per-step model routing (reviving the dead `model_policy` config): local model for tool selection, summarization, and compaction; a frontier model, when opted in, for patch synthesis and multi-file reasoning.
- Requires seeded-bug eval fixtures that report pass rates **per model**, so the local-versus-cloud gap is a measured number. Every acceptance gate that includes "the agent patches the source" is labeled model-dependent with a defined fixture difficulty.

## Security Invariants (constant across every milestone)

- Typed registered tools only; no raw shell, ever.
- Source disclosure to a model, source modification, and cloud inference are each **new trust boundaries** requiring their own explicit consent mechanism (attestation scopes) — they are never treated as capabilities already covered by the existing execution-consent model.
- Single-use signed approvals bind the fully resolved argv — and, for patches, the content hashes of every touched file (fail closed on base-hash mismatch).
- Unattended (no-approval) tools must take no workspace-controlled configuration that alters execution. Anything that reads workspace config able to load plugins (clang-tidy, eslint, Gradle) stays approval-gated.
- Toolchain discovery never executes anything as a profiling side effect; discovery probes are path/registry reads, and probe executions (e.g., `vswhere`) run as registered rules.
- Network access is a separate, explicitly gated rule family; offline remains the default everywhere.
- The read tier enforces a sensitive-file deny/redact policy (`.env`, keystores, `google-services.json`, key material), with each domain pack contributing deny patterns.
- Every milestone keeps the existing review culture: independent Codex review plus Claude max-effort review before the milestone is recorded as complete in `docs/review-log.md`.

## Milestones

### M0 — Agent Core (delivered as v0.2a, v0.2b, v0.3, v0.4)

Everything else depends on this. M0 as originally drafted was three-to-four milestones behind one compound exit gate; after the joint review with the Codex v0.2 plan it is decomposed into four smaller, per-commit-reviewed milestones. The executable spec for the first two is [milestone-v0.2.md](milestone-v0.2.md). Sizing figures anywhere in this document are planning guidance, not acceptance criteria — milestones exit on their gates, not on their estimates.

- **v0.2a — Durable substrate**: versioned SQLite migrations (replacing ad hoc column patching), scoped workspace attestations in a new `workspace_attestations` table (`source_read` / `code_execution`, existing records migrated to `code_execution` only, legacy table emptied so a scope-unaware v0.1 binary fails closed) with revocation enforced at dequeue and at every source-bearing operation, `RunKind` (verification / investigation), `QUEUED` / `CANCELLED` / `INCOMPLETE` / `CANCEL_REQUESTED` statuses with CAS extensions and a queued-run restart requeue path, a single-worker queue owning starts *and* resumes with 202 semantics and node-boundary cancellation, a durable run-event log with `GET /runs/{id}/events?after=` (replacing the earlier SSE ambition), tool-registry consolidation under a static-constants-only rule-generation invariant, agent-triggered or automatic `review-commit` brought under `source_read` (a direct operator CLI invocation remains explicit per-run consent — identical rule in milestone-v0.2.md), and an MCP-specific safe projection applied to all MCP run methods (fixing the `trace_path` leak, which affects `create_run`/`start_run` too, and excluding approval challenges and absolute paths).
- **v0.2b — Read-only investigation capability**: the Windows-hardened unattended read tier (component-wise handle-based no-follow, ADS/hard-link/8.3/device-name handling, all reparse points refused in v0.2, sensitive-file deny policy, line-preserving redaction and instruction neutralization — redacted/neutralized spans are uncitable, and run-level `INCOMPLETE` has the same enumerated causes as milestone-v0.2.md), the iterative decide→dispatch→observe loop as a side-by-side graph with a binding v0.3 retirement clause, token-denominated observation budgets with command-output distillation, citation validation against the durable event log, and the model-reliability track this milestone's gate depends on: measured context calibration (16K/24K/32K/48K, budgets derived from what passes), history compaction with a runtime-owned observation catalog, bounded retry with no JSON-repair layer, and the calibrated token estimator. Telemetry (token/cost accounting, structured logging) lands here and is load-bearing, not polish.
- **v0.3 — Mutation and cloud** (strategic grouping only — a separate implementation specification, at the same rigor as milestone-v0.2.md, is required before execution): the patch protocol (structured edits, ≤8 files/≤128 KB, approval digest binding `sha256(patch)` + per-file `sha256(pre-image)`, atomic apply with rollback, unified-diff review UI); per-rule argument validators with the generic pytest `-k`/node-id rule as first consumer and fuzz suites in the same PR; the typed git rule family (`commit -m` with validated message, branch create/switch, diff — the patch protocol without commit capability strands edits in the worktree); the mid-run steering channel; per-project environment-variable requests that must intersect server-owned policy and operator approval — workspace configuration can never authorize host variables; and native Anthropic + OpenAI clients with per-step model routing, riding on a new `model_egress` attestation scope so cloud consent is per-workspace and default-off by mechanism, not by promise. The investigation graph absorbs approval interrupts and the v0.1 plan-once graph retires in this same milestone.
- **v0.4 — Fix loop**: the read failure → grep → read → patch → filtered re-run gate, honestly labeled model-dependent with defined fixture difficulty (the earlier draft gated M0 on the local model completing this loop while disclaiming local-model patch synthesis two sections earlier — that contradiction is resolved by gating v0.2b on investigation and v0.4 on the fix loop per model tier); rule-supplied static env and per-rule timeout/output limits (Android, C++, and PyTorch all need it); working-tree review-to-fix integration groundwork; OS-level filesystem/network/process isolation groundwork (autonomy promotion itself stays deferred behind full sandboxing, per the deferral section below).

Exit gates: v0.2a/v0.2b gates as specified in [milestone-v0.2.md](milestone-v0.2.md) (seven cases × three goal variants, at least 2/3 per case and ≥19/21 aggregate — which guarantees at least one complete 7/7 suite — with zero unsupported citations and retries reported independently); v0.3 patch TOCTOU and validator injection gates; v0.4 fix-loop gate per model tier, with the same fixture run under local, Anthropic, and OpenAI clients as the calibration comparison.

### M1 — Android Inner Loop (priority #1)

- Static workspace profiling upgrade: Kotlin-DSL detection (`build.gradle.kts` / `settings.gradle.kts`), SDK/NDK/JDK discovery (`local.properties`, `ANDROID_SDK_ROOT`), module topology from settings files, version catalogs, manifest/resource indexing — all without executing Gradle.
- Parameterized Gradle rule family: module- and variant-scoped tasks (`:feature:login:testDebugUnitTest`, `--tests com.example.FooTest`), with cmd.exe-metacharacter (BatBadBut-class) charset validators for `gradlew.bat`, landing **in the same PR as the injection fuzz suite**.
- JUnit XML and Gradle log parsers producing typed observations (failure class/method/message mapped to file:line; configuration vs compile vs test vs offline-cache-miss classification); report files preferred over the 2 MB-capped stdout.
- Run ergonomics consumption: per-rule long-build timeouts, Gradle daemon policy with a registered `gradlew --stop` escape hatch.

Exit gates: single-test observe→patch→re-run loop on a Kotlin-DSL multi-module project on Windows; injection-resistance suite green.

### M2 — Android Completion + iOS Windows-Buildable Slices (parallel)

- Android: lint stack (Android Lint per-variant, ktlint, detekt, typed findings); lockfile-aware network-gated dependency resolution (offline stays default; online rule offered only on detected cache miss, lockfile diff shown to the approver); NDK/CMake native pack (shared clang diagnostics parser); device tier v1 — read-only `adb devices`/`logcat -d`(filtered, redacted)/`getprop`, each requiring explicit approval despite being read-only: device logs are a distinct disclosure surface (they can contain other applications' data and personal information), so read-only does not mean unattended here.
- iOS (all pure-Python, fixture-testable on Windows — ~60% of the pack): parameterized xcodebuild rule catalog and validators (`-scheme`/`-destination`/`-only-testing`), zero-execution project discovery, xcresult/log parsers, `RunnerBackend` seam so host-unavailable tools degrade gracefully.
- **Hardware action item, due now:** procure a macOS host (physical Mac or macOS CI) during M2–M3; it gates M4.

Exit gates: Android lint/dependency/NDK/device-v1 gates; iOS policy negative-test suite green on Windows.

### M3 — C/C++ Pack (priority #2)

- `Domain.CPP` + `CppAdapter`, fixing the NDK adapter's false-positive detection of plain C++ repos (single owner for this fix — it appears in both packs).
- Toolchain discovery: vswhere/MSVC, clang, MinGW, Ninja — with `vswhere` executed as a registered rule from registry-derived install roots, never as an implicit profiling side effect.
- Parameterized `cmake` configure (approval-gated — configure executes project code) and build; `CMakePresets.json` with digest-pinned presets; `ctest -R/-L/-E` filters with failure parsing.
- `compile_commands.json` as the read-tier code-navigation backbone; clang-format check and clang-tidy (tidy approval-gated: workspace config loads plugins); native diagnostics parser shared with the NDK pack; commit-review wiring for `Domain.CPP` (the C++ lifetime/lambda checks already exist there).
- Deferred to M6: sanitizer builds (verification depth, not loop-enabling); MSBuild/.sln stretch.

### M4 — iOS Execution via Mac-as-Host (priority #1b, gated on hardware)

- macOS toolchain discovery, simulator lifecycle (`xcrun simctl` — `list --json` unattended; boot/shutdown/install gated), SwiftLint/swift-format, SPM/CocoaPods network-gated resolution, validation pass on macOS CI plus one manual pass on real hardware.
- Sequenced after M3 **only** because of the host constraint; if a Mac materializes earlier, M4 swaps ahead of M3 with no dependency breakage (all code prerequisites land in M2). The supported mode is running Inverse-Agent on the Mac; the Windows-driven remote runner daemon is deferred to M7.

### M5 — Django + React Full-Stack Pack (priority #3)

- Django to frontier depth: single-label `manage.py test app.tests.TestCase.test_x --keepdb`, pytest-django detection and node-id rules, `makemigrations` as a patch-protocol mutation (migration file arrives as a reviewable DIFF artifact), test-database strategy with engine/name surfaced in the approval challenge (fail closed on inconclusive settings parse).
- New `FrontendAdapter`: package-manager detection by lockfile (npm/pnpm/yarn as distinct aliases — `.cmd` shims must not cross-match); network-gated `--frozen-lockfile --ignore-scripts` install with the lockfile hash in the approval card and a postinstall-canary regression test; refusal of registry-redirecting `.npmrc` before any approval; tsc/eslint/prettier (approval-gated — workspace configs load plugins; validators pin script paths under `node_modules`); vitest/jest single-test filters with JSON reporters distilled to <2K tokens; vite/webpack production build.
- `pip`/`uv` install gating designed alongside npm's (same network-gated, lockfile-pinned pattern) — Django and ML both need it.
- Multi-domain run plumbing: `RunSpec.domains` set, per-tool domain derivation, policy built once per run; **monorepo subdirectory execution** (the runner currently hard-requires cwd == workspace root — per-rule validated sub-cwd support is required for `backend/` + `frontend/` layouts, which are exactly this pack's target).
- Full-stack loop recipe: changed-path → related-test mapping feeding the replan loop (component → its vitest file; app module → its Django label).
- Dev-server items excluded here (moved to M6).

### M6 — Long-Running / Background-Tier Consumers

Batched because they share one dependency — leased background processes with readiness probes and guaranteed teardown. Note this is *not* the v0.2a run queue (which serializes run executions inside the service); the long-running-process supervisor is its own deliverable, built here:

- Django `runserver` smoke and vite dev-server preview, including a new first-class `http.probe` tool (loopback-pinned host, response-size caps, its own approval classification) instead of stretching one approval over start+probe+teardown.
- Android device tier v2: `installDebug`, filtered `connectedDebugAndroidTest`, uninstall — one approval each.
- C++ sanitizer builds (ASan/UBSan) as first-class verification; MSBuild stretch if demand exists.
- Windows hygiene, explicitly owned here: orphan-process sweep on next run, long-path (MAX_PATH) enablement, antivirus-interference diagnostics.

### M7 — ML Research Pack (priority #4) + Remote macOS Runner

ML pack (reuses the most machinery, lands last per stated priority):

- `pytorch.run_experiment`: validated experiment grammar (script, typed flag allowlist, Hydra-style overrides, seeds, device selection); configs snapshotted into the run directory and argv pointed at the immutable snapshot so the approval digest covers what actually executes.
- GPU telemetry as unattended read (`nvidia-smi` via fixed-location discovery, never PATH), including VRAM-contention preflight against the LM Studio server sharing the same GPU.
- Observation tools: in-process TensorBoard event parsing (no TensorFlow dependency), stdout metric extraction, checkpoint inspection with a **hard no-unpickle invariant** (zip member listing + static pickletools opcode walk; safetensors parsed natively) backed by a no-unpickle regression test.
- Budgeted background training: `RunSpec.budget` gets PyTorch semantics (wall-clock, GPU-hours, checkpoint bytes); `stop_conditions` grammar (`metric:val_loss:nan`, stall patience); supervisor-owned termination — the approval token authorizes process start, budgets govern lifetime.
- Runner-written experiment ledger with list/compare/report tools (making `pytorch.report` real); dataset-root confinement with offline-by-default child env (`WANDB_MODE=offline`, `HF_HUB_OFFLINE=1`); notebook stance: unattended read, patch-protocol edits, never execute.
- Pre-merge ML contract gate: the commit-review PyTorch contracts (eval-mode, normalization leakage, gradient control, state restoration) extracted into a shared module and run deterministically against every agent-authored patch in the fix loop.

Remote macOS runner daemon (the Windows-driven iOS parity target): honestly sized at 4–6 weeks, not 2–3 — it is the project's first network-listening execution service and needs host-bound approval digests, a runner-side replay store, workspace sync, runner-side redaction, streaming/cancel transport, and its own threat-model review. Mac-as-host (M4) already delivers a working iOS story; the daemon buys convenience, not capability, which is why it is last.

## Deferred Beyond This Plan

- All other languages/platforms (Flutter, Node backends, Go, Rust, …) — revisit after M7, once all four priority stacks have reached parity.
- Emulator/AVD lifecycle, `adb shell` beyond `getprop`, multi-device orchestration.
- Promotion of `BOUNDED_AUTO`/`WORKFLOW_AUTO` autonomy (rule-scoped standing approvals): requires OS-level sandboxing and its own security design; the iterative loop multiplies approval interrupts, so this pressure will grow — it must be answered with containment, not by weakening single-use approvals.
- MCP **client** support (consuming external tool servers wrapped in the same typed-rule + approval machinery) — natural post-M5 addition.
- Remote/team access, release/publication pipeline cadence.

## Standing Risks To Hold The Line On

- Parameterized argv is the first non-frozen input surface in the runner; every validator ships with a fuzz/negative suite in the same PR, and schedule pressure never shortens security work.
- npm's real trust boundary is the human's network-gated approval over a specific lockfile hash — the first vitest/eslint run executes `node_modules` code even with lifecycle scripts disabled; the UI must present it that way.
- Approval binds argv, not transitive build inputs (existing TOCTOU residual); config snapshotting (ML) and patch base-hashes (core) close the instances this plan creates.
- Local-model loops can thrash (action repetition, ignored evidence); budgets and the repeated-action guard bound the damage, and the docs/UI must set expectations for local-only mode explicitly.

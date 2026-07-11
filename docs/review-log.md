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

## 2026-07-11 - Multi-Domain Commit Review Benchmark

Inverse-Agent now materializes and scores isolated Android, iOS, C/C++, Django, and PyTorch commit-review tasks plus the public Inverse-Agent docs-only control commit `482fa05`. The reviewer uses bounded immutable Git extraction, opaque file IDs, strict changed-line evidence, two general model scouts, a focused PyTorch evaluation-mode scout, local schema enforcement, a model adjudicator, and narrow deterministic candidates that are presented only after adjudicator acceptance.

Successive independent Codex audits found and drove closure of executable-discovery trust, source-instruction handling, truncation completeness, benchmark-schema validation, goal/secret sanitization, mode and byte-only changes, candidate-budget fairness, Python import context, Windows fixture Git-state handling, and non-UTF-8 decode loss. Regression coverage now includes ancestor package initializers, case-insensitive Windows Git-state aliases, and model restatement collapse.

The local `inverse-gpt-oss-20b` gate initially passed the full six-case suite twice consecutively with zero missing and zero unmatched findings in both runs. An independent Codex xhigh audit then blocked the milestone on a static-only benchmark escape, multiline redaction that could hide later code, missing-object gitlink dereferencing, broad same-file semantic collapse, relative package-attribute context, and mixed content/EOL completeness.

The remediation added adjudicated model-origin provenance requirements, nonempty structured finding bodies, root-local restatement collapse, fail-closed source/context redaction, raw-mode gitlink handling, package-initializer context for relative attributes, and independent line-ending metadata. All six findings received targeted regressions. A second independent audit found three deeper edge cases: same-kind mixed-EOL redistribution, scout reservations displacing a full static-finding budget, and omission of the containing initializer for relative submodule imports. The implementation now aligns logical lines before comparing exact terminators, gives static findings absolute candidate priority, and requests the containing package initializer for every relative `from` import. Those findings also have targeted regressions.

After the second remediation, `inverse-gpt-oss-20b` again passed the six-case suite twice consecutively with model support in every seeded domain and zero missing or unmatched findings. Full verification passed with 226 tests, 3 platform skips, 86.74% coverage, Ruff and changed-file formatting clean, strict Mypy clean for Windows/Linux/macOS, lock and diff checks clean, and verified wheel/source-distribution assets.

A third independent audit found four P2 gaps adjacent to the prior regressions: inserted mixed-EOL lines, context omitted for imports targeting another changed file, nearby independent defects collapsed by proximity, and C++ lifetime matching across function boundaries. The implementation now combines global and aligned terminator comparison, extracts imported symbols even from changed targets, uses detector-supplied multi-label root lines instead of proximity, and scopes C++ lifetime proof to one function. A PyTorch AST contradiction filter additionally rejects claims that stacked Subsets from normalized input are raw. Targeted tests cover all five behaviors.

The post-third-review live suite passed `6/6` twice consecutively with required model provenance and zero missing or unmatched findings. Full verification passed with 233 tests, 3 platform skips, 86.83% coverage, Ruff and changed-file formatting clean, strict Mypy clean for Windows/Linux/macOS, lock and diff checks clean, and verified wheel/source-distribution assets.

A fourth independent audit found seven remaining gaps: PyTorch contradiction proof could cross function boundaries, follow variable names instead of complete aliases, and double-count rejected candidates; mixed-EOL insertions or deletions could evade completeness checks when they reused an existing terminator kind; nested C++ lambdas could share one enclosing detector scope; multiple Android bridges shared one registration/root set; and partial imported-symbol extraction could appear complete. The remediation now requires function-local assignment and return provenance for PyTorch contradiction filtering, counts each filtered candidate once, detects mixed-EOL creation and structural edits, partitions nested C++ scopes, emits per-bridge roots, and retains partial Python context only with `INCOMPLETE`.

The fifth independent audit found nine further issues: diff headers collided with `++`/`--` source, PyTorch contradiction proof ignored overwrites, aggregate provenance let static rules supply model misses, Android bridges crossed WebView receivers, parameterless C++ lambdas escaped scope partitioning, namespace-package imports lost submodule context, Django taint and iOS callback detection crossed functions, and CR-only line counts were wrong. The remediation uses hunk state rather than prefix-shaped headers, ordered kill-aware PyTorch provenance, per-expectation accepted-scout scoring, receiver and lexical scopes, namespace submodule candidates, and one shared byte-line parser. Side-consistent diff markers in model evidence are accepted only after exact matching fails.

The strengthened suite adds gradient control as a fourth independent PyTorch validity contract and a focused state-restoration scout pass. Two consecutive post-remediation runs passed `6/6` with every seeded expectation backed by adjudicated model provenance. The first used only Android's explicitly allowed redundant-navigation alternative; the second had zero missing and zero unmatched findings. Full verification passed with 254 tests, 3 platform skips, 87.31% coverage, Ruff and changed-file formatting clean, strict Mypy clean for Windows/Linux/macOS, lock and diff checks clean.

Subsequent independent audits closed object-store mutation races, insufficient benchmark grounding, heuristic assignment pruning, incomplete per-response model attribution, mixed-side evidence ambiguity, and false-positive detector edges. Follow-up remediation moved Git snapshots outside the target repository, made fixture commits byte- and metadata-deterministic, tightened Android origin guards and Swift callback boundaries, recognized unchanged or external C++ storage, and isolated ambient model configuration in tests. The scorer now performs an exact state search over the bounded 20-finding universe, and every v2 expectation grounds path, side, line, and evidence.

The post-reboot live gate exposed two final reliability gaps rather than relaxing the benchmark: scouts could cite a redaction placeholder or choose an adjacent forward-pass line, and independent PyTorch contracts competed for attention or produced duplicate restoration restatements. Redaction markers are now invalid evidence, supplied multi-line evidence prefers an explicitly named control only when that quoted line is present, a focused mode scout isolates inference behavior, and removed state-snapshot/restoration lines share one narrow root set. The unchanged six-case suite then passed `6/6` twice consecutively with zero missing and zero unmatched findings; all 18 responses in each run carried the requested endpoint-reported model ID.

A fresh independent Codex audit then found five adjacent issues: adjudicator severity corrections could erase model provenance, restoration roots could cross nearby Python functions, multiline C++ `extern` declarations looked automatic, benchmark assets were absent from distributions, and the scorer documentation still described reusable findings. The remediation uses severity-independent origin identities while retaining corrected severities, scopes restoration roots through the old-source AST, scans complete C++ declaration prefixes, documents the disjoint assignment contract, and adds a packaged `benchmark-review builtin` suite. Wheel/sdist verification now requires every benchmark domain, the specification, source-checkout fixtures, and no cache bytecode; a byte-for-byte test prevents packaged fixture drift.

The amended built-in suite passed `6/6` twice consecutively with zero missing and zero unmatched findings, and all 18 responses in each run carried the requested endpoint-reported model ID. At that checkpoint, local verification reported 294 passed, 3 platform skips, 86.09% coverage, Ruff and changed-file formatting clean, strict Mypy clean for Windows/Linux/macOS, lock and diff checks clean, and verified wheel/source-distribution workbench plus benchmark assets.

The next independent Codex xhigh audit blocked on three release behaviors: a full static-candidate budget could displace a valid model-only finding and still produce `PASS`, `review-commit` returned success when actionable findings existed, and benchmark output described endpoint-supplied model IDs as verified identity. The remediation adds explicit candidate-budget truncation metadata and fails such reviews as `INCOMPLETE`, maps `PASS`/`FINDINGS`/`INCOMPLETE` to exit codes `0`/`1`/`3`, and reports `endpoint_model_consistent` while still failing the benchmark when any successful response omits or disagrees with the requested model ID.

A follow-up reproduction found that deterministic findings themselves were clipped before the central overflow check. Static deduplication now preserves the complete bounded set until candidate merging; a 21-file regression proves that an omitted static candidate produces `INCOMPLETE`, reports 21 static signals, and does not inflate the discarded-model count. Origin-aware accounting still counts a displaced model candidate. The same independent reviewer then reproduced all overflow, exit-code, and provenance cases and returned PASS with no actionable P0-P3 findings.

After the final remediation, two fresh `inverse-gpt-oss-20b` runs passed the packaged suite `6/6` with zero missing, unmatched, or candidate-truncated findings. Each run received 20 successful responses, all 20 carrying the requested endpoint-reported model ID. Final pre-commit verification: 302 passed, 3 platform skips, 86.05% coverage, Ruff and changed-file formatting clean, strict Mypy clean for Windows/Linux/macOS, lock and diff checks clean, and verified wheel/source-distribution plus isolated installed-wheel benchmark assets. The public-commit Claude Fable 5 ultracode-effort review remains required before the milestone is complete.

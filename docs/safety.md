# Safety Policy

## Threat Model

The planner, model endpoint, and workspace content are untrusted. The approval signer, runner policy, authenticated control plane, and local state directory are trusted. Model output may select only registered tools; it cannot supply an executable, argv, approval bit, or approval token.

## Enforced Controls

- A workspace must be explicitly attested as trusted before assisted or automated execution. Advisory planning does not execute code.
- Command rules match exact normalized argv. Unknown flags, additional arguments, alternate scripts, and output paths are refused.
- The runner executes its own resolved absolute executable, never the caller's executable string.
- Workspace-local executables are allowed only for approval-gated rules.
- Each capability is signed and bound to the resolved workspace, domain, rule, and argv. Capabilities expire and are consumed once through a SQLite uniqueness constraint.
- The control plane fails to start without distinct operator and approver credentials. Human identity comes from server configuration, not request text. Only `/health` is public, and the CLI binds the server to loopback.
- The child environment is allowlisted and does not inherit Inverse-Agent credentials.
- Output capture is bounded, decoded defensively, and redacted before it enters traces.
- Timeout cleanup targets process groups on POSIX and process trees on Windows.
- Model endpoints are loopback-only by default. Remote inference requires dual operator opt-in and HTTPS. Redirects and environment proxies are disabled, responses are size-capped, and endpoint failures never trigger a deterministic fallback.
- Model API keys are environment-only configuration and remain excluded from workspace subprocess environments, traces, fingerprints, and startup summaries.

## Tool Hardening

Git commands use exact read shapes with global/system configuration, terminal prompts, pagers, and fsmonitor disabled. Gradle commands use the absolute project wrapper, run offline, and always require approval because configuration evaluates project code. Django, PyTorch, CMake, and Xcode actions likewise require approval.

## Residual Risks

Inverse-Agent does not yet provide an OS-level network namespace, filesystem virtualization, or container boundary. A trusted workspace can still perform any action available to its process identity after a human approves execution. Strong isolation requires a VM, container sandbox, dedicated build host, or CI runner.

Approval binds the action, not every transitive file imported by a build system. Operators should avoid modifying a workspace while an approval is pending. The service refuses state directories under the workspace root; production deployments should additionally protect state with OS ACLs.

Redaction is defense in depth, not a proof that arbitrary secrets can be detected. Raw logs and source remain local by default; external inference or artifact upload should use explicit egress policies.

A loopback model server is still an untrusted network peer. A compromised local process can impersonate it, return malicious plans, or observe planning prompts. Exact tool validation and human approval remain mandatory at every autonomy level, including bounded and workflow automation.

The model transport reapplies a shrinking monotonic deadline before response headers and every body chunk. Python's standard HTTP header parser may perform multiple socket reads under the remaining header timeout, so a deliberately slow opted-in remote endpoint can still extend one planning call. Keep inference loopback-local where possible and supervise long-running service processes externally.

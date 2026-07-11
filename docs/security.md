# Security

The canonical threat model, trust boundaries, and residual risks are documented in
[safety.md](safety.md).

## Model Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `INVERSE_AGENT_MODEL_NAME` | unset | OpenAI-compatible model identifier |
| `INVERSE_AGENT_MODEL_BASE_URL` | unset | API root ending in `/v1` |
| `INVERSE_AGENT_MODEL_API_KEY` | unset | Environment-only bearer credential |
| `INVERSE_AGENT_MODEL_TIMEOUT_SECONDS` | `60` | Request timeout from 1 to 600 seconds |
| `INVERSE_AGENT_MODEL_MAX_ACTIONS` | `8` | Plan action budget from 1 to 32 |
| `INVERSE_AGENT_MODEL_ALLOW_REMOTE` | `0` | First half of remote-endpoint dual opt-in |

Command-line model values override environment values. There is deliberately no API-key flag. Non-loopback endpoints also require `--model-allow-remote` and HTTPS. Configuration is frozen when `serve` or `mcp` starts; restart the process to apply changes.

Run records and traces store a non-secret planner fingerprint. A planned run refuses to start if the model configuration changes before planning. A workflow already waiting for approval resumes from its durable plan without invoking the model again.

## Browser Credentials

The operator and approver are separate bearer credentials. The operator token is stored in `sessionStorage` so it is scoped to one browser tab. The approver token exists only in JavaScript memory and is cleared by a reload. Neither credential is accepted in a URL, cookie, or request body, and requests never send both credentials together.

The workbench is same-origin and loopback-only. It does not enable CORS. Host validation rejects non-loopback hostnames. Normal application responses disable caching and supply a restrictive Content Security Policy, Trusted Types enforcement, MIME sniffing protection, frame prohibition, referrer suppression, and cross-origin isolation headers. Dynamic content is constructed with DOM text nodes; HTML injection sinks and third-party assets are absent.

`scripts/start-workbench.ps1` creates fresh operator and approver credentials for each server process. It stores only the approval signing secret as a per-user environment variable so a durable pending approval remains verifiable after a normal restart. Session credentials are printed to the local terminal and are not written to the repository or browser URL.

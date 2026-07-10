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

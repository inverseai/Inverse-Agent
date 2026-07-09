# Architecture

Inverse-Agent separates agent orchestration, local execution, and product control-plane concerns.

## Planes

- Local execution plane: repo files, builds, tests, devices, GPUs, and artifacts stay on the runner host.
- Control plane: users, runs, approvals, artifact handles, aggregate metrics, and dashboards.
- Inference plane: cloud no-retention, local/self-hosted, or hybrid model endpoint per `WorkspaceProfile`.

## OSS Composition

- LangGraph is used for durable workflow orchestration when available.
- MCP is the adapter protocol direction; v0.1 ships typed MCP-style tool contracts.
- OpenHands, SWE-agent, and Aider inform CodeAct, ACI, and patch workflow design without a hard fork.

## Domains

- Django is the first executable workflow.
- PyTorch support is scoped to research engineering, not autonomous scientific judgment.
- Android/NDK and iOS adapters are OS/toolchain-aware and safe by default.


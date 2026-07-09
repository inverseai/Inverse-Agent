# Safety Policy

## Command Execution

- Commands are structured argv lists, never shell strings.
- Execution is default-deny and scoped to the workspace root.
- Shell metacharacters, unchecked working directories, destructive git/file actions, release signing, and publishing are refused or approval-gated.
- Child processes receive a curated environment instead of the full host environment.
- v0.1 approval-gates commands marked as network-prone, but it does not provide OS-level network isolation. Untrusted repositories still require an external sandbox, VM, or CI isolation layer before running code.

## Data Egress

- The control plane receives metadata only by default.
- Raw source, diffs, logs, notebooks, datasets, model weights, signing material, and experiment outputs require explicit allowlisting.
- Secret-like content is redacted before artifact upload or review prompts.
- The local FastAPI app confines path-based profiling to a configured workspace root and can require an API token.

## Autonomy

Autonomy is scoped per workspace and domain. A successful Django gate does not grant Android, iOS, NDK, or PyTorch autonomy.

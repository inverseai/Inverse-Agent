"""Command-line interface for profiling and durable agent runs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.control_plane import create_app
from inverse_agent.dogfood import evaluate_workspace, save_evaluation
from inverse_agent.eval import json_default
from inverse_agent.mcp_server import create_mcp_server
from inverse_agent.models import AutonomyLevel, Domain, RunStatus
from inverse_agent.service import AgentService


def default_state_dir() -> str:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return str(Path(os.environ["LOCALAPPDATA"]) / "Inverse-Agent" / "state")
    if os.environ.get("XDG_STATE_HOME"):
        return str(Path(os.environ["XDG_STATE_HOME"]) / "inverse-agent")
    return str(Path.home() / ".local" / "state" / "inverse-agent")


def profile_command(args: argparse.Namespace) -> int:
    profile = detect_workspace(Path(args.workspace))
    print(json.dumps(asdict(profile), default=json_default, indent=2))
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    result = evaluate_workspace(Path(args.workspace))
    if args.output:
        save_evaluation(result, Path(args.output).resolve())
    print(json.dumps({**asdict(result), "passed": result.passed}, default=json_default, indent=2))
    return 0 if result.passed else 1


def start_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    service = _service(args, workspace.parent)
    try:
        created = service.create_run(
            goal=args.goal,
            workspace=workspace,
            domain=Domain(args.domain),
            autonomy_level=AutonomyLevel(args.autonomy),
        )
        record = service.start(created.run_id)
        print(json.dumps(asdict(record), default=json_default, indent=2))
        return 2 if record.status == RunStatus.WAITING_FOR_APPROVAL.value else 0
    finally:
        service.close()


def approve_command(args: argparse.Namespace) -> int:
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        record = service.approve_and_resume(
            args.run_id,
            approved_by=args.approved_by,
            expected_action_digest=args.action_digest,
        )
        print(json.dumps(asdict(record), default=json_default, indent=2))
        return 2 if record.status == RunStatus.WAITING_FOR_APPROVAL.value else 0
    finally:
        service.close()


def trust_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        result = service.trust_workspace(workspace, trusted_by=args.trusted_by)
        print(json.dumps(result, indent=2))
        return 0
    finally:
        service.close()


def serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("uvicorn is required to serve the control plane") from exc
    api_token = os.environ.get("INVERSE_AGENT_API_TOKEN", "")
    if not api_token:
        raise ValueError("INVERSE_AGENT_API_TOKEN is required")
    approver_token = os.environ.get("INVERSE_AGENT_APPROVER_TOKEN", "")
    approver_id = os.environ.get("INVERSE_AGENT_APPROVER_ID", "")
    if not approver_token or not approver_id:
        raise ValueError("INVERSE_AGENT_APPROVER_TOKEN and INVERSE_AGENT_APPROVER_ID are required")
    service = _service(args, Path(args.workspace_root).resolve())
    app = create_app(
        service=service,
        api_token=api_token,
        approver_tokens={approver_token: approver_id},
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


def mcp_command(args: argparse.Namespace) -> int:
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        create_mcp_server(service).run("stdio")
        return 0
    finally:
        service.close()


def _service(args: argparse.Namespace, workspace_root: Path) -> AgentService:
    secret = os.environ.get("INVERSE_AGENT_APPROVAL_SECRET", "").encode()
    if len(secret) < 32:
        raise ValueError("INVERSE_AGENT_APPROVAL_SECRET must contain at least 32 bytes")
    return AgentService(
        workspace_root=workspace_root,
        state_dir=Path(args.state_dir).resolve(),
        approval_secret=secret,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inverse-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="Detect a workspace profile")
    profile.add_argument("workspace")
    profile.set_defaults(func=profile_command)

    evaluate = sub.add_parser("evaluate", help="Run the reproducible advisory dogfood evaluation")
    evaluate.add_argument("workspace")
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=evaluate_command)

    start = sub.add_parser("start", help="Create and start a durable workflow")
    start.add_argument("workspace")
    start.add_argument("--domain", required=True, choices=[item.value for item in Domain])
    start.add_argument("--goal", default="Run the domain verification workflow")
    start.add_argument(
        "--autonomy",
        type=int,
        default=AutonomyLevel.ASSISTED.value,
        choices=[item.value for item in AutonomyLevel],
    )
    start.add_argument("--state-dir", default=default_state_dir())
    start.set_defaults(func=start_command)

    approve = sub.add_parser("approve", help="Approve the current pending action and resume")
    approve.add_argument("run_id")
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--action-digest", required=True)
    approve.add_argument("--workspace-root", required=True)
    approve.add_argument("--state-dir", default=default_state_dir())
    approve.set_defaults(func=approve_command)

    trust = sub.add_parser("trust-workspace", help="Attest a workspace before executing its code")
    trust.add_argument("workspace")
    trust.add_argument("--trusted-by", required=True)
    trust.add_argument("--workspace-root", required=True)
    trust.add_argument("--state-dir", default=default_state_dir())
    trust.set_defaults(func=trust_command)

    serve = sub.add_parser("serve", help="Serve the authenticated local control plane")
    serve.add_argument("--workspace-root", required=True)
    serve.add_argument("--state-dir", default=default_state_dir())
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=serve_command)

    mcp = sub.add_parser("mcp", help="Serve policy-enforced tools over MCP stdio")
    mcp.add_argument("--workspace-root", required=True)
    mcp.add_argument("--state-dir", default=default_state_dir())
    mcp.set_defaults(func=mcp_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

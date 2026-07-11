"""Command-line interface for profiling and durable agent runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, replace
from importlib.resources import as_file, files
from pathlib import Path

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.commit_review import ReviewDomain, review_commit
from inverse_agent.control_plane import create_app
from inverse_agent.dogfood import evaluate_workspace, save_evaluation
from inverse_agent.eval import json_default
from inverse_agent.mcp_server import create_mcp_server
from inverse_agent.model_config import PlannerResolution, resolve_planner
from inverse_agent.models import AutonomyLevel, Domain, RunStatus, WorkspaceProfile
from inverse_agent.redaction import redact_text
from inverse_agent.review_benchmark import BenchmarkModelProvenance, run_benchmark_suite
from inverse_agent.service import AgentService

BUILTIN_BENCHMARK_SUITE = "builtin"


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
    planner = resolve_planner(args=args, require_model=True).planner if args.use_model else None
    result = evaluate_workspace(Path(args.workspace), planner=planner)
    if args.output:
        save_evaluation(result, Path(args.output).resolve())
    print(json.dumps({**asdict(result), "passed": result.passed}, default=json_default, indent=2))
    return 0 if result.passed else 1


def start_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    service = _service(args, workspace)
    try:
        created = service.create_run(
            goal=args.goal,
            workspace=workspace,
            domain=Domain(args.domain),
            autonomy_level=AutonomyLevel(args.autonomy),
        )
        record = service.start(created.run_id)
        print(json.dumps(asdict(record), default=json_default, indent=2))
        if record.status == RunStatus.WAITING_FOR_APPROVAL.value:
            return 2
        return 0 if record.status == RunStatus.SUCCEEDED.value else 1
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
        if record.status == RunStatus.WAITING_FOR_APPROVAL.value:
            return 2
        return 0 if record.status == RunStatus.SUCCEEDED.value else 1
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
    resolution = resolve_planner(args=args)
    _report_planner(resolution)
    service = _service(args, Path(args.workspace_root).resolve(), resolution=resolution)
    app = create_app(
        service=service,
        api_token=api_token,
        approver_tokens={approver_token: approver_id},
        planner_summary=resolution.config.safe_summary(),
    )
    print(f"Inverse-Agent workbench: http://127.0.0.1:{args.port}", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


def mcp_command(args: argparse.Namespace) -> int:
    resolution = resolve_planner(args=args)
    _report_planner(resolution)
    service = _service(args, Path(args.workspace_root).resolve(), resolution=resolution)
    try:
        create_mcp_server(service).run("stdio")
        return 0
    finally:
        service.close()


def model_check_command(args: argparse.Namespace) -> int:
    started = time.monotonic()
    try:
        resolution = resolve_planner(args=args, require_model=True)
        profile = WorkspaceProfile(
            root=Path.cwd(),
            domains={Domain.GENERIC},
            commands={"generic.inspect": ["unused"]},
        )
        plan = resolution.planner.plan(
            goal="Select the supplied diagnostic tool",
            domain=Domain.GENERIC,
            profile=profile,
            available_tools=("generic.inspect",),
        )
        payload = {
            "ok": True,
            "planner": resolution.config.safe_summary(),
            "latency_seconds": round(time.monotonic() - started, 3),
            "actions": [action.tool_name for action in plan.actions],
            "rationale": plan.rationale,
        }
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        error = redact_text(str(exc)).text
        print(json.dumps({"ok": False, "error": error}, indent=2), file=sys.stderr)
        return 1


def review_commit_command(args: argparse.Namespace) -> int:
    resolution = resolve_planner(args=args, require_model=True)
    if resolution.client is None:  # pragma: no cover - guarded by require_model
        raise RuntimeError("structured review requires a model client")
    report = review_commit(
        Path(args.workspace).resolve(),
        args.commit,
        domain=ReviewDomain(args.domain),
        goal=args.goal,
        client=resolution.client,
    )
    print(json.dumps(asdict(report), default=json_default, indent=2))
    return {"PASS": 0, "FINDINGS": 1, "INCOMPLETE": 3}[report.verdict]


def benchmark_review_command(args: argparse.Namespace) -> int:
    prepared_output = _prepare_benchmark_output(args.output)
    output_path, temporary_output = prepared_output or (None, None)
    try:
        resolution = resolve_planner(args=args, require_model=True)
        if resolution.client is None:  # pragma: no cover - guarded by require_model
            raise RuntimeError("review benchmark requires a model client")
        repository_root = Path(args.repository_root).resolve() if args.repository_root else None
        with _benchmark_suite_path(args.suite) as suite_path:
            result = run_benchmark_suite(
                suite_path,
                client=resolution.client,
                repository_root=repository_root,
            )
        requested_model = resolution.config.model
        base_url = resolution.config.base_url
        if requested_model is None or base_url is None:  # pragma: no cover - require_model guard
            raise RuntimeError("review benchmark model provenance is unavailable")
        reported_models = resolution.client.observed_response_models
        successful_responses = resolution.client.successful_response_count
        attributed_responses = resolution.client.attributed_response_count
        endpoint_model_consistent = (
            reported_models == (requested_model,)
            and successful_responses > 0
            and attributed_responses == successful_responses
        )
        result = replace(
            result,
            passed=result.passed and endpoint_model_consistent,
            model_provenance=BenchmarkModelProvenance(
                kind=resolution.config.kind,
                requested_model=requested_model,
                base_url=base_url,
                config_fingerprint=resolution.config.fingerprint,
                endpoint_reported_models=reported_models,
                successful_responses=successful_responses,
                attributed_responses=attributed_responses,
                endpoint_model_consistent=endpoint_model_consistent,
            ),
        )
        payload = json.dumps(asdict(result), default=json_default, indent=2)
        if output_path is not None and temporary_output is not None:
            temporary_output.write_text(payload + "\n", encoding="utf-8")
            os.replace(temporary_output, output_path)
            temporary_output = None
        print(payload)
        return 0 if result.passed else 1
    finally:
        if temporary_output is not None:
            with suppress(OSError):
                temporary_output.unlink()


@contextmanager
def _benchmark_suite_path(value: str) -> Iterator[Path]:
    if value != BUILTIN_BENCHMARK_SUITE:
        yield Path(value).resolve()
        return
    packaged_root = files("inverse_agent.benchmark_assets").joinpath("commit_review")
    with as_file(packaged_root) as extracted_root:
        yield Path(extracted_root) / "suite.json"


def _prepare_benchmark_output(value: str | None) -> tuple[Path, Path] | None:
    if not value:
        return None
    output_path = Path(value).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not output_path.is_file():
            raise ValueError("benchmark output path must be a file")
        if output_path.stat().st_mode & 0o222 == 0:
            raise PermissionError("benchmark output file is not writable")
        try:
            with output_path.open("ab"):
                pass
        except OSError as exc:
            raise PermissionError("benchmark output file is not writable") from exc
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    return output_path, temporary_path


def _service(
    args: argparse.Namespace,
    workspace_root: Path,
    *,
    resolution: PlannerResolution | None = None,
) -> AgentService:
    secret = os.environ.get("INVERSE_AGENT_APPROVAL_SECRET", "").encode()
    if len(secret) < 32:
        raise ValueError("INVERSE_AGENT_APPROVAL_SECRET must contain at least 32 bytes")
    selected = resolution or resolve_planner(args=args)
    return AgentService(
        workspace_root=workspace_root,
        state_dir=Path(args.state_dir).resolve(),
        approval_secret=secret,
        planner=selected.planner,
        planner_fingerprint=selected.config.fingerprint,
    )


def _report_planner(resolution: PlannerResolution) -> None:
    print(
        "Inverse-Agent planner: " + json.dumps(resolution.config.safe_summary(), sort_keys=True),
        file=sys.stderr,
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model")
    parser.add_argument("--model-base-url")
    parser.add_argument("--model-timeout-seconds", type=int)
    parser.add_argument("--model-max-actions", type=int)
    parser.add_argument("--model-allow-remote", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inverse-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="Detect a workspace profile")
    profile.add_argument("workspace")
    profile.set_defaults(func=profile_command)

    evaluate = sub.add_parser("evaluate", help="Run the reproducible advisory dogfood evaluation")
    evaluate.add_argument("workspace")
    evaluate.add_argument("--output")
    evaluate.add_argument("--use-model", action="store_true")
    _add_model_arguments(evaluate)
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
    _add_model_arguments(start)
    start.set_defaults(func=start_command)

    approve = sub.add_parser("approve", help="Approve the current pending action and resume")
    approve.add_argument("run_id")
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--action-digest", required=True)
    approve.add_argument("--workspace-root", required=True)
    approve.add_argument("--state-dir", default=default_state_dir())
    _add_model_arguments(approve)
    approve.set_defaults(func=approve_command)

    trust = sub.add_parser("trust-workspace", help="Attest a workspace before executing its code")
    trust.add_argument("workspace")
    trust.add_argument("--trusted-by", required=True)
    trust.add_argument("--workspace-root", required=True)
    trust.add_argument("--state-dir", default=default_state_dir())
    _add_model_arguments(trust)
    trust.set_defaults(func=trust_command)

    serve = sub.add_parser("serve", help="Serve the authenticated local control plane")
    serve.add_argument("--workspace-root", required=True)
    serve.add_argument("--state-dir", default=default_state_dir())
    serve.add_argument("--port", type=int, default=8765)
    _add_model_arguments(serve)
    serve.set_defaults(func=serve_command)

    mcp = sub.add_parser("mcp", help="Serve policy-enforced tools over MCP stdio")
    mcp.add_argument("--workspace-root", required=True)
    mcp.add_argument("--state-dir", default=default_state_dir())
    _add_model_arguments(mcp)
    mcp.set_defaults(func=mcp_command)

    model_check = sub.add_parser("model-check", help="Verify model-backed structured planning")
    _add_model_arguments(model_check)
    model_check.set_defaults(func=model_check_command)

    review = sub.add_parser("review-commit", help="Review an immutable Git commit")
    review.add_argument("workspace")
    review.add_argument("commit")
    review.add_argument("--domain", required=True, choices=[item.value for item in ReviewDomain])
    review.add_argument(
        "--goal",
        default="Review this commit for introduced correctness and security defects",
    )
    _add_model_arguments(review)
    review.set_defaults(func=review_commit_command)

    benchmark_review = sub.add_parser(
        "benchmark-review",
        help="Run the multi-domain commit-review acceptance suite",
    )
    benchmark_review.add_argument(
        "suite",
        help="Suite JSON path, or 'builtin' for the packaged acceptance suite",
    )
    benchmark_review.add_argument("--repository-root")
    benchmark_review.add_argument("--output")
    _add_model_arguments(benchmark_review)
    benchmark_review.set_defaults(func=benchmark_review_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

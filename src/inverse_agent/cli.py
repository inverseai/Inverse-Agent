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
from inverse_agent.attestations import AttestationScope
from inverse_agent.commit_review import ReviewDomain, review_commit
from inverse_agent.control_plane import create_app
from inverse_agent.dogfood import evaluate_workspace, save_evaluation
from inverse_agent.eval import json_default
from inverse_agent.investigation_model import ModelInvestigationPlanner
from inverse_agent.mcp_server import create_mcp_server
from inverse_agent.model_config import PlannerResolution, resolve_planner
from inverse_agent.models import AutonomyLevel, Domain, RunKind, RunStatus, WorkspaceProfile
from inverse_agent.redaction import redact_text
from inverse_agent.review_benchmark import BenchmarkModelProvenance, run_benchmark_suite
from inverse_agent.service import AgentService, RunRecord

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
    if getattr(args, "no_wait", False):
        raise ValueError("--no-wait requires submission to a running control plane")
    workspace = Path(args.workspace).resolve()
    service = _service(args, workspace)
    try:
        created = service.create_run(
            goal=args.goal,
            workspace=workspace,
            domain=Domain(args.domain),
            kind=RunKind(args.kind),
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
    if getattr(args, "no_wait", False):
        raise ValueError("--no-wait requires submission to a running control plane")
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        record = service.approve_and_resume(
            args.run_id,
            approved_by=args.approved_by,
            expected_action_digest=args.action_digest,
            expected_challenge_id=args.challenge_id,
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
        result = service.trust_workspace(
            workspace,
            trusted_by=args.trusted_by,
            scope=AttestationScope(args.scope),
        )
        print(json.dumps(result, indent=2))
        return 0
    finally:
        service.close()


def revoke_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        scope = AttestationScope(args.scope)
        revoked = service.revoke_workspace(workspace, scope=scope)
        print(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "scope": scope.value,
                    "revoked": revoked,
                },
                indent=2,
            )
        )
        return 0
    finally:
        service.close()


def list_runs_command(args: argparse.Namespace) -> int:
    if not 1 <= args.limit <= 500 or args.offset < 0:
        raise ValueError("list pagination is out of range")
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        records = service.list(limit=args.limit, offset=args.offset)
        print(json.dumps([asdict(record) for record in records], default=json_default, indent=2))
        return 0
    finally:
        service.close()


def plan_command(args: argparse.Namespace) -> int:
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        print(json.dumps(service.plan_view(args.run_id), default=json_default, indent=2))
        return 0
    finally:
        service.close()


def trace_command(args: argparse.Namespace) -> int:
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        print(json.dumps(service.trace_preview(args.run_id), default=json_default, indent=2))
        return 0
    finally:
        service.close()


def events_command(args: argparse.Namespace) -> int:
    if args.after < 0 or not 1 <= args.limit <= 200:
        raise ValueError("event pagination is out of range")
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        events = service.events(args.run_id, after=args.after, limit=args.limit)
        next_cursor = events[-1].sequence if events else args.after
        print(
            json.dumps(
                {
                    "events": [asdict(event) for event in events],
                    "next_cursor": next_cursor,
                },
                default=json_default,
                indent=2,
            )
        )
        return 0
    finally:
        service.close()


def decline_command(args: argparse.Namespace) -> int:
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        record = service.decline(
            args.run_id,
            declined_by=args.declined_by,
            expected_action_digest=args.action_digest,
            expected_challenge_id=args.challenge_id,
        )
        print(json.dumps(asdict(record), default=json_default, indent=2))
        return 0
    finally:
        service.close()


def cancel_command(args: argparse.Namespace) -> int:
    service = _service(args, Path(args.workspace_root).resolve())
    try:
        record = service.cancel(args.run_id)
        print(json.dumps(asdict(record), default=json_default, indent=2))
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
            and not resolution.client.observed_response_models_overflowed
            and not resolution.client.response_model_mismatch_observed
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
    investigation_planner_factory = None
    model_client = selected.client
    if model_client is not None and selected.config.investigation_available:
        context_tokens = selected.config.context_tokens
        estimator_bytes_per_token = selected.config.estimator_bytes_per_token
        assert context_tokens is not None
        assert estimator_bytes_per_token is not None
        ModelInvestigationPlanner(
            client=model_client,
            context_tokens=context_tokens,
            estimator_bytes_per_token=estimator_bytes_per_token,
        )

        def investigation_planner_factory(record: RunRecord) -> ModelInvestigationPlanner:
            profile = detect_workspace(Path(record.workspace))
            prefix = f"{record.domain}."
            commands = (
                tuple(sorted(name for name in profile.commands if name.startswith(prefix)))
                if record.autonomy_level != AutonomyLevel.ADVISORY.value
                else ()
            )
            return ModelInvestigationPlanner(
                client=model_client,
                context_tokens=context_tokens,
                estimator_bytes_per_token=estimator_bytes_per_token,
                allowed_commands=commands,
            )

    return AgentService(
        workspace_root=workspace_root,
        state_dir=Path(args.state_dir).resolve(),
        approval_secret=secret,
        planner=selected.planner,
        planner_fingerprint=selected.config.fingerprint,
        investigation_planner_factory=investigation_planner_factory,
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
    parser.add_argument(
        "--model-context-tokens",
        type=int,
        choices=(16_384, 24_576, 32_768, 49_152),
    )
    parser.add_argument("--model-estimator-bytes-per-token", type=float)
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
    start.add_argument(
        "--kind",
        default=RunKind.VERIFICATION.value,
        choices=[item.value for item in RunKind],
    )
    start.add_argument("--goal", default="Run the domain verification workflow")
    start.add_argument(
        "--autonomy",
        type=int,
        default=AutonomyLevel.ASSISTED.value,
        choices=[item.value for item in AutonomyLevel],
    )
    start.add_argument("--state-dir", default=default_state_dir())
    start.add_argument(
        "--no-wait",
        action="store_true",
        help="Only supported when submitting to a separately running control plane",
    )
    _add_model_arguments(start)
    start.set_defaults(func=start_command)

    approve = sub.add_parser("approve", help="Approve the current pending action and resume")
    approve.add_argument("run_id")
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--action-digest", required=True)
    approve.add_argument("--challenge-id", required=True)
    approve.add_argument("--workspace-root", required=True)
    approve.add_argument("--state-dir", default=default_state_dir())
    approve.add_argument(
        "--no-wait",
        action="store_true",
        help="Only supported when submitting to a separately running control plane",
    )
    _add_model_arguments(approve)
    approve.set_defaults(func=approve_command)

    trust = sub.add_parser("trust-workspace", help="Attest a workspace before executing its code")
    trust.add_argument("workspace")
    trust.add_argument("--trusted-by", required=True)
    trust.add_argument(
        "--scope",
        default=AttestationScope.CODE_EXECUTION.value,
        choices=(
            AttestationScope.SOURCE_READ.value,
            AttestationScope.CODE_EXECUTION.value,
        ),
    )
    trust.add_argument("--workspace-root", required=True)
    trust.add_argument("--state-dir", default=default_state_dir())
    _add_model_arguments(trust)
    trust.set_defaults(func=trust_command)

    revoke = sub.add_parser("revoke-workspace", help="Revoke one workspace attestation scope")
    revoke.add_argument("workspace")
    revoke.add_argument(
        "--scope",
        required=True,
        choices=(
            AttestationScope.SOURCE_READ.value,
            AttestationScope.CODE_EXECUTION.value,
        ),
    )
    revoke.add_argument("--workspace-root", required=True)
    revoke.add_argument("--state-dir", default=default_state_dir())
    _add_model_arguments(revoke)
    revoke.set_defaults(func=revoke_command)

    list_runs = sub.add_parser("list-runs", help="List durable runs")
    list_runs.add_argument("--workspace-root", required=True)
    list_runs.add_argument("--state-dir", default=default_state_dir())
    list_runs.add_argument("--limit", type=int, default=100)
    list_runs.add_argument("--offset", type=int, default=0)
    _add_model_arguments(list_runs)
    list_runs.set_defaults(func=list_runs_command)

    get_plan = sub.add_parser("get-plan", help="Read a durable run plan")
    get_plan.add_argument("run_id")
    get_plan.add_argument("--workspace-root", required=True)
    get_plan.add_argument("--state-dir", default=default_state_dir())
    _add_model_arguments(get_plan)
    get_plan.set_defaults(func=plan_command)

    get_trace = sub.add_parser("get-trace", help="Read a bounded redacted run trace")
    get_trace.add_argument("run_id")
    get_trace.add_argument("--workspace-root", required=True)
    get_trace.add_argument("--state-dir", default=default_state_dir())
    _add_model_arguments(get_trace)
    get_trace.set_defaults(func=trace_command)

    events = sub.add_parser("events", help="Read ordered durable run events")
    events.add_argument("run_id")
    events.add_argument("--workspace-root", required=True)
    events.add_argument("--state-dir", default=default_state_dir())
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--limit", type=int, default=200)
    _add_model_arguments(events)
    events.set_defaults(func=events_command)

    decline = sub.add_parser("decline", help="Decline the current pending action")
    decline.add_argument("run_id")
    decline.add_argument("--declined-by", required=True)
    decline.add_argument("--action-digest", required=True)
    decline.add_argument("--challenge-id", required=True)
    decline.add_argument("--workspace-root", required=True)
    decline.add_argument("--state-dir", default=default_state_dir())
    _add_model_arguments(decline)
    decline.set_defaults(func=decline_command)

    cancel = sub.add_parser("cancel", help="Cancel a durable run")
    cancel.add_argument("run_id")
    cancel.add_argument("--workspace-root", required=True)
    cancel.add_argument("--state-dir", default=default_state_dir())
    _add_model_arguments(cancel)
    cancel.set_defaults(func=cancel_command)

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

    benchmark_investigation = sub.add_parser(
        "benchmark-investigation",
        help="Run the seven-case semantic investigation acceptance gate",
    )
    benchmark_investigation.add_argument("--output")
    benchmark_investigation.add_argument(
        "--model",
        help="Drive the gate with a model identifier instead of the scripted solver",
    )
    benchmark_investigation.add_argument(
        "--model-base-url",
        help="OpenAI-compatible endpoint (default http://127.0.0.1:1234/v1)",
    )
    benchmark_investigation.add_argument(
        "--model-context-tokens",
        type=int,
        choices=(16_384, 24_576, 32_768, 49_152),
        help=(
            "Calibrated endpoint context capacity; provide the flag or "
            "INVERSE_AGENT_MODEL_CONTEXT_TOKENS for model runs"
        ),
    )
    benchmark_investigation.add_argument(
        "--model-estimator-bytes-per-token",
        type=float,
        help=(
            "Calibrated conservative UTF-8 bytes-per-token floor; defaults to "
            "INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN and is required for model runs"
        ),
    )
    benchmark_investigation.add_argument(
        "--model-allow-remote",
        action="store_true",
        help="Permit a non-loopback model endpoint (requires https)",
    )
    benchmark_investigation.set_defaults(func=benchmark_investigation_command)
    return parser


def benchmark_investigation_command(args: argparse.Namespace) -> int:
    from inverse_agent.investigation_benchmark import (
        BenchmarkCase,
        default_cases,
        run_benchmark,
        run_benchmark_with_planner,
    )

    cases = default_cases()
    endpoint_model_consistent = True
    provenance: dict[str, object] | None = None
    model_requested = args.model is not None
    if model_requested:
        if (
            not isinstance(args.model, str)
            or not args.model.strip()
            or args.model != args.model.strip()
        ):
            raise ValueError(
                "--model must be a non-empty identifier without surrounding whitespace"
            )
        import os

        from inverse_agent.investigation_model import ModelInvestigationPlanner
        from inverse_agent.planner import OpenAICompatibleClient, validate_model_endpoint

        base_url = args.model_base_url or os.environ.get(
            "INVERSE_AGENT_MODEL_BASE_URL", "http://127.0.0.1:1234/v1"
        )
        # Remote endpoints require the same dual opt-in as every other model path:
        # the environment variable AND the CLI flag, together, plus https.
        env_allows_remote = os.environ.get("INVERSE_AGENT_MODEL_ALLOW_REMOTE", "0") in {
            "1",
            "true",
            "True",
        }
        allow_remote = env_allows_remote and bool(getattr(args, "model_allow_remote", False))
        normalized_url = validate_model_endpoint(base_url, allow_remote=allow_remote)
        client = OpenAICompatibleClient(
            base_url=normalized_url,
            model=args.model,
            api_key=os.environ.get("INVERSE_AGENT_MODEL_API_KEY") or None,
            allow_remote=allow_remote,
            timeout_seconds=120,
        )
        context_tokens = args.model_context_tokens
        if context_tokens is None:
            raw_context = os.environ.get("INVERSE_AGENT_MODEL_CONTEXT_TOKENS")
            if raw_context is None:
                raise ValueError(
                    "model investigation requires a calibrated model context-token value"
                )
            try:
                context_tokens = int(raw_context)
            except ValueError as exc:
                raise ValueError("INVERSE_AGENT_MODEL_CONTEXT_TOKENS must be an integer") from exc
        if context_tokens not in {16_384, 24_576, 32_768, 49_152}:
            raise ValueError("model context tokens must be one of 16384, 24576, 32768, or 49152")
        estimator_bytes_per_token = args.model_estimator_bytes_per_token
        if estimator_bytes_per_token is None:
            raw_estimator = os.environ.get("INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN")
            if raw_estimator is None:
                raise ValueError(
                    "model investigation requires a calibrated estimator bytes-per-token value"
                )
            try:
                estimator_bytes_per_token = float(raw_estimator)
            except ValueError as exc:
                raise ValueError(
                    "INVERSE_AGENT_MODEL_ESTIMATOR_BYTES_PER_TOKEN must be numeric"
                ) from exc
        if not 1.0 <= estimator_bytes_per_token <= 4.0:
            raise ValueError("model estimator bytes per token must be between 1.0 and 4.0")

        from inverse_agent.investigation import AgentBudget

        def factory(case: BenchmarkCase, goal: str) -> ModelInvestigationPlanner:
            del goal
            return ModelInvestigationPlanner(
                client=client,
                context_tokens=context_tokens,
                estimator_bytes_per_token=estimator_bytes_per_token,
                goal_hint=case.model_hint,
                allowed_commands=case.command_tools,
            )

        # The gate uses the same calibrated contract as production investigations.
        budget = AgentBudget()
        result = run_benchmark_with_planner(
            cases,
            factory,
            budget=budget,
            expected_model=args.model,
            model_client=client,
        )
        # Every successful response must be attributed to exactly the requested
        # model; a substituted or unattributed model is gate-fatal.
        endpoint_model_consistent = (
            client.observed_response_models == (args.model,)
            and not client.observed_response_models_overflowed
            and not client.response_model_mismatch_observed
            and client.successful_response_count > 0
            and client.attributed_response_count == client.successful_response_count
        )
        provenance = {
            "requested_model": args.model,
            "reported_models": list(client.observed_response_models),
            "reported_models_overflowed": client.observed_response_models_overflowed,
            "model_mismatch_observed": client.response_model_mismatch_observed,
            "successful_responses": client.successful_response_count,
            "attributed_responses": client.attributed_response_count,
            "endpoint_model_consistent": endpoint_model_consistent,
            "context_tokens": context_tokens,
            "estimator_bytes_per_token": estimator_bytes_per_token,
        }
    else:
        result = run_benchmark(cases)
    gate_passed = result.gate_passed and endpoint_model_consistent
    summary = {
        "planner": "model" if model_requested else "deterministic",
        "cases_passed": result.cases_passed,
        "total_cases": result.total_cases,
        "variants_passed": result.variants_passed,
        "total_variants": result.total_variants,
        "gate_passed": gate_passed,
        "integrity_failures": list(result.integrity_failures),
        "model_provenance": provenance,
        "variants": [
            {
                "case": variant.case,
                "passed": variant.passed,
                "verdict": variant.verdict,
                "reason": variant.reason,
                "integrity_failures": list(variant.integrity_failures),
                "decisions_used": variant.decisions_used,
                "tool_calls_used": variant.tool_calls_used,
                "command_calls_used": variant.command_calls_used,
                "physical_requests_used": variant.physical_requests_used,
                "completion_tokens_used": variant.completion_tokens_used,
                "completion_tokens_charged": variant.completion_tokens_charged,
                "completion_tokens_requested": variant.completion_tokens_requested,
                "observation_bytes_used": variant.observation_bytes_used,
                "active_seconds": variant.active_seconds,
                "transport_retries": variant.transport_retries,
                "schema_retries": variant.schema_retries,
                "model_calls": [asdict(call) for call in variant.model_calls],
                "model_endpoint_audit": (
                    asdict(variant.model_endpoint_audit)
                    if variant.model_endpoint_audit is not None
                    else None
                ),
                "command_audit": [asdict(item) for item in variant.command_audit],
            }
            for variant in result.variants
        ],
    }
    if args.output:
        Path(args.output).resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if gate_passed else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

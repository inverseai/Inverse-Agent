"""Authenticated FastAPI control plane backed by the durable agent service."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from secrets import compare_digest
from typing import Any, cast

from pydantic import BaseModel, Field

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.eval import json_default
from inverse_agent.models import AutonomyLevel, Domain
from inverse_agent.service import AgentService, RunRecord

API_VERSION = "2026-07-10.v1"
UI_ASSETS = {
    "app.css": "text/css",
    "app.js": "text/javascript",
}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
        "font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'; require-trusted-types-for 'script'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}


class RunCreate(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)
    workspace: str
    domain: Domain
    autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED


class WorkspaceTrustCreate(BaseModel):
    workspace: str


class ApprovalCreate(BaseModel):
    action_digest: str = Field(min_length=64, max_length=64)


def create_app(
    *,
    service: AgentService,
    api_token: str,
    approver_tokens: dict[str, str],
    planner_summary: dict[str, str | int | bool | None] | None = None,
) -> Any:
    if not api_token:
        raise ValueError("control-plane API token is required")
    if not approver_tokens or any(not token or not identity for token, identity in approver_tokens.items()):
        raise ValueError("at least one approver token and identity are required")
    if api_token in approver_tokens:
        raise ValueError("operator and approver tokens must be distinct")
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query
        from fastapi.responses import Response
        from starlette.middleware.trustedhost import TrustedHostMiddleware
    except Exception as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("fastapi and pydantic are required") from exc

    app = FastAPI(
        title="Inverse-Agent Control Plane",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "::1", "[::1]"],
    )
    ui_files = _load_ui_assets()

    @app.middleware("http")
    async def add_security_headers(_request: Any, call_next: Any) -> Any:
        response = await call_next(_request)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        return response

    def require_auth(x_inverse_agent_token: str | None = Header(default=None)) -> None:
        if not _tokens_match(x_inverse_agent_token or "", api_token):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    def require_approver(
        x_inverse_agent_approval_token: str | None = Header(default=None),
    ) -> str:
        supplied = x_inverse_agent_approval_token or ""
        for token, identity in approver_tokens.items():
            if _tokens_match(supplied, token):
                return identity
        raise HTTPException(status_code=401, detail="invalid or missing approver token")

    @app.get("/", include_in_schema=False)
    def ui_index() -> Any:
        return Response(content=ui_files["index.html"], media_type="text/html")

    @app.get("/assets/{name}", include_in_schema=False)
    def ui_asset(name: str) -> Any:
        media_type = UI_ASSETS.get(name)
        if media_type is None:
            raise HTTPException(status_code=404, detail="asset not found")
        return Response(content=ui_files[name], media_type=media_type)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runtime", dependencies=[Depends(require_auth)])
    def runtime() -> dict[str, Any]:
        safe_planner = {
            key: value
            for key, value in (planner_summary or {"kind": "unknown"}).items()
            if key
            in {
                "kind",
                "model",
                "base_url",
                "timeout_seconds",
                "max_actions",
                "allow_remote",
                "api_key_set",
            }
        }
        return {
            "api_version": API_VERSION,
            "workspace_root": str(service.workspace_root),
            "planner": safe_planner,
        }

    @app.get("/approver/session")
    def approver_session(approver: str = Depends(require_approver)) -> dict[str, str]:
        return {"approver": approver}

    @app.get("/profile", dependencies=[Depends(require_auth)])
    def profile(path: str) -> dict[str, Any]:
        try:
            workspace = _resolve_workspace(Path(path), service.workspace_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        payload = _json_safe(asdict(detect_workspace(workspace)))
        payload["trust"] = service.workspace_trust_status(workspace)
        return payload

    @app.get("/workspaces/trust-status", dependencies=[Depends(require_auth)])
    def trust_status(path: str) -> dict[str, Any]:
        try:
            workspace = _resolve_workspace(Path(path), service.workspace_root)
            return service.workspace_trust_status(workspace)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/runs", dependencies=[Depends(require_auth)])
    def create_run(body: RunCreate) -> dict[str, Any]:
        try:
            record = service.create_run(
                goal=body.goal,
                workspace=Path(body.workspace),
                domain=body.domain,
                autonomy_level=body.autonomy_level,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _run_view(record)

    @app.post("/workspaces/trust")
    def trust_workspace(
        body: WorkspaceTrustCreate,
        approver: str = Depends(require_approver),
    ) -> dict[str, Any]:
        try:
            return service.trust_workspace(Path(body.workspace), trusted_by=approver)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/runs", dependencies=[Depends(require_auth)])
    def list_runs(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        return [_run_view(record) for record in service.list(limit=limit, offset=offset)]

    @app.get("/runs/{run_id}", dependencies=[Depends(require_auth)])
    def get_run(run_id: str) -> dict[str, Any]:
        return _run_view(_require_run(service, run_id, HTTPException))

    @app.get("/runs/{run_id}/plan", dependencies=[Depends(require_auth)])
    def get_plan(run_id: str) -> dict[str, Any]:
        _require_run(service, run_id, HTTPException)
        return service.plan_view(run_id)

    @app.get("/runs/{run_id}/trace", dependencies=[Depends(require_auth)])
    def get_trace(run_id: str) -> dict[str, Any]:
        _require_run(service, run_id, HTTPException)
        try:
            return service.trace_preview(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/start", dependencies=[Depends(require_auth)])
    def start_run(run_id: str) -> dict[str, Any]:
        _require_run(service, run_id, HTTPException)
        try:
            return _run_view(service.start(run_id))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/approvals")
    def approve(
        run_id: str,
        body: ApprovalCreate,
        approver: str = Depends(require_approver),
    ) -> dict[str, Any]:
        _require_run(service, run_id, HTTPException)
        try:
            return _run_view(
                service.approve_and_resume(
                    run_id,
                    approved_by=approver,
                    expected_action_digest=body.action_digest,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/decline")
    def decline(
        run_id: str,
        body: ApprovalCreate,
        approver: str = Depends(require_approver),
    ) -> dict[str, Any]:
        _require_run(service, run_id, HTTPException)
        try:
            return _run_view(
                service.decline(
                    run_id,
                    declined_by=approver,
                    expected_action_digest=body.action_digest,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


def _tokens_match(supplied: str, expected: str) -> bool:
    return compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _resolve_workspace(path: Path, allowed_root: Path) -> Path:
    candidate = path.resolve()
    if not candidate.is_relative_to(allowed_root.resolve()):
        raise ValueError("path outside configured workspace root")
    if not candidate.is_dir():
        raise ValueError("workspace directory does not exist")
    return candidate


def _require_run(service: AgentService, run_id: str, exception_type: Any) -> Any:
    try:
        return service.get(run_id)
    except KeyError as exc:
        raise exception_type(status_code=404, detail=str(exc)) from exc


def _json_safe(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value, default=json_default)))


def _load_ui_assets() -> dict[str, bytes]:
    package = files("inverse_agent.ui")
    names = ("index.html", *UI_ASSETS)
    return {name: package.joinpath(name).read_bytes() for name in names}


def _run_view(record: RunRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload.pop("trace_path", None)
    payload["has_trace"] = record.trace_path is not None
    payload["plan"] = list(record.plan)
    return payload

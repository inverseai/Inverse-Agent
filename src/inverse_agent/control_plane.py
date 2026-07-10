"""Authenticated FastAPI control plane backed by the durable agent service."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from secrets import compare_digest
from typing import Any, cast

from pydantic import BaseModel, Field

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.eval import json_default
from inverse_agent.models import AutonomyLevel, Domain
from inverse_agent.service import AgentService


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
) -> Any:
    if not api_token:
        raise ValueError("control-plane API token is required")
    if not approver_tokens or any(not token or not identity for token, identity in approver_tokens.items()):
        raise ValueError("at least one approver token and identity are required")
    if api_token in approver_tokens:
        raise ValueError("operator and approver tokens must be distinct")
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except Exception as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("fastapi and pydantic are required") from exc

    app = FastAPI(
        title="Inverse-Agent Control Plane",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def require_auth(x_inverse_agent_token: str | None = Header(default=None)) -> None:
        if not compare_digest(x_inverse_agent_token or "", api_token):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    def require_approver(
        x_inverse_agent_approval_token: str | None = Header(default=None),
    ) -> str:
        supplied = x_inverse_agent_approval_token or ""
        for token, identity in approver_tokens.items():
            if compare_digest(supplied, token):
                return identity
        raise HTTPException(status_code=401, detail="invalid or missing approver token")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/profile", dependencies=[Depends(require_auth)])
    def profile(path: str) -> dict[str, Any]:
        try:
            workspace = _resolve_workspace(Path(path), service.workspace_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return _json_safe(asdict(detect_workspace(workspace)))

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
        return asdict(record)

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
    def list_runs() -> list[dict[str, Any]]:
        return [asdict(record) for record in service.list()]

    @app.get("/runs/{run_id}", dependencies=[Depends(require_auth)])
    def get_run(run_id: str) -> dict[str, Any]:
        return asdict(_require_run(service, run_id, HTTPException))

    @app.post("/runs/{run_id}/start", dependencies=[Depends(require_auth)])
    def start_run(run_id: str) -> dict[str, Any]:
        _require_run(service, run_id, HTTPException)
        try:
            return asdict(service.start(run_id))
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
            return asdict(
                service.approve_and_resume(
                    run_id,
                    approved_by=approver,
                    expected_action_digest=body.action_digest,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


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
    import json

    return cast(dict[str, Any], json.loads(json.dumps(value, default=json_default)))

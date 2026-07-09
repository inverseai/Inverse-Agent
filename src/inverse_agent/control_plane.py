"""Minimal FastAPI control-plane API."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from secrets import compare_digest
from typing import Any

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.eval import json_default


def create_app(workspace_root: Path | None = None, api_token: str | None = None) -> Any:
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("fastapi is required to create the control-plane app") from exc

    app = FastAPI(title="Inverse-Agent Control Plane", version="0.1.0")
    runs: dict[str, dict[str, Any]] = {}
    allowed_root = (workspace_root or Path.cwd()).resolve()

    def require_auth(x_inverse_agent_token: str | None = Header(default=None)) -> None:
        if api_token and not compare_digest(x_inverse_agent_token or "", api_token):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    def resolve_workspace(path: str) -> Path:
        candidate = Path(path).resolve()
        try:
            candidate.relative_to(allowed_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="path outside configured workspace root") from exc
        return candidate

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/profile")
    def profile(path: str, _: None = Depends(require_auth)) -> dict[str, Any]:
        return _json_safe(asdict(detect_workspace(resolve_workspace(path))))

    @app.post("/runs/{run_id}/approvals")
    def approve(run_id: str, approval: dict[str, Any], _: None = Depends(require_auth)) -> dict[str, Any]:
        runs.setdefault(run_id, {"approvals": []})
        runs[run_id].setdefault("approvals", []).append(approval)
        return {"run_id": run_id, "approval_count": len(runs[run_id]["approvals"])}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
        return runs.get(run_id, {"run_id": run_id, "status": "unknown"})

    return app


def _json_safe(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value, default=json_default))

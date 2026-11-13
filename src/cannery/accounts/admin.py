"""Read-only operator summaries. No remote commands or filesystem mutation."""

import shutil
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from cannery.accounts.context import identity, role, template_context
from cannery.benchmark.state import PerformanceJob
from cannery.builds.state import BuildState
from cannery.numerics.state import NumericalState
from cannery.registry.models import SpecVersion
from cannery.registry.registrations import Registration
from cannery.settings import Settings


def require_admin(settings: Settings) -> None:
    identity(settings)
    if role(settings) != "admin":
        raise HTTPException(403, "Administrator role required")


def summary(settings: Settings, engine: Engine) -> dict[str, Any]:
    require_admin(settings)
    with Session(engine) as db:
        queues = {}
        for label, status in [
            ("build", BuildState.status),
            ("validation", NumericalState.status),
            ("benchmark", PerformanceJob.status),
            ("registration", Registration.status),
        ]:
            queues[label] = dict(db.execute(select(status, func.count()).group_by(status)).all())
        snapshots = db.scalar(select(func.count()).select_from(SpecVersion))
    try:
        capacity = shutil.disk_usage(settings.snapshot_root)
        disk = {"total_bytes": capacity.total, "free_bytes": capacity.free}
    except OSError:
        disk = None
    return {
        "queues": queues,
        "frozen_specifications": snapshots,
        "snapshot_filesystem": disk,
        "artifact_backend": settings.artifact_backend,
        "note": "Queue state does not establish remote device health.",
    }


def install_admin(
    app: FastAPI,
    settings: Settings,
    engine: Engine,
    templates: Jinja2Templates,
    context: dict[str, object],
) -> None:
    @app.get("/api/v1/admin/summary")
    def status() -> dict[str, Any]:
        return summary(settings, engine)

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    def admin(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="admin.html",
            context={
                **context,
                **template_context(settings),
                "title": "Administration",
                "summary": summary(settings, engine),
            },
            headers={"Cache-Control": "no-store"},
        )

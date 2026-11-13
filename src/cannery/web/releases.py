"""Published Global Best partitions, one record curve and immutable release details."""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cannery.devices import Device
from cannery.releases import catalog
from cannery.releases.details import detail
from cannery.releases.discovery_api import selected, selection_error
from cannery.settings import Settings


def chart(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {"points": [], "line": ""}
    times = [datetime.fromisoformat(p["published_at"]).timestamp() for p in points]
    span = max(times[-1] - times[0], 1.0)
    ceiling = max(1.0, max(p["geomean_speedup"] for p in points)) * 1.1
    dots = [
        p | {"x": 50 + (t - times[0]) / span * 700, "y": 220 - p["geomean_speedup"] / ceiling * 180}
        for p, t in zip(points, times, strict=True)
    ]
    line = f"M {dots[0]['x']} {dots[0]['y']}"
    for dot in dots[1:]:
        line += f" H {dot['x']} V {dot['y']}"
    return {
        "points": dots,
        "line": line,
        "reference_y": 220 - 180 / ceiling,
        "start": points[0]["published_at"],
        "end": points[-1]["published_at"],
    }


def install_release_web(
    app: FastAPI,
    settings: Settings,
    engine: Engine,
    templates: Jinja2Templates,
    context: dict[str, object],
) -> None:
    if not settings.release_enabled:
        return

    @app.get("/global-best", response_class=HTMLResponse, include_in_schema=False)
    def global_best(
        request: Request,
        kernel: str | None = None,
        spec_version: str | None = None,
        device: Device = "a5",
        cohort: str | None = None,
        rtol: str | None = None,
        atol: str | None = None,
    ) -> HTMLResponse:
        values: dict[str, Any] = {
            "kernel": kernel,
            "device": device,
            "spec_version": spec_version,
            "cohort": cohort,
            "rtol": rtol,
            "atol": atol,
        }
        with Session(engine) as db:
            values["kernels"] = catalog.kernels(db)
            if kernel:
                try:
                    selection = selected(
                        request,
                        kernel,
                        int(spec_version) if spec_version else None,
                        device,
                        cohort or None,
                        float(rtol) if rtol else None,
                        float(atol) if atol else None,
                    )
                except ValueError:
                    raise HTTPException(422, "Invalid specification or tolerance") from None
                try:
                    values["ranking"] = catalog.leaderboard(db, selection)
                    values["curve"] = chart(catalog.history(db, selection)["points"])
                    values["cohorts"] = values["ranking"]["cohorts"]
                except catalog.SelectionError as error:
                    values["error"] = str(error)
                    values["cohorts"] = error.cohorts or []
        return templates.TemplateResponse(
            request=request,
            name="releases.html",
            context=context | {"title": "Global Best"} | values,
        )

    @app.get("/releases/{identifier}", response_class=HTMLResponse, include_in_schema=False)
    def release(request: Request, identifier: UUID) -> HTMLResponse:
        try:
            with Session(engine) as db:
                item, publication = catalog.release_info(db, identifier)
                value = detail(db, item, publication)
        except catalog.SelectionError as error:
            raise selection_error(error) from None
        return templates.TemplateResponse(
            request=request,
            name="releases.html",
            context=context | {"title": "Global Best", "release": value},
        )

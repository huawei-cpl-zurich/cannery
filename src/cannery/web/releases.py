"""Published Global Best partitions, one record curve and immutable release details."""

from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlencode
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cannery.accounts.context import template_context
from cannery.devices import Device
from cannery.releases import catalog
from cannery.releases.details import detail
from cannery.releases.discovery_api import selected, selection_error
from cannery.settings import Settings
from cannery.web.downloads import origin, snippets
from cannery.web.presentation import configure, number

DIRECTORY_COLUMNS = {
    "kernel": "Kernel",
    "device": "Device",
    "gain": "Gain",
    "implementation": "Implementation",
    "submissions": "Published submissions",
}


def chart(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {"points": [], "line": ""}
    times = [datetime.fromisoformat(p["published_at"]).timestamp() for p in points]
    span = times[-1] - times[0]
    ceiling = max(1.0, max(p["geomean_speedup"] for p in points)) * 1.1
    dots = [
        p
        | {
            "x": 50 + (t - times[0]) / span * 700 if span else 400,
            "y": 220 - p["geomean_speedup"] / ceiling * 180,
        }
        for p, t in zip(points, times, strict=True)
    ]
    line = f"M {dots[0]['x']} {dots[0]['y']}"
    best = dots[0]
    for dot in dots[1:]:
        if dot["geomean_speedup"] > best["geomean_speedup"]:
            line += f" H {dot['x']} V {dot['y']}"
            best = dot
    if best is not dots[-1]:
        line += f" H {dots[-1]['x']}"
    ticks = [{"y": 220 - i * 45, "label": number(ceiling * i / 4)} for i in range(5)]
    axis_times = [(50, times[0]), (400, (times[0] + times[-1]) / 2), (750, times[-1])]
    if times[0] == times[-1]:
        axis_times = [(400, times[0])]
    dates = []
    for x, t in axis_times:
        moment = datetime.fromtimestamp(t, UTC)
        time = moment.strftime("%H:%M:%S")
        if 0 < span < 2 and moment.microsecond:
            time += f".{moment.microsecond:06d}".rstrip("0")
        dates.append({"x": x, "date": moment.strftime("%d %b %Y"), "time": time})
    return {
        "ticks": ticks,
        "dates": dates,
        "best": number(best["geomean_speedup"]),
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
    configure(templates)
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
        q: str = "",
        catalog_device: Literal["", "a3", "a5"] = "",
        sort: Literal["kernel", "device", "gain", "implementation", "submissions"] | None = None,
        direction: Literal["asc", "desc"] | None = None,
        device_sort: Literal["", "asc", "desc"] = "",
    ) -> HTMLResponse:
        column = sort or ("device" if device_sort else "kernel")
        order = direction or (device_sort if sort is None and device_sort else "asc")
        values: dict[str, Any] = {
            "kernel": kernel,
            "q": q,
            "catalog_device": catalog_device,
            "sort": column,
            "direction": order,
            "device": device,
            "spec_version": spec_version,
            "cohort": cohort,
            "rtol": rtol,
            "atol": atol,
        }
        with Session(engine) as db:
            directory = catalog.directory(db)
            values["kernels"] = bool(directory)
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
                    ranking = catalog.leaderboard(db, selection)
                    curve = chart(catalog.evolution(db, selection))
                    values.update(ranking=ranking, curve=curve, cohorts=ranking["cohorts"])
                except catalog.SelectionError as error:
                    values["error"] = str(error)
                    values["cohorts"] = error.cohorts or []
            rows = []
            for item in directory:
                if q.strip().casefold() not in item["kernel"].casefold():
                    continue
                row_device = item["device"]
                if catalog_device and row_device != catalog_device:
                    continue
                active = (
                    "ranking" in values
                    and (item["kernel"], row_device) == (kernel, device)
                    and values["ranking"]["spec_version"] == item["version"]
                    and not (cohort or rtol or atol)
                )
                params = {"kernel": item["kernel"], "device": row_device}
                if q:
                    params["q"] = q
                if catalog_device:
                    params["catalog_device"] = catalog_device
                if sort or direction or device_sort:
                    params.update(sort=column, direction=order)
                params["spec_version"] = str(item["version"])
                rows.append(
                    item
                    | {
                        "active": active,
                        "url": "/global-best?" + urlencode(params) + "#kernel-results",
                    }
                )
            if column in ("gain", "implementation"):
                available = [row for row in rows if row["best"]]
                if column == "gain":
                    available.sort(key=lambda row: row["best"]["speedup"], reverse=order == "desc")
                else:
                    available.sort(
                        key=lambda row: row["best"]["name"].casefold(), reverse=order == "desc"
                    )
                rows = available + [row for row in rows if not row["best"]]
            else:
                sort_key = "submission_count" if column == "submissions" else column
                rows.sort(key=lambda row: row[sort_key], reverse=order == "desc")
            values["kernel_rows"] = rows
            sort_params = dict(request.query_params)
            sort_params.pop("device_sort", None)
            headers = []
            for key, title in DIRECTORY_COLUMNS.items():
                active = key == column
                next_order = "desc" if active and order == "asc" else "asc"
                if key in ("gain", "submissions") and not active:
                    next_order = "desc"
                params = sort_params | {"sort": key, "direction": next_order}
                headers.append(
                    {
                        "title": title,
                        "current": ("ascending" if order == "asc" else "descending")
                        if active
                        else "",
                        "next": "ascending" if next_order == "asc" else "descending",
                        "url": "/global-best?" + urlencode(params),
                    }
                )
            values["directory_headers"] = headers
        return templates.TemplateResponse(
            request=request,
            name="releases.html",
            context=context | template_context(settings) | {"title": "Global Best"} | values,
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
            context=context
            | template_context(settings)
            | {
                "title": "Global Best",
                "release": value,
                "download_snippets": snippets(
                    value, origin(request, settings), settings.hub_enabled
                ),
            },
        )

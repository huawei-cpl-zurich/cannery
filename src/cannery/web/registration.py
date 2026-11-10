"""Git registration form and namespace-owned asynchronous status pages."""

from urllib.parse import parse_qs
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from cannery.registry.api import RegistrationRequest, identity, job_info, submit
from cannery.registry.registrations import Registration
from cannery.settings import Settings
from cannery.web.examples import NAMESPACE, REGISTRATION


def install_registration(
    app: FastAPI,
    settings: Settings,
    engine: Engine,
    templates: Jinja2Templates,
    context: dict[str, object],
) -> None:
    def render(request: Request, **values: object) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="registration.html",
            context={
                **context,
                "title": "Register Kernel",
                "namespace": settings.dev_namespace,
                **values,
            },
        )

    @app.get("/register-kernel", response_class=HTMLResponse, include_in_schema=False)
    def form(request: Request) -> HTMLResponse:
        namespace = identity(settings)
        available = namespace == NAMESPACE
        return render(
            request,
            example_available=available,
            example=REGISTRATION
            if available and request.query_params.get("example") == "helloworld"
            else {},
        )

    @app.post("/register-kernel", include_in_schema=False, response_model=None)
    async def register(request: Request) -> HTMLResponse | RedirectResponse:
        identity(settings)
        if request.headers.get("origin") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Same-origin form submission required")
        if (
            request.headers.get("content-type", "").split(";")[0]
            != "application/x-www-form-urlencoded"
        ):
            raise HTTPException(415, "Form encoding required")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 16384:
                raise HTTPException(413, "Form too large")
        try:
            fields = parse_qs(
                data.decode("utf-8", errors="replace"), keep_blank_values=True, max_num_fields=12
            )
            if any(len(values) != 1 for values in fields.values()) or set(fields) - {
                "url",
                "commit",
                "path",
                "dependencies",
                "name",
            }:
                raise ValueError("Unexpected form fields")
            values = {key: entries[0].strip() for key, entries in fields.items()}
            registration = RegistrationRequest.model_validate(
                {
                    "name": values.get("name") or None,
                    "source": {
                        "url": values.get("url", ""),
                        "commit": values.get("commit", ""),
                        "path": values.get("path", ""),
                        "dependencies": values.get("dependencies", "").split(),
                    },
                }
            )
        except (ValueError, ValidationError):
            return render(
                request, error="Enter a Git repository URL, full commit SHA and source directory."
            )
        key = await run_in_threadpool(submit, settings, engine, registration)
        return RedirectResponse(f"/registrations/{key}", status_code=303)

    @app.get("/registrations/{key}", response_class=HTMLResponse, include_in_schema=False)
    def status(request: Request, key: UUID) -> HTMLResponse:
        namespace = identity(settings)
        with Session(engine) as session:
            job = session.get(Registration, key)
            if job is None or job.namespace != namespace:
                raise HTTPException(404, "Registration not found")
            return render(request, job=job_info(job, session))

"""Namespace-owned submissions, source intake status and commit-only revisions."""

from typing import Any
from urllib.parse import parse_qs
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from cannery.accounts.context import identity, template_context
from cannery.registry.models import Kernel, SpecVersion
from cannery.releases.api import publish, withdraw_publication
from cannery.settings import Settings
from cannery.submissions.api import (
    RevisionRequest,
    SubmitRequest,
    get_info,
    list_info,
    revise,
    submit,
)
from cannery.web.examples import (
    CATLASS_SUBMISSION,
    NAMESPACE,
    NATIVE_SUBMISSION,
    NUMERICAL_CATLASS,
    NUMERICAL_NATIVE,
    NUMERICAL_SPEC_LABEL,
    SPEC_LABEL,
    SUBMISSION,
)


async def form_data(request: Request, allowed: set[str]) -> dict[str, str]:
    if request.headers.get("origin") != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "Same-origin form submission required")
    if request.headers.get("content-type", "").split(";")[0] != "application/x-www-form-urlencoded":
        raise HTTPException(415, "Form encoding required")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > 16384:
            raise HTTPException(413, "Form too large")
        data.extend(chunk)
    try:
        fields = parse_qs(
            data.decode("utf-8", errors="replace"), keep_blank_values=True, max_num_fields=16
        )
        if set(fields) - allowed or any(len(values) != 1 for values in fields.values()):
            raise ValueError("Unexpected form fields")
        return {key: values[0].strip() for key, values in fields.items()}
    except ValueError:
        raise HTTPException(400, "Invalid form fields") from None


def submission_pending(job: dict[str, Any], benchmark_enabled: bool = False) -> bool:
    if job.get("publication", {}).get("status") == "publishing":
        return True
    if job["numerical"] and job["numerical"]["status"] == "blocked":
        return False
    if job["status"] in {
        "queued",
        "intaking",
        "build_queued",
        "building",
        "awaiting_validation",
        "validation_queued",
        "validating",
        "benchmark_queued",
        "benchmarking",
    }:
        return True
    return bool(
        benchmark_enabled
        and job["status"] == "awaiting_benchmark"
        and not (job.get("benchmark") and job["benchmark"]["status"] == "blocked")
    )


def install_submission_web(
    app: FastAPI,
    settings: Settings,
    engine: Engine,
    templates: Jinja2Templates,
    context: dict[str, object],
) -> None:
    def specs() -> list[dict[str, str]]:
        namespace = identity(settings)
        with Session(engine) as session:
            return [
                {
                    "id": str(spec.id),
                    "label": f"{namespace}/{spec.manifest['kernel']['name']}@spec-v{spec.version}",
                }
                for spec in session.scalars(
                    select(SpecVersion)
                    .join(Kernel)
                    .where(Kernel.namespace == namespace)
                    .order_by(SpecVersion.created_at.desc())
                )
                if spec.manifest["kernel"]["namespace"] == namespace
            ]

    def render(request: Request, title: str, **values: object) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="submissions.html",
            context={
                **context,
                **template_context(settings),
                "release_enabled": settings.release_enabled,
                "title": title,
                **values,
            },
        )

    def parse_request(fields: dict[str, str], namespace: str) -> tuple[str, SubmitRequest]:
        spec_id = UUID(fields.get("spec_id", ""))
        with Session(engine) as session:
            spec = session.get(SpecVersion, spec_id)
            if spec is None or spec.manifest["kernel"]["namespace"] != namespace:
                raise ValueError("Frozen specification not found")
            name, version = spec.manifest["kernel"]["name"], spec.version
        body = SubmitRequest.model_validate(
            {
                "spec_version": version,
                "device": fields.get("device"),
                "language": fields.get("language"),
                "source": {
                    "url": fields.get("url"),
                    "commit": fields.get("commit"),
                    "path": fields.get("path"),
                    "dependencies": fields.get("dependencies", "").split(),
                },
            }
        )
        return name, body

    @app.get("/submit-implementation", response_class=HTMLResponse, include_in_schema=False)
    def form(request: Request) -> HTMLResponse:
        choices = specs()
        example_id = request.query_params.get("example")
        numerical = example_id in ("helloworld-native-v2", "helloworld-catlass-v2")
        label = NUMERICAL_SPEC_LABEL if numerical else SPEC_LABEL
        sample = next((spec for spec in choices if spec["label"] == label), None)
        presets = {
            "helloworld": SUBMISSION,
            "helloworld-native": NATIVE_SUBMISSION,
            "helloworld-catlass": CATLASS_SUBMISSION,
            "helloworld-native-v2": NUMERICAL_NATIVE,
            "helloworld-catlass-v2": NUMERICAL_CATLASS,
        }
        selected = sample is not None and example_id in presets
        values = presets.get(example_id or "", SUBMISSION)
        return render(
            request,
            "Submit Implementation",
            specs=choices,
            example_available=any(spec["label"] == SPEC_LABEL for spec in choices),
            numerical_available=any(spec["label"] == NUMERICAL_SPEC_LABEL for spec in choices),
            example_namespace=identity(settings) == NAMESPACE,
            example={**values, "spec_id": sample["id"]} if selected and sample else {},
        )

    @app.post("/submit-implementation", include_in_schema=False, response_model=None)
    async def create(request: Request) -> HTMLResponse | RedirectResponse:
        namespace = identity(settings)
        fields = await form_data(
            request, {"spec_id", "device", "language", "url", "commit", "path", "dependencies"}
        )
        try:
            name, body = await run_in_threadpool(parse_request, fields, namespace)
        except ValueError:
            return render(
                request,
                "Submit Implementation",
                specs=await run_in_threadpool(specs),
                error="Choose a frozen specification and valid pinned Git source/target.",
            )
        key = await run_in_threadpool(submit, settings, engine, namespace, name, body)
        return RedirectResponse(f"/submissions/{key}", status_code=303)

    @app.get("/my-submissions", response_class=HTMLResponse, include_in_schema=False)
    def mine(request: Request) -> HTMLResponse:
        namespace = identity(settings)
        with Session(engine) as session:
            jobs = list_info(session, namespace, publication=settings.release_enabled)
        return render(
            request,
            "My Submissions",
            jobs=jobs,
            pending=any(submission_pending(job, settings.benchmark_enabled) for job in jobs),
        )

    @app.get("/submissions/{identifier}", response_class=HTMLResponse, include_in_schema=False)
    def detail(request: Request, identifier: UUID) -> HTMLResponse:
        namespace = identity(settings)
        with Session(engine) as session:
            job = get_info(session, namespace, identifier, publication=settings.release_enabled)
        return render(
            request,
            "My Submissions",
            job=job,
            pending=submission_pending(job, settings.benchmark_enabled),
        )

    if settings.release_enabled:

        @app.post("/submissions/{identifier}/publish", include_in_schema=False)
        async def publication(request: Request, identifier: UUID) -> RedirectResponse:
            identity(settings)
            await form_data(request, set())
            await run_in_threadpool(publish, settings, engine, identifier)
            return RedirectResponse(f"/submissions/{identifier}", status_code=303)

        @app.post("/submissions/{identifier}/withdraw", include_in_schema=False)
        async def withdrawal(request: Request, identifier: UUID) -> RedirectResponse:
            identity(settings)
            await form_data(request, set())
            await run_in_threadpool(withdraw_publication, settings, engine, identifier)
            return RedirectResponse(f"/submissions/{identifier}", status_code=303)

    @app.post("/submissions/{identifier}/revisions", include_in_schema=False)
    async def revision(request: Request, identifier: UUID) -> RedirectResponse:
        identity(settings)
        fields = await form_data(request, {"commit"})
        try:
            body = RevisionRequest.model_validate(fields)
        except ValueError:
            raise HTTPException(400, "A full commit SHA is required") from None
        key = await run_in_threadpool(revise, settings, engine, identifier, body.commit)
        return RedirectResponse(f"/submissions/{key}", status_code=303)

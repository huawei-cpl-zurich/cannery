"""Published metadata is installation-wide; private submissions remain separate."""

from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cannery.releases import catalog
from cannery.releases.contract import SHA
from cannery.releases.details import detail
from cannery.settings import Settings
from cannery.submissions.manifest import KernelPath


def selected(
    request: Request,
    kernel: KernelPath,
    spec_version: Annotated[int | None, Query(ge=1)] = None,
    device: Literal["a5"] = "a5",
    cohort: Annotated[str | None, Query(pattern=SHA)] = None,
    rtol: Annotated[float | None, Query(ge=0, allow_inf_nan=False)] = None,
    atol: Annotated[float | None, Query(ge=0, allow_inf_nan=False)] = None,
) -> catalog.Selection:
    allowed = {"kernel", "spec_version", "device", "cohort", "rtol", "atol"}
    if request.url.path.endswith(("/leaderboard", "/releases")):
        allowed |= {"limit", "offset"}
    if any(
        k not in allowed or len(request.query_params.getlist(k)) != 1 for k in request.query_params
    ):
        raise HTTPException(422, "Unsupported or repeated query field")
    try:
        return catalog.Selection(
            kernel=kernel,
            spec_version=spec_version,
            device=device,
            cohort=cohort,
            rtol=rtol,
            atol=atol,
        )
    except ValidationError:
        raise HTTPException(422, "Supply both finite rtol and atol") from None


def selection_error(error: catalog.SelectionError) -> HTTPException:
    return HTTPException(error.status, {"code": str(error), "cohorts": error.cohorts})


def install_discovery(app: FastAPI, settings: Settings, engine: Engine) -> None:
    if not settings.release_enabled:
        return

    @app.get("/api/v1/releases/kernels")
    def kernels() -> list[dict[str, Any]]:
        with Session(engine) as db:
            return catalog.kernels(db)

    @app.get("/api/v1/releases")
    def releases(
        selection: Annotated[catalog.Selection, Depends(selected)],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        try:
            with Session(engine) as db:
                return catalog.listing(db, selection, limit, offset)
        except catalog.SelectionError as error:
            raise selection_error(error) from None

    @app.get("/api/v1/releases/{identifier}")
    def release(identifier: UUID) -> dict[str, Any]:
        try:
            with Session(engine) as db:
                item, pub = catalog.release_info(db, identifier)
                return detail(db, item, pub)
        except catalog.SelectionError as error:
            raise selection_error(error) from None

    @app.get("/api/v1/leaderboard")
    def leaderboard(
        selection: Annotated[catalog.Selection, Depends(selected)],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        try:
            with Session(engine) as db:
                return catalog.leaderboard(db, selection, limit, offset)
        except catalog.SelectionError as error:
            raise selection_error(error) from None

    @app.get("/api/v1/leaderboard/history")
    def history(selection: Annotated[catalog.Selection, Depends(selected)]) -> dict[str, Any]:
        try:
            with Session(engine) as db:
                return catalog.history(db, selection)
        except catalog.SelectionError as error:
            raise selection_error(error) from None

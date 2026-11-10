from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine

from cannery.settings import Settings
from cannery.web.registration import install_registration
from cannery.web.submissions import install_submission_web

ROOT = Path(__file__).parent
PAGES = {
    "global-best": ("Global Best", "No published results yet."),
    "my-submissions": ("My Submissions", "No submissions yet."),
    "register-kernel": ("Register Kernel", "Register a pinned Git specification."),
    "submit-implementation": (
        "Submit Implementation",
        "Submit pinned Git source for a frozen specification.",
    ),
}


def install_web(app: FastAPI, settings: Settings, engine: Engine) -> None:
    templates = Jinja2Templates(directory=ROOT / "templates")
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/", include_in_schema=False)
    async def home() -> RedirectResponse:
        return RedirectResponse("/global-best")

    def render(request: Request, title: str, message: str) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="page.html",
            context={
                "title": title,
                "message": message,
                "pages": PAGES,
                "admin": settings.dev_identity_enabled and settings.dev_role == "admin",
                "development": settings.dev_identity_enabled,
            },
        )

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    async def admin(request: Request) -> HTMLResponse:
        if not settings.dev_identity_enabled:
            raise HTTPException(401, "Authentication required")
        if settings.dev_role != "admin":
            raise HTTPException(403, "Administrator role required")
        return render(
            request, "Administration", "Worker and storage management is not available yet."
        )

    install_registration(
        app,
        settings,
        engine,
        templates,
        {
            "pages": PAGES,
            "admin": settings.dev_identity_enabled and settings.dev_role == "admin",
            "development": settings.dev_identity_enabled,
        },
    )

    install_submission_web(
        app,
        settings,
        engine,
        templates,
        {
            "pages": PAGES,
            "admin": settings.dev_identity_enabled and settings.dev_role == "admin",
            "development": settings.dev_identity_enabled,
        },
    )

    @app.get("/{page}", response_class=HTMLResponse, include_in_schema=False)
    async def page_view(request: Request, page: str) -> HTMLResponse:
        if page not in PAGES:
            raise HTTPException(404, "Page not found")
        if page == "my-submissions" and not settings.dev_identity_enabled:
            raise HTTPException(401, "Authentication required")
        return render(request, *PAGES[page])

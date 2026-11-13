"""Local-account browser entry points; mutations use the shared auth API."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from cannery.accounts.context import identity, template_context
from cannery.settings import Settings


def install_account_web(
    app: FastAPI, settings: Settings, templates: Jinja2Templates, context: dict[str, object]
) -> None:
    if not settings.auth_enabled:
        return

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    def login(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="accounts.html",
            context={
                **context,
                **template_context(settings),
                "title": "Sign in",
                "login": True,
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/account", response_class=HTMLResponse, include_in_schema=False)
    def account(request: Request) -> HTMLResponse:
        identity(settings)
        return templates.TemplateResponse(
            request=request,
            name="accounts.html",
            context={
                **context,
                **template_context(settings),
                "title": "Account",
                "login": False,
            },
            headers={"Cache-Control": "no-store"},
        )

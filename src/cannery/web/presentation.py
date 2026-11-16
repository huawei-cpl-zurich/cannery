"""Display formatting and commit-pinned upstream links; never modify saved values."""

import math
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi.templating import Jinja2Templates

from cannery.git_intake import GitSource


def number(value: int | float | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if not math.isfinite(value):
        return "undefined"
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        mantissa, exponent = f"{value:.3e}".split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def publication_date(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(UTC).strftime("%d %b %Y")


def source_links(source: dict[str, Any]) -> dict[str, str | None] | None:
    source_ref = GitSource.model_validate(source)
    url = urlsplit(source_ref.url)
    hosts = {"github.com", "gitlab.com", "gitcode.com", "gitee.com"}
    default_port = 443 if url.scheme == "https" else 22
    if url.hostname not in hosts or url.port not in (None, default_port):
        return {"repository": source_ref.url, "code": None} if url.scheme == "https" else None
    path = url.path.rstrip("/").removesuffix(".git")
    if len(path.strip("/").split("/")) < 2:
        return None
    repository = f"https://{url.hostname}{path}"
    prefix = "/-/tree/" if url.hostname == "gitlab.com" else "/tree/"
    code = repository + prefix + source_ref.commit
    if source_ref.path != ".":
        code += "/" + quote(source_ref.path, safe="/")
    return {"repository": repository, "code": code}


def configure(templates: Jinja2Templates) -> None:
    templates.env.filters.update(
        number=number, publication_date=publication_date, source_links=source_links
    )

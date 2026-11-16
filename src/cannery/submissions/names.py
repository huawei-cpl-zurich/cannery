"""Human labels are metadata, never execution or deduplication identity."""

from typing import Annotated
from uuid import UUID

from pydantic import BeforeValidator, StringConstraints, TypeAdapter

from cannery.submissions.models import SubmissionConflict


class ImplementationNameConflict(SubmissionConflict):
    pass


def printable(value: object) -> object:
    if isinstance(value, str) and any(not char.isprintable() for char in value):
        raise ValueError("Implementation names must contain only printable characters")
    return value


ImplementationName = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=80),
    BeforeValidator(printable),
]


NAME = TypeAdapter(ImplementationName | None)


def label(name: str | None, identifier: UUID) -> str:
    return name if name is not None else f"Implementation {str(identifier)[:8]}"


def check_name(stored: str | None, supplied: str | None) -> None:
    if supplied is not None and stored != supplied:
        raise ImplementationNameConflict(
            "This source already has a submission with a fixed implementation name; "
            "use the existing submission."
        )

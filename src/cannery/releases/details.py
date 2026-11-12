"""Public projection of the immutable numerical attempt bound to a release."""

from typing import Any

from sqlalchemy.orm import Session

from cannery.numerics.comparison import Comparison
from cannery.numerics.state import NumericalResult
from cannery.releases.catalog import describe, manifest_of
from cannery.releases.models import Publication, Release


def detail(db: Session, release: Release, publication: Publication) -> dict[str, Any]:
    manifest = manifest_of(release)
    result = db.get(NumericalResult, manifest.numerical_attempt)
    cases = []
    if result and result.report:
        for case in result.report["cases"]:
            cases.append(
                {
                    "id": case["id"],
                    "input_unchanged": case["input_unchanged"],
                    "spec_default": Comparison(**case["spec_default"]).as_dict(),
                    "submission_requested": Comparison(**case["submission_requested"]).as_dict(),
                }
            )
    return describe(release, publication) | {"numerical_cases": cases}

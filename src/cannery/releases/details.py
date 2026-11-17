"""Public projection of the immutable numerical attempt bound to a release."""

from typing import Any

from sqlalchemy.orm import Session

from cannery.numerics.comparison import Comparison
from cannery.numerics.state import NumericalResult
from cannery.releases.catalog import describe, manifest_of
from cannery.releases.models import HubSnapshot, Publication, Release
from cannery.releases.tensor_contract import TensorReleaseManifest


def tensor_comparison(data: dict[str, Any]) -> dict[str, Any]:
    if data["metric"] == "torch_close":
        return Comparison(**{k: v for k, v in data.items() if k != "metric"}).as_dict()
    if data["metric"] != "exact":
        raise ValueError("Unsupported published comparison")
    return {
        "rtol": None,
        "atol": None,
        "max_relative_error": None,
        "rms_error": None,
        "nonfinite_mismatches": 0,
        **{
            key: data[key]
            for key in ("metric", "passed", "elements", "mismatches", "max_absolute_error")
        },
    }


def detail(db: Session, release: Release, publication: Publication) -> dict[str, Any]:
    manifest = manifest_of(release)
    result = db.get(NumericalResult, manifest.numerical_attempt)
    cases = []
    if result and result.report:
        for case in result.report["cases"]:
            if isinstance(manifest, TensorReleaseManifest):
                for name in manifest.interface.outputs:
                    output = case["outputs"][name]
                    cases.append(
                        {
                            "id": f"{case['id']}/{name}",
                            "input_unchanged": case["readonly_inputs_unchanged"]
                            and case["mutable_aliases_consistent"],
                            "spec_default": tensor_comparison(output["spec_default"]),
                            "submission_requested": tensor_comparison(
                                output["submission_requested"]
                            ),
                        }
                    )
                continue
            cases.append(
                {
                    "id": case["id"],
                    "input_unchanged": case["input_unchanged"],
                    "spec_default": Comparison(**case["spec_default"]).as_dict(),
                    "submission_requested": Comparison(**case["submission_requested"]).as_dict(),
                }
            )
    snapshot = db.get(HubSnapshot, release.id)
    hub = (
        {
            "repo_id": manifest.kernel,
            "revision": snapshot.revision,
            "release_revision": f"release-{release.id}",
        }
        if snapshot
        else None
    )
    return describe(release, publication) | {"numerical_cases": cases, "hub": hub}

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from test_release_contract import release_data as release_data
from test_tensor_release_assets import assets as assets
from test_tensor_release_contract import tensor_release as tensor_release

from cannery.artifacts import digest
from cannery.releases.catalog import describe, manifest_of
from cannery.releases.client import ReleaseClient
from cannery.releases.document import read_release_document
from cannery.releases.resolution import ResolveRequest
from cannery.releases.tensor_contract import TensorReleaseManifest


@pytest.fixture
def serving(assets):
    store, manifest, _, _ = assets
    raw = manifest.encoded()
    pin = digest(raw)
    requests = []

    def respond(request):
        path = request.url.path
        requests.append(path)
        if path.endswith("/resolve"):
            return httpx.Response(
                200, json={"id": str(manifest.submission_id), "manifest_sha256": pin}
            )
        if path.endswith("/manifest"):
            return httpx.Response(200, content=raw)
        if path.endswith("/package"):
            return httpx.Response(200, content=store.get(manifest.package))
        return httpx.Response(404)

    client = ReleaseClient("https://cannery.example", transport=httpx.MockTransport(respond))
    return client, manifest, requests


def test_tensor_download_verifies_cache_and_repairs_corruption_without_execution(serving, tmp_path):
    client, manifest, requests = serving
    request = ResolveRequest(
        kernel=manifest.kernel, runtime=manifest.cohort.runtime, language="triton"
    )
    cached = client.get(request, tmp_path)
    assert isinstance(cached.manifest, TensorReleaseManifest)
    assert cached.manifest.variants() == manifest.variants()
    assert client.get(request, tmp_path) == cached
    assert sum(path.endswith("/package") for path in requests) == 1
    cached.package_path.write_bytes(b"corrupt")
    assert client.get(request, tmp_path) == cached
    assert sum(path.endswith("/package") for path in requests) == 2
    assert not list(tmp_path.rglob(".download-*"))


@pytest.mark.parametrize(
    "change", [{"shape": [2, 3]}, {"language": "catlass"}, {"runtime": {"bad": "runtime"}}]
)
def test_tensor_resolution_never_treats_unary_shape_as_full_call_domain(serving, tmp_path, change):
    client, manifest, requests = serving
    value = {
        "kernel": manifest.kernel,
        "runtime": manifest.cohort.runtime,
        "language": "triton",
    } | change
    with pytest.raises(ValueError):
        client.get(ResolveRequest.model_validate(value), tmp_path)
    assert not any(path.endswith("/package") for path in requests)


def test_published_tensor_projection_exposes_descriptors_not_inputs_or_oracle(assets):
    _, manifest, _, _ = assets
    release = SimpleNamespace(
        document=manifest.model_dump(mode="json"),
        id=manifest.submission_id,
        candidate_id=manifest.submission_id,
        device=manifest.device,
        language=manifest.language,
        cohort_sha256=manifest.cohort_sha256,
        manifest={"sha256": digest(manifest.encoded())},
        submission=SimpleNamespace(implementation_name="Triton example", namespace="team"),
    )
    pub = SimpleNamespace(published_at=datetime.now(UTC), actor={"namespace": "team"})
    assert manifest_of(release) == manifest
    row = describe(release, pub)
    assert row["abi"] == "tensor-v2" and row["language"] == "triton"
    assert row["interface"] == {"inputs": ["x", "cache"], "outputs": ["updated", "indices"]}
    assert row["cases"]["small"]["outputs"]["indices"]["dtype"] == "int32"
    assert row["specializations"] is None
    assert "data" not in json.dumps(row["cases"])

    from pathlib import Path

    from fastapi import Request
    from fastapi.templating import Jinja2Templates

    from cannery.web.presentation import configure

    templates = Jinja2Templates(directory=Path(__file__).parents[1] / "src/cannery/web/templates")
    configure(templates)
    page = templates.get_template("releases.html").render(
        release=row | {"numerical_cases": []},
        pages={},
        download_snippets={},
        request=Request(
            {"type": "http", "path": "/releases/example", "query_string": b"", "headers": []}
        ),
    )
    assert "Static named tensor domains: 2 cases" in page
    assert "dynamic float32" not in page


def test_version_dispatch_rejects_duplicate_keys_and_noncanonical_documents(assets):
    _, manifest, _, _ = assets
    raw = manifest.encoded()
    assert read_release_document(raw, digest(raw)) == manifest
    for bad in [b"[]", raw + b" ", b'{"schema_version":1,"schema_version":2}', b"x" * 262145]:
        with pytest.raises(ValueError):
            read_release_document(bad, digest(bad))


def test_named_public_numeric_details_keep_integer_verification_exact(assets):
    import math

    from cannery.numerics.comparison_v2 import compare_tensor
    from cannery.numerics.state import NumericalResult
    from cannery.releases.details import detail

    _, manifest, _, _ = assets
    release = SimpleNamespace(
        document=manifest.model_dump(mode="json"),
        id=manifest.submission_id,
        candidate_id=manifest.submission_id,
        device=manifest.device,
        language=manifest.language,
        cohort_sha256=manifest.cohort_sha256,
        manifest={"sha256": digest(manifest.encoded())},
        submission=SimpleNamespace(implementation_name="Triton example", namespace="team"),
    )
    rows = []
    for name, case in manifest.cases.items():
        outputs = {}
        for output, descriptor in case.outputs.items():
            raw = bytes(math.prod(descriptor.shape) * (2 if descriptor.dtype == "float16" else 4))
            comparison = compare_tensor(raw, raw, descriptor.dtype, rtol=0.001, atol=0.0001)
            outputs[output] = {"spec_default": comparison, "submission_requested": comparison}
        rows.append(
            {
                "id": name,
                "outputs": outputs,
                "readonly_inputs_unchanged": True,
                "mutable_aliases_consistent": True,
            }
        )
    db = SimpleNamespace(
        get=lambda model, _: (
            SimpleNamespace(report={"cases": rows}) if model is NumericalResult else None
        )
    )
    result = detail(
        db, release, SimpleNamespace(published_at=datetime.now(UTC), actor={"namespace": "team"})
    )
    assert len(result["numerical_cases"]) == 4
    indices = next(row for row in result["numerical_cases"] if row["id"] == "small/indices")
    assert indices["spec_default"]["metric"] == "exact"
    assert indices["spec_default"]["max_relative_error"] is None

"""Database-only fixtures for the disposable Compose acceptance project."""

import json
from pathlib import Path

from sqlalchemy import delete, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from cannery.database import make_engine
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree, SnapshotStore
from cannery.registry.freeze import SmokeEvidence, freeze_spec, kernel_key
from cannery.registry.manifest import validate_spec
from cannery.registry.models import SpecVersion
from cannery.settings import Settings
from cannery.submissions.models import Candidate, RequestedTarget, Submission
from cannery.submissions.requests import new_submission

manifest = {
    "schema_version": 1,
    "kernel": {"namespace": "acceptance", "name": "identity"},
    "spec": {
        "version": 1,
        "reference": {"file": "reference.py", "function": "reference"},
        "cases": {"file": "cases.py", "function": "generate_cases"},
        "interface": {"inputs": ["x"], "outputs": ["result"]},
        "validation": {"metric": "torch_close", "default_rtol": 0.001, "default_atol": 0.0001},
    },
}
files = {
    "spec/cannery.yaml": json.dumps(manifest).encode(),
    "spec/reference.py": b"def reference(x): return x\n",
    "spec/cases.py": b"def generate_cases(): yield {'x': 1}\n",
    "spec/environment.lock": b'{"schema_version":1,"python":"3.12","torch":"2.6.0+cpu"}',
}
store = SnapshotStore(Path("/tmp/frozen-spec-fixture"))
binding = store.commit(
    kernel_key("acceptance", "identity"),
    GitSource(url="https://example.invalid/fixture.git", commit="a" * 40, path="spec"),
    SelectedTree(files, dict.fromkeys(files, "100644"), "b" * 40),
)
spec = validate_spec(files, "spec")
# Synthetic evidence exercises DB invariants only. CPU reference execution is
# independently verified by the M1 sandbox acceptance tests.
evidence = SmokeEvidence(
    spec_sha256=spec.content_sha256,
    image_id="sha256:" + "a" * 64,
    python="3.12",
    torch="2.6.0+cpu",
    cases=1,
)
engine = make_engine(Settings())
try:
    frozen = freeze_spec(
        engine,
        spec,
        binding,
        evidence,
        store=store,
        namespace="acceptance",
        allow_new_version=False,
    )
    again = freeze_spec(
        engine,
        spec,
        binding,
        evidence,
        store=store,
        namespace="acceptance",
        allow_new_version=False,
    )
    assert again.id == frozen.id
    for statement in (
        update(SpecVersion).where(SpecVersion.id == frozen.id).values(version=99),
        delete(SpecVersion).where(SpecVersion.id == frozen.id),
    ):
        try:
            with engine.begin() as connection:
                connection.execute(statement)
        except DBAPIError as error:
            assert getattr(error.orig, "sqlstate", None) == "P0001"
        else:
            raise AssertionError("PostgreSQL accepted immutable spec mutation")
    with engine.connect() as connection:
        assert (
            connection.scalar(select(SpecVersion.version).where(SpecVersion.id == frozen.id)) == 1
        )
    request = new_submission(
        engine,
        "acceptance",
        frozen.id,
        binding.source,
        RequestedTarget(device="a5", language="catlass"),
    )
    # Database-only candidate fixture; actual intake/provenance is tested separately.
    with Session(engine) as session, session.begin():
        session.add(
            Candidate(
                id=request,
                binding_id=request,
                binding={},
                manifest={},
                snapshot={},
                content_sha256="c" * 64,
                file_sha256={},
            )
        )
    for statement in (
        update(Submission).where(Submission.id == request).values(target={}),
        delete(Submission).where(Submission.id == request),
        update(Candidate).where(Candidate.id == request).values(binding={"changed": True}),
        delete(Candidate).where(Candidate.id == request),
    ):
        try:
            with engine.begin() as connection:
                connection.execute(statement)
        except DBAPIError as error:
            assert getattr(error.orig, "sqlstate", None) == "P0001"
        else:
            raise AssertionError("PostgreSQL accepted immutable submission mutation")
    print("PASS: PostgreSQL frozen spec, submission and candidate mutation guards")
finally:
    engine.dispose()

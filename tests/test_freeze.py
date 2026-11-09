import pytest
from spec_fixtures import spec_files
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from cannery.database import Base
from cannery.git_intake import GitSource
from cannery.git_intake.snapshots import SelectedTree, SnapshotStore
from cannery.registry.freeze import SmokeEvidence, freeze_spec, kernel_key
from cannery.registry.manifest import validate_spec
from cannery.registry.models import SpecConflict, SpecVersion


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'specs.db'}")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def store(tmp_path):
    return SnapshotStore(tmp_path / "snapshots")


def frozen_arguments(store, version=1, reference=b"def reference(x): return x\n"):
    spec = validate_spec(spec_files(version, reference), "spec")
    files = spec_files(version, reference)
    binding = store.commit(
        kernel_key("team", "identity"),
        GitSource(url="https://example.com/repo.git", commit="a" * 40, path="spec"),
        SelectedTree(files, dict.fromkeys(files, "100644"), "c" * 40),
    )
    evidence = SmokeEvidence(
        spec_sha256=spec.content_sha256,
        image_id="sha256:" + "a" * 64,
        python="3.12",
        torch="2.6.0+cpu",
        cases=1,
    )
    return spec, binding, evidence


def test_idempotent_v1_and_new_version_do_not_mutate_old_spec(engine, store) -> None:
    first = freeze_spec(
        engine, *frozen_arguments(store), store=store, namespace="team", allow_new_version=False
    )
    again = freeze_spec(
        engine, *frozen_arguments(store), store=store, namespace="team", allow_new_version=False
    )
    assert again.id == first.id
    second = freeze_spec(
        engine,
        *frozen_arguments(store, 2, b"def reference(x): return x + 1\n"),
        store=store,
        namespace="team",
        allow_new_version=True,
    )
    assert first.version == 1 and second.version == 2
    with Session(engine) as session:
        old = session.get(SpecVersion, first.id)
        assert old.content_sha256 == first.content_sha256
        old.version = 99
        with pytest.raises(SpecConflict, match="cannot be updated"):
            session.commit()


def test_new_versions_require_admin_and_changed_content(engine, store) -> None:
    freeze_spec(
        engine, *frozen_arguments(store), store=store, namespace="team", allow_new_version=False
    )
    for args, admin in (
        (frozen_arguments(store, 2), True),
        (frozen_arguments(store, 2, b"changed"), False),
        (frozen_arguments(store, 3, b"changed"), True),
    ):
        with pytest.raises(SpecConflict):
            freeze_spec(engine, *args, store=store, namespace="team", allow_new_version=admin)
    with Session(engine) as session:
        assert len(session.scalars(select(SpecVersion)).all()) == 1


def test_namespace_and_runtime_mismatch_fail_without_database_side_effects(engine, store) -> None:
    args = frozen_arguments(store)
    with pytest.raises(SpecConflict, match="namespace"):
        freeze_spec(engine, *args, store=store, namespace="other", allow_new_version=True)
    spec, binding, evidence = args
    evidence = evidence.model_copy(update={"torch": "other"})
    with pytest.raises(SpecConflict, match="runtime"):
        freeze_spec(
            engine, spec, binding, evidence, store=store, namespace="team", allow_new_version=True
        )
    with Session(engine) as session:
        assert session.scalars(select(SpecVersion)).all() == []


def test_mismatched_snapshot_or_smoke_content_is_rejected(engine, store):
    one, _, evidence = frozen_arguments(store)
    two, binding, _ = frozen_arguments(store, 2, b"different reference")
    with pytest.raises(SpecConflict, match="retained snapshot"):
        freeze_spec(
            engine, one, binding, evidence, store=store, namespace="team", allow_new_version=True
        )
    with pytest.raises(SpecConflict, match="retained snapshot"):
        freeze_spec(
            engine, two, binding, evidence, store=store, namespace="team", allow_new_version=True
        )
    with Session(engine) as session:
        assert session.scalars(select(SpecVersion)).all() == []

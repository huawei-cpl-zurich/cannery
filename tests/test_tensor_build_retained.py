from types import SimpleNamespace

import pytest
from test_baseline_ingress import ingress as ingress
from test_tensor_numerical_retained import retained  # noqa: F401

from cannery.builds.coordinator import retained_tensor_spec
from cannery.builds.executor import ExecutionFailure
from cannery.registry.manifest import SpecError


def test_build_reads_exact_frozen_tensor_without_rerunning_oracle(retained):  # noqa: F811
    engine, key, executor, control, tensor, _ = retained
    if not tensor:
        with pytest.raises(SpecError):
            retained_tensor_spec(engine, key, executor)
        return
    first = retained_tensor_spec(engine, key, executor)
    assert first == retained_tensor_spec(engine, key, executor)
    assert first.suite.generator == "sha256-tensor-v2"
    assert first.contract == control["contract"]
    assert control["calls"] == 1


def test_build_rejects_missing_submission(retained):  # noqa: F811
    from uuid import uuid4

    engine, _, executor, _, _, _ = retained
    with pytest.raises(ExecutionFailure, match="retained_spec_missing"):
        retained_tensor_spec(engine, uuid4(), executor)


def test_build_rejects_tampered_snapshot_before_compilation(retained):  # noqa: F811
    engine, key, executor, control, tensor, files = retained
    changed = dict(files)
    changed["spec/reference.py"] += b"\n# changed\n"
    executor.snapshots = SimpleNamespace(read=lambda _: SimpleNamespace(files=changed))
    expected = ExecutionFailure if tensor else SpecError
    with pytest.raises(expected):
        retained_tensor_spec(engine, key, executor)
    assert control["calls"] == 1


def test_build_retains_snapshot_outage_for_retry_without_oracle_work(retained):  # noqa: F811
    engine, key, executor, control, _, _ = retained

    def unavailable(binding):
        raise OSError("snapshot storage temporarily unavailable")

    executor.snapshots = SimpleNamespace(read=unavailable)
    with pytest.raises(OSError):
        retained_tensor_spec(engine, key, executor)
    assert control["calls"] == 1

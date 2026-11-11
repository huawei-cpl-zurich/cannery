import base64
import json
from uuid import uuid4

import pytest
from test_baseline_format import fixture
from test_build_executor import Memory
from test_recipe_spec import recipe_spec

from cannery.artifacts import ArtifactError
from cannery.numerics.baseline_format import baseline_key
from cannery.numerics.oracle_job import OracleFailure, OracleJob, OraclePending


class Docker:
    def __init__(self):
        self.info = None
        self.calls = []
        self.creates = self.starts = 0
        self.fault = None
        self.reply = b""

    def __call__(self, args, limit=1024 * 1024):
        self.calls.append(args)
        if args[0] == "ps":
            return 0, b"id\n" if self.info else b""
        if args[0] == "inspect":
            return 0, json.dumps([self.info]).encode()
        if args[0] == "create":
            self.creates += 1
            if self.fault == "create_missing":
                raise OraclePending("lost create request")
            self.info = {
                "Id": "id",
                "Config": {
                    "Labels": {
                        "cannery.oracle.owner": args[args.index("--label") + 1].split("=")[1]
                    }
                },
                "State": {"Status": "created", "ExitCode": 0},
            }
            if self.fault == "create":
                raise OraclePending("lost create reply")
        if args[0] == "start":
            self.starts += 1
            self.info["State"]["Status"] = "running"
            if self.fault == "start":
                raise OraclePending("lost start reply")
        if args[0] == "logs":
            return 0, self.reply
        if args[0] == "rm":
            if self.fault == "cleanup":
                return 1, b""
            self.info = None
        return 0, b""

    def complete(self, outputs):
        self.info["State"]["Status"] = "exited"
        self.reply = json.dumps(
            {"outputs": {k: base64.b64encode(v).decode() for k, v in outputs.items()}}
        ).encode()


@pytest.fixture
def setup_job(tmp_path, monkeypatch):
    spec, suite, outputs = fixture()
    files, _, _ = recipe_spec()
    docker = Docker()
    monkeypatch.setattr("cannery.numerics.oracle_job.docker", docker)
    memory = Memory()
    attempt = uuid4()

    def job(fence=lambda: None):
        return OracleJob(
            tmp_path,
            "test-baseline-control",
            baseline_key(spec.content_sha256, suite, spec.environment),
            attempt,
            "sha256:" + "a" * 64,
            memory,
            fence,
        )

    def progress():
        return job().progress(spec, suite, files, "spec")

    return job, progress, docker, memory, spec, suite, outputs, files


def complete(setup_job):
    job, progress, docker, memory, spec, suite, outputs, files = setup_job
    for _ in range(2):
        with pytest.raises(OraclePending):
            progress()
    docker.complete(outputs)
    return job, progress, docker, memory


def test_detached_lifecycle_restarts_controller_without_rerunning_reference(setup_job):
    job, progress, docker, memory = complete(setup_job)
    reference = progress()
    assert reference.repository == "private"
    assert progress() == reference
    assert docker.creates == docker.starts == 1 and docker.info is None
    args = next(c for c in docker.calls if c[0] == "create")
    assert "--device" not in args and args[args.index("--network") + 1] == "none"
    assert any("volume-subpath=" in a and a.endswith("/input") for a in args)
    assert "--read-only" in args and "--cap-drop" in args
    assert (job().journal / "artifact.json").exists()
    assert set((job().journal / "input").iterdir()) == {job().journal / "input/plan.json"}


@pytest.mark.parametrize("fault", ["create", "start"])
def test_transport_loss_observes_same_created_or_running_container(setup_job, fault):
    _, progress, docker, _, _, _, outputs, _ = setup_job
    docker.fault = fault
    with pytest.raises(OraclePending):
        progress()
    with pytest.raises(OraclePending):
        progress()
    docker.fault = None
    docker.complete(outputs)
    assert progress().sha256
    assert docker.creates == docker.starts == 1


def test_ambiguous_missing_container_is_never_resubmitted(setup_job):
    _, progress, docker, *_ = setup_job
    docker.fault = "create_missing"
    with pytest.raises(OraclePending):
        progress()
    docker.fault = None
    with pytest.raises(OraclePending, match="Ambiguous"):
        progress()
    assert docker.creates == 1 and docker.starts == 0


def test_private_upload_failure_reuses_completed_reference_output(setup_job):
    _, progress, docker, memory = complete(setup_job)
    original = memory.put

    def unavailable(*args):
        raise ArtifactError("injected disconnect")

    memory.put = unavailable
    with pytest.raises(ArtifactError):
        progress()
    memory.put = original
    assert progress().sha256 and docker.starts == 1


def test_cleanup_failure_preserves_private_checkpoint_without_new_forward(setup_job):
    job, progress, docker, memory = complete(setup_job)
    docker.fault = "cleanup"
    with pytest.raises(OraclePending, match="cleanup"):
        progress()
    assert (job().journal / "artifact.json").exists()
    puts = memory.puts
    docker.fault = None
    assert progress().sha256 and memory.puts == puts and docker.starts == 1


def test_corrupt_cached_output_blocks_without_rebuilding(setup_job):
    _, progress, docker, memory = complete(setup_job)
    ref = progress()
    memory.data[ref.sha256] = b"corrupted"
    with pytest.raises(ArtifactError):
        progress()
    assert docker.starts == 1 and docker.creates == 1


def test_configuration_change_and_ownership_conflict_fail_before_effects(setup_job):
    job, progress, docker, _, spec, suite, _, files = setup_job
    with pytest.raises(OraclePending):
        progress()
    old = docker.creates
    docker.info["Config"]["Labels"]["cannery.oracle.owner"] = "different"
    with pytest.raises(OracleFailure, match="ownership"):
        progress()
    other = job()
    other.image = "sha256:" + "b" * 64
    with pytest.raises(OracleFailure, match="configuration"):
        other.progress(spec, suite, files, "spec")
    assert docker.creates == old and not any(c[0] == "rm" for c in docker.calls)


def test_lost_lease_prevents_new_docker_dispatch(setup_job):
    job, _, docker, _, spec, suite, _, files = setup_job

    def lost():
        raise OraclePending("lease lost")

    with pytest.raises(OraclePending, match="lease lost"):
        job(lost).progress(spec, suite, files, "spec")
    assert not docker.calls

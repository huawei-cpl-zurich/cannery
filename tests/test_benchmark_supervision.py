import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from test_build_sandbox import profile, setup

from cannery.builds.sandbox import job_script


def hardware(monkeypatch, *, changed=False):
    original = Path.read_bytes

    def read(path):
        if str(path) == "/usr/local/Ascend/driver/version.info":
            return b"driver1"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    return {
        "hardware_id": hashlib.sha256(
            json.dumps({"Serial Number": "serial1", "DDie ID": "die1"}, sort_keys=True).encode()
        ).hexdigest(),
        "driver": hashlib.sha256(b"driver1").hexdigest(),
    }


def run_supervisor(config, token, identity):
    namespace = {}
    try:
        exec(
            compile(
                job_script(
                    config,
                    token,
                    "runtime",
                    runtime_program="pass",
                    capture_runtime=True,
                    benchmark=identity,
                ),
                "<supervisor>",
                "exec",
            ),
            namespace,
        )
    finally:
        if namespace.get("held"):
            namespace["held"].close()


@pytest.mark.parametrize("fault", [None, "post_health", "identity"])
def test_benchmark_checks_physical_identity_health_before_and_after(
    tmp_path, monkeypatch, capsys, fault
):
    config, token, calls = setup(tmp_path, monkeypatch)
    identity = hardware(monkeypatch)
    delegate = subprocess.run
    launched, board = False, 0

    def run(args, **kwargs):
        nonlocal launched, board
        if args[0] == "npu-smi" and args[args.index("-t") + 1] == "board":
            board += 1
            serial = "changed" if board > 1 and fault == "identity" else "serial1"
            return SimpleNamespace(
                returncode=0, stdout=f"Serial Number : {serial}\nDDie ID : die1\n"
            )
        if args[:2] == ["docker", "run"]:
            launched = True
        if launched and fault == "post_health" and args[0] == "npu-smi":
            return SimpleNamespace(returncode=0, stdout="Health Status : ERROR")
        return delegate(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    if fault == "post_health":
        with pytest.raises(SystemExit) as failure:
            run_supervisor(config, token, identity)
        assert failure.value.code == 75
    elif fault == "identity":
        with pytest.raises(ValueError, match="cohort changed"):
            run_supervisor(config, token, identity)
    else:
        run_supervisor(config, token, identity)
        assert len(capsys.readouterr().out.strip()) == 64
        health = [c for c in calls if c[0] == "npu-smi"]
        assert len(health) == 8 and board == 2
    assert not (Path(config.workspace) / "device-1.owner").exists()
    docker = next(c for c in calls if c[:2] == ["docker", "run"])
    assert docker[docker.index("--ulimit") + 1] == "fsize=134217728:134217728"


def test_identity_phase_is_bounded_and_never_launches_container(tmp_path, monkeypatch, capsys):
    config, token = profile(tmp_path / "worker"), uuid4()
    expected = hardware(monkeypatch)

    def run(args, **kwargs):
        assert args == ["npu-smi", "info", "-t", "board", "-i", "1"]
        assert kwargs["timeout"] == 12 and kwargs["check"]
        return SimpleNamespace(stdout="Serial Number : serial1\nDDie ID : die1\n")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit) as result:
        exec(compile(job_script(config, token, "identify"), "<identify>", "exec"), {})
    assert result.value.code == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_performance_settings_cannot_weaken_other_job_phases(tmp_path):
    config = profile(tmp_path / "worker")
    for phase, capture in [("compile", True), ("runtime", False), ("identify", False)]:
        with pytest.raises(ValueError):
            job_script(
                config,
                uuid4(),
                phase,
                capture_runtime=capture,
                benchmark={"hardware_id": "x", "driver": "x"},
            )

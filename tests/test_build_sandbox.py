import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cannery.builds.sandbox import WorkerProfile, job_script


def profile(root):
    return WorkerProfile(
        target="test-a5-host",
        runtime="cann",
        workspace=str(root),
        image="sha256:" + "a" * 64,
        npu=1,
        device_group=1000,
        readonly_mounts=("/usr/local/Ascend/driver",),
        compiler="test",
        fingerprint={
            "machine": "aarch64",
            "python": "3.12",
            "torch": "2.9",
            "torch_npu": "2.9",
            "cann": "9.3",
            "soc": "Ascend950DT",
        },
    )


def supervise(config, token, phase):
    namespace = {}
    try:
        exec(compile(job_script(config, token, phase), "<supervisor>", "exec"), namespace)
    finally:
        # The real supervisor is a process: emulate its descriptor cleanup on exit.
        if namespace.get("held"):
            namespace["held"].close()


def execute(config, token, phase):
    with pytest.raises(SystemExit) as result:
        supervise(config, token, phase)
    assert result.value.code == 0


def setup(tmp_path, monkeypatch, fault=None):
    config, token, calls = profile(tmp_path / "worker"), uuid4(), []
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/cann/current")
    monkeypatch.setattr("resource.setrlimit", lambda *_: None)
    execute(config, token, "prepare")

    def run(args, **kwargs):
        calls.append(args)
        if args[0] == "npu-smi":
            kind = args[args.index("-t") + 1]
            if fault == "timeout":
                raise subprocess.TimeoutExpired(args, 12)
            text = {
                "health": "Health Status : OK",
                "usages": "HBM Usage Rate(%) : 4\nNPU Utilization(%) : 0",
                "proc-mem": "No process in device",
                "current-fault-event": "Fault Event Count : 0",
            }[kind]
            if fault == kind:
                text = "unavailable or busy"
            return SimpleNamespace(returncode=0, stdout=text)
        if args[:2] == ["docker", "run"]:
            assert kwargs["timeout"] == 180
            (Path(config.workspace) / token.hex / "output/kernel.so").write_bytes(b"CPU simulation")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", run)
    return config, token, calls


def test_device_less_compile_has_no_network_secrets_or_host_home(tmp_path, monkeypatch):
    config, token, calls = setup(tmp_path, monkeypatch)
    # The actual wrapper exits normally after the bounded Docker command.
    supervise(config, token, "compile")
    args = next(call for call in calls if call[:2] == ["docker", "run"])
    assert "--device" not in args and "--privileged" not in args
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("--user") + 1] == "10001:10001"
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert "--read-only" in args and args[args.index("--log-driver") + 1] == "none"
    assert all("docker.sock" not in arg and "TOKEN" not in arg for arg in args)
    assert "--pull" in args and "never" in args


@pytest.mark.parametrize(
    "fault", ["health", "usages", "proc-mem", "current-fault-event", "timeout"]
)
def test_unhealthy_busy_or_unresponsive_device_never_launches(tmp_path, monkeypatch, fault):
    config, token, calls = setup(tmp_path, monkeypatch, fault)
    with pytest.raises(SystemExit) as result:
        supervise(config, token, "runtime")
    assert result.value.code == 75
    assert not any(call[:2] == ["docker", "run"] for call in calls)
    assert not (Path(config.workspace) / "device-1.owner").exists()


def test_runtime_exposes_one_physical_device_and_releases_owned_lease(tmp_path, monkeypatch):
    config, token, calls = setup(tmp_path, monkeypatch)
    supervise(config, token, "runtime")
    args = next(call for call in calls if call[:2] == ["docker", "run"])
    devices = [args[index + 1] for index, arg in enumerate(args) if arg == "--device"]
    assert "/dev/davinci1" in devices and "/dev/davinci0" not in devices
    assert "ASCEND_RT_VISIBLE_DEVICES=0" in args
    assert args[args.index("--tmpfs") + 1] == "/tmp:rw,exec,nosuid,nodev,size=512m"
    assert not (Path(config.workspace) / "device-1.owner").exists()
    assert calls[-1][:3] == ["docker", "rm", "-f"]


def test_stale_device_owner_requires_explicit_cleanup(tmp_path, monkeypatch):
    config, token, calls = setup(tmp_path, monkeypatch)
    marker = Path(config.workspace) / "device-1.owner"
    marker.write_text(token.hex)
    with pytest.raises(SystemExit) as result:
        supervise(config, token, "runtime")
    assert result.value.code == 75 and marker.exists()
    execute(config, token, "cleanup")
    assert not marker.exists()


def test_profile_cannot_mount_broad_or_relative_paths(tmp_path):
    for path in (
        "/",
        "/home",
        "../secret",
        "/root/../secret",
        "/var/lib,extra",
        "/var/run/docker.sock",
        "/home/user/.ssh",
    ):
        with pytest.raises(ValueError):
            profile(tmp_path).model_validate(
                profile(tmp_path).model_dump() | {"readonly_mounts": [path]}
            )


def test_prepare_makes_input_readable_despite_private_supervisor_umask(tmp_path):
    import os

    config, token = profile(tmp_path / "worker"), uuid4()
    previous = os.umask(0o077)
    try:
        execute(config, token, "prepare")
    finally:
        os.umask(previous)
    root = Path(config.workspace) / token.hex
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "input").stat().st_mode & 0o777 == 0o755


def test_container_output_symlink_cannot_redirect_host_log_writes(tmp_path, monkeypatch):
    config, token, calls = setup(tmp_path, monkeypatch)
    victim = tmp_path / "host-secret"
    victim.write_text("preserve this host file")
    (Path(config.workspace) / token.hex / "output/runtime.log").symlink_to(victim)
    supervise(config, token, "runtime")
    assert victim.read_text() == "preserve this host file"


def test_failed_removal_after_timeout_preserves_device_owner_and_scratch(tmp_path, monkeypatch):
    config, token, calls = setup(tmp_path, monkeypatch)
    original = subprocess.run

    def broken(args, **kwargs):
        if args[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(args, 180)
        if args[:3] == ["docker", "rm", "-f"] or args[:2] == ["docker", "ps"]:
            return SimpleNamespace(returncode=1, stdout="daemon unavailable")
        return original(args, **kwargs)

    monkeypatch.setattr("subprocess.run", broken)
    with pytest.raises(RuntimeError, match="preserve device ownership"):
        supervise(config, token, "runtime")
    root = Path(config.workspace)
    assert (root / "device-1.owner").read_text() == token.hex
    assert (root / token.hex).exists()
    with pytest.raises(RuntimeError, match="preserve device ownership"):
        execute(config, token, "cleanup")
    assert (root / "device-1.owner").exists() and (root / token.hex).exists()


def test_profile_roundtrip_preserves_generated_job_contract(tmp_path):
    import json

    original = profile(tmp_path)
    restored = WorkerProfile.model_validate_json(
        json.dumps(original.model_dump(mode="json"), sort_keys=True)
    )
    token = uuid4()
    assert job_script(original, token, "runtime") == job_script(restored, token, "runtime")

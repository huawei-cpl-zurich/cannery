import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_a3_contract import a3_profile
from test_build_sandbox import setup, supervise

HEALTH = {
    "health": "Health Status : OK\nError Code : NA\nError Information : NA\n",
    "usages": "HBM Usage Rate(%) : 5\nAicore Usage Rate(%) : 0\nAivector Usage Rate(%) : 0\n",
    "proc-mem": "No process in device",
    "ecc": "HBM Double Bit Error Count : 0\nHBM Double Bit Aggregate Total Err Cnt : 0\n",
}


@pytest.mark.parametrize(
    "fault", [None, "health", "error", "memory", "vector", "ecc", "missing", "timeout"]
)
def test_a3_health_is_bounded_and_blocks_dispatch_on_faults(tmp_path, monkeypatch, fault):
    _, token, calls = setup(tmp_path, monkeypatch)
    config = a3_profile(tmp_path)
    original = subprocess.run

    def run(args, **kwargs):
        if args[0] != "npu-smi":
            return original(args, **kwargs)
        calls.append(args)
        assert args[-2:] == ["-c", "0"] and kwargs["timeout"] == 12
        if fault == "timeout":
            raise subprocess.TimeoutExpired(args, 12)
        kind = args[args.index("-t") + 1]
        text = HEALTH[kind]
        if fault == "health" and kind == "health":
            text = text.replace("OK", "Warning")
        if fault == "error" and kind == "health":
            text = text.replace("Error Code : NA", "Error Code : 123")
        if fault == "memory" and kind == "usages":
            text = text.replace(" : 5", " : 50")
        if fault == "vector" and kind == "usages":
            text = text.replace("Aivector Usage Rate(%) : 0", "Aivector Usage Rate(%) : 1")
        if fault == "ecc" and kind == "ecc":
            text = text.replace("Count : 0", "Count : 1")
        if fault == "missing" and kind == "usages":
            text = "HBM Usage Rate(%) : 5"
        return SimpleNamespace(returncode=0, stdout=text)

    monkeypatch.setattr(subprocess, "run", run)
    if fault:
        with pytest.raises(SystemExit) as error:
            supervise(config, token, "runtime")
        assert error.value.code == 75
        assert not any(args[:2] == ["docker", "run"] for args in calls)
    else:
        supervise(config, token, "runtime")
        runtime = next(args for args in calls if args[:2] == ["docker", "run"])
        assert "/dev/devmm_svm" in runtime and "/dev/ummu/tid" not in runtime
        assert f"ASCEND_RT_VISIBLE_DEVICES={config.npu}" in runtime
    assert not (Path(config.workspace) / f"device-{config.npu}.owner").exists()


@pytest.mark.parametrize(
    "die", ["00000001 00000002 00000003 00000004 00000005", "NA", "invalid", "00000000 " * 5]
)
def test_a3_physical_identity_requires_card_serial_and_valid_vdie(
    tmp_path, monkeypatch, capsys, die
):
    config = a3_profile(tmp_path)
    from uuid import uuid4

    read = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: b"driver" if str(p) == "/usr/local/Ascend/driver/version.info" else read(p),
    )

    def run(args, **kwargs):
        assert args[:4] == ["npu-smi", "info", "-t", "board"]
        return SimpleNamespace(
            stdout=f"VDie ID : {die}\n" if "-c" in args else "Serial Number : fixture-card\n"
        )

    monkeypatch.setattr(subprocess, "run", run)
    if die.startswith("00000001"):
        with pytest.raises(SystemExit) as error:
            supervise(config, uuid4(), "identify")
        assert error.value.code == 0
        result = json.loads(capsys.readouterr().out)
        assert set(result) == {"hardware_id", "driver"} and len(result["hardware_id"]) == 64
    else:
        with pytest.raises(ValueError):
            supervise(config, uuid4(), "identify")

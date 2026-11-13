"""Render the actual overlay merge without starting services or exposing secrets."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_a3_overlay_is_opt_in_and_keeps_existing_workers_on_bridge():
    environment = os.environ | {
        "COMPOSE_PROJECT_NAME": "cannery-m7b-config-test",
        "CANNERY_OPERATOR_UID": "1000",
        "CANNERY_OPERATOR_GID": "1000",
        "CANNERY_DOCKER_GID": "998",
        "CANNERY_OPERATOR_HOME": "/home/operator",
        "CANNERY_REMOTE_SKILL_ROOT": "/installed/remote-access",
        "CANNERY_REMOTE_AGENT_ROOT": "/installed/remote-agent",
        "CANNERY_WORKER_PROFILE": "/operator/a5.json",
        "CANNERY_A3_WORKER_PROFILE": "/operator/a3.json",
        "CANNERY_LOADER_WHEEL": "/operator/loader-a5.whl",
        "CANNERY_BUILD_LOADER_SHA256": "a" * 64,
        "CANNERY_A3_LOADER_WHEEL": "/operator/loader-a3.whl",
        "CANNERY_A3_LOADER_SHA256": "b" * 64,
        "CANNERY_ARTIFACT_READ_TOKEN": "config-read-placeholder",
        "CANNERY_ARTIFACT_WRITE_TOKEN": "config-write-placeholder",
    }
    arguments = ["docker", "compose"]
    for name in [
        "compose.yaml",
        "compose.artifacts.yaml",
        "compose.builds.yaml",
        "compose.numerics.yaml",
        "compose.benchmarks.yaml",
        "compose.releases.yaml",
        "compose.a3.yaml",
    ]:
        arguments += ["-f", str(ROOT / name)]
    result = subprocess.run(
        arguments + ["config", "--format", "json"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert not any(name.endswith("-a3") for name in json.loads(result.stdout)["services"])
    result = subprocess.run(
        arguments + ["--profile", "a3", "config", "--format", "json"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    services = json.loads(result.stdout)["services"]
    for name in ["builder", "validator", "benchmarker"]:
        assert "network_mode" not in services[name]
        assert services[name]["environment"]["CANNERY_BUILD_LOADER_SHA256"] == "a" * 64
    for name in ["builder-a3", "validator-a3", "benchmarker-a3"]:
        node = services[name]
        assert node["profiles"] == ["a3"] and node["network_mode"] == "host"
        assert node["environment"]["CANNERY_BUILD_LOADER_SHA256"] == "b" * 64
        mounts = {m["target"]: m for m in node["volumes"]}
        assert mounts["/etc/cannery/worker.json"]["source"] == "/operator/a3.json"
        assert mounts["/installed/remote-agent"]["read_only"]
        assert mounts["/var/lib/cannery/builds"]["source"] == "build-control"
    for name in ["postgres", "artifacts"]:
        assert all(port["host_ip"] == "127.0.0.1" for port in services[name]["ports"])

"""Retained local Docker oracle jobs; completed reference work is never restarted."""

import base64
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any
from uuid import UUID

from cannery.artifacts import ArtifactRef, ArtifactStore, digest, verified
from cannery.builds.remote import durable_write
from cannery.numerics.baseline_format import (
    MAX_PACKAGE_BYTES,
    baseline_key,
    pack_baseline,
    unpack_baseline,
)
from cannery.numerics.baseline_v2 import baseline_key_v2, pack_baseline_v2, unpack_baseline_v2
from cannery.numerics.oracle_worker import worker_source
from cannery.numerics.oracle_worker_v2 import worker_source_v2
from cannery.numerics.recipes import RecipeSuite, parse_recipes
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.registry.manifest import RecipeEntrypoint, ValidatedSpec, validate_spec
from cannery.registry.tensor_manifest import ValidatedTensorSpec, validate_tensor_spec

CPU_FLAGS = (
    "create --init --rm --pull never --network none --read-only --user 0:0 "
    "--cap-drop ALL --cap-add SETUID --cap-add SETGID --cap-add KILL "
    "--security-opt no-new-privileges --memory 2g --cpus 1 --pids-limit 64 --log-driver none "
    "--tmpfs /work:rw,noexec,nosuid,nodev,size=32m,mode=755 "
    "--tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777"
).split()


class OraclePending(Exception):
    """Observe the same container/checkpoint; never resubmit uncertain work."""


class OracleFailure(ValueError):
    pass


def docker(arguments: list[str], limit: int = 1024 * 1024) -> tuple[int, bytes]:
    try:
        with TemporaryFile() as output, TemporaryFile() as errors:
            result = subprocess.run(
                ["docker", *arguments], stdout=output, stderr=errors, timeout=20
            )
            if output.tell() > limit or errors.tell() > 65536:
                raise OraclePending("Docker reply exceeded budget")
            output.seek(0)
            return result.returncode, output.read(limit + 1)
    except (OSError, subprocess.TimeoutExpired):
        raise OraclePending("Docker operation unavailable; observe retained job") from None


class OracleJob:
    def __init__(
        self,
        root: Path,
        volume: str,
        key: str,
        attempt: UUID,
        image: str,
        artifacts: ArtifactStore,
        fence: Callable[[], None],
    ) -> None:
        if (
            re.fullmatch(r"[a-f0-9]{64}", key) is None
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", volume) is None
            or re.fullmatch(r"sha256:[a-f0-9]{64}", image) is None
        ):
            raise ValueError("Invalid trusted oracle job configuration")
        if os.geteuid() == 10001:
            raise ValueError("Oracle controller UID must differ from reserved author UID 10001")
        self.key, self.attempt, self.volume, self.image = key, attempt, volume, image
        self.artifacts, self.fence = artifacts, fence
        self.relative = key + "/" + attempt.hex
        self.journal = root / self.relative
        self.journal.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.name = "cannery-oracle-" + attempt.hex

    def inspect(self) -> dict[str, Any] | None:
        code, data = docker(
            [
                "ps",
                "-a",
                "--no-trunc",
                "--filter",
                "name=^/" + self.name + "$",
                "--format",
                "{{.ID}}",
            ]
        )
        if code:
            raise OraclePending("Docker inventory unavailable")
        ids = data.decode().split()
        if not ids:
            return None
        if len(ids) != 1:
            raise OracleFailure("Ambiguous oracle container")
        code, data = docker(["inspect", ids[0]])
        if code:
            raise OraclePending("Oracle container observation unavailable")
        info: dict[str, Any] = json.loads(data)[0]
        if info.get("Config", {}).get("Labels", {}).get("cannery.oracle.owner") != self.attempt.hex:
            raise OracleFailure("Oracle container ownership mismatch")
        return info

    def cleanup(self) -> None:
        info = self.inspect()
        if info is not None:
            self.fence()
            docker(["rm", "-f", info["Id"]])
            if self.inspect() is not None:
                raise OraclePending("Oracle container cleanup pending")

    def progress(
        self,
        spec: ValidatedSpec,
        suite: RecipeSuite | RecipeSuiteV2,
        files: dict[str, bytes],
        path: str,
    ) -> ArtifactRef:
        if spec.manifest.schema_version == 2:
            frozen = validate_tensor_spec(files, path)
            if frozen.spec != spec or frozen.suite != suite:
                raise OracleFailure("Retained tensor oracle source mismatch")
            return self._progress_tensor(frozen, files, path)
        if not isinstance(suite, RecipeSuite):
            raise OracleFailure("Oracle recipe version mismatch")
        if validate_spec(files, path) != spec:
            raise OracleFailure("Retained oracle source mismatch")
        cases = spec.manifest.spec.cases
        prefix = "" if path == "." else path + "/"
        if (
            not isinstance(cases, RecipeEntrypoint)
            or parse_recipes(files[prefix + cases.recipe_file]) != suite
            or baseline_key(spec.content_sha256, suite, spec.environment) != self.key
        ):
            raise OracleFailure("Oracle recipe identity mismatch")
        plan = {
            "path": path,
            "output_owner": [os.geteuid(), os.getegid()],
            "spec": spec.manifest.spec.model_dump(mode="json"),
            "suite": suite.model_dump(mode="json"),
            "environment": spec.environment.model_dump(mode="json"),
            "files": {k: base64.b64encode(v).decode() for k, v in files.items()},
        }

        def validate(package: bytes, reference: ArtifactRef) -> None:
            unpack_baseline(
                package,
                reference.sha256,
                spec_sha256=spec.content_sha256,
                suite=suite,
                environment=spec.environment,
            )

        return self._progress(
            plan,
            worker_source(checkpoint=True),
            lambda data: self.publish(spec, suite, data),
            validate,
        )

    def _progress(
        self,
        plan: dict[str, Any],
        source: str,
        publish: Callable[[bytes], ArtifactRef],
        validate: Callable[[bytes, ArtifactRef], None],
    ) -> ArtifactRef:
        context = json.dumps(
            {
                "plan_sha256": digest(json.dumps(plan, sort_keys=True).encode()),
                "image": self.image,
                "worker_sha256": digest(source.encode()),
            },
            sort_keys=True,
        ).encode()
        frozen = self.journal / "context.json"
        if frozen.exists() and frozen.read_bytes() != context:
            raise OracleFailure("Active oracle job configuration changed")
        self.fence()
        durable_write(frozen, context)
        saved = self.journal / "artifact.json"
        if saved.exists():
            reference = ArtifactRef.model_validate_json(saved.read_bytes())
            if reference.repository != "private" or reference.revision != reference.sha256:
                raise OracleFailure("Baseline checkpoint must reference private pinned storage")
            package = verified(self.artifacts.get(reference), reference)
            validate(package, reference)
            self.cleanup()
            (self.journal / "output/result.json").unlink(missing_ok=True)
            (self.journal / "input/plan.json").unlink(missing_ok=True)
            return reference
        result = self.journal / "output/result.json"
        if result.exists():
            if (
                result.is_symlink()
                or not result.is_file()
                or result.stat().st_size > MAX_PACKAGE_BYTES * 2
            ):
                raise OracleFailure("Invalid reference output checkpoint")
            reference = publish(result.read_bytes())
            self.cleanup()
            result.unlink()
            (self.journal / "input/plan.json").unlink(missing_ok=True)
            return reference
        info = self.inspect()
        if info is None:
            dispatch = self.journal / "create.dispatch"
            if dispatch.exists():
                raise OraclePending("Ambiguous oracle creation; do not repeat reference")
            source_dir = self.journal / "input"
            source_dir.mkdir(mode=0o755, exist_ok=True)
            source_dir.chmod(0o755)
            (self.journal / "output").mkdir(mode=0o700, exist_ok=True)
            durable_write(source_dir / "plan.json", json.dumps(plan, sort_keys=True).encode())
            (source_dir / "plan.json").chmod(0o444)
            self.fence()
            durable_write(dispatch, b"dispatching\n")
            docker(
                CPU_FLAGS
                + [
                    "--name",
                    self.name,
                    "--label",
                    "cannery.oracle.owner=" + self.attempt.hex,
                    "--mount",
                    f"type=volume,src={self.volume},dst=/input,readonly,volume-subpath={self.relative}/input",
                    "--mount",
                    f"type=volume,src={self.volume},dst=/output,volume-subpath={self.relative}/output",
                    self.image,
                    "python",
                    "-I",
                    "-c",
                    source,
                ]
            )
            raise OraclePending("Oracle container creation submitted")
        state = info["State"]
        if state["Status"] == "created":
            marker = self.journal / "start.dispatch"
            if marker.exists():
                raise OraclePending("Oracle start uncertain; observe without restarting")
            self.fence()
            durable_write(marker, b"dispatching\n")
            docker(["start", info["Id"]])
            raise OraclePending("Oracle reference running")
        if state["Status"] not in ("exited", "dead"):
            raise OraclePending("Oracle reference running")
        if (self.journal / "output/result.json").exists():
            return self._progress(plan, source, publish, validate)
        self.cleanup()
        raise OracleFailure("CPU reference exited without an output checkpoint")

    def _progress_tensor(
        self,
        frozen: ValidatedTensorSpec,
        files: dict[str, bytes],
        path: str,
    ) -> ArtifactRef:
        spec, suite, contract = frozen.spec, frozen.suite, frozen.contract
        outputs = contract.output_contract()
        if baseline_key_v2(spec.content_sha256, suite, spec.environment, outputs) != self.key:
            raise OracleFailure("Tensor oracle identity mismatch")
        plan = {
            "path": path,
            "output_owner": [os.geteuid(), os.getegid()],
            "spec": spec.manifest.spec.model_dump(mode="json"),
            "suite": suite.model_dump(mode="json"),
            "contract": contract.model_dump(mode="json"),
            "environment": spec.environment.model_dump(mode="json"),
            "files": {k: base64.b64encode(v).decode() for k, v in files.items()},
        }

        def validate(package: bytes, reference: ArtifactRef) -> None:
            unpack_baseline_v2(
                package,
                reference.sha256,
                spec_sha256=spec.content_sha256,
                suite=suite,
                environment=spec.environment,
                contract=outputs,
            )

        return self._progress(
            plan,
            worker_source_v2(checkpoint=True),
            lambda data: self.publish_tensor(frozen, data),
            validate,
        )

    def publish_tensor(self, frozen: ValidatedTensorSpec, data: bytes) -> ArtifactRef:
        spec, suite, contract = frozen.spec, frozen.suite, frozen.contract
        try:
            reply = json.loads(data)
            if (
                isinstance(reply, dict)
                and set(reply) == {"error"}
                and reply["error"]
                in (
                    "reference_failed",
                    "reference_timeout",
                )
            ):
                raise OracleFailure("CPU reference failed: " + reply["error"])
            if (
                not isinstance(reply, dict)
                or set(reply) != {"outputs"}
                or not isinstance(reply["outputs"], dict)
                or not all(isinstance(values, dict) for values in reply["outputs"].values())
            ):
                raise ValueError
            outputs = {
                case: {
                    name: base64.b64decode(value, validate=True) for name, value in values.items()
                }
                for case, values in reply["outputs"].items()
            }
            package = pack_baseline_v2(
                spec.content_sha256,
                suite,
                spec.environment,
                self.image,
                contract.output_contract(),
                outputs,
            )
        except OracleFailure:
            self.cleanup()
            raise
        except (ValueError, TypeError):
            self.cleanup()
            raise OracleFailure("Invalid CPU tensor reference output") from None
        return self._publish_package(package)

    def publish(self, spec: ValidatedSpec, suite: RecipeSuite, data: bytes) -> ArtifactRef:
        try:
            reply = json.loads(data)
            if set(reply) == {"error"} and reply["error"] in (
                "reference_failed",
                "reference_timeout",
            ):
                raise OracleFailure("CPU reference failed: " + reply["error"])
            if set(reply) != {"outputs"} or not isinstance(reply["outputs"], dict):
                raise ValueError
            outputs = {k: base64.b64decode(v, validate=True) for k, v in reply["outputs"].items()}
            package = pack_baseline(
                spec.content_sha256, suite, spec.environment, self.image, outputs
            )
        except OracleFailure:
            self.cleanup()
            raise
        except (ValueError, TypeError):
            self.cleanup()
            raise OracleFailure("Invalid CPU reference output") from None
        return self._publish_package(package)

    def _publish_package(self, package: bytes) -> ArtifactRef:
        self.fence()
        reference = self.artifacts.put("baselines/" + self.key + "/outputs.tar", package)
        if reference.repository != "private" or reference.revision != reference.sha256:
            raise OracleFailure("Baseline requires private pinned storage")
        if verified(self.artifacts.get(reference), reference) != package:
            raise OracleFailure("Baseline private readback mismatch")
        self.fence()
        durable_write(self.journal / "artifact.json", reference.model_dump_json().encode())
        return reference

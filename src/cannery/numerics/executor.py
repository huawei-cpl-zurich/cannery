"""Regenerate candidate inputs on A5; compare outputs only in the controller."""

import io
import json
import re
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from cannery.artifacts import ArtifactRef, digest, verified
from cannery.builds.executor import BuildExecutor, ExecutionFailure
from cannery.builds.remote import RemoteFailure, RemotePending, durable_write
from cannery.builds.sandbox import job_script
from cannery.numerics.baseline_format import BaselineBundle
from cannery.numerics.evaluation import evaluate
from cannery.numerics.recipes import RecipeSuite
from cannery.numerics.validation_runtime import parse_results, runtime_program
from cannery.registry.manifest import ValidatedSpec
from cannery.submissions.manifest import SubmissionManifest
from cannery.submissions.models import Candidate


class NumericalExecutor(BuildExecutor):
    def validate(
        self,
        candidate: Candidate,
        attempt: UUID,
        fence: Callable[[], None],
        *,
        spec: ValidatedSpec,
        suite: RecipeSuite,
        baseline: BaselineBundle,
        binary: ArtifactRef,
        baseline_artifact: ArtifactRef,
    ) -> dict[str, Any]:
        self.fence = fence
        manifest = SubmissionManifest.model_validate(candidate.manifest)
        build = manifest.implementation.build
        if build.abi == "catlass-unary-f32-v1":
            covered = {tuple(s.shape) for s in build.specializations or ()}
            if any(tuple(next(iter(c.inputs.values())).shape) not in covered for c in suite.cases):
                raise ExecutionFailure("unsupported_case_shape")
        if binary.repository != "private" or binary.revision != binary.sha256:
            raise ExecutionFailure("invalid_admitted_artifact")
        if baseline.metadata["spec_sha256"] != spec.content_sha256:
            raise ExecutionFailure("baseline_identity_mismatch")
        if self.wheel.stat().st_size > 2 * 1024 * 1024:
            raise ExecutionFailure("loader_size_limit")
        wheel = self.wheel.read_bytes()
        if digest(wheel) != self.wheel_sha256 or not re.fullmatch(
            r"[A-Za-z0-9_.-]+\.whl", self.wheel.name
        ):
            raise ExecutionFailure("loader_digest_mismatch")
        program = runtime_program()
        context = json.dumps(
            {
                "profile": self.profile.model_dump(mode="json"),
                "source": candidate.content_sha256,
                "wheel": self.wheel_sha256,
                "binary": binary.model_dump(mode="json"),
                "baseline": baseline_artifact.model_dump(mode="json"),
                "baseline_key": baseline.metadata["key"],
                "suite": suite.sha256,
                "defaults": spec.manifest.spec.validation.model_dump(mode="json"),
                "requested": manifest.validation.model_dump(mode="json"),
                "job_contract": digest(
                    job_script(
                        self.profile,
                        attempt,
                        "runtime",
                        runtime_program=program,
                        capture_runtime=True,
                    )
                ),
            },
            sort_keys=True,
        ).encode()
        frozen = self.remote.journal / "context.json"
        if frozen.exists() and frozen.read_bytes() != context:
            raise ExecutionFailure("validation_context_changed")
        fence()
        durable_write(frozen, context)
        checkpoint = self.remote.journal / "numerical-report.json"
        if checkpoint.exists():
            report = json.loads(checkpoint.read_bytes())
            if not isinstance(report, dict):
                raise ExecutionFailure("invalid_report_checkpoint")
            return report  # Written only by this trusted controller.
        package = verified(self.artifacts.get(binary), binary)
        try:
            with tarfile.open(fileobj=io.BytesIO(package), mode="r:") as archive:
                member = archive.extractfile("metadata.json")
                if member is None:
                    raise ValueError
                metadata = json.loads(member.read(16385))
            if (
                metadata["source_sha256"] != candidate.content_sha256
                or metadata["runtime"] != self.profile.fingerprint
            ):
                raise ValueError
        except (ValueError, KeyError, tarfile.TarError):
            raise ExecutionFailure("admitted_artifact_mismatch") from None
        self.operation("prepare", job_script(self.profile, attempt, "prepare"))
        self.upload(
            attempt,
            "validation",
            {
                "admitted.tar": package,
                "artifact.json": binary.model_dump_json().encode(),
                self.wheel.name: wheel,
                "cases.json": json.dumps(
                    {
                        "suite": suite.model_dump(mode="json"),
                        "input_sha256": {
                            c["id"]: c["input_sha256"] for c in baseline.metadata["cases"]
                        },
                    },
                    sort_keys=True,
                ).encode(),
            },
        )
        index_path = self.remote.journal / "runtime-index"
        index = int(index_path.read_text()) if index_path.exists() else 0
        if index >= 360:
            raise ExecutionFailure("device_unavailable")
        try:
            handle = self.operation(
                f"runtime-{index}",
                job_script(
                    self.profile, attempt, "runtime", runtime_program=program, capture_runtime=True
                ),
                npu=self.profile.npu,
            )
        except RemoteFailure as error:
            if error.exit_code == 75:
                durable_write(index_path, str(index + 1).encode())
                raise RemotePending("Waiting for a healthy device") from None
            raise ExecutionFailure("numerical_runtime_failed", retryable=True) from None
        checksum = self.remote.call("logs", handle, "--tail", "2").get("content", "").strip()
        if not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ExecutionFailure("numerical_evidence_missing")
        output = self.remote.journal / "results.json"
        fence()
        self.remote.transfer(
            "download",
            str(Path(self.profile.workspace) / attempt.hex / "output/results.json"),
            str(output),
        )
        if output.is_symlink() or output.stat().st_size > 63 * 1024 * 1024:
            raise ExecutionFailure("invalid_numerical_output")
        raw = output.read_bytes()
        if digest(raw) != checksum:
            raise ExecutionFailure("numerical_digest_mismatch")
        try:
            values = parse_results(raw, suite)
            report = evaluate(
                baseline,
                suite,
                {k: v.data for k, v in values.items()},
                {k: v.input_sha256 for k, v in values.items()},
                spec.manifest.spec.validation,
                manifest.validation,
            )
        except ValueError:
            raise ExecutionFailure("invalid_numerical_output") from None
        report.update(
            binary_sha256=binary.sha256,
            baseline_sha256=baseline_artifact.sha256,
            candidate_sha256=candidate.content_sha256,
            output_sha256=checksum,
        )
        fence()
        durable_write(checkpoint, json.dumps(report, sort_keys=True, allow_nan=False).encode())
        return report

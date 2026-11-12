"""Retained reference/candidate timing jobs, verified only by the controller."""

import io
import json
import re
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from cannery.artifacts import ArtifactRef, digest, verified
from cannery.benchmark.contract import Cohort, cache_key, score, summarize
from cannery.benchmark.evidence import read_timings, timing_artifact
from cannery.benchmark.runtime import parse_runtime, runtime_program
from cannery.builds.executor import BuildExecutor, ExecutionFailure
from cannery.builds.remote import RemoteFailure, RemotePending, durable_write
from cannery.builds.sandbox import job_script
from cannery.numerics.baseline_format import BaselineBundle
from cannery.numerics.evaluation import evaluate
from cannery.numerics.recipes import RecipeSuite
from cannery.registry.manifest import ValidatedSpec
from cannery.submissions.manifest import RequestedValidation


class BenchmarkExecutor(BuildExecutor):
    def benchmark(
        self,
        attempt: UUID,
        fence: Callable[[], None],
        *,
        kind: Literal["reference", "candidate"],
        spec: ValidatedSpec,
        suite: RecipeSuite,
        baseline: BaselineBundle,
        cohort: Cohort,
        requested: RequestedValidation,
        binary: ArtifactRef | None = None,
        candidate_sha256: str | None = None,
        reference_samples: dict[str, list[float]] | None = None,
        reference_files: dict[str, bytes] | None = None,
        reference_path: str | None = None,
    ) -> tuple[dict[str, Any], ArtifactRef]:
        self.fence = fence
        if baseline.metadata["spec_sha256"] != spec.content_sha256:
            raise ExecutionFailure("baseline_identity_mismatch")
        if (
            cohort.image != self.profile.image
            or cohort.runtime != self.profile.fingerprint
            or cohort.loader_sha256 != self.wheel_sha256
        ):
            raise ExecutionFailure("performance_cohort_mismatch")
        if self.wheel.stat().st_size > 2 * 1024 * 1024 or not re.fullmatch(
            r"[A-Za-z0-9_.-]+\.whl", self.wheel.name
        ):
            raise ExecutionFailure("loader_size_limit")
        wheel = self.wheel.read_bytes()
        if digest(wheel) != self.wheel_sha256:
            raise ExecutionFailure("loader_digest_mismatch")
        if binary is not None and (
            binary.repository != "private" or binary.revision != binary.sha256
        ):
            raise ExecutionFailure("invalid_admitted_artifact")
        program = runtime_program(kind)
        identity = {"hardware_id": cohort.hardware_id, "driver": cohort.driver}
        script = job_script(
            self.profile,
            attempt,
            "runtime",
            runtime_program=program,
            capture_runtime=True,
            benchmark=identity,
        )
        context = json.dumps(
            {
                "profile": self.profile.model_dump(mode="json"),
                "kind": kind,
                "key": cache_key(spec.content_sha256, suite, cohort),
                "cohort": cohort.model_dump(mode="json"),
                "candidate": candidate_sha256,
                "binary": binary.model_dump(mode="json") if binary else None,
                "cpu_baseline": baseline.metadata["key"],
                "defaults": spec.manifest.spec.validation.model_dump(mode="json"),
                "requested": requested.model_dump(mode="json"),
                "reference_timings": digest(json.dumps(reference_samples, sort_keys=True).encode()),
                "job_contract": digest(script),
            },
            sort_keys=True,
        ).encode()
        frozen = self.remote.journal / "context.json"
        if frozen.exists() and frozen.read_bytes() != context:
            raise ExecutionFailure("benchmark_context_changed")
        fence()
        durable_write(frozen, context)
        checkpoint = self.remote.journal / "benchmark-report.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_bytes())
            if not isinstance(saved, dict) or not isinstance(saved.get("report"), dict):
                raise ExecutionFailure("invalid_benchmark_checkpoint")
            artifact = ArtifactRef.model_validate(saved["artifact"])
            read_timings(self.artifacts.get(artifact), artifact, spec.content_sha256, suite, cohort)
            return saved["report"], artifact
        plan: dict[str, Any] = {
            "suite": suite.model_dump(mode="json"),
            "runtime": cohort.runtime,
            "input_sha256": {c["id"]: c["input_sha256"] for c in baseline.metadata["cases"]},
        }
        files = {self.wheel.name: wheel}
        if kind == "reference":
            if reference_files is None or reference_path is None:
                raise ExecutionFailure("retained_spec_missing")
            from cannery.registry.manifest import validate_spec

            checked = validate_spec(reference_files, reference_path)
            if checked != spec:
                raise ExecutionFailure("retained_spec_mismatch")
            files.update({"tree/" + k: v for k, v in reference_files.items()})
            plan.update(
                reference={
                    "path": reference_path,
                    **spec.manifest.spec.reference.model_dump(mode="json"),
                },
                input_name=spec.manifest.spec.interface.inputs[0],
                output_name=spec.manifest.spec.interface.outputs[0],
            )
        else:
            if binary is None or candidate_sha256 is None or reference_samples is None:
                raise ExecutionFailure("admitted_candidate_missing")
            package = verified(self.artifacts.get(binary), binary)
            try:
                with tarfile.open(fileobj=io.BytesIO(package), mode="r:") as archive:
                    member = archive.extractfile("metadata.json")
                    if member is None:
                        raise ValueError
                    metadata = json.loads(member.read(16385))
                if (
                    metadata["source_sha256"] != candidate_sha256
                    or metadata["runtime"] != self.profile.fingerprint
                ):
                    raise ValueError
            except (ValueError, KeyError, tarfile.TarError):
                raise ExecutionFailure("admitted_artifact_mismatch") from None
            files.update(
                {"admitted.tar": package, "artifact.json": binary.model_dump_json().encode()}
            )
        files["cases.json"] = json.dumps(plan, sort_keys=True).encode()
        self.operation("prepare", job_script(self.profile, attempt, "prepare"))
        self.upload(attempt, "benchmark", files)
        index_path = self.remote.journal / "runtime-index"
        index = int(index_path.read_text()) if index_path.exists() else 0
        if index >= 360:
            raise ExecutionFailure("device_unavailable")
        try:
            handle = self.operation(f"runtime-{index}", script, npu=self.profile.npu)
        except RemoteFailure as error:
            if error.exit_code == 75:
                durable_write(index_path, str(index + 1).encode())
                raise RemotePending("Waiting for a healthy idle device") from None
            raise ExecutionFailure("benchmark_runtime_failed", retryable=True) from None
        checksum = self.remote.call("logs", handle, "--tail", "2").get("content", "").strip()
        if not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ExecutionFailure("benchmark_evidence_missing")
        output = self.remote.journal / "results.json"
        fence()
        self.remote.transfer(
            "download",
            str(Path(self.profile.workspace) / attempt.hex / "output/results.json"),
            str(output),
            large_output=True,
        )
        if output.is_symlink() or output.stat().st_size > 128 * 1024 * 1024:
            raise ExecutionFailure("invalid_benchmark_output")
        raw = output.read_bytes()
        if digest(raw) != checksum:
            raise ExecutionFailure("benchmark_digest_mismatch")
        try:
            values = parse_runtime(raw, suite)
            checks = [
                evaluate(
                    baseline,
                    suite,
                    {k: v.data for k, v in tensors.items()},
                    {k: v.input_sha256 for k, v in tensors.items()},
                    spec.manifest.spec.validation,
                    requested,
                )
                for tensors in (values.before, values.after)
            ]
            passed = all(c["passed"] for c in checks)
            if kind == "reference":
                summary = {
                    "cases": [
                        {"id": c.id, "reference": summarize(values.samples[c.id])}
                        if c.id in values.samples
                        else {"id": c.id, "excluded": "empty_correctness_only"}
                        for c in suite.cases
                    ]
                }
            else:
                assert reference_samples is not None
                summary = score(suite, reference_samples, values.samples) if passed else {}
            packed = timing_artifact(spec.content_sha256, suite, cohort, values.samples)
        except ValueError:
            raise ExecutionFailure("invalid_benchmark_output") from None
        fence()
        artifact = self.artifacts.put(f"performance/{kind}/{attempt.hex}/samples.json", packed)
        read_timings(self.artifacts.get(artifact), artifact, spec.content_sha256, suite, cohort)
        report = {
            "protocol": cohort.protocol,
            "passed": passed,
            "kind": kind,
            "key": cache_key(spec.content_sha256, suite, cohort),
            "cohort": cohort.model_dump(mode="json"),
            "correctness": checks,
            **summary,
        }
        if not passed:
            report.pop("geomean_speedup", None)
        fence()
        durable_write(
            checkpoint,
            json.dumps(
                {"report": report, "artifact": artifact.model_dump(mode="json")},
                sort_keys=True,
                allow_nan=False,
            ).encode(),
        )
        return report, artifact

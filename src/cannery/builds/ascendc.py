"""Trusted launch-wrapper generation for the initial A5 Ascend C ABI."""

from dataclasses import dataclass

from cannery.submissions.manifest import SubmissionManifest


class BuildError(ValueError):
    pass


@dataclass(frozen=True)
class BuildPlan:
    wrapper: str
    command: tuple[str, ...]


class AscendCAdapter:
    def preflight(self, manifest: SubmissionManifest) -> None:
        implementation = manifest.implementation
        if (implementation.device, implementation.language, implementation.build.abi) != (
            "a5",
            "ascendc",
            "unary-f32-v1",
        ):
            raise BuildError("Unsupported build contract; requires A5 Ascend C unary-f32-v1")
        entry, _ = implementation.entrypoint.split(":")
        # Initial ABI supports one translation unit and declared headers only.
        if any(
            path != entry and not path.endswith((".h", ".hpp"))
            for path in implementation.build.inputs
        ):
            raise BuildError("unary-f32-v1 supports one entrypoint translation unit plus headers")
        # Paths are emitted into an include directive, never interpreted as shell text.
        if any(
            not all(c.isalnum() or c in "_./-" for c in path)
            for path in implementation.build.inputs
        ):
            raise BuildError("Unsupported build input spelling")

    def build(self, manifest: SubmissionManifest) -> BuildPlan:
        """Return trusted inputs; caller must compile in a device-less sandbox."""
        self.preflight(manifest)
        entry, symbol = manifest.implementation.entrypoint.split(":")
        wrapper = f"""#include <cstdint>
#include "source/{entry}"
extern "C" void cannery_launch(void* stream, void* input, void* output, uint64_t count) {{
    {symbol}<<<1, nullptr, stream>>>((GM_ADDR)input,
                                   (GM_ADDR)output, count);
}}
"""
        optimization = (
            "-O2" if manifest.implementation.build.options.optimization == "release" else "-O0"
        )
        return BuildPlan(
            wrapper,
            (
                "bisheng",
                "-shared",
                "-fPIC",
                "-x",
                "asc",
                "launch.asc",
                "-I",
                "source",
                "--npu-arch=dav-3510",
                optimization,
                "-o",
                "kernel.so",
            ),
        )

"""Generate the trusted compiler driver; importing this module never imports CATLASS."""

import inspect
import json
import math

from cannery.builds import catlass
from cannery.builds.ascendc import AscendCAdapter, BuildError, BuildPlan
from cannery.submissions.manifest import SubmissionManifest


class CatlassAdapter:
    def preflight(self, manifest: SubmissionManifest) -> None:
        impl = manifest.implementation
        if (impl.device, impl.language, impl.build.abi) != (
            "a5",
            "catlass",
            "catlass-unary-f32-v1",
        ) or impl.build.specializations is None:
            raise BuildError("Requires A5 CATLASS static unary float32 specializations")
        if impl.build.options.optimization != "release":
            raise BuildError("CATLASS binary admission requires release optimization")
        if not any(math.prod(item.shape) for item in impl.build.specializations):
            raise BuildError("At least one CATLASS specialization must be nonempty")

    def build(self, manifest: SubmissionManifest) -> BuildPlan:
        self.preflight(manifest)
        impl = manifest.implementation
        assert impl.build.specializations is not None
        entry, symbol = impl.entrypoint.split(":")
        parameters = {
            "entry": entry,
            "symbol": symbol,
            "counts": sorted({math.prod(s.shape) for s in impl.build.specializations} - {0}),
        }
        # Copy only trusted source helpers into the standalone compile driver.
        # The author container does not contain the control-plane application.
        helpers = "\n".join(
            inspect.getsource(value)
            for value in (
                catlass._native_object,
                catlass.NativeVariant,
                catlass.validate_export,
                catlass.native_wrapper,
            )
        )
        prelude = (
            "import hashlib,importlib.util,json,os,re,shutil,subprocess,sys\n"
            "from pathlib import Path\nfrom dataclasses import dataclass\nfrom typing import Any\n"
            "class BuildError(ValueError): pass\n"
            f"MAX_OBJECT_BYTES={catlass.MAX_OBJECT_BYTES!r}\n"
            f"ACL_BRIDGE={catlass.ACL_BRIDGE!r}\n"
            f"PARAMETERS=json.loads({json.dumps(parameters, sort_keys=True)!r})\n"
        )
        return BuildPlan(prelude + helpers + COMPILER, ("python", "launch.py"), "launch.py")


COMPILER = r"""

def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compile_exports(plan, source, output):
    import catlass as framework
    import catlass.execution as execution
    import catlass.tla as tla
    from catlass.tla.runtime import make_fake_tensor
    from catlass.compiler_bridge import kernel_abi_to_dict

    pins = plan['catlass']
    if (framework.__version__ != pins['version']
            or _sha256(execution.__file__) != pins['execution_sha256']
            or _sha256(shutil.which('hivmc-a5')) != pins['hivmc_sha256']):
        raise BuildError('CATLASS compiler fingerprint mismatch')
    cache = Path(os.environ['CATLASS_DSL_CACHE_DIR'])
    bc = cache / 'bc' / pins['bitcode_key']
    if (_sha256(bc / 'manifest.json') != pins['manifest_sha256']
            or _sha256(bc / 'meta_op.aiv.c310.bc') != pins['bitcode_sha256']):
        raise BuildError('CATLASS bitcode fingerprint mismatch')
    source = source.resolve()
    entry = (source / PARAMETERS['entry']).resolve()
    if not entry.is_relative_to(source):
        raise BuildError('Entrypoint outside source tree')
    compiler = shutil.which('bisheng')
    cann = Path(os.environ['ASCEND_HOME_PATH'])
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('cannery_candidate', entry)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    kernel = getattr(candidate, PARAMETERS['symbol'])
    variants = []
    for count in PARAMETERS['counts']:
        sample = make_fake_tensor(tla.Float32, (count,), (1,))
        compiled = tla.compile(kernel, sample, sample, options='--npu-arch 3510')
        module = compiled.jit_module
        path = Path(compiled.kernel_binary_path)
        if path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()):
            raise BuildError('Export outside private compiler cache')
        with path.open('rb') as stream:
            binary = stream.read(MAX_OBJECT_BYTES + 1)
        metadata = {
            'entrypoint': module.entrypoint, 'kernel_mode': module.kernel_mode,
            'kernel_abi': kernel_abi_to_dict(module.execution_args.kernel_abi),
            'uses_scalar_print': module.uses_scalar_print,
            'uses_tensor_print': module.uses_tensor_print, 'is_mixed': module.is_mixed,
            'ub_static_bytes': module.ub_static_bytes, 'ub_dynamic_base': module.ub_dynamic_base,
        }
        variants.append(validate_export(count, binary, metadata))
    native = output.with_suffix('.cpp')
    native.write_text(native_wrapper(variants))
    subprocess.run([
        compiler, '-shared', '-fPIC', '-O2', '-std=c++17', '-I', str(cann / 'include'),
        str(native), '-L', str(cann / 'lib64'), '-lascendcl', '-lstdc++', '-pthread',
        '-Wl,-z,defs', '-o', str(output),
    ], check=True, stdout=sys.stderr, timeout=60)


if __name__ == '__main__':
    compile_exports(json.loads(Path('/input/build.json').read_bytes()),
                    Path('source'), Path('kernel.so'))
"""


def adapter_for(manifest: SubmissionManifest) -> "CatlassAdapter | AscendCAdapter":
    adapter = (
        CatlassAdapter() if manifest.implementation.language == "catlass" else AscendCAdapter()
    )
    adapter.preflight(manifest)
    return adapter

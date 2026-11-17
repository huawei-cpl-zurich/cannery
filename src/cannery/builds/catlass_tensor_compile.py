"""Trusted static tensor compilation driver; no framework import in the controller."""

import inspect
import json
from pathlib import Path

from cannery.builds import catlass, catlass_tensor, tensor_native
from cannery.builds.ascendc import BuildError, BuildPlan
from cannery.builds.tensor_signatures import tensor_variants
from cannery.registry.tensor_manifest import ValidatedTensorSpec
from cannery.submissions.manifest import SubmissionManifest


def package_sha256(root: Path) -> str:
    """Match the frozen toolchain inventory: paths + NUL + raw SHA256 digests."""
    import hashlib

    result = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").digest()
        result.update(str(path.relative_to(root)).encode() + b"\0" + checksum)
    return result.hexdigest()


class CatlassTensorAdapter:
    def build(
        self, manifest: SubmissionManifest, frozen: ValidatedTensorSpec, abi_header: str
    ) -> BuildPlan:
        impl = manifest.implementation
        if (
            (manifest.schema_version, impl.language, impl.device, impl.build.abi)
            != (2, "catlass", "a5", "tensor-v2")
            or impl.build.options.optimization != "release"
            or frozen.spec.manifest.schema_version != 2
            or manifest.spec_version != frozen.spec.manifest.spec.version
            or manifest.kernel
            != f"{frozen.spec.manifest.kernel.namespace}/{frozen.spec.manifest.kernel.name}"
            or not isinstance(abi_header, str)
            or not abi_header
            or "cannery_prepare_tensor_v2" not in abi_header
            or len(abi_header.encode()) > 16384
        ):
            raise BuildError(
                "Requires matching A5 tensor spec, CATLASS submission and pinned ABI header"
            )
        entry, symbol = impl.entrypoint.split(":")
        parameters = {
            "entry": entry,
            "symbol": symbol,
            "variants": tensor_variants(frozen),
            "abi_header": abi_header,
        }
        helpers = "\n".join(
            inspect.getsource(value)
            for value in (
                package_sha256,
                catlass._native_object,
                catlass_tensor.StaticTensorExport,
                catlass_tensor.validate_static_export,
                tensor_native._conditions,
                tensor_native.native_tensor_wrapper,
            )
        )
        prelude = (
            "import copy,hashlib,importlib.util,json,math,os,re,shutil,struct,subprocess,sys\n"
            "from pathlib import Path\nfrom dataclasses import dataclass\n"
            "from typing import Any,Literal\n"
            "class BuildError(ValueError): pass\n"
            f"MAX_OBJECT_BYTES={catlass.MAX_OBJECT_BYTES!r}\n"
            f"MLIR_TYPES={catlass_tensor.MLIR_TYPES!r}\n"
            f"DTYPES={tensor_native.DTYPES!r}\nWIDTHS={tensor_native.WIDTHS!r}\n"
            f"PREPARE={tensor_native.PREPARE!r}\n"
            f"PARAMETERS=json.loads({json.dumps(parameters, sort_keys=True)!r})\n"
        )
        return BuildPlan(prelude + helpers + COMPILER, ("python", "launch.py"), "launch.py")


COMPILER = r"""

def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compile_tensor_exports(plan, source, output):
    pins=plan.get('catlass')
    if (not isinstance(pins,dict) or not pins.get('package_sha256')
        or not pins.get('aic_bitcode_sha256')):
        raise BuildError('Complete CATLASS tensor toolchain pins are required')
    import catlass as framework
    import catlass.execution as execution
    from catlass.compiler_bridge import kernel_abi_to_dict
    compiler=shutil.which('bisheng')
    if (not compiler or framework.__version__!=pins['version']
        or _sha256(execution.__file__)!=pins['execution_sha256']
        or _sha256(shutil.which('hivmc-a5'))!=pins['hivmc_sha256']
        or package_sha256(Path(framework.__file__).parent)!=pins['package_sha256']):
        raise BuildError('CATLASS tensor compiler fingerprint mismatch')
    cache=Path(os.environ['CATLASS_DSL_CACHE_DIR'])
    bc=cache/'bc'/pins['bitcode_key']
    for name,expected in [('manifest.json',pins['manifest_sha256']),
        ('meta_op.aiv.c310.bc',pins['bitcode_sha256']),
        ('meta_op.aic.c310.bc',pins['aic_bitcode_sha256'])]:
        if _sha256(bc/name)!=expected: raise BuildError('CATLASS tensor bitcode mismatch')
    source=source.resolve()
    entry=(source/PARAMETERS['entry']).resolve()
    if not entry.is_relative_to(source): raise BuildError('Entrypoint outside source tree')
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('cannery_tensor_candidate',entry)
    candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate
    spec.loader.exec_module(candidate)
    build_variant=getattr(candidate,PARAMETERS['symbol'])
    variants=[]
    for signature in PARAMETERS['variants']:
        exported=build_variant(copy.deepcopy(signature))
        if type(exported) is not dict or set(exported)!={'artifact','bindings','block_num'}:
            raise BuildError('Expected one compiled artifact, pointer bindings and block count')
        artifact=exported['artifact'];module=artifact.jit_module
        path=Path(artifact.kernel_binary_path)
        if path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()):
            raise BuildError('Native object outside private compiler cache')
        with path.open('rb') as stream: binary=stream.read(MAX_OBJECT_BYTES+1)
        metadata={'entrypoint':module.entrypoint,'kernel_mode':module.kernel_mode,
            'kernel_abi':kernel_abi_to_dict(module.execution_args.kernel_abi),
            'uses_scalar_print':module.uses_scalar_print,'uses_tensor_print':module.uses_tensor_print,
            'is_mixed':module.is_mixed,'ub_static_bytes':module.ub_static_bytes,
            'ub_dynamic_base':module.ub_dynamic_base}
        native=validate_static_export(binary,metadata,signature,
            exported['bindings'],exported['block_num'])
        variants.append((signature,native))
    (output.parent/'tensor_abi.h').write_text(PARAMETERS['abi_header'])
    native=output.with_suffix('.cpp');native.write_text(native_tensor_wrapper(variants))
    cann=Path(os.environ['ASCEND_HOME_PATH'])
    if (cann/'aarch64-linux/include/acl/acl.h').is_file(): cann=cann/'aarch64-linux'
    subprocess.run([compiler,'-shared','-fPIC','-O2','-std=c++20','-I',str(cann/'include'),
        '-I',str(output.parent),str(native),'-L',str(cann/'lib64'),'-lascendcl','-lstdc++',
        '-pthread','-Wl,-z,defs','-o',str(output)],check=True,stdout=sys.stderr,timeout=60)


if __name__=='__main__':
    compile_tensor_exports(json.loads(Path('/input/build.json').read_bytes()),
                           Path('source'),Path('kernel.so'))
"""

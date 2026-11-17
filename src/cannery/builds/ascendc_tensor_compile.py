"""Pinned, device-free Ascend C compilation for frozen tensor variants."""

import inspect
import json

from cannery.builds import ascendc_tensor, catlass, catlass_tensor, tensor_native
from cannery.builds.ascendc import BuildError, BuildPlan
from cannery.builds.tensor_signatures import tensor_build_parameters
from cannery.registry.tensor_manifest import ValidatedTensorSpec
from cannery.submissions.manifest import SubmissionManifest


class AscendCTensorAdapter:
    def build(
        self, manifest: SubmissionManifest, frozen: ValidatedTensorSpec, abi_header: str
    ) -> BuildPlan:
        if not manifest.implementation.entrypoint.split(":")[0].endswith(".py"):
            raise BuildError("Ascend C tensor builds require a Python source factory")
        parameters = tensor_build_parameters(manifest, frozen, abi_header, "ascendc")
        helpers = "\n".join(
            inspect.getsource(value)
            for value in (
                catlass._native_object,
                catlass_tensor.StaticTensorExport,
                ascendc_tensor.ascendc_sections,
                ascendc_tensor.ascendc_records,
                ascendc_tensor.validate_ascendc_export,
                tensor_native._conditions,
                tensor_native.native_tensor_wrapper,
            )
        )
        prelude = (
            "import copy,hashlib,importlib.util,json,math,os,re,shutil,struct,subprocess,sys\n"
            "from pathlib import Path\nfrom dataclasses import dataclass\n"
            "from typing import Any,Literal\n"
            "class BuildError(ValueError): pass\n"
            f"MAX_OBJECT_BYTES={catlass.MAX_OBJECT_BYTES!r}\nDTYPES={tensor_native.DTYPES!r}\n"
            f"WIDTHS={tensor_native.WIDTHS!r}\nPREPARE={tensor_native.PREPARE!r}\n"
            f"PARAMETERS=json.loads({json.dumps(parameters, sort_keys=True)!r})\n"
        )
        return BuildPlan(prelude + helpers + COMPILER, ("python", "launch.py"), "launch.py")


COMPILER = r"""
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def compile_ascendc(plan,source,output):
    pins=plan.get('ascendc')
    if not isinstance(pins,dict):raise BuildError('Ascend C toolchain pins required')
    compiler=shutil.which('bisheng');linker=shutil.which('ld.lld')
    if sha(compiler)!=pins['bisheng_sha256'] or sha(linker)!=pins['ld_sha256']:
        raise BuildError('Ascend C compiler fingerprint mismatch')
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
    source=source.resolve();entry=(source/PARAMETERS['entry']).resolve()
    if not entry.is_relative_to(source):raise BuildError('Entrypoint outside source tree')
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('cannery_ascendc_candidate',entry)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    factory=getattr(module,PARAMETERS['symbol']);variants=[]
    for index,signature in enumerate(PARAMETERS['variants']):
        request=factory(copy.deepcopy(signature))
        fields={'source','entrypoint','bindings','block_num'}
        if (type(request) is not dict or not fields<=set(request)
            or set(request)-fields-{'include_dirs'}):
            raise BuildError('Expected Ascend C source and static export declaration')
        directories=request.get('include_dirs',[])
        if type(directories) is not list or len(directories)>16:
            raise BuildError('Invalid Ascend C include directories')
        includes=['-I',str(source)]
        for directory in directories:
            if type(directory) is not str or Path(directory).is_absolute():
                raise BuildError('Invalid Ascend C include directory')
            path=(source/directory).resolve()
            if not path.is_relative_to(source) or not path.is_dir():
                raise BuildError('Ascend C include directory outside source')
            includes+=['-I',str(path)]
        code=request['source']
        if type(code) is not str or not 1<=len(code.encode())<=131072:
            raise BuildError('Invalid generated Ascend C source size')
        path=output.parent/f'variant-{index}.asc';path.write_text(code)
        obj=path.with_suffix('.o');binary=path.with_suffix('.bin')
        subprocess.run([compiler,'-O2','-x','asc','--npu-arch=dav-3510','--cce-aicore-only',
            *includes,str(path),'-o',str(obj)],check=True,stdout=sys.stderr,timeout=120)
        if obj.stat().st_size>MAX_OBJECT_BYTES:raise BuildError('Ascend C object size limit')
        subprocess.run([linker,'-m','aicorelinux','-Ttext=0',str(obj),'-o',str(binary)],
            check=True,stdout=sys.stderr,timeout=60)
        if binary.stat().st_size>MAX_OBJECT_BYTES:raise BuildError('Ascend C image size limit')
        export=validate_ascendc_export(binary.read_bytes(),request['entrypoint'],signature,
                                       request['bindings'],request['block_num'])
        variants.append((signature,export))
    (output.parent/'tensor_abi.h').write_text(PARAMETERS['abi_header'])
    cpp=output.with_suffix('.cpp');cpp.write_text(native_tensor_wrapper(variants))
    root=Path(os.environ['ASCEND_HOME_PATH'])
    if (root/'aarch64-linux/include/acl/acl.h').is_file():root=root/'aarch64-linux'
    subprocess.run([compiler,'-shared','-fPIC','-O2','-std=c++20','-I',str(root/'include'),
        '-I',str(output.parent),str(cpp),'-L',str(root/'lib64'),'-lascendcl','-lstdc++',
        '-pthread','-Wl,-z,defs','-o',str(output)],check=True,stdout=sys.stderr,timeout=60)


if __name__=='__main__':
    compile_ascendc(json.loads(Path('/input/build.json').read_bytes()),Path('source'),Path('kernel.so'))
"""

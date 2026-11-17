"""Pinned offline Triton compilation of frozen pointer-only tensor signatures."""

import inspect
import json

from cannery.builds import catlass, catlass_tensor, tensor_native, triton_tensor
from cannery.builds.ascendc import BuildPlan
from cannery.builds.catlass_tensor_compile import package_sha256
from cannery.builds.tensor_signatures import tensor_build_parameters
from cannery.registry.tensor_manifest import ValidatedTensorSpec
from cannery.submissions.manifest import SubmissionManifest


class TritonTensorAdapter:
    def build(
        self, manifest: SubmissionManifest, frozen: ValidatedTensorSpec, abi_header: str
    ) -> BuildPlan:
        parameters = tensor_build_parameters(manifest, frozen, abi_header, "triton")
        helpers = "\n".join(
            inspect.getsource(value)
            for value in (
                package_sha256,
                catlass._native_object,
                catlass_tensor.StaticTensorExport,
                triton_tensor.validate_triton_export,
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


def compile_triton(plan,source,output):
    pins=plan.get('triton')
    if not isinstance(pins,dict):raise BuildError('Triton toolchain pins required')
    root=Path(os.environ['ASCEND_HOME_PATH'])
    compiler=root/'tools/bishengir/bin/bishengir-compile'
    linker=shutil.which('bisheng')
    if sha(compiler)!=pins['bishengir_sha256'] or sha(linker)!=pins['bisheng_sha256']:
        raise BuildError('Triton compiler fingerprint mismatch')
    os.environ['PATH']=str(compiler.parent)+os.pathsep+os.environ['PATH']
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    package=Path(triton.__file__).parent
    if (package.resolve()!=Path(pins['package_root']).resolve()
        or triton.__version__!=pins['version'] or package_sha256(package)!=pins['package_sha256']):
        raise BuildError('Triton frontend fingerprint mismatch')
    # The read-only namespace mount intentionally excludes distribution entry points.
    from triton.backends import Backend,backends
    from triton.backends.ascend.compiler import AscendBackend
    from triton.backends.ascend.driver import NPUDriver
    backends.clear();backends['ascend']=Backend(AscendBackend,NPUDriver)
    source=source.resolve();entry=(source/PARAMETERS['entry']).resolve()
    if not entry.is_relative_to(source):raise BuildError('Entrypoint outside source tree')
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('cannery_triton_candidate',entry)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    factory=getattr(module,PARAMETERS['symbol'])
    arch=plan['runtime']['soc'];variants=[];grids=[]
    for signature in PARAMETERS['variants']:
        request=factory(copy.deepcopy(signature))
        if type(request) is not dict or set(request)!={'kernel','bindings','constants','grid'}:
            raise BuildError('Expected Triton kernel, pointer bindings, constants and grid')
        fn=request['kernel'];bindings=request['bindings'];constants=request['constants']
        if type(bindings) is not list or not 1<=len(bindings)<=64 or type(constants) is not dict:
            raise BuildError('Invalid Triton source signature')
        named={}
        for binding in bindings:
            if (type(binding) is not list or len(binding)!=3 or type(binding[0]) is not str
                or binding[0] in named or binding[1] not in ('input','output')):
                raise BuildError('Invalid Triton pointer binding')
            named[binding[0]]=binding[1:]
        if set(named)&set(constants) or set(named)|set(constants)!=set(fn.arg_names):
            raise BuildError('Every native argument must be a frozen pointer or build constant')
        if len(fn.arg_names)!=len(set(fn.arg_names)):
            raise BuildError('Duplicate Triton argument names')
        order=[name for name in fn.arg_names if name in named]
        ordered=[named[name] for name in order];types={}
        for name,(direction,tensor_name) in zip(order,ordered,strict=True):
            tensor=next(t for t in signature[direction+'s'] if t['name']==tensor_name)
            dtype={'float32':'fp32','float16':'fp16','bfloat16':'bf16','int8':'i8',
                   'int32':'i32','int64':'i64','bool':'i1'}[tensor['dtype']]
            types[name]='*'+dtype
        compiled=triton.compile(ASTSource(fn,signature=types,constexprs=constants),
                                target=GPUTarget('npu',arch,0))
        metadata=compiled.metadata._asdict()
        if type(metadata.get('cluster_dims')) is tuple:
            metadata['cluster_dims']=list(metadata['cluster_dims'])
        grid=request['grid']
        if type(grid) is not list:raise BuildError('Expected static three-dimensional grid')
        grid=tuple(grid)
        native=validate_triton_export(compiled.asm['npubin'],metadata,signature,ordered,grid,arch=arch)
        variants.append((signature,native));grids.append(grid)
    (output.parent/'tensor_abi.h').write_text(PARAMETERS['abi_header'])
    cpp=output.with_suffix('.cpp');cpp.write_text(native_tensor_wrapper(variants,triton_grids=grids))
    if (root/'aarch64-linux/include/acl/acl.h').is_file():root=root/'aarch64-linux'
    subprocess.run([linker,'-shared','-fPIC','-O2','-std=c++20','-I',str(root/'include'),
        '-I',str(output.parent),str(cpp),'-L',str(root/'lib64'),'-lascendcl','-lstdc++',
        '-pthread','-Wl,-z,defs','-o',str(output)],check=True,stdout=sys.stderr,timeout=60)


if __name__=='__main__':
    compile_triton(json.loads(Path('/input/build.json').read_bytes()),Path('source'),Path('kernel.so'))
"""

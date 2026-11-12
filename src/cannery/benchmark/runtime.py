"""Service-owned offline A5 timing runner and bounded result protocol."""

import inspect
import json
from dataclasses import dataclass
from typing import Literal

from cannery.benchmark.contract import PROTOCOL, SAMPLES, WARMUP, measure, parse_samples
from cannery.numerics.recipes import RecipeSuite, _generate, input_sha256
from cannery.numerics.validation_runtime import TensorResult, parse_results
from cannery.registry.manifest import unique_pairs

SETUP = r"""
wire=os.fdopen(os.dup(sys.stdout.fileno()),'wb')
os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
import torch,torch_npu
torch.npu.set_device(0)
assert torch.npu.device_count()==1
acl=ctypes.CDLL(os.environ['ASCEND_HOME_PATH']+'/lib64/libascendcl.so')
physical_fn=getattr(acl,'aclrtGetPhyDevIdByUserDevId',None) or acl.aclrtGetPhyDevIdByLogicDevId
physical_fn.argtypes=[ctypes.c_int32,ctypes.POINTER(ctypes.c_int32)]
physical_fn.restype=ctypes.c_int
physical=ctypes.c_int32(-1)
assert physical_fn(0,ctypes.byref(physical))==0
assert physical.value==int(os.environ['CANNERY_PHYSICAL_DEVICE'])
plan=json.loads(Path('/input/cases.json').read_bytes())
assert plan['suite']['generator']=='sha256-f32-v1'
wheels=list(Path('/input').glob('*.whl'))
assert len(wheels)==1
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
               '--target','/tmp/loader',str(wheels[0])],check=True,timeout=30,stdout=sys.stderr)
sys.path.insert(0,'/tmp/loader')
from cannery_kernels import load_cannery_package,runtime_fingerprint
assert runtime_fingerprint()==plan['runtime']
"""
CANDIDATE = r"""
package=Path('/input/admitted.tar').read_bytes()
artifact=json.loads(Path('/input/artifact.json').read_bytes())
assert hashlib.sha256(package).hexdigest()==artifact['sha256']
forward=load_cannery_package(package,artifact['sha256'],trust_remote_code=True).forward
"""
REFERENCE = r"""
root=Path('/input/tree')/plan['reference']['path']
sys.path[:0]=[str(root),'/input/tree']
module_spec=importlib.util.spec_from_file_location('_reference',root/plan['reference']['file'])
module=importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name]=module
module_spec.loader.exec_module(module)
reference=getattr(module,plan['reference']['function'])

def forward(x):
    value=reference(**{plan['input_name']:x})
    if isinstance(value,dict):
        assert set(value)=={plan['output_name']}
        return value[plan['output_name']]
    if isinstance(value,(tuple,list)):
        assert len(value)==1
        return value[0]
    return value
"""
RUN = r"""
def raw(value):
    cpu=value.detach().to('cpu').resolve_conj().resolve_neg().contiguous()
    return ctypes.string_at(cpu.data_ptr(),cpu.numel()*4)

def record(case,recipe,value,x):
    assert isinstance(value,torch.Tensor)
    assert tuple(value.shape)==tuple(recipe.shape) and value.dtype==torch.float32
    assert value.device==x.device
    return {'id':case['id'],'shape':list(value.shape),'dtype':'float32',
            'input_sha256':input_sha256(recipe,raw(x)),
            'data':base64.b64encode(raw(value)).decode('ascii')}

before_cases,after_cases,timings=[],[],[]
with torch.no_grad():
    for case in plan['suite']['cases']:
        name,values=next(iter(case['inputs'].items()))
        recipe=SimpleNamespace(**values)
        data=_generate(recipe,case['id'],name)
        assert input_sha256(recipe,data)==plan['input_sha256'][case['id']]
        x=(torch.frombuffer(bytearray(data),dtype=torch.float32).clone().reshape(recipe.shape)
           if data else torch.empty(recipe.shape,dtype=torch.float32)).to('npu')
        if recipe.layout=='transpose':
            backing=torch.empty(tuple(reversed(recipe.shape)),dtype=torch.float32,device='npu')
            view=backing.t()
            view.copy_(x)
            x=view
        strides=x.stride()
        if data:
            samples,before,after=measure(forward,x,torch.npu.synchronize,time.perf_counter_ns)
            timings.append({'id':case['id'],'samples_us':samples})
        else:
            before=forward(x)
            torch.npu.synchronize()
            after=forward(x)
            torch.npu.synchronize()
        assert tuple(x.shape)==tuple(recipe.shape) and x.dtype==torch.float32
        assert x.stride()==strides
        before_cases.append(record(case,recipe,before,x))
        after_cases.append(record(case,recipe,after,x))
        del x,before,after
wire.write(json.dumps({'protocol':PROTOCOL,
    'timing':{'protocol':PROTOCOL,'warmup':WARMUP,'cases':timings},
    'before':{'schema_version':1,'cases':before_cases},
    'after':{'schema_version':1,'cases':after_cases}},separators=(',',':'),allow_nan=False).encode())
wire.close()
"""


def runtime_program(kind: Literal["reference", "candidate"]) -> str:
    if kind not in {"reference", "candidate"}:
        raise ValueError("Unknown benchmark job kind")
    return (
        "from __future__ import annotations\n"
        "import base64,ctypes,hashlib,importlib.util,json,math,os,struct,subprocess,sys,time\n"
        "from pathlib import Path\nfrom types import SimpleNamespace\n"
        f"PROTOCOL={PROTOCOL!r}\nWARMUP={WARMUP}\nSAMPLES={SAMPLES}\n"
        + inspect.getsource(_generate)
        + inspect.getsource(input_sha256)
        + inspect.getsource(measure)
        + SETUP
        + (REFERENCE if kind == "reference" else CANDIDATE)
        + RUN
    )


@dataclass(frozen=True)
class TimingResult:
    samples: dict[str, list[float]]
    before: dict[str, TensorResult]
    after: dict[str, TensorResult]


def parse_runtime(raw: bytes, suite: RecipeSuite) -> TimingResult:
    # Two full output sets require up to ~86 MiB with base64; inputs are never stored.
    if type(raw) is not bytes or len(raw) > 128 * 1024 * 1024:
        raise ValueError("Benchmark output exceeds budget")
    try:
        payload = json.loads(raw, object_pairs_hook=unique_pairs)
        if (
            set(payload) != {"protocol", "timing", "before", "after"}
            or payload["protocol"] != PROTOCOL
        ):
            raise ValueError
        return TimingResult(
            parse_samples(json.dumps(payload["timing"]).encode(), suite),
            parse_results(json.dumps(payload["before"]).encode(), suite),
            parse_results(json.dumps(payload["after"]).encode(), suite),
        )
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        raise ValueError("Invalid benchmark output") from None

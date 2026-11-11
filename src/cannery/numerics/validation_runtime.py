"""A5 output protocol; expected CPU outputs never enter the candidate job."""

import base64
import inspect
import json
import math
import re
from dataclasses import dataclass

from cannery.numerics.recipes import RecipeSuite, _generate, input_sha256
from cannery.registry.manifest import unique_pairs

RUNTIME = r"""
# Reserve the result channel before importing native libraries: logs use stderr.
wire=os.fdopen(os.dup(sys.stdout.fileno()),'wb')
os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
wheels=list(Path('/input').glob('*.whl'))
assert len(wheels)==1
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
               '--target','/tmp/loader',str(wheels[0])],check=True,timeout=30,stdout=sys.stderr)
sys.path.insert(0,'/tmp/loader')
import torch,torch_npu
from cannery_kernels import load_cannery_package
assert torch.npu.device_count()==1
acl=ctypes.CDLL(os.environ['ASCEND_HOME_PATH']+'/lib64/libascendcl.so')
physical_fn=getattr(acl,'aclrtGetPhyDevIdByUserDevId',None) or acl.aclrtGetPhyDevIdByLogicDevId
physical_fn.argtypes=[ctypes.c_int32,ctypes.POINTER(ctypes.c_int32)]
physical_fn.restype=ctypes.c_int
physical=ctypes.c_int32(-1)
assert physical_fn(0,ctypes.byref(physical))==0
assert physical.value==int(os.environ['CANNERY_PHYSICAL_DEVICE'])
torch.npu.set_device(0)
package=Path('/input/admitted.tar').read_bytes()
reference=json.loads(Path('/input/artifact.json').read_bytes())
assert hashlib.sha256(package).hexdigest()==reference['sha256']
kernel=load_cannery_package(package,reference['sha256'],trust_remote_code=True)
plan=json.loads(Path('/input/cases.json').read_bytes())
assert plan['suite']['generator']=='sha256-f32-v1'


def raw(value):
    cpu=value.detach().to('cpu').resolve_conj().resolve_neg().contiguous()
    return ctypes.string_at(cpu.data_ptr(),cpu.numel()*4)


results=[]
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
    y=kernel.forward(x)
    torch.npu.synchronize()
    assert tuple(y.shape)==tuple(recipe.shape) and y.dtype==torch.float32 and y.device==x.device
    assert tuple(x.shape)==tuple(recipe.shape) and x.dtype==torch.float32
    results.append({'id':case['id'],'shape':list(y.shape),'dtype':'float32',
                    'input_sha256':input_sha256(recipe,raw(x)),
                    'data':base64.b64encode(raw(y)).decode('ascii')})
wire.write(json.dumps({'schema_version':1,'cases':results},separators=(',',':')).encode())
wire.close()
"""


def runtime_program() -> str:
    return (
        "from __future__ import annotations\n"
        "import base64,ctypes,hashlib,json,math,os,struct,subprocess,sys\n"
        "from pathlib import Path\nfrom types import SimpleNamespace\n"
        + inspect.getsource(_generate)
        + inspect.getsource(input_sha256)
        + RUNTIME
    )


@dataclass(frozen=True)
class TensorResult:
    data: bytes
    input_sha256: str


def parse_results(data: bytes, suite: RecipeSuite) -> dict[str, TensorResult]:
    if type(data) is not bytes or len(data) > 63 * 1024 * 1024:
        raise ValueError("Validation output exceeds budget")
    try:
        reply = json.loads(data, object_pairs_hook=unique_pairs)
        if (
            set(reply) != {"schema_version", "cases"}
            or type(reply["schema_version"]) is not int
            or reply["schema_version"] != 1
            or not isinstance(reply["cases"], list)
            or [r["id"] for r in reply["cases"]] != [c.id for c in suite.cases]
        ):
            raise ValueError
        outputs = {}
        for case, entry in zip(suite.cases, reply["cases"], strict=True):
            recipe = next(iter(case.inputs.values()))
            if (
                set(entry) != {"id", "shape", "dtype", "input_sha256", "data"}
                or entry["shape"] != list(recipe.shape)
                or not all(type(v) is int for v in entry["shape"])
                or entry["dtype"] != "float32"
                or re.fullmatch(r"[a-f0-9]{64}", entry["input_sha256"]) is None
            ):
                raise ValueError
            values = base64.b64decode(entry["data"], validate=True)
            if len(values) != 4 * math.prod(recipe.shape):
                raise ValueError
            outputs[case.id] = TensorResult(values, entry["input_sha256"])
        return outputs
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise ValueError("Invalid numerical validation output") from None

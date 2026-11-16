"""Generate named tensor inputs on-device; never send cached oracle outputs."""

import inspect

from cannery.numerics.recipes_v2 import ITEM_BYTES, _float_bytes, _generate_v2, input_sha256_v2
from cannery.numerics.validation_runtime import RUNTIME

TENSOR_RUNTIME = r"""
plan=json.loads(Path('/input/cases.json').read_bytes())
assert plan['suite']['schema_version']==2
assert plan['suite']['generator']=='sha256-tensor-v2'
assert plan['contract']['schema_version']==2
assert kernel.metadata['schema_version']==3 and kernel.metadata['abi']=='tensor-v2'
input_order=[item['name'] for item in kernel.metadata['variants'][0]['inputs']]


def raw(value):
    cpu=value.detach().to('cpu').contiguous()
    return ctypes.string_at(cpu.data_ptr(),cpu.numel()*cpu.element_size())


def state(value):
    return (tuple(value.shape),tuple(value.stride()),value.dtype,value.device,
            value.data_ptr(),value.storage_offset(),value.is_conj(),value.is_neg(),
            torch_npu.get_npu_format(value))


def tensor(data,recipe):
    dtype=getattr(torch,recipe.dtype)
    cpu=(torch.frombuffer(bytearray(data),dtype=dtype).clone().reshape(recipe.shape)
         if data else torch.empty(recipe.shape,dtype=dtype))
    result=torch_npu.empty_with_format(recipe.shape,dtype=dtype,device='npu',acl_format=2)
    result.copy_(cpu)
    return result


results=[]
for case in plan['suite']['cases']:
    definition=plan['contract']['cases'][case['id']]
    assert set(input_order)==set(case['inputs']) and len(input_order)==len(case['inputs'])
    inputs,recipes,before={},{},{}
    for name,values in case['inputs'].items():
        recipe=SimpleNamespace(**values)
        assert recipe.layout=='contiguous'
        recipe.nbytes=math.prod(recipe.shape)*ITEM_BYTES[recipe.dtype]
        data=_generate_v2(recipe,case['id'],name)
        assert input_sha256_v2(recipe,data)==plan['input_sha256'][case['id']][name]
        inputs[name]=tensor(data,recipe)
        recipes[name]=recipe
        before[name]=state(inputs[name])
    outputs=kernel.forward(*(inputs[name] for name in input_order),
                           **{name:s['value'] for name,s in definition['scalars'].items()})
    torch.npu.synchronize()
    assert isinstance(outputs,dict) and set(outputs)==set(definition['outputs'])
    for name,value in inputs.items():
        assert state(value)==before[name]
    records={}
    for name,descriptor in definition['outputs'].items():
        value=outputs[name]
        assert tuple(value.shape)==tuple(descriptor['shape'])
        assert value.dtype==getattr(torch,descriptor['dtype'])
        assert value.device==inputs[input_order[0]].device and value.is_contiguous()
        assert not value.is_neg() and not value.is_conj() and torch_npu.get_npu_format(value)==2
        alias=definition['aliases'].get(name)
        if alias is not None:
            assert value is inputs[alias]
        else:
            for original in inputs.values():
                assert not (value.numel() and original.numel() and
                    value.data_ptr()<original.data_ptr()+original.numel()*original.element_size()
                    and
                    original.data_ptr()<value.data_ptr()+value.numel()*value.element_size())
        records[name]={'shape':list(value.shape),'dtype':descriptor['dtype'],
                       'data':base64.b64encode(raw(value)).decode('ascii')}
    results.append({'id':case['id'],'outputs':records,'inputs':{
        name:input_sha256_v2(recipes[name],raw(value)) for name,value in inputs.items()}})
wire.write(json.dumps({'schema_version':2,'cases':results},separators=(',',':')).encode())
wire.close()
"""


def tensor_runtime_program() -> str:
    # Reuse the pinned wheel/device/bootstrap contract without changing v1 bytes.
    bootstrap = RUNTIME.split("plan=json.loads", 1)[0]
    return (
        "from __future__ import annotations\n"
        "import base64,ctypes,hashlib,json,math,os,struct,subprocess,sys\n"
        "from pathlib import Path\nfrom types import SimpleNamespace\n"
        + f"ITEM_BYTES={ITEM_BYTES!r}\n"
        + inspect.getsource(_float_bytes)
        + inspect.getsource(_generate_v2)
        + inspect.getsource(input_sha256_v2)
        + bootstrap
        + TENSOR_RUNTIME
    )

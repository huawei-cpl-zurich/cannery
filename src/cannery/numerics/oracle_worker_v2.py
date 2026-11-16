"""Trusted CPU tensor oracle source; canonical reference runs under another UID."""

import inspect
from string import Template

from cannery.numerics.oracle_worker import SUPERVISOR_PREFIX, SUPERVISOR_SUFFIX
from cannery.numerics.recipes_v2 import ITEM_BYTES, _float_bytes, _generate_v2

CHILD_V2 = r"""
import base64,contextlib,ctypes,importlib.util,json,sys
from pathlib import Path
import torch

def raw(value):
    contiguous=value.detach().resolve_conj().resolve_neg().contiguous()
    return ctypes.string_at(contiguous.data_ptr(),contiguous.numel()*contiguous.element_size())

first=json.loads(sys.stdin.readline())
root=Path('/work')/first['path']
sys.path[:0]=[str(root),'/work']
with open('/dev/null','w') as quiet,contextlib.redirect_stdout(quiet):
    spec=importlib.util.spec_from_file_location('_canonical_reference',root/first['file'])
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    forward=getattr(module,first['function'])
print(json.dumps({'nonce':first['nonce'],'ready':True}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    with open('/dev/null','w') as quiet,contextlib.redirect_stdout(quiet):
        inputs={}
        snapshots={}
        for name,description in request['inputs'].items():
            data=base64.b64decode(description['data'],validate=True)
            dtype=getattr(torch,description['dtype'])
            shape=description['shape']
            value=(torch.frombuffer(bytearray(data),dtype=dtype).clone().reshape(shape)
                   if data else torch.empty(shape,dtype=dtype))
            inputs[name]=value
            storage=(value.data_ptr(),value.is_neg(),value.is_conj())
            snapshots[name]=(tuple(shape),dtype,value.stride(),data,storage)
        scalars={name:record['value'] for name,record in request['scalars'].items()}
        with torch.no_grad():
            outputs=forward(**inputs,**scalars)
        if not isinstance(outputs,dict) or set(outputs)!=set(request['outputs']):
            raise ValueError('Reference must return every named output')
        mutable=set(request['aliases'].values())
        for name,value in inputs.items():
            shape,dtype,strides,data,storage=snapshots[name]
            if (tuple(value.shape)!=shape or value.dtype!=dtype or value.stride()!=strides
                    or (value.data_ptr(),value.is_neg(),value.is_conj())!=storage
                    or value.device.type!='cpu' or name not in mutable and raw(value)!=data):
                raise ValueError('Reference changed input metadata or read-only values')
        encoded={}
        for name,description in request['outputs'].items():
            value=outputs[name]
            if (not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                    or value.dtype!=getattr(torch,description['dtype'])
                    or tuple(value.shape)!=tuple(description['shape'])):
                raise ValueError('Reference output differs from its tensor contract')
            if name in request['aliases'] and value is not inputs[request['aliases'][name]]:
                raise ValueError('Mutable output must return the declared input object')
            encoded[name]={'shape':description['shape'],'dtype':description['dtype'],
                          'data':base64.b64encode(raw(value)).decode('ascii')}
    print(json.dumps({'nonce':request['nonce'],'outputs':encoded}),flush=True)
"""

TENSOR_CASES = r"""        for case in plan['suite']['cases']:
            contract=plan['contract']['cases'][case['id']]
            inputs={}
            for name,values in case['inputs'].items():
                assert values['layout']=='contiguous'
                data=_generate_v2(SimpleNamespace(**values),case['id'],name)
                inputs[name]={'shape':values['shape'],'dtype':values['dtype'],
                              'data':base64.b64encode(data).decode('ascii')}
            response=exchange(inputs=inputs,outputs=contract['outputs'],
                              aliases=contract['aliases'],scalars=contract['scalars'])
            assert set(response['outputs'])==set(contract['outputs'])
            outputs[case['id']]={}
            for name,description in contract['outputs'].items():
                record=response['outputs'][name]
                assert (record['shape']==description['shape']
                        and record['dtype']==description['dtype'])
                value=base64.b64decode(record['data'],validate=True)
                dtype=description['dtype']
                assert len(value)==math.prod(description['shape'])*ITEM_BYTES[dtype]
                if dtype=='bool':assert all(v<=1 for v in value)
                if not definition['validation']['equal_nan']:
                    if dtype in ('float16','float32'):
                        code='<e' if dtype=='float16' else '<f'
                        assert not any(math.isnan(v) for (v,) in struct.iter_unpack(code,value))
                    elif dtype=='bfloat16':
                        assert not any(math.isnan(struct.unpack('<f',b'\0\0'+value[i:i+2])[0])
                                       for i in range(0,len(value),2))
                outputs[case['id']][name]=base64.b64encode(value).decode('ascii')
"""


def worker_source_v2(*, checkpoint: bool = False) -> str:
    """Emit a stdlib supervisor; caller validates the frozen suite/contract first."""
    return (
        "from __future__ import annotations\n"
        "import base64,hashlib,importlib.metadata,json,math,os,secrets,signal,struct\n"
        "import subprocess,sys\n"
        "from pathlib import Path\nfrom types import SimpleNamespace\n"
        + "OUTPUT_OWNER=(0,0)\n"
        + f"RESULT_FILE={'/output/result.json' if checkpoint else None!r}\n"
        + f"ITEM_BYTES={ITEM_BYTES!r}\n"
        + f"CHILD={CHILD_V2!r}\n"
        + inspect.getsource(_float_bytes)
        + inspect.getsource(_generate_v2)
        + Template(SUPERVISOR_PREFIX).substitute(reply_limit="48*1024*1024")
        + TENSOR_CASES
        + SUPERVISOR_SUFFIX
    )

"""Offline named-tensor timing runner; expected outputs stay in the controller."""

import inspect
import json
from dataclasses import dataclass
from typing import Literal

from cannery.benchmark.contract import SAMPLES, WARMUP
from cannery.benchmark.runtime import CANDIDATE, SETUP
from cannery.benchmark.tensor_timing import TENSOR_PROTOCOL, measure_tensor, parse_tensor_samples
from cannery.numerics.recipes_v2 import (
    ITEM_BYTES,
    RecipeSuiteV2,
    _float_bytes,
    _generate_v2,
    input_sha256_v2,
)
from cannery.numerics.tensor_contract import TensorContractV2
from cannery.numerics.validation_protocol_v2 import TensorResultsV2, parse_tensor_results
from cannery.numerics.validation_runtime_v2 import TENSOR_RUNTIME, TENSOR_THREADS
from cannery.registry.manifest import unique_pairs

REFERENCE = r"""
root=Path('/input/tree')/plan['reference']['path']
sys.path[:0]=[str(root),'/input/tree']
spec=importlib.util.spec_from_file_location('_reference',root/plan['reference']['file'])
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
reference=getattr(module,plan['reference']['function'])
def forward(*args,**scalars):
    return reference(**dict(zip(plan['input_order'],args,strict=True)),**scalars)
"""

RUN = r"""
assert plan['contract']['schema_version']==2
input_order=plan['input_order']
assert len(input_order)==len(set(input_order))
before_cases,after_cases,timings=[],[],[]
with torch.no_grad():
    for case in plan['suite']['cases']:
        definition=plan['contract']['cases'][case['id']]
        assert set(input_order)==set(case['inputs'])
        inputs,recipes,states,originals={},{},{},{}
        mutable=set(definition['aliases'].values())
        for name,values in case['inputs'].items():
            recipe=SimpleNamespace(**values)
            assert recipe.layout=='contiguous'
            recipe.nbytes=math.prod(recipe.shape)*ITEM_BYTES[recipe.dtype]
            data=_generate_v2(recipe,case['id'],name)
            assert input_sha256_v2(recipe,data)==plan['input_sha256'][case['id']][name]
            inputs[name]=tensor(data,recipe)
            recipes[name]=recipe
            states[name]=state(inputs[name])
            if name in mutable: originals[name]=inputs[name].clone()

        def reset():
            for name in mutable: inputs[name].copy_(originals[name])

        def invoke():
            return forward(*(inputs[name] for name in input_order),
                **{name:scalar['value'] for name,scalar in definition['scalars'].items()})

        def record(outputs):
            assert isinstance(outputs,dict) and set(outputs)==set(definition['outputs'])
            for name,value in inputs.items(): assert state(value)==states[name]
            records={}
            for name,desc in definition['outputs'].items():
                value=outputs[name]
                assert isinstance(value,torch.Tensor)
                assert list(value.shape)==desc['shape']
                assert value.dtype==getattr(torch,desc['dtype'])
                assert value.device==inputs[input_order[0]].device and value.is_contiguous()
                assert not value.is_neg() and not value.is_conj()
                assert torch_npu.get_npu_format(value)==2
                alias=definition['aliases'].get(name)
                if alias is not None: assert value is inputs[alias]
                else:
                    for original in inputs.values():
                        assert not (value.numel() and original.numel() and
                            value.data_ptr()<original.data_ptr()+original.numel()*original.element_size()
                            and
                            original.data_ptr()<value.data_ptr()+value.numel()*value.element_size())
                records[name]={'shape':list(value.shape),'dtype':desc['dtype'],
                    'data':base64.b64encode(raw(value)).decode('ascii')}
            return {'id':case['id'],'outputs':records,'inputs':{
                name:input_sha256_v2(recipes[name],raw(value)) for name,value in inputs.items()}}

        nonempty=any(recipe.nbytes for recipe in recipes.values()) or any(
            math.prod(desc['shape']) for desc in definition['outputs'].values())
        if nonempty:
            samples,before,after=measure_tensor(invoke,reset,torch.npu.synchronize,
                                                time.perf_counter_ns,snapshot=record)
            timings.append({'id':case['id'],'samples_us':samples})
        else:
            reset();before=invoke();torch.npu.synchronize();before=record(before)
            reset();after=invoke();torch.npu.synchronize();after=record(after)
        before_cases.append(before);after_cases.append(after)
wire.write(json.dumps({'protocol':TENSOR_PROTOCOL,
    'timing':{'protocol':TENSOR_PROTOCOL,'warmup':WARMUP,'cases':timings},
    'before':{'schema_version':2,'cases':before_cases},
    'after':{'schema_version':2,'cases':after_cases}},separators=(',',':'),allow_nan=False).encode())
wire.close()
"""


def tensor_benchmark_program(kind: Literal["reference", "candidate"]) -> str:
    if kind not in {"reference", "candidate"}:
        raise ValueError("Unknown tensor benchmark kind")
    # Share the numerical runner's byte conversion and descriptor checks.
    helpers = "def raw" + TENSOR_RUNTIME.split("def raw", 1)[1].split("\n\nresults=[]", 1)[0]
    return (
        "from __future__ import annotations\n"
        "import base64,ctypes,hashlib,importlib.util,json,math,os,struct,subprocess,sys,time\n"
        "from pathlib import Path\nfrom types import SimpleNamespace\n"
        f"TENSOR_PROTOCOL={TENSOR_PROTOCOL!r}\nWARMUP={WARMUP}\nSAMPLES={SAMPLES}\n"
        f"ITEM_BYTES={ITEM_BYTES!r}\n"
        + inspect.getsource(_float_bytes)
        + inspect.getsource(_generate_v2)
        + inspect.getsource(input_sha256_v2)
        + inspect.getsource(measure_tensor)
        + TENSOR_THREADS
        + SETUP.replace("sha256-f32-v1", "sha256-tensor-v2")
        + (REFERENCE if kind == "reference" else CANDIDATE)
        + helpers
        + RUN
    )


@dataclass(frozen=True)
class TensorTimingResult:
    samples: dict[str, list[float]]
    before: TensorResultsV2
    after: TensorResultsV2


def parse_tensor_runtime(
    raw: bytes, suite: RecipeSuiteV2, contract: TensorContractV2
) -> TensorTimingResult:
    if type(raw) is not bytes or len(raw) > 128 * 1024 * 1024:
        raise ValueError("Tensor benchmark output exceeds budget")
    try:
        payload = json.loads(raw, object_pairs_hook=unique_pairs)
        if (
            set(payload) != {"protocol", "timing", "before", "after"}
            or payload["protocol"] != TENSOR_PROTOCOL
        ):
            raise ValueError
        return TensorTimingResult(
            parse_tensor_samples(json.dumps(payload["timing"]).encode(), suite, contract),
            parse_tensor_results(json.dumps(payload["before"]).encode(), suite, contract),
            parse_tensor_results(json.dumps(payload["after"]).encode(), suite, contract),
        )
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        raise ValueError("Invalid tensor benchmark output") from None

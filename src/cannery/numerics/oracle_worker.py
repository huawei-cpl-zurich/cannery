"""Trusted CPU supervisor source; author reference runs under a separate UID."""

import inspect

from cannery.numerics.recipes import _generate

CHILD = r"""
import base64,contextlib,ctypes,importlib.util,json,sys
from pathlib import Path
import torch


def raw(value):
    contiguous=value.detach().resolve_conj().resolve_neg().contiguous()
    return ctypes.string_at(contiguous.data_ptr(),contiguous.numel()*4)


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
        data=base64.b64decode(request['data'],validate=True)
        shape=request['shape']
        x=(torch.frombuffer(bytearray(data),dtype=torch.float32).clone().reshape(shape)
           if data else torch.empty(shape,dtype=torch.float32))
        if request['layout']=='transpose':
            backing=torch.empty(tuple(reversed(shape)),dtype=torch.float32)
            view=backing.t()
            view.copy_(x)
            x=view
        strides=x.stride()
        with torch.no_grad():
            output=forward(**{request['input_name']:x})
        if (tuple(x.shape)!=tuple(shape) or x.dtype!=torch.float32
                or x.stride()!=strides or raw(x)!=data):
            raise ValueError('Reference mutated its input')
        if isinstance(output,dict):
            if set(output)!={request['output_name']}:raise ValueError('Wrong output names')
            output=output[request['output_name']]
        elif isinstance(output,(tuple,list)):
            if len(output)!=1:raise ValueError('Wrong output arity')
            output=output[0]
        if (not isinstance(output,torch.Tensor) or output.dtype!=torch.float32
                or output.device.type!='cpu' or tuple(output.shape)!=tuple(shape)):
            raise ValueError('Reference output must preserve float32 CPU shape')
        value=base64.b64encode(raw(output)).decode('ascii')
    print(json.dumps({'nonce':request['nonce'],'shape':shape,'dtype':'float32','data':value}),flush=True)
"""

SUPERVISOR = r"""

def publish(result):
    encoded=json.dumps(result).encode()
    if RESULT_FILE is None:
        print(encoded.decode(),flush=True)
        return
    uid,gid=os.geteuid(),os.getegid()
    try:
        os.setegid(OUTPUT_OWNER[1])
        os.seteuid(OUTPUT_OWNER[0])
        path=Path(RESULT_FILE)
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        temporary.replace(path)
        descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        os.seteuid(uid)
        os.setegid(gid)


def expired(*_):
    if RESULT_FILE is not None:publish({'error':'reference_timeout'})
    os._exit(124)


def demote():
    os.setgroups([])
    os.setgid(10001)
    os.setuid(10001)


def main(plan):
    global OUTPUT_OWNER
    OUTPUT_OWNER=plan.get('output_owner',[0,0])
    assert len(OUTPUT_OWNER)==2 and all(type(v) is int and v>=0 for v in OUTPUT_OWNER)
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(70)
    assert sys.byteorder=='little'
    assert f'{sys.version_info.major}.{sys.version_info.minor}'==plan['environment']['python']
    assert importlib.metadata.version('torch')==plan['environment']['torch']
    root=Path('/work')
    total=0
    assert len(plan['files'])<=256
    for name,encoded in plan['files'].items():
        relative=Path(name)
        assert (not relative.is_absolute() and '..' not in relative.parts
                and '.git' not in relative.parts)
        destination=root/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        data=base64.b64decode(encoded,validate=True)
        total+=len(data)
        assert total<=16*1024*1024
        destination.write_bytes(data)
        destination.chmod(0o444)
    worker=subprocess.Popen([sys.executable,'-I','-c',CHILD],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,preexec_fn=demote,
        env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','OMP_NUM_THREADS':'1',
             'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    try:
        def exchange(**request):
            nonce=secrets.token_hex(16)
            worker.stdin.write(json.dumps(dict(nonce=nonce,**request))+'\n')
            worker.stdin.flush()
            result=json.loads(worker.stdout.readline(6*1024*1024))
            assert isinstance(result,dict) and result.get('nonce')==nonce
            return result
        definition=plan['spec']
        ready=exchange(path=plan['path'],**definition['reference'])
        assert ready.get('ready') is True
        print('oracle reference initialized',file=sys.stderr,flush=True)
        outputs={}
        for case in plan['suite']['cases']:
            name,values=next(iter(case['inputs'].items()))
            recipe=SimpleNamespace(**values)
            data=_generate(recipe,case['id'],name)
            response=exchange(input_name=name,output_name=definition['interface']['outputs'][0],
                shape=values['shape'],layout=values['layout'],data=base64.b64encode(data).decode('ascii'))
            assert response.get('shape')==values['shape'] and response.get('dtype')=='float32'
            value=base64.b64decode(response['data'],validate=True)
            assert len(value)==4*math.prod(values['shape'])
            if not definition['validation']['equal_nan']:
                assert not any(math.isnan(v) for (v,) in struct.iter_unpack('<f',value))
            outputs[case['id']]=base64.b64encode(value).decode('ascii')
    finally:
        worker.kill()
        worker.wait(timeout=2)
    publish({'outputs':outputs})


try:
    main(json.loads(Path('/input/plan.json').read_bytes()))
except Exception:
    if RESULT_FILE is not None:publish({'error':'reference_failed'})
    raise
"""


def worker_source(*, checkpoint: bool = False) -> str:
    return (
        "from __future__ import annotations\n"
        "import base64,hashlib,importlib.metadata,json,math,os,secrets,signal,struct\n"
        "import subprocess,sys\n"
        "from pathlib import Path\nfrom types import SimpleNamespace\n"
        + "OUTPUT_OWNER=(0,0)\n"
        + f"RESULT_FILE={'/output/result.json' if checkpoint else None!r}\n"
        + f"CHILD={CHILD!r}\n"
        + inspect.getsource(_generate)
        + SUPERVISOR
    )

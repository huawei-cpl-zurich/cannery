"""Generate trusted host supervision for isolated Ascend C build/runtime jobs."""

import json
from pathlib import PurePosixPath
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from cannery.devices import Device, matches_soc
from cannery.registry.manifest import StrictModel


class CatlassToolchain(StrictModel):
    """Trusted operator pins and the prebuilt, read-only AIV helper cache."""

    version: str = Field(min_length=1, max_length=128)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    execution_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    hivmc_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bitcode_key: str = Field(pattern=r"^[a-f0-9]{16}$")
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bitcode_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bitcode_root: str

    @model_validator(mode="after")
    def confined_cache(self) -> Self:
        path = PurePosixPath(self.bitcode_root)
        if (
            not path.is_absolute()
            or len(path.parts) < 5
            or ".." in path.parts
            or any(c in self.bitcode_root for c in "\n\r\x00,")
        ):
            raise ValueError("Expected a specific absolute helper bitcode cache")
        return self


class WorkerProfile(StrictModel):
    device: Device = "a5"
    toolchain_in_image: bool = False
    target: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    runtime: str | None = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    workspace: str
    image: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    npu: int = Field(ge=0, le=63)
    device_group: int = Field(ge=0)
    toolchain_read_group: int | None = Field(default=None, ge=0)
    readonly_mounts: tuple[str, ...]
    transport_devices: tuple[str, ...] = ()
    fingerprint: dict[str, str]
    compiler: str = Field(min_length=1, max_length=4096)
    catlass: CatlassToolchain | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value: dict[str, Any] = handler(self)
        # Keep existing A5 attempt contexts byte-identical across this upgrade.
        if self.device == "a5":
            value.pop("device", None)
        if not self.toolchain_in_image:
            value.pop("toolchain_in_image", None)
        return value

    @model_validator(mode="after")
    def confined_profile(self) -> Self:
        for value in (self.workspace, *self.readonly_mounts):
            path = PurePosixPath(value)
            if (
                not path.is_absolute()
                or ".." in path.parts
                or len(path.parts) < 3
                or any(c in value for c in "\n\r\x00,")
            ):
                raise ValueError("Expected a specific absolute worker path")
        import re

        if any(
            not re.fullmatch(r"/dev/uburma/udmac[0-9]+d[0-9]+e[0-9]+", value)
            for value in self.transport_devices
        ):
            raise ValueError("Unsupported Unified Bus transport device")
        driver_roots = ("/usr/local/Ascend/driver", "/usr/local/dcmi")
        allowed_prefixes = tuple(value + "/" for value in driver_roots) + (
            "/usr/lib64/",
            "/usr/lib/aarch64-linux-gnu/",
        )
        if any(
            value not in driver_roots and not value.startswith(allowed_prefixes)
            for value in self.readonly_mounts
        ):
            raise ValueError("Only driver and runtime library mounts are supported")
        if set(self.fingerprint) != {"machine", "python", "torch", "torch_npu", "cann", "soc"}:
            raise ValueError("Incomplete runtime fingerprint")
        if (
            self.fingerprint["machine"] != "aarch64"
            or not matches_soc(self.device, self.fingerprint["soc"])
            or not all(self.fingerprint.values())
        ):
            raise ValueError("Device and runtime fingerprint disagree")
        if self.device == "a3" and self.catlass is not None:
            raise ValueError("CATLASS DSL toolchains require A5")
        if self.toolchain_in_image and self.catlass is not None:
            raise ValueError("CATLASS DSL requires host-mounted pinned toolchains")
        return self


# Executed only by the trusted remote host supervisor, never imported by author code.
SUPERVISOR = r"""
import fcntl, hashlib, json, os, re, resource, shutil, stat, subprocess, sys
from pathlib import Path

root = Path(config['workspace'])
root.mkdir(mode=0o700, parents=True, exist_ok=True)
attempt = root / token
source, output = attempt / 'input', attempt / 'output'
names = {kind: 'cannery-' + token + '-' + kind for kind in ['compile', 'runtime']}
lease = root / ('device-' + str(config['npu']) + '.owner')
lock = root / ('device-' + str(config['npu']) + '.lock')

def remove_containers():
    for name in names.values():
        removed = subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=15)
        if removed.returncode:
            remaining = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=^/' + name + '$',
                                        '--format', '{{.ID}}'], capture_output=True,
                                       text=True, timeout=15)
            if remaining.returncode or remaining.stdout.strip():
                raise RuntimeError('Container cleanup uncertain; preserve device ownership')

def check_device():
    for kind in ['health', 'usages', 'proc-mem', 'current-fault-event']:
        try:
            result = subprocess.run(['npu-smi', 'info', '-t', kind, '-i', str(config['npu'])],
                                    capture_output=True, text=True, timeout=12)
        except (OSError, subprocess.TimeoutExpired):
            raise SystemExit(75)
        text = result.stdout
        if result.returncode or not text.strip():
            raise SystemExit(75)
        if kind == 'health' and not re.search(r'Health Status\s*:\s*OK', text):
            raise SystemExit(75)
        if kind == 'usages':
            memory = re.search(r'HBM Usage Rate\(%\)\s*:\s*(\d+)', text)
            if not memory or int(memory.group(1)) >= 50:
                raise SystemExit(75)
            if not re.search(r'NPU Utilization\(%\)\s*:\s*0\b', text):
                raise SystemExit(75)
        if kind == 'proc-mem' and 'No process in device' not in text:
            raise SystemExit(75)
        if kind == 'current-fault-event' and not re.search(r'Fault Event Count\s*:\s*0\b', text):
            raise SystemExit(75)

def device_identity():
    reply=subprocess.run(['npu-smi','info','-t','board','-i',str(config['npu'])],
                         capture_output=True,text=True,timeout=12,check=True).stdout
    values={}
    for name in ['Serial Number','DDie ID']:
        found=re.search(r'^\s*'+name+r'\s*:\s*(\S[^\r\n]*)$',reply,re.M)
        if not found or found.group(1).strip() in {'NA','N/A','Unknown'}:
            raise ValueError('Stable physical device identity unavailable')
        values[name]=found.group(1).strip()
    version=Path('/usr/local/Ascend/driver/version.info').read_bytes()
    if not 0<len(version)<=16384:raise ValueError('Invalid driver identity')
    return {'hardware_id':hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest(),
            'driver':hashlib.sha256(version).hexdigest()}

if phase == 'identify':
    print(json.dumps(device_identity(),sort_keys=True))
    raise SystemExit(0)

if phase == 'prepare':
    attempt.mkdir(mode=0o700, exist_ok=True)
    source.mkdir(mode=0o755, exist_ok=True)
    source.chmod(0o755)
    output.mkdir(mode=0o777, exist_ok=True)
    output.chmod(0o777)
    raise SystemExit(0)

if phase == 'cleanup':
    remove_containers()
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(75)
        if lease.exists() and lease.read_text() == token:
            lease.unlink()
    if attempt.exists():
        shutil.rmtree(attempt)
    raise SystemExit(0)

benchmark=config.get('benchmark')
if benchmark and device_identity()!=benchmark:
    raise ValueError('Performance device cohort changed')
output_limit=128*1024*1024 if benchmark else 63*1024*1024

args = ['docker', 'run', '--rm', '--init', '--pull', 'never', '--name', names[phase],
        '--network', 'none', '--user', '10001:10001', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--read-only', '--log-driver', 'none',
        '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=512m' if phase == 'runtime'
        else '/tmp:rw,noexec,nosuid,nodev,size=512m', '--shm-size', '512m', '--cpus', '4',
        '--memory', '8g' if phase == 'runtime' else '4g', '--pids-limit', '256',
        '--ulimit', 'fsize=' + ('134217728:134217728' if benchmark else '67108864:67108864'),
        '--env', 'HOME=/tmp',
        '--env', 'ASCEND_PROCESS_LOG_PATH=/tmp/ascend/log',
        '--mount', f'type=bind,src={source},dst=/input,readonly']
if not config.get('toolchain_in_image', False):
    cann = os.environ['ASCEND_HOME_PATH']
    if (not Path(cann).is_absolute() or len(Path(cann).parts) < 4
            or not cann.startswith(('/opt/', '/usr/local/Ascend/'))):
        raise ValueError('Invalid registered CANN root')
    args += ['--mount', f'type=bind,src={cann},dst={cann},readonly']
    for key in ['PATH', 'LD_LIBRARY_PATH', 'ASCEND_HOME_PATH',
                'ASCEND_TOOLKIT_HOME', 'ASCEND_OPP_PATH']:
        args += ['--env', key]
if config['toolchain_read_group'] is not None:
    args += ['--group-add', str(config['toolchain_read_group'])]
held = None
if phase == 'compile':
    plan = json.loads((source / 'build.json').read_bytes())
    if plan.get('wrapper_name', 'launch.asc') == 'launch.py':
        pins = config.get('catlass')
        if not pins or plan.get('catlass') != pins:
            raise ValueError('CATLASS toolchain pins unavailable or mismatched')
        ir = Path(os.environ['CATLASS_DSL_ASCENDNPU_IR_ROOT'])
        if (not ir.is_absolute() or len(ir.parts) < 4 or '..' in ir.parts
                or not str(ir).startswith('/opt/') or ',' in str(ir)):
            raise ValueError('Invalid registered AscendNPU-IR root')
        args += ['--tmpfs', '/tmp/cache:rw,noexec,nosuid,nodev,size=256m,mode=1777',
                 '--env', 'CATLASS_DSL_CACHE_DIR=/tmp/cache',
                 '--mount', f"type=bind,src={pins['bitcode_root']},dst=/tmp/cache/bc,readonly",
                 '--mount', f'type=bind,src={ir.parent},dst={ir.parent},readonly']
        for key in ['PYTHONPATH', 'CATLASS_DSL_ASCENDNPU_IR_ROOT',
                    'CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR']:
            args += ['--env', key]
    command = COMPILE
else:
    held = lock.open('a')
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(75)
    # A stale owner is quarantined until its retained operation is reconciled.
    if lease.exists():
        raise SystemExit(75)
    check_device()
    with lease.open('x') as marker:
        marker.write(token)
        marker.flush()
        os.fsync(marker.fileno())
    args += ['--group-add', str(config['device_group']), '--ulimit', 'memlock=-1:-1',
             '--env', 'ASCEND_RT_VISIBLE_DEVICES=0',
             '--env', 'CANNERY_PHYSICAL_DEVICE=' + str(config['npu']),
             '--tmpfs', '/ascend:rw,noexec,nosuid,nodev,size=32m,mode=1777',
             '--env', 'ASCEND_VISIBLE_DEVICES=' + str(config['npu'])]
    for path in [f"/dev/davinci{config['npu']}", '/dev/davinci_manager',
                 '/dev/hisi_hdc', '/dev/ummu/tid']:
        args += ['--device', path]
    for path in config['transport_devices']:
        args += ['--device', path]
    for path in config['readonly_mounts']:
        args += ['--mount', f'type=bind,src={path},dst={path},readonly']
    command = RUNTIME
args += [config['image'], 'python', '-c', command]
resource.setrlimit(resource.RLIMIT_FSIZE, ((128 if benchmark else 64) * 1024 * 1024,) * 2)
capturing = phase == 'compile' or (phase == 'runtime' and config.get('capture_runtime', False))
result_path = output / ('kernel.so' if phase == 'compile' else 'results.json')
try:
    with (attempt / (phase + '.log')).open('w+b') as log, \
            open(result_path if capturing else os.devnull, 'wb') as binary_output:
        try:
            result = subprocess.run(args, stdout=binary_output if capturing else log,
                                    stderr=log, timeout=180)
        finally:
            log.seek(max(0, os.fstat(log.fileno()).st_size - 16384))
            sys.stderr.buffer.write(b'Last 16 KiB of isolated job log:\n' + log.read(16384))
            sys.stderr.flush()
    if result.returncode:
        raise SystemExit(1)
    if benchmark:
        check_device()
        if device_identity()!=benchmark:raise ValueError('Performance device cohort changed')
    if capturing:
        binary = result_path
        if (not stat.S_ISREG(binary.lstat().st_mode)
                or not 0 < binary.stat().st_size <= output_limit):
            raise ValueError('Invalid captured output')
        print(hashlib.sha256(binary.read_bytes()).hexdigest())
finally:
    remove_containers()
    if held is not None:
        if lease.exists() and lease.read_text() == token:
            lease.unlink()
        held.close()
"""

COMPILE = r"""
import json, pathlib, subprocess, sys
plan = json.loads(pathlib.Path('/input/build.json').read_text())
version = subprocess.run(['bisheng', '--version'], capture_output=True, text=True,
                         check=True, timeout=10).stdout
assert plan['compiler'] in version, 'compiler fingerprint mismatch'
print(version[:4096], file=sys.stderr, flush=True)
work = pathlib.Path('/tmp/work')
work.mkdir()
(work / 'source').symlink_to('/input/tree/' + plan['path'], target_is_directory=True)
wrapper = plan.get('wrapper_name', 'launch.asc')
assert wrapper in ['launch.asc', 'launch.py']
(work / wrapper).write_text(plan['wrapper'])
subprocess.run(plan['command'], cwd=work, stdout=sys.stderr, check=True, timeout=120)
binary = work / 'kernel.so'
assert binary.is_file() and not binary.is_symlink() and binary.stat().st_size <= 63*1024*1024
sys.stdout.buffer.write(binary.read_bytes())
"""

RUNTIME = r"""
import ctypes, hashlib, json, os, pathlib, subprocess, sys
wheels = list(pathlib.Path('/input').glob('*.whl'))
assert len(wheels) == 1
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                '--target', '/tmp/loader', str(wheels[0])], check=True, timeout=30)
sys.path.insert(0, '/tmp/loader')
import torch, torch_npu
from cannery_kernels import load_cannery_package
assert torch.npu.device_count() == 1, 'runtime device mask failed'
acl = ctypes.CDLL(os.environ['ASCEND_HOME_PATH'] + '/lib64/libascendcl.so')
get_physical = getattr(acl, 'aclrtGetPhyDevIdByUserDevId', acl.aclrtGetPhyDevIdByLogicDevId)
get_physical.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
get_physical.restype = ctypes.c_int
physical = ctypes.c_int32(-1)
assert get_physical(0, ctypes.byref(physical)) == 0
assert physical.value == int(os.environ['CANNERY_PHYSICAL_DEVICE']), 'physical device mismatch'
torch.npu.set_device(0)
package = pathlib.Path('/input/admitted.tar').read_bytes()
reference = json.loads(pathlib.Path('/input/artifact.json').read_text())
assert hashlib.sha256(package).hexdigest() == reference['sha256']
kernel = load_cannery_package(package, reference['sha256'], trust_remote_code=True)
shapes = ([tuple(item['shape']) for item in kernel.metadata['specializations']]
          if kernel.metadata['abi'] == 'catlass-unary-f32-v1' else [(17,), (0,), (3, 7)])
for shape in shapes:
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape).npu()
    if x.ndim == 2 and kernel.metadata['abi'] == 'unary-f32-v1':
        x = x.t()
    result = kernel.forward(x)
    torch.npu.synchronize()
    assert result.shape == x.shape and result.dtype == x.dtype and result.device == x.device
print('binary loaded and called; numerical correctness and benchmarks remain pending')
"""


def job_script(
    profile: WorkerProfile,
    attempt: UUID,
    phase: Literal["prepare", "compile", "runtime", "cleanup", "identify"],
    *,
    runtime_program: str | None = None,
    capture_runtime: bool = False,
    benchmark: dict[str, str] | None = None,
) -> bytes:
    if (runtime_program is not None or capture_runtime) and phase != "runtime":
        raise ValueError("Custom runtime programs apply only to runtime jobs")
    if benchmark is not None and (
        phase != "runtime" or not capture_runtime or set(benchmark) != {"hardware_id", "driver"}
    ):
        raise ValueError("Benchmark identity applies only to captured runtime jobs")
    config = profile.model_dump(mode="json")
    if benchmark is not None:
        config["benchmark"] = benchmark
    if capture_runtime:
        config["capture_runtime"] = True
    program = RUNTIME if runtime_program is None else runtime_program
    if profile.catlass is None:
        config.pop("catlass")
    serialized = json.dumps(config, sort_keys=True)
    return (
        "#!/usr/bin/env python3\nimport json\n"
        + f"config = json.loads({serialized!r})\n"
        + f"token = {attempt.hex!r}\nphase = {phase!r}\n"
        + f"COMPILE = {COMPILE!r}\nRUNTIME = {program!r}\n"
        + SUPERVISOR
    ).encode()

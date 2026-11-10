"""Generate trusted host supervision for isolated Ascend C build/runtime jobs."""

from pathlib import PurePosixPath
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from cannery.registry.manifest import StrictModel


class WorkerProfile(StrictModel):
    target: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    runtime: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    workspace: str
    image: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    npu: int = Field(ge=0, le=63)
    device_group: int = Field(ge=0)
    readonly_mounts: tuple[str, ...]
    fingerprint: dict[str, str]
    compiler: str = Field(min_length=1, max_length=4096)

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
            or not self.fingerprint["soc"].startswith("Ascend950")
            or not all(self.fingerprint.values())
        ):
            raise ValueError("Expected an A5 runtime fingerprint")
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
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15)

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

cann = os.environ['ASCEND_HOME_PATH']
if (not Path(cann).is_absolute() or len(Path(cann).parts) < 4
        or not cann.startswith(('/opt/', '/usr/local/Ascend/'))):
    raise ValueError('Invalid registered CANN root')
args = ['docker', 'run', '--rm', '--init', '--pull', 'never', '--name', names[phase],
        '--network', 'none', '--user', '10001:10001', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--read-only', '--log-driver', 'none',
        '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=512m' if phase == 'runtime'
        else '/tmp:rw,noexec,nosuid,nodev,size=512m', '--shm-size', '512m', '--cpus', '4',
        '--memory', '8g' if phase == 'runtime' else '4g', '--pids-limit', '256',
        '--ulimit', 'fsize=67108864:67108864', '--env', 'HOME=/tmp',
        '--env', 'ASCEND_PROCESS_LOG_PATH=/tmp/ascend/log',
        '--mount', f'type=bind,src={source},dst=/input,readonly',
        '--mount', f'type=bind,src={cann},dst={cann},readonly']
for key in ['PATH', 'LD_LIBRARY_PATH', 'ASCEND_HOME_PATH',
            'ASCEND_TOOLKIT_HOME', 'ASCEND_OPP_PATH']:
    args += ['--env', key]
held = None
if phase == 'compile':
    args += ['--mount', f'type=bind,src={output},dst=/output']
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
    with lease.open('x') as marker:
        marker.write(token)
        marker.flush()
        os.fsync(marker.fileno())
    args += ['--group-add', str(config['device_group']), '--ulimit', 'memlock=-1:-1',
             '--env', 'ASCEND_RT_VISIBLE_DEVICES=' + str(config['npu']),
             '--env', 'ASCEND_VISIBLE_DEVICES=' + str(config['npu'])]
    for path in [f"/dev/davinci{config['npu']}", '/dev/davinci_manager',
                 '/dev/hisi_hdc', '/dev/ummu/tid']:
        args += ['--device', path]
    for path in config['readonly_mounts']:
        args += ['--mount', f'type=bind,src={path},dst={path},readonly']
    command = RUNTIME
args += [config['image'], 'python', '-c', command]
resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024,) * 2)
try:
    with (output / (phase + '.log')).open('w+b') as log:
        try:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        finally:
            log.seek(max(0, os.fstat(log.fileno()).st_size - 16384))
            sys.stderr.buffer.write(b'Last 16 KiB of isolated job log:\n' + log.read(16384))
            sys.stderr.flush()
    if result.returncode:
        raise SystemExit(1)
    if phase == 'compile':
        binary = output / 'kernel.so'
        if (not stat.S_ISREG(binary.lstat().st_mode)
                or not 0 < binary.stat().st_size <= 63 * 1024 * 1024):
            raise ValueError('Invalid compiled output')
        print(hashlib.sha256(binary.read_bytes()).hexdigest())
finally:
    remove_containers()
    if held is not None:
        if lease.exists() and lease.read_text() == token:
            lease.unlink()
        held.close()
"""

COMPILE = r"""
import json, pathlib, shutil, subprocess
plan = json.loads(pathlib.Path('/input/build.json').read_text())
version = subprocess.run(['bisheng', '--version'], capture_output=True, text=True,
                         check=True, timeout=10).stdout
assert plan['compiler'] in version, 'compiler fingerprint mismatch'
print(version[:4096], flush=True)
work = pathlib.Path('/tmp/work')
work.mkdir()
(work / 'source').symlink_to('/input/tree/' + plan['path'], target_is_directory=True)
(work / 'launch.asc').write_text(plan['wrapper'])
subprocess.run(plan['command'], cwd=work, check=True, timeout=120)
shutil.copyfile(work / 'kernel.so', '/output/kernel.so')
"""

RUNTIME = r"""
import hashlib, json, pathlib, subprocess, sys
wheels = list(pathlib.Path('/input').glob('*.whl'))
assert len(wheels) == 1
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                '--target', '/tmp/loader', str(wheels[0])], check=True, timeout=30)
sys.path.insert(0, '/tmp/loader')
import torch, torch_npu
from cannery_kernels import load_cannery_package
assert torch.npu.device_count() == 1, 'runtime device mask failed'
torch.npu.set_device(0)
package = pathlib.Path('/input/admitted.tar').read_bytes()
reference = json.loads(pathlib.Path('/input/artifact.json').read_text())
assert hashlib.sha256(package).hexdigest() == reference['sha256']
kernel = load_cannery_package(package, reference['sha256'], trust_remote_code=True)
for shape in [(17,), (0,), (3, 7)]:
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape).npu()
    if x.ndim == 2:
        x = x.t()
    result = kernel.forward(x)
    torch.npu.synchronize()
    assert result.shape == x.shape and result.dtype == x.dtype and result.device == x.device
print('binary loaded and called; numerical correctness and benchmarks remain pending')
"""


def job_script(
    profile: WorkerProfile,
    attempt: UUID,
    phase: Literal["prepare", "compile", "runtime", "cleanup"],
) -> bytes:
    return (
        "#!/usr/bin/env python3\nimport json\n"
        + f"config = json.loads({profile.model_dump_json()!r})\n"
        + f"token = {attempt.hex!r}\nphase = {phase!r}\n"
        + f"COMPILE = {COMPILE!r}\nRUNTIME = {RUNTIME!r}\n"
        + SUPERVISOR
    ).encode()

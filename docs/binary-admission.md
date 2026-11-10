# Private A5 binary admission

M3 accepts opt-in Ascend C submissions with `implementation.build.abi:
unary-f32-v1`. Existing manifests without an ABI remain source-only and keep their
canonical hashes. The wrapper accepts contiguous float32 input/output buffers,
a uint64 element count and the current NPU stream. Author code defines
`extern "C" __global__ __vector__ __aicore__ void forward(GM_ADDR, GM_ADDR, uint64_t)`.
The loader materializes noncontiguous input and preserves shape, dtype and device.
Only one source translation unit plus headers is supported.

The coordinator reads the retained Git snapshot, checks the frozen binding,
compiles through the installed `remote-access` skill, stages a hashed package in
the [private backend](private-artifacts.md), reads it back and runs an isolated
A5 loadability smoke. Successful admission becomes `awaiting_validation`;
correctness scores, benchmarks and publication remain pending. No Hugging Face
account, endpoint or artifact repository is required.

## Operator setup

Build the `cannery-runtime` wheel from the reviewed `cannery-kernels` repository,
retain its original wheel filename and record its SHA256. Provision the installed
remote-access skill, SSH configuration, ignored target registry and user-enabled
NPU policy on the coordinator host. The overlay mounts the existing user-wide
NPU policy directory read-only; it never grants device eligibility. Register the remote host's runtime in that
registry; CANN paths are resolved from the selected runtime rather than copied
into job sources. The profile is trusted operator configuration, never submission
input. Give it a dedicated empty remote workspace and a pinned Docker image ID.

Example profile shape (replace every inventory-dependent value):

```json
{
  "target": "bz-a5-2-host",
  "runtime": "cannery-cann-weekly",
  "workspace": "/home/worker/cannery-builds",
  "image": "sha256:<64 hex digits>",
  "npu": 2,
  "device_group": 1000,
  "toolchain_read_group": null,
  "transport_devices": [],
  "readonly_mounts": ["/usr/local/Ascend/driver", "/usr/local/dcmi"],
  "compiler": "exact bisheng version substring",
  "fingerprint": {
    "machine": "aarch64", "python": "3.12", "torch": "2.9.0+cpu",
    "torch_npu": "2.9.0.post9", "cann": "9.3.0", "soc": "Ascend950DT_9582"
  }
}
```

Inventory the actual driver library mounts and Unified Bus nodes. Expose only one
physical davinci device; inside the container it is logical device 0. The runtime
checks the driver's physical mapping before loading the binary. Some installations
ship root:root 0750 CANN packages: an explicit `toolchain_read_group: 0` permits
reading those packages while keeping UID 10001, capabilities removed, readonly
root/toolchain and no network. Prefer a readable dedicated toolchain installation
when available. This field grants supplemental file-group access, never UID 0.

The Compose overlay runs the **trusted coordinator** with operator SSH credentials
and a Docker socket for the registered VPN route. Its inputs must be operator-owned.
Author compilation/loading occurs only in separate restricted containers with no
credentials, home directory or Docker socket. Do not mount author worktrees into
the coordinator or replace the pinned image/profile during an active attempt.

Export paths to existing files and the installed skill's resolved directory:

```bash
export CANNERY_OPERATOR_HOME="$HOME"
export CANNERY_REMOTE_SKILL_ROOT="$(realpath ~/.agents/skills/remote-access)"
export CANNERY_WORKER_PROFILE=/absolute/path/worker.json
export CANNERY_LOADER_WHEEL=/absolute/path/cannery_kernels_runtime-0.1.0-py3-none-any.whl
export CANNERY_BUILD_LOADER_SHA256="$(sha256sum "$CANNERY_LOADER_WHEEL" | cut -d' ' -f1)"
# Preserve the service's existing port, identity and repository allowlist settings.
docker compose --env-file ~/.config/cannery/artifacts.env \
  -f compose.yaml -f compose.artifacts.yaml -f compose.builds.yaml build
docker compose --env-file ~/.config/cannery/artifacts.env \
  -f compose.yaml -f compose.artifacts.yaml -f compose.builds.yaml run --rm migrate
docker compose --env-file ~/.config/cannery/artifacts.env \
  -f compose.yaml -f compose.artifacts.yaml -f compose.builds.yaml up -d
```

Keep PostgreSQL, snapshots, artifact-data and build-control volumes. Never use
`down -v` against a live installation. Build-control contains retained job handles,
frozen configuration and cleanup checkpoints; preserve it across restarts. Startup
requires private storage and pinned profile/wheel settings and fails closed otherwise.

## Submit and recover

Use the public example's `candidates/helloworld_ascendc` directory and a full commit
SHA, after registering `team/helloworld@spec-v1` and allowlisting the exact Git URL:

```bash
cannery submit team/helloworld --spec-version 1 --device a5 --lang ascendc \
  --repo https://github.com/huawei-cpl-zurich/helloworld-kernel.git \
  --commit FULL_COMMIT_SHA --path candidates/helloworld_ascendc \
  --url http://localhost:8095 --wait-build
```

`--wait-build` also waits for intake. Inspect build status in the API, CLI or web
submission view. Logs and worker coordinates stay in private evidence, not public
status. There are at most three attempts and a 30-minute admission deadline.
Transport uncertainty resumes the same retained handle; it never blindly repeats
compilation. Busy/unhealthy devices are deferred. Lost ownership fences new work.
An uncertain container removal preserves the device lease and scratch for recovery.

After terminal recovery, the cleanup sweep removes owned containers and remote
scratch, retains private evidence and prunes bulky local inputs. Do not manually
delete an owner marker or dispatch journal while a retained job may still run.
Use the registered client's observe/cancel operations and reconcile that handle.

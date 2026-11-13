# A3 worker rollout and supervisor upgrades

The A3 track targets the registered `gz-a3` machine, which currently reports
Ascend910B4-1, CANN 9.1 and Python 3.11 / PyTorch 2.10. CATLASS DSL is A5-only.
Native Ascend C uses the existing unary float32 ABI and the verified ASC
compiler target `dav-2201`. Kernel declarations must carry an explicit kernel
kind, as in the example's `__vector__` declaration. Compatible C++ headers may
use that adapter; there is no dedicated CATLASS C++ build/launch contract.

Worker profiles pin device, compiler, image and exact runtime fingerprint. A5 is
the backward-compatible default. `toolchain_in_image` uses the pinned image's SDK
environment instead of host mounts; a host with no registered SDK runtime uses
an explicit null runtime. Device health/routing and validated deployment setup
must be installed before activating A3 services.

## Mandatory drain before changing supervisor code or toolchain pins

Attempt contexts pin the complete supervisor program as well as source, profile
and loader. A code/toolchain upgrade is not a supported hot-resume operation.
Profile serialization remains compatible, but changed supervisor hashes fail
closed with `worker_profile_changed`. Never reset contexts or delete receipts to
force an unfinished attempt onto new code, even after an admission checkpoint.

1. Build new images while the old deployment is running. Record old image/source
   identities and retain the old loader wheel and operator configuration.
2. Stop the API, let old coordinators finish queued/active work, then stop **all**
   application writers using the current Compose overlays.
3. Run `cannery maintenance require-drained --project cannery`. If it fails,
   restart the **old containers/images** to finish or reconcile the retained work.
   Do not start new images, cancel unrelated NPU work, or discard journals.
4. Capture and verify the five-component backup as described in [maintenance](maintenance.md).
   Keep writers stopped and run `require-drained` again before deployment.
5. Start new images only after both checks succeed. Verify readiness, retained A5
   publications and baseline state. A3 remains disabled until hardware acceptance.

An unavailable database, running writer, or queued/active stage blocks the check.
Recovering an interrupted job from backup requires its pinned old source/images
and configuration; complete that recovery before upgrading.

Remote-agent transfers are observable but cannot be cancelled by `cpl-remote`.
Only confirmed failed/cancelled transfers may be retried, at most three times.
A missing handle or ambiguous dispatch stays quarantined: never resubmit or
remove its remote attempt directory until the original operation is reconciled.
Pipeline deadlines still terminate SQL work; unresolved cleanup journals remain
for operator reconciliation rather than claiming cleanup succeeded. Restore the
listener's retained transfer state and observe the same handle before cleanup.

## Dual-worker deployment

Keep the existing A5 profile and loader pins unchanged. Install the validated A3
profile separately at `~/.config/cannery/a3-worker.json`, and retain the 0.3.0
loader wheel outside temporary task directories. Pin the verified SDK image,
compiler and full runtime fingerprint from the actual registered A3 target.
For the current setup the image contains the SDK, the host route is
`gz-a3-host`, `runtime` is null, `device` is `a3`, and physical NPU 4 maps to
logical device 0. Do not copy those hardware selections onto another server.

Add the following operator environment values to the private service environment
file (use absolute paths; keep existing A5 values):

```dotenv
CANNERY_A3_WORKER_PROFILE=/absolute/path/to/a3-worker.json
CANNERY_A3_LOADER_WHEEL=/absolute/path/to/cannery_kernels_runtime-0.3.0-py3-none-any.whl
CANNERY_A3_LOADER_SHA256=<sha256 of that exact wheel>
CANNERY_REMOTE_AGENT_ROOT=/absolute/installed/remote-agent
CANNERY_A3_DATABASE_PORT=15435
CANNERY_A3_ARTIFACT_PORT=18096
```

The remote-agent client root must match the path already named in the user-wide
route registry. The overlay mounts it read-only. A3 coordinators use the host
network to reach the registered loopback listener; submitted code runs remotely
in its isolated sandbox. PostgreSQL and the private artifact backend gain only
loopback host ports. Check those ports are free before rollout. A5 coordinators
keep their bridge-network addresses and original pins.

After the mandatory drain, verified backup and hardware acceptance, start with:

```bash
export COMPOSE_FILE=compose.yaml:compose.artifacts.yaml:compose.builds.yaml:compose.numerics.yaml:compose.benchmarks.yaml:compose.releases.yaml:compose.a3.yaml
docker compose --env-file "$HOME/.config/cannery/service.env" --profile a3 up -d
```

Omitting the `a3` profile leaves all three A3 coordinators disabled. All workers
share the backed-up build-control volume; claims and cleanup are scoped to the
configured device route. Stop A3 workers as well as A5 workers before upgrades,
backup or restore. Existing published A5 releases and the cached CPU baseline
are reused, not recomputed or republished.

Local verification includes the rendered Compose configuration and an opt-in
Docker drill with two worker nodes and PostgreSQL. The drill partitions the A3
node's network, lets A5 finish, then verifies A3 recovers the original fenced
attempts without duplicate effects:

```bash
CANNERY_DEVICE_DOCKER=1 pytest -q tests/test_device_workers_docker.py
```

This drill verifies SQL routing and recovery, not NPU execution. Use actual A3
acceptance for device behavior, and keep A3/A5 timing cohorts separate.

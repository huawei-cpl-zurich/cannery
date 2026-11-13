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

# Reference smoke diagnostics

M1 accepts arbitrary author Python. Its smoke run is a bounded diagnostic for
ordinary import, interface and environment errors, not independent proof that a
particular Python callable executed. An author program can emulate its worker
protocol. Every result explicitly carries `assurance: diagnostic`; `cases` counts
observed case responses, not independently attested function executions. Neither
numerical correctness nor performance is established by these observations.

The trusted supervisor independently checks the installed runtime and owns final
result output, while author code runs under a different non-root UID. It cannot
access supervisor stdout/memory, host credentials or the Docker socket. Network,
root filesystem, memory, CPU, process count, temporary storage, output and elapsed
time are restricted. A supervisor-owned timer remains active if the host loses
contact with Docker. Failed cleanup raises a recoverable sandbox handle.

Registration authorization comes from namespace/role policy. Immutable source,
snapshot provenance, retained refs and content digests are verified separately
before freezing. Smoke observations never replace those checks or authorize a
new spec version. M5 must perform its own correctness and benchmark evaluation;
it cannot treat the M1 diagnostic as proof.

## CPU acceptance

With the package installed and Docker available, run
`python scripts/reference_acceptance.py`. This builds a task-owned CPU image,
checks ordinary reference diagnostics and sandbox boundaries, and removes its
resources afterward. It also demonstrates adaptive protocol emulation returning
only diagnostic assurance; it never treats that observation as callable proof.
The CPU CI reference job runs the same command without an NPU or host PyTorch.

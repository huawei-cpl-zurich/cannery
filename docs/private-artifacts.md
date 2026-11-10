# Private artifact storage

Cannery can store generated binaries and evidence in its own artifact service.
This backend does not require or contact Hugging Face. Source intake remains Git-only.

Generate two distinct random secrets (at least 32 characters) and keep them in an
operator-owned, uncommitted environment file. Set `CANNERY_ARTIFACT_READ_TOKEN`
and `CANNERY_ARTIFACT_WRITE_TOKEN`. Do not put these secrets in submission job
containers. Only the trusted coordinator receives the writer secret; consumers
receive only the reader secret.

Start storage alongside an existing Compose project without restarting its services:

```sh
docker compose --env-file /path/to/private.env -p cannery \
  -f compose.yaml -f compose.artifacts.yaml up -d --build artifacts
```

There is no published host port. The internal address is `http://artifacts:8000`.
Trusted clients use `CANNERY_ARTIFACT_BACKEND=private`,
`CANNERY_ARTIFACT_URL=http://artifacts:8000` and their explicit credentials.
The opt-in overlay does not switch existing workers or restart the API.
Keep the service on a trusted private network; use TLS if moving it off-host.

`PUT /v1/blobs/{sha256}` accepts at most 64 MiB of raw bytes with the writer bearer
credential. Uploads have a 30-second receive deadline. A mismatched digest or
incomplete upload cannot publish a blob. Publication fsyncs data and the volume
directory and never overwrites an existing object. Repeating an identical PUT is
safe. `GET /v1/blobs/{sha256}` requires the reader credential and validates stored
bytes before serving them. Clients verify downloads and read back uploads before
accepting them. There are no listing, deletion or overwrite endpoints.

The named `artifact-data` volume is durable. Back it up alongside PostgreSQL and
Git snapshots; do not remove it during routine worktree or job cleanup. Artifacts
are immutable blobs, so interrupted transfers may be retried using the same digest.
A storage outage or corruption is an explicit failure, never a successful build.
A crash during a transfer may leave an unpublished temporary file; remove those
only during maintenance with the service stopped. No automatic garbage collection
is enabled in this first storage slice.

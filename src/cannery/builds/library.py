"""Shared bounded native library packaging; independent of worker execution."""

import io
import json
import re
import tarfile
from typing import Any

from cannery.builds.ascendc import BuildError


class ExecutionFailure(BuildError):
    def __init__(self, code: str, retryable: bool = False) -> None:
        self.code, self.retryable = code, retryable
        super().__init__(code)


def check_library(binary: bytes, source: str) -> None:
    """Require a bounded AArch64 ELF shared library and a retained source identity."""
    if (
        type(binary) is not bytes
        or type(source) is not str
        or not 64 <= len(binary) <= 63 * 1024 * 1024
        or binary[:6] != b"\x7fELF\x02\x01"
        or int.from_bytes(binary[16:18], "little") != 3
        or int.from_bytes(binary[18:20], "little") != 183
        or not re.fullmatch(r"[a-f0-9]{64}", source)
    ):
        raise ExecutionFailure("invalid_compiled_binary")


def archive_library(
    binary: bytes, metadata: dict[str, Any], *, metadata_limit: int = 16384
) -> bytes:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > metadata_limit:
        raise ExecutionFailure("invalid_worker_profile")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, value in [("metadata.json", encoded), ("kernel.so", binary)]:
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(value), 0o400
            archive.addfile(item, io.BytesIO(value))
    return output.getvalue()

"""Read pinned Ascend C pointer-only AIV exports without loading native code."""

import re
import struct
from typing import Any

from cannery.builds.ascendc import BuildError
from cannery.builds.catlass import _native_object
from cannery.builds.catlass_tensor import StaticTensorExport


def ascendc_sections(binary: bytes) -> dict[str, bytes]:
    """Bounded little-endian ELF64 section reader for compiler output, not host ELF."""
    if not _native_object(binary):
        raise BuildError("Invalid Ascend C executable")
    try:
        offset = struct.unpack_from("<Q", binary, 40)[0]
        width, count, names_index = struct.unpack_from("<HHH", binary, 58)
        if width != 64 or not 1 <= count <= 512 or not 0 < names_index < count:
            raise ValueError
        if offset < 64 or offset + count * width > len(binary):
            raise ValueError
        headers = [
            struct.unpack_from("<IIQQQQIIQQ", binary, offset + i * width) for i in range(count)
        ]
        names = headers[names_index]
        if names[1] != 3 or names[4] + names[5] > len(binary):
            raise ValueError
        table = binary[names[4] : names[4] + names[5]]
        sections = {}
        seen = set()
        for header in headers[1:]:
            name_offset, kind, _, _, start, size, *_ = header
            if name_offset >= len(table):
                raise ValueError
            end = table.index(b"\0", name_offset)
            name = table[name_offset:end].decode("ascii")
            if not name or name in seen:
                raise ValueError
            seen.add(name)
            if kind == 8:  # NOBITS is memory storage, not file data.
                continue
            if start + size > len(binary):
                raise ValueError
            sections[name] = binary[start : start + size]
        return sections
    except (ValueError, IndexError, UnicodeError, struct.error):
        raise BuildError("Invalid Ascend C ELF section table") from None


def ascendc_records(raw: bytes) -> dict[int, list[bytes]]:
    """The pinned compiler emits uint16 tag/length followed by its payload."""
    result: dict[int, list[bytes]] = {}
    offset = 0
    while offset < len(raw):
        if offset + 4 > len(raw):
            raise BuildError("Truncated Ascend C metadata record")
        tag, size = struct.unpack_from("<HH", raw, offset)
        offset += 4
        if not size or offset + size > len(raw):
            raise BuildError("Invalid Ascend C metadata record size")
        result.setdefault(tag, []).append(raw[offset : offset + size])
        offset += size
    return result


def validate_ascendc_export(
    binary: bytes,
    entrypoint: str,
    signature: dict[str, Any],
    bindings: list[list[str]],
    block_num: int,
) -> StaticTensorExport:
    """Only the measured AIV-only, no cross-core sync, GM-pointer ABI is supported.

    Scalars specialize source before compilation. Mixed Cube/vector, runtime
    scalar arguments, workspace and other compiler metadata require new proofs.
    """
    try:
        if (
            type(entrypoint) is not str
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", entrypoint) is None
            or type(block_num) is not int
            or not 1 <= block_num <= 65535
            or type(bindings) is not list
            or not 1 <= len(bindings) <= 64
        ):
            raise ValueError
        sections = ascendc_sections(binary)
        name = ".ascend.meta." + entrypoint
        if {key for key in sections if key.startswith(".ascend.meta.")} != {name}:
            raise ValueError
        size = len(bindings) * 8
        if sections["__CCE_KernelArgSize"] != struct.pack("<II", size, size):
            raise ValueError
        records = ascendc_records(sections[name])
        if set(records) != {1, 2, 7, 8, 9, 10, 12, 16, 17}:
            raise ValueError
        for tag, value in [(1, 3), (2, 0), (9, 0), (10, 0), (12, 2)]:
            if records[tag] != [struct.pack("<I", value)]:
                raise ValueError
        if records[16] != [struct.pack("<III", len(bindings), size, 0)]:
            raise ValueError
        if records[17] != [
            struct.pack("<IIIIHHIIII", 0, i, 8 * i, 8, 8, 2, 0, 0, 0, 0)
            for i in range(len(bindings))
        ]:
            raise ValueError
        for tag in [7, 8]:
            if len(records[tag]) != 1 or len(records[tag][0]) != 4:
                raise ValueError
            if struct.unpack("<I", records[tag][0])[0] > 253952:
                raise ValueError
        resolved = []
        for binding in bindings:
            if (
                type(binding) is not list
                or len(binding) != 2
                or binding[0] not in ("input", "output")
                or type(binding[1]) is not str
            ):
                raise ValueError
            direction, name = binding
            position = next(
                i for i, t in enumerate(signature[direction + "s"]) if t["name"] == name
            )
            resolved.append((direction, position))
        for index, output in enumerate(signature["outputs"]):
            if ("output", index) not in resolved and not any(
                direction == "input" and signature["inputs"][position]["name"] == output["alias"]
                for direction, position in resolved
            ):
                raise ValueError
        return StaticTensorExport(binary, entrypoint, "aiv", block_num, tuple(resolved))
    except (ValueError, KeyError, TypeError, StopIteration, struct.error):
        raise BuildError("Unsupported Ascend C static pointer export") from None

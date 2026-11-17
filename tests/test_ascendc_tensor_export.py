import json
import struct
from pathlib import Path

import pytest

from cannery.builds.ascendc import BuildError
from cannery.builds.ascendc_tensor import ascendc_records, ascendc_sections, validate_ascendc_export


def fixture():
    data = json.loads((Path(__file__).parent / "fixtures/ascendc/aiv-add128.json").read_text())
    return {
        "__CCE_KernelArgSize": bytes.fromhex(data["argument_sizes"]),
        ".ascend.meta.ascendc_tensor_add": bytes.fromhex(data["metadata"]),
    }


def elf(sections):
    names = b"\0.shstrtab\0"
    contents = [names]
    for name, raw in sections.items():
        names += name.encode() + b"\0"
        contents.append(raw)
    contents[0] = names
    count = len(contents) + 1
    result = bytearray(64 + 64 * count)
    result[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HH", result, 16, 2, 4137)
    struct.pack_into("<Q", result, 40, 64)
    struct.pack_into("<HHH", result, 58, 64, count, 1)
    for index, (name, raw) in enumerate(zip([".shstrtab", *sections], contents, strict=True), 1):
        struct.pack_into(
            "<IIQQQQIIQQ",
            result,
            64 + 64 * index,
            names.index(name.encode() + b"\0"),
            3 if index == 1 else 7,
            0,
            0,
            len(result),
            len(raw),
            0,
            0,
            4,
            0,
        )
        result.extend(raw)
    return bytes(result)


def validate(sections, bindings=None, **kwargs):
    signature = {
        "inputs": [{"name": name} for name in ["a", "b"]],
        "outputs": [{"name": "out", "alias": None}],
    }
    return validate_ascendc_export(
        elf(sections),
        "ascendc_tensor_add",
        signature,
        bindings or [["input", "a"], ["input", "b"], ["output", "out"]],
        kwargs.get("blocks", 1),
    )


def test_observed_native_aiv_pointer_metadata_is_accepted():
    sections = fixture()
    assert ascendc_sections(elf(sections))["__CCE_KernelArgSize"] == struct.pack("<II", 24, 24)
    export = validate(sections)
    assert export.mode == "aiv" and export.block_num == 1
    assert export.bindings == (("input", 0), ("input", 1), ("output", 0))


@pytest.mark.parametrize("tag", [1, 2, 9, 10, 12, 16, 17])
def test_changed_native_abi_or_requirements_are_rejected(tag):
    sections = fixture()
    key = ".ascend.meta.ascendc_tensor_add"
    records = ascendc_records(sections[key])
    records[tag][0] = bytes(len(records[tag][0]))
    sections[key] = b"".join(
        struct.pack("<HH", key, len(raw)) + raw for key, values in records.items() for raw in values
    )
    # Some observed records already contain zero; use a nonzero byte mutation.
    if tag in [2, 9, 10]:
        records[tag][0] = struct.pack("<I", 1)
        sections[key] = b"".join(
            struct.pack("<HH", k, len(raw)) + raw for k, values in records.items() for raw in values
        )
    with pytest.raises(BuildError):
        validate(sections)


@pytest.mark.parametrize("raw", [b"\1", b"\1\0\0\0", b"\1\0\10\0abcd"])
def test_truncated_metadata_rejected(raw):
    with pytest.raises(BuildError):
        ascendc_records(raw)


@pytest.mark.parametrize("change", ["args", "extra", "missing", "unknown", "duplicate", "pointer"])
def test_unsupported_records_or_kernel_domains_rejected(change):
    sections = fixture()
    key = ".ascend.meta.ascendc_tensor_add"
    if change == "args":
        sections["__CCE_KernelArgSize"] = struct.pack("<II", 32, 32)
    elif change == "extra":
        sections[".ascend.meta.other_kernel"] = sections[key]
    elif change == "missing":
        del sections[key]
    elif change == "unknown":
        sections[key] += struct.pack("<HHI", 99, 4, 0)
    elif change == "duplicate":
        sections[key] += struct.pack("<HHI", 1, 4, 3)
    else:
        with pytest.raises(BuildError):
            validate(sections, [["input", "a"]] * 3)
        return
    with pytest.raises(BuildError):
        validate(sections)


@pytest.mark.parametrize(
    "field,value,format", [(40, 2**63, "Q"), (58, 63, "H"), (60, 513, "H"), (62, 999, "H")]
)
def test_out_of_bounds_elf_tables_fail_closed(field, value, format):
    raw = bytearray(elf(fixture()))
    struct.pack_into("<" + format, raw, field, value)
    with pytest.raises(BuildError):
        ascendc_sections(bytes(raw))


@pytest.mark.parametrize("blocks", [0, 65536, True])
def test_invalid_grid_is_rejected(blocks):
    with pytest.raises(BuildError):
        validate(fixture(), blocks=blocks)

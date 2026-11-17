import copy
import json
from pathlib import Path

import pytest
from test_catlass_export import binary

from cannery.builds.ascendc import BuildError
from cannery.builds.catlass_tensor import validate_static_export


def fixture():
    metadata = json.loads(
        (Path(__file__).parent / "fixtures/catlass/tensor-matmul128.json").read_bytes()
    )
    signature = {
        "inputs": [{"name": name, "shape": [128, 128], "dtype": "float16"} for name in ["a", "b"]],
        "outputs": [{"name": "out", "shape": [128, 128], "dtype": "float32", "alias": None}],
        "scalars": [],
    }
    return metadata, signature, [["input", "a"], ["input", "b"], ["output", "out"]]


def test_actual_static_matrix_descriptor_and_ordered_bindings():
    metadata, signature, bindings = fixture()
    export = validate_static_export(binary(), metadata, signature, bindings, 1)
    assert export.entrypoint == "basic_mmad_kernel" and export.mode == "aic"
    assert export.bindings == (("input", 0), ("input", 1), ("output", 0))
    metadata["kernel_mode"] = "aiv"
    assert validate_static_export(binary(), metadata, signature, bindings, 48).block_num == 48
    # Compile-time parameters can leave gaps in logical indices; physical byte
    # offsets and verified pointer types determine the native argument order.
    for argument, logical in zip(metadata["kernel_abi"]["arguments"], [0, 2, 9], strict=True):
        argument["logical_index"] = logical
    assert (
        validate_static_export(binary(), metadata, signature, bindings, 1).bindings
        == export.bindings
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("kernel_mode",), "mix"),
        (("is_mixed",), True),
        (("uses_scalar_print",), True),
        (("uses_tensor_print",), True),
        (("ub_dynamic_base",), 0),
        (("ub_static_bytes",), True),
        (("ub_static_bytes",), 253953),
        (("entrypoint",), "bad();"),
        (("kernel_abi", "schema_version"), 4),
        (("kernel_abi", "schema_version"), True),
        (("kernel_abi", "total_size"), 32),
        (("kernel_abi", "entrypoint"), "other"),
        (("kernel_abi", "arguments", 0, "index"), True),
        (("kernel_abi", "arguments", 0, "offset"), 4),
        (("kernel_abi", "arguments", 0, "logical_index"), True),
        (("kernel_abi", "arguments", 0, "logical_index"), 128),
        (("kernel_abi", "arguments", 0, "kind"), "memref_field"),
        (("kernel_abi", "arguments", 0, "mlir_type"), "memref<?x?xf16, #hivm.address_space<gm>>"),
        (
            ("kernel_abi", "arguments", 0, "mlir_type"),
            "memref<128x128xf32, #hivm.address_space<gm>>",
        ),
    ],
)
def test_unsupported_or_forged_exports_fail_closed(path, value):
    metadata, signature, bindings = fixture()
    target = metadata
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(BuildError):
        validate_static_export(binary(), metadata, signature, bindings, 1)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        [["input", "missing"]],
        [["scalar", "a"]],
        [["input"]],
        [["input", "a", "extra"]],
        [["input", 0]],
        [["input", "a"]] * 65,
    ],
)
def test_invalid_pointer_bindings_rejected(bad):
    metadata, signature, _ = fixture()
    with pytest.raises(BuildError):
        validate_static_export(binary(), metadata, signature, bad, 1)


def test_mutable_output_can_share_an_input_pointer_but_fresh_output_must_be_bound():
    metadata, signature, bindings = fixture()
    metadata["kernel_abi"]["arguments"].pop()
    metadata["kernel_abi"]["total_size"] = 16
    bindings.pop()
    with pytest.raises(BuildError):
        validate_static_export(binary(), metadata, signature, bindings, 1)
    signature["outputs"][0].update(dtype="float16", alias="a")
    assert validate_static_export(binary(), metadata, signature, bindings, 1).bindings == (
        ("input", 0),
        ("input", 1),
    )


@pytest.mark.parametrize("blocks", [0, -1, 65536, True, 1.0])
def test_invalid_blocks_or_native_object_rejected(blocks):
    metadata, signature, bindings = fixture()
    with pytest.raises(BuildError):
        validate_static_export(binary(), metadata, signature, bindings, blocks)
    with pytest.raises(BuildError):
        validate_static_export(b"bad", metadata, signature, bindings, 1)


def test_metadata_extra_fields_and_duplicate_offsets_rejected():
    metadata, signature, bindings = fixture()
    duplicate = copy.deepcopy(metadata)
    duplicate["kernel_abi"]["arguments"][1]["offset"] = 0
    for value in [metadata | {"untrusted": True}, duplicate]:
        with pytest.raises(BuildError):
            validate_static_export(binary(), value, signature, bindings, 1)

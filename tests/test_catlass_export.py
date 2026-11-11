"""CPU bridge verification. The ACL stub does not establish NPU execution."""

import copy
import ctypes
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cannery.builds.ascendc import BuildError
from cannery.builds.catlass import native_wrapper, validate_export

FIXTURES = Path(__file__).parent / "fixtures/catlass"


def metadata(count=17):
    value = json.loads((FIXTURES / "export17.json").read_bytes())
    for argument in value["kernel_abi"]["arguments"]:
        argument["mlir_type"] = f"memref<{count}xf32, #hivm.address_space<gm>>"
    return value


def binary(count=17):
    value = bytearray(64)
    value[:7] = b"\x7fELF\x02\x01\x01"
    value[16:18] = (2).to_bytes(2, "little")
    value[18:20] = (4137).to_bytes(2, "little")
    value[8] = count  # Stub-only marker; no device object is executed in CPU tests.
    return bytes(value)


def test_actual_compiler_descriptor_and_bounded_native_objects():
    variant = validate_export(17, binary(), metadata())
    assert variant.count == 17 and variant.entrypoint == "forward"
    host_binary = bytearray(binary())
    host_binary[18:20] = (183).to_bytes(2, "little")
    for count, data in [
        (0, binary()),
        (True, binary()),
        (1048577, binary()),
        (17, b"bad"),
        (17, bytes(host_binary)),
    ]:
        with pytest.raises(BuildError):
            validate_export(count, data, metadata())


@pytest.mark.parametrize(
    "path,value",
    [
        (("kernel_mode",), "mix"),
        (("is_mixed",), True),
        (("uses_scalar_print",), True),
        (("uses_tensor_print",), True),
        (("ub_dynamic_base",), 0),
        (("ub_static_bytes",), True),
        (("entrypoint",), 'forward";arbitrary();'),
        (("kernel_abi", "schema_version"), 4),
        (("kernel_abi", "entrypoint"), "different"),
        (("kernel_abi", "total_size"), 24),
        (("kernel_abi", "arguments", 1, "offset"), 4),
        (("kernel_abi", "arguments", 0, "index"), False),
        (("kernel_abi", "arguments", 0, "kind"), "memref_field"),
        (("kernel_abi", "arguments", 0, "alignment"), 8),
        (("kernel_abi", "arguments", 0, "storage_size"), 4),
        (("kernel_abi", "arguments", 0, "mlir_type"), "memref<?xf32>"),
        (("kernel_abi", "arguments", 0, "mlir_type"), "memref<18xf32, #hivm.address_space<gm>>"),
    ],
)
def test_unimplemented_abi_modes_are_rejected_before_native_generation(path, value):
    record = copy.deepcopy(metadata())
    parent = record
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(BuildError, match="Unsupported CATLASS export ABI"):
        validate_export(17, binary(), record)


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    root = tmp_path_factory.mktemp("catlass-bridge")
    (root / "acl").mkdir()
    shutil.copyfile(FIXTURES / "acl.h", root / "acl/acl.h")
    variants = [validate_export(n, binary(n), metadata(n)) for n in (17, 21)]
    source = root / "launch.cpp"
    source.write_text(native_wrapper(variants))
    library = root / "kernel.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-I",
            str(root),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return library


def load(bridge, root):
    # A new path gives every failure-injection case independent sticky error state.
    path = root / "kernel.so"
    shutil.copyfile(bridge, path)
    lib = ctypes.CDLL(str(path))
    lib.cannery_launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint64]
    lib.cannery_launch.restype = ctypes.c_int
    return lib


def call(lib, count):
    values = ctypes.c_float * count
    x, y = values(*range(count)), values()
    status = lib.cannery_launch(99, x, y, count)
    return status, list(y)


def test_offline_bridge_dispatch_payload_stream_and_handle_reuse(bridge, tmp_path):
    lib = load(bridge, tmp_path)
    for count in (17, 21, 17):
        status, values = call(lib, count)
        assert status == 0 and values == [i + 1 for i in range(count)]
    assert lib.test_loads() == 2 and lib.test_launches() == 3
    assert call(lib, 18)[0] == -1
    assert lib.cannery_launch(None, None, None, 0) == 0
    assert lib.test_loads() == 2 and lib.test_launches() == 3


@pytest.mark.parametrize("stage", range(1, 7))
def test_acl_failure_propagates_and_never_blindly_retries(bridge, tmp_path, stage):
    lib = load(bridge, tmp_path)
    lib.test_fault(stage)
    assert call(lib, 17)[0] == 100 + stage
    lib.test_fault(0)
    assert call(lib, 17)[0] == 100 + stage
    assert lib.test_loads() == 1 and lib.test_launches() == 0

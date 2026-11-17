"""CPU ACL simulation checks native packing/lifecycle, not NPU correctness."""

import copy
import ctypes as c
import subprocess

import pytest
from test_catlass_export import binary
from test_catlass_tensor_export import fixture

from cannery.builds.ascendc import BuildError
from cannery.builds.catlass_tensor import validate_static_export
from cannery.builds.tensor_native import native_tensor_wrapper

# Test stand-in for the pinned runtime wheel's header. Real A5 proofs compile
# against the actual wheel header and use the real SDK descriptors.
HEADER = """#include <stdint.h>
typedef struct {void* data; uint64_t sizes[4],strides[4];uint32_t rank,dtype;} CanneryTensorV2;
typedef struct {uint32_t kind,reserved;uint64_t bits;} CanneryScalarV2;
"""
ACL = r"""
#include <cstdint>
#include <cstddef>
using aclrtContext=void*; using aclrtBinHandle=void*; using aclrtFuncHandle=void*;
static int fault=0,loads=0,unloads=0,launches=0,blocks_seen=0;
static uintptr_t context=1;
extern "C" void test_fault(int value){fault=value;}
extern "C" void test_context(int value){context=value;}
extern "C" int test_loads(){return loads;}
extern "C" int test_unloads(){return unloads;}
extern "C" int test_launches(){return launches;}
extern "C" int test_blocks(){return blocks_seen;}
constexpr int ACL_RT_BINARY_LOAD_OPT_MAGIC=2,ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD=3;
constexpr uint32_t ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE=0x41415246;
constexpr uint32_t ACL_RT_BINARY_MAGIC_ELF_AICORE=0x43554245;
struct aclrtBinaryLoadOption {int type;union {uint32_t magic;int isLazyLoad;} value;};
struct aclrtBinaryLoadOptions {aclrtBinaryLoadOption* options;size_t numOpt;};
static int aclrtGetCurrentContext(aclrtContext* out) {
  *out=reinterpret_cast<void*>(context);return fault==4 ? 104 : 0;
}
static int aclrtBinaryLoadFromData(const void* data,size_t size,
    const aclrtBinaryLoadOptions* opts,aclrtBinHandle* out) {
  ++loads;
  if(fault==1 || ((fault==5 || fault==8) && loads==2)) return 100+fault;
  if(fault==6) return 0;
  if(size<64 || opts->numOpt!=2 || opts->options[0].type!=3 ||
      opts->options[0].value.isLazyLoad!=0 || opts->options[1].type!=2) return 199;
  *out=reinterpret_cast<void*>(uintptr_t(static_cast<const unsigned char*>(data)[8])+1);
  return 0;
}
static int aclrtBinaryGetFunction(aclrtBinHandle binary,const char*,aclrtFuncHandle* out) {
  if(fault==2) return 102;
  if(fault!=7) *out=binary;
  return 0;
}
static int aclrtBinaryUnLoad(aclrtBinHandle){++unloads;return fault==8 ? 208 : 0;}
static int aclrtLaunchKernelWithHostArgs(aclrtFuncHandle fn,uint32_t blocks,
    void* stream,void* cfg,void* args,size_t size,void* placeholders,size_t count) {
  if(fault==3) return 103;
  if(stream!=reinterpret_cast<void*>(99) || cfg || placeholders || count || size!=24) return 199;
  auto** pointers=static_cast<float**>(args);
  for(uintptr_t i=0;i<reinterpret_cast<uintptr_t>(fn)-1;++i)
      pointers[2][i]=pointers[0][i]+pointers[1][i];
  ++launches;blocks_seen=blocks;return 0;
}
"""


class Tensor(c.Structure):
    _fields_ = [
        ("data", c.c_void_p),
        ("sizes", c.c_uint64 * 4),
        ("strides", c.c_uint64 * 4),
        ("rank", c.c_uint32),
        ("dtype", c.c_uint32),
    ]


class Scalar(c.Structure):
    _fields_ = [("kind", c.c_uint32), ("reserved", c.c_uint32), ("bits", c.c_uint64)]


def variants():
    result = []
    for count, alias in [(17, False), (21, False), (0, False), (17, True)]:
        metadata, signature, bindings = fixture()
        signature["inputs"] = [
            {"name": name, "shape": [count], "dtype": "float32"} for name in ["a", "b"]
        ]
        signature["outputs"] = [
            {"name": "out", "shape": [count], "dtype": "float32", "alias": "a" if alias else None}
        ]
        signature["scalars"] = [
            {"name": "flag", "kind": "bool", "value": True},
            {"name": "offset", "kind": "int64", "value": -7},
            {"name": "eps", "kind": "float64", "value": -0.0 if alias else 0.0},
        ]
        for argument in metadata["kernel_abi"]["arguments"]:
            argument["mlir_type"] = f"memref<{count}xf32, #hivm.address_space<gm>>"
        export = validate_static_export(
            binary(count), metadata, signature, bindings, 2 if alias else 1
        )
        result.append((signature, export))
    return result


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    root = tmp_path_factory.mktemp("tensor-native")
    (root / "acl").mkdir()
    (root / "acl/acl.h").write_text(ACL)
    (root / "tensor_abi.h").write_text(HEADER)
    (root / "bridge.cpp").write_text(native_tensor_wrapper(variants()))
    path = root / "bridge.so"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-shared",
            "-fPIC",
            "-pthread",
            "-I",
            str(root),
            str(root / "bridge.cpp"),
            "-o",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return path


def load(bridge, tmp_path):
    target = tmp_path / "library.so"
    target.write_bytes(bridge.read_bytes())
    lib = c.CDLL(str(target))
    lib.cannery_prepare_tensor_v2.restype = c.c_int32
    lib.cannery_launch_tensor_v2.argtypes = [
        c.c_void_p,
        c.c_uint32,
        c.POINTER(Tensor),
        c.c_uint32,
        c.POINTER(Tensor),
        c.c_uint32,
        c.POINTER(Scalar),
    ]
    lib.cannery_launch_tensor_v2.restype = c.c_int32
    return lib


def call(lib, count=17, alias=False, broken=False):
    values = c.c_float * count
    a, b, out = values(*range(count)), values(*([2] * count)), values()
    if alias:
        out = a

    def descriptor(data):
        return Tensor(
            c.addressof(data) if count else 0, (c.c_uint64 * 4)(count), (c.c_uint64 * 4)(1), 1, 1
        )

    inputs = (Tensor * 2)(descriptor(a), descriptor(b))
    outputs = (Tensor * 1)(descriptor(out))
    scalars = (Scalar * 3)(
        Scalar(1, 0, 1), Scalar(2, int(broken), 2**64 - 7), Scalar(3, 0, 2**63 if alias else 0)
    )
    return lib.cannery_launch_tensor_v2(99, 2, inputs, 1, outputs, 3, scalars), list(out)


def test_initialization_dispatch_alias_scalar_bits_and_no_forward_loads(bridge, tmp_path):
    lib = load(bridge, tmp_path)
    assert call(lib)[0] != 0 and lib.test_loads() == 0
    assert lib.cannery_prepare_tensor_v2() == 0 and lib.test_loads() == 4
    assert lib.cannery_prepare_tensor_v2() == 0 and lib.test_loads() == 4
    for count, alias in [(17, False), (21, False), (0, False), (17, True)]:
        status, output = call(lib, count, alias)
        assert status == 0 and output == [i + 2 for i in range(count)]
        assert lib.test_blocks() == (2 if alias else 1)
    assert lib.test_loads() == 4 and lib.test_launches() == 4
    assert call(lib, 18)[0] != 0 and call(lib, broken=True)[0] != 0
    lib.test_context(2)
    assert call(lib)[0] != 0 and lib.cannery_prepare_tensor_v2() != 0
    lib.test_context(1)
    lib.test_fault(3)
    assert call(lib)[0] == 103
    lib.test_fault(0)
    assert call(lib)[0] == 0


@pytest.mark.parametrize(
    "fault,status,unloads", [(1, 101, 0), (2, 102, 1), (5, 105, 1), (6, -14, 0), (7, -15, 1)]
)
def test_initialization_failures_release_partial_handles(bridge, tmp_path, fault, status, unloads):
    lib = load(bridge, tmp_path)
    lib.test_fault(fault)
    assert lib.cannery_prepare_tensor_v2() == status
    assert lib.test_unloads() == unloads and lib.test_launches() == 0 and call(lib)[0] != 0


def test_missing_or_failed_context_does_not_load_modules(bridge, tmp_path):
    lib = load(bridge, tmp_path)
    lib.test_context(0)
    assert lib.cannery_prepare_tensor_v2() != 0 and lib.test_loads() == 0
    lib.test_fault(4)
    assert lib.cannery_prepare_tensor_v2() == 104 and lib.test_loads() == 0


@pytest.mark.parametrize(
    "kind", ["duplicate", "bad_shape", "scalar", "alias", "binding", "interface"]
)
def test_invalid_native_variants_fail_before_compilation(kind):
    values = variants()
    if kind == "duplicate":
        values.append(copy.deepcopy(values[0]))
    elif kind == "bad_shape":
        values[0][0]["inputs"][0]["shape"] = [True]
    elif kind == "scalar":
        values[0][0]["scalars"][0]["value"] = 1
    elif kind == "alias":
        values[0][0]["outputs"][0]["alias"] = "missing"
    elif kind == "binding":
        values[0] = (
            values[0][0],
            values[0][1].__class__(binary(), "forward", "aiv", 1, (("input", 99),)),
        )
    else:
        values[1][0]["inputs"][0]["name"] = "different"
    with pytest.raises(BuildError):
        native_tensor_wrapper(values)


def test_parallel_preparation_and_calls_share_handles_without_corruption(bridge, tmp_path):
    import sys

    script = """
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'tests')
from test_tensor_native import load,call
lib=load(Path(sys.argv[1]),Path(sys.argv[2]))
with ThreadPoolExecutor(max_workers=8) as pool:
    assert list(pool.map(lambda _:lib.cannery_prepare_tensor_v2(),range(32)))==[0]*32
    results=list(pool.map(lambda _:call(lib),range(64)))
assert lib.test_loads()==4 and lib.test_launches()==64
assert all(status==0 and values==[i+2 for i in range(17)] for status,values in results)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(bridge), str(tmp_path)],
        check=True,
        capture_output=True,
        timeout=10,
    )


def test_failed_partial_unload_is_terminal_and_never_loads_more_handles(bridge, tmp_path):
    lib = load(bridge, tmp_path)
    lib.test_fault(8)
    assert lib.cannery_prepare_tensor_v2() == 208
    assert lib.test_loads() == 2 and lib.test_unloads() == 1
    lib.test_fault(0)
    for _ in range(3):
        assert lib.cannery_prepare_tensor_v2() == 208
        assert call(lib)[0] != 0
    assert lib.test_loads() == 2 and lib.test_unloads() == 1
    assert lib.test_launches() == 0

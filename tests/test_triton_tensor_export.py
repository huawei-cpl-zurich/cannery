import json
from pathlib import Path

import pytest
from test_catlass_export import binary

from cannery.builds.ascendc import BuildError
from cannery.builds.triton_tensor import validate_triton_export


def fixture():
    metadata = json.loads((Path(__file__).parent / "fixtures/triton/matmul512.json").read_text())
    signature = {
        "inputs": [{"name": name, "shape": [512, 512], "dtype": "float16"} for name in ["a", "b"]],
        "outputs": [{"name": "out", "shape": [512, 512], "dtype": "float16", "alias": None}],
        "scalars": [],
    }
    return metadata, signature, [["input", "a"], ["input", "b"], ["output", "out"]]


def test_real_compiler_metadata_accepts_static_mixed_matrix_export():
    metadata, signature, bindings = fixture()
    export = validate_triton_export(
        binary(), metadata, signature, bindings, (64, 1, 1), arch="Ascend950DT_9582"
    )
    assert export.entrypoint == "matmul_kernel" and export.mode == "aic" and export.block_num == 64
    assert export.bindings == (("input", 0), ("input", 1), ("output", 0))
    signature["outputs"][0]["alias"] = "a"
    export = validate_triton_export(
        binary(),
        metadata,
        signature,
        bindings[:2] + [["input", "a"]],
        (64, 1, 1),
        arch="Ascend950DT_9582",
    )
    assert export.bindings[-1] == ("input", 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_pure_simt", True),
        ("parallel_mode", "simt"),
        ("workspace_size", 64),
        ("workspace_size", False),
        ("sync_block_lock_layout", 1),
        ("lock_num", 8),
        ("lock_num", True),
        ("lock_num", "0"),
        ("lock_num", -1),
        ("program_grid_transforms", []),
        ("program_grid_mapping_applied", True),
        ("row_coalescing_applied", True),
        ("debug", True),
        ("arch", "Ascend910B"),
        ("target_arch", "other"),
        ("compile_on_910_95", False),
        ("kernel_name", "other"),
        ("name", "not a symbol"),
        ("mix_mode", "other"),
        ("tensor_kinds", [0, 0]),
        ("tensor_kinds", [0, 0, True]),
        ("instrumentation_mode", "profile"),
        ("cluster_dims", [2, 1, 1]),
        ("num_ctas", 2),
        ("launch_cooperative_grid", True),
    ],
)
def test_unsupported_runtime_requirements_fail_before_native_generation(field, value):
    metadata, signature, bindings = fixture()
    metadata[field] = value
    with pytest.raises(BuildError):
        validate_triton_export(
            binary(), metadata, signature, bindings, (64, 1, 1), arch="Ascend950DT_9582"
        )


@pytest.mark.parametrize(
    "bindings,grid",
    [
        ([["input", "a"], ["input", "b"], ["output", "missing"]], (64, 1, 1)),
        ([["input", "a"], ["input", "b"], ["input", "a"]], (64, 1, 1)),
        ([["input", "a"], ["input", "b"], ["output", "out"]], (0, 1, 1)),
        ([["input", "a"], ["input", "b"], ["output", "out"]], (65535, 2, 1)),
    ],
)
def test_invalid_bindings_and_grids_fail_closed(bindings, grid):
    metadata, signature, _ = fixture()
    with pytest.raises(BuildError):
        validate_triton_export(
            binary(), metadata, signature, bindings, grid, arch="Ascend950DT_9582"
        )

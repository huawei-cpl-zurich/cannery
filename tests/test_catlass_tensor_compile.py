import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_build_sandbox import catlass_pins
from test_catlass_export import binary
from test_tensor_spec_reader import tensor_files
from test_tensor_submission_reader import tensor_files as submission_files

from cannery.builds.ascendc import BuildError
from cannery.builds.catlass_tensor_compile import CatlassTensorAdapter, package_sha256
from cannery.builds.sandbox import CatlassToolchain
from cannery.registry.tensor_manifest import validate_tensor_spec
from cannery.submissions.manifest import validate_tensor_submission


def plan():
    manifest = validate_tensor_submission(submission_files("catlass"), "candidate").manifest
    frozen = validate_tensor_spec(tensor_files(), "spec")
    return CatlassTensorAdapter().build(manifest, frozen, "int cannery_prepare_tensor_v2(void);")


def test_standalone_driver_and_legacy_profile_serialization(tmp_path):
    source = plan().wrapper
    compile(source, "<compiler>", "exec")
    assert "import cannery" not in source and "compile_tensor_exports" in source
    assert plan().command == ("python", "launch.py")
    pins = catlass_pins()
    model = CatlassToolchain.model_validate(pins)
    assert model.model_dump(mode="json") == pins
    assert model.model_dump(exclude_none=True) == pins
    assert "package_sha256" not in model.model_dump(exclude={"package_sha256"})
    updated = CatlassToolchain.model_validate(
        pins | {"package_sha256": "a" * 64, "aic_bitcode_sha256": "b" * 64}
    )
    assert updated.model_dump()["package_sha256"] == "a" * 64
    root = tmp_path / "package"
    root.mkdir()
    (root / "module.py").write_bytes(b"one")
    expected = hashlib.sha256(b"module.py\0" + hashlib.sha256(b"one").digest()).hexdigest()
    assert package_sha256(root) == expected
    (root / "__pycache__").mkdir()
    (root / "__pycache__/ignored.pyc").write_bytes(b"ignored")
    assert package_sha256(root) == expected
    (root / "module.py").write_bytes(b"two")
    assert package_sha256(root) != expected


@pytest.mark.parametrize("kind", ["kernel", "version", "header", "backend"])
def test_wrong_binding_or_loader_header_rejected(kind):
    manifest = validate_tensor_submission(submission_files("catlass"), "candidate").manifest
    frozen = validate_tensor_spec(tensor_files(), "spec")
    header = "int cannery_prepare_tensor_v2(void);"
    if kind == "kernel":
        manifest = manifest.model_copy(update={"kernel": "other/kernel"})
    elif kind == "version":
        manifest = manifest.model_copy(update={"spec_version": 2})
    elif kind == "header":
        header = "old header without initialization"
    else:
        manifest = validate_tensor_submission(submission_files("triton"), "candidate").manifest
    with pytest.raises(BuildError):
        CatlassTensorAdapter().build(manifest, frozen, header)


@pytest.fixture
def compilation(tmp_path, monkeypatch):
    namespace = {"__name__": "compiler_test"}
    exec(plan().wrapper, namespace)
    root = tmp_path / "package"
    root.mkdir()
    (root / "__init__.py").write_text("package")
    (root / "execution.py").write_text("execution")
    framework = ModuleType("catlass")
    framework.__file__ = str(root / "__init__.py")
    framework.__path__ = []
    framework.__version__ = "pinned"
    execution = ModuleType("catlass.execution")
    execution.__file__ = str(root / "execution.py")
    bridge = ModuleType("catlass.compiler_bridge")
    bridge.kernel_abi_to_dict = lambda v: v
    framework.execution = execution
    framework.compiler_bridge = bridge
    for name, module in [
        ("catlass", framework),
        ("catlass.execution", execution),
        ("catlass.compiler_bridge", bridge),
    ]:
        monkeypatch.setitem(__import__("sys").modules, name, module)
    tools = {name: tmp_path / name for name in ["bisheng", "hivmc-a5"]}
    for name, path in tools.items():
        path.write_text(name)
    monkeypatch.setattr(namespace["shutil"], "which", lambda name: str(tools[name]))
    cache = tmp_path / "cache"
    bc = cache / "bc/6f88b68ae684fbab"
    bc.mkdir(parents=True)
    pins = {
        "version": "pinned",
        "execution_sha256": namespace["_sha256"](execution.__file__),
        "hivmc_sha256": namespace["_sha256"](tools["hivmc-a5"]),
        "package_sha256": package_sha256(root),
        "bitcode_key": "6f88b68ae684fbab",
    }
    for name, key in [
        ("manifest.json", "manifest_sha256"),
        ("meta_op.aiv.c310.bc", "bitcode_sha256"),
        ("meta_op.aic.c310.bc", "aic_bitcode_sha256"),
    ]:
        (bc / name).write_text(name)
        pins[key] = namespace["_sha256"](bc / name)
    monkeypatch.setenv("CATLASS_DSL_CACHE_DIR", str(cache))
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "cann"))
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.py").write_text("from fixture_exports import build_variant as forward\n")
    control = {"calls": 0, "fault": None, "commands": []}

    def build_variant(signature):
        control["calls"] += 1
        bindings = [["input", "x"], ["input", "cache"], ["output", "indices"]]
        arguments = []
        for index, (direction, name) in enumerate(bindings):
            tensor = next(t for t in signature[direction + "s"] if t["name"] == name)
            shape = "x".join(map(str, tensor["shape"]))
            dtype = {"float16": "f16", "int32": "i32"}[tensor["dtype"]]
            arguments.append(
                {
                    "index": index,
                    "logical_index": index,
                    "kind": "pointer",
                    "field": None,
                    "scalar": None,
                    "offset": 8 * index,
                    "storage_size": 8,
                    "alignment": 4,
                    "mlir_type": f"memref<{shape}x{dtype}, #hivm.address_space<gm>>",
                }
            )
        metadata = {
            "schema_version": 3,
            "entrypoint": "forward",
            "total_size": 24,
            "arguments": arguments,
        }
        module = SimpleNamespace(
            entrypoint="forward",
            kernel_mode="aiv",
            execution_args=SimpleNamespace(kernel_abi=metadata),
            uses_scalar_print=False,
            uses_tensor_print=False,
            is_mixed=False,
            ub_static_bytes=0,
            ub_dynamic_base=-1,
        )
        path = (tmp_path if control["fault"] == "escape" else cache) / "object.o"
        path.write_bytes(binary())
        artifact = SimpleNamespace(jit_module=module, kernel_binary_path=path)
        return {"artifact": artifact, "bindings": bindings, "block_num": 1}

    monkeypatch.setitem(
        __import__("sys").modules, "fixture_exports", SimpleNamespace(build_variant=build_variant)
    )

    def compile_host(command, **kwargs):
        control["commands"].append((command, kwargs))
        Path(command[-1]).write_bytes(b"host-library")

    monkeypatch.setattr(namespace["subprocess"], "run", compile_host)
    return namespace, {"catlass": pins}, source, tmp_path / "kernel.so", control, root, bc


def test_tensor_factory_exports_alias_and_compiles_native_bridge(compilation):
    ns, pins, source, output, control, _, _ = compilation
    ns["compile_tensor_exports"](pins, source, output)
    assert control["calls"] == 1 and output.read_bytes() == b"host-library"
    native = output.with_suffix(".cpp").read_text()
    assert "cannery_prepare_tensor_v2" in native and "out[0].data==in[1].data" in native
    assert "in[0].data,in[1].data,out[1].data" in native
    assert output.with_name("tensor_abi.h").read_text() == ns["PARAMETERS"]["abi_header"]
    assert len(control["commands"]) == 1 and control["commands"][0][1]["timeout"] == 60


@pytest.mark.parametrize("fault", ["package", "aic", "missing_pin", "escape"])
def test_changed_pins_and_unconfined_exports_fail_before_link(compilation, fault):
    ns, pins, source, output, control, root, bc = compilation
    if fault == "package":
        (root / "extra.py").write_text("changed")
    elif fault == "aic":
        (bc / "meta_op.aic.c310.bc").write_text("changed")
    elif fault == "missing_pin":
        pins["catlass"].pop("package_sha256")
    else:
        control["fault"] = "escape"
    with pytest.raises(ValueError):
        ns["compile_tensor_exports"](pins, source, output)
    assert not control["commands"] and not output.exists()

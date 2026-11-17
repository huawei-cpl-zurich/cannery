import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_catlass_export import binary
from test_tensor_spec_reader import tensor_files
from test_tensor_submission_reader import tensor_files as submission_files
from test_triton_tensor_export import fixture as metadata_fixture

from cannery.builds.catlass_tensor_compile import package_sha256
from cannery.builds.sandbox import TritonToolchain
from cannery.builds.triton_tensor_compile import TritonTensorAdapter
from cannery.registry.tensor_manifest import validate_tensor_spec
from cannery.submissions.manifest import validate_tensor_submission


def plan():
    manifest = validate_tensor_submission(submission_files(), "candidate").manifest
    return TritonTensorAdapter().build(
        manifest,
        validate_tensor_spec(tensor_files(), "spec"),
        "int cannery_prepare_tensor_v2(void);",
    )


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    ns = {"__name__": "triton_compiler_test"}
    exec(plan().wrapper, ns)
    package = tmp_path / "package/triton"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("pinned frontend")
    framework = ModuleType("triton")
    framework.__file__ = str(package / "__init__.py")
    framework.__version__ = "3.6.0"
    framework.__path__ = []
    compiler_module = ModuleType("triton.compiler")
    compiler_module.ASTSource = lambda fn, **kwargs: SimpleNamespace(fn=fn, **kwargs)
    backend = ModuleType("triton.backends.compiler")
    backend.GPUTarget = lambda *args: args
    controls = {"fault": None, "calls": [], "commands": []}
    metadata = metadata_fixture()[0]

    def compile_source(src, *, target):
        controls["calls"].append((src, target))
        assert src.signature == {"x": "*fp16", "cache": "*fp16", "indices": "*i32"}
        values = copy.deepcopy(metadata)
        if controls["fault"] == "workspace":
            values["workspace_size"] = 16
        return SimpleNamespace(
            asm={"npubin": binary()}, metadata=SimpleNamespace(_asdict=lambda: values)
        )

    framework.compile = compile_source
    for name, module in [
        ("triton", framework),
        ("triton.compiler", compiler_module),
        ("triton.backends.compiler", backend),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    root = tmp_path / "cann"
    tool = root / "tools/bishengir/bin/bishengir-compile"
    tool.parent.mkdir(parents=True)
    tool.write_text("trusted compiler")
    linker = tmp_path / "bisheng"
    linker.write_text("trusted linker")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    monkeypatch.setattr(ns["shutil"], "which", lambda _: str(linker))
    pins = {
        "version": "3.6.0",
        "package_root": str(package),
        "package_sha256": package_sha256(package),
        "bishengir_sha256": ns["sha"](tool),
        "bisheng_sha256": ns["sha"](linker),
    }
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.py").write_text("from fixture_triton_exports import factory as forward\n")

    def factory(signature):
        if controls["fault"] == "mutate":
            signature["inputs"][0]["shape"][:] = [99]
        names = ["x", "cache", "indices", "OFFSET"]
        if controls["fault"] == "unbound":
            names.append("unbound_runtime_scalar")
        return {
            "kernel": SimpleNamespace(arg_names=names),
            "grid": [1, 1, 1],
            "constants": {"OFFSET": 1},
            "bindings": [
                ["indices", "output", "indices"],
                ["cache", "input", "cache"],
                ["x", "input", "x"],
            ],
        }

    monkeypatch.setitem(sys.modules, "fixture_triton_exports", SimpleNamespace(factory=factory))

    def link(command, **kwargs):
        controls["commands"].append(command)
        Path(command[-1]).write_bytes(b"host-library")

    monkeypatch.setattr(ns["subprocess"], "run", link)
    config = {"triton": pins, "runtime": {"soc": "Ascend950DT_9582"}}
    return ns, config, source, tmp_path / "kernel.so", controls, package, tool


def test_offline_driver_compiles_frozen_types_and_native_grid(compiler):
    ns, config, source, output, control, _, _ = compiler
    assert "import cannery." not in plan().wrapper
    ns["compile_triton"](config, source, output)
    assert output.read_bytes() == b"host-library" and len(control["calls"]) == 1
    args, target = control["calls"][0]
    assert target == ("npu", "Ascend950DT_9582", 0) and args.constexprs == {"OFFSET": 1}
    assert "void* sync; void* workspace;" in output.with_suffix(".cpp").read_text()
    assert control["commands"][0][-2:] == ["-o", str(output)]


@pytest.mark.parametrize("fault", ["package", "compiler", "workspace", "unbound"])
def test_changed_or_unsupported_toolchains_and_exports_never_link(compiler, fault):
    ns, config, source, output, control, package, tool = compiler
    if fault == "package":
        (package / "changed.py").write_text("changed")
    elif fault == "compiler":
        tool.write_text("changed")
    else:
        control["fault"] = fault
    with pytest.raises(ValueError):
        ns["compile_triton"](config, source, output)
    assert not control["commands"] and not output.exists()


def test_author_callback_cannot_change_native_dispatch_domain(compiler):
    ns, config, source, output, control, _, _ = compiler
    before = copy.deepcopy(ns["PARAMETERS"]["variants"])
    control["fault"] = "mutate"
    ns["compile_triton"](config, source, output)
    assert ns["PARAMETERS"]["variants"] == before
    assert "sizes[0]==UINT64_C(99)" not in output.with_suffix(".cpp").read_text()


@pytest.mark.parametrize(
    "root", ["/", "relative/triton", "/opt/a/../triton", "/home/u/bad,root/triton"]
)
def test_toolchain_requires_specific_unambiguous_package_root(root):
    with pytest.raises(ValueError):
        TritonToolchain(
            version="3.6.0",
            package_root=root,
            package_sha256="a" * 64,
            bishengir_sha256="b" * 64,
            bisheng_sha256="c" * 64,
        )

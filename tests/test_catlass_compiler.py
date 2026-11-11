"""Trusted driver tests use a fake framework, never an actual NPU or compiler."""

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from submission_fixtures import submission_files
from test_catlass_contract import native_manifest
from test_catlass_export import binary, metadata

from cannery.builds.ascendc import BuildError
from cannery.builds.catlass_compile import CatlassAdapter
from cannery.submissions.manifest import validate_submission


def manifest(files, value):
    return validate_submission(
        {**files, "candidate/submission.yaml": yaml.safe_dump(value).encode()}, "candidate"
    ).manifest


def test_driver_generation_never_imports_author_python_and_keeps_source_namespace():
    files, value = native_manifest()
    value["implementation"]["entrypoint"] = "launch.py:forward"
    value["implementation"]["build"]["inputs"] = ["launch.py"]
    files["candidate/launch.py"] = b"raise AssertionError('must not execute')"
    plan = CatlassAdapter().build(manifest(files, value))
    assert plan.wrapper_name == "launch.py" and plan.command == ("python", "launch.py")
    assert '"counts": [17, 21]' in plan.wrapper
    assert "import cannery" not in plan.wrapper
    assert "raise AssertionError('must not execute')" not in plan.wrapper
    namespace = {"__name__": "driver_definition_only"}
    exec(compile(plan.wrapper, "<driver>", "exec"), namespace)
    assert callable(namespace["compile_exports"])
    assert "catlass" not in sys.modules


def test_source_only_or_empty_only_contract_has_no_compiler_plan():
    files = submission_files()
    value = yaml.safe_load(files["candidate/submission.yaml"])
    with pytest.raises(BuildError):
        CatlassAdapter().build(manifest(files, value))
    files, value = native_manifest()
    value["implementation"]["build"]["specializations"] = [{"shape": [0], "dtype": "float32"}]
    with pytest.raises(BuildError, match="nonempty"):
        CatlassAdapter().build(manifest(files, value))


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    files, value = native_manifest()
    plan = CatlassAdapter().build(manifest(files, value))
    namespace = {"__name__": "driver_definition_only"}
    exec(compile(plan.wrapper, "<driver>", "exec"), namespace)
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "kernel.py").write_text("forward = object()\n")
    cache = tmp_path / "cache"
    bc = cache / "bc/0123456789abcdef"
    bc.mkdir(parents=True)
    (bc / "manifest.json").write_bytes(b"trusted manifest")
    (bc / "meta_op.aiv.c310.bc").write_bytes(b"trusted bitcode")
    tool = tmp_path / "hivmc-a5"
    tool.write_bytes(b"trusted compiler")
    execution_file = tmp_path / "execution.py"
    execution_file.write_bytes(b"pinned framework")
    pins = dict(
        version="pinned",
        bitcode_key="0123456789abcdef",
        execution_sha256=hashlib.sha256(execution_file.read_bytes()).hexdigest(),
        hivmc_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((bc / "manifest.json").read_bytes()).hexdigest(),
        bitcode_sha256=hashlib.sha256((bc / "meta_op.aiv.c310.bc").read_bytes()).hexdigest(),
    )
    counts, commands = [], []
    options = {"bad_abi": False}

    def compile_kernel(kernel, x, y, *, options):
        count = x.shape[0]
        counts.append(count)
        assert x is y and options == "--npu-arch 3510"
        directory = cache / str(count)
        directory.mkdir()
        path = directory / "kernel.o"
        path.write_bytes(binary(count))
        record = metadata(count)
        if compiler_options["bad_abi"]:
            record["kernel_mode"] = "mix"
        module = SimpleNamespace(**{k: v for k, v in record.items() if k != "kernel_abi"})
        module.execution_args = SimpleNamespace(kernel_abi=record["kernel_abi"])
        return SimpleNamespace(jit_module=module, kernel_binary_path=path)

    compiler_options = options
    framework = ModuleType("catlass")
    framework.__version__ = "pinned"
    framework.__path__ = []
    tla = ModuleType("catlass.tla")
    tla.Float32 = "float32"
    tla.compile = compile_kernel
    runtime = ModuleType("catlass.tla.runtime")
    runtime.make_fake_tensor = lambda dtype, shape, stride: SimpleNamespace(shape=shape)
    bridge = ModuleType("catlass.compiler_bridge")
    bridge.kernel_abi_to_dict = lambda value: value
    execution = ModuleType("catlass.execution")
    execution.__file__ = str(execution_file)
    framework.tla = tla
    framework.execution = execution
    for module in [framework, tla, runtime, bridge, execution]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("CATLASS_DSL_CACHE_DIR", str(cache))
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "cann"))
    monkeypatch.setattr(
        "shutil.which", lambda name: str(tool) if name == "hivmc-a5" else "/bisheng"
    )

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] and kwargs["timeout"] == 60
        Path(command[-1]).write_bytes(b"mock host ELF")

    monkeypatch.setattr("subprocess.run", run)
    return (
        namespace["compile_exports"],
        pins,
        source,
        tmp_path / "kernel.so",
        counts,
        commands,
        options,
    )


def test_all_unique_positive_variants_export_before_native_host_compilation(compiler):
    compile_exports, pins, source, output, counts, commands, _ = compiler
    compile_exports({"catlass": pins}, source, output)
    assert counts == [17, 21] and len(commands) == 1
    assert commands[0][:4] == ["/bisheng", "-shared", "-fPIC", "-O2"]
    assert "-Wl,-z,defs" in commands[0]
    assert output.read_bytes() == b"mock host ELF"
    assert "object_0" in output.with_suffix(".cpp").read_text()


@pytest.mark.parametrize(
    "field", ["version", "execution_sha256", "hivmc_sha256", "manifest_sha256", "bitcode_sha256"]
)
def test_mismatched_toolchain_never_imports_author_or_invokes_compiler(compiler, field):
    compile_exports, pins, source, output, counts, commands, _ = compiler
    (source / "kernel.py").write_text("raise AssertionError('must not import')")
    pins[field] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        compile_exports({"catlass": pins}, source, output)
    assert not counts and not commands and not output.exists()


def test_unsupported_native_abi_never_reaches_cpp_linker(compiler):
    compile_exports, pins, source, output, counts, commands, options = compiler
    options["bad_abi"] = True
    with pytest.raises(ValueError, match="Unsupported CATLASS export ABI"):
        compile_exports({"catlass": pins}, source, output)
    assert counts == [17] and not commands and not output.exists()

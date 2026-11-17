import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_ascendc_tensor_export import elf, fixture
from test_tensor_spec_reader import tensor_files
from test_tensor_submission_reader import tensor_files as submission_files

from cannery.builds.ascendc_tensor_compile import AscendCTensorAdapter
from cannery.registry.tensor_manifest import validate_tensor_spec
from cannery.submissions.manifest import validate_tensor_submission


def plan():
    files = submission_files("ascendc")
    value = yaml.safe_load(files["candidate/submission.yaml"])
    value["implementation"]["entrypoint"] = "kernel.py:forward"
    value["implementation"]["build"]["inputs"].append("kernel.py")
    files["candidate/submission.yaml"] = yaml.safe_dump(value).encode()
    files["candidate/kernel.py"] = b"def forward(signature): pass\n"
    manifest = validate_tensor_submission(files, "candidate").manifest
    return AscendCTensorAdapter().build(
        manifest,
        validate_tensor_spec(tensor_files(), "spec"),
        "int cannery_prepare_tensor_v2(void);",
    )


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    ns = {"__name__": "ascendc_compiler_test"}
    exec(plan().wrapper, ns)
    root = tmp_path / "cann"
    root.mkdir()
    tools = {name: root / name for name in ["bisheng", "ld.lld"]}
    for name, path in tools.items():
        path.write_text("pinned " + name)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    monkeypatch.setattr(ns["shutil"], "which", lambda name: str(tools[name]))
    config = {
        "ascendc": {
            "bisheng_sha256": ns["sha"](tools["bisheng"]),
            "ld_sha256": ns["sha"](tools["ld.lld"]),
        }
    }
    control = {"fault": None, "commands": []}
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.py").write_text("from fixture_ascendc_exports import factory as forward\n")

    def factory(signature):
        if control["fault"] == "mutate":
            signature["inputs"][0]["shape"][:] = [99]
        return {
            "source": '#include "genuine_upstream.h"\n',
            "entrypoint": "ascendc_tensor_add",
            "bindings": [["input", "x"], ["input", "cache"], ["output", "indices"]],
            "block_num": 1,
            "include_dirs": control.get("include_dirs", []),
        }

    monkeypatch.setitem(sys.modules, "fixture_ascendc_exports", SimpleNamespace(factory=factory))

    def run(command, **kwargs):
        control["commands"].append(command)
        assert kwargs["check"] is True and kwargs["timeout"] <= 120
        out = Path(command[-1])
        if out.suffix == ".o":
            out.write_bytes(b"relocatable")
        elif out.suffix == ".bin":
            sections = fixture()
            if control["fault"] == "abi":
                sections["__CCE_KernelArgSize"] = bytes(8)
            out.write_bytes(elf(sections))
        else:
            out.write_bytes(b"host-library")

    monkeypatch.setattr(ns["subprocess"], "run", run)
    return ns, config, source, tmp_path / "kernel.so", control, tools


def test_offline_driver_validates_compiler_metadata_and_links_static_bridge(compiler):
    ns, config, source, out, control, _ = compiler
    assert "import cannery." not in plan().wrapper
    ns["compile_ascendc"](config, source, out)
    assert out.read_bytes() == b"host-library" and len(control["commands"]) == 3
    assert "--cce-aicore-only" in control["commands"][0]
    assert control["commands"][1][1:4] == ["-m", "aicorelinux", "-Ttext=0"]
    assert "-Wl,-z,defs" in control["commands"][2]
    assert "scalar[0].bits==UINT64_C(1)" in out.with_suffix(".cpp").read_text()


@pytest.mark.parametrize("tool", ["bisheng", "ld.lld"])
def test_changed_compiler_or_linker_rejected_before_author_import(compiler, tool):
    ns, config, source, out, control, tools = compiler
    tools[tool].write_text("changed")
    with pytest.raises(ValueError, match="fingerprint"):
        ns["compile_ascendc"](config, source, out)
    assert not control["commands"]


def test_unsupported_compiled_abi_never_produces_host_bridge(compiler):
    ns, config, source, out, control, _ = compiler
    control["fault"] = "abi"
    with pytest.raises(ValueError):
        ns["compile_ascendc"](config, source, out)
    assert len(control["commands"]) == 2 and not out.exists()


def test_author_mutation_cannot_change_native_dispatch(compiler):
    ns, config, source, out, control, _ = compiler
    control["fault"] = "mutate"
    frozen = copy.deepcopy(ns["PARAMETERS"]["variants"])
    ns["compile_ascendc"](config, source, out)
    assert ns["PARAMETERS"]["variants"] == frozen
    assert "sizes[0]==UINT64_C(99)" not in out.with_suffix(".cpp").read_text()


def test_existing_cpp_tensor_declarations_fail_with_explicit_factory_requirement():
    manifest = validate_tensor_submission(submission_files("ascendc"), "candidate").manifest
    with pytest.raises(ValueError, match="Python source factory"):
        AscendCTensorAdapter().build(
            manifest,
            validate_tensor_spec(tensor_files(), "spec"),
            "int cannery_prepare_tensor_v2(void);",
        )


@pytest.mark.parametrize("directory", ["../", "/tmp", "missing"])
def test_source_include_paths_cannot_escape_snapshot(compiler, directory):
    ns, config, source, out, control, _ = compiler
    control["include_dirs"] = [directory]
    with pytest.raises(ValueError, match="include directory"):
        ns["compile_ascendc"](config, source, out)
    assert not control["commands"]

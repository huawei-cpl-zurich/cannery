import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from typer.testing import CliRunner

from cannery.cli import app
from cannery.web.downloads import snippets


def release():
    return {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "kernel": "team/identity",
        "device": "a5",
        "runtime": {
            "machine": "aarch64",
            "python": "3.12",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "cann": "9.3.0",
            "soc": "Ascend950DT_9582",
        },
        "tolerances": [{"rtol": 0.000123456789, "atol": 0.0}],
        "specializations": [{"shape": [2, 3], "dtype": "float32"}],
        "hub": {
            "repo_id": "team/identity",
            "release_revision": "release-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        },
    }


def test_python_snippet_sets_endpoint_and_uses_pinned_public_loader(monkeypatch):
    loader = Mock()
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=loader))
    monkeypatch.setenv("HF_ENDPOINT", "https://original.example")
    code = snippets(release(), "https://cannery.example", True)["python"]
    exec(compile(code, "<download snippet>", "exec"), {})
    import os

    assert os.environ["HF_ENDPOINT"] == "https://cannery.example"
    loader.assert_called_once_with(
        "team/identity",
        revision="release-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        backend="cann",
        trust_remote_code=True,
    )


def test_cli_snippet_works_without_runtime_detection_and_keeps_full_precision(
    tmp_path, monkeypatch
):
    item = release()
    lines = snippets(item, "https://cannery.example", True)["cli"].splitlines()
    end = lines.index("CANNERY_RUNTIME")
    runtime = json.loads("\n".join(lines[1:end]))
    filename = shlex.split(lines[0])[2]
    (tmp_path / filename).write_text(json.dumps(runtime))
    monkeypatch.chdir(tmp_path)
    from cannery.releases.client import ReleaseClient

    calls = []

    def download(self, selection, cache):
        calls.append(selection)
        return SimpleNamespace(
            manifest_path=Path("release.json"),
            manifest_sha256="a" * 64,
            package_path=Path("kernel.tar"),
            manifest=SimpleNamespace(package=SimpleNamespace(sha256="b" * 64)),
        )

    monkeypatch.setattr(ReleaseClient, "get", download)
    result = CliRunner().invoke(app, shlex.split(lines[end + 1])[1:])
    assert result.exit_code == 0, result.output
    request = calls[0]
    assert str(request.release_id) == item["id"] and request.kernel == item["kernel"]
    assert request.runtime == item["runtime"] and request.shape == (2, 3)
    assert request.rtol == 0.000123456789 and request.atol == 0


def test_python_not_advertised_when_hub_or_snapshot_is_missing():
    assert snippets(release(), "https://cannery.example", False)["python"] is None
    item = release() | {"hub": None}
    assert snippets(item, "https://cannery.example", True)["python"] is None
    assert "--release-id" in snippets(item, "https://cannery.example", True)["cli"]

import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from cannery.git_intake import GitSource
from cannery.git_intake.process import IntakeError, git
from cannery.git_intake.snapshots import SelectedTree, SnapshotStore, extract_tree


def source(commit):
    return GitSource(url="https://example.com/repo.git", commit=commit, path="kernels/op")


def test_only_selected_files_survive_upstream_loss_and_gc(tmp_path, bare_upstream) -> None:
    upstream, first, second = bare_upstream
    store = SnapshotStore(tmp_path / "snapshots")
    one = store.commit("a" * 64, source(first), extract_tree(upstream, source(first)))
    two = store.commit("a" * 64, source(second), extract_tree(upstream, source(second)))
    assert one.internal_snapshot_sha != first
    assert two.internal_snapshot_sha != second
    repository = store.root / ("a" * 64 + ".git")
    assert (
        git(repository, "rev-parse", two.internal_snapshot_sha + "^").strip().decode()
        == one.internal_snapshot_sha
    )
    with pytest.raises(IntakeError):
        git(repository, "cat-file", "-e", first)
    shutil.rmtree(upstream)
    git(repository, "gc", "--prune=now")
    assert store.read(one).files == {"kernels/op/reference.py": b"version = 1\n"}
    assert store.read(two).files == {"kernels/op/reference.py": b"version = 2\n"}
    refs = git(repository, "for-each-ref", "--format=%(refname)").decode()
    assert refs.count("refs/cannery/snapshots/") == 2


def test_duplicate_concurrent_imports_are_idempotent(tmp_path, bare_upstream) -> None:
    upstream, first, _ = bare_upstream
    selected = extract_tree(upstream, source(first))
    store = SnapshotStore(tmp_path / "snapshots")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: store.commit("b" * 64, source(first), selected), range(2))
        )
    assert results[0] == results[1]
    assert store.read(results[0]).files == selected.files


@pytest.mark.parametrize(
    "mode,data",
    [
        ("120000", b"/etc/passwd"),
        ("160000", b"x"),
        ("100644", b"version https://git-lfs.github.com/spec/v1\n"),
    ],
)
def test_unsupported_selected_files_are_rejected(tmp_path, bare_upstream, mode, data) -> None:
    upstream, first, _ = bare_upstream
    if mode in ("120000", "160000"):
        oid = (
            first
            if mode == "160000"
            else git(upstream, "hash-object", "-w", "--stdin", data=data).strip().decode()
        )
        tree = (
            git(
                upstream,
                "mktree",
                data=f"{mode} {'commit' if mode == '160000' else 'blob'} {oid}\tbad\n".encode(),
            )
            .strip()
            .decode()
        )
        from cannery.git_intake.snapshots import read_tree

        with pytest.raises(IntakeError):
            read_tree(upstream, tree)
    else:
        selected = extract_tree(upstream, source(first))
        selected.files["kernels/op/pointer"] = data
        # Construct a retained tree with this blob, then verify read-time rejection.
        selected.modes["kernels/op/pointer"] = "100644"
        store = SnapshotStore(tmp_path / "store")
        binding = store.commit("c" * 64, source(first), selected)
        with pytest.raises(IntakeError, match="LFS"):
            store.read(binding)


def test_invalid_store_paths_and_byte_budget(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    selected = SelectedTree({".git/config": b"bad"}, {".git/config": "100644"}, "a" * 40)
    with pytest.raises(IntakeError):
        store.commit("a" * 64, source("a" * 40), selected)
    with pytest.raises(IntakeError):
        store.commit("../escape", source("a" * 40), selected)


def test_store_enforces_file_and_byte_limits_before_writing(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    for files in ({"huge": b"x" * (16 * 1024 * 1024 + 1)}, {f"file{i}": b"" for i in range(257)}):
        selected = SelectedTree(files, dict.fromkeys(files, "100644"), "a" * 40)
        with pytest.raises(IntakeError):
            store.commit("a" * 64, source("a" * 40), selected)
    assert list(tmp_path.iterdir()) == []

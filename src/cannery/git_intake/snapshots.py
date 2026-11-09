"""Selected Git-tree bytes and synthetic, protected per-kernel history."""

import fcntl
import hashlib
import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from cannery.git_intake import GitSource, SnapshotBinding, relative_source_path
from cannery.git_intake.process import IntakeError, clean_environment, git

MAX_BYTES = 16 * 1024 * 1024
MAX_FILES = 256


@dataclass(frozen=True)
class SelectedTree:
    files: dict[str, bytes]
    modes: dict[str, str]
    external_tree_id: str


def safe_path(path: str) -> str:
    relative_source_path(path)
    if any(part.lower() == ".git" for part in path.split("/")):
        raise IntakeError("Git administration paths are forbidden")
    return path


def read_tree(repository: Path, revision: str, paths: tuple[str, ...] = ()) -> SelectedTree:
    records = git(repository, "ls-tree", "-r", "-z", "--full-tree", revision, "--", *paths)
    entries = [record for record in records.split(b"\0") if record]
    if not entries or len(entries) > MAX_FILES:
        raise IntakeError("Selected tree exceeds file-count budget or is empty")
    files = {}
    modes = {}
    total = 0
    for entry in entries:
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        try:
            name = safe_path(raw_path.decode("utf-8"))
        except (UnicodeError, ValueError):
            raise IntakeError("Selected tree contains unsafe paths") from None
        if kind != "blob" or mode not in ("100644", "100755"):
            raise IntakeError("Symlinks and submodules are unsupported")
        size = int(git(repository, "cat-file", "-s", oid).strip())
        total += size
        if total > MAX_BYTES:
            raise IntakeError("Selected tree exceeds byte budget")
        data = git(repository, "cat-file", "blob", oid)
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise IntakeError("Git LFS pointers are unsupported")
        files[name], modes[name] = data, mode
    return SelectedTree(files, modes, "")


def extract_tree(repository: Path, source: GitSource) -> SelectedTree:
    location = source.commit + "^{tree}" if source.path == "." else f"{source.commit}:{source.path}"
    tree_id = git(repository, "rev-parse", "--verify", location).decode().strip()
    if git(repository, "cat-file", "-t", tree_id).strip() != b"tree":
        raise IntakeError("Source path must name a committed directory")
    paths = () if source.path == "." else (source.path,) + source.dependencies
    for path in source.dependencies:
        git(repository, "rev-parse", "--verify", f"{source.commit}:{path}")
    selected = read_tree(repository, source.commit, paths)
    return SelectedTree(selected.files, selected.modes, tree_id)


@contextmanager
def kernel_lock(root: Path, key: str) -> Iterator[None]:
    with (root / f"{key}.lock").open("a") as stream:
        deadline = time.monotonic() + 5
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise IntakeError("Kernel snapshot store is busy") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, key: str, source: GitSource, selected: SelectedTree) -> SnapshotBinding:
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise IntakeError("Invalid internal repository identity")
        if not selected.files or len(selected.files) > MAX_FILES:
            raise IntakeError("Invalid selected tree")
        if sum(map(len, selected.files.values())) > MAX_BYTES:
            raise IntakeError("Selected tree exceeds byte budget")
        for name in selected.files:
            safe_path(name)
            if selected.modes[name] not in ("100644", "100755"):
                raise IntakeError("Unsupported Git file mode")
        content = {
            name: [selected.modes[name], hashlib.sha256(data).hexdigest()]
            for name, data in sorted(selected.files.items())
        }
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        identity = hashlib.sha256((source.model_dump_json() + digest).encode()).hexdigest()
        reference = f"refs/cannery/snapshots/{identity}"
        repository = self.root / f"{key}.git"
        with kernel_lock(self.root, key):
            if not repository.exists():
                git(repository, "init", "--bare", str(repository))
            refs = git(repository, "for-each-ref", "--format=%(refname) %(objectname)")
            existing = dict(line.split() for line in refs.decode().splitlines())
            if reference in existing:
                commit = existing[reference]
            else:
                env = clean_environment()
                env.update(
                    GIT_AUTHOR_NAME="CANNery",
                    GIT_COMMITTER_NAME="CANNery",
                    GIT_AUTHOR_EMAIL="cannery@localhost",
                    GIT_COMMITTER_EMAIL="cannery@localhost",
                )
                with TemporaryDirectory(dir=self.root) as temporary:
                    env["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
                    records = []
                    for name, data in sorted(selected.files.items()):
                        oid = git(repository, "hash-object", "-w", "--stdin", data=data).strip()
                        records.append(
                            selected.modes[name].encode()
                            + b" "
                            + oid
                            + b"\t"
                            + name.encode()
                            + b"\0"
                        )
                    git(
                        repository,
                        "update-index",
                        "-z",
                        "--index-info",
                        data=b"".join(records),
                        env=env,
                    )
                    tree = git(repository, "write-tree", env=env).decode().strip()
                parent = existing.get("refs/heads/accepted")
                arguments = ["commit-tree", tree]
                if parent:
                    arguments += ["-p", parent]
                commit = (
                    git(
                        repository,
                        *arguments,
                        data=("CANNery snapshot\n" + source.model_dump_json() + "\n").encode(),
                        env=env,
                    )
                    .decode()
                    .strip()
                )
                transaction = (
                    f"start\nupdate refs/heads/accepted {commit} "
                    f"{parent or '0' * 40}\ncreate {reference} {commit}\nprepare\ncommit\n"
                )
                git(repository, "update-ref", "--stdin", data=transaction.encode())
        return SnapshotBinding(
            source=source,
            internal_repository=key,
            internal_snapshot_sha=commit,
            external_tree_id=selected.external_tree_id,
            content_sha256=digest,
        )

    def read(self, binding: SnapshotBinding) -> SelectedTree:
        selected = read_tree(
            self.root / f"{binding.internal_repository}.git", binding.internal_snapshot_sha
        )
        content = {
            name: [selected.modes[name], hashlib.sha256(data).hexdigest()]
            for name, data in sorted(selected.files.items())
        }
        if (
            hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            != binding.content_sha256
        ):
            raise IntakeError("Retained snapshot digest mismatch")
        return selected

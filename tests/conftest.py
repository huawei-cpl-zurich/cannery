import subprocess

import pytest


@pytest.fixture
def bare_upstream(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    git("init", "--initial-branch=main")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "core.hooksPath", "/dev/null")
    (source / "kernels" / "op").mkdir(parents=True)
    selected = source / "kernels" / "op" / "reference.py"
    selected.write_text("version = 1\n")
    (source / "unrelated.txt").write_text("not selected\n")
    git("add", ".")
    git("commit", "-m", "first")
    first = git("rev-parse", "HEAD")
    selected.write_text("version = 2\n")
    git("commit", "-am", "second")
    second = git("rev-parse", "HEAD")
    bare = tmp_path / "upstream.git"
    git("clone", "--bare", str(source), str(bare))
    return bare, first, second

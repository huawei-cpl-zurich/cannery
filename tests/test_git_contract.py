import subprocess

import pytest
from pydantic import ValidationError

from cannery.git_intake import GitSource, SnapshotBinding


@pytest.mark.parametrize(
    "change",
    [
        {"type": "archive"},
        {"url": "/tmp/source"},
        {"url": "file:///tmp/source"},
        {"url": "https://token:secret@example.com/repo"},
        {"url": "https://host/repo?token=secret"},
        {"url": "ssh://host:bad/repo"},
        {"commit": "main"},
        {"commit": "abc123"},
        {"path": "../kernel"},
        {"path": "/kernel"},
        {"dependencies": ["../shared"]},
        {"archive": "source.zip"},
    ],
)
def test_invalid_source_binding(change) -> None:
    values = dict(url="https://example.com/repo.git", commit="a" * 40, path="kernels/op")
    with pytest.raises(ValidationError):
        GitSource(**{**values, **change})


def test_source_and_snapshot_identities_are_separate() -> None:
    source = GitSource(
        url="ssh://git@example.com/repo.git",
        commit="a" * 40,
        path="kernels/op",
        dependencies=("shared/header.h",),
    )
    binding = SnapshotBinding(
        source=source,
        internal_repository="kernel-123",
        internal_snapshot_sha="b" * 40,
        external_tree_id="c" * 40,
        content_sha256="d" * 64,
    )
    assert binding.source.commit != binding.internal_snapshot_sha
    assert binding.source.dependencies == ("shared/header.h",)


def test_bare_fixture_retains_two_pinned_source_versions(bare_upstream) -> None:
    bare, first, second = bare_upstream
    for revision, expected in ((first, "version = 1"), (second, "version = 2")):
        output = subprocess.check_output(
            [
                "git",
                "--git-dir",
                str(bare),
                "show",
                f"{revision}:kernels/op/reference.py",
            ],
            text=True,
        )
        assert output.strip() == expected
    # This is a test fixture, not a production importer or upstream mirror policy.

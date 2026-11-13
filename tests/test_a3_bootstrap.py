"""The A3 driver bootstrap must fail before the submitted runtime can execute."""

import ast
import ctypes
from types import SimpleNamespace

import pytest

from cannery.builds.sandbox import A3_BOOTSTRAP


@pytest.mark.parametrize(
    "fault", [None, "mask", "physical", "bounding", "uid", "thread_caps", "thread_privileges"]
)
def test_bootstrap_drops_privileges_on_every_thread_before_author_code(fault):
    events = []

    def physical(_, pointer):
        pointer._obj.value = 3 if fault == "physical" else 4
        return 0

    def prctl(kind, value, *args):
        events.append((kind, value))
        return -1 if fault == "bounding" and kind == 24 and value == 8 else 0

    class Status:
        def read_text(self):
            cap = "1" if fault == "thread_caps" else "0"
            priv = "0" if fault == "thread_privileges" else "1"
            return f"Uid: 10001 10001 10001 10001\nCapEff: {cap}\nCapPrm: 0\nNoNewPrivs: {priv}"

    class Path:
        def __init__(self, name):
            self.name = name

        def exists(self):
            return False

        def read_text(self):
            assert self.name == "/proc/sys/kernel/cap_last_cap"
            return "40"

        def glob(self, pattern):
            assert self.name == "/proc/self/task" and pattern == "*/status"
            return [Status(), Status()]

    library = SimpleNamespace(
        aclrtGetPhyDevIdByUserDevId=physical,
        aclrtGetPhyDevIdByLogicDevId=physical,
        prctl=prctl,
        fflush=lambda _: None,
    )
    fake_ctypes = SimpleNamespace(
        CDLL=lambda *args, **kwargs: library,
        c_int32=ctypes.c_int32,
        c_int=ctypes.c_int,
        byref=ctypes.byref,
        POINTER=ctypes.POINTER,
    )
    fake_os = SimpleNamespace(
        dup=lambda _: 3,
        dup2=lambda *args: None,
        close=lambda _: None,
        environ={"ASCEND_HOME_PATH": "/sdk", "CANNERY_PHYSICAL_DEVICE": "4"},
        setgroups=lambda _: events.append("groups"),
        setgid=lambda _: events.append("gid"),
        setuid=lambda _: events.append("uid"),
        getuid=lambda: 0 if fault == "uid" else 10001,
        geteuid=lambda: 10001,
    )
    torch = SimpleNamespace(
        npu=SimpleNamespace(
            device_count=lambda: 2 if fault == "mask" else 1,
            set_device=lambda _: events.append("initialize"),
            synchronize=lambda: None,
        ),
        empty=lambda *args, **kwargs: None,
    )
    tree = ast.parse(A3_BOOTSTRAP.replace("CANNERY_DEVICE_GROUP", "1002"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    scope = dict(
        ctypes=fake_ctypes,
        os=fake_os,
        pathlib=SimpleNamespace(Path=Path),
        torch=torch,
        sys=SimpleNamespace(
            stdout=SimpleNamespace(fileno=lambda: 1, flush=lambda: None),
            stderr=SimpleNamespace(fileno=lambda: 2),
        ),
    )

    def runtime():
        exec(compile(tree, "bootstrap", "exec"), scope)
        events.append("author")

    if fault:
        with pytest.raises(AssertionError):
            runtime()
        assert "author" not in events
    else:
        runtime()
        assert events.index("initialize") < events.index("uid") < events.index("author")
        assert [(24, cap) for cap in range(41)] == [
            event for event in events if isinstance(event, tuple) and event[0] == 24
        ]

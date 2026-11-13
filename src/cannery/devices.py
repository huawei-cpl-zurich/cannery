"""Allowlisted hardware families for operator-pinned device tracks."""

from typing import Literal

Device = Literal["a3", "a5"]


def matches_soc(device: Device, soc: str) -> bool:
    return (
        device == "a3"
        and soc.startswith("Ascend910B")
        or device == "a5"
        and soc.startswith("Ascend950")
    )


def compiler_arch(device: Device) -> str:
    return {"a3": "dav-2201", "a5": "dav-3510"}[device]

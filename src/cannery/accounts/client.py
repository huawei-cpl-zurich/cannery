"""Origin-bound CLI credentials stored in private files, never URL parameters."""

import hashlib
import ipaddress
import json
import os
import stat
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


def origin(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or any(
            (
                parsed.username,
                parsed.password,
                parsed.query,
                parsed.fragment,
                parsed.path.strip("/"),
            )
        )
    ):
        raise ValueError("Expected an HTTP service origin without credentials or path")
    return f"{parsed.scheme}://{parsed.netloc.lower()}"


def secure(url: str) -> str:
    value = origin(url)
    parsed = urlsplit(value)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if parsed.scheme != "https" and not loopback:
        raise ValueError("Authentication requires HTTPS or a loopback service")
    return value


def location(url: str) -> Path:
    return (
        Path.home()
        / ".config/cannery/tokens"
        / (hashlib.sha256(origin(url).encode()).hexdigest() + ".json")
    )


def private_directory(path: Path, *, create: bool = False) -> bool:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise ValueError("Cannot read private CLI credential directory") from None
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError("CLI credential directory must be private and user-owned")
    return True


def headers(url: str) -> dict[str, str]:
    path = location(url)
    if not private_directory(path.parent):
        return {}
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {}
    except OSError:
        raise ValueError("Cannot read private CLI credentials") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise ValueError("CLI credentials must be user-owned private files")
        with os.fdopen(descriptor, encoding="utf-8", closefd=False) as source:
            value = json.loads(source.read(4096))
        if (
            not isinstance(value, dict)
            or value.get("origin") != secure(url)
            or not isinstance(value.get("token"), str)
            or not 32 <= len(value["token"]) <= 128
            or not value["token"].isascii()
            or not value["token"].replace("_", "").replace("-", "").isalnum()
        ):
            raise ValueError("Invalid CLI credentials")
        return {"Authorization": "Bearer " + value["token"]}
    finally:
        os.close(descriptor)


def save(url: str, token: str) -> None:
    value = {"origin": secure(url), "token": token}
    path = location(url)
    private_directory(path.parent, create=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as target:
            json.dump(value, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

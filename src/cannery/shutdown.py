"""Let idle PID-1 coordinators stop on SIGTERM without discarding durable attempts."""

import signal
import threading
from collections.abc import Callable
from types import FrameType


def stop_on_term(event: threading.Event) -> Callable[[], None]:
    def stop(number: int, frame: FrameType | None) -> None:
        event.set()

    previous = signal.signal(signal.SIGTERM, stop)

    def restore() -> None:
        signal.signal(signal.SIGTERM, previous)

    return restore

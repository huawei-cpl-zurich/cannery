import os
import selectors
import signal
import subprocess
import sys
import threading

from cannery.shutdown import stop_on_term


def test_sigterm_stops_idle_coordinator_and_restores_previous_handler():
    previous = signal.getsignal(signal.SIGTERM)
    event = threading.Event()
    restore = stop_on_term(event)
    try:
        signal.raise_signal(signal.SIGTERM)
        assert event.is_set()
    finally:
        restore()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_reconciler_stops_batch_without_dispatching_later_jobs(monkeypatch):
    import cannery.reconcile as module

    delivered = []

    class Engine:
        disposed = False

        def dispose(self):
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(module, "make_engine", lambda _: engine)
    monkeypatch.setattr(module, "reconcile_probes", lambda _: ["first", "second"])
    monkeypatch.setattr(module, "due_registrations", lambda _: ["registration"])
    monkeypatch.setattr(module, "due_intakes", lambda _: ["submission"])

    def send(key):
        delivered.append(key)
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(module.probe, "send", send)
    monkeypatch.setattr(module, "publish_registration", lambda *args: delivered.append("bad"))
    monkeypatch.setattr(module, "publish_submission", lambda *args: delivered.append("bad"))
    module.main()
    assert delivered == ["first"] and engine.disposed


def test_actual_process_exits_successfully_after_sigterm(tmp_path):
    script = (
        "import threading; from cannery.shutdown import stop_on_term; "
        "event=threading.Event(); restore=stop_on_term(event); "
        "print('ready',flush=True); event.wait(60); restore(); assert event.is_set()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=3), "Subprocess readiness deadline exceeded"
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=3) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)

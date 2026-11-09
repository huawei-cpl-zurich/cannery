"""Single-destination CONNECT proxy: Git cannot follow alternates to other hosts."""

import select
import socket
import socketserver
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def tunnel(host: str, port: int, address: str) -> Iterator[str]:
    authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            self.connection.settimeout(2)
            try:
                first = self.rfile.readline(8193).decode("ascii").split()
                if len(first) != 3 or first[:2] != ["CONNECT", authority]:
                    self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                    return
                budget = 16384
                while budget > 0:
                    line = self.rfile.readline(min(budget, 8193))
                    budget -= len(line)
                    if line == b"\r\n":
                        break
                    if not line:
                        return
                else:
                    return
                with socket.create_connection((address, port), timeout=3) as upstream:
                    self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                    self.wfile.flush()
                    end = time.monotonic() + 30
                    while time.monotonic() < end:
                        readable, _, _ = select.select([self.connection, upstream], [], [], 2)
                        for source in readable:
                            data = source.recv(65536)
                            if not data:
                                return
                            target = upstream if source is self.connection else self.connection
                            target.sendall(data)
            except (OSError, UnicodeError):
                return

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        block_on_close = False

    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=3)

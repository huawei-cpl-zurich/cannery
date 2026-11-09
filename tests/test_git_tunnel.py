import socket
import threading

import pytest

from cannery.git_intake.tunnel import tunnel


@pytest.mark.parametrize("host", ["allowed.test", "::1"])
def test_tunnel_pins_destination_and_rejects_alternates(host) -> None:
    with socket.socket() as target:
        target.bind(("127.0.0.1", 0))
        target.listen()
        port = target.getsockname()[1]

        def echo():
            connection, _ = target.accept()
            with connection:
                connection.settimeout(3)
                assert connection.recv(4) == b"ping"
                connection.sendall(b"pong")

        thread = threading.Thread(target=echo)
        thread.start()
        with tunnel(host, port, "127.0.0.1") as proxy:
            proxy_port = int(proxy.rsplit(":", 1)[1])
            with socket.create_connection(("127.0.0.1", proxy_port), timeout=3) as client:
                client.sendall(b"CONNECT metadata.test:443 HTTP/1.1\r\n\r\n")
                assert b"403" in client.recv(1024)
            with socket.create_connection(("127.0.0.1", proxy_port), timeout=3) as client:
                authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
                client.sendall(f"CONNECT {authority} HTTP/1.1\r\n\r\n".encode())
                assert b"200" in client.recv(1024)
                client.sendall(b"ping")
                assert client.recv(4) == b"pong"
        thread.join(timeout=3)
        assert not thread.is_alive()

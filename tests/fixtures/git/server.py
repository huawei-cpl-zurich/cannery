"""Private HTTPS smart-Git fixture; synthetic credentials, no credential logging."""

import base64
import json
import ssl
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path("/fixtures")
REPO = ROOT / "repo.git"


def command(*args):
    return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()


def seed():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "revisions.json").exists():
        return
    command(
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(ROOT / "key.pem"),
        "-out",
        str(ROOT / "ca.pem"),
        "-subj",
        "/CN=git-fixture",
        "-addext",
        "subjectAltName=DNS:git-fixture",
        "-days",
        "2",
    )
    work = ROOT / "work"
    command("git", "init", str(work))
    (work / "spec").mkdir()
    (work / "shared").mkdir()
    (work / "shared/helper.py").write_text("def identity(x): return x\n")
    (work / "unrelated.txt").write_text("must never be retained\n")
    (work / "spec/environment.lock").write_text(
        json.dumps({"schema_version": 1, "python": "3.12", "torch": "2.6.0+cpu"})
    )
    (work / "spec/reference.py").write_text(
        "from shared.helper import identity\ndef reference(x): return identity(x)\n"
    )
    (work / "spec/cases.py").write_text(
        "import torch\ndef generate_cases(): yield {'x': torch.ones(2)}\n"
    )
    template = {
        "schema_version": 1,
        "kernel": {"namespace": "team", "name": "identity"},
        "spec": {
            "version": 1,
            "reference": {"file": "reference.py", "function": "reference"},
            "cases": {"file": "cases.py", "function": "generate_cases"},
            "interface": {"inputs": ["x"], "outputs": ["result"]},
            "validation": {
                "metric": "torch_close",
                "equal_nan": False,
                "default_rtol": 0.001,
                "default_atol": 0.0001,
            },
        },
    }
    revisions = {}
    for name, version in (("v1", 1), ("v2", 2), ("bad", 3)):
        template["spec"]["version"] = version
        (work / "spec/cannery.yaml").write_text(json.dumps(template))
        if version == 2:
            (work / "spec/reference.py").write_text("def reference(x): return x + 1\n")
        if name == "bad":
            (work / "spec/cases.py").write_text("def generate_cases(): return iter(())\n")
        command("git", "-C", str(work), "add", ".")
        command(
            "git",
            "-C",
            str(work),
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=x@y",
            "commit",
            "-m",
            name,
        )
        revisions[name] = command("git", "-C", str(work), "rev-parse", "HEAD")
    command("git", "clone", "--bare", str(work), str(REPO))
    (ROOT / "revisions.json").write_text(json.dumps(revisions))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.serve()

    def do_POST(self):
        self.serve()

    def serve(self):
        expected = "Basic " + base64.b64encode(b"oauth2:fixture-token").decode()
        if self.headers.get("Authorization") != expected:
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Basic realm=fixture")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        location = urlsplit(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= 1024 * 1024:
            self.send_error(413)
            return
        environment = {
            "PATH": "/usr/bin:/bin",
            "GIT_PROJECT_ROOT": str(ROOT),
            "GIT_HTTP_EXPORT_ALL": "1",
            "PATH_INFO": location.path,
            "QUERY_STRING": location.query,
            "REQUEST_METHOD": self.command,
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": str(length),
            "REMOTE_USER": "fixture",
            "GIT_PROTOCOL": self.headers.get("Git-Protocol", ""),
        }
        result = subprocess.run(
            ["git", "http-backend"],
            input=self.rfile.read(length),
            env=environment,
            capture_output=True,
            timeout=10,
            check=True,
        )
        headers, _, body = result.stdout.partition(b"\r\n\r\n")
        pairs = [line.split(b":", 1) for line in headers.split(b"\r\n") if b":" in line]
        status = next((int(v.strip().split()[0]) for k, v in pairs if k.lower() == b"status"), 200)
        self.send_response(status)
        for key, value in pairs:
            if key.lower() != b"status":
                self.send_header(key.decode(), value.strip().decode())
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    seed()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ROOT / "ca.pem", ROOT / "key.pem")
    server = ThreadingHTTPServer(("0.0.0.0", 8443), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()

import json

import yaml


def spec_files(version=1, reference=b"def reference(x): return x\n"):
    manifest = {
        "schema_version": 1,
        "kernel": {"namespace": "team", "name": "identity", "description": "Identity"},
        "spec": {
            "version": version,
            "reference": {"file": "reference.py", "function": "reference"},
            "cases": {"file": "cases.py", "function": "generate_cases"},
            "interface": {"inputs": ["x"], "outputs": ["result"]},
            "validation": {"metric": "torch_close", "default_rtol": 0.001, "default_atol": 0.0001},
        },
    }
    return {
        "spec/cannery.yaml": yaml.safe_dump(manifest).encode(),
        "spec/reference.py": reference,
        "spec/cases.py": b"import torch\ndef generate_cases(): yield {'x': torch.ones(2)}\n",
        "spec/environment.lock": json.dumps(
            {"schema_version": 1, "python": "3.12", "torch": "2.6.0+cpu"}
        ).encode(),
    }

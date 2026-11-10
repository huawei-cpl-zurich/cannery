import yaml


def submission_files(language="catlass"):
    filename = "kernel.py" if language == "catlass" else "kernel.cpp"
    manifest = {
        "schema_version": 1,
        "kernel": "team/identity",
        "spec_version": 1,
        "implementation": {
            "device": "a5",
            "language": language,
            "entrypoint": filename + ":forward",
            "build": {
                "adapter": language,
                "inputs": [filename],
                "artifact": "identity_a5",
                "options": {"optimization": "release"},
            },
        },
        "validation": {"requested_rtol": 0.001, "requested_atol": 0.0001},
    }
    return {
        "candidate/submission.yaml": yaml.safe_dump(manifest).encode(),
        "candidate/" + filename: b"raise RuntimeError('must never execute at intake')\n",
    }

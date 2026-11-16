import hashlib
import io
import json
import struct
import tarfile

import pytest
from test_baseline_format import rewrite
from test_recipe_spec import recipe_spec

from cannery.numerics.baseline_v2 import (
    OutputContractV2,
    baseline_key_v2,
    pack_baseline_v2,
    unpack_baseline_v2,
)
from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.registry.manifest import validate_spec


def fixture():
    files, _, _ = recipe_spec()
    environment = validate_spec(files, "spec").environment
    suite = RecipeSuiteV2.model_validate(
        {
            "schema_version": 2,
            "generator": "sha256-tensor-v2",
            "cases": [
                {
                    "id": "quantize",
                    "inputs": {
                        "x": {
                            "shape": [2, 3],
                            "dtype": "float16",
                            "distribution": "uniform",
                            "seed": 42,
                        },
                        "mask": {
                            "shape": [2],
                            "dtype": "bool",
                            "distribution": "randint",
                            "low": 0,
                            "high": 1,
                            "seed": 43,
                        },
                    },
                }
            ],
        }
    )
    contract = OutputContractV2.model_validate(
        {
            "cases": {
                "quantize": {
                    "values": {"shape": [2, 3], "dtype": "int8"},
                    "scales": {"shape": [2], "dtype": "float32"},
                    "empty": {"shape": [0], "dtype": "int64"},
                }
            }
        }
    )
    outputs = {
        "quantize": {
            "values": b"\x01\xff\x00\x7f\x80\x05",
            "scales": struct.pack("<2f", 0.5, 1.0),
            "empty": b"",
        }
    }
    return suite, contract, environment, outputs


def pack(suite, contract, environment, outputs):
    return pack_baseline_v2("b" * 64, suite, environment, "sha256:" + "a" * 64, contract, outputs)


def unpack(package, suite, contract, environment, **changes):
    return unpack_baseline_v2(
        package,
        hashlib.sha256(package).hexdigest(),
        **(
            {
                "spec_sha256": "b" * 64,
                "suite": suite,
                "contract": contract,
                "environment": environment,
            }
            | changes
        ),
    )


def test_mixed_outputs_roundtrip_without_storing_inputs():
    suite, contract, environment, outputs = fixture()
    data = pack(suite, contract, environment, outputs)
    assert pack(suite, contract, environment, outputs) == data
    restored = unpack(data, suite, contract, environment)
    assert restored.outputs == outputs
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        assert set(archive.getnames()) == {
            "metadata.json",
            "outputs/quantize/values.bin",
            "outputs/quantize/scales.bin",
            "outputs/quantize/empty.bin",
        }
    assert restored.metadata["cases"][0]["outputs"]["values"]["dtype"] == "int8"
    assert set(restored.metadata["cases"][0]["inputs"]) == {"x", "mask"}


def test_identity_pins_spec_recipes_environment_and_output_contract():
    suite, contract, environment, outputs = fixture()
    key = baseline_key_v2("b" * 64, suite, environment, contract)
    changed = OutputContractV2.model_validate(
        {"cases": {"quantize": {"values": {"shape": [6], "dtype": "int8"}}}}
    )
    assert baseline_key_v2("b" * 64, suite, environment, changed) != key
    assert baseline_key_v2("c" * 64, suite, environment, contract) != key
    data = pack(suite, contract, environment, outputs)
    with pytest.raises(ValueError):
        unpack(data, suite, contract, environment, spec_sha256="c" * 64)
    with pytest.raises(ValueError):
        unpack(data, suite, changed, environment)


@pytest.mark.parametrize(
    "extra",
    [
        ("../escape", tarfile.REGTYPE),
        ("outputs/quantize/values.bin", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
        ("directory", tarfile.DIRTYPE),
    ],
)
def test_unexpected_duplicate_and_nonregular_members_fail(extra):
    suite, contract, environment, outputs = fixture()
    data = rewrite(pack(suite, contract, environment, outputs), extra=extra)
    with pytest.raises(ValueError):
        unpack(data, suite, contract, environment)


def test_payload_and_metadata_tampering_missing_outputs_and_wrong_digest_fail():
    suite, contract, environment, outputs = fixture()
    data = pack(suite, contract, environment, outputs)
    for filename in ["outputs/quantize/values.bin", "outputs/quantize/scales.bin", "metadata.json"]:

        def mutate(name, value, filename=filename):
            if name != filename:
                return value
            if name == "metadata.json":
                metadata = json.loads(value)
                metadata["spec_sha256"] = "c" * 64
                return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            return bytes([value[0] ^ 1]) + value[1:]

        with pytest.raises(ValueError):
            unpack(rewrite(data, mutate=mutate), suite, contract, environment)
    with pytest.raises(ValueError):
        unpack_baseline_v2(
            data,
            "0" * 64,
            spec_sha256="b" * 64,
            suite=suite,
            environment=environment,
            contract=contract,
        )
    for bad in [{}, {"quantize": {"values": b""}}, outputs | {"extra": {}}]:
        with pytest.raises(ValueError):
            pack(suite, contract, environment, bad)
    with pytest.raises(ValueError):
        unpack(data[:1024], suite, contract, environment)


def test_output_contract_bounds_and_names():
    for cases in [
        {},
        {"../unsafe": {"x": {"shape": [1], "dtype": "float32"}}},
        {"c": {}},
        {"c": {"x": {"shape": [1024, 1025], "dtype": "float32"}}},
        {"c": {f"x{i}": {"shape": [1], "dtype": "int8"} for i in range(33)}},
        {"c": {f"x{i}": {"shape": [1048576], "dtype": "int64"} for i in range(5)}},
    ]:
        with pytest.raises(ValueError):
            OutputContractV2.model_validate({"cases": cases})
    suite, _, environment, _ = fixture()
    contract = OutputContractV2.model_validate(
        {"cases": {"quantize": {"flags": {"shape": [1], "dtype": "bool"}}}}
    )
    with pytest.raises(ValueError, match="Boolean"):
        pack(suite, contract, environment, {"quantize": {"flags": b"\x02"}})

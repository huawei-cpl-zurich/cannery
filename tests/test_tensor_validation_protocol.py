import base64
import copy
import json

import pytest
from test_tensor_numerical_contract import fixture

from cannery.numerics.recipes_v2 import RecipeSuiteV2
from cannery.numerics.tensor_contract import TensorContractV2
from cannery.numerics.validation_protocol_v2 import parse_tensor_results


def protocol():
    recipes, definition = fixture()
    recipes["cases"].append(copy.deepcopy(recipes["cases"][0]))
    recipes["cases"][1]["id"] = "second"
    definition["cases"]["second"] = copy.deepcopy(definition["cases"]["small"])
    for case in definition["cases"].values():
        case["outputs"]["empty"] = {"shape": [0], "dtype": "bool"}
        case["outputs"]["flags"] = {"shape": [1], "dtype": "bool"}
    suite = RecipeSuiteV2.model_validate(recipes)
    contract = TensorContractV2.model_validate(definition)
    value = {
        "schema_version": 2,
        "cases": [
            {
                "id": case.id,
                "inputs": dict.fromkeys(case.inputs, "a" * 64),
                "outputs": {
                    name: {
                        "shape": list(desc.shape),
                        "dtype": desc.dtype,
                        "data": base64.b64encode(bytes(desc.nbytes)).decode(),
                    }
                    for name, desc in contract.cases[case.id].outputs.items()
                },
            }
            for case in suite.cases
        ],
    }
    return suite, contract, value


def test_named_mixed_dtype_empty_outputs_roundtrip():
    suite, contract, value = protocol()
    result = parse_tensor_results(json.dumps(value).encode(), suite, contract)
    assert list(result.outputs) == ["small", "second"]
    assert result.outputs["small"] == {
        "updated": bytes(24),
        "indices": bytes(8),
        "empty": b"",
        "flags": b"\0",
    }
    assert result.input_hashes["second"] == {"x": "a" * 64, "cache": "a" * 64}


@pytest.mark.parametrize(
    "kind",
    [
        "missing_case",
        "duplicate_case",
        "order",
        "extra_case",
        "version",
        "missing_input",
        "extra_input",
        "hash",
        "missing_output",
        "extra_output",
        "shape",
        "bool_dimension",
        "dtype",
        "data",
        "noncanonical_bool",
        "extra_field",
        "null_hash",
        "null_output",
    ],
)
def test_malformed_replies_fail_closed(kind):
    suite, contract, value = protocol()
    entry = value["cases"][0]
    output = entry["outputs"]["updated"]
    if kind == "missing_case":
        value["cases"].pop()
    elif kind == "duplicate_case":
        value["cases"][1] = copy.deepcopy(entry)
    elif kind == "order":
        value["cases"].reverse()
    elif kind == "extra_case":
        value["cases"].append(copy.deepcopy(entry))
    elif kind == "version":
        value["schema_version"] = True
    elif kind == "missing_input":
        entry["inputs"].pop("x")
    elif kind == "extra_input":
        entry["inputs"]["unexpected"] = "a" * 64
    elif kind == "hash":
        entry["inputs"]["x"] = "A" * 64
    elif kind == "missing_output":
        entry["outputs"].pop("indices")
    elif kind == "extra_output":
        entry["outputs"]["unexpected"] = output
    elif kind == "shape":
        output["shape"] = [12]
    elif kind == "bool_dimension":
        entry["outputs"]["flags"]["shape"] = [True]
    elif kind == "dtype":
        output["dtype"] = "int8"
    elif kind == "data":
        output["data"] = "!!!!"
    elif kind == "noncanonical_bool":
        entry["outputs"]["flags"]["data"] = "Ag=="
    elif kind == "extra_field":
        entry["passed"] = True
    elif kind == "null_hash":
        entry["inputs"]["x"] = None
    else:
        entry["outputs"]["updated"] = None
    with pytest.raises(ValueError, match="Invalid tensor"):
        parse_tensor_results(json.dumps(value).encode(), suite, contract)


@pytest.mark.parametrize(
    "data",
    [
        b"{",
        b"[]",
        b"null",
        b'{"schema_version":2,"schema_version":2,"cases":[]}',
        b'{"schema_version":2,"cases":[null]}',
        b'{"schema_version":2,"cases":{}}',
    ],
)
def test_truncated_nonobject_and_duplicate_keys_rejected(data):
    suite, contract, _ = protocol()
    with pytest.raises(ValueError):
        parse_tensor_results(data, suite, contract)


def test_oversized_and_nonbytes_output_rejected():
    suite, contract, _ = protocol()
    for data in [bytes(63 * 1024 * 1024 + 1), bytearray(b"{}")]:
        with pytest.raises(ValueError, match="budget"):
            parse_tensor_results(data, suite, contract)

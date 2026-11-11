import base64
import copy
import json

import pytest
from test_recipe_spec import recipe_spec

from cannery.numerics.recipes import generate, input_sha256, parse_recipes
from cannery.numerics.validation_runtime import parse_results, runtime_program


def protocol():
    files, _, _ = recipe_spec()
    suite = parse_recipes(files["spec/recipes.yaml"])
    cases = []
    for case in suite.cases:
        recipe = case.inputs["x"]
        data = generate(recipe, case.id, "x")
        cases.append(
            {
                "id": case.id,
                "shape": list(recipe.shape),
                "dtype": "float32",
                "input_sha256": input_sha256(recipe, data),
                "data": base64.b64encode(data).decode(),
            }
        )
    return suite, {"schema_version": 1, "cases": cases}


def test_generated_runtime_uses_only_pinned_sdk_and_recipe_metadata():
    source = runtime_program()
    compile(source, "<validation-runtime>", "exec")
    assert "import catlass" not in source and "import cannery." not in source
    assert "resolve_neg()" in source and "backing=torch.empty" in source
    assert "expected_outputs" not in source and "baseline.tar" not in source
    assert "wire=os.fdopen" in source and "os.dup2" in source


def test_complete_output_protocol_roundtrip_including_empty_tensor():
    suite, value = protocol()
    result = parse_results(json.dumps(value).encode(), suite)
    assert list(result) == [c.id for c in suite.cases]
    assert result["empty"].data == b""
    for case in suite.cases:
        recipe = case.inputs["x"]
        data = generate(recipe, case.id, "x")
        assert result[case.id].data == data
        assert result[case.id].input_sha256 == input_sha256(recipe, data)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "order",
        "shape",
        "dtype",
        "bytes",
        "hash",
        "extra",
        "version",
        "bool_dimension",
    ],
)
def test_missing_malformed_or_forged_result_descriptors_rejected(kind):
    suite, value = protocol()
    entry = value["cases"][0]
    if kind == "missing":
        value["cases"].pop()
    elif kind == "duplicate":
        value["cases"][1] = copy.deepcopy(entry)
    elif kind == "order":
        value["cases"].reverse()
    elif kind == "shape":
        entry["shape"] = [6]
    elif kind == "dtype":
        entry["dtype"] = "float16"
    elif kind == "bytes":
        entry["data"] = "AA=="
    elif kind == "hash":
        entry["input_sha256"] = "invalid"
    elif kind == "extra":
        entry["passed"] = True
    elif kind == "version":
        value["schema_version"] = True
    else:
        entry["shape"] = [True, 6]
    with pytest.raises(ValueError):
        parse_results(json.dumps(value).encode(), suite)


def test_duplicate_json_keys_and_truncated_payload_fail_closed():
    suite, _ = protocol()
    for data in [b'{"schema_version":1,"schema_version":1,"cases":[]}', b"{", b"[]"]:
        with pytest.raises(ValueError):
            parse_results(data, suite)
